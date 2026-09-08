"""news_nlp.eval.sampling: the low_conf / target_<x> / representative strata
over a two-tier store."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import pytest
from conftest import EVAL_LOW_IDS, write_stage_predictions

import news_nlp as db_module
from news_nlp.eval import sampling
from news_nlp.eval.sampling import EvalItem, sample_for_stage
from news_nlp.taxonomy import CATEGORY_SLUGS


def _buckets(items: list[EvalItem]) -> tuple[list[EvalItem], list[EvalItem]]:
    return (
        [it for it in items if it.bucket == "low_conf"],
        [it for it in items if it.bucket == "representative"],
    )


def _key(items: list[EvalItem]) -> list[tuple[int, str]]:
    return [(it.article_id, it.bucket) for it in items]


def test_sentiment_split_counts_and_membership(eval_conn: db_module.NewsNlpDatabase) -> None:
    # target_frac=0.0 isolates the low_conf/representative split this test is
    # actually about -- see test_target_frac_zero_matches_legacy_two_bucket_ordering
    # for why that's a faithful (not a workaround) way to test this.
    items = sample_for_stage(
        eval_conn, "sentiment", size=10, low_conf_frac=0.6, target_frac=0.0, seed=1
    )
    low, rep = _buckets(items)
    assert len(items) == 10
    assert len(low) == 6
    assert len(rep) == 4
    assert {it.article_id for it in low} <= set(EVAL_LOW_IDS)
    assert {it.article_id for it in low}.isdisjoint({it.article_id for it in rep})
    low_mean = sum(it.prediction["score"] for it in low) / len(low)
    rep_mean = sum(it.prediction["score"] for it in rep) / len(rep)
    assert low_mean < rep_mean


def test_sampling_is_deterministic_under_seed(eval_conn: db_module.NewsNlpDatabase) -> None:
    a = sample_for_stage(eval_conn, "sentiment", size=10, seed=7)
    b = sample_for_stage(eval_conn, "sentiment", size=10, seed=7)
    c = sample_for_stage(eval_conn, "sentiment", size=10, seed=8)
    assert _key(a) == _key(b)
    assert _key(a) != _key(c)


def test_category_low_conf_rows_are_near_threshold_or_other(
    eval_conn: db_module.NewsNlpDatabase,
) -> None:
    items = sample_for_stage(eval_conn, "category", size=6, low_conf_frac=0.5, seed=3)
    low, _ = _buckets(items)
    assert low
    for it in low:
        pred = it.prediction
        assert pred["label"] == "other" or 0.3 <= pred["score"] <= 0.5


def test_every_item_carries_body_text_and_prediction(
    eval_conn: db_module.NewsNlpDatabase,
) -> None:
    for stage in ("sentiment", "category", "ner", "c_summary"):
        items = sample_for_stage(eval_conn, stage, size=8, seed=1)
        assert items
        for it in items:
            assert it.body_text.strip()
            assert isinstance(it.prediction, dict)
            assert it.prediction
            assert it.stratum_population > 0


def test_sentiment_uses_full_body_text_category_stays_capped(
    eval_store_paths: tuple[Path, Path],
) -> None:
    """sentiment must see the body_text FinBERT actually scored (the whole
    article, per run_sentiment_stage's chunk-and-average) -- category stays
    capped at _MAX_BODY_CHARS since it deliberately classifies the lead
    chunk only (see src/news_nlp/eval/sampling.py's _UNCAPPED_STAGES note)."""
    source, results = eval_store_paths
    long_body = "Acme Corp reported strong quarterly results. " * 400
    assert len(long_body) > sampling._MAX_BODY_CHARS

    # eval_conn attaches SOURCE read-only, so mutate it directly first.
    raw = sqlite3.connect(source)
    raw.execute("UPDATE articles SET body_text = ? WHERE id = 1", (long_body,))
    raw.commit()
    raw.close()

    conn = db_module.connect_pipeline(results_db=results, source_db=source)
    try:
        # id 1 is the single lowest-scoring row for both stages in eval_store_paths.
        sentiment_items = sample_for_stage(conn, "sentiment", size=1, low_conf_frac=1.0, seed=1)
        category_items = sample_for_stage(conn, "category", size=1, low_conf_frac=1.0, seed=1)
    finally:
        db_module.detach_source(conn)
        conn.close()

    assert sentiment_items[0].article_id == 1
    assert sentiment_items[0].body_text == long_body
    assert "truncated" not in sentiment_items[0].body_text

    assert category_items[0].article_id == 1
    assert category_items[0].body_text != long_body
    assert category_items[0].body_text.endswith("[... truncated ...]")


def test_sentiment_target_negative_captures_near_miss_row(
    eval_store_paths: tuple[Path, Path],
) -> None:
    """A row whose argmax is positive but whose raw `negative` score clears
    _SENTIMENT_TARGET_THRESHOLD must land in target_negative, not
    representative -- the crux false-negative-recall-enrichment property this
    redesign exists for."""
    source, results = eval_store_paths
    conn = db_module.connect_pipeline(results_db=results, source_db=source)
    try:
        write_stage_predictions(
            conn, 15, sentiment_score=0.95, sentiment_label="positive", sentiment_negative=0.40
        )
        items = sample_for_stage(
            conn, "sentiment", size=12, low_conf_frac=8 / 12, target_frac=0.6, seed=1
        )
    finally:
        db_module.detach_source(conn)
        conn.close()

    by_id = {it.article_id: it for it in items}
    assert 15 in by_id
    assert by_id[15].bucket == "target_negative"


def test_category_priority_order_resolves_dual_threshold_row(
    eval_store_paths: tuple[Path, Path],
) -> None:
    """A row clearing >= _CATEGORY_TARGET_THRESHOLD on two slugs claims only
    the higher-priority (first-listed in _CATEGORY_TARGET_WEIGHTS) stratum."""
    source, results = eval_store_paths
    conn = db_module.connect_pipeline(results_db=results, source_db=source)
    try:
        write_stage_predictions(
            conn,
            16,
            category_distribution={
                "partnerships_business_dev": 0.25,
                "labor_human_capital": 0.25,
            },
        )
        items = sample_for_stage(
            conn, "category", size=12, low_conf_frac=8 / 12, target_frac=1.0, seed=1
        )
    finally:
        db_module.detach_source(conn)
        conn.close()

    by_id = {it.article_id: it for it in items}
    assert 16 in by_id
    assert by_id[16].bucket == "target_partnerships_business_dev"


def test_strata_are_pairwise_disjoint(eval_store_paths: tuple[Path, Path]) -> None:
    source, results = eval_store_paths
    conn = db_module.connect_pipeline(results_db=results, source_db=source)
    try:
        write_stage_predictions(conn, 15, sentiment_label="positive", sentiment_negative=0.40)
        write_stage_predictions(
            conn, 16, sentiment_label="neutral", sentiment_negative=0.36, sentiment_neutral=0.5
        )
        for stage in ("sentiment", "category", "ner", "c_summary"):
            items = sample_for_stage(
                conn, stage, size=18, low_conf_frac=8 / 18, target_frac=1.0, seed=2
            )
            ids = [it.article_id for it in items]
            assert len(ids) == len(set(ids)), f"{stage}: duplicate article_id across strata"
    finally:
        db_module.detach_source(conn)
        conn.close()


def test_stratum_population_matches_independent_sql_count(
    eval_store_paths: tuple[Path, Path],
) -> None:
    source, results = eval_store_paths
    conn = db_module.connect_pipeline(results_db=results, source_db=source)
    try:
        write_stage_predictions(conn, 15, sentiment_label="positive", sentiment_negative=0.40)
        items = sample_for_stage(
            conn, "sentiment", size=12, low_conf_frac=8 / 12, target_frac=0.6, seed=1
        )
        (expected,) = conn.execute(
            "SELECT COUNT(*) FROM article_sentiment WHERE negative >= ?",
            (sampling._SENTIMENT_TARGET_THRESHOLD,),
        ).fetchone()
    finally:
        db_module.detach_source(conn)
        conn.close()

    target_items = [it for it in items if it.bucket == "target_negative"]
    assert target_items
    assert all(it.stratum_population == expected for it in target_items)


def test_target_frac_zero_matches_legacy_two_bucket_ordering(
    eval_conn: db_module.NewsNlpDatabase,
) -> None:
    """target_frac=0 (or a stage with no target strata, i.e. ner always) must
    reproduce the exact pre-stratification rng sequence -- 'representative' is
    just 'random' renamed."""
    for stage in ("sentiment", "category", "ner", "c_summary"):
        new = sample_for_stage(
            eval_conn, stage, size=10, low_conf_frac=0.6, target_frac=0.0, seed=3
        )

        # Hand-rolled replica of the pre-stratification two-bucket algorithm.
        rng = random.Random(3)  # noqa: S311 -- test-only replica, not cryptography
        n_low = round(10 * 0.6)
        chosen_low = sampling._low_conf_ids(eval_conn, stage, n_low)
        rest = [i for i in sampling._all_ids(eval_conn, stage) if i not in set(chosen_low)]
        rng.shuffle(rest)
        chosen_random = rest[: 10 - len(chosen_low)]

        assert [it.article_id for it in new if it.bucket == "low_conf"] == chosen_low
        assert [it.article_id for it in new if it.bucket == "representative"] == chosen_random


def test_category_target_weights_sum_to_one() -> None:
    assert sum(sampling._CATEGORY_TARGET_WEIGHTS.values()) == pytest.approx(1.0)


def test_category_target_slugs_are_valid_taxonomy_slugs() -> None:
    assert set(sampling._CATEGORY_TARGET_WEIGHTS) <= set(CATEGORY_SLUGS)


def test_target_thresholds_stay_below_the_argmax_guarantee_boundary() -> None:
    """A threshold >= 0.5 would mathematically exclude every false-negative
    candidate for that class (only one class in a distribution over mutually
    exclusive classes can exceed 0.5) -- silently defeating soft-probability
    stratification's whole purpose. See sampling.py's module-level comment."""
    assert sampling._SENTIMENT_TARGET_THRESHOLD < 0.5
    assert sampling._CATEGORY_TARGET_THRESHOLD < 0.5


def test_requires_article_body_text(results_db_path: Path) -> None:
    """A store whose `articles` has no `body_text` column (the lean RESULTS
    store, opened single-file) can't be evaluated -- eval needs the source text."""
    lean = db_module.connect(results_db_path)
    try:
        with pytest.raises(RuntimeError, match="SOURCE store"):
            sample_for_stage(lean, "sentiment", size=2)
    finally:
        lean.close()
