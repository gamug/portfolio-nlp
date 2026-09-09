"""news_nlp.eval.sampling: the 60/40 low-conf / random split over a two-tier store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from conftest import EVAL_LOW_IDS

import news_nlp as db_module
from news_nlp.eval import sampling
from news_nlp.eval.sampling import EvalItem, sample_for_stage


def _buckets(items: list[EvalItem]) -> tuple[list[EvalItem], list[EvalItem]]:
    return (
        [it for it in items if it.bucket == "low_conf"],
        [it for it in items if it.bucket == "random"],
    )


def _key(items: list[EvalItem]) -> list[tuple[int, str]]:
    return [(it.article_id, it.bucket) for it in items]


def test_sentiment_split_counts_and_membership(eval_conn: db_module.NewsNlpDatabase) -> None:
    items = sample_for_stage(eval_conn, "sentiment", size=10, low_conf_frac=0.6, seed=1)
    low, rnd = _buckets(items)
    assert len(items) == 10
    assert len(low) == 6
    assert len(rnd) == 4
    assert {it.article_id for it in low} <= set(EVAL_LOW_IDS)
    assert {it.article_id for it in low}.isdisjoint({it.article_id for it in rnd})
    low_mean = sum(it.prediction["score"] for it in low) / len(low)
    rnd_mean = sum(it.prediction["score"] for it in rnd) / len(rnd)
    assert low_mean < rnd_mean


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


def test_sentiment_still_truncates_past_the_safety_ceiling(
    eval_store_paths: tuple[Path, Path],
) -> None:
    """sentiment is uncapped in practice (see the prior test) but not
    unconditionally: a body past _SENTIMENT_MAX_BODY_CHARS still gets
    truncated, so a pathologically large body_text can't produce an oversized
    judge request that fails outright instead of just losing tail context."""
    source, results = eval_store_paths
    pathological_body = "Acme Corp reported strong quarterly results. " * 3000
    assert len(pathological_body) > sampling._SENTIMENT_MAX_BODY_CHARS

    raw = sqlite3.connect(source)
    raw.execute("UPDATE articles SET body_text = ? WHERE id = 1", (pathological_body,))
    raw.commit()
    raw.close()

    conn = db_module.connect_pipeline(results_db=results, source_db=source)
    try:
        items = sample_for_stage(conn, "sentiment", size=1, low_conf_frac=1.0, seed=1)
    finally:
        db_module.detach_source(conn)
        conn.close()

    assert items[0].article_id == 1
    assert items[0].body_text != pathological_body
    assert items[0].body_text.endswith("[... truncated ...]")
    assert len(items[0].body_text) < len(pathological_body)


def test_requires_article_body_text(results_db_path: Path) -> None:
    """A store whose `articles` has no `body_text` column (the lean RESULTS
    store, opened single-file) can't be evaluated -- eval needs the source text."""
    lean = db_module.connect(results_db_path)
    try:
        with pytest.raises(RuntimeError, match="SOURCE store"):
            sample_for_stage(lean, "sentiment", size=2)
    finally:
        lean.close()
