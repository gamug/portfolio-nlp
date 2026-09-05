"""news_nlp.eval.sampling: the 60/40 low-conf / random split over a two-tier store."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import EVAL_LOW_IDS

import news_nlp as db_module
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


def test_requires_article_body_text(results_db_path: Path) -> None:
    """A store whose `articles` has no `body_text` column (the lean RESULTS
    store, opened single-file) can't be evaluated -- eval needs the source text."""
    lean = db_module.connect(results_db_path)
    try:
        with pytest.raises(RuntimeError, match="SOURCE store"):
            sample_for_stage(lean, "sentiment", size=2)
    finally:
        lean.close()
