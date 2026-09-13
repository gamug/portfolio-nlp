"""Sentiment stage: title-only scoring.

PLAN.md Work item 4 step 1 (chosen 2026-09-13, after prototyping and
real-data-validating an entity-scoped chunk/sentence-weighting alternative
the same week -- see docs/evaluation.md's 2026-09-13 follow-up for that
comparison). Financial-news headlines conventionally state the primary
event and its direction plainly (inverted-pyramid style); scoring the
title directly sidesteps the whole multi-sentence-aggregation problem
(chunking, weighting, per-company reasoning) the alternative design tried
to solve -- one short text, one forward pass, no aggregation at all.
"""

import sqlite3
from typing import Any

import pytest
import torch
from conftest import seed_article

import news_nlp as db
import pipeline

# --- run_sentiment_stage integration -----------------------------------------


class FakeSentimentEncoding(dict):
    """Mimics the one piece of a real tokenizer's BatchEncoding
    run_sentiment_stage actually uses: a dict-like object with a `.to()`
    that returns itself, holding a single fake token id the fake model
    below reads to route each call to the right canned logits for that
    exact title."""

    def __init__(self, token_id: int) -> None:
        super().__init__(
            {
                "input_ids": torch.tensor([[token_id]], dtype=torch.long),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
            }
        )

    def to(self, device: Any) -> "FakeSentimentEncoding":
        return self


class FakeSentimentTokenizer:
    """Routes each exact title text to a stable id via `vocab` -- the fake
    model below looks predictions up by that id."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab

    def __call__(self, text: str, **kwargs: Any) -> FakeSentimentEncoding:
        return FakeSentimentEncoding(self._vocab[text])


class FakeSentimentModel:
    def __init__(self, id2label: dict[int, str], id2logits: dict[int, list[float]]) -> None:
        self.config = type("Config", (), {"id2label": id2label})()
        self._id2logits = id2logits

    def to(self, device: Any) -> "FakeSentimentModel":
        return self

    def eval(self) -> "FakeSentimentModel":
        return self

    def __call__(self, **kwargs: Any) -> Any:
        token_id = int(kwargs["input_ids"][0, 0].item())
        logits = torch.tensor([self._id2logits[token_id]])
        return type("Output", (), {"logits": logits})()


def _patch_sentiment_model(
    monkeypatch: pytest.MonkeyPatch,
    id2label: dict[int, str],
    vocab: dict[str, int],
    id2logits: dict[int, list[float]],
) -> None:
    monkeypatch.setattr(
        pipeline.AutoTokenizer, "from_pretrained", lambda *_a, **_k: FakeSentimentTokenizer(vocab)
    )
    monkeypatch.setattr(
        pipeline.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *_a, **_k: FakeSentimentModel(id2label, id2logits),
    )


def test_run_sentiment_stage_scores_the_title_not_the_body(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core design contract: the model is called with the article's
    `title`, not its `body_text` -- a body that would score very
    differently from the title must not leak into the result."""
    title = "Acme Corp posts record profit, raises full-year guidance"
    # A deliberately opposite-sounding body -- if this got scored instead
    # of the title, the assertions below would fail.
    body = "Rival Beta Inc warned of steep losses and slashed its outlook."
    seed_article(conn, id=1, title=title, body_text=body)
    conn.commit()

    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    vocab = {title: 0}
    id2logits = {0: [8.0, -8.0, -8.0]}  # positive
    _patch_sentiment_model(monkeypatch, id2label, vocab, id2logits)

    pipeline.run_sentiment_stage(conn)

    detail = db.get_article_detail(conn, 1)
    assert detail is not None
    sentiment = detail["sentiment"]
    assert sentiment is not None
    assert sentiment["label"] == "positive"
    assert sentiment["positive"] > sentiment["negative"]


class _FailIfCalledModel:
    """A fake model whose `__call__` raises -- used to prove a blank title
    never reaches the model at all, not just that it produces no row."""

    def __init__(self) -> None:
        self.config = type(
            "Config", (), {"id2label": {0: "positive", 1: "negative", 2: "neutral"}}
        )()

    def to(self, device: Any) -> "_FailIfCalledModel":
        return self

    def eval(self) -> "_FailIfCalledModel":
        return self

    def __call__(self, **kwargs: Any) -> Any:
        raise AssertionError("model should not be called for a blank title")


def test_run_sentiment_stage_skips_articles_with_no_title(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank title has nothing for the model to score -- must not crash,
    must not call the model, and must leave no article_sentiment row."""
    seed_article(conn, id=1, title="", body_text="Some body text.")
    conn.commit()

    monkeypatch.setattr(
        pipeline.AutoTokenizer, "from_pretrained", lambda *_a, **_k: FakeSentimentTokenizer({})
    )
    monkeypatch.setattr(
        pipeline.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *_a, **_k: _FailIfCalledModel(),
    )

    pipeline.run_sentiment_stage(conn)

    detail = db.get_article_detail(conn, 1)
    assert detail is not None
    assert detail["sentiment"] is None


def test_run_sentiment_stage_reads_title_via_new_dedicated_query(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit-level check of the pass-through contract: run_sentiment_stage
    must fetch via `db.fetch_pending_sentiment_titles` (title, not
    body_text), not the old bare-(id, body_text) `fetch_pending_articles`.
    Stubbed to return no rows so the stage exits before needing a
    tokenizer or model."""
    calls: list[dict[str, Any]] = []

    def fake_fetch(
        _conn: Any, limit: int | None = None, *, sample_seed: int | None = None
    ) -> list[Any]:
        calls.append({"limit": limit, "sample_seed": sample_seed})
        return []

    monkeypatch.setattr(pipeline.db, "fetch_pending_sentiment_titles", fake_fetch)

    pipeline.run_sentiment_stage(conn, limit=50)

    assert calls == [{"limit": 50, "sample_seed": None}]


def test_run_sentiment_stage_passes_sample_seed_through_to_fetch_pending_sentiment_titles(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_sentiment_stage(..., sample_seed=...)` must reach
    `db.fetch_pending_sentiment_titles` unchanged -- the same pass-through
    contract as NER's/category's `sample_seed`, for a deliberate targeted
    reprocessing pass."""
    calls: list[dict[str, Any]] = []

    def fake_fetch(
        _conn: Any, limit: int | None = None, *, sample_seed: int | None = None
    ) -> list[Any]:
        calls.append({"limit": limit, "sample_seed": sample_seed})
        return []

    monkeypatch.setattr(pipeline.db, "fetch_pending_sentiment_titles", fake_fetch)

    pipeline.run_sentiment_stage(conn, limit=10_000, sample_seed=1)

    assert calls == [{"limit": 10_000, "sample_seed": 1}]
