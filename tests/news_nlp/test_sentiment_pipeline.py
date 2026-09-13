"""Sentiment stage: entity-scoped, chunk-level aggregation.

PLAN.md Work item 4 step 1 (chosen 2026-09-12, revised to chunk
granularity 2026-09-13): FinBERT has no per-company reasoning of its own.
The original aggregation (chunk-level, one ~510-token multi-sentence
forward pass per chunk, combined via a plain token-count-weighted mean
across chunks) gave a chunk about a *different* company, or generic market
commentary, the same say in the article's score as a chunk actually about
its subject. This weights chunks naming the article's own company/ticker
(`_SENTIMENT_SUBJECT_WEIGHT`) over everything else
(`_SENTIMENT_BASELINE_WEIGHT`) before averaging.

(A first version of this fix, shipped 2026-09-12, scored each *sentence*
individually instead of each chunk -- reverted the next day after
real-data evaluation showed it regressing `recall_negative` more than
expected; see `pipeline.run_sentiment_stage`'s docstring "Revision
history" for the full account. This file tests the current, chunk-level
version only -- the sentence-level version's own commit is still in git
history if that account needs corroborating.)
"""

import sqlite3
from typing import Any

import pytest
import torch
from conftest import seed_article

import news_nlp as db
import pipeline

# --- _text_mentions_subject / _normalize_company_name (pure) ----------------


def test_ticker_mention_is_word_boundary_matched() -> None:
    assert pipeline._text_mentions_subject("ACME shares rose today.", None, "ACME")
    assert pipeline._text_mentions_subject("acme shares rose today.", None, "ACME")
    # Not a substring match inside an unrelated word.
    assert not pipeline._text_mentions_subject("The ACMEX fund fell today.", None, "ACME")


def test_company_name_matches_despite_corporate_suffix_mismatch() -> None:
    # Article's own company field carries a suffix the text doesn't (or a
    # different one) -- both should still match after normalization.
    assert pipeline._text_mentions_subject("Acme reported earnings.", "Acme Corp.", None)
    assert pipeline._text_mentions_subject("Acme Corporation reported earnings.", "Acme Corp", None)


def test_different_company_does_not_match() -> None:
    assert not pipeline._text_mentions_subject(
        "Rival Beta Inc warned of steep losses.", "Acme Corp", "ACME"
    )


def test_no_company_or_ticker_never_matches() -> None:
    assert not pipeline._text_mentions_subject("Acme Corp reported earnings.", None, None)


def test_sentiment_chunk_weights_assigns_subject_vs_baseline() -> None:
    chunks = [
        pipeline.Chunk(text="Acme Corp reported record profit.", start_char=0, end_char=34),
        pipeline.Chunk(text="Rival Beta Inc warned of steep losses.", start_char=35, end_char=74),
    ]
    weights = pipeline._sentiment_chunk_weights(chunks, "Acme Corp", "ACME")
    assert weights == [pipeline._SENTIMENT_SUBJECT_WEIGHT, pipeline._SENTIMENT_BASELINE_WEIGHT]


# --- run_sentiment_stage integration: a mixed-company article ---------------


class FakeSentimentEncoding(dict):
    """Mimics the one piece of a real tokenizer's BatchEncoding
    run_sentiment_stage actually uses: a dict-like object with a `.to()`
    that returns itself, holding a single fake token id run_sentiment_stage
    never inspects directly -- only the fake model below reads it, to route
    each call to the right canned logits for that exact chunk."""

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
    """`.encode()` reports a fixed per-sentence token count (used only by
    `chunk_text` to decide packing -- 200 keeps exactly two sentences per
    ~510-token chunk, deterministically splitting a 4-sentence body into
    two chunks). `__call__` routes each exact *chunk* text to a stable id
    via `vocab` -- the fake model below looks predictions up by that id."""

    def __init__(self, vocab: dict[str, int], tokens_per_sentence: int = 200) -> None:
        self._vocab = vocab
        self._tokens_per_sentence = tokens_per_sentence

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return ["x"] * self._tokens_per_sentence

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


def test_run_sentiment_stage_is_not_dragged_negative_by_a_different_companys_bad_news(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The concrete failure this aggregation fixes: an article that's
    actually good news for its own subject company (a chunk naming Acme,
    confidently positive) but mentions a *different* company's bad news
    along the way (a chunk about "Beta Inc", confidently negative). A
    plain unweighted mean over these 2 equal-sized chunks calls the
    article NEGATIVE-leaning-to-a-toss-up -- entity-scoped weighting must
    call it POSITIVE instead, since only the Acme-naming chunk is actually
    about this article's subject.
    """
    s1 = "Acme Corp reported record profit and raised its full-year guidance."
    s2 = "Acme's chief executive said demand remains strong across all regions."
    s3 = "Rival Beta Inc warned of steep losses and slashed its own outlook."
    s4 = "Beta Inc shares fell sharply on the news."
    body = " ".join((s1, s2, s3, s4))
    seed_article(conn, id=1, company="Acme Corp", ticker="ACME", body_text=body)
    conn.commit()

    chunk1_text = f"{s1} {s2}"  # the Acme-naming chunk (subject weight)
    chunk2_text = f"{s3} {s4}"  # the Beta-naming chunk (baseline weight)

    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    # id2label order: [positive_logit, negative_logit, neutral_logit]
    id2logits = {
        0: [8.0, -8.0, -8.0],  # chunk1 (Acme): confidently positive
        1: [-8.0, 8.0, -8.0],  # chunk2 (Beta): confidently negative
    }
    vocab = {chunk1_text: 0, chunk2_text: 1}

    monkeypatch.setattr(
        pipeline.AutoTokenizer, "from_pretrained", lambda *_a, **_k: FakeSentimentTokenizer(vocab)
    )
    monkeypatch.setattr(
        pipeline.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *_a, **_k: FakeSentimentModel(id2label, id2logits),
    )

    pipeline.run_sentiment_stage(conn)

    detail = db.get_article_detail(conn, 1)
    assert detail is not None
    sentiment = detail["sentiment"]
    assert sentiment is not None
    assert sentiment["label"] == "positive"
    assert sentiment["positive"] > sentiment["negative"]


def test_run_sentiment_stage_reads_company_and_ticker_via_new_dedicated_query(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit-level check of the pass-through contract: run_sentiment_stage
    must fetch via `db.fetch_pending_sentiment_articles` (company/ticker
    included), not the old bare-(id, body_text) `fetch_pending_articles` --
    entity-scoped weighting has nothing to scope against otherwise.
    Stubbed to return no rows so the stage exits before needing a tokenizer
    or model."""
    calls: list[dict[str, Any]] = []

    def fake_fetch(
        _conn: Any, limit: int | None = None, *, sample_seed: int | None = None
    ) -> list[Any]:
        calls.append({"limit": limit, "sample_seed": sample_seed})
        return []

    monkeypatch.setattr(pipeline.db, "fetch_pending_sentiment_articles", fake_fetch)

    pipeline.run_sentiment_stage(conn, limit=50)

    assert calls == [{"limit": 50, "sample_seed": None}]


def test_run_sentiment_stage_passes_sample_seed_through_to_fetch_pending_sentiment_articles(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_sentiment_stage(..., sample_seed=...)` must reach
    `db.fetch_pending_sentiment_articles` unchanged -- the same pass-through
    contract as NER's `sample_seed` (T-025), now needed for sentiment's own
    resample (`scripts/resample_sentiment_2026_09_12.py`, TASKS.md
    T-034/T-035) -- always pass one for a reprocessing run (see
    `run_sentiment_stage`'s docstring on why an unseeded run is risky)."""
    calls: list[dict[str, Any]] = []

    def fake_fetch(
        _conn: Any, limit: int | None = None, *, sample_seed: int | None = None
    ) -> list[Any]:
        calls.append({"limit": limit, "sample_seed": sample_seed})
        return []

    monkeypatch.setattr(pipeline.db, "fetch_pending_sentiment_articles", fake_fetch)

    pipeline.run_sentiment_stage(conn, limit=10_000, sample_seed=1)

    assert calls == [{"limit": 10_000, "sample_seed": 1}]
