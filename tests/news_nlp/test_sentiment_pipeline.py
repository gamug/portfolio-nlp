"""Sentiment stage: entity-scoped, sentence-level aggregation.

PLAN.md Work item 4 step 1 (chosen 2026-09-12): FinBERT was fine-tuned on
Financial PhraseBank -- single, standalone sentences -- and has no
per-company reasoning of its own. The old aggregation (chunk-level, one
~510-token multi-sentence forward pass per chunk, combined via a plain
token-count-weighted mean across chunks) gave a sentence about a
*different* company, or generic market commentary, the same say in the
article's score as a sentence actually about its subject. This scores each
sentence individually and weights sentences naming the article's own
company/ticker (`_SENTIMENT_SUBJECT_WEIGHT`) over everything else
(`_SENTIMENT_BASELINE_WEIGHT`) before averaging.
"""

import sqlite3
from typing import Any

import pytest
import torch
from conftest import seed_article

import news_nlp as db
import pipeline

# --- _sentence_mentions_subject / _normalize_company_name (pure) ------------


def test_ticker_mention_is_word_boundary_matched() -> None:
    assert pipeline._sentence_mentions_subject("ACME shares rose today.", None, "ACME")
    assert pipeline._sentence_mentions_subject("acme shares rose today.", None, "ACME")
    # Not a substring match inside an unrelated word.
    assert not pipeline._sentence_mentions_subject("The ACMEX fund fell today.", None, "ACME")


def test_company_name_matches_despite_corporate_suffix_mismatch() -> None:
    # Article's own company field carries a suffix the sentence doesn't (or
    # a different one) -- both should still match after normalization.
    assert pipeline._sentence_mentions_subject("Acme reported earnings.", "Acme Corp.", None)
    assert pipeline._sentence_mentions_subject(
        "Acme Corporation reported earnings.", "Acme Corp", None
    )


def test_different_company_does_not_match() -> None:
    assert not pipeline._sentence_mentions_subject(
        "Rival Beta Inc warned of steep losses.", "Acme Corp", "ACME"
    )


def test_no_company_or_ticker_never_matches() -> None:
    assert not pipeline._sentence_mentions_subject("Acme Corp reported earnings.", None, None)


def test_sentiment_sentence_weights_assigns_subject_vs_baseline() -> None:
    sentences = [
        "Acme Corp reported record profit.",
        "Rival Beta Inc warned of steep losses.",
    ]
    weights = pipeline._sentiment_sentence_weights(sentences, "Acme Corp", "ACME")
    assert weights == [pipeline._SENTIMENT_SUBJECT_WEIGHT, pipeline._SENTIMENT_BASELINE_WEIGHT]


# --- run_sentiment_stage integration: a mixed-company article ---------------


class FakeSentimentEncoding(dict):
    """Mimics the one piece of a real tokenizer's BatchEncoding
    run_sentiment_stage actually uses: a dict-like object with a `.to()`
    that returns itself, holding a single fake token id run_sentiment_stage
    never inspects directly -- only the fake model below reads it, to route
    each call to the right canned logits for that exact sentence."""

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
    """Routes each exact sentence text to a stable per-sentence id via
    `vocab` -- the fake model below looks predictions up by that id."""

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


def test_run_sentiment_stage_is_not_dragged_negative_by_a_different_companys_bad_news(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The concrete failure this aggregation fixes: an article that's
    actually good news for its own subject company (2 sentences naming
    Acme, confidently positive) but mentions a *different* company's bad
    news along the way (3 sentences about "Beta Inc"/generic market
    commentary, confidently negative). A plain unweighted mean over these
    5 sentences calls the article NEGATIVE (3 confident negatives outvote 2
    confident positives) -- entity-scoped weighting must call it POSITIVE
    instead, since only the Acme-naming sentences are actually about this
    article's subject.
    """
    sentences = [
        "Acme Corp reported record profit and raised its full-year guidance.",
        "Acme's chief executive said demand remains strong across all regions.",
        "Rival Beta Inc warned of steep losses and slashed its own outlook.",
        "Beta Inc shares fell sharply on the news.",
        "Broader market sentiment remained weak amid recession fears.",
    ]
    body = " ".join(sentences)
    seed_article(conn, id=1, company="Acme Corp", ticker="ACME", body_text=body)
    conn.commit()

    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    # id2label order: [positive_logit, negative_logit, neutral_logit]
    positive_logits = [8.0, -8.0, -8.0]
    negative_logits = [-8.0, 8.0, -8.0]
    id2logits = {
        0: positive_logits,  # "Acme Corp reported record profit..."
        1: positive_logits,  # "Acme's chief executive said..."
        2: negative_logits,  # "Rival Beta Inc warned..."
        3: negative_logits,  # "Beta Inc shares fell..."
        4: negative_logits,  # "Broader market sentiment..."
    }
    vocab = {s: i for i, s in enumerate(sentences)}

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

    def fake_fetch(_conn: Any, limit: int | None = None) -> list[Any]:
        calls.append({"limit": limit})
        return []

    monkeypatch.setattr(pipeline.db, "fetch_pending_sentiment_articles", fake_fetch)

    pipeline.run_sentiment_stage(conn, limit=50)

    assert calls == [{"limit": 50}]
