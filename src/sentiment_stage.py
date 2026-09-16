"""Sentiment stage `Feature`/`Inference` (PLAN.md Work item 10 / TASKS.md
T-083), migrating `pipeline.py`'s entity-scoped chunk-level sentiment logic
onto the FTI base classes from `src/fti.py`.

Not `Trainer` -- training (`train_sentiment.py`) operates on raw labeled
*sentences* (no chunking, no subject-weighting), a fundamentally different
input shape from inference's full-article chunking; `train_sentiment.py`
gets its own `SentimentTrainer` instead of reusing `SentimentFeature` for a
shape that doesn't fit it. Keeping this module separate from
`train_sentiment.py` also keeps the heavier training-only dependency chain
(`evaluate`, `datasets`, HF `Trainer`) out of what `pipeline.py` imports on
every normal run.

`AutoTokenizer`/`AutoModelForSequenceClassification` are imported here at
module scope (not received via `pipeline.py`) per `fti.Inference`'s own
`load_model` contract -- `pipeline.py` still imports both for `run_category_
stage`'s own use, so existing hermetic tests' `monkeypatch.setattr(pipeline.
AutoTokenizer, ...)` calls remain valid: monkeypatching a class object
mutates it globally, wherever the patched class is later called from.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import news_nlp as db
from chunking import Chunk, chunk_text
from fti import Feature, FeatureBatch, Inference

# Full weight for a chunk naming the article's own subject company/ticker;
# a lower baseline otherwise -- FinBERT has no per-company reasoning of its
# own, so a sentence about a *different* company (or generic market
# commentary) would otherwise count as much as one actually about the
# article's subject. See docs/evaluation.md's 2026-09-12/13 follow-ups for
# the real-data evidence behind this design and its known, disclosed limits
# (no coreference resolution; ~40% of directional predictions are still
# false alarms on multi-company/mixed-signal articles this can't net out).
_SENTIMENT_SUBJECT_WEIGHT = 1.0
_SENTIMENT_BASELINE_WEIGHT = 0.35
# Strips common corporate suffixes so "Acme Corp." / "Acme Corporation"
# both match a sentence naming just "Acme".
_CORP_SUFFIX_RE = re.compile(
    r"\b(incorporated|inc|corporation|corp|company|co|limited|ltd|plc|llc|"
    r"holdings?|group)\.?\b",
    re.IGNORECASE,
)


def _normalize_company_name(name: str) -> str:
    """Strip corporate suffixes and punctuation for a looser company-name
    comparison -- see `_text_mentions_subject`."""
    name = _CORP_SUFFIX_RE.sub(" ", name)
    name = re.sub(r"[^\w\s]", " ", name)
    return " ".join(name.split()).strip().lower()


def _text_mentions_subject(text: str, company: str | None, ticker: str | None) -> bool:
    """True if `text` names the article's own subject company or ticker --
    the signal `_sentiment_chunk_weights` uses to scope FinBERT's read
    toward the company this article is actually about, instead of treating
    a chunk about a different company (or generic market commentary) as
    equally informative. Generic over its input granularity (originally
    written for single sentences, now applied per ~510-token chunk -- see
    `SentimentInference`'s module docstring for why the granularity
    changed)."""
    lowered = text.lower()
    if ticker and re.search(rf"\b{re.escape(ticker.lower())}\b", lowered):
        return True
    if company:
        normalized = _normalize_company_name(company)
        if normalized and normalized in _normalize_company_name(text):
            return True
    return False


def _sentiment_chunk_weights(
    chunks: list[Chunk], company: str | None, ticker: str | None
) -> list[float]:
    return [
        _SENTIMENT_SUBJECT_WEIGHT
        if _text_mentions_subject(ch.text, company, ticker)
        else _SENTIMENT_BASELINE_WEIGHT
        for ch in chunks
    ]


@dataclass(frozen=True)
class SentimentFeatures:
    """One article's chunked-and-weighted input to the sentiment model."""

    chunks: list[Chunk]
    weights: list[float]


class SentimentFeature(Feature[Any, SentimentFeatures]):
    """`row` is `(article_id, company, ticker, body_text)`, matching
    `db.fetch_pending_sentiment_articles`'s return shape."""

    def extract_one(self, tokenizer: Any, row: Any) -> SentimentFeatures:
        _article_id, company, ticker, body_text = row
        chunks = chunk_text(body_text, tokenizer, max_tokens=510)
        return SentimentFeatures(
            chunks=chunks, weights=_sentiment_chunk_weights(chunks, company, ticker)
        )


class SentimentInference(Inference[Any, SentimentFeatures]):
    """Entity-scoped, chunk-level aggregation (PLAN.md Work item 4 step 1,
    chosen 2026-09-12, revised to chunk granularity 2026-09-13; migrated
    onto the FTI hierarchy 2026-09-16, PLAN.md Work item 10 / TASKS.md
    T-083). Scores each ~510-token, sentence-packed chunk in one forward
    pass -- preserving several sentences' worth of real discourse context
    per call -- then combines chunks weighted by `_sentiment_chunk_weights`
    (full weight for chunks naming the article's own company/ticker, a
    lower baseline for everything else) instead of a plain token-count-
    weighted mean, which gave a chunk about a different company, or
    generic market commentary, the same say in the article's score as a
    chunk actually about its subject.

    **Revision history**: the first version of this fix (2026-09-12)
    scored each *sentence* individually, matching FinBERT's own Financial
    PhraseBank fine-tuning granularity. Real-data evaluation the next day
    (`docs/evaluation.md`) showed `recall_negative` regressing more than
    expected. A live probe run earlier in that diagnosis had already shown
    whole-chunk scoring correctly handling a mixed-sentiment passage in one
    forward pass, while naive per-sentence averaging did not -- evidence,
    in hindsight, that decontextualizing down to single sentences threw
    away real discourse signal (negation, contrast, expectation-relative
    framing) that a several-sentence chunk preserves. This revision kept
    the entity-scoped *weighting* idea (validated separately, on its own
    merits) but moved the unit it's applied to back to chunk level, which
    is also ~7-10x fewer forward passes per article than per-sentence
    scoring (a chunk covers many sentences).

    Not batched across chunks/articles (unlike NER's `_ner_batch` /
    category's `CATEGORY_BATCH_SIZE`) -- correctness first, matching how
    NER's own batching was sequenced (PLAN.md Work item 7): a throughput
    pass is a natural, separate follow-up once this aggregation is
    validated against real data.

    **Model swap (2026-09-13)**: the model this scores moved from base
    `ProsusAI/finbert` to `gamug/FinBERT-financial-news`, a continued
    fine-tune on real, LLM-labeled in-domain sentences -- this weighting
    scheme and the model swap were evaluated together and selected as one
    decision, not two independent ones (`docs/evaluation.md`'s 2026-09-13
    follow-ups have the full four-candidate comparison this was chosen
    from).

    `MODEL_NAME`/`MODEL_REVISIONS` below duplicate `pipeline.py`'s own
    `SENTIMENT_MODEL`/`MODEL_REVISIONS[SENTIMENT_MODEL]` as static
    defaults -- `pipeline.run_sentiment_stage`'s thin wrapper always passes
    `model_name=`/`revision=` explicitly (reading its own module globals
    fresh at call time, so a resample script's `pipeline.SENTIMENT_MODEL =
    ...`/`pipeline.MODEL_REVISIONS[...] = "local"` monkeypatch keeps
    working exactly as before), so these class-level defaults only matter
    for a hypothetical direct construction outside that call path (e.g. a
    future `news_nlp.eval` reuse, TASKS.md T-089) -- disclosed, acceptable
    duplication, not a second, independently-drifting source of truth for
    the real call path today.
    """

    MODEL_NAME: ClassVar[str] = "gamug/FinBERT-financial-news"
    MODEL_REVISIONS: ClassVar[dict[str, str]] = {
        "gamug/FinBERT-financial-news": "072712344f1f82e54391e6721b0b39e7b944e898",
    }
    STAGE_NAME: ClassVar[str] = "sentiment"

    def batch_size(self) -> int:
        """1, not the base class's `None` default -- sentiment never
        batches its forward pass across articles today (the weighted
        average is already computed per article), and today's
        `run_sentiment_stage` commits once per article. The base
        `Inference.run()`'s default `None` would run every pending article
        through one `predict_batch`/`write_predictions`/commit instead,
        silently trading real per-article crash resilience for a bigger
        one-shot commit -- a genuine behavioral regression no existing test
        would catch. Returning `1` here preserves the exact commit
        granularity `pipeline.py`'s original loop had."""
        return 1

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_name, revision=self.revision
            )
            .to(self.device)
            .eval()
        )
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    def fetch_pending(
        self, conn: db.NewsNlpDatabase, limit: int | None, sample_seed: int | None
    ) -> list[Any]:
        return db.fetch_pending_sentiment_articles(conn, limit=limit, sample_seed=sample_seed)

    def predict_batch(self, features: FeatureBatch[SentimentFeatures]) -> list[Any]:
        """`features.items` always has exactly one entry (`batch_size() ==
        1`). Returns `[None]` when the article had no chunks (today's
        `if chunks:` guard -- no write for that article) or a one-item list
        with `{"label", "score", "positive", "negative", "neutral"}`."""
        (article_features,) = features.items
        chunks, weights = article_features.chunks, article_features.weights
        if not chunks:
            return [None]

        weighted_probs = torch.zeros(len(self.id2label))
        total_weight = 0.0
        for chunk, weight in zip(chunks, weights, strict=True):
            inputs = self.tokenizer(
                chunk.text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1).cpu()
            weighted_probs += probs * weight
            total_weight += weight

        avg_probs = (weighted_probs / total_weight).tolist()
        class_probs = {self.id2label[i]: p for i, p in enumerate(avg_probs)}
        label = max(class_probs, key=class_probs.__getitem__)
        return [
            {
                "label": label,
                "score": class_probs[label],
                "positive": class_probs.get("positive", 0.0),
                "negative": class_probs.get("negative", 0.0),
                "neutral": class_probs.get("neutral", 0.0),
            }
        ]

    def write_predictions(
        self, conn: db.NewsNlpDatabase, rows: Sequence[Any], predictions: list[Any]
    ) -> None:
        for (article_id, *_rest), prediction in zip(rows, predictions, strict=True):
            if prediction is None:
                continue
            db.write_sentiment(
                conn,
                article_id,
                label=prediction["label"],
                score=prediction["score"],
                positive=prediction["positive"],
                negative=prediction["negative"],
                neutral=prediction["neutral"],
                model_name=self.model_name,
            )
