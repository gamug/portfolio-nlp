"""NER stage `Feature`/`Inference` (PLAN.md Work item 10 / TASKS.md T-084),
migrating `pipeline.py`'s cross-article batched NER logic onto the FTI base
classes from `src/fti.py`.

Not `Trainer` -- training (`train_ner.py`) fine-tunes from FiNER-ORD's own
sentence-level token/label rows, a fundamentally different input shape from
inference's chunked full-article text; `train_ner.py` gets its own
`NerTrainer` instead.

Real batching here is cross-*article*, not per-row (unlike sentiment):
`NerFeature.extract_batch` flattens every chunk from every article in the
batch into a single, consistently padded tokenizer call, so its natural
output is one shared `NerBatchFeatures` object for the whole batch, not one
item per article -- `NerInference.predict_batch` does the forward pass and
then regroups entities back to their owning article.

`AutoTokenizer`/`AutoModelForTokenClassification` are imported here at
module scope (not received via `pipeline.py`) per `fti.Inference`'s own
`load_model` contract -- `pipeline.py` still imports `AutoTokenizer` for
`run_category_stage`'s/`run_company_summary_stage`'s own use, so existing
hermetic tests' `monkeypatch.setattr(pipeline.AutoTokenizer, ...)` calls
remain valid (monkeypatching a class object mutates it globally, wherever
the patched class is later called from); `AutoModelForTokenClassification`
is NER-only, so tests patching it must target `ner_stage.
AutoModelForTokenClassification` instead.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

import news_nlp as db
from chunking import Chunk, chunk_text, merge_char_spans
from fti import Feature, FeatureBatch, Inference

# Last-resort net for the write path: a single-character span is almost
# certainly junk (see merge_bio_predictions' word-id fix for the actual root
# cause this exists alongside, not instead of).
_MIN_ENTITY_TEXT_LEN = 2


def merge_bio_predictions(
    pred_ids: list[int],
    word_ids: list[int | None],
    offsets: list[tuple[int, int]],
    probs: list[list[float]],
    id2label: dict[int, str],
) -> list[dict[str, Any]]:
    """Convert token-level BIO predictions (with char offsets local to the
    chunk) into merged entity spans local to the chunk.

    Word-boundary aware: only a word's *first* WordPiece subword ever decides
    a span boundary (open, close, or same-type continuation). A continuation
    subword (``word_id == prev_word_id``) never independently closes,
    redirects, or starts a span -- it only extends whatever the owning word's
    first subword already decided, because ``train_ner.py``'s
    ``make_tokenize_fn`` masks every continuation subword to
    ``IGNORED_LABEL_ID`` in the training loss: the model gets zero training
    signal for what to predict there, so treating its raw argmax there as a
    real decision (the pre-fix behavior) was training/inference-inconsistent.
    Concretely, this fixes "3M" tokenized as ["3", "##M"] emitting a bogus
    standalone "3"/ORG entity when the model's untrained-for continuation
    prediction on "##M" happened not to be I-ORG (see docs/evaluation.md's
    2026-09-10 NER follow-up for the real-data-confirmed scale of this).

    A continuation subword's own probability is excluded from the entity's
    averaged ``score`` for the same reason -- it was never calibrated against
    any target at that position, so including it would make the average less
    honest, not more informative.
    """
    entities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    prev_word_id: int | None = None

    for pred_id, word_id, (start, end), tok_probs in zip(
        pred_ids, word_ids, offsets, probs, strict=True
    ):
        if word_id is None:  # special/padding token
            prev_word_id = None
            continue

        is_continuation = word_id == prev_word_id
        prev_word_id = word_id

        if is_continuation:
            if current is not None:
                current["end_char"] = end
            continue

        # First subword of a word -- the only prediction the model was
        # actually trained to produce a meaningful label for.
        label = id2label[pred_id]
        score = tok_probs[pred_id]

        if label == "O":
            if current:
                entities.append(current)
                current = None
            continue

        bio, tag_type = label.split("-", 1)
        if bio == "B" or current is None or current["entity_type"] != tag_type:
            if current:
                entities.append(current)
            current = {
                "entity_type": tag_type,
                "start_char": start,
                "end_char": end,
                "scores": [score],
            }
        else:
            current["end_char"] = end
            current["scores"].append(score)

    if current:
        entities.append(current)
    return entities


@dataclass(frozen=True)
class NerBatchFeatures:
    """One `predict_batch` call's flattened, tokenized input -- every chunk
    from every article in this batch as a single, consistently padded unit
    (NER's real batching is cross-article, not per-row), plus the
    bookkeeping `predict_batch` needs to regroup entities back to their
    owning article afterward."""

    inputs: dict[str, Any]
    offsets_batch: list[list[tuple[int, int]]]
    word_ids_batch: list[list[int | None]]
    owner: list[int]
    flat_chunk_starts: list[int]
    body_texts: list[str]
    num_rows: int


class NerFeature(Feature[Any, NerBatchFeatures]):
    """`rows` is a list of `(article_id, body_text)`, matching
    `db.fetch_pending_articles`'s return shape. Overrides `extract_batch`
    directly (not `extract_one`) since NER's real batching is cross-article,
    not expressible as independent per-row extraction."""

    def extract_batch(self, tokenizer: Any, rows: Sequence[Any]) -> FeatureBatch[NerBatchFeatures]:
        flat_texts: list[str] = []
        flat_chunk_starts: list[int] = []
        owner: list[int] = []  # index into rows, one entry per flat_texts entry
        body_texts = [body_text for _article_id, body_text in rows]

        for i, (_article_id, body_text) in enumerate(rows):
            chunks: list[Chunk] = chunk_text(body_text, tokenizer, max_tokens=510)
            for ch in chunks:
                flat_texts.append(ch.text)
                flat_chunk_starts.append(ch.start_char)
                owner.append(i)

        if not flat_texts:
            return FeatureBatch(
                items=[
                    NerBatchFeatures(
                        inputs={},
                        offsets_batch=[],
                        word_ids_batch=[],
                        owner=[],
                        flat_chunk_starts=[],
                        body_texts=body_texts,
                        num_rows=len(rows),
                    )
                ]
            )

        tokenized = tokenizer(
            flat_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            return_offsets_mapping=True,
        )
        offsets_batch = tokenized.pop("offset_mapping").tolist()
        # Must be read before `inputs` is rebuilt as a plain dict below --
        # .word_ids() lives on the BatchEncoding, not the dict.
        word_ids_batch = [tokenized.word_ids(batch_index=i) for i in range(len(flat_texts))]
        return FeatureBatch(
            items=[
                NerBatchFeatures(
                    inputs=dict(tokenized),
                    offsets_batch=offsets_batch,
                    word_ids_batch=word_ids_batch,
                    owner=owner,
                    flat_chunk_starts=flat_chunk_starts,
                    body_texts=body_texts,
                    num_rows=len(rows),
                )
            ]
        )


class NerInference(Inference[Any, NerBatchFeatures]):
    """Cross-article batched NER (PLAN.md Work item 7; migrated onto the FTI
    hierarchy 2026-09-16, PLAN.md Work item 10 / TASKS.md T-084). Flattens
    every chunk of every article in a batch into one padded forward pass,
    then regroups entities back to their owning article -- see
    `merge_bio_predictions` for the word-boundary-aware BIO merge and
    `pipeline.py`'s former `NER_BATCH_SIZE` comment for why this constant is
    an articles-per-batch count, not a forward-pass-width count.

    `MODEL_NAME`/`MODEL_REVISIONS` below duplicate `pipeline.py`'s own
    `NER_MODEL`/`MODEL_REVISIONS[NER_MODEL]` as static defaults --
    `pipeline.run_ner_stage`'s thin wrapper always passes `model_name=`/
    `revision=`/`batch_size=` explicitly (reading its own module globals
    fresh at call time, so a test's `pipeline.NER_BATCH_SIZE = ...`
    monkeypatch between two calls keeps working exactly as before) -- see
    `sentiment_stage.SentimentInference`'s docstring for the same disclosed,
    acceptable duplication.
    """

    MODEL_NAME: ClassVar[str] = "gamug/sec-bert-finer-ord-ner"
    MODEL_REVISIONS: ClassVar[dict[str, str]] = {
        "gamug/sec-bert-finer-ord-ner": "ba7b9e43e4aa023ec5691f955b276dc58158354c",
    }
    STAGE_NAME: ClassVar[str] = "ner"

    def batch_size(self) -> int:
        """The constructor's `batch_size` override, or `8` (matching
        today's `NER_BATCH_SIZE`) if none was given -- the real call path
        (`pipeline.run_ner_stage`) always passes one explicitly, so this
        fallback only matters for a hypothetical direct construction
        outside that path."""
        return self._batch_size_override if self._batch_size_override is not None else 8

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self.model = (
            AutoModelForTokenClassification.from_pretrained(self.model_name, revision=self.revision)
            .to(self.device)
            .eval()
        )
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def fetch_pending(
        self, conn: db.NewsNlpDatabase, limit: int | None, sample_seed: int | None
    ) -> list[Any]:
        return db.fetch_pending_articles(
            conn, "article_entities", limit=limit, sample_seed=sample_seed
        )

    def predict_batch(self, features: FeatureBatch[NerBatchFeatures]) -> list[Any]:
        """Returns one entities list per row in this batch, same order --
        NOT one per `features.items` entry (always exactly 1, the whole
        cross-article batch)."""
        (batch,) = features.items
        per_article_entities: list[list[dict[str, Any]]] = [[] for _ in range(batch.num_rows)]
        if not batch.owner:
            return per_article_entities

        inputs = {k: v.to(self.device) for k, v in batch.inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu()
            pred_ids_batch = probs.argmax(-1).tolist()

        for i in range(len(batch.owner)):
            chunk_entities = merge_bio_predictions(
                pred_ids_batch[i],
                batch.word_ids_batch[i],
                batch.offsets_batch[i],
                probs[i].tolist(),
                self.id2label,
            )
            article_idx = batch.owner[i]
            chunk_start = batch.flat_chunk_starts[i]
            body_text = batch.body_texts[article_idx]
            for e in chunk_entities:
                start = chunk_start + e["start_char"]
                end = chunk_start + e["end_char"]
                per_article_entities[article_idx].append(
                    {
                        "entity_type": e["entity_type"],
                        "text": body_text[start:end],
                        "start_char": start,
                        "end_char": end,
                        "score": sum(e["scores"]) / len(e["scores"]),
                    }
                )

        for entities in per_article_entities:
            entities[:] = merge_char_spans(entities)
            # Last-resort net, not a substitute for merge_bio_predictions'
            # word-boundary fix above: drops any single-character junk span
            # that fix doesn't structurally prevent. A length floor, not a
            # digit-specific check -- excludes_bare_digit's mistake
            # (portfolio_common.db.dialect.SqliteDialect) was living
            # downstream, in two read-side aggregate queries, and only ever
            # catching bare digits. Revisit if this starts hiding a new real
            # bug class.
            entities[:] = [e for e in entities if len(e["text"].strip()) >= _MIN_ENTITY_TEXT_LEN]

        return per_article_entities

    def write_predictions(
        self, conn: db.NewsNlpDatabase, rows: Sequence[Any], predictions: list[Any]
    ) -> None:
        for (article_id, _body_text), entities in zip(rows, predictions, strict=True):
            db.write_entities(conn, article_id, entities, model_name=self.model_name)
