"""Company summary (`c_summary`) stage `Feature`/`Inference`/`Trainer`
(PLAN.md Work item 10 / TASKS.md T-086, `Trainer` wiring TASKS.md T-097),
migrating `pipeline.py`'s batched chunk-then-reduce summarization onto the
FTI base classes from `src/fti.py`.

`SummaryTrainer` is a trivial, body-less subclass of `fti.NoOpTrainer` --
SUMMARY_MODEL (`sshleifer/distilbart-cnn-12-6`) is a pretrained,
off-the-shelf checkpoint; there is no fine-tuning step to wrap (same
reasoning as `category_stage.py`'s `CategoryTrainer`). It exists as its own
named class, not a bare `fti.NoOpTrainer` reference (T-086 originally left
it as just a docstring mention, never actually instantiated anywhere
reachable outside `fti.py`'s own unit test), for the same reason
`CategoryTrainer` does: a real class for the future stage->Trainer registry
(TASKS.md T-099) to look up, matching sentiment's/NER's own
`SentimentTrainer`/`NerTrainer`.

`hierarchical_summarize_batch`'s own recursive leaf-chunk + reduce-pass
loop can't be hoisted out to a pure `Feature.extract_batch` step the way
NER's cross-article chunking was -- each reduce pass re-chunks and
re-summarizes the *previous* pass's model output, so it's inherently
model-dependent at every step. `SummaryFeature` is accordingly the
simplest of the four stages: it only builds each row's raw input text
(`db.build_company_summary_input`), and all chunking/reduction happens
inside `SummaryInference.predict_batch`.

`AutoTokenizer`/`AutoModelForSeq2SeqLM` are imported here at module scope
(not received via `pipeline.py`) per `fti.Inference`'s own `load_model`
contract -- this is the last of the four stages to migrate, so
`pipeline.py` no longer imports either class for any other stage's sake
after this change; every test that monkeypatches `pipeline.AutoTokenizer`
now needs to target its own already-migrated stage module instead (see
this migration's PR description for the full retarget list).
"""

from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import news_nlp as db
from chunking import chunk_text
from fti import Feature, FeatureBatch, Inference, NoOpTrainer

# BART-large-cnn's own cap is 1024 tokens; 1000 leaves headroom for the
# BOS/EOS tokens the tokenizer adds on top of chunk_text's count.
SUMMARY_MAX_INPUT_TOKENS = 1000
# Matches bart-large-cnn's published default generation config.
SUMMARY_MAX_OUTPUT_TOKENS = 142
SUMMARY_MIN_OUTPUT_TOKENS = 56
# Safety valve for the recursive reduce below -- each pass's summaries are
# far shorter than what fed them, so this converges in 1-2 passes in
# practice; this just bounds the pathological case.
MAX_REDUCE_PASSES = 6
# Disclosed duplication of pipeline.py's own SUMMARY_BATCH_SIZE (see
# SummaryInference's docstring) -- only used as hierarchical_summarize_batch's
# default `batch_size` for a hypothetical direct call outside the real
# pipeline.run_company_summary_stage call path, which always passes its own
# fresh SUMMARY_BATCH_SIZE through explicitly.
SUMMARY_BATCH_SIZE = 4


def _summarize_batch(
    texts: list[str], tokenizer: Any, model: Any, device: torch.device
) -> list[str]:
    """Run SUMMARY_MODEL (distilbart-cnn-12-6) generation on a batch of
    chunks that already fit within the model's input cap, in one forward
    pass -- the same batching principle category's classifier applies by
    pooling multiple articles' (premise, hypothesis) pairs into one call.
    Split out as its own function so tests can monkeypatch it and exercise
    hierarchical_summarize_batch's chunk/reduce control flow without loading
    a real model."""
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=1024, padding=True
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=SUMMARY_MAX_OUTPUT_TOKENS,
            min_length=SUMMARY_MIN_OUTPUT_TOKENS,
            num_beams=4,
        )
    return [s.strip() for s in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]


def _summarize_in_batches(
    texts: list[str], tokenizer: Any, model: Any, device: torch.device, batch_size: int
) -> list[str]:
    """Run _summarize_batch over `texts` in chunks of `batch_size`,
    concatenating results in order -- each generate() call's width stays
    bounded by batch_size regardless of how many texts are pending (the
    call *count* still grows with len(texts), one call per
    batch_size-sized slice)."""
    results: list[str] = []
    for start in range(0, len(texts), batch_size):
        results.extend(
            _summarize_batch(texts[start : start + batch_size], tokenizer, model, device)
        )
    return results


def _leaf_summarize_batch(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_input_tokens: int,
    batch_size: int,
) -> tuple[list[list[str]], list[int]]:
    """Chunk every text on sentence boundaries (chunk_text) and
    batch-summarize the whole pool of leaf chunks together. Returns
    (summaries_per_text, num_chunks), each indexed the same as `texts`."""
    per_text_chunks = [chunk_text(t, tokenizer, max_tokens=max_input_tokens) for t in texts]
    num_chunks = [len(c) for c in per_text_chunks]

    flat_texts: list[str] = []
    owner: list[int] = []
    for i, chunks in enumerate(per_text_chunks):
        for ch in chunks:
            flat_texts.append(ch.text)
            owner.append(i)

    flat_summaries = _summarize_in_batches(flat_texts, tokenizer, model, device, batch_size)

    summaries_per_text: list[list[str]] = [[] for _ in texts]
    for o, s in zip(owner, flat_summaries, strict=True):
        summaries_per_text[o].append(s)

    return summaries_per_text, num_chunks


def _reduce_pass(
    pending: set[int],
    summaries_per_text: list[list[str]],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_input_tokens: int,
    batch_size: int,
) -> None:
    """Run one reduce pass in place over every text index in `pending`: join
    each one's current summaries, re-chunk, and batch-summarize the pooled
    result across all of them -- same batching principle as the leaf pass."""
    flat_texts: list[str] = []
    owner: list[int] = []
    for i in sorted(pending):
        combined = " ".join(summaries_per_text[i])
        for ch in chunk_text(combined, tokenizer, max_tokens=max_input_tokens):
            flat_texts.append(ch.text)
            owner.append(i)

    flat_summaries = _summarize_in_batches(flat_texts, tokenizer, model, device, batch_size)

    regrouped: dict[int, list[str]] = {i: [] for i in pending}
    for o, s in zip(owner, flat_summaries, strict=True):
        regrouped[o].append(s)
    for i in pending:
        summaries_per_text[i] = regrouped[i]


def hierarchical_summarize_batch(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_input_tokens: int = SUMMARY_MAX_INPUT_TOKENS,
    batch_size: int = SUMMARY_BATCH_SIZE,
) -> list[tuple[str, int]]:
    """Batched chunk-then-reduce summarization: chunks and reduces every
    text in `texts` independently (same per-text contract as a single-text
    version would have -- sentence-boundary chunking via chunk_text, then a
    recursive reduce pass over each text's own joined chunk-summaries until
    they collapse to one), but pools the model calls across every text still
    pending at each pass into batch_size-sized generate() calls instead of
    one call per text -- the same batching principle category's classifier
    applies to its forward pass. Returns (summary_text, num_chunks) pairs
    in the same order as `texts`, where num_chunks is each text's own
    leaf-level chunk count (>1 means that text needed a reduce pass).
    """
    n = len(texts)
    if n == 0:
        return []

    summaries_per_text, num_chunks = _leaf_summarize_batch(
        texts, tokenizer, model, device, max_input_tokens, batch_size
    )

    passes = 0
    pending = {i for i in range(n) if len(summaries_per_text[i]) > 1}
    while pending and passes < MAX_REDUCE_PASSES:
        _reduce_pass(
            pending, summaries_per_text, tokenizer, model, device, max_input_tokens, batch_size
        )
        passes += 1
        pending = {i for i in pending if len(summaries_per_text[i]) > 1}

    if pending:
        # MAX_REDUCE_PASSES exhausted without collapsing to one chunk for
        # some texts -- force a final pass; generate()'s own truncation=True
        # keeps this bounded even though it means the tail gets dropped.
        forced_positions = sorted(pending)
        forced_texts = [" ".join(summaries_per_text[i]) for i in forced_positions]
        forced_summaries = _summarize_in_batches(forced_texts, tokenizer, model, device, batch_size)
        for i, s in zip(forced_positions, forced_summaries, strict=True):
            summaries_per_text[i] = [s]

    return [
        (summaries_per_text[i][0] if summaries_per_text[i] else "", num_chunks[i]) for i in range(n)
    ]


class SummaryFeature(Feature[Any, str]):
    """`row` is a `Row`/dict-like object, matching
    `db.fetch_pending_company_summaries`'s return shape -- passed straight
    through to `db.build_company_summary_input`, which does all the actual
    text assembly."""

    def extract_one(self, tokenizer: Any, row: Any) -> str:
        return db.build_company_summary_input(row)


class SummaryTrainer(NoOpTrainer):
    """No fine-tuning step exists for `c_summary`'s pretrained checkpoint
    (see module docstring) -- behaviorally identical to `NoOpTrainer`, kept
    as its own named subclass only so this stage has a real `Trainer` type
    to hand to a future stage->Trainer registry (TASKS.md T-099)."""


class SummaryInference(Inference[Any, str]):
    """Batched chunk-then-reduce summarization (migrated onto the FTI
    hierarchy 2026-09-16, PLAN.md Work item 10 / TASKS.md T-086) -- see
    `hierarchical_summarize_batch` for the actual leaf-chunk + iterative-
    reduce-pass algorithm, which stays here (not `SummaryFeature`) since
    each reduce pass re-chunks and re-summarizes the *previous* pass's
    model output, making it inherently model-dependent at every step.

    `MODEL_NAME`/`MODEL_REVISIONS` below duplicate `pipeline.py`'s own
    `SUMMARY_MODEL`/`MODEL_REVISIONS[SUMMARY_MODEL]` as static defaults --
    `pipeline.run_company_summary_stage`'s thin wrapper always passes
    `model_name=`/`revision=`/`batch_size=` explicitly (reading its own
    module globals fresh at call time) -- see
    `sentiment_stage.SentimentInference`'s docstring for the same
    disclosed, acceptable duplication.
    """

    MODEL_NAME: ClassVar[str] = "sshleifer/distilbart-cnn-12-6"
    MODEL_REVISIONS: ClassVar[dict[str, str]] = {
        "sshleifer/distilbart-cnn-12-6": "a4f8f3ea906ed274767e9906dbaede7531d660ff",
    }
    STAGE_NAME: ClassVar[str] = "company_summary"

    def batch_size(self) -> int:
        """The constructor's `batch_size` override, or `SUMMARY_BATCH_SIZE`
        if none was given -- the real call path
        (`pipeline.run_company_summary_stage`) always passes one
        explicitly, so this fallback only matters for a hypothetical
        direct construction outside that path."""
        return (
            self._batch_size_override
            if self._batch_size_override is not None
            else SUMMARY_BATCH_SIZE
        )

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(self.model_name, revision=self.revision)
            .to(self.device)
            .eval()
        )

    def fetch_pending(
        self, conn: db.NewsNlpDatabase, limit: int | None, sample_seed: int | None
    ) -> list[Any]:
        return db.fetch_pending_company_summaries(conn, limit=limit)

    def predict_batch(self, features: FeatureBatch[str]) -> list[Any]:
        texts = features.items
        return hierarchical_summarize_batch(
            texts,
            self.tokenizer,
            self.model,
            self.device,
            max_input_tokens=SUMMARY_MAX_INPUT_TOKENS,
            batch_size=self.batch_size(),
        )

    def write_predictions(
        self, conn: db.NewsNlpDatabase, rows: Sequence[Any], predictions: list[Any]
    ) -> None:
        for row, (summary_text, num_chunks) in zip(rows, predictions, strict=True):
            if summary_text:
                db.write_company_summary(
                    conn, row["article_id"], summary_text, num_chunks, self.model_name
                )
