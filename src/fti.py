"""Feature/Train/Inference (FTI) base classes shared across the four ML
stages (sentiment, NER, category, `c_summary`).

Formalizes a pattern that already existed informally in `pipeline.py` (each
stage: fetch pending rows, load a model at a pinned revision, chunk/batch/
predict, write results, free the model) into three independently-importable
base classes instead of four separate sets of module-level functions.
`sector_summary` (heuristic, deterministic, no model) is explicitly excluded
-- it has no Feature/Train/Inference shape to share (see
`news_nlp/sector_summary/`). PLAN.md Work item 10 / SPEC.md FR-011-FR-013.

Plain classes with `NotImplementedError`-raising template methods, not
`abc.ABC` (no precedent anywhere in this codebase) and not `typing.Protocol`
(reserved elsewhere in this repo -- `news_nlp/eval/judges.py`'s `JudgeAgent`
-- for a genuinely swappable *external* dependency, not this module's shape:
four concrete stages sharing real code via inheritance). Matches this
repo's existing concrete-inheritance precedent (`NewsNlpDatabase
(TwoTierDatabase)`, `train_sentiment.py`'s `WeightedLossTrainer(Trainer)`).

`Feature`, `Trainer`, and `Inference` are three independent top-level
classes, not nested under one umbrella `Stage` class: category and
`c_summary` have no training step today (pretrained/zero-shot as shipped),
and sharing one `NoOpTrainer` between them only works if `Trainer` is a
free-standing, independently-importable class -- nesting it inside a
per-stage class would force each stage to re-declare "no training" itself,
exactly what SPEC.md FR-011 says not to do. `Feature` is likewise reused by
both `Inference` (scoring) and, later, a stage's own `Trainer` (building
training examples) and `news_nlp/eval/`'s redesigned inference step (T-089,
FR-013).

No concrete stage subclasses live here yet -- migrating sentiment/NER/
category/`c_summary` onto these base classes is TASKS.md T-083-T-086,
deliberately out of scope for this module's introduction (T-082).
"""

import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from news_nlp.db import NewsNlpDatabase

# (stage_name, processed_count, total_count) -> None -- same shape as
# pipeline.py's own ProgressCallback, kept in sync deliberately.
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class FeatureBatch[FeatureT]:
    """One stage's extracted features for a batch of articles, order
    preserved. `FeatureT` is each stage's own per-article feature shape
    (e.g. a future `SentimentFeatures(chunks, weights)` or
    `CategoryFeatures(premise)`) -- deliberately not unified across stages,
    since forcing one shared field set onto every stage's own input shape
    would just recreate the "fake step to satisfy an interface" problem
    this module exists to avoid for `Trainer` (see `NoOpTrainer`)."""

    items: list[FeatureT]


@dataclass(frozen=True)
class TrainConfig:
    """Empty base marker for a stage's own training configuration. Every
    concrete stage subclasses this with its own fields (or none) instead of
    sharing one rigid dataclass -- e.g. a future `SentimentTrainConfig
    (weighted: bool = False, base_model: str = ..., output_dir: str = ...)`
    mirroring `train_sentiment.py`'s `--weighted`/`--base-model` flags, vs.
    a future `NerTrainConfig` with no fields at all (`train_ner.py` takes no
    CLI arguments today)."""


@dataclass(frozen=True)
class TrainedArtifact:
    """What a `Trainer.train()` call produced. `output_dir=None` marks a
    no-op trainer's result (category/`c_summary` -- see `NoOpTrainer`):
    there is nothing to point at, because nothing was trained. `metrics` is
    a plain dict, not a fixed schema -- mirrors `train_sentiment.py`'s
    richer `metrics_payload` (test_metrics/dataset_sizes/base_model/...),
    which has fields `train_ner.py`'s print-only-today metrics don't;
    forcing one shared metrics dataclass across both would either lose
    sentiment's fields or force NER to grow ones it doesn't have."""

    output_dir: str | None
    metrics: dict[str, Any] | None = None


class Feature[RowT, FeatureT]:
    """Base class for a stage's feature-extraction step: turns raw DB
    row(s) into the initial, stage-specific input shape its model needs --
    chunks+weights for sentiment, title+lead-chunk premises for category,
    leaf-level chunk texts for `c_summary`, the chunked-and-flattened input
    NER's own forward pass needs. Owns no model, no GPU, no DB connection --
    a pure function of (tokenizer, row(s)) -> `FeatureBatch[FeatureT]`, so
    it is unit-testable without a live model and reusable by a stage's
    `Inference` (scoring), later its `Trainer` (building training
    examples), and `news_nlp/eval/`'s redesigned inference step (T-089).

    `RowT` is whatever shape that stage's own `db.fetch_pending_*` already
    returns (a plain tuple for sentiment/NER/category today, a `Row`-like
    mapping for `c_summary`) -- deliberately not unified, since unifying
    four already-different return shapes isn't this module's job.

    Scope note: `Feature` owns the *initial* raw-row -> first-model-input
    transformation only. A stage whose own algorithm needs more chunking
    later in its own generation loop (`c_summary`'s recursive reduce pass,
    which calls the model repeatedly on its own prior output) does that
    directly inside its own `Inference.predict_batch`, reusing the free
    `chunking.chunk_text` helper the same way it can today -- `Feature` is
    the blessed "raw row -> this stage's first model input" step, not a
    mandatory choke point for every `chunk_text` call a stage's loop ever
    makes. NER's and `c_summary`'s current implementations
    (`pipeline._ner_batch`, `pipeline.hierarchical_summarize_batch`)
    interleave preprocessing with real model calls -- migrating them
    (T-084/T-086) means physically splitting the pure-preprocessing part
    into `extract_batch` and the model-calling part (forward pass,
    `merge_bio_predictions`, the reduce loop) into `predict_batch`.
    """

    def extract_one(self, tokenizer: Any, row: RowT) -> FeatureT:
        """Extract this stage's feature shape for exactly one row. Every
        concrete stage implements this, even a stage whose real batching is
        cross-article (NER) -- `extract_batch`'s default calls this in a
        loop; NER's own override replaces the loop, but this method still
        documents NER's per-article shape on its own."""
        raise NotImplementedError

    def extract_batch(self, tokenizer: Any, rows: Sequence[RowT]) -> FeatureBatch[FeatureT]:
        """Extract features for a batch of rows, `rows`' order preserved.
        Default: call `extract_one` once per row -- correct for any stage
        whose batching is just "loop over articles independently"
        (sentiment, category, `c_summary`'s leaf pass). NER overrides this
        directly instead, since its real batching (`_ner_batch` today) is
        genuinely cross-article: every chunk of every article in the batch
        is flattened into one padded tokenizer call, which cannot be
        expressed as independent per-row `extract_one` calls."""
        return FeatureBatch(items=[self.extract_one(tokenizer, row) for row in rows])


class Trainer[ConfigT: TrainConfig]:
    """Base class for a stage's one-time fine-tuning step (will wrap
    `train_sentiment.py`'s/`train_ner.py`'s existing `main()` logic once
    migrated). Not part of `run_pipeline` -- invoked standalone, same as
    today's scripts."""

    def train(self, config: ConfigT) -> TrainedArtifact:
        """Run this stage's full training loop against `config` and return
        where the result landed. A concrete subclass owns everything
        `train_sentiment.py`'s/`train_ner.py`'s `main()` does today:
        resolving the training pool, building HF `Dataset`s, constructing
        an HF `Trainer` (or a `Trainer` subclass like `WeightedLossTrainer`),
        `.train()`, writing metrics JSON where the stage does that today,
        `save_pretrained`/`save_model`."""
        raise NotImplementedError


class NoOpTrainer(Trainer[TrainConfig]):
    """Explicit, documented "this stage ships pretrained/zero-shot, there
    is no training step" answer -- shared verbatim by category's and
    `c_summary`'s future `Trainer` subclasses (SPEC.md FR-011 requires this
    be a real, documented subclass, not an omission). Calling `train()` is
    always safe and does no work, so nothing downstream needs an
    `isinstance` check to special-case "stages without training"."""

    def train(self, config: TrainConfig) -> TrainedArtifact:
        return TrainedArtifact(output_dir=None, metrics=None)


class Inference[RowT, FeatureT]:
    """Base class wrapping one stage's `run_<stage>_stage` shape today:
    fetch pending rows, print the banner, load a model at a pinned
    revision, iterate in batches calling this stage's own `Feature` +
    forward pass + write, report progress, always free the model.

    Every concrete subclass MUST import `AutoTokenizer`/`AutoModelForXxx`
    (whichever HF auto class it needs) as bare module-scope names inside
    its own defining module and call `.from_pretrained` on them directly
    inside `load_model` -- never receive them as a pre-bound attribute or
    via `importlib` -- so existing hermetic tests'
    `monkeypatch.setattr(<module>.AutoXxx, "from_pretrained", ...)`
    mechanism keeps working once a stage migrates its call site (only the
    monkeypatch *import path* changes, per TASKS.md T-083-T-086's own
    accepted criteria -- never the mechanism itself).
    """

    #: HF repo id for this stage's model -- a plain class attribute, same
    #: swappability as today's module-level SENTIMENT_MODEL/NER_MODEL/...:
    #: an experiment script can still do
    #: `SentimentInference.MODEL_NAME = "..."` exactly the way
    #: scripts/resample_sentiment_v3_2026_09_15.py does
    #: `pipeline.SENTIMENT_MODEL = ...` today -- no harder to swap.
    MODEL_NAME: ClassVar[str]
    #: {repo_id: pinned_revision_sha} for *this stage's* model(s) --
    #: mirrors today's shared MODEL_REVISIONS dict, but scoped per stage
    #: class rather than one dict shared across all four, since FR-011
    #: wants each stage's Inference importable/usable independently of the
    #: others.
    MODEL_REVISIONS: ClassVar[dict[str, str]]
    #: This stage's name for the banner / `on_progress`'s first positional
    #: argument -- "sentiment" / "ner" / "category" / "company_summary"
    #: (must match existing hermetic tests' exact on_progress assertions).
    STAGE_NAME: ClassVar[str]

    def __init__(
        self,
        feature: "Feature[RowT, FeatureT]",
        *,
        model_name: str | None = None,
        revision: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        """`model_name` overrides `type(self).MODEL_NAME` for just this
        instance (a scoped alternative to the class-attribute monkeypatch
        above -- e.g. `news_nlp.eval`'s own per-experiment model, T-089).
        `revision` likewise overrides `type(self).MODEL_REVISIONS[model_name]`
        directly -- needed by a caller reading its own current model-name/
        revision pin from elsewhere (e.g. `pipeline.py`'s
        `SENTIMENT_MODEL`/`MODEL_REVISIONS` module globals, which a resample
        script can monkeypatch before calling the stage) and passing it
        through explicitly, rather than relying on this class's own
        `MODEL_REVISIONS` dict being object-identical to that other source of
        truth. `device` overrides the auto-detected default the same way;
        default is always `torch.device("cuda" if torch.cuda.is_available()
        else "cpu")` -- never hardcoded "cuda", preserving the constitution's
        CPU-fallback requirement (AI behavior #2)."""
        self.feature = feature
        self.model_name = model_name or type(self).MODEL_NAME
        self._revision_override = revision
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Any = None
        self.tokenizer: Any = None

    @property
    def revision(self) -> str:
        return self._revision_override or type(self).MODEL_REVISIONS[self.model_name]

    def load_model(self) -> None:
        """Load this stage's tokenizer + model at `self.revision`,
        `.to(self.device).eval()`, assign to `self.tokenizer`/`self.model`.
        Must call `AutoTokenizer.from_pretrained`/
        `AutoModelForXxx.from_pretrained` as bare names imported at the top
        of the concrete subclass's own module -- see class docstring."""
        raise NotImplementedError

    def free_model(self) -> None:
        """Drop the loaded model/tokenizer and free GPU memory -- the same
        `del model, tokenizer; free_gpu()` discipline every stage already
        follows today, generalized once here instead of reimplemented four
        times. Overridable for a stage that ever needs extra cleanup; no
        stage needs to override it today."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def fetch_pending(
        self, conn: NewsNlpDatabase, limit: int | None, sample_seed: int | None
    ) -> list[RowT]:
        """This stage's own `db.fetch_pending_*` call. A stage whose fetch
        helper doesn't support sampling yet (category, `c_summary` --
        neither `fetch_pending_category_articles` nor
        `fetch_pending_company_summaries` takes `sample_seed` today) simply
        ignores the argument in its own override rather than `run` needing
        an `if stage supports it` branch."""
        raise NotImplementedError

    def batch_size(self) -> int | None:
        """Rows per `predict_batch` call, or `None` to run every pending
        row through one `predict_batch` call (sentiment's current shape --
        no batching across articles yet). Returns `NER_BATCH_SIZE`/
        `CATEGORY_BATCH_SIZE`/`SUMMARY_BATCH_SIZE`'s value for the other
        three once migrated. A method, not a class attribute, so a subclass
        can size it off `self.device` later without changing `run`'s loop."""
        return None

    def predict_batch(self, features: FeatureBatch[FeatureT]) -> list[Any]:
        """Run this stage's forward pass (+ postprocessing -- NER's
        `merge_bio_predictions`, category's two-level softmax, `c_summary`'s
        reduce loop) over one `FeatureBatch`, returning one stage-specific
        prediction per item, same order. The only method that touches
        `self.model`; split out from `write_predictions` so a future caller
        (T-089's eval reuse, SPEC.md FR-013) can score an article without
        writing anything to the RESULTS store."""
        raise NotImplementedError

    def write_predictions(
        self, conn: NewsNlpDatabase, rows: Sequence[RowT], predictions: list[Any]
    ) -> None:
        """Persist one batch's predictions via this stage's own
        `db.write_sentiment`/`db.write_entities`/`db.write_category`/
        `db.write_company_summary`. Does not commit -- `run` commits once
        per batch after this returns."""
        raise NotImplementedError

    def run(
        self,
        conn: NewsNlpDatabase,
        limit: int | None = None,
        on_progress: ProgressCallback | None = None,
        *,
        sample_seed: int | None = None,
    ) -> None:
        """The exact external contract every `run_<stage>_stage` function
        has today. Concrete stages should not need to override this --
        everything stage-specific lives in the methods above.

        Reports `on_progress` once per row within a batch (not once per
        batch), matching today's real `run_ner_stage`/`run_category_stage`
        behavior (`pipeline.py`'s own per-article `idx += 1; on_progress(...)`
        inside their batch loops) even though `batch_size()` may group
        several rows into one `predict_batch`/`write_predictions` call.

        `load_model`/`free_model` are wrapped in `try`/`finally` -- a small,
        deliberate hardening over today's `run_*_stage` functions, which
        leak a loaded model on a mid-batch exception. Doesn't change any
        currently-tested behavior; called out here rather than introduced
        silently."""
        rows = self.fetch_pending(conn, limit, sample_seed)
        total = len(rows)
        print(f"\n=== {type(self).STAGE_NAME} stage ({self.model_name}) on {self.device} ===")
        if on_progress:
            on_progress(type(self).STAGE_NAME, 0, total)
        if total == 0:
            return  # never loads a model when nothing is pending

        self.load_model()
        try:
            idx = 0
            step = self.batch_size() or total
            for start in range(0, total, step):
                batch_rows = rows[start : start + step]
                features = self.feature.extract_batch(self.tokenizer, batch_rows)
                predictions = self.predict_batch(features)
                self.write_predictions(conn, batch_rows, predictions)
                conn.commit()
                for _ in batch_rows:
                    idx += 1
                    if on_progress:
                        on_progress(type(self).STAGE_NAME, idx, total)
        finally:
            self.free_model()
