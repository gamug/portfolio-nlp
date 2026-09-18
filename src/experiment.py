"""`ExperimentSpec` -- a single, JSON-validated file that fully describes one
model experiment for any of the four ML stages (sentiment/NER/category/
`c_summary`; `sector_summary` excluded -- no Feature/Train/Inference shape,
FR-012/FR-017). PLAN.md Work item 11 step 1 / TASKS.md T-098, SPEC.md
FR-017.

Reproducibility guarantee, not a security mechanism: strict pydantic
validation rejects an ambiguous or unpinned config before any real
train/eval work starts, with a message naming exactly what's wrong --
matching this project's other schema-validated config
(`news_nlp.eval.config.EvalSettings`).

Nested specs mirror the real shapes they configure, not an invented generic
one:

- `PretrainSpec.base_model` / `.split` / `.hyperparameters` line up with
  `SentimentTrainConfig`'s/`NerTrainConfig`'s own real dataclass fields
  (`src/train_sentiment.py`/`src/train_ner.py`) -- `base_model` and the
  train/test split knobs (`TrainTestSplitSpec`, mirroring TASKS.md T-096's
  `split_seed`/`test_frac`/`val_frac`) are structured fields because every
  trainable stage has them; `hyperparameters` is a free-form dict for
  whatever's left (e.g. sentiment's `weighted`), validated by name against
  that stage's real `TrainConfig` subclass so a typo'd or nonexistent
  hyperparameter fails fast instead of being silently ignored by
  `dataclasses.replace`-style construction downstream.
- `EvalSpec` is a direct passthrough subset of
  `news_nlp.eval.config.EvalSettings`'s own non-secret fields (sample
  size/stratification/seed/run name/candidate override) -- deliberately
  excludes `llm_api_key`/`llm_model`/`llm_url`/`mlflow_tracking_uri`, which
  come from `.env`/CLI flags, never a git-tracked JSON file.
- `PublishSpec` only records *that* a Hub publish was requested and where
  -- it does not template a model card. TASKS.md T-099's own scope note:
  actually executing a `publish.enabled=true` spec still needs the same
  explicit, separate confirmation any Hub push already requires (the
  existing `scripts/publish_*.py` pattern) -- this schema doesn't change
  that gate, just lets a spec name the intent up front.

Stage rejects `pretrain` outright for category/`c_summary`: neither ships
a trainable checkpoint today, or ever will (both zero-shot/pretrained as
designed, FR-011) -- `CategoryTrainer`/`SummaryTrainer` are trivial
`NoOpTrainer` subclasses (TASKS.md T-097) with nothing to configure.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fti import NoOpTrainer, TrainConfig, Trainer
from news_nlp.eval.config import EvalSettings
from news_nlp.eval.provenance import code_version
from news_nlp.eval.runner import run_eval

#: Where `run_experiment` writes its git-tracked result records (TASKS.md
#: T-099) -- `experiments/`'s own JSON-spec siblings (TASKS.md T-101) live
#: one directory up, at `experiments/`.
RESULTS_DIR = Path("experiments/results")

#: The four ML stages this schema covers -- `sector_summary` deliberately
#: excluded (see module docstring).
Stage = Literal["sentiment", "ner", "category", "c_summary"]

#: Stages with no trainable checkpoint -- `pretrain.enabled` is always
#: rejected for these (FR-011).
_NO_PRETRAIN_STAGES: frozenset[str] = frozenset({"category", "c_summary"})

#: `TrainTestSplitSpec`'s own fields, plus `PretrainSpec.base_model` --
#: structured fields every trainable stage's real `TrainConfig` happens to
#: share the same names for (`SentimentTrainConfig`, TASKS.md T-096), so
#: they're never also valid `hyperparameters` keys -- a spec author can't
#: set `base_model` two different ways.
_STRUCTURAL_TRAIN_FIELDS: frozenset[str] = frozenset(
    {"base_model", "split_seed", "test_frac", "val_frac"}
)


def _train_config_class(stage: Stage) -> type[TrainConfig]:
    """This stage's real `Trainer[ConfigT]`'s `ConfigT` -- imported lazily
    (not at module scope) so validating a spec never pays the
    torch/transformers import cost `train_sentiment`/`train_ner` carry,
    unless that stage is actually the one being validated."""
    if stage == "sentiment":
        from train_sentiment import SentimentTrainConfig  # noqa: PLC0415

        return SentimentTrainConfig
    if stage == "ner":
        from train_ner import NerTrainConfig  # noqa: PLC0415

        return NerTrainConfig
    # category/c_summary: pretrain is always rejected before this is ever
    # called (see ExperimentSpec's own validator) -- the bare marker class
    # is the honest answer for "no fields" rather than a stage-specific
    # empty subclass neither category_stage.py nor summary_stage.py define.
    return TrainConfig


def _trainer_class(stage: Stage) -> type[Trainer[Any]]:
    """This stage's real `Trainer` subclass, mirroring
    `_train_config_class` (same lazy-import reasoning). category/
    `c_summary` structurally can never reach `run_experiment`'s training
    step (`ExperimentSpec`'s own validator rejects `pretrain.enabled` for
    both) -- `NoOpTrainer` is still the correct, symmetrical answer here,
    same as `_train_config_class`'s bare `TrainConfig`."""
    if stage == "sentiment":
        from train_sentiment import SentimentTrainer  # noqa: PLC0415

        return SentimentTrainer
    if stage == "ner":
        from train_ner import NerTrainer  # noqa: PLC0415

        return NerTrainer
    return NoOpTrainer


def _allowed_hyperparameter_keys(stage: Stage) -> frozenset[str]:
    fields = {f.name for f in dataclasses.fields(_train_config_class(stage))}
    return frozenset(fields - _STRUCTURAL_TRAIN_FIELDS)


class _StrictModel(BaseModel):
    """Shared base: an unrecognized field (a typo'd key, a renamed one)
    fails validation instead of being silently dropped -- a reproducibility
    guarantee, matching this schema's own stated purpose (module
    docstring)."""

    model_config = ConfigDict(extra="forbid")


class TrainTestSplitSpec(_StrictModel):
    """Mirrors `SentimentTrainConfig`'s `split_seed`/`test_frac`/`val_frac`
    (TASKS.md T-096) -- `None` for any field means "use that stage's own
    `TrainConfig` default", not a spec-level default duplicated here, so
    this schema never drifts from the real one if a stage's own default
    ever changes."""

    split_seed: int | None = None
    test_frac: float | None = Field(default=None, gt=0.0, lt=1.0)
    val_frac: float | None = Field(default=None, gt=0.0, lt=1.0)


class PretrainSpec(_StrictModel):
    """Whether this experiment trains a new checkpoint, and from what."""

    enabled: bool = False
    base_model: str | None = None
    split: TrainTestSplitSpec = Field(default_factory=TrainTestSplitSpec)
    #: Stage-specific knobs beyond `base_model`/`split` (e.g. sentiment's
    #: `weighted`) -- validated by name against that stage's real
    #: `TrainConfig` fields in `ExperimentSpec`'s own validator, not here
    #: (needs `stage`, a sibling field, not available on this nested model
    #: alone).
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class EvalSpec(_StrictModel):
    """Direct passthrough subset of `EvalSettings`' own non-secret fields
    (see module docstring) -- `sample_size`/stratification/`seed`/
    `run_name` always apply; `candidate_model`/`candidate_revision` are
    only user-settable for an eval-only spec (`pretrain.enabled=False`) --
    a `pretrain.enabled=true` spec auto-resolves them downstream
    (TASKS.md T-099) to the freshly-trained local checkpoint, so setting
    either here too would be an unreproducible ambiguity: which model
    actually got evaluated?"""

    sample_size: int = Field(default=80, gt=0)
    low_conf_frac: float = Field(default=0.2, ge=0.0, le=1.0)
    target_frac: float = Field(default=0.6, ge=0.0, le=1.0)
    seed: int | None = None
    max_workers: int = Field(default=4, gt=0)
    run_name: str | None = None
    candidate_model: str | None = None
    candidate_revision: str | None = None
    candidate_prescore_size: int | None = None
    check_regression: bool = False
    regression_tolerance: float = Field(default=0.05, ge=0.0)


class PublishSpec(_StrictModel):
    """Records *that* a Hub publish is wanted and where -- does not
    template a model card (see module docstring)."""

    enabled: bool = False
    repo_id: str | None = None


class ExperimentSpec(_StrictModel):
    """One file, one experiment: whether it trains a new checkpoint (and
    from what), how the result is evaluated, and whether to publish it.
    `stage`-generic across sentiment/NER/category/`c_summary`."""

    name: str = Field(min_length=1)
    stage: Stage
    pretrain: PretrainSpec = Field(default_factory=PretrainSpec)
    eval: EvalSpec
    publish: PublishSpec = Field(default_factory=PublishSpec)

    @model_validator(mode="after")
    def _validate_cross_field_rules(self) -> ExperimentSpec:
        if self.pretrain.enabled:
            if self.stage in _NO_PRETRAIN_STAGES:
                raise ValueError(
                    f"pretrain.enabled is not valid for stage={self.stage!r} -- "
                    f"{sorted(_NO_PRETRAIN_STAGES)} ship pretrained/zero-shot, "
                    "with no trainable checkpoint (FR-011)"
                )
            if not self.pretrain.base_model:
                raise ValueError(
                    "pretrain.base_model is required (and must be a real, "
                    "pinned HF repo id) when pretrain.enabled is true"
                )
            allowed = _allowed_hyperparameter_keys(self.stage)
            unknown = set(self.pretrain.hyperparameters) - allowed
            if unknown:
                raise ValueError(
                    f"pretrain.hyperparameters has unknown key(s) {sorted(unknown)} for "
                    f"stage={self.stage!r} -- allowed: {sorted(allowed)}"
                )
            if self.eval.candidate_model is not None or self.eval.candidate_revision is not None:
                raise ValueError(
                    "eval.candidate_model/candidate_revision must be unset when "
                    "pretrain.enabled is true -- auto-resolved downstream to the "
                    "freshly-trained local checkpoint, never user-supplied here"
                )

        if self.publish.enabled:
            if not self.pretrain.enabled:
                raise ValueError("publish.enabled requires pretrain.enabled")
            if not self.publish.repo_id:
                raise ValueError("publish.repo_id is required when publish.enabled is true")

        return self


class ExperimentResult(BaseModel):
    """What one `run_experiment` call produced: the resolved spec it ran
    (so the result file is self-describing, not just a bag of metrics),
    this checkout's `code_version`, the training step's own
    `TrainedArtifact` fields (`None`/`None` for an eval-only spec), and
    `run_eval`'s own returned dict verbatim. `result_path` is where this
    object itself got written -- the same path a caller could re-read it
    from."""

    spec: ExperimentSpec
    code_version: str
    train_output_dir: str | None
    train_metrics: dict[str, Any] | None
    eval_results: dict[str, dict[str, Any]]
    result_path: str


def run_experiment(
    spec: ExperimentSpec,
    *,
    source_db: str | None = None,
    results_db: str | None = None,
) -> ExperimentResult:
    """Run one experiment end to end (PLAN.md Work item 11 steps 1/3,
    TASKS.md T-099): train a new checkpoint if `spec.pretrain.enabled`,
    evaluate it (or, for an eval-only spec, whatever `spec.eval` already
    names) via `news_nlp.eval.runner.run_eval` -- reused verbatim, not
    reimplemented -- then write a git-tracked
    `experiments/results/<spec.name>.result.json`.

    On success after training, `candidate_model`/`candidate_revision`
    auto-resolve to the fresh local checkpoint (`revision="local"`, the
    established convention
    `scripts/resample_sentiment_v4_2026_09_15.py` already set --
    `from_pretrained` ignores `revision` entirely for a local directory
    path) -- `ExperimentSpec`'s own validator already guarantees
    `spec.eval.candidate_model`/`candidate_revision` are unset whenever
    `spec.pretrain.enabled`, so there is no ambiguity to resolve between a
    user-supplied value and this one.

    Publishing (`spec.publish.enabled`) is recorded in the resolved
    `spec` this function writes out, but never executed here -- a Hub push
    keeps the same explicit, separate confirmation any of the existing
    `scripts/publish_*.py` scripts already require (TASKS.md T-073's own
    still-pending status is the live proof that gate isn't bypassed); this
    task does not change it.

    `check_regression=True` reuses `run_eval`'s own existing behavior
    verbatim, including its `SystemExit(1)` on a regression past
    tolerance (the same behavior `cli/news_nlp_eval.py` already has for
    this flag, uncaught) -- deliberately not wrapped in a
    try/except here, so no result file is written for a run that
    regressed past tolerance. A future CLI (TASKS.md T-100) decides
    what to do with that; this function doesn't paper over it.
    """
    train_output_dir: str | None = None
    train_metrics: dict[str, Any] | None = None
    candidate_model = spec.eval.candidate_model
    candidate_revision = spec.eval.candidate_revision

    if spec.pretrain.enabled:
        config_cls = _train_config_class(spec.stage)
        trainer_cls = _trainer_class(spec.stage)
        config_kwargs: dict[str, Any] = {"base_model": spec.pretrain.base_model}
        if spec.pretrain.split.split_seed is not None:
            config_kwargs["split_seed"] = spec.pretrain.split.split_seed
        if spec.pretrain.split.test_frac is not None:
            config_kwargs["test_frac"] = spec.pretrain.split.test_frac
        if spec.pretrain.split.val_frac is not None:
            config_kwargs["val_frac"] = spec.pretrain.split.val_frac
        config_kwargs.update(spec.pretrain.hyperparameters)
        config = config_cls(**config_kwargs)

        artifact = trainer_cls().train(config)
        train_output_dir = artifact.output_dir
        train_metrics = artifact.metrics
        candidate_model = train_output_dir
        candidate_revision = "local"

    settings = EvalSettings.load(
        sample_size=spec.eval.sample_size,
        low_conf_frac=spec.eval.low_conf_frac,
        target_frac=spec.eval.target_frac,
        seed=spec.eval.seed,
        max_workers=spec.eval.max_workers,
        run_name=spec.eval.run_name or spec.name,
        candidate_model=candidate_model,
        candidate_revision=candidate_revision,
        candidate_prescore_size=spec.eval.candidate_prescore_size,
    )
    eval_results = run_eval(
        [spec.stage],
        settings=settings,
        source_db=source_db,
        results_db=results_db,
        check_regression=spec.eval.check_regression,
        regression_tolerance=spec.eval.regression_tolerance,
    )

    result = ExperimentResult(
        spec=spec,
        code_version=code_version(),
        train_output_dir=train_output_dir,
        train_metrics=train_metrics,
        eval_results=eval_results,
        result_path=str(RESULTS_DIR / f"{spec.name}.result.json"),
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(result.result_path).write_text(result.model_dump_json(indent=2) + "\n")
    return result
