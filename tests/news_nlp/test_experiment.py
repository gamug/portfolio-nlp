"""`experiment.ExperimentSpec`: strict pydantic validation for the
JSON-driven experiment schema (PLAN.md Work item 11 step 1, TASKS.md
T-098, SPEC.md FR-017). Schema/validation only -- `run_experiment` itself
(TASKS.md T-099) is untested here, same split as the other tasks in this
work item.

Nested specs are constructed as real `PretrainSpec`/`EvalSpec`/
`PublishSpec` instances, not dict literals -- this project has no pydantic
mypy plugin configured, so a plain dict doesn't type-check against a
nested `BaseModel` field even though pydantic itself would coerce it at
runtime. `model_validate` calls (the realistic JSON-loading path) use raw
dicts on purpose, where that's exactly what's being exercised.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from experiment import EvalSpec, ExperimentSpec, PretrainSpec, PublishSpec, TrainTestSplitSpec
from news_nlp.eval.config import (
    DEFAULT_LOW_CONF_FRAC,
    DEFAULT_MAX_WORKERS,
    DEFAULT_REGRESSION_TOLERANCE,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_TARGET_FRAC,
)


def test_minimal_eval_only_spec_is_valid() -> None:
    spec = ExperimentSpec(name="sentiment_v2_eval_only", stage="sentiment", eval=EvalSpec())
    assert spec.pretrain.enabled is False
    assert spec.publish.enabled is False
    assert spec.eval.sample_size == 80  # EvalSettings' own default


def test_eval_defaults_match_eval_settings_defaults() -> None:
    """EvalSpec is a passthrough subset of EvalSettings -- its defaults must
    stay in sync, not silently drift to a different value."""
    spec = ExperimentSpec(name="x", stage="ner", eval=EvalSpec())
    assert spec.eval.sample_size == DEFAULT_SAMPLE_SIZE
    assert spec.eval.low_conf_frac == DEFAULT_LOW_CONF_FRAC
    assert spec.eval.target_frac == DEFAULT_TARGET_FRAC
    assert spec.eval.max_workers == DEFAULT_MAX_WORKERS
    assert spec.eval.regression_tolerance == DEFAULT_REGRESSION_TOLERANCE


def test_pretrain_spec_valid_for_sentiment_with_weighted_hyperparameter() -> None:
    spec = ExperimentSpec(
        name="sentiment_v4_class_weighted",
        stage="sentiment",
        pretrain=PretrainSpec(
            enabled=True, base_model="ProsusAI/finbert", hyperparameters={"weighted": True}
        ),
        eval=EvalSpec(sample_size=2000, seed=1),
    )
    assert spec.pretrain.hyperparameters == {"weighted": True}


def test_pretrain_spec_valid_for_ner_with_no_hyperparameters() -> None:
    """NerTrainConfig has zero fields today (train_ner.py takes no CLI
    flags) -- an empty hyperparameters dict must still validate."""
    spec = ExperimentSpec(
        name="ner_production",
        stage="ner",
        pretrain=PretrainSpec(enabled=True, base_model="nlpaueb/sec-bert-base"),
        eval=EvalSpec(),
    )
    assert spec.pretrain.hyperparameters == {}


@pytest.mark.parametrize("stage", ["category", "c_summary"])
def test_pretrain_is_rejected_for_no_trainable_checkpoint_stages(stage: str) -> None:
    with pytest.raises(ValidationError, match="not valid for stage"):
        ExperimentSpec(
            name="bad",
            stage=stage,  # type: ignore[arg-type]
            pretrain=PretrainSpec(enabled=True, base_model="x"),
            eval=EvalSpec(),
        )


def test_pretrain_enabled_without_base_model_is_rejected() -> None:
    with pytest.raises(ValidationError, match="base_model is required"):
        ExperimentSpec(
            name="bad", stage="sentiment", pretrain=PretrainSpec(enabled=True), eval=EvalSpec()
        )


def test_pretrain_with_unknown_hyperparameter_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown key"):
        ExperimentSpec(
            name="bad",
            stage="sentiment",
            pretrain=PretrainSpec(
                enabled=True,
                base_model="ProsusAI/finbert",
                hyperparameters={"bogus_flag": True},
            ),
            eval=EvalSpec(),
        )


def test_pretrain_with_any_hyperparameter_key_is_rejected_for_ner() -> None:
    """NerTrainConfig has no fields at all -- every hyperparameters key is
    unknown for this stage, not just a specific typo."""
    with pytest.raises(ValidationError, match="unknown key"):
        ExperimentSpec(
            name="bad",
            stage="ner",
            pretrain=PretrainSpec(
                enabled=True, base_model="nlpaueb/sec-bert-base", hyperparameters={"anything": 1}
            ),
            eval=EvalSpec(),
        )


def test_hyperparameters_cannot_duplicate_structural_split_fields() -> None:
    """base_model/split_seed/test_frac/val_frac are structured fields
    (PretrainSpec.base_model / .split) -- setting them again via
    hyperparameters must be rejected as unknown, not silently accepted as
    a second way to say the same thing."""
    with pytest.raises(ValidationError, match="unknown key"):
        ExperimentSpec(
            name="bad",
            stage="sentiment",
            pretrain=PretrainSpec(
                enabled=True, base_model="ProsusAI/finbert", hyperparameters={"split_seed": 7}
            ),
            eval=EvalSpec(),
        )


def test_publish_enabled_without_pretrain_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"publish\.enabled requires pretrain\.enabled"):
        ExperimentSpec(
            name="bad",
            stage="sentiment",
            eval=EvalSpec(),
            publish=PublishSpec(enabled=True, repo_id="x/y"),
        )


def test_publish_enabled_without_repo_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"publish\.repo_id is required"):
        ExperimentSpec(
            name="bad",
            stage="sentiment",
            pretrain=PretrainSpec(enabled=True, base_model="ProsusAI/finbert"),
            eval=EvalSpec(),
            publish=PublishSpec(enabled=True),
        )


def test_publish_valid_when_pretrain_enabled_and_repo_id_set() -> None:
    spec = ExperimentSpec(
        name="ok",
        stage="sentiment",
        pretrain=PretrainSpec(enabled=True, base_model="ProsusAI/finbert"),
        eval=EvalSpec(),
        publish=PublishSpec(enabled=True, repo_id="gamug/FinBERT-financial-news"),
    )
    assert spec.publish.repo_id == "gamug/FinBERT-financial-news"


def test_eval_candidate_model_is_rejected_when_pretrain_enabled() -> None:
    with pytest.raises(ValidationError, match=r"must be unset when pretrain\.enabled"):
        ExperimentSpec(
            name="bad",
            stage="sentiment",
            pretrain=PretrainSpec(enabled=True, base_model="ProsusAI/finbert"),
            eval=EvalSpec(candidate_model="some-model"),
        )


def test_eval_candidate_revision_is_rejected_when_pretrain_enabled() -> None:
    with pytest.raises(ValidationError, match=r"must be unset when pretrain\.enabled"):
        ExperimentSpec(
            name="bad",
            stage="sentiment",
            pretrain=PretrainSpec(enabled=True, base_model="ProsusAI/finbert"),
            eval=EvalSpec(candidate_revision="local"),
        )


def test_eval_candidate_fields_are_allowed_for_an_eval_only_spec() -> None:
    """An eval-only spec (pretrain disabled) is exactly how a candidate
    model gets scored without training a new checkpoint -- must not be
    rejected."""
    spec = ExperimentSpec(
        name="sentiment_v5_downstream_check",
        stage="sentiment",
        eval=EvalSpec(candidate_model="models/finbert-v5", candidate_revision="local"),
    )
    assert spec.eval.candidate_model == "models/finbert-v5"


def test_unknown_top_level_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(
            {"name": "bad", "stage": "sentiment", "eval": {}, "typo_field": True}
        )


def test_unknown_nested_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(
            {"name": "bad", "stage": "sentiment", "eval": {"typo_field": True}}
        )


def test_invalid_stage_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate({"name": "bad", "stage": "sector_summary", "eval": {}})


def test_json_round_trip_preserves_every_field() -> None:
    spec = ExperimentSpec(
        name="sentiment_v4_class_weighted",
        stage="sentiment",
        pretrain=PretrainSpec(
            enabled=True,
            base_model="ProsusAI/finbert",
            split=TrainTestSplitSpec(split_seed=7, test_frac=0.15, val_frac=0.15),
            hyperparameters={"weighted": True},
        ),
        eval=EvalSpec(sample_size=2000, seed=1, run_name="v4"),
        publish=PublishSpec(enabled=True, repo_id="gamug/FinBERT-financial-news"),
    )
    reparsed = ExperimentSpec.model_validate_json(spec.model_dump_json())
    assert reparsed == spec
