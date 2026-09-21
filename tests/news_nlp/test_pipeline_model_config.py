"""`pipeline_config.PipelineModelsConfig`: strict pydantic validation for
`config/pipeline_models.json`, the git-tracked production model-selection
file (SPEC.md FR-018, PLAN.md Work item 14).

Schema/validation and the pipeline.py wiring contract only -- not the
model-loading behavior itself, unchanged from before this work item and
already covered by each stage's own hermetic pipeline test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import pipeline
from pipeline_config import PipelineModelsConfig, load_pipeline_models_config


def test_real_config_file_parses() -> None:
    config = load_pipeline_models_config()
    assert config.sentiment.model == "gamug/FinBERT-financial-news"
    assert config.sentiment.revision == "93863fcb7252874e7c0339081b34f691f9e17ff6"
    assert config.ner.model == "gamug/sec-bert-finer-ord-ner"
    assert config.ner.revision == "ba7b9e43e4aa023ec5691f955b276dc58158354c"
    assert config.ner.batch_size == 8
    assert config.category.model == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    assert config.category.revision == "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8"
    assert config.category.batch_size == 8
    assert config.c_summary.model == "sshleifer/distilbart-cnn-12-6"
    assert config.c_summary.revision == "a4f8f3ea906ed274767e9906dbaede7531d660ff"
    assert config.c_summary.batch_size == 4


def test_pipeline_globals_match_a_fresh_config_load() -> None:
    """Catches a future copy-paste mistake in pipeline.py's field-to-global
    mapping (e.g. swapping ner.model/category.model)."""
    config = load_pipeline_models_config()
    assert config.sentiment.model == pipeline.SENTIMENT_MODEL
    assert config.ner.model == pipeline.NER_MODEL
    assert config.category.model == pipeline.CATEGORY_MODEL
    assert config.c_summary.model == pipeline.SUMMARY_MODEL
    assert pipeline.MODEL_REVISIONS[pipeline.SENTIMENT_MODEL] == config.sentiment.revision
    assert pipeline.MODEL_REVISIONS[pipeline.NER_MODEL] == config.ner.revision
    assert pipeline.MODEL_REVISIONS[pipeline.CATEGORY_MODEL] == config.category.revision
    assert pipeline.MODEL_REVISIONS[pipeline.SUMMARY_MODEL] == config.c_summary.revision
    assert config.ner.batch_size == pipeline.NER_BATCH_SIZE
    assert config.category.batch_size == pipeline.CATEGORY_BATCH_SIZE
    assert config.c_summary.batch_size == pipeline.SUMMARY_BATCH_SIZE


def test_model_revisions_is_a_real_mutable_dict() -> None:
    """scripts/resample_sentiment_v{3,4,5}_2026_09_15.py mutate
    pipeline.MODEL_REVISIONS[candidate] = "local" in-process at runtime --
    protects that pattern from a future refactor to an immutable mapping."""
    assert type(pipeline.MODEL_REVISIONS) is dict


_VALID: dict[str, dict[str, str | int]] = {
    "sentiment": {"model": "a/b", "revision": "abc"},
    "ner": {"model": "a/b", "revision": "abc", "batch_size": 8},
    "category": {"model": "a/b", "revision": "abc", "batch_size": 8},
    "c_summary": {"model": "a/b", "revision": "abc", "batch_size": 4},
}


def test_valid_config_round_trips() -> None:
    config = PipelineModelsConfig.model_validate(_VALID)
    assert config.sentiment.model == "a/b"


def test_unknown_top_level_key_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineModelsConfig.model_validate({**_VALID, "sector_summary": {"model": "a/b"}})


def test_missing_stage_rejected() -> None:
    bad = {k: v for k, v in _VALID.items() if k != "ner"}
    with pytest.raises(ValidationError):
        PipelineModelsConfig.model_validate(bad)


def test_non_positive_batch_size_rejected() -> None:
    bad = {**_VALID, "ner": {**_VALID["ner"], "batch_size": 0}}
    with pytest.raises(ValidationError):
        PipelineModelsConfig.model_validate(bad)


def test_sentiment_batch_size_field_rejected() -> None:
    """sentiment has no batch_size knob -- nothing in sentiment_stage.py
    reads one, so extra="forbid" should reject it rather than silently
    accept a dead field."""
    bad = {**_VALID, "sentiment": {**_VALID["sentiment"], "batch_size": 1}}
    with pytest.raises(ValidationError):
        PipelineModelsConfig.model_validate(bad)


def test_load_from_explicit_path(tmp_path: Path) -> None:
    config_file = tmp_path / "pipeline_models.json"
    config_file.write_text(
        '{"sentiment": {"model": "a/b", "revision": "abc"}, '
        '"ner": {"model": "a/b", "revision": "abc", "batch_size": 1}, '
        '"category": {"model": "a/b", "revision": "abc", "batch_size": 1}, '
        '"c_summary": {"model": "a/b", "revision": "abc", "batch_size": 1}}'
    )
    config = load_pipeline_models_config(path=config_file)
    assert config.ner.batch_size == 1
