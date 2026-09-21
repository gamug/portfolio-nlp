"""Schema + loader for `config/pipeline_models.json` -- the production model
selection (name/revision/batch size per ML stage) that `pipeline.py` loads
at import time. SPEC.md FR-018, PLAN.md Work item 14.

A git-tracked, fixed-path config file, not a runtime-selectable one:
constitution AI-behavior #1 ("model selection is pinned, not dynamic --
swapping a model is a spec-level change") rules out a `--model-config
<path>` CLI flag, since that would let anyone point the pipeline at an
unreviewed file. Changing a model still means editing this file and going
through normal PR review -- only the format moved, from a Python literal to
JSON, matching `experiment.py`'s existing JSON+pydantic convention
(`ExperimentSpec`).

`_StrictModel` (`extra="forbid"`) is duplicated from `experiment.py` rather
than imported -- it's a private name there, and the two schemas should stay
independently importable/testable without coupling one to the other.

`sentiment` has no `batch_size` field, unlike the other three stages:
`sentiment_stage.SentimentInference.batch_size()` is a hardcoded override
returning `1` (see that file), not a tunable constant `pipeline.py` reads --
there is nothing for a `sentiment.batch_size` field to move.

No shape/regex validation on `revision` (matching `experiment.py`'s own
`EvalSpec.candidate_revision`) -- just requires it non-empty.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "pipeline_models.json"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SentimentModelConfig(_StrictModel):
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)


class StageModelConfig(_StrictModel):
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    batch_size: int = Field(gt=0)


class PipelineModelsConfig(_StrictModel):
    sentiment: SentimentModelConfig
    ner: StageModelConfig
    category: StageModelConfig
    c_summary: StageModelConfig


def load_pipeline_models_config(path: Path = CONFIG_PATH) -> PipelineModelsConfig:
    return PipelineModelsConfig.model_validate_json(path.read_text())
