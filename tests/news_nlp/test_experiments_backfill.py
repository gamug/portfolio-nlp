"""Every backfilled `experiments/*.json` spec (TASKS.md T-101, PLAN.md Work
item 11 step 5) stays valid against `ExperimentSpec` -- catches a future
schema change silently breaking one of these historical records. Also
locks in `experiments/README.md`'s own disclosed gap #3: the v2/v3
sentiment specs are deliberately content-identical except for `name`
(same base_model/hyperparameters/split -- the real difference is which
data file happens to exist on disk at train time, which `ExperimentSpec`
has no field for) -- a future edit "de-duplicating" them without updating
the README would be a real regression, not a cleanup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiment import ExperimentSpec

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
_SPEC_PATHS = sorted(_EXPERIMENTS_DIR.glob("*.json"))


def test_experiments_directory_has_the_expected_eight_specs() -> None:
    names = {p.stem for p in _SPEC_PATHS}
    assert names == {
        "sentiment_v2_chunklevel_finetuned",
        "sentiment_v3_downsampled",
        "sentiment_v4_class_weighted",
        "sentiment_v5_secbert_base",
        "sentiment_chunklevel_base_finbert",
        "ner_production",
        "category_production",
        "c_summary_production",
    }


@pytest.mark.parametrize("path", _SPEC_PATHS, ids=lambda p: p.stem)
def test_backfilled_spec_validates(path: Path) -> None:
    spec = ExperimentSpec.model_validate_json(path.read_text())
    # The file's own name always matches spec.name -- <name>.json is how
    # cli/run_experiment.py's own result file naming (<spec.name>.result.json)
    # stays discoverable back to the config that produced it.
    assert spec.name == path.stem


def test_v2_and_v3_sentiment_specs_are_content_identical_except_name() -> None:
    v2 = ExperimentSpec.model_validate_json(
        (_EXPERIMENTS_DIR / "sentiment_v2_chunklevel_finetuned.json").read_text()
    )
    v3 = ExperimentSpec.model_validate_json(
        (_EXPERIMENTS_DIR / "sentiment_v3_downsampled.json").read_text()
    )
    assert v2.model_copy(update={"name": "x"}) == v3.model_copy(update={"name": "x"})


def test_category_and_c_summary_production_specs_never_enable_pretrain() -> None:
    """FR-011: neither stage has a trainable checkpoint -- ExperimentSpec
    itself would reject pretrain.enabled for either, but the backfilled
    "production config" specs should never even attempt it."""
    for name in ("category_production", "c_summary_production"):
        spec = ExperimentSpec.model_validate_json((_EXPERIMENTS_DIR / f"{name}.json").read_text())
        assert spec.pretrain.enabled is False
