"""`train_ner.NerTrainConfig`'s `base_model` field (TASKS.md T-099, PLAN.md
Work item 11) -- previously this class had no fields at all, which left
`ExperimentSpec.pretrain.base_model` silently ignored for the NER stage
(`NerTrainer.train` always fine-tuned the hardcoded `MODEL_NAME` regardless
of what config it was handed). `NerTrainer.train()`'s actual fine-tuning
path (real GPU work, a real dataset download) is out of scope for this
hermetic suite, same as `test_train_sentiment.py`'s own scope boundary --
this only exercises the config dataclass's own defaults.
"""

from __future__ import annotations

import train_ner as tn


def test_ner_train_config_default_matches_module_constant() -> None:
    config = tn.NerTrainConfig()
    assert config.base_model == tn.MODEL_NAME


def test_ner_train_config_base_model_is_overridable() -> None:
    config = tn.NerTrainConfig(base_model="some-other/checkpoint")
    assert config.base_model == "some-other/checkpoint"
