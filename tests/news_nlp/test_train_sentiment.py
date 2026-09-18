"""`train_sentiment.stratified_split`/`SentimentTrainConfig`: the train/test
split is now genuinely config-driven (TASKS.md T-096, PLAN.md Work item 11)
-- previously `stratified_split(rows)` was always called with no arguments,
silently relying on this module's own hardcoded `_SEED`/`_TEST_FRAC`/
`_VAL_FRAC` constants with no way to override them from a config or CLI
flag. These tests only exercise the pure, non-GPU split logic and the
config dataclass's own defaults -- `SentimentTrainer.train()`'s actual
fine-tuning path (real GPU work) is out of scope for this hermetic suite,
same as every other stage's training script.
"""

from __future__ import annotations

import train_sentiment as ts


def _rows(n_per_label: int) -> list[dict]:
    """`n_per_label` rows for each of the three sentiment labels, distinct
    text per row so misplacement is detectable."""
    rows = []
    for label_id in ts.LABEL2ID.values():
        for i in range(n_per_label):
            rows.append({"text": f"label{label_id}-row{i}", "label": label_id})
    return rows


def test_stratified_split_defaults_match_module_constants() -> None:
    """`SentimentTrainConfig`'s new fields default to this module's own
    pre-existing `_SEED`/`_TEST_FRAC`/`_VAL_FRAC` -- every existing
    invocation's behavior is unchanged unless a spec now explicitly
    overrides them."""
    config = ts.SentimentTrainConfig()
    assert config.split_seed == ts._SEED
    assert config.test_frac == ts._TEST_FRAC
    assert config.val_frac == ts._VAL_FRAC

    rows = _rows(20)
    default_call = ts.stratified_split(rows)
    explicit_call = ts.stratified_split(
        rows, seed=config.split_seed, test_frac=config.test_frac, val_frac=config.val_frac
    )
    assert default_call == explicit_call


def test_stratified_split_is_stratified_per_label() -> None:
    """Each label contributes proportionally to every split -- a plain
    random split could leave a rare class's test/validation slice empty."""
    rows = _rows(20)  # 60 rows total, 20 per label
    splits = ts.stratified_split(rows, seed=1, test_frac=0.1, val_frac=0.1)
    assert len(splits["train"]) + len(splits["validation"]) + len(splits["test"]) == 60
    for name, expected_per_label in (("test", 2), ("validation", 2)):
        counts = {label_id: 0 for label_id in ts.LABEL2ID.values()}
        for row in splits[name]:
            counts[row["label"]] += 1
        assert all(c == expected_per_label for c in counts.values()), (name, counts)


def test_stratified_split_seed_is_reproducible() -> None:
    rows = _rows(20)
    first = ts.stratified_split(rows, seed=7)
    second = ts.stratified_split(rows, seed=7)
    assert first == second


def test_stratified_split_test_frac_and_val_frac_are_config_driven() -> None:
    """Overriding test_frac/val_frac actually changes the split sizes --
    not just accepted and silently ignored."""
    rows = _rows(20)  # 20 per label
    small = ts.stratified_split(rows, seed=1, test_frac=0.1, val_frac=0.1)
    big = ts.stratified_split(rows, seed=1, test_frac=0.3, val_frac=0.3)
    assert len(big["test"]) > len(small["test"])
    assert len(big["validation"]) > len(small["validation"])
    assert len(big["train"]) < len(small["train"])


def test_stratified_split_min_one_row_per_label_even_at_tiny_frac() -> None:
    """`max(1, int(n * frac))` guarantees at least one row per label in
    test/validation, even when the fraction would otherwise round to 0."""
    rows = _rows(3)  # 3 per label -- 3 * 0.1 rounds to 0
    splits = ts.stratified_split(rows, seed=1, test_frac=0.1, val_frac=0.1)
    for label_id in ts.LABEL2ID.values():
        assert any(r["label"] == label_id for r in splits["test"])
        assert any(r["label"] == label_id for r in splits["validation"])


def test_sentiment_train_config_split_fields_are_overridable() -> None:
    config = ts.SentimentTrainConfig(split_seed=99, test_frac=0.25, val_frac=0.05)
    assert config.split_seed == 99
    assert config.test_frac == 0.25
    assert config.val_frac == 0.05
    # weighted/base_model (pre-existing fields) are untouched by this change.
    assert config.weighted is False
    assert config.base_model == ts.MODEL_NAME
