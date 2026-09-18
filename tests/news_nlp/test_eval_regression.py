"""news_nlp.eval.regression.check_regression against a tmp MLflow file store."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("mlflow")

from news_nlp.eval.regression import check_regression
from news_nlp.eval.tracking import log_to_mlflow


def _seed_runs(uri: str, stage: str, metric: str, values: list[float]) -> None:
    for v in values:
        log_to_mlflow(
            stage=stage,
            params={"stage": stage},
            metrics={metric: v},
            judgements=[],
            system_prompt="p",
            tracking_uri=uri,
        )
        time.sleep(0.05)


def test_flags_a_drop_past_tolerance(tmp_path: Path) -> None:
    uri = str(tmp_path / "mlruns")
    _seed_runs(uri, "category", "accuracy_vs_judge", [0.90, 0.62])

    rr = check_regression("category", {"accuracy_vs_judge": 0.62}, tolerance=0.05, tracking_uri=uri)
    assert rr.previous == 0.90
    assert rr.current == 0.62
    assert rr.regressed is True
    assert "REGRESSED" in rr.describe()


def test_small_drop_within_tolerance_is_ok(tmp_path: Path) -> None:
    # sentiment's headline is recall_negative (see metrics.HEADLINE) -- missing
    # real negative-sentiment articles is the regression that matters here.
    uri = str(tmp_path / "mlruns")
    _seed_runs(uri, "sentiment", "recall_negative", [0.80, 0.78])

    rr = check_regression("sentiment", {"recall_negative": 0.78}, tolerance=0.05, tracking_uri=uri)
    assert rr.regressed is False


def test_no_prior_run_is_not_a_regression(tmp_path: Path) -> None:
    rr = check_regression(
        "ner", {"micro_f1": 0.5}, tolerance=0.05, tracking_uri=str(tmp_path / "mlruns")
    )
    assert rr.previous is None
    assert rr.regressed is False
    assert "no prior run" in rr.describe()


def test_regression_check_is_experiment_scoped(tmp_path: Path) -> None:
    """A lone candidate-model run must never be flagged as a regression
    just because production's ("base") own number happens to be higher --
    2026-09-18 fix (docs/evaluation.md)."""
    uri = str(tmp_path / "mlruns")
    log_to_mlflow(
        stage="category",
        params={"stage": "category"},
        metrics={"accuracy_vs_judge": 0.90},
        judgements=[],
        system_prompt="p",
        tracking_uri=uri,
        experiment="base",
    )
    time.sleep(0.05)
    log_to_mlflow(
        stage="category",
        params={"stage": "category"},
        metrics={"accuracy_vs_judge": 0.30},
        judgements=[],
        system_prompt="p",
        tracking_uri=uri,
        experiment="candidate",
    )
    time.sleep(0.05)

    rr = check_regression(
        "category",
        {"accuracy_vs_judge": 0.30},
        tolerance=0.05,
        tracking_uri=uri,
        experiment="candidate",
    )
    assert rr.previous is None
    assert rr.regressed is False
