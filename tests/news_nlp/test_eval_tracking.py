"""news_nlp.eval.tracking: MLflow file-store logging + previous-run lookup."""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

pytest.importorskip("mlflow")

import mlflow

from news_nlp.eval.tracking import log_to_mlflow, previous_headline


def _uri(tmp_path: Path) -> str:
    return str(tmp_path / "mlruns")


def test_log_to_mlflow_writes_run_metrics_and_artifacts(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    run_id = log_to_mlflow(
        stage="sentiment",
        params={"stage": "sentiment", "sample_size": 10, "seed": 1},
        metrics={"macro_f1_vs_judge": 0.82, "n": 10.0, "bogus_nan": math.nan},
        judgements=[{"article_id": 1, "bucket": "low_conf", "verdict": {"agrees": True}}],
        system_prompt="SYS PROMPT",
        tracking_uri=uri,
    )
    assert run_id

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["news_nlp_eval/sentiment"])
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.macro_f1_vs_judge"] == 0.82
    # non-finite metric filtered out
    assert "metrics.bogus_nan" not in runs.columns

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert {"judgements.json", "judge_prompt.md"} <= artifacts


def test_log_to_mlflow_run_name_is_cosmetic_only(tmp_path: Path) -> None:
    """--run-name labels the run in the MLflow UI's run list without changing
    which experiment it lands in or breaking previous_headline's ordering."""
    uri = _uri(tmp_path)
    run_id = log_to_mlflow(
        stage="sentiment",
        params={"stage": "sentiment"},
        metrics={"precision_negative": 0.5},
        judgements=[],
        system_prompt="p",
        tracking_uri=uri,
        run_name="v5-sec-bert-base",
    )
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.info.run_name == "v5-sec-bert-base"
    # still the same fixed experiment, named runs and default-named runs coexist
    exp = client.get_experiment_by_name("news_nlp_eval/sentiment")
    assert run.info.experiment_id == exp.experiment_id


def test_previous_headline_returns_prior_run_value(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    for value in (0.90, 0.70):
        log_to_mlflow(
            stage="category",
            params={"stage": "category"},
            metrics={"accuracy_vs_judge": value},
            judgements=[],
            system_prompt="p",
            tracking_uri=uri,
        )
        time.sleep(0.05)  # keep the two runs' start_time ordered in the file store
    # most recent run is 0.70; the prior one is 0.90
    assert previous_headline("category", "accuracy_vs_judge", uri) == 0.90


def test_previous_headline_none_without_history(tmp_path: Path) -> None:
    assert previous_headline("ner", "micro_f1", _uri(tmp_path)) is None
