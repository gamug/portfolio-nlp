"""MLflow logging for an eval run: metrics + a per-row ``judgements.json`` artifact.

One MLflow run per ``(stage, invocation)``, in experiment ``news_nlp_eval/<stage>``.
The tracking URI defaults to ``./mlruns`` (a local file store, ``uv run mlflow ui``);
set ``MLFLOW_TRACKING_URI`` / ``--mlflow-uri`` to a shared server for the scheduled
``--check-regression`` job. Imports ``mlflow`` at module load -- the ``eval`` group.
"""

from __future__ import annotations

import math
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

_BUCKET_SPLIT_TAG = "60/40 low_conf/random"
_MIN_RUNS_TO_COMPARE = 2  # the fresh run plus one prior


def experiment_name(stage: str) -> str:
    return f"news_nlp_eval/{stage}"


def log_to_mlflow(
    *,
    stage: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    judgements: list[dict[str, Any]],
    system_prompt: str,
    tracking_uri: str,
) -> str:
    """Create the run, log params/metrics/artifacts, return the MLflow run id."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name(stage))
    with mlflow.start_run() as run:
        mlflow.set_tags({"stage": stage, "bucket_split": _BUCKET_SPLIT_TAG})
        mlflow.log_params(params)
        # mlflow rejects non-finite / non-numeric metric values; filter defensively.
        clean = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, int | float) and math.isfinite(v)
        }
        mlflow.log_metrics(clean)
        mlflow.log_dict({"stage": stage, "judgements": judgements}, "judgements.json")
        mlflow.log_text(system_prompt, "judge_prompt.md")
        return str(run.info.run_id)


def previous_headline(stage: str, metric: str, tracking_uri: str) -> float | None:
    """The value of *metric* on the run just before the most recent one for
    *stage*, or ``None`` if there is no such prior run. Called after the fresh
    run is logged, so index 0 is that fresh run and index 1 is the comparison."""
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name(stage))
    if exp is None:
        return None
    runs = client.search_runs(
        [exp.experiment_id], order_by=["start_time DESC"], max_results=_MIN_RUNS_TO_COMPARE
    )
    if len(runs) < _MIN_RUNS_TO_COMPARE:
        return None
    value = runs[1].data.metrics.get(metric)
    return float(value) if value is not None and math.isfinite(value) else None
