"""Write eval runs + per-row verdicts to the RESULTS store.

``eval_run`` / ``eval_judgement`` DDL lives in ``news_nlp.schema``; ``init_schema``
creates them. Column lists route through ``conn.dialect`` like the rest of
``news_nlp.queries``; the caller commits.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from news_nlp.db import NewsNlpDatabase

_RUN_COLS = (
    "stage",
    "started_at",
    "sample_size",
    "low_conf_n",
    "random_n",
    "seed",
    "judge_model",
    "judge_url",
    "code_version",
    "status",
)
_JUDGEMENT_COLS = (
    "run_id",
    "article_id",
    "bucket",
    "model_prediction_json",
    "verdict_json",
    "correct",
    "severity",
    "rationale",
    "judged_at",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_eval_run(
    conn: NewsNlpDatabase,
    *,
    stage: str,
    sample_size: int,
    low_conf_n: int,
    random_n: int,
    seed: int | None,
    judge_model: str,
    judge_url: str,
    code_version: str,
) -> int:
    """Insert a ``running`` ``eval_run`` row; return its id."""
    cur = conn.execute(
        conn.dialect.insert("eval_run", _RUN_COLS),
        (
            stage,
            _now(),
            sample_size,
            low_conf_n,
            random_n,
            seed,
            judge_model,
            judge_url,
            code_version,
            "running",
        ),
    )
    row_id = cur.lastrowid
    if row_id is None:  # pragma: no cover -- INSERT always sets it for an AUTOINCREMENT PK
        raise RuntimeError("eval_run INSERT returned no rowid")
    return int(row_id)


def record_judgement(
    conn: NewsNlpDatabase,
    run_id: int,
    *,
    article_id: int,
    bucket: str,
    prediction: dict[str, Any],
    verdict: dict[str, Any],
    correct: bool | None,
    severity: int | None,
    rationale: str,
) -> None:
    conn.execute(
        conn.dialect.insert("eval_judgement", _JUDGEMENT_COLS),
        (
            run_id,
            article_id,
            bucket,
            json.dumps(prediction, default=str),
            json.dumps(verdict, default=str),
            None if correct is None else int(correct),
            severity,
            rationale,
            _now(),
        ),
    )


def finish_eval_run(
    conn: NewsNlpDatabase,
    run_id: int,
    *,
    metrics: dict[str, float],
    mlflow_run_id: str | None,
    status: str,
    error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE eval_run SET finished_at = ?, metrics_json = ?, mlflow_run_id = ?, "
        "status = ?, error = ? WHERE id = ?",
        (_now(), json.dumps(metrics), mlflow_run_id, status, error, run_id),
    )
