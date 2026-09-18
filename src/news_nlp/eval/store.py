"""Write (and, for judge-verdict reuse, read) eval runs + per-row model
inferences/judge verdicts in the RESULTS store.

``eval_run``/``eval_inference``/``eval_verdict``/``eval_confusion`` DDL
lives in ``news_nlp.schema``; ``init_schema`` creates them. Column lists
route through ``conn.dialect`` like the rest of ``news_nlp.queries``; the
caller commits.

``eval_inference``/``eval_verdict`` (TASKS.md T-090, SPEC.md FR-014) split
what the now-legacy ``eval_judgement`` held in one row -- the sampled model
inference and the LLM judge's verdict on it -- into two tables, both
carrying ``article_id``/``task``/``experiment`` so more than one
experiment's data for the same article/task can coexist. ``record_judgement``
is gone; ``record_inference``/``record_verdict`` replace it at the same call
site (``news_nlp.eval.runner._run_stage``).

``find_verdict_json`` (TASKS.md T-091, SPEC.md FR-015) is the one
verdict-reuse read function here -- looked up before judging, so an
unchanged ``(article_id, task, experiment)`` key skips the judge LLM call
and reuses the stored verdict instead of re-judging.

``record_confusion_cells`` (TASKS.md T-092, SPEC.md FR-016) writes one
``eval_confusion`` row per distinct ``(true_label, predicted_label)`` cell
observed in a sentiment/category run's own sample -- built from the SAME
judge verdicts ``record_verdict`` already persists, no new judge calls.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
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
    "strata_json",
)
_INFERENCE_COLS = (
    "run_id",
    "article_id",
    "task",
    "experiment",
    "bucket",
    "prediction_json",
    "created_at",
)
_VERDICT_COLS = (
    "inference_id",
    "run_id",
    "article_id",
    "task",
    "experiment",
    "verdict_json",
    "correct",
    "severity",
    "rationale",
    "judged_at",
)
_CONFUSION_COLS = (
    "run_id",
    "task",
    "experiment",
    "true_label",
    "predicted_label",
    "count",
    "created_at",
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
    strata_json: str = "{}",
) -> int:
    """Insert a ``running`` ``eval_run`` row; return its id.

    ``strata_json`` is the per-stratum ``{bucket: {"population": N_h, "n":
    n_h}}`` bookkeeping the Horvitz-Thompson reweighting in
    ``news_nlp.eval.metrics`` needs (see ``docs/evaluation.md``); defaults to
    ``'{}'`` for callers that don't (yet) have it.
    """
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
            strata_json,
        ),
    )
    row_id = cur.lastrowid
    if row_id is None:  # pragma: no cover -- INSERT always sets it for an AUTOINCREMENT PK
        raise RuntimeError("eval_run INSERT returned no rowid")
    return int(row_id)


def record_inference(
    conn: NewsNlpDatabase,
    run_id: int,
    *,
    article_id: int,
    task: str,
    experiment: str,
    bucket: str,
    prediction: dict[str, Any],
) -> int:
    """Insert one ``eval_inference`` row; return its id (for
    ``record_verdict``'s ``inference_id`` FK)."""
    cur = conn.execute(
        conn.dialect.insert("eval_inference", _INFERENCE_COLS),
        (
            run_id,
            article_id,
            task,
            experiment,
            bucket,
            json.dumps(prediction, default=str),
            _now(),
        ),
    )
    row_id = cur.lastrowid
    if row_id is None:  # pragma: no cover -- INSERT always sets it for an AUTOINCREMENT PK
        raise RuntimeError("eval_inference INSERT returned no rowid")
    return int(row_id)


def record_verdict(
    conn: NewsNlpDatabase,
    inference_id: int,
    run_id: int,
    *,
    article_id: int,
    task: str,
    experiment: str,
    verdict: dict[str, Any],
    correct: bool | None,
    severity: int | None,
    rationale: str,
) -> None:
    conn.execute(
        conn.dialect.insert("eval_verdict", _VERDICT_COLS),
        (
            inference_id,
            run_id,
            article_id,
            task,
            experiment,
            json.dumps(verdict, default=str),
            None if correct is None else int(correct),
            severity,
            rationale,
            _now(),
        ),
    )


def find_verdict_json(
    conn: NewsNlpDatabase, *, article_id: int, task: str, experiment: str
) -> str | None:
    """The most recent ``eval_verdict.verdict_json`` already recorded for
    this exact ``(article_id, task, experiment)`` key, or ``None`` -- an
    indexed lookup via ``idx_eval_verdict_article_task_experiment``, not a
    scan (TASKS.md T-091, SPEC.md FR-015). ``ORDER BY judged_at DESC LIMIT
    1`` is a safety net, not a correctness requirement -- nothing should
    ever produce more than one row per key once this reuse mechanism is in
    place, but the query stays well-defined even if it does."""
    row = conn.execute(
        "SELECT verdict_json FROM eval_verdict WHERE article_id = ? AND task = ? "
        "AND experiment = ? ORDER BY judged_at DESC LIMIT 1",
        (article_id, task, experiment),
    ).fetchone()
    return row[0] if row is not None else None


def record_confusion_cells(
    conn: NewsNlpDatabase,
    run_id: int,
    *,
    task: str,
    experiment: str,
    counts: Mapping[tuple[str, str], int],
) -> None:
    """Insert one ``eval_confusion`` row per distinct ``(true_label,
    predicted_label)`` cell in *counts* (typically
    ``collections.Counter(metrics.confusion_pairs(...))``) -- a no-op if
    *counts* is empty (TASKS.md T-092, SPEC.md FR-016)."""
    for (true_label, predicted_label), count in counts.items():
        conn.execute(
            conn.dialect.insert("eval_confusion", _CONFUSION_COLS),
            (run_id, task, experiment, true_label, predicted_label, count, _now()),
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
