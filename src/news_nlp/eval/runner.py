"""Orchestrate an eval run: sample -> judge (concurrently) -> aggregate -> persist.

``run_eval`` opens one two-tier ``connect_pipeline`` connection, then per stage
draws the stratified ``low_conf`` / ``target_<x>`` / ``representative`` sample
(``news_nlp.eval.sampling``), judges every row through a ``ThreadPoolExecutor``
(each task builds its own stateless ``Agent`` over a shared ``OpenAIModel``),
aggregates, writes ``eval_run`` / ``eval_judgement`` rows + an MLflow run, and
optionally checks for a headline-metric regression against the previous run.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

import news_nlp as db
from news_nlp.eval.config import EvalSettings
from news_nlp.eval.judges import JUDGES, load_prompt
from news_nlp.eval.metrics import HEADLINE, aggregate
from news_nlp.eval.model import build_judge_agent, build_model
from news_nlp.eval.provenance import code_version
from news_nlp.eval.regression import check_regression as _check_regression
from news_nlp.eval.sampling import STAGES, EvalItem, sample_for_stage
from news_nlp.eval.store import create_eval_run, finish_eval_run, record_judgement
from news_nlp.eval.tracking import log_to_mlflow

_LABEL_STAGES = {"sentiment", "category"}


def _correct_and_severity(stage: str, dumped: dict[str, Any]) -> tuple[bool | None, int | None]:
    if dumped.get("parse_failed") or stage not in _LABEL_STAGES:
        return None, None
    return bool(dumped["agrees"]), int(dumped["severity"])


def _run_stage(
    conn: db.NewsNlpDatabase,
    stage: str,
    settings: EvalSettings,
    *,
    want_regression: bool,
    tolerance: float,
) -> dict[str, Any]:
    prompt = load_prompt(stage)
    items = sample_for_stage(
        conn,
        stage,
        size=settings.sample_size,
        low_conf_frac=settings.low_conf_frac,
        target_frac=settings.target_frac,
        seed=settings.seed,
    )
    bucket_counts = Counter(it.bucket for it in items)
    low_conf_n = bucket_counts.get("low_conf", 0)
    random_n = len(items) - low_conf_n  # sum of every non-low_conf stratum
    population_by_bucket: dict[str, int] = {}
    for it in items:
        population_by_bucket.setdefault(it.bucket, it.stratum_population)
    strata_meta = {
        bucket: {"population": population_by_bucket[bucket], "n": n}
        for bucket, n in bucket_counts.items()
    }

    run_id = create_eval_run(
        conn,
        stage=stage,
        sample_size=len(items),
        low_conf_n=low_conf_n,
        random_n=random_n,
        seed=settings.seed,
        judge_model=settings.llm_model,
        judge_url=settings.llm_url,
        code_version=code_version(),
        strata_json=json.dumps(strata_meta),
    )
    conn.commit()

    try:
        model = build_model(settings)
        judge = JUDGES[stage]

        def judge_one(item: EvalItem) -> BaseModel:
            return judge(build_judge_agent(model, prompt), item)

        verdicts: list[BaseModel] = []
        with ThreadPoolExecutor(max_workers=settings.max_workers) as pool:
            for verdict in tqdm(
                pool.map(judge_one, items), total=len(items), desc=f"judge {stage}"
            ):
                verdicts.append(verdict)

        metrics = aggregate(stage, items, verdicts)

        judgements: list[dict[str, Any]] = []
        for item, verdict in zip(items, verdicts, strict=True):
            dumped = verdict.model_dump()
            correct, severity = _correct_and_severity(stage, dumped)
            record_judgement(
                conn,
                run_id,
                article_id=item.article_id,
                bucket=item.bucket,
                prediction=item.prediction,
                verdict=dumped,
                correct=correct,
                severity=severity,
                rationale=str(dumped.get("rationale", "")),
            )
            judgements.append(
                {
                    "article_id": item.article_id,
                    "bucket": item.bucket,
                    "prediction": item.prediction,
                    "verdict": dumped,
                }
            )

        mlflow_run_id = log_to_mlflow(
            stage=stage,
            params={
                "stage": stage,
                "sample_size": settings.sample_size,
                "n_judged": len(items),
                "low_conf_n": low_conf_n,
                "random_n": random_n,
                **{f"n_{bucket}": n for bucket, n in bucket_counts.items()},
                "seed": settings.seed,
                "judge_model": settings.llm_model,
                "judge_url": settings.llm_url,
                "code_version": code_version(),
            },
            metrics=metrics,
            judgements=judgements,
            system_prompt=prompt,
            tracking_uri=settings.mlflow_tracking_uri,
        )
        finish_eval_run(conn, run_id, metrics=metrics, mlflow_run_id=mlflow_run_id, status="ok")
        conn.commit()
    except Exception as exc:
        finish_eval_run(
            conn, run_id, metrics={}, mlflow_run_id=None, status="error", error=str(exc)
        )
        conn.commit()
        raise

    result: dict[str, Any] = {
        "eval_run_id": run_id,
        "mlflow_run_id": mlflow_run_id,
        "n_judged": len(items),
        "headline_metric": HEADLINE[stage],
        "headline_value": metrics.get(HEADLINE[stage]),
        "metrics": metrics,
        "regressed": False,
    }
    if want_regression:
        rr = _check_regression(
            stage, metrics, tolerance=tolerance, tracking_uri=settings.mlflow_tracking_uri
        )
        result["regressed"] = rr.regressed
        result["regression"] = rr.describe()
    return result


def run_eval(
    stages: list[str] | None = None,
    *,
    settings: EvalSettings | None = None,
    source_db: str | None = None,
    results_db: str | None = None,
    check_regression: bool = False,
    regression_tolerance: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Run the LLM-as-judge evaluation for *stages* (default: all four).

    Returns ``{stage: {eval_run_id, mlflow_run_id, n_judged, headline_metric,
    headline_value, metrics, regressed, regression?}}``. Raises ``SystemExit(1)``
    if ``check_regression`` and any stage's headline metric dropped past
    ``regression_tolerance``.
    """
    settings = settings or EvalSettings.load()
    chosen = list(stages) if stages else list(STAGES)
    bad = [s for s in chosen if s not in STAGES]
    if bad:
        raise ValueError(f"unknown stage(s) {bad}; valid: {list(STAGES)}")

    conn = db.connect_pipeline(results_db=results_db, source_db=source_db)
    db.init_schema(conn)  # idempotent; ensures eval_run / eval_judgement exist
    results: dict[str, dict[str, Any]] = {}
    try:
        for stage in chosen:
            results[stage] = _run_stage(
                conn,
                stage,
                settings,
                want_regression=check_regression,
                tolerance=regression_tolerance,
            )
    finally:
        db.detach_source(conn)
        conn.close()

    if check_regression and any(r["regressed"] for r in results.values()):
        for r in results.values():
            if "regression" in r:
                print(r["regression"])
        raise SystemExit(1)
    return results


def summary_table(results: dict[str, dict[str, Any]]) -> str:
    """A compact per-stage summary for the CLI."""
    lines = [f"{'stage':<12} {'n':>4}  {'headline':<24} {'value':>8}  eval_run / mlflow"]
    for stage, r in results.items():
        val = r.get("headline_value")
        val_s = f"{val:.4f}" if isinstance(val, int | float) else "n/a"
        mlflow_id = str(r.get("mlflow_run_id") or "")[:8]
        lines.append(
            f"{stage:<12} {r['n_judged']:>4}  {r['headline_metric']:<24} {val_s:>8}  "
            f"{r['eval_run_id']} / {mlflow_id}"
        )
        if r.get("regression"):
            lines.append(f"  {r['regression']}")
    return "\n".join(lines)
