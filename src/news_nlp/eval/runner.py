"""Orchestrate an eval run: sample -> judge (concurrently) -> aggregate -> persist.

``run_eval`` opens one two-tier ``connect_pipeline`` connection, then per stage
draws the stratified ``low_conf`` / ``target_<x>`` / ``representative`` sample
(``news_nlp.eval.sampling``), judges every row through a ``ThreadPoolExecutor``
(each task builds its own stateless ``Agent`` over a shared ``OpenAIModel``),
aggregates, writes ``eval_run``/``eval_inference``/``eval_verdict`` rows + an
MLflow run, and optionally checks for a headline-metric regression against
the previous run.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

import news_nlp as db
from news_nlp.eval.candidate import candidate_scored_connection
from news_nlp.eval.config import EvalSettings
from news_nlp.eval.judges import JUDGES, load_prompt
from news_nlp.eval.metrics import HEADLINE, aggregate
from news_nlp.eval.model import build_judge_agent, build_model
from news_nlp.eval.provenance import code_version
from news_nlp.eval.regression import check_regression as _check_regression
from news_nlp.eval.sampling import STAGES, EvalItem, sample_for_stage
from news_nlp.eval.store import (
    create_eval_run,
    finish_eval_run,
    record_inference,
    record_verdict,
)
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
    source_db: str,
    want_regression: bool,
    tolerance: float,
) -> dict[str, Any]:
    prompt = load_prompt(stage)
    if settings.candidate_model:
        # Score a fresh sample with the candidate model, via that stage's
        # own FTI Inference subclass, into a throwaway scratch RESULTS file
        # -- never the production tables -- then draw the same low_conf/
        # target_<x>/representative sample from THAT connection instead, so
        # stratification reflects the candidate's own scores (TASKS.md
        # T-089, SPEC.md FR-013). The scratch connection only needs to live
        # long enough to materialize `items` below.
        assert settings.candidate_revision, "run_eval validates this before calling _run_stage"
        with candidate_scored_connection(
            source_db,
            stage,
            model_name=settings.candidate_model,
            revision=settings.candidate_revision,
            limit=settings.candidate_prescore_size or settings.sample_size,
            sample_seed=settings.seed,
        ) as sample_conn:
            items = sample_for_stage(
                sample_conn,
                stage,
                size=settings.sample_size,
                low_conf_frac=settings.low_conf_frac,
                target_frac=settings.target_frac,
                seed=settings.seed,
            )
    else:
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

    # TASKS.md T-090 / SPEC.md FR-014: which experiment produced this run's
    # inferences. candidate_model (T-089's own field, the most specific
    # "which model produced this" signal) wins when a candidate run is in
    # progress; run_name is the fallback for someone labeling a
    # production-model run without swapping models; "base" is the default
    # for an unlabeled production run, matching FR-014's own example values.
    experiment = settings.candidate_model or settings.run_name or "base"

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
            inference_id = record_inference(
                conn,
                run_id,
                article_id=item.article_id,
                task=stage,
                experiment=experiment,
                bucket=item.bucket,
                prediction=item.prediction,
            )
            record_verdict(
                conn,
                inference_id,
                run_id,
                article_id=item.article_id,
                task=stage,
                experiment=experiment,
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
            run_name=settings.run_name,
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
    if settings.candidate_model:
        if len(chosen) != 1:
            # A candidate model swap is inherently stage-specific (a
            # different model architecture per stage) -- silently applying
            # it to "all" or several stages at once would be a user error,
            # not a real multi-stage experiment. Mirrors
            # scripts/resample_sentiment_v*'s own one-stage-at-a-time
            # precedent.
            raise ValueError(
                "candidate_model requires exactly one stage, got "
                f"{chosen!r} -- pass stages=['<one stage>']"
            )
        if not settings.candidate_revision:
            # candidate_scored_connection requires a real revision (see its
            # own docstring) -- fail fast here with a clear message rather
            # than a deep-stack ValueError from inside _run_stage.
            raise ValueError(
                "candidate_revision is required when candidate_model is set "
                "(pass any placeholder for a local checkpoint path)"
            )

    conn = db.connect_pipeline(results_db=results_db, source_db=source_db)
    # connect_pipeline above already raised if SOURCE isn't configured, so
    # this resolves to a real path -- same resolution it did internally.
    resolved_source_db = str(db.source_db_path(source_db))
    db.init_schema(conn)  # idempotent; ensures eval_run/eval_inference/eval_verdict exist
    results: dict[str, dict[str, Any]] = {}
    try:
        for stage in chosen:
            results[stage] = _run_stage(
                conn,
                stage,
                settings,
                source_db=resolved_source_db,
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
