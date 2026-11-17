"""The service of the stepped pipeline (T-219): run each step, run the whole pipeline, serve the stage models.

Run:
    uv run apps/pipeline_api.py            # -> http://127.0.0.1:8003/docs
    uvicorn apps.pipeline_api:app --host 0.0.0.0 --port 8003     # the Docker image

Every run is a job (one at a time, the GPU is single), in memory; a step runs through the CLI's own parser and functions (``news_nlp.service``), so
the rows are the CLI's. Revisions are explicit, never "latest". No authentication (owner, 2026-11-17): keep the port on a private network. Design and
what it does not do: ``docs/pipeline_api.md``. It does not replace the v1 API (``apps/news_nlp_api.py``) yet: T-220 removes that one.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import APIRouter, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import news_nlp as db
from news_nlp.service import steps_runner as runner
from news_nlp.service.jobs import Busy, GpuGate, JobManager
from news_nlp.service.stage_models import StageModels, StageUnavailable

db.env.load_env_file()  # the shell wins over the file; read once, before any connection (tests/news_nlp/test_wi18_scripts.py)


class StepRequest(BaseModel):
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="the options of cli/news_nlp_steps.py without the leading dashes (summary_revision, limit, ...); revisions are explicit",
    )


class PipelineStepRequest(StepRequest):
    step: str


class PipelineRequest(BaseModel):
    steps: list[PipelineStepRequest] = Field(
        min_length=1, description="the steps to run, in the order of the pipeline"
    )


class PredictRequest(BaseModel):
    items: list[dict[str, Any]] = Field(min_length=1)
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="what builds the model, not an item (companies: threshold and candidates, both required)",
    )


BUSY_TIMEOUT_MS = 3000


def default_connection() -> Any:
    """The stores of the environment, to read only: no ``init_schema`` (DDL and migrations contend with a job that writes) and ``query_only``."""
    conn = db.connect_pipeline(results_db=None, source_db=None)
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA query_only = ON")
    return conn


def step_status(
    conn: Any, step: str, revision: str | None, snapshot_revision: str | None
) -> dict[str, Any]:
    """The counts of a step, read-only (no model, no print): what ``--status`` shows without its details."""
    from news_nlp import steps  # noqa: PLC0415

    if step == "snapshot":
        import news_snapshot_stage as snap  # noqa: PLC0415

        return dict(snap.status(conn))
    if step == "period":
        import news_period_stage as period  # noqa: PLC0415

        return dict(period.status(conn, snapshot_revision, tz=period.batch_timezone()))
    shown = revision or steps.latest_revision(conn, step)
    report = dict(steps.status(conn, step, shown))
    report["revision"] = shown
    report["revisions"] = steps.revisions(conn, step)
    return report


def _steps_router(jobs: JobManager, connection: Callable[[], Any]) -> APIRouter:
    router = APIRouter()

    @router.get("/steps")
    def list_steps() -> dict[str, Any]:
        return {
            "order": list(runner.PIPELINE_ORDER),
            "not_through_the_api": sorted(runner.NOT_THROUGH_THE_API),
            "note": "options are those of cli/news_nlp_steps.py without the dashes; run POST /steps/{step}/run",
        }

    @router.get("/steps/{step}/status")
    def status(
        step: str,
        revision: str | None = Query(
            None, description="the model revision to count (default: the latest with state)"
        ),
        snapshot_revision: str | None = Query(
            None, description="period only: the snapshot revision"
        ),
    ) -> dict[str, Any]:
        if step not in runner.PIPELINE_ORDER:
            raise HTTPException(404, f"unknown step {step!r}")
        try:
            conn = connection()
            try:
                return step_status(conn, step, revision, snapshot_revision)
            finally:
                conn.close()
        except (
            sqlite3.OperationalError,
            RuntimeError,
        ) as exc:  # a locked store, no tables yet, no SOURCE configured
            raise HTTPException(503, f"the status could not be read: {exc}") from exc

    @router.post("/steps/{step}/run", status_code=202)
    def run_step(step: str, request: StepRequest) -> dict[str, Any]:
        if step not in runner.PIPELINE_ORDER:
            raise HTTPException(404, f"unknown step {step!r}")
        return _submit(jobs, "step", [(step, request.options)])

    @router.post("/pipeline/run", status_code=202)
    def run_pipeline(request: PipelineRequest) -> dict[str, Any]:
        names = [s.step for s in request.steps]
        if names != [n for n in runner.PIPELINE_ORDER if n in names]:
            raise HTTPException(
                422,
                f"steps must be unique and in the order of the pipeline: {', '.join(runner.PIPELINE_ORDER)}",
            )
        return _submit(jobs, "pipeline", [(s.step, s.options) for s in request.steps])

    return router


def _submit(jobs: JobManager, kind: str, items: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    try:
        return jobs.submit(kind, jobs.plan(items)).public()
    except runner.StepRefused as exc:
        raise HTTPException(422, str(exc)) from exc
    except Busy as exc:
        raise HTTPException(409, str(exc)) from exc


def _jobs_router(jobs: JobManager) -> APIRouter:
    router = APIRouter()

    @router.get("/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [j.public() for j in jobs.listing()]

    @router.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(
                404, f"no job {job_id!r} (jobs are kept in memory: a restart forgets them)"
            )
        return job.public()

    @router.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        job, reason = jobs.cancel(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id!r}")
        if reason is not None:
            raise HTTPException(409, reason)
        return job.public()

    return router


def _fti_router(stage_models: StageModels, gate: GpuGate) -> APIRouter:
    router = APIRouter()

    @router.get("/fti/stages")
    def fti_stages() -> dict[str, Any]:
        return {"stages": stage_models.stages(), "gpu_holder": gate.holder}

    @router.post("/fti/{stage}/predict")
    def fti_predict(stage: str, request: PredictRequest) -> dict[str, Any]:
        try:
            return {
                "stage": stage,
                "results": stage_models.predict(stage, request.items, request.options),
            }
        except KeyError as exc:
            raise HTTPException(404, f"unknown stage {stage!r}") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        except (StageUnavailable, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc

    return router


def create_app(
    manager: JobManager | None = None,
    models: StageModels | None = None,
    connection: Callable[[], Any] = default_connection,
) -> FastAPI:
    app = FastAPI(title="news-nlp pipeline", version="1")
    gate = manager.gate if manager is not None else GpuGate()
    jobs = manager or JobManager(gate)
    stage_models = models or StageModels(gate, connection)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(_steps_router(jobs, connection))
    app.include_router(_jobs_router(jobs))
    app.include_router(_fti_router(stage_models, gate))
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8003)
