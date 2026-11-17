"""One background run at a time, jobs kept in memory (T-219, owner 2026-11-17).

The service has one GPU and a step runs for hours, so there is one worker: a second request while a job (or a stage-model call) holds the GPU is
refused with ``Busy`` (HTTP 409), not queued. Jobs live in memory and are lost with the process, which is safe: every step is idempotent and its state is in the
tables, so a restart resumes where the run stopped. A running step is not stopped in the middle (the CLI has no hook for it): ``cancel`` stops a pipeline
job **between steps**; to stop a step, restart the service.
"""

from __future__ import annotations

import argparse
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from news_nlp.service import steps_runner as runner

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED, SKIPPED = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "skipped",
)
ACTIVE = (QUEUED, RUNNING)
MAX_JOBS_KEPT = 50


class Busy(RuntimeError):
    """The GPU is in use by a job or by a stage-model call."""


class GpuGate:
    """One holder at a time. A job holds it from its start to its end; a stage-model call holds it for the call."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.holder: str | None = None

    def acquire(self, owner: str) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        self.holder = owner
        return True

    def release(self) -> None:
        self.holder = None
        self._lock.release()


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class StepRun:
    step: str
    options: dict[str, Any]
    args: argparse.Namespace = field(repr=False, default_factory=argparse.Namespace)
    status: str = QUEUED
    started_at: str | None = None
    finished_at: str | None = None
    outputs: list[Any] = field(default_factory=list)
    log: str = ""
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "options": self.options,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outputs": self.outputs,
            "log": self.log,
            "error": self.error,
        }


@dataclass
class Job:
    id: str
    kind: str  # "step" or "pipeline"
    steps: list[StepRun]
    status: str = QUEUED
    created_at: str = field(default_factory=now)
    started_at: str | None = None
    finished_at: str | None = None
    progress: str = ""
    cancel_requested: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "cancel_requested": self.cancel_requested,
            "steps": [s.public() for s in self.steps],
        }


Executor = Callable[
    [argparse.Namespace, Callable[[str], None] | None], runner.StepOutcome
]  # (args, on_progress) -> outcome


class JobManager:
    def __init__(
        self, gate: GpuGate | None = None, executor: Executor | None = None, spawn: bool = True
    ) -> None:
        self.gate = gate or GpuGate()
        self._executor: Executor = executor or runner.execute
        self._spawn = spawn  # tests run the job in the calling thread
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []

    # -- requests ------------------------------------------------------------------------------------------------------

    def plan(self, items: Sequence[tuple[str, Mapping[str, Any]]]) -> list[StepRun]:
        """Validate every step up front with the CLI's parser: a refusal is raised before anything starts (``StepRefused``)."""
        steps = []
        for step, options in items:
            if options.get("status"):
                raise runner.StepRefused(
                    "'status' is not a run option: read GET /steps/{step}/status"
                )
            steps.append(StepRun(step, dict(options), runner.parse_step(step, options)))
        return steps

    def submit(self, kind: str, steps: list[StepRun]) -> Job:
        if not self.gate.acquire("job"):
            raise Busy(f"the GPU is in use by {self.gate.holder}")
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, steps=steps)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            for old in self._order[:-MAX_JOBS_KEPT]:
                if self._jobs[old].status not in ACTIVE:
                    del self._jobs[old]
            self._order = [i for i in self._order if i in self._jobs]
        if self._spawn:
            try:
                threading.Thread(
                    target=self._run, args=(job,), name=f"job-{job.id}", daemon=True
                ).start()
            except RuntimeError:  # the thread did not start: the GPU must not stay held
                job.status, job.finished_at = FAILED, now()
                self.gate.release()
                raise
        else:
            self._run(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def listing(self) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in reversed(self._order) if i in self._jobs]

    def cancel(self, job_id: str) -> tuple[Job | None, str | None]:
        """Ask a job to stop before its next step. Returns the job and, when no stop is possible, the reason (nothing is marked)."""
        job = self.get(job_id)
        if job is None:
            return None, None
        if job.status not in ACTIVE:
            return job, f"the job already ended ({job.status})"
        if not any(s.status == QUEUED for s in job.steps):
            return job, (
                "a running step cannot be stopped in the middle and no step is left to skip: "
                "restart the service to stop it (every step resumes where it stopped)"
            )
        job.cancel_requested = True
        return job, None

    # -- the worker ----------------------------------------------------------------------------------------------------

    def _run(self, job: Job) -> None:
        try:
            job.status, job.started_at = RUNNING, now()
            failed = False
            for index, step in enumerate(job.steps):
                if failed or job.cancel_requested:
                    for rest in job.steps[index:]:
                        rest.status = SKIPPED
                    break
                step.status, step.started_at = RUNNING, now()
                outcome = self._executor(step.args, lambda line: setattr(job, "progress", line))
                step.outputs, step.log = outcome.outputs, outcome.log
                step.finished_at = now()
                if outcome.ok:
                    step.status = SUCCEEDED
                else:
                    step.status, step.error, failed = FAILED, outcome.message, True
            if failed:
                job.status = FAILED
            elif job.cancel_requested and any(s.status == SKIPPED for s in job.steps):
                job.status = CANCELLED
            else:
                job.status = SUCCEEDED
            job.finished_at = now()
        finally:
            self.gate.release()
