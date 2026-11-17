"""The service of the stepped pipeline (T-219): the CLI's own parser and functions behind jobs, the stage models, the routes. No model, no network."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from fastapi.testclient import TestClient

import apps.pipeline_api as api
import news_nlp as db_module
from news_nlp.service import jobs as jobs_module
from news_nlp.service import steps_runner as runner
from news_nlp.service.jobs import Busy, GpuGate, JobManager
from news_nlp.service.stage_models import BUILDERS, STAGES, StageModels, StageUnavailable, jsonable

REVISIONS = {"summary_revision": "summary-a", "companies_revision": "companies-b"}


# --- the command line the API builds and the CLI's own refusals -------------------------------------------------------


def test_options_become_the_cli_command_line() -> None:
    argv = runner.build_argv("category", {**REVISIONS, "limit": 5, "status": False, "x": None})
    assert argv == [
        "--step",
        "category",
        "--summary-revision",
        "summary-a",
        "--companies-revision",
        "companies-b",
        "--limit",
        "5",
    ]  # False and None are left out
    assert runner.build_argv("period", {"catch_up": True})[-1] == "--catch-up"
    assert runner.build_argv("period", {"catch-up": True})[-1] == "--catch-up"


@pytest.mark.parametrize("option", sorted(runner.NOT_THROUGH_THE_API))
def test_an_option_that_acts_on_files_or_rows_is_not_available_through_the_api(option: str) -> None:
    with pytest.raises(runner.StepRefused, match="not available through the API"):
        runner.build_argv("scope", {option: True})


def test_an_unknown_step_and_the_step_as_an_option_are_refused() -> None:
    with pytest.raises(runner.StepRefused, match="unknown step"):
        runner.build_argv("nope", {})
    with pytest.raises(runner.StepRefused, match="not an option"):
        runner.build_argv("scope", {"step": "cut"})


def test_the_cli_parser_and_its_checks_refuse_what_the_command_line_refuses() -> None:
    args = runner.parse_step("category", {**REVISIONS, "limit": 5})
    assert args.step == "category" and args.limit == 5
    with pytest.raises(runner.StepRefused, match="--limit must be at least 1"):
        runner.parse_step("scope", {"limit": 0})
    with pytest.raises(runner.StepRefused):  # an option of another step is refused, not ignored
        runner.parse_step("scope", {"reading": "units5"})
    with pytest.raises(runner.StepRefused):
        runner.parse_step("scope", {"no_such_option": "x"})


def test_every_step_of_the_pipeline_is_a_step_of_the_cli() -> None:
    assert tuple(runner.cli().STEPS) == runner.PIPELINE_ORDER


# --- what a step printed ----------------------------------------------------------------------------------------------


def test_the_documents_of_stdout_are_split_from_the_other_lines() -> None:
    text = 'sector step, revision x\n{"a": 1}\n{"b": [1, 2]}\nrefused later\n'
    documents, other = runner.split_documents(text)
    assert documents == [{"a": 1}, {"b": [1, 2]}]
    assert other == "sector step, revision x\nrefused later"


def test_the_tail_keeps_the_last_progress_line_of_a_rewritten_bar() -> None:
    seen: list[str] = []
    tail = runner.Tail(seen.append)
    tail.write("10%|#  | 1/10\r20%|## | 2/10\r30%|###| 3/10")
    assert seen[-1] == "30%|###| 3/10"


def _args() -> argparse.Namespace:
    return argparse.Namespace(step="scope")


def test_a_step_that_prints_a_result_ends_ok_with_its_documents_and_progress() -> None:
    progress: list[str] = []

    def fake(_args: argparse.Namespace) -> None:
        print(json.dumps({"done": 3}))
        print("bar 1/3", file=sys.stderr)

    outcome = runner.execute(_args(), progress.append, runner=fake)
    assert outcome.ok and outcome.outputs == [{"done": 3}] and progress == ["bar 1/3"]


def test_a_sys_exit_message_is_a_refusal_and_an_exception_is_a_failure_not_a_crash() -> None:
    def refuse(_args: argparse.Namespace) -> None:
        sys.exit("refused: no companies")

    def crash(_args: argparse.Namespace) -> None:
        raise RuntimeError("the GPU is lost")

    first = runner.execute(_args(), runner=refuse)
    assert not first.ok and first.message == "refused: no companies"
    second = runner.execute(_args(), runner=crash)
    assert not second.ok and second.message == "RuntimeError: the GPU is lost"


def test_a_clean_sys_exit_is_ok() -> None:
    def clean(_args: argparse.Namespace) -> None:
        sys.exit(0)

    assert runner.execute(_args(), runner=clean).ok


def test_the_real_cli_path_connects_dispatches_and_reports_the_clis_own_refusal(
    results_db_path: Path, source_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.setenv("SOURCE_DATABASE_URL", str(source_db_path))
    args = runner.parse_step("period", {"snapshot_revision": "snapshot-x", "catch_up": True})
    outcome = runner.execute(args)  # the CLI's main_period, on a real empty store
    assert not outcome.ok
    assert "no company snapshots at snapshot-x" in outcome.message


# --- the jobs ---------------------------------------------------------------------------------------------------------


def _manager(
    results: Mapping[str, runner.StepOutcome] | None = None,
    on_run: Callable[[str, JobManager], None] | None = None,
) -> tuple[JobManager, list[str]]:
    ran: list[str] = []
    holder: list[JobManager] = []

    def executor(
        args: argparse.Namespace, progress: Callable[[str], None] | None
    ) -> runner.StepOutcome:
        ran.append(args.step)
        if progress:
            progress(f"running {args.step}")
        if on_run:
            on_run(args.step, holder[0])
        return (results or {}).get(
            args.step, runner.StepOutcome(ok=True, outputs=[{"step": args.step}])
        )

    manager = JobManager(executor=executor, spawn=False)
    holder.append(manager)
    return manager, ran


def test_a_job_runs_its_steps_in_order_and_keeps_the_outputs_and_the_progress() -> None:
    manager, ran = _manager()
    job = manager.submit("pipeline", manager.plan([("scope", {}), ("cut", {})]))
    assert ran == ["scope", "cut"] and job.status == jobs_module.SUCCEEDED
    assert [s.outputs for s in job.steps] == [[{"step": "scope"}], [{"step": "cut"}]]
    assert job.progress == "running cut" and job.finished_at
    assert manager.get(job.id) is job and manager.listing() == [job]


def test_a_failed_step_stops_the_pipeline_and_the_rest_are_skipped() -> None:
    manager, ran = _manager({"cut": runner.StepOutcome(ok=False, message="refused: x")})
    job = manager.submit("pipeline", manager.plan([("scope", {}), ("cut", {}), ("summary", {})]))
    assert ran == ["scope", "cut"]
    assert [s.status for s in job.steps] == ["succeeded", "failed", "skipped"]
    assert job.status == jobs_module.FAILED and job.steps[1].error == "refused: x"


def test_a_cancel_stops_a_pipeline_between_steps() -> None:
    def cancel_on_scope(step: str, manager: JobManager) -> None:
        if step == "scope":
            manager.cancel(manager.listing()[0].id)

    manager, ran = _manager(on_run=cancel_on_scope)
    job = manager.submit("pipeline", manager.plan([("scope", {}), ("cut", {})]))
    assert ran == ["scope"] and job.status == jobs_module.CANCELLED
    assert [s.status for s in job.steps] == ["succeeded", "skipped"]


def test_the_gpu_is_held_by_a_job_and_a_second_request_is_refused_not_queued() -> None:
    gate = GpuGate()
    manager = JobManager(gate=gate, executor=lambda *_a: runner.StepOutcome(ok=True), spawn=False)
    steps = manager.plan([("scope", {})])
    assert gate.acquire("a stage model")
    with pytest.raises(Busy, match="a stage model"):
        manager.submit("step", steps)
    gate.release()
    manager.submit("step", steps)
    assert gate.holder is None  # the job released it


def test_a_run_with_the_status_flag_is_refused_and_a_bad_option_stops_before_the_start() -> None:
    manager, ran = _manager()
    with pytest.raises(runner.StepRefused, match="status"):
        manager.plan([("scope", {"status": True})])
    with pytest.raises(runner.StepRefused):
        manager.plan([("scope", {}), ("cut", {"limit": 0})])
    assert ran == [] and manager.listing() == []


# --- the stage models -------------------------------------------------------------------------------------------------


class Score(NamedTuple):
    label: str
    probability: float


class FakeStage:
    description = "items: {text}"

    def __init__(self, _conn: Callable[[], Any]) -> None:
        self.calls = 0

    def predict(self, items: Sequence[Mapping[str, Any]]) -> list[Any]:
        self.calls += 1
        return [Score(str(i["text"]), 0.5) for i in items]


def _models(gate: GpuGate | None = None, builders: Mapping[str, Any] | None = None) -> StageModels:
    return StageModels(
        gate or GpuGate(), lambda: None, builders or {"fake": FakeStage, "other": FakeStage}
    )


def test_results_of_a_finder_are_plain_json() -> None:
    class Plain:
        def item(self) -> float:
            return 0.25

    assert jsonable([Score("a", 1.0), None, {"k": (1, 2)}, Plain()]) == [
        {"label": "a", "probability": 1.0},
        None,
        {"k": [1, 2]},
        0.25,
    ]


def test_a_stage_model_answers_and_only_one_stays_loaded() -> None:
    models = _models()
    assert models.predict("fake", [{"text": "x"}]) == [{"label": "x", "probability": 0.5}]
    assert models.stages()["fake"]["loaded"] == "yes"
    models.predict("other", [{"text": "y"}])
    assert models.stages()["fake"]["loaded"] == "no" and models.stages()["other"]["loaded"] == "yes"


def test_a_stage_call_while_a_job_holds_the_gpu_is_refused_and_releases_it_after_a_failure() -> (
    None
):
    gate = GpuGate()
    models = _models(gate)
    assert gate.acquire("job")
    with pytest.raises(Busy):
        models.predict("fake", [{"text": "x"}])
    gate.release()
    with pytest.raises(ValueError, match="between 1 and"):
        models.predict("fake", [])
    assert gate.holder is None
    with pytest.raises(KeyError):
        models.predict("nope", [{"text": "x"}])


# --- the routes -------------------------------------------------------------------------------------------------------


@pytest.fixture
def client_and_ran() -> Iterator[tuple[TestClient, list[str], GpuGate]]:
    manager, ran = _manager()
    models = _models(manager.gate)
    yield TestClient(api.create_app(manager, models, connection=lambda: None)), ran, manager.gate  # type: ignore[arg-type,return-value]


def test_health_answers_for_the_docker_healthcheck(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    assert client_and_ran[0].get("/health").json() == {"status": "ok"}


def test_the_steps_are_listed_in_the_order_of_the_pipeline_with_what_is_not_through_the_api(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    body = client_and_ran[0].get("/steps").json()
    assert body["order"][0] == "scope" and body["order"][-1] == "period"
    assert "reset" in body["not_through_the_api"] and "fetch" in body["not_through_the_api"]


def test_a_step_run_is_a_job_that_can_be_read(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    client, ran, _gate = client_and_ran
    response = client.post("/steps/scope/run", json={"options": {"limit": 5}})
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "succeeded" and ran == ["scope"]
    assert client.get(f"/jobs/{job['id']}").json()["steps"][0]["outputs"] == [{"step": "scope"}]
    assert [j["id"] for j in client.get("/jobs").json()] == [job["id"]]


def test_a_refused_option_is_422_with_the_cli_message_and_nothing_starts(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    client, ran, _gate = client_and_ran
    assert client.post("/steps/scope/run", json={"options": {"limit": 0}}).status_code == 422
    forbidden = client.post("/steps/scope/run", json={"options": {"reset": True}})
    assert (
        forbidden.status_code == 422
        and "not available through the API" in forbidden.json()["detail"]
    )
    assert client.post("/steps/nope/run", json={}).status_code == 404
    assert ran == []


def test_a_run_while_the_gpu_is_busy_is_409(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    client, ran, gate = client_and_ran
    assert gate.acquire("a long job")
    response = client.post("/steps/scope/run", json={})
    assert response.status_code == 409 and "a long job" in response.json()["detail"]
    assert ran == []


def test_the_whole_pipeline_runs_its_steps_in_order_and_refuses_another_order(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    client, ran, _gate = client_and_ran
    wrong = client.post("/pipeline/run", json={"steps": [{"step": "cut"}, {"step": "scope"}]})
    assert wrong.status_code == 422 and ran == []
    ok = client.post("/pipeline/run", json={"steps": [{"step": "scope"}, {"step": "cut"}]})
    assert ok.status_code == 202 and ran == ["scope", "cut"] and ok.json()["kind"] == "pipeline"


def test_an_unknown_job_is_404_and_a_finished_job_is_not_cancelled(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    client, _ran, _gate = client_and_ran
    assert client.get("/jobs/none").status_code == 404
    assert client.post("/jobs/none/cancel").status_code == 404
    job = client.post("/steps/scope/run", json={}).json()
    assert client.post(f"/jobs/{job['id']}/cancel").json()["cancel_requested"] is False


def test_the_stage_models_are_listed_and_asked_through_the_routes(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    client, _ran, gate = client_and_ran
    assert set(client.get("/fti/stages").json()["stages"]) == {"fake", "other"}
    ok = client.post("/fti/fake/predict", json={"items": [{"text": "x"}]})
    assert ok.json() == {"stage": "fake", "results": [{"label": "x", "probability": 0.5}]}
    assert client.post("/fti/nope/predict", json={"items": [{"text": "x"}]}).status_code == 404
    assert client.post("/fti/fake/predict", json={"items": []}).status_code == 422
    assert gate.acquire("a job")
    assert client.post("/fti/fake/predict", json={"items": [{"text": "x"}]}).status_code == 409


def test_a_stage_that_cannot_be_served_is_503() -> None:
    def unavailable(_conn: Callable[[], Any]) -> Any:
        raise StageUnavailable("no universe")

    models = _models(builders={"sector": unavailable})
    client = TestClient(api.create_app(_manager()[0], models, connection=lambda: None))  # type: ignore[arg-type,return-value]
    response = client.post("/fti/sector/predict", json={"items": [{"title": "t", "text": "x"}]})
    assert response.status_code == 503 and "no universe" in response.json()["detail"]


def test_the_real_stage_builders_are_the_stages_that_classify() -> None:
    assert tuple(BUILDERS) == STAGES


# --- the status of a step ---------------------------------------------------------------------------------------------


def test_the_status_of_a_step_is_read_from_the_tables_without_a_model(
    results_db_path: Path, source_db_path: Path
) -> None:
    def connection() -> Any:
        conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source_db_path)
        db_module.init_schema(conn)
        return conn

    client = TestClient(api.create_app(_manager()[0], _models(), connection=connection))
    body = client.get("/steps/scope/status").json()
    assert body["revision"] is None and body["revisions"] == []
    assert client.get("/steps/nope/status").status_code == 404
