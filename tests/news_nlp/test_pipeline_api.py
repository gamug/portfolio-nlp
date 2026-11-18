"""The service of the stepped pipeline (T-219): the CLI's own parser and functions behind jobs, the stage models, the routes. No model, no network."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
from fastapi.testclient import TestClient

import apps.pipeline_api as api
import news_nlp as db_module
from news_category_stage import LABELS
from news_nlp.service import jobs as jobs_module
from news_nlp.service import steps_runner as runner
from news_nlp.service.jobs import Busy, GpuGate, JobManager
from news_nlp.service.stage_models import (
    BUILDERS,
    MAX_TEXT_CHARS,
    STAGES,
    StageModels,
    StageUnavailable,
    _Category,
    _Companies,
    _Scope,
    _Sector,
    _Sentiment,
    jsonable,
)
from news_summary_llm import UNCAPPED_RUN_ITEMS

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
    job = manager.submit(
        "pipeline", manager.plan([("scope", {}), ("cut", {}), ("summary", {"max_calls": 10})])
    )
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

    def __init__(self, _conn: Callable[[], Any], _options: Mapping[str, Any] | None = None) -> None:
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


def test_an_unknown_job_is_404_and_a_finished_job_cannot_be_cancelled_409(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    client, _ran, _gate = client_and_ran
    assert client.get("/jobs/none").status_code == 404
    assert client.post("/jobs/none/cancel").status_code == 404
    job = client.post("/steps/scope/run", json={}).json()
    ended = client.post(f"/jobs/{job['id']}/cancel")
    assert ended.status_code == 409 and "already ended" in ended.json()["detail"]


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
    def unavailable(_conn: Callable[[], Any], _options: Mapping[str, Any]) -> Any:
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


# --- review of #209: what the API must refuse, and what it must not claim ----------------------------------------------


#: each abbreviation of a refused option, on the step where that option is legal for the CLI (so the CLI's own "does not apply to this step"
#: cannot be what refuses it), and the name of the option the parse sets
ABBREVIATIONS = [
    ("scope", {"fetc": True}, "fetch"),
    ("scope", {"allow-unp": True}, "allow_unpinned"),
    ("scope", {"model-d": "x"}, "model_dir"),
    ("scope", {"results-d": "x.db"}, "results_db"),
    ("scope", {"source-d": "y.db"}, "source_db"),
    (
        "snapshot",
        {
            "rese": True,
            "summary_revision": "s",
            "companies_revision": "c",
            "cut_revision": "u",
            "category_revision": "k",
            "sentiment_revision": "t",
        },
        "reset",
    ),
]


@pytest.mark.parametrize(("step", "options", "_name"), ABBREVIATIONS)
def test_an_abbreviation_of_a_refused_option_is_refused(
    step: str, options: dict[str, Any], _name: str
) -> None:
    """The CLI's parser takes any unique prefix of an option; ``rese`` was ``--reset``, which deletes the snapshot rows."""
    with pytest.raises(runner.StepRefused):
        runner.parse_step(step, options)


@pytest.mark.parametrize(("step", "options", "_name"), ABBREVIATIONS)
def test_the_exact_names_lock_refuses_each_abbreviation_on_its_own(
    step: str, options: dict[str, Any], _name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_reached", lambda _args: [])  # the second lock is off
    with pytest.raises(runner.StepRefused, match="unrecognized arguments"):
        runner.parse_step(step, options)


@pytest.mark.parametrize(("step", "options", "name"), ABBREVIATIONS)
def test_the_result_lock_refuses_each_one_on_its_own_when_abbreviations_get_through(
    step: str, options: dict[str, Any], name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_parse = argparse.ArgumentParser.parse_args

    def abbreviations_on(self: argparse.ArgumentParser, *a: Any, **k: Any) -> Any:
        self.allow_abbrev = True  # the first lock is off
        return real_parse(self, *a, **k)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", abbreviations_on)
    with pytest.raises(runner.StepRefused, match=f"{name} is not available through the API"):
        runner.parse_step(step, options)


def test_the_summary_through_the_api_needs_an_explicit_cap_on_the_paid_calls() -> None:
    with pytest.raises(runner.StepRefused, match="needs max_calls"):
        runner.parse_step("summary", {"items": "single"})
    assert runner.parse_step("summary", {"items": "single", "max_calls": 100}).max_calls == 100
    assert (
        runner.parse_step("summary", {"items": "single", "summariser": "local"}).summariser
        == "local"
    )


def test_the_summary_caps_are_bounded_not_only_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runner.MAX_CALLS_ENV, raising=False)
    monkeypatch.delenv(runner.CONCURRENCY_ENV, raising=False)
    ceiling = runner.calls_ceiling()
    assert ceiling == UNCAPPED_RUN_ITEMS  # the CLI's own allowance, until a deployment sets another
    assert (
        runner.parse_step("summary", {"items": "single", "max_calls": ceiling}).max_calls == ceiling
    )
    with pytest.raises(runner.StepRefused, match=f"over the ceiling of {ceiling} calls"):
        runner.parse_step("summary", {"items": "single", "max_calls": ceiling + 1})
    with pytest.raises(runner.StepRefused, match="over the ceiling of 16"):
        runner.parse_step("summary", {"items": "single", "max_calls": 10, "concurrency": 17})
    assert runner.parse_step("summary", {"items": "single", "max_calls": 10, "concurrency": 16})
    with pytest.raises(
        runner.StepRefused, match="only apply to the API summariser"
    ):  # the CLI's own rule
        runner.parse_step(
            "summary", {"items": "single", "summariser": "local", "concurrency": 100000}
        )
    monkeypatch.setenv(runner.MAX_CALLS_ENV, "100")
    monkeypatch.setenv(runner.CONCURRENCY_ENV, "2")
    with pytest.raises(runner.StepRefused, match="over the ceiling of 100 calls"):
        runner.parse_step("summary", {"items": "single", "max_calls": 101})
    with pytest.raises(runner.StepRefused, match="over the ceiling of 2"):
        runner.parse_step("summary", {"items": "single", "max_calls": 10, "concurrency": 3})
    monkeypatch.setenv(runner.MAX_CALLS_ENV, "not a number")
    assert (
        runner.calls_ceiling() == UNCAPPED_RUN_ITEMS
    )  # an unreadable value is the default, not a crash


def test_a_thread_that_does_not_start_leaves_the_gpu_free(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = GpuGate()
    manager = JobManager(gate=gate, executor=lambda *_a: runner.StepOutcome(ok=True), spawn=True)
    steps = manager.plan([("scope", {})])

    def no_thread(self: Any) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", no_thread)
    with pytest.raises(RuntimeError, match="can't start"):
        manager.submit("step", steps)
    assert gate.holder is None


def test_cancel_stops_a_pipeline_that_has_a_step_left_and_says_no_for_the_last_step() -> None:
    answers: dict[str, tuple[Any, str | None]] = {}

    def cancel_while_running(step: str, manager: JobManager) -> None:
        job_id = manager.listing()[0].id
        answers[step] = manager.cancel(job_id)

    manager, ran = _manager(on_run=cancel_while_running)
    job = manager.submit("pipeline", manager.plan([("scope", {}), ("cut", {})]))
    assert answers["scope"][1] is None  # the cut is still to run: the stop is accepted
    assert ran == ["scope"] and job.status == jobs_module.CANCELLED
    lone, _ = _manager(on_run=cancel_while_running)
    single = lone.submit("step", lone.plan([("scope", {})]))
    reason = answers["scope"][1]  # overwritten by the single-step run: nothing to skip
    assert reason is not None and "cannot be stopped in the middle" in reason
    assert single.status == jobs_module.SUCCEEDED and single.cancel_requested is False


def test_cancel_of_a_running_single_step_is_409_through_the_route() -> None:
    holder: list[JobManager] = []
    outcome: dict[str, Any] = {}

    def executor(_args: argparse.Namespace, _progress: Any) -> runner.StepOutcome:
        job_id = holder[0].listing()[0].id
        outcome["response"] = TestClient(api.create_app(holder[0], _models(holder[0].gate))).post(
            f"/jobs/{job_id}/cancel"
        )
        return runner.StepOutcome(ok=True)

    manager = JobManager(executor=executor, spawn=False)
    holder.append(manager)
    manager.submit("step", manager.plan([("scope", {})]))
    assert outcome["response"].status_code == 409
    assert "restart the service" in outcome["response"].json()["detail"]


# --- the real stage builders, with their heavy parts stubbed ------------------------------------------------------------


@pytest.fixture
def universe(results_db_path: Path, source_db_path: Path) -> Callable[[], Any]:
    return lambda: db_module.connect_pipeline(results_db=results_db_path, source_db=source_db_path)


def test_the_scope_stage_model_refuses_a_folder_that_is_not_the_pinned_model(
    tmp_path: Path,
) -> None:
    (tmp_path / "training_record.json").write_text('{"fragment_chars": 800}', encoding="utf-8")
    with pytest.raises(StageUnavailable, match="not the pinned model"):
        _Scope(lambda: None, {}, model_dir=tmp_path)
    with pytest.raises(StageUnavailable, match="missing"):
        _Scope(lambda: None, {}, model_dir=tmp_path / "nowhere")


def test_the_category_stage_model_builds_and_answers_with_a_stub_scorer() -> None:
    class Scorer:
        model_name, model_revision = "stub", "r0"

        def score(self, pairs: Sequence[tuple[str, str]]) -> list[list[float]]:
            return [[1.0 if i == 0 else 0.0 for i in range(len(LABELS))] for _ in pairs]

    model = _Category(lambda: None, {}, scorer=Scorer())
    [result] = [jsonable(r) for r in model.predict([{"text": "t", "company": "Apple"}])]
    assert result["label"] == LABELS[0] and set(result) >= {"label", "score", "group_label"}
    with pytest.raises(ValueError, match="text, company"):
        model.predict([{"text": "t"}])


def test_the_sentiment_stage_model_needs_the_relevance_and_answers_with_a_stub_scorer() -> None:
    class Scorer:
        def identity(self) -> str:
            return "stub"

        def probabilities(self, pairs: Sequence[tuple[str, str]]) -> list[Any]:
            return [([0.1, 0.8, 0.1], 1) for _ in pairs]

    model = _Sentiment(lambda: None, {}, scorer=Scorer())
    [result] = [
        jsonable(r) for r in model.predict([{"text": "t", "company": "Apple", "relevance": 0.9}])
    ]
    assert result["label"] in {"positive", "negative", "neutral"}
    with pytest.raises(ValueError, match="relevance"):
        model.predict([{"text": "t", "company": "Apple"}])


def test_the_companies_stage_model_needs_the_threshold_and_the_candidates_like_the_cli(
    universe: Callable[[], Any],
) -> None:
    class Spans:
        model_name, model_revision = "stub", "r0"

        def spans(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
            return [[] for _ in texts]

    class Relevance:
        repo, revision = "stub", "r0"

        def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
            return [0.9 for _ in pairs]

    with pytest.raises(ValueError, match="threshold, candidates"):
        _Companies(universe, {}, spans=Spans(), scorer=Relevance())
    with pytest.raises(ValueError, match="candidates"):
        _Companies(universe, {"threshold": 0.5}, spans=Spans(), scorer=Relevance())
    model = _Companies(
        universe, {"threshold": 0.5, "candidates": "ner"}, spans=Spans(), scorer=Relevance()
    )
    assert model.predict([{"text": "Nothing about a company"}]) == [[]]


def test_the_sector_stage_model_builds_from_the_universe_and_answers_with_a_stub_embedder(
    universe: Callable[[], Any],
) -> None:
    def embed(texts: list[str]) -> Any:
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(len(texts), 8))
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    model = _Sector(universe, {}, embed=embed)
    [result] = [
        jsonable(r) for r in model.predict([{"title": "Oil prices", "text": "Crude rose."}])
    ]
    assert set(result) >= {"status", "sector"}


def test_the_stage_options_that_build_a_model_rebuild_it_when_they_change() -> None:
    built: list[Mapping[str, Any]] = []

    class Counting(FakeStage):
        options = ("threshold",)

        def __init__(
            self, conn: Callable[[], Any], options: Mapping[str, Any] | None = None
        ) -> None:
            super().__init__(conn, options)
            built.append(dict(options or {}))

    models = _models(builders={"fake": Counting})
    models.predict("fake", [{"text": "x"}], {"threshold": 0.5})
    models.predict("fake", [{"text": "x"}], {"threshold": 0.5})
    models.predict("fake", [{"text": "x"}], {"threshold": 0.6})
    assert built == [{"threshold": 0.5}, {"threshold": 0.6}]


# --- the status route reads while a job writes ------------------------------------------------------------------------


@pytest.fixture
def real_store(
    results_db_path: Path, source_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.setenv("SOURCE_DATABASE_URL", str(source_db_path))
    conn = db_module.connect_pipeline(results_db=results_db_path, source_db=source_db_path)
    db_module.init_schema(conn)
    conn.commit()
    conn.close()
    return results_db_path


@pytest.mark.parametrize("step", list(runner.PIPELINE_ORDER))
def test_the_status_of_every_step_reads_an_empty_store_with_the_default_connection(
    real_store: Path, step: str
) -> None:
    client = TestClient(api.create_app(_manager()[0], _models()))
    response = client.get(f"/steps/{step}/status")
    assert response.status_code == 200, response.text


def test_the_status_reads_while_another_connection_holds_a_write_transaction(
    real_store: Path,
) -> None:
    """A job writes for hours: the status must neither wait for it nor write (no init_schema, query_only)."""
    writer = sqlite3.connect(real_store, isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    try:
        client = TestClient(api.create_app(_manager()[0], _models()))
        assert client.get("/steps/scope/status").status_code == 200
    finally:
        writer.execute("ROLLBACK")
        writer.close()


def test_the_default_connection_cannot_write(real_store: Path) -> None:
    conn = api.default_connection()
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM body_scope")
    finally:
        conn.close()


def test_a_store_that_cannot_be_read_is_503_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def locked() -> Any:
        raise sqlite3.OperationalError("database is locked")

    client = TestClient(api.create_app(_manager()[0], _models(), connection=locked))
    response = client.get("/steps/scope/status")
    assert response.status_code == 503 and "locked" in response.json()["detail"]


# --- the size of what a request may carry ------------------------------------------------------------------------------


def test_a_text_over_the_cap_is_refused_and_one_at_the_cap_is_answered() -> None:
    models = _models()
    assert models.predict("fake", [{"text": "x" * MAX_TEXT_CHARS}])
    with pytest.raises(ValueError, match=f"at most {MAX_TEXT_CHARS:,}"):
        models.predict("fake", [{"text": "x" * (MAX_TEXT_CHARS + 1)}])


def test_the_routes_refuse_a_long_text_and_a_body_over_the_limit(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    client, _ran, _gate = client_and_ran
    long_text = client.post("/fti/fake/predict", json={"items": [{"text": "x" * 50_001}]})
    assert long_text.status_code == 422 and "at most 50,000" in long_text.json()["detail"]
    huge = client.post(
        "/fti/fake/predict",
        content=b"x" * (api.MAX_BODY_BYTES + 1),
        headers={"content-type": "application/json"},
    )
    assert huge.status_code == 413 and "bytes" in huge.json()["detail"]


def test_the_body_limit_counts_a_chunked_upload_that_declares_no_length(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    """The ``Content-Length`` check was bypassed by chunking: 6 MB was buffered and parsed before the text cap refused it."""
    client, _ran, _gate = client_and_ran

    def chunks(megabytes: int) -> Iterator[bytes]:
        yield b'{"items": [{"text": "'
        for _ in range(megabytes):
            yield b"x" * 1_000_000
        yield b'"}]}'

    over = client.post(
        "/fti/fake/predict", content=chunks(5), headers={"content-type": "application/json"}
    )
    assert over.status_code == 413 and "bytes" in over.json()["detail"]
    under = client.post(
        "/fti/fake/predict", content=chunks(1), headers={"content-type": "application/json"}
    )
    assert under.status_code == 422  # passes the limit, refused later by the 50,000-character cap
    assert "at most 50,000" in under.json()["detail"]


def test_a_cross_site_simple_request_cannot_start_a_run(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    """A page can send a POST with ``text/plain`` (or none) without a preflight. It is refused today because FastAPI wants a JSON content type;
    that is a property of the installed library, so a test keeps it: an upgrade that changes it would let any web page start a paid run."""
    client, ran, _gate = client_and_ran
    body = '{"options": {"max_calls": 10}}'
    for headers in ({"content-type": "text/plain"}, {}):
        for route in ("/steps/summary/run", "/steps/scope/run", "/pipeline/run"):
            response = client.post(route, content=body, headers=headers)
            assert response.status_code == 422, (route, headers, response.text)
    assert ran == [] and client.get("/jobs").json() == []


def test_an_option_a_stage_does_not_take_is_refused_and_does_not_rebuild_the_model() -> None:
    built: list[Mapping[str, Any]] = []

    class Counting(FakeStage):
        options = ("threshold",)

        def __init__(
            self, conn: Callable[[], Any], options: Mapping[str, Any] | None = None
        ) -> None:
            super().__init__(conn, options)
            built.append(dict(options or {}))

    class TakesNone(FakeStage):
        def __init__(
            self, conn: Callable[[], Any], options: Mapping[str, Any] | None = None
        ) -> None:
            super().__init__(conn, options)
            built.append({"none": True})

    models = _models(builders={"counting": Counting, "none": TakesNone})
    for junk in range(3):  # a caller cannot make the GPU reload by varying an option nobody reads
        with pytest.raises(ValueError, match=r"takes no option junk"):
            models.predict("none", [{"text": "x"}], {"junk": junk})
    with pytest.raises(ValueError, match=r"takes no option other.*it takes: threshold"):
        models.predict("counting", [{"text": "x"}], {"threshold": 0.5, "other": 1})
    assert built == []  # nothing was built, no GPU was touched
    models.predict("counting", [{"text": "x"}], {"threshold": 0.5})
    models.predict("counting", [{"text": "x"}], {"threshold": 0.5})
    assert built == [{"threshold": 0.5}]  # equal declared options: built once


def test_the_route_says_422_for_an_option_the_stage_does_not_take(
    client_and_ran: tuple[TestClient, list[str], GpuGate],
) -> None:
    client, _ran, gate = client_and_ran
    response = client.post(
        "/fti/fake/predict", json={"items": [{"text": "x"}], "options": {"junk": 1}}
    )
    assert response.status_code == 422 and "takes no option junk" in response.json()["detail"]
    assert gate.holder is None
