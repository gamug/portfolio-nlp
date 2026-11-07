"""The summaries of the backtest cut can be stopped and resumed (T-148a, scripts/cut_summaries)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import pytest
import torch
from conftest import REAL_LOAD_ENV_FILE

from news_nlp import triage

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CUT = _script("cut_summaries_2026_11_07")


@pytest.fixture(autouse=True)
def _fixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The versions and device are read from the machine; the tests fix them."""
    monkeypatch.setattr(
        CUT, "environment", lambda d=None: f"torch 1, transformers 1, {d or 'cuda'}"
    )


def _index(n: int) -> list[dict[str, str]]:
    return [{"body_hash": f"h{i}", "article_ids": str(i)} for i in range(n)]


def _processor(calls: list[str]) -> Callable[[Sequence[dict[str, str]]], list[dict[str, str]]]:
    def process(part: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        calls.extend(r["body_hash"] for r in part)
        return [
            {
                "body_hash": r["body_hash"],
                "article_ids": r["article_ids"],
                "rule_scope": "single",
                "gold_scope": "single",
                "status": CUT.SUMMARISED,
                "summary": f"summary of {r['body_hash']}",
                "num_chunks": "1",
                "error": "",
            }
            for r in part
        ]

    return process


def test_the_index_lists_each_body_once_in_order_of_its_first_article() -> None:
    pairs = [(5, "Same body."), (2, "Other body."), (9, "Same body."), (3, "  "), (4, "")]
    index = CUT.build_index(pairs)
    assert [r["article_ids"] for r in index] == ["2", "5 9"]
    assert index[1]["body_hash"] == triage.body_hash("Same body.")


def test_a_stopped_job_resumes_where_it_stopped_and_never_repeats_a_chunk(tmp_path: Path) -> None:
    parts = CUT.slices(_index(10), 3)  # 4 chunks
    calls: list[str] = []
    messages: list[str] = []
    made = CUT.run_chunks(tmp_path, parts, _processor(calls), max_chunks=2, log=messages.append)
    assert made == 2 and len(calls) == 6
    assert sorted(p.name for p in tmp_path.glob("chunk_*")) == [
        "chunk_00001.csv",
        "chunk_00002.csv",
    ]
    made = CUT.run_chunks(tmp_path, parts, _processor(calls), log=messages.append)
    assert made == 2 and len(calls) == 10  # only the 4 bodies of the last two chunks
    assert sorted(calls) == sorted(f"h{i}" for i in range(10))  # nobody twice
    again: list[str] = []
    assert CUT.run_chunks(tmp_path, parts, _processor(again), log=messages.append) == 0
    assert again == []  # a finished job does nothing


def test_a_stop_request_is_honoured_between_chunks(tmp_path: Path) -> None:
    parts = CUT.slices(_index(9), 3)
    calls: list[str] = []
    seen = {"n": 0}

    def stop_after_first() -> bool:
        seen["n"] += 1
        return seen["n"] > 1

    made = CUT.run_chunks(
        tmp_path, parts, _processor(calls), should_stop=stop_after_first, log=lambda _m: None
    )
    assert made == 1 and len(calls) == 3


def test_a_killed_write_leaves_no_half_chunk(tmp_path: Path) -> None:
    parts = CUT.slices(_index(6), 3)
    (tmp_path / "chunk_00001.csv.tmp").write_text("half a chunk", encoding="utf-8")

    def dies(_part: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        CUT.run_chunks(tmp_path, parts, dies, log=lambda _m: None)
    assert not list(tmp_path.glob("chunk_*.csv"))  # nothing whole was written
    calls: list[str] = []
    CUT.run_chunks(tmp_path, parts, _processor(calls), log=lambda _m: None)
    assert not list(tmp_path.glob("*.tmp"))  # the leftover was removed at the start
    assert len(calls) == 6


def test_a_chunk_that_is_not_its_slice_stops_the_run(tmp_path: Path) -> None:
    parts = CUT.slices(_index(6), 3)
    CUT.run_chunks(tmp_path, parts, _processor([]), max_chunks=1, log=lambda _m: None)
    changed = CUT.slices(_index(7)[1:], 3)  # another index: same file names, other bodies
    with pytest.raises(SystemExit, match="does not hold the bodies"):
        CUT.run_chunks(tmp_path, changed, _processor([]), log=lambda _m: None)


def test_the_manifest_refuses_a_job_made_under_other_conditions(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    now = {"since": "2021-12-31", "summariser": "m@r1", "chunk_size": 250}
    CUT.check_manifest(path, now)
    CUT.check_manifest(path, dict(now))  # the same conditions: fine
    with pytest.raises(SystemExit, match="other conditions"):
        CUT.check_manifest(path, {**now, "summariser": "m@r2"})
    assert json.loads(path.read_text(encoding="utf-8")) == {**now, "ner_newest_id": ""}  # untouched


def test_status_counts_what_is_done_and_loads_nothing(tmp_path: Path) -> None:
    parts = CUT.slices(_index(5), 2)
    CUT.run_chunks(tmp_path, parts, _processor([]), max_chunks=2, log=lambda _m: None)
    lines = CUT.status(tmp_path, parts)
    assert lines[0].startswith("2 of 3 chunks done (4 of 5 bodies)")
    assert "'single': 4" in lines[1] and "summaries so far: 4" in lines[2]


def test_conditions_digest_the_index_and_the_rules(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    CUT.atomic_write(index, CUT.INDEX_FIELDS, _index(3))
    args = argparse.Namespace(since="2021-12-31", chunk_size=250)
    first = CUT.conditions(args, index, "10:99")
    CUT.atomic_write(index, CUT.INDEX_FIELDS, _index(4))
    assert CUT.conditions(args, index, "10:99")["index"] != first["index"]
    assert first["summariser"].startswith("sshleifer/distilbart-cnn-12-6@")
    assert CUT.conditions(args, index, "11:100")["ner_spans"] == "11:100"  # new spans: new job


def test_a_second_run_while_the_first_is_alive_stops_and_a_dead_lock_is_taken_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts = CUT.slices(_index(3), 3)
    (tmp_path / CUT.LOCK_NAME).write_text("424242", encoding="utf-8")
    monkeypatch.setattr(CUT, "_alive", lambda pid: True)  # the first run is still finishing
    (tmp_path / "chunk_00001.csv.424242.tmp").write_text("its half chunk", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"another run \(pid 424242"):
        CUT.run_chunks(tmp_path, parts, _processor([]), log=lambda _m: None)
    assert (tmp_path / "chunk_00001.csv.424242.tmp").exists()  # the other run's file is not touched
    monkeypatch.setattr(CUT, "_alive", lambda pid: False)  # it died
    assert CUT.run_chunks(tmp_path, parts, _processor([]), log=lambda _m: None) == 1
    assert not (tmp_path / CUT.LOCK_NAME).exists()  # released at the end
    assert not list(tmp_path.glob("*.tmp"))


def test_alive_tells_a_running_process_from_a_finished_one() -> None:
    assert CUT._alive(os.getpid())
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    assert not CUT._alive(done.pid)  # a pid is not reused within this instant


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows probe runs only on Windows")
def test_the_windows_probe_itself_on_a_live_and_a_finished_process() -> None:
    assert CUT._alive_windows(os.getpid())
    exit_259 = subprocess.Popen([sys.executable, "-c", "raise SystemExit(259)"])  # STILL_ACTIVE
    exit_259.wait()
    assert not CUT._alive_windows(exit_259.pid)
    assert not CUT._alive_windows(0x7FFFFFF0)  # no such pid
    assert CUT._started(os.getpid()) != ""


def test_a_stale_pid_file_and_its_lock_are_cleared_in_one_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the crash was: ``run.lock`` and ``run.lock.<pid>`` of a killed run, then a resume."""
    (tmp_path / CUT.LOCK_NAME).write_text("41956", encoding="utf-8")
    (tmp_path / f"{CUT.LOCK_NAME}.41956").write_text("41956", encoding="utf-8")
    monkeypatch.setattr(CUT, "_alive", lambda pid: False)
    with CUT.locked(tmp_path):
        assert (tmp_path / CUT.LOCK_NAME).read_text(encoding="utf-8").split()[0] == str(os.getpid())
        assert not (tmp_path / f"{CUT.LOCK_NAME}.41956").exists()
    assert not list(tmp_path.glob(f"{CUT.LOCK_NAME}*"))


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (OSError(1, "Operation not permitted"), "NTFS"),  # EPERM: no hard links here
        (OSError(18, "Invalid cross-device link"), "NTFS"),  # EXDEV
        (PermissionError(13, "Access is denied"), "something holds the file"),  # EACCES
        (OSError(5, "I/O error"), "something holds the file"),
    ],
)
def test_a_failed_hard_link_names_the_cause_that_the_error_tells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError, message: str
) -> None:
    def no_link(*_a: object) -> None:
        raise error

    monkeypatch.setattr(CUT.os, "link", no_link)
    with pytest.raises(SystemExit, match=message), CUT.locked(tmp_path):
        pass
    assert not list(tmp_path.glob(f"{CUT.LOCK_NAME}*"))  # the private file is removed


def test_the_windows_error_for_a_file_system_without_hard_links_names_ntfs() -> None:
    exc = OSError(22, "Incorrect function")
    exc.winerror = 1  # type: ignore[attr-defined]
    assert "NTFS" in CUT._cannot_lock(Path("x"), exc)


def test_a_stale_private_file_is_judged_by_its_start_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CUT, "_alive", lambda pid: True)
    monkeypatch.setattr(CUT, "_started", lambda pid: "222" if pid != os.getpid() else "1")
    reused = tmp_path / f"{CUT.LOCK_NAME}.41956"
    mine_still = tmp_path / f"{CUT.LOCK_NAME}.41957"
    reused.write_text("41956 111", encoding="utf-8")  # another process holds that pid now
    mine_still.write_text("41957 222", encoding="utf-8")  # the process that wrote it
    with CUT.locked(tmp_path):
        assert not reused.exists()
        assert mine_still.exists()


def test_on_windows_the_pid_is_never_probed_with_os_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``os.kill(pid, 0)`` terminates the process on Windows (and raised WinError 11 in a real run)."""

    def boom(*_a: object) -> None:
        raise AssertionError("os.kill must not be called on Windows")

    monkeypatch.setattr(CUT.sys, "platform", "win32")
    monkeypatch.setattr(CUT.os, "kill", boom)
    monkeypatch.setattr(CUT, "_alive_windows", lambda pid: pid == 7)
    assert CUT._alive(7)
    assert not CUT._alive(8)


class _FakeWin32:
    """What ``_Win32`` answers, set per test: the handle (0 = OpenProcess failed), the last error,
    the wait result and the creation time."""

    def __init__(
        self,
        handle: int = 1,
        error: int = 0,
        wait: int = 0x102,
        created: int | None = 5,
        sync_denied: bool = False,
    ) -> None:
        self.handle, self.error, self.wait_result, self.created_at = handle, error, wait, created
        self.sync_denied = sync_denied  # a protected process: refuses the wait right only
        self.closed: list[int] = []

    def open(self, _pid: int, *, wait: bool = False) -> int:
        if wait and self.sync_denied:
            self.error = 5
            return 0
        return self.handle

    def last_error(self) -> int:
        return self.error

    def wait(self, _handle: int) -> int:
        return self.wait_result

    def created(self, _handle: int) -> int | None:
        return self.created_at

    def close(self, handle: int) -> None:
        self.closed.append(handle)


@pytest.mark.parametrize(
    ("api", "alive"),
    [
        (_FakeWin32(wait=0x102), True),  # WAIT_TIMEOUT: running
        (_FakeWin32(wait=0), False),  # WAIT_OBJECT_0: ended
        (_FakeWin32(wait=0xFFFFFFFF), True),  # WAIT_FAILED: unknown, so alive
        (_FakeWin32(handle=0, error=87), False),  # no such pid
        (_FakeWin32(handle=0, error=5), True),  # access denied: exists
        (_FakeWin32(handle=0, error=31), True),  # any other error: unknown, so alive
    ],
)
def test_the_windows_probe_answers_dead_only_when_the_os_says_so(
    api: _FakeWin32, alive: bool
) -> None:
    assert CUT._alive_windows(1234, api) is alive  # type: ignore[arg-type]
    assert api.closed == ([api.handle] if api.handle else [])  # the handle is always closed


def test_a_protected_process_still_gives_its_start_time_without_the_wait_right(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _FakeWin32(sync_denied=True, created=99)
    monkeypatch.setattr(CUT.sys, "platform", "win32")
    assert CUT._alive_windows(4, api)  # type: ignore[arg-type]  # access denied: exists
    assert CUT._started(4, api) == "99"  # type: ignore[arg-type]  # read with the query right alone


def test_a_truncated_proc_stat_gives_no_start_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stat = tmp_path / "stat"
    stat.write_text("1 (cut) S 0 1", encoding="utf-8")
    real_read = Path.read_text
    monkeypatch.setattr(CUT.sys, "platform", "linux")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *a, **k: real_read(stat if "/proc/" in str(self) else self, *a, **k),
    )
    assert CUT._started(1) == ""


def test_the_windows_start_time_is_empty_when_it_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(CUT.sys, "platform", "win32")
    assert CUT._started(1, _FakeWin32(created=77)) == "77"  # type: ignore[arg-type]
    assert CUT._started(1, _FakeWin32(created=None)) == ""  # type: ignore[arg-type]
    assert CUT._started(1, _FakeWin32(handle=0)) == ""  # type: ignore[arg-type]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_the_start_time_of_a_running_process_is_stable() -> None:
    assert CUT._started(os.getpid()) != ""
    assert CUT._started(os.getpid()) == CUT._started(os.getpid())


def test_a_pid_another_process_now_holds_does_not_keep_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / CUT.LOCK_NAME).write_text("41956 111", encoding="utf-8")
    monkeypatch.setattr(CUT, "_alive", lambda pid: True)  # some program has that pid now
    monkeypatch.setattr(CUT, "_started", lambda pid: "222")  # ... but it started later
    with CUT.locked(tmp_path):
        pass
    (tmp_path / CUT.LOCK_NAME).write_text("41956 222", encoding="utf-8")  # the same process
    with pytest.raises(SystemExit, match=r"another run \(pid 41956"), CUT.locked(tmp_path):
        pass
    (tmp_path / CUT.LOCK_NAME).write_text("41956", encoding="utf-8")  # old format: pid only
    with pytest.raises(SystemExit, match=r"another run \(pid 41956"), CUT.locked(tmp_path):
        pass


def test_a_lock_released_between_the_link_and_the_read_is_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / CUT.LOCK_NAME).write_text("41956", encoding="utf-8")
    real_read = Path.read_text

    def vanishing(self: Path, *a: Any, **k: Any) -> str:
        if self.name == CUT.LOCK_NAME:
            self.unlink()  # the other run lets go right now
            raise FileNotFoundError(str(self))
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", vanishing)
    with CUT.locked(tmp_path):
        assert (tmp_path / CUT.LOCK_NAME).exists()


def test_ctrl_break_stops_like_ctrl_c_where_it_exists() -> None:
    assert {CUT.signal.SIGINT, CUT.signal.SIGTERM} <= set(CUT.STOP_SIGNALS)
    if hasattr(CUT.signal, "SIGBREAK"):  # Windows
        assert CUT.signal.SIGBREAK in CUT.STOP_SIGNALS


def test_the_lock_is_released_when_a_chunk_fails(tmp_path: Path) -> None:
    def dies(_part: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        raise RuntimeError("model error")

    with pytest.raises(RuntimeError):
        CUT.run_chunks(tmp_path, CUT.slices(_index(3), 3), dies, log=lambda _m: None)
    assert not (tmp_path / CUT.LOCK_NAME).exists()


class _Cursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows

    def fetchone(self) -> tuple[object, ...]:
        return self.rows[0]

    def __iter__(self) -> Iterator[tuple[object, ...]]:
        return iter(self.rows)


class _Conn:
    """Two articles of one body under two headlines, and NER spans for the second only."""

    def __init__(self, body: str) -> None:
        self.body = body

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> _Cursor:
        if "FROM source.articles a" in sql:
            title, url = {1: ("Own title", "x/own"), 2: ("Stocks to watch today", "x/roundup")}[
                int(params[0])  # type: ignore[call-overload]
            ]
            return _Cursor([(title, self.body, url)])
        if "article_entities" in sql:
            return _Cursor([("Acme",)] if params == (2,) else [])
        raise AssertionError(sql)


def _processor_under_test(
    monkeypatch: pytest.MonkeyPatch, scope: str, body: str
) -> tuple[Callable[[Sequence[dict[str, str]]], list[dict[str, str]]], dict[str, object]]:
    seen: dict[str, object] = {"loaded": 0, "inputs": [], "classified_title": None}
    monkeypatch.setattr(CUT.size_triage, "load_universe", lambda _c: [])
    monkeypatch.setattr(CUT.size_triage, "sector_terms", lambda _u: {})

    def classify(_r, title, *_a, **_k):  # type: ignore[no-untyped-def]
        seen["classified_title"] = title
        return triage.Scope(scope, 1, 1.0), None

    monkeypatch.setattr(CUT.size_triage, "classify", classify)

    def load(_device: str | None = None) -> Callable[[list[str]], list[tuple[str, int]]]:
        seen["loaded"] = int(seen["loaded"]) + 1  # type: ignore[call-overload]

        def summarise(texts: list[str]) -> list[tuple[str, int]]:
            seen["inputs"] = texts
            return [("a summary", 2) for _ in texts]

        return summarise

    monkeypatch.setattr(CUT, "load_summariser", load)
    return CUT.make_processor(_Conn(body)), seen  # type: ignore[arg-type]


def test_the_processor_summarises_a_single_body_from_its_own_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = "Acme rose."
    process, seen = _processor_under_test(monkeypatch, "single", body)
    entry = {"body_hash": triage.body_hash(body), "article_ids": "1 2"}
    [row] = process([entry])
    assert (row["rule_scope"], row["gold_scope"], row["summary"], row["num_chunks"]) == (
        "single",
        "single",
        "a summary",
        "2",
    )
    assert seen["inputs"] == [f"Title-Own title\nBody-{body}"]  # not the round-up headline's
    assert seen["classified_title"] == "Stocks to watch today"  # the round-up test reads that one
    assert row["spans"] == "1"  # the spans of the first article that has any


def test_a_multi_body_gets_a_scope_and_no_summary_and_no_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = "Acme and Beta rose."
    process, seen = _processor_under_test(monkeypatch, "multi", body)
    [row] = process([{"body_hash": triage.body_hash(body), "article_ids": "1"}])
    assert (row["rule_scope"], row["summary"], row["num_chunks"]) == ("multi", "", "")
    assert seen["loaded"] == 0 and row["spans"] == "0"


def test_a_body_that_changed_stops_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    process, _ = _processor_under_test(monkeypatch, "single", "Acme rose.")
    with pytest.raises(SystemExit, match="no longer matches"):
        process([{"body_hash": "not-the-hash", "article_ids": "1"}])


def test_the_slow_span_count_is_repeated_only_when_ner_wrote_something(tmp_path: Path) -> None:
    class Conn:
        def __init__(self, newest: int, spans: tuple[int, int]) -> None:
            self.newest, self.spans, self.slow_runs = newest, spans, 0

        def execute(self, sql: str, _p: tuple[object, ...] = ()) -> _Cursor:
            if sql == CUT.NEWEST_ENTITY_SQL:
                return _Cursor([(self.newest,)])
            self.slow_runs += 1
            return _Cursor([self.spans])

    path = tmp_path / "manifest.json"
    conn = Conn(10, (5, 10))
    spans, newest = CUT.entities_basis(conn, "2021-12-31", path)  # type: ignore[arg-type]
    assert (spans, newest, conn.slow_runs) == ("5:10", "10", 1)
    now = {"ner_spans": spans, "x": 1}
    CUT.check_manifest(path, now, newest)
    assert CUT.entities_basis(conn, "2021-12-31", path)[0] == "5:10" and conn.slow_runs == 1
    # NER wrote for other articles: the table has a newer row, the window's count is the same
    conn.newest = 11
    spans, newest = CUT.entities_basis(conn, "2021-12-31", path)  # type: ignore[arg-type]
    assert conn.slow_runs == 2
    CUT.check_manifest(path, now, newest)  # accepted, bookkeeping refreshed
    assert json.loads(path.read_text(encoding="utf-8"))["ner_newest_id"] == "11"
    # NER wrote for the window: other spans, the job is refused
    conn.newest, conn.spans = 12, (6, 12)
    spans, newest = CUT.entities_basis(conn, "2021-12-31", path)  # type: ignore[arg-type]
    with pytest.raises(SystemExit, match="other conditions"):
        CUT.check_manifest(path, {"ner_spans": spans, "x": 1}, newest)


def test_new_ner_spans_are_refused_unless_accepted_and_nothing_else_is(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    now = {"ner_spans": "5:10", "summariser": "m@r1"}
    CUT.check_manifest(path, now, "10")
    changed = {**now, "ner_spans": "6:12"}
    with pytest.raises(SystemExit, match=r"--accept-new-spans"):
        CUT.check_manifest(path, changed, "12")
    CUT.check_manifest(path, changed, "12", accept_new_spans=True, chunks_done=7)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["ner_spans"] == "6:12"
    assert saved["ner_spans_history"] == [{"ner_spans": "5:10", "chunks_made_under_it": 7}]
    CUT.check_manifest(path, changed, "13")  # the same basis again: no new history
    assert len(json.loads(path.read_text(encoding="utf-8"))["ner_spans_history"]) == 1
    other = {**changed, "summariser": "m@r2"}
    with pytest.raises(SystemExit, match="other conditions"):
        CUT.check_manifest(path, other, "13", accept_new_spans=True)  # only spans may be accepted


def test_the_lock_always_has_its_pid_and_can_be_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with CUT.locked(tmp_path):
        assert (tmp_path / CUT.LOCK_NAME).read_text(encoding="utf-8").split()[0].isdigit()
        assert not list(tmp_path.glob("run.lock.*"))  # the private file is gone
    (tmp_path / CUT.LOCK_NAME).write_text("424242", encoding="utf-8")
    monkeypatch.setattr(CUT, "_alive", lambda pid: True)
    with pytest.raises(SystemExit, match="--break-lock"), CUT.locked(tmp_path):
        pass
    with CUT.locked(tmp_path, break_lock=True):  # a pid reused after a restart
        pass
    assert not (tmp_path / CUT.LOCK_NAME).exists()


def test_an_empty_summary_is_retried_once_and_then_marked_never_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def flaky(texts: list[str]) -> list[tuple[str, int]]:
        calls.append(texts)
        return [("" if t.endswith("bad") and len(calls) == 1 else "ok", 1) for t in texts]

    assert CUT.summarise_with_retry(flaky, ["a", "b bad"]) == [("ok", 1, ""), ("ok", 1, "")]
    assert calls == [["a", "b bad"], ["b bad"]]
    process, _ = _processor_under_test(monkeypatch, "single", "Acme rose.")
    monkeypatch.setattr(
        CUT, "load_summariser", lambda _d=None: lambda texts: [("", 0)] * len(texts)
    )
    process, _ = _processor_under_test(monkeypatch, "single", "Acme rose.")
    monkeypatch.setattr(
        CUT, "load_summariser", lambda _d=None: lambda texts: [("", 0)] * len(texts)
    )
    entry = {"body_hash": triage.body_hash("Acme rose."), "article_ids": "1"}
    [row] = process([entry])
    assert row["status"] == CUT.EMPTY and row["summary"] == ""
    parts = [[entry]]
    CUT.run_chunks(tmp_path, parts, lambda _p: [row], log=lambda _m: None)
    assert any("came back empty: 1" in line for line in CUT.status(tmp_path, parts))


def _finished_cut(out: Path, monkeypatch: pytest.MonkeyPatch, n: int = 5) -> None:
    index = _index(n)
    CUT.atomic_write(out / CUT.INDEX_NAME, CUT.INDEX_FIELDS, index)
    args = argparse.Namespace(since="2021-12-31", chunk_size=2)
    CUT.check_manifest(out / CUT.MANIFEST_NAME, CUT.conditions(args, out / CUT.INDEX_NAME, "1:1"))
    CUT.run_chunks(out, CUT.slices(index, 2), _processor([]), log=lambda _m: None)


def test_read_cut_summaries_checks_the_cut_before_it_returns_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _finished_cut(tmp_path, monkeypatch)
    rows = CUT.read_cut_summaries(tmp_path)
    assert sorted(rows) == sorted(f"h{i}" for i in range(5))
    assert rows["h3"]["summary"] == "summary of h3"
    (tmp_path / "chunk_00003.csv").unlink()
    with pytest.raises(CUT.CutSummariesError, match="not finished"):
        CUT.read_cut_summaries(tmp_path)
    assert len(CUT.read_cut_summaries(tmp_path, allow_partial=True)) == 4
    manifest = json.loads((tmp_path / CUT.MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["summariser"] = "other/model@abc"
    (tmp_path / CUT.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CUT.CutSummariesError, match="the pipeline pins"):
        CUT.read_cut_summaries(tmp_path, allow_partial=True)
    with pytest.raises(CUT.CutSummariesError, match="not found"):
        CUT.read_cut_summaries(tmp_path / "nowhere")


def test_one_bad_body_costs_only_itself_and_is_recorded() -> None:
    def breaks_on_one(texts: list[str]) -> list[tuple[str, int]]:
        if any("poison" in x for x in texts):
            raise RuntimeError("CUDA out of memory " + "x" * 400)
        return [("ok", 1) for _ in texts]

    done = CUT.summarise_with_retry(breaks_on_one, ["a", "poison body", "b"])
    assert [d[0] for d in done] == ["ok", "", "ok"]  # the good ones survive the bad batch
    assert done[1][2].startswith("RuntimeError: CUDA out of memory") and len(done[1][2]) < 200

    def stops(_t: list[str]) -> list[tuple[str, int]]:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):  # a Ctrl-C is never swallowed as a body error
        CUT.summarise_with_retry(stops, ["a"])


def test_a_failed_body_is_written_marked_and_counted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = "Acme rose."
    process, _ = _processor_under_test(monkeypatch, "single", body)

    def always_raises(_texts: list[str]) -> list[tuple[str, int]]:
        raise RuntimeError("tokenizer error")

    monkeypatch.setattr(CUT, "load_summariser", lambda _d=None: always_raises)
    entry = {"body_hash": triage.body_hash(body), "article_ids": "1"}
    [row] = process([entry])
    assert (row["status"], row["summary"], row["error"]) == (
        CUT.FAILED,
        "",
        "RuntimeError: tokenizer error",
    )
    parts = [[entry]]
    CUT.run_chunks(tmp_path, parts, lambda _p: [row], log=lambda _m: None)  # the chunk is written
    assert any("failed: 1" in line for line in CUT.status(tmp_path, parts))


def test_a_chunk_with_a_missing_summary_and_no_reason_is_refused(tmp_path: Path) -> None:
    parts = CUT.slices(_index(2), 2)
    CUT.run_chunks(tmp_path, parts, _processor([]), log=lambda _m: None)
    rows = CUT.read_rows(tmp_path / "chunk_00001.csv")
    rows[1]["summary"] = ""  # the right hashes, but an empty summary that says it is fine
    CUT.atomic_write(tmp_path / "chunk_00001.csv", CUT.CHUNK_FIELDS, rows)
    with pytest.raises(SystemExit, match="not sound"):
        CUT.done_chunks(tmp_path, parts)
    rows[1]["status"] = CUT.EMPTY  # the same row, honestly marked
    CUT.atomic_write(tmp_path / "chunk_00001.csv", CUT.CHUNK_FIELDS, rows)
    assert CUT.done_chunks(tmp_path, parts) == {1}


def test_a_chunk_file_is_synced_before_its_name_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    monkeypatch.setattr(CUT.os, "fsync", synced.append)
    CUT.atomic_write(tmp_path / "chunk_00001.csv", CUT.CHUNK_FIELDS, [])
    assert len(synced) == 2  # the file, then its directory


def test_a_changed_environment_is_refused_unless_accepted(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    now = {"environment": "torch 2, transformers 4, cuda", "summariser": "m@r"}
    CUT.check_manifest(path, now, "1")
    cpu = {**now, "environment": "torch 2, transformers 4, cpu"}
    with pytest.raises(SystemExit, match="--accept-environment-change"):
        CUT.check_manifest(path, cpu, "1")
    with pytest.raises(SystemExit, match="other conditions"):  # the other flag does not help
        CUT.check_manifest(path, cpu, "1", accept_new_spans=True)
    CUT.check_manifest(path, cpu, "1", accept_environment_change=True, chunks_done=4)
    history = json.loads(path.read_text(encoding="utf-8"))["ner_spans_history"]
    assert history == [{"environment": "torch 2, transformers 4, cuda", "chunks_made_under_it": 4}]


DEVICES: list[str | None] = []


def _wired_main(
    monkeypatch: pytest.MonkeyPatch, out: Path, *flags: str, spans: str = "1:1"
) -> list[str]:
    """``main()`` with the database, the index and the processor replaced; returns the calls."""
    calls: list[str] = []
    monkeypatch.setattr(CUT.db, "connect_pipeline", object)

    def fake_index(_conn: object, path: Path, _since: str, **_kw: bool) -> list[dict[str, str]]:
        CUT.atomic_write(path, CUT.INDEX_FIELDS, _index(5))
        return _index(5)

    monkeypatch.setattr(CUT, "load_or_build_index", fake_index)
    monkeypatch.setattr(CUT, "entities_basis", lambda *_a: (spans, "7"))

    def make(
        _conn: object, device: str | None = None
    ) -> Callable[[Sequence[dict[str, str]]], list[dict[str, str]]]:
        calls.append("make_processor")
        DEVICES.append(device)
        return _processor(calls)

    monkeypatch.setattr(CUT, "make_processor", make)
    monkeypatch.setenv("SOURCE_DATABASE_URL", "/crawl/urls.db")  # the run asks for it up front
    monkeypatch.setattr(sys, "argv", ["c", "--out", str(out), "--chunk-size", "2", *flags])
    return calls


def test_main_makes_chunks_and_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _wired_main(monkeypatch, tmp_path, "--max-chunks", "2")
    CUT.main()
    assert sorted(p.name for p in tmp_path.glob("chunk_*")) == [
        "chunk_00001.csv",
        "chunk_00002.csv",
    ]
    assert not (tmp_path / CUT.LOCK_NAME).exists()
    assert "2 chunks made in this run" in capsys.readouterr().out
    assert calls.count("h0") == 1


def test_main_status_loads_no_model_and_takes_no_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _wired_main(monkeypatch, tmp_path, "--status")
    CUT.main()
    assert "no cut in" in capsys.readouterr().out  # nothing is built just to read a status
    assert not list(tmp_path.iterdir())
    CUT.atomic_write(tmp_path / CUT.INDEX_NAME, CUT.INDEX_FIELDS, _index(5))
    (tmp_path / CUT.LOCK_NAME).write_text("1", encoding="utf-8")  # another run is "in progress"
    monkeypatch.setattr(CUT, "_alive", lambda pid: True)
    CUT.main()  # --status is allowed while a run is in progress
    assert "0 of 3 chunks done" in capsys.readouterr().out
    assert calls == [] and not (tmp_path / CUT.MANIFEST_NAME).exists()  # read only


def test_main_passes_the_accept_flags_to_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired_main(monkeypatch, tmp_path, "--max-chunks", "1")
    CUT.main()
    _wired_main(monkeypatch, tmp_path, "--max-chunks", "1", spans="9:9")
    with pytest.raises(SystemExit, match="--accept-new-spans"):
        CUT.main()
    _wired_main(monkeypatch, tmp_path, "--max-chunks", "1", "--accept-new-spans", spans="9:9")
    CUT.main()
    manifest = json.loads((tmp_path / CUT.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["ner_spans"] == "9:9"
    assert manifest["ner_spans_history"][0]["chunks_made_under_it"] == 1


def test_a_second_start_cannot_touch_the_manifest_while_a_run_holds_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired_main(monkeypatch, tmp_path, "--max-chunks", "1")
    CUT.main()
    before = (tmp_path / CUT.MANIFEST_NAME).read_bytes()
    (tmp_path / CUT.LOCK_NAME).write_text("424242", encoding="utf-8")
    monkeypatch.setattr(CUT, "_alive", lambda pid: True)
    _wired_main(monkeypatch, tmp_path, "--accept-new-spans", spans="9:9")
    with pytest.raises(SystemExit, match="another run"):
        CUT.main()
    assert (tmp_path / CUT.MANIFEST_NAME).read_bytes() == before  # not rewritten, no history


def test_the_private_lock_file_of_a_dead_run_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "run.lock.555").write_text("555", encoding="utf-8")
    (tmp_path / "run.lock.556").write_text("556", encoding="utf-8")
    monkeypatch.setattr(CUT, "_alive", lambda pid: pid == 556)
    with CUT.locked(tmp_path):
        assert not (tmp_path / "run.lock.555").exists()  # its run is gone
        assert (tmp_path / "run.lock.556").exists()  # still alive: left alone


def _chunk_with_a_failed_row(tmp_path: Path) -> list[list[dict[str, str]]]:
    parts: list[list[dict[str, str]]] = CUT.slices(_index(4), 2)
    CUT.run_chunks(tmp_path, parts, _processor([]), log=lambda _m: None)
    rows = CUT.read_rows(tmp_path / "chunk_00001.csv")
    rows[1].update(status=CUT.FAILED, summary="", num_chunks="", error="RuntimeError: oom")
    CUT.atomic_write(tmp_path / "chunk_00001.csv", CUT.CHUNK_FIELDS, rows)
    return parts


def test_redo_failed_makes_only_the_failed_bodies_again_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    parts = _chunk_with_a_failed_row(tmp_path)
    again: list[str] = []
    messages: list[str] = []
    made = CUT.run_chunks(tmp_path, parts, _processor(again), redo_failed=True, log=messages.append)
    assert made == 1 and again == ["h1"]  # chunk 2 was fine; in chunk 1 only the failed body
    rows = CUT.read_rows(tmp_path / "chunk_00001.csv")
    assert [(r["status"], r["summary"]) for r in rows] == [
        ("summarised", "summary of h0"),
        ("summarised", "summary of h1"),
    ]
    assert not (tmp_path / "redo" / "chunk_00001.csv").exists()  # the new file replaced it
    assert any("1 bodies with no summary in 1 chunks" in m for m in messages)


def test_a_kill_after_the_old_chunk_was_set_aside_loses_nothing(tmp_path: Path) -> None:
    parts = _chunk_with_a_failed_row(tmp_path)

    def dies(_part: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        raise RuntimeError("killed")

    with pytest.raises(RuntimeError):
        CUT.run_chunks(tmp_path, parts, dies, redo_failed=True, log=lambda _m: None)
    assert (tmp_path / "redo" / "chunk_00001.csv").exists()  # the old rows are still there
    again: list[str] = []
    CUT.run_chunks(tmp_path, parts, _processor(again), log=lambda _m: None)  # no flag this time
    assert again == ["h1"]  # it still finds the old file and keeps the good row


def test_status_tells_what_to_run_when_bodies_have_no_summary(tmp_path: Path) -> None:
    parts = _chunk_with_a_failed_row(tmp_path)
    assert "--redo-failed" in CUT.status(tmp_path, parts)[0]


def test_the_reader_refuses_gaps_and_damage_unless_told(tmp_path: Path) -> None:
    index = _index(4)
    CUT.atomic_write(tmp_path / CUT.INDEX_NAME, CUT.INDEX_FIELDS, index)
    args = argparse.Namespace(since="2021-12-31", chunk_size=2)
    CUT.check_manifest(
        tmp_path / CUT.MANIFEST_NAME, CUT.conditions(args, tmp_path / CUT.INDEX_NAME, "1:1")
    )
    parts = _chunk_with_a_failed_row(tmp_path)
    with pytest.raises(CUT.CutSummariesError, match="redo-failed"):
        CUT.read_cut_summaries(tmp_path)
    rows = CUT.read_cut_summaries(tmp_path, allow_gaps=True)
    assert rows["h1"]["status"] == CUT.FAILED and rows["h0"]["status"] == CUT.SUMMARISED
    damaged = CUT.read_rows(tmp_path / "chunk_00002.csv")
    damaged[0]["summary"] = ""  # right hashes, a missing summary that claims to be fine
    CUT.atomic_write(tmp_path / "chunk_00002.csv", CUT.CHUNK_FIELDS, damaged)
    with pytest.raises(CUT.CutSummariesError, match="not sound"):
        CUT.read_cut_summaries(tmp_path, allow_gaps=True)
    assert len(parts) == 2


def test_the_cut_summarises_the_scopes_the_sentiment_draw_expects() -> None:
    draw = _script("draw_sentiment_gold_2026_11_05")
    gold_of_cut = {triage.RULE_SCOPE_TO_GOLD[s] for s in CUT.SUMMARY_RULE_SCOPES}
    # the rules never predict investing_general (D25 summarises it): the one known difference
    assert gold_of_cut == set(draw.SUMMARY_SCOPES) - {"investing_general"}
    summarised_by_rules = {
        s for s, g in triage.RULE_SCOPE_TO_GOLD.items() if g in draw.SUMMARY_SCOPES
    }
    assert summarised_by_rules == set(CUT.SUMMARY_RULE_SCOPES)  # no scope left out or added


def test_a_killed_manifest_write_leaves_the_old_manifest_and_a_truncated_one_is_explained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    now = {"since": "2021-12-31", "summariser": "m@r1", "chunk_size": 250}
    CUT.check_manifest(path, now, "1")
    before = path.read_text(encoding="utf-8")

    def dies(_tmp: Path, _path: Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(CUT, "_publish", dies)
    with pytest.raises(KeyboardInterrupt):
        CUT.check_manifest(
            path, now, "2"
        )  # NER wrote: the manifest is refreshed, and the kill hits
    assert path.read_text(encoding="utf-8") == before  # never half a file
    monkeypatch.undo()
    path.write_text(before[:20], encoding="utf-8")  # a manifest truncated by an older version
    with pytest.raises(SystemExit, match="cannot be read"):
        CUT.check_manifest(path, now, "2")
    with pytest.raises(CUT.CutSummariesError, match="cannot be read"):
        CUT.read_cut_summaries(tmp_path)


def test_status_slices_the_index_as_the_job_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _wired_main(monkeypatch, tmp_path, "--max-chunks", "1")  # a job with --chunk-size 2
    CUT.main()
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["c", "--out", str(tmp_path), "--status"])  # default 250
    CUT.main()
    assert "1 of 3 chunks done (2 of 5 bodies)" in capsys.readouterr().out


def test_a_redo_file_that_is_not_the_slice_stops_the_run(tmp_path: Path) -> None:
    parts = _chunk_with_a_failed_row(tmp_path)
    other = CUT.read_rows(tmp_path / "chunk_00002.csv")
    (tmp_path / "chunk_00001.csv").unlink()  # the owner deleted a damaged chunk ...
    (tmp_path / "redo").mkdir()  # ... and a leftover of another slice waits in redo/
    CUT.atomic_write(tmp_path / "redo" / "chunk_00001.csv", CUT.CHUNK_FIELDS, other)
    with pytest.raises(SystemExit, match=r"redo/chunk_00001\.csv does not hold the bodies"):
        CUT.run_chunks(tmp_path, parts, _processor([]), log=lambda _m: None)


def test_an_index_with_no_manifest_is_listed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / CUT.INDEX_NAME
    CUT.atomic_write(path, CUT.INDEX_FIELDS, _index(2))  # listed for some other --since

    class Conn:
        def execute(self, *_a: object) -> Conn:
            return self

        def fetchone(self) -> list[int]:
            return [0]

    monkeypatch.setattr(CUT, "build_index", lambda _pairs: _index(3))
    kept = CUT.load_or_build_index(Conn(), path, "2022-01-01", manifest_exists=True)  # type: ignore[arg-type]
    assert len(kept) == 2
    fresh = CUT.load_or_build_index(Conn(), path, "2022-01-01", manifest_exists=False)  # type: ignore[arg-type]
    assert len(fresh) == 3 and len(CUT.read_rows(path)) == 3


def test_chunks_without_their_manifest_stop_the_run_and_keep_the_index(tmp_path: Path) -> None:
    path = tmp_path / CUT.INDEX_NAME
    CUT.atomic_write(path, CUT.INDEX_FIELDS, _index(2))
    before = path.read_bytes()
    with pytest.raises(SystemExit, match="Nothing was changed"):
        CUT.load_or_build_index(
            object(),  # type: ignore[arg-type]
            path,
            "2022-01-01",
            manifest_exists=False,
            chunks_exist=True,
        )
    assert path.read_bytes() == before  # the index the chunks were made from is intact


@pytest.mark.parametrize(
    ("where", "lost"),
    [("", "manifest"), ("redo", "manifest"), ("", "index")],
)
def test_main_stops_when_chunks_exist_and_the_manifest_or_the_index_is_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, lost: str
) -> None:
    """Through ``main()`` itself: the real ``load_or_build_index`` gets the arguments it needs."""
    parts = CUT.slices(_index(4), 2)
    CUT.atomic_write(tmp_path / CUT.INDEX_NAME, CUT.INDEX_FIELDS, _index(4))
    CUT.run_chunks(tmp_path, parts, _processor([]), max_chunks=1, log=lambda _m: None)
    CUT.atomic_text(tmp_path / CUT.MANIFEST_NAME, "{}")
    if where:
        (tmp_path / where).mkdir()
        (tmp_path / "chunk_00001.csv").rename(tmp_path / where / "chunk_00001.csv")
    (tmp_path / (CUT.MANIFEST_NAME if lost == "manifest" else CUT.INDEX_NAME)).unlink()
    before = sorted(p.name for p in tmp_path.rglob("*"))
    monkeypatch.setattr(CUT.db, "connect_pipeline", object)
    monkeypatch.setenv("SOURCE_DATABASE_URL", "/crawl/urls.db")
    monkeypatch.setattr(sys, "argv", ["c", "--out", str(tmp_path), "--chunk-size", "2"])
    with pytest.raises(SystemExit, match="Nothing was changed"):
        CUT.main()
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


class _FakeBar:
    """Stands in for ``tqdm`` and records what the bar was told."""

    made: ClassVar[list[_FakeBar]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.updates: list[int] = []
        self.closed = False
        _FakeBar.made.append(self)

    def update(self, n: int) -> None:
        self.updates.append(n)

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def write(_message: str) -> None:
        return None


def test_the_bar_counts_bodies_starts_after_the_chunks_on_disk_and_shows_the_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CUT, "tqdm", _FakeBar)
    _FakeBar.made.clear()
    parts = CUT.slices(_index(10), 3)  # chunks of 3, 3, 3, 1 bodies
    CUT.run_chunks(
        tmp_path, parts, _processor([]), max_chunks=1, log=lambda _m: None, device_label="cuda"
    )
    CUT.run_chunks(tmp_path, parts, _processor([]), log=lambda _m: None, device_label="cuda")
    first, second = _FakeBar.made
    assert first.kwargs["total"] == 10 and first.kwargs["initial"] == 0
    assert first.kwargs["postfix"] == {"device": "cuda"}
    assert first.updates == [3] and first.closed
    assert second.kwargs["initial"] == 3  # the chunk already on disk counts as done
    assert second.updates == [3, 3, 1] and second.closed


def test_the_bar_is_closed_when_a_chunk_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CUT, "tqdm", _FakeBar)
    _FakeBar.made.clear()

    def dies(_part: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        CUT.run_chunks(tmp_path, CUT.slices(_index(3), 3), dies, log=lambda _m: None)
    assert _FakeBar.made[0].closed


def _fake_torch(monkeypatch: pytest.MonkeyPatch, *, cuda: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _i=0: "Fake RTX 9000")


def test_the_banner_says_cuda_or_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_torch(monkeypatch, cuda=True)
    has_cuda, line = CUT.device_banner()
    assert has_cuda and line.startswith("device: CUDA (Fake RTX 9000")
    _fake_torch(monkeypatch, cuda=False)
    has_cuda, line = CUT.device_banner()
    assert not has_cuda and line.startswith("device: CPU, no CUDA")


def test_main_prints_the_device_and_require_cuda_stops_a_cpu_run_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_torch(monkeypatch, cuda=False)
    out = tmp_path / "cut"  # not made yet
    calls = _wired_main(monkeypatch, out, "--require-cuda")
    with pytest.raises(SystemExit, match="CUDA is not available"):
        CUT.main()
    assert "device: CPU, no CUDA" in capsys.readouterr().out
    assert calls == [] and not out.exists()  # not even the folder
    _wired_main(monkeypatch, out, "--max-chunks", "1")  # without the flag a CPU run goes on
    DEVICES.clear()
    CUT.main()
    assert "device: CPU, no CUDA" in capsys.readouterr().out
    assert DEVICES == ["cpu"]  # the model is told the device the banner printed
    _fake_torch(monkeypatch, cuda=True)
    _wired_main(monkeypatch, tmp_path / "gpu", "--max-chunks", "1")
    DEVICES.clear()
    CUT.main()
    assert DEVICES == ["cuda"]


def test_the_real_bar_is_off_without_a_terminal_and_the_default_log_does_not_need_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    CUT.run_chunks(tmp_path, CUT.slices(_index(4), 2), _processor([]), device_label="cuda")
    captured = capsys.readouterr()
    assert "summaries" not in captured.err  # stderr is not a tty here: no bar
    assert "chunk 2/2 written" in captured.out  # the default log (tqdm.write) still prints


def test_the_manifest_records_the_device_main_decided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_torch(monkeypatch, cuda=True)  # torch says cuda, the decision says cpu: the decision wins
    args = argparse.Namespace(since="2021-12-31", chunk_size=2)
    index = tmp_path / CUT.INDEX_NAME
    CUT.atomic_write(index, CUT.INDEX_FIELDS, _index(2))
    assert CUT.conditions(args, index, "1:1", "cpu")["environment"].endswith(", cpu")
    assert CUT.conditions(args, index, "1:1")["environment"].endswith(", cuda")


def test_main_loads_the_env_file_and_stops_early_without_a_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("SOURCE_DATABASE_URL", "DATABASE_URL"):  # recorded, so the test leaves no trace
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    out = tmp_path / "cut"
    _wired_main(monkeypatch, out, "--max-chunks", "1")
    monkeypatch.delenv("SOURCE_DATABASE_URL")  # _wired_main set it: this test starts without
    _fake_torch(monkeypatch, cuda=False)
    monkeypatch.setattr(CUT.db.env, "load_env_file", REAL_LOAD_ENV_FILE)
    env_file = tmp_path / ".env"
    monkeypatch.setattr(CUT.db.env, "DEFAULT_ENV_FILE", env_file)
    with pytest.raises(SystemExit, match=r"SOURCE_DATABASE_URL is not set.*does not exist"):
        CUT.main()  # no .env, no variable: one line, and no folder
    assert not out.exists()
    env_file.write_text("DATABASE_URL=/x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"exists but has no SOURCE_DATABASE_URL"):
        CUT.main()
    env_file.write_text("SOURCE_DATABASE_URL=/crawl/urls.db\n", encoding="utf-8")
    DEVICES.clear()
    CUT.main()  # main itself read the file
    assert os.environ["SOURCE_DATABASE_URL"] == "/crawl/urls.db" and DEVICES == ["cpu"]


def test_the_env_file_loader_fills_what_is_unset_and_the_shell_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("SOURCE_DATABASE_URL", "DATABASE_URL"):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    env = tmp_path / ".env"
    assert REAL_LOAD_ENV_FILE(env) is None  # no file: nothing happens
    env.write_text("SOURCE_DATABASE_URL=D:/urls.db\nDATABASE_URL=D:/nlp.db\n", encoding="utf-8")
    monkeypatch.setenv("DATABASE_URL", "from-the-shell")
    assert REAL_LOAD_ENV_FILE(env) == env
    assert os.environ["SOURCE_DATABASE_URL"] == "D:/urls.db"
    assert os.environ["DATABASE_URL"] == "from-the-shell"
