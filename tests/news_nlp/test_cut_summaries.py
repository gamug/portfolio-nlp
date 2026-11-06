"""The summaries of the backtest cut can be stopped and resumed (T-148a, scripts/cut_summaries)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType

import pytest

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
                "summary": f"summary of {r['body_hash']}",
                "num_chunks": "1",
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

    def load() -> Callable[[list[str]], list[tuple[str, int]]]:
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
        assert (tmp_path / CUT.LOCK_NAME).read_text(encoding="utf-8").isdigit()
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

    assert CUT.summarise_with_retry(flaky, ["a", "b bad"]) == [("ok", 1), ("ok", 1)]
    assert calls == [["a", "b bad"], ["b bad"]]
    process, _ = _processor_under_test(monkeypatch, "single", "Acme rose.")
    monkeypatch.setattr(CUT, "load_summariser", lambda: lambda texts: [("", 0)] * len(texts))
    process, _ = _processor_under_test(monkeypatch, "single", "Acme rose.")
    monkeypatch.setattr(CUT, "load_summariser", lambda: lambda texts: [("", 0)] * len(texts))
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
