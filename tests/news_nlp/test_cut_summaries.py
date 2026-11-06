"""The summaries of the backtest cut can be stopped and resumed (T-148a, scripts/cut_summaries)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
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
    assert json.loads(path.read_text(encoding="utf-8")) == now  # the old manifest is untouched


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
    first = CUT.conditions(args, index)
    CUT.atomic_write(index, CUT.INDEX_FIELDS, _index(4))
    assert CUT.conditions(args, index)["index"] != first["index"]
    assert first["summariser"].startswith("sshleifer/distilbart-cnn-12-6@")
