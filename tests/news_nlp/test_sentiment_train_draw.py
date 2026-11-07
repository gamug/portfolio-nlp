"""The draw of the training summaries for sentiment (T-148c): who is never drawn, balance by class."""

from __future__ import annotations

import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = _script("draw_sentiment_train_2026_11_07")


def _row(n: int, summary: str = "A summary long enough to be read by the owner.") -> dict[str, str]:
    return {
        "body_hash": f"h{n:04d}",
        "article_ids": str(n),
        "rule_scope": "single",
        "status": "summarised",
        "summary": summary,
    }


def test_candidates_leave_out_the_gold_and_held_out_bodies_and_unreadable_lengths() -> None:
    rows = [_row(i) for i in range(10)] + [_row(10, "short"), _row(11, "x" * 5000)]
    got = TRAIN.candidates(rows, {"h0003", "h0004"}, pool=100, seed=1)
    hashes = {r["body_hash"] for r in got}
    assert not hashes & {"h0003", "h0004", "h0010", "h0011"}
    assert len(hashes) == 8


def test_candidates_are_seeded_and_capped() -> None:
    rows = [_row(i) for i in range(50)]
    first = TRAIN.candidates(rows, set(), pool=20, seed=7)
    assert [r["body_hash"] for r in first] == [
        r["body_hash"] for r in TRAIN.candidates(rows, set(), pool=20, seed=7)
    ]
    assert len(first) == 20
    assert [r["body_hash"] for r in first] != [
        r["body_hash"] for r in TRAIN.candidates(rows, set(), pool=20, seed=8)
    ]


def test_the_draw_takes_the_same_number_of_each_predicted_class_or_all_of_a_short_one() -> None:
    scored = (
        [{**_row(i), "text": "t", "predicted": "neutral"} for i in range(30)]
        + [{**_row(100 + i), "text": "t", "predicted": "positive"} for i in range(30)]
        + [{**_row(200 + i), "text": "t", "predicted": "negative"} for i in range(4)]
    )
    chosen, population = TRAIN.draw(scored, per_class=10, seed=3)
    assert Counter(t["predicted"] for t in chosen) == {"neutral": 10, "positive": 10, "negative": 4}
    assert population == {"neutral": 30, "positive": 30, "negative": 4}
    assert {t["population"] for t in chosen if t["predicted"] == "negative"} == {"4"}
    assert all(t["text_hash"] for t in chosen)
    assert len({t["body_hash"] for t in chosen}) == len(chosen)


def test_the_run_stops_on_a_cut_that_is_not_finished(tmp_path: Path) -> None:
    for number in (1, 2, 4):  # chunk 3 is missing
        with (tmp_path / f"chunk_{number:05d}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(_row(1)))
            writer.writeheader()
            writer.writerow(_row(number))
    with pytest.raises(SystemExit, match="finish the cut first"):
        TRAIN.cut_rows(tmp_path)
    (tmp_path / "chunk_00003.csv").write_text(
        (tmp_path / "chunk_00002.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert len(TRAIN.cut_rows(tmp_path)) == 4


def test_only_summarised_rows_with_a_summary_are_read(tmp_path: Path) -> None:
    rows = [_row(1), {**_row(2), "status": "by_scope", "summary": ""}, {**_row(3), "summary": " "}]
    with (tmp_path / "chunk_00001.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert [r["body_hash"] for r in TRAIN.cut_rows(tmp_path)] == ["h0001"]
