"""The owner's answers on the two fresh sector draws (2026-11-19): the tracked marks, the scorer's figures, and the figures the
docs and the pipeline reports quote. A change to any of them has to change this test too."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKS = ROOT / "gold" / "sector_fresh_marks_2026_11_19.csv"
SCORER = ROOT / "scripts" / "score_sector_fresh_2026_11_19.py"
DOCS = [
    ROOT / "docs" / "sector_training.md",
    ROOT / "pipeline_metrics_report.md",
    ROOT / "pipeline_metrics_report_es.md",
]


def _scored() -> str:
    done = subprocess.run(  # noqa: S603 -- runs this repo's own scorer with this interpreter
        [sys.executable, str(SCORER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return done.stdout


def _marks() -> list[dict[str, str]]:
    with MARKS.open(encoding="utf-8-sig", newline="") as fh:
        rows: list[dict[str, str]] = list(csv.DictReader(fh))
    return rows


def test_the_tracked_marks_are_the_357_answers_of_the_two_draws() -> None:
    rows = _marks()
    assert len(rows) == 357
    assert sum(r["control"] == "yes" for r in rows) == 20
    # 57 of 2026-11-13 (40 assigned, 5 controls, 12 sent out) and 300 of 2026-11-18 (165, 15, 120)
    assert sum(r["status"] == "assigned" and r["control"] == "no" for r in rows) == 205
    assert sum(r["status"] == "out_of_scope" for r in rows) == 132


def test_the_scorer_prints_the_figures_the_docs_quote() -> None:
    out = _scored()
    for line in (
        "assigned, all:                      141/165 = 85.5%",
        "assigned, without those:            141/144 = 97.9%",
        "controls answered yes:              0/15 = 0.0%",
        "sent out, the owner agrees:         95/120 = 79.2%",
        "assigned, all:                      35/40 = 87.5%",
        "assigned, without those:            35/38 = 92.1%",
        "controls answered yes:              0/5 = 0.0%",
        "sent out, the owner agrees:         6/12 = 50.0%",
    ):
        assert line in out, line
    # the one-vs-rest table counts the 28 sent-out rows with no sector named as unknown, and as wrong in the precision
    assert "truth unknown for 28 sent-out rows" in out
    assert (
        "none no sector | 132 | 101 | 101/132 = 76.5% [68.6, 82.9] | 28 | 124 | 101 | 81.5%" in out
    )


def test_the_docs_and_reports_quote_the_combined_figures() -> None:
    combined = ("176/205 = 85.9%", "176/182 = 96.7%", "101/132 = 76.5%")
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for pattern in combined:
            assert pattern in text, f"{doc.name}: {pattern}"
    # the stricter reading (rows 20 and 46 as `no`) is in the doc; the reports give it as a percentage
    assert "99/132 = 75.0%" in DOCS[0].read_text(encoding="utf-8")
    for report in DOCS[1:]:
        assert "75.0%" in report.read_text(encoding="utf-8"), report.name
