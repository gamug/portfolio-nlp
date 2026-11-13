"""The table of candidate thresholds for the main company (T-169): the ratios, the inclusive threshold, the empty classes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "relevance_threshold_candidates",
    ROOT / "scripts" / "relevance_threshold_candidates_2026_11_13.py",
)
assert spec and spec.loader
script = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = script
spec.loader.exec_module(script)


def test_each_threshold_counts_the_rows_at_or_over_it() -> None:
    scores = [0.9, 0.5, 0.5, 0.1]
    relevant = [True, True, False, False]
    (row,) = script.candidates(scores, relevant, (0.5,))
    assert row["relevant_kept"] == 1.0  # 0.9 and 0.5 are main: the threshold itself counts
    assert row["not_relevant_main"] == 0.5
    assert row["precision"] == 2 / 3
    assert row["under"] == 0.25


def test_a_higher_threshold_keeps_fewer_relevant_rows_and_a_class_with_no_rows_has_no_ratio() -> (
    None
):
    table = script.candidates([0.2, 0.6, 0.95], [True, True, True], (0.1, 0.5, 0.99))
    assert [r["relevant_kept"] for r in table] == [1.0, 2 / 3, 0.0]
    assert all(
        r["not_relevant_main"] is None for r in table
    )  # no not-relevant row: nothing to divide by
    assert table[2]["precision"] is None  # nothing is called main at 0.99


def test_the_scores_are_the_mean_of_the_repeats_and_a_neutral_mark_is_not_relevant(
    tmp_path: Path,
) -> None:
    oof = tmp_path / "oof.json"
    oof.write_text(json.dumps({"keys": ["a#summary", "b#summary"], "R": [[0.2, 0.8], [0.4, 1.0]]}))
    gold = tmp_path / "gold.csv"
    gold.write_text("key,sentiment\na#summary,neutral\nb#summary,positive\n")
    scores, relevant = script.load(oof, (gold,))
    assert scores == [0.30000000000000004, 0.9] or [round(s, 6) for s in scores] == [0.3, 0.9]
    assert relevant == [False, True]
