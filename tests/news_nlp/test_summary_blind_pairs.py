"""The choice of the 24 pairs of the blind re-scoring (T-206): reproducible, filtered as the doc says, and the committed file is its result."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "blind", ROOT / "scripts" / "summary_blind_pairs_2026_11_15.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample() -> list[dict[str, Any]]:
    rows = []
    for i in range(1, 91):
        rows.append(
            {
                "article_id": i,
                "tier": 1 + i % 3,
                "language": "es" if i % 7 == 0 else "en",
                "body": "x" * (500 + 150 * i),  # the longer ones are over 9,000 characters
            }
        )
    return rows


def test_the_choice_is_reproducible_filtered_and_balanced() -> None:
    mod = _mod()
    sample = _sample()
    a = mod.choose(sample)
    assert a == mod.choose(sample)
    assert len(a) == 24 and sorted(c["item"] for c in a) == list(range(1, 25))
    by_id = {r["article_id"]: r for r in sample}
    for tier in (1, 2, 3):
        assert sum(1 for c in a if c["tier"] == tier) == 8
    assert all(
        by_id[c["article_id"]]["language"] == "en" and len(by_id[c["article_id"]]["body"]) < 9000
        for c in a
    )  # type: ignore[arg-type]
    assert {c["x_arm"] for c in a} == {"B", "D"}
    assert mod.choose(sample, seed=8) != a


def test_the_committed_blind_file_is_the_choice_of_the_script_when_the_scratch_is_there() -> None:
    mod = _mod()
    if not (mod.RESULTS / "sample.json").exists():
        pytest.skip("the git-ignored scratch of the experiment is not here")
    assert mod.check(mod.RESULTS) == 0


def test_the_committed_blind_file_has_24_items_8_per_tier() -> None:
    with (ROOT / "docs" / "summarization_results_2026_11_15" / "blind_24.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 24 and [int(r["item"]) for r in rows] == list(range(1, 25))
    assert {t: sum(1 for r in rows if r["tier"] == t) for t in ("1", "2", "3")} == {
        "1": 8,
        "2": 8,
        "3": 8,
    }
