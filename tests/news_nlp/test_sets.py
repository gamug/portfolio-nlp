"""The named sets of tests are complete: every test file is in a topic of ``tests/sets.py``."""

from __future__ import annotations

from pathlib import Path

from tests.sets import HEAVY, TOPICS


def test_every_test_file_is_in_a_topic_and_every_name_exists() -> None:
    files = {p.name for p in Path(__file__).parent.glob("test_*.py")} - {"test_sets.py"}
    placed = set().union(*TOPICS.values())
    assert files - placed == set(), (
        f"add these test files to a topic in tests/sets.py: {files - placed}"
    )
    assert placed - files == set(), f"these names in tests/sets.py are not files: {placed - files}"
    assert HEAVY - files == set(), f"HEAVY names that are not files: {HEAVY - files}"
