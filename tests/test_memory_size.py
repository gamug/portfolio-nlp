"""The planning files stay small enough to read (constitution AI behavior #15, `.specify/memory/INDEX.md`).

The limits are a guard, not a target: when one fails, move the closed or long-form text to `CHANGELOG.md`
or `.specify/memory/archive/` (verbatim, IDs unchanged) instead of raising the limit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MEMORY = Path(__file__).resolve().parents[1] / ".specify" / "memory"

#: Characters (the files are read by a model, so the unit that matters is tokens, about a quarter of it).
LIMITS = {"TASKS.md": 70_000, "PLAN.md": 80_000, "SPEC.md": 80_000}
#: One open task entry: its line plus the indented lines under it.
MAX_TASK_ENTRY = 4_000

_TASK = re.compile(r"^(\s*)- \[[ x]\] ")


def _text(name: str) -> str:
    return (MEMORY / name).read_text(encoding="utf-8").replace("\r\n", "\n")


@pytest.mark.parametrize(("name", "limit"), sorted(LIMITS.items()))
def test_planning_file_is_under_its_size_limit(name: str, limit: int) -> None:
    size = len(_text(name))
    assert size <= limit, (
        f"{name} is {size:,} characters (limit {limit:,}): move closed or long-form text to "
        "CHANGELOG.md or .specify/memory/archive/ (see INDEX.md)"
    )


def test_no_task_entry_is_longer_than_the_limit() -> None:
    lines = _text("TASKS.md").split("\n")
    too_long: list[str] = []
    for i, line in enumerate(lines):
        match = _TASK.match(line)
        if not match:
            continue
        end = i + 1
        while (
            end < len(lines)
            and not _TASK.match(lines[end])
            and not lines[end].startswith("#")
            and lines[end].startswith(" ")
        ):
            end += 1
        size = len("\n".join(lines[i:end]))
        if size > MAX_TASK_ENTRY:
            ident = re.search(r"\*\*(T-\w+)\*\*", line)
            too_long.append(f"{ident.group(1) if ident else line[:40]} ({size:,})")
    assert not too_long, (
        f"task entries over {MAX_TASK_ENTRY:,} characters: {', '.join(too_long)}; keep a status, the "
        "decision, what is open and a pointer, and move the history to .specify/memory/archive/"
    )


def test_index_lists_every_archive_file() -> None:
    index = (MEMORY / "INDEX.md").read_text(encoding="utf-8")
    for path in sorted((MEMORY / "archive").glob("*.md")):
        assert path.name in index, f"{path.name} is not described in INDEX.md"
