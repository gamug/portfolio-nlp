"""The planning files stay small enough to read (constitution AI behavior #15, `.specify/memory/INDEX.md`).

The limits are a guard, not a target: when one fails, move the closed or long-form text to `CHANGELOG.md`
or `.specify/memory/archive/` (verbatim, IDs unchanged) instead of raising the limit.
"""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

import pytest

MEMORY = Path(__file__).resolve().parents[1] / ".specify" / "memory"

#: Characters (the files are read by a model, so the unit that matters is tokens, about a quarter of it).
LIMITS = {"TASKS.md": 52_000, "PLAN.md": 73_000, "SPEC.md": 72_000}
#: One `## Work item N` section of PLAN.md (Work item 18 is the largest, 53 KB on 2026-11-08).
MAX_PLAN_SECTION = 58_000
#: One line of PLAN.md or TASKS.md (the longest on 2026-11-08 is 4.4 KB).
MAX_LINE = 5_000
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


def test_no_plan_section_or_line_is_over_its_limit() -> None:
    plan = _text("PLAN.md").split("\n")
    heads = [i for i, line in enumerate(plan) if line.startswith("## ")] + [len(plan)]
    big = [
        f"{plan[a][:40]} ({len(chr(10).join(plan[a:b])):,})"
        for a, b in pairwise(heads)
        if len(chr(10).join(plan[a:b])) > MAX_PLAN_SECTION
    ]
    assert not big, f"PLAN.md sections over {MAX_PLAN_SECTION:,} characters: {big}"
    for name in ("PLAN.md", "TASKS.md"):
        long = [i + 1 for i, line in enumerate(_text(name).split("\n")) if len(line) > MAX_LINE]
        assert not long, f"{name} lines over {MAX_LINE:,} characters: {long[:5]}"


def _task_id(line: str) -> str:
    found = re.search(r"\*\*(T-\w+)\*\*", line)
    return found.group(1) if found else line[:40]


def test_tasks_holds_no_checked_task() -> None:
    checked = [
        _task_id(line) for line in _text("TASKS.md").split("\n") if re.match(r"^\s*- \[x\] ", line)
    ]
    assert not checked, (
        f"checked tasks in TASKS.md: {', '.join(checked)}; a done or superseded task moves verbatim "
        'to CHANGELOG.md, section "Tasks closed while their work item is still open" '
        "(constitution AI #15)"
    )


def test_index_lists_every_archive_file() -> None:
    index = (MEMORY / "INDEX.md").read_text(encoding="utf-8")
    for path in sorted((MEMORY / "archive").glob("*.md")):
        assert path.name in index, f"{path.name} is not described in INDEX.md"
