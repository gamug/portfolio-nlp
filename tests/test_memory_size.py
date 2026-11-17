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

#: A task is recognised by the shape of its line (`- [ ]` / `- [x]` / `- [X]`), not by where it is: a
#: checkbox line inside a fenced block or a table cell would count as one.
_TASK = re.compile(r"^(\s*)- \[[ xX]\] ")


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
        _task_id(line)
        for line in _text("TASKS.md").split("\n")
        if re.match(r"^\s*- \[[xX]\] ", line)
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


def test_no_task_id_is_used_twice_in_tasks() -> None:
    """Two entries with one ID (a merge of two branches that each took the next number) make a reference ambiguous."""
    ids = re.findall(r"^\s*- \[[ xX]\] \*\*(T-[0-9a-z]+)\*\*", _text("TASKS.md"), re.MULTILINE)
    twice = sorted({i for i in ids if ids.count(i) > 1})
    assert twice == [], f"task IDs used by more than one entry in TASKS.md: {twice}"


#: IDs that are in both files on purpose: a task checked off in CHANGELOG.md as a whole whose remaining part was restored to
#: TASKS.md (its CHANGELOG line says so).
PARTLY_CLOSED = frozenset({"T-148c2c"})
_ID_SHAPE = r"^\s*- \[{box}\] \*\*(T-[0-9a-z]+)\*\*"


def _ids_in_both(tasks_text: str, changelog_text: str) -> list[str]:
    """The IDs of an entry of TASKS.md (open) that is also a **checked** entry of CHANGELOG.md (closed). An unchecked line of
    CHANGELOG.md is a moved work-item section, not a closed task."""
    open_ids = set(re.findall(_ID_SHAPE.format(box="[ xX]"), tasks_text, re.MULTILINE))
    closed_ids = set(re.findall(_ID_SHAPE.format(box="[xX]"), changelog_text, re.MULTILINE))
    return sorted((open_ids & closed_ids) - PARTLY_CLOSED)


def test_no_task_id_is_both_open_and_closed() -> None:
    """The same defect one file over: a PR that closes its task into CHANGELOG.md while another entry of TASKS.md has that ID
    leaves one number for two tasks and no test failing (T-219 in #206 and #207)."""
    both = _ids_in_both(_text("TASKS.md"), _text("CHANGELOG.md"))
    assert both == [], f"task IDs in both TASKS.md and CHANGELOG.md: {both}"


def test_the_id_check_flags_a_task_open_in_one_file_and_checked_in_the_other() -> None:
    tasks = "- [ ] **T-219** *(owner; category)* the measure\n- [ ] **T-220** next\n"
    changelog = "- [x] **T-219** *(done)* the API\n- [x] **T-148c2c** *(closed as a whole)* x\n"
    assert _ids_in_both(tasks, changelog) == ["T-219"]  # T-220 is in one file only


def test_the_id_check_lets_the_partly_closed_task_and_an_unchecked_line_through() -> None:
    tasks = "- [ ] **T-148c2c** *(restored)* the rest\n- [ ] **T-185** run step 3\n"
    changelog = (
        "- [x] **T-148c2c** *(closed as a whole)* x\n  - [ ] **T-185** a moved work-item section\n"
    )
    assert _ids_in_both(tasks, changelog) == []
