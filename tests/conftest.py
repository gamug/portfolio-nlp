"""Test sets (``tests/sets.py``): ``pytest --set fast`` / ``--set wi18`` / ``--set eval`` ...

pytest imports every test module while it collects, before ``-m`` can deselect anything, so a
heavy file (torch, transformers, mlflow, the pipeline: 50 to 400 s on this disk) costs its import
even when it is deselected. ``--set`` therefore skips the files outside the chosen sets before they
are imported. ``--set`` can be repeated; a file is collected when it is in any chosen set. The names
are ``fast``, ``heavy`` and the topics (``wi18``, ``db``, ``eval``, ``stages``, ``api``). Without
``--set`` everything runs, as CI does. The marks are applied too, so ``-m`` still works.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.sets import HEAVY, TOPICS


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--set",
        action="append",
        default=[],
        choices=["fast", "heavy", *TOPICS],
        help="run only the test files of this set (repeatable): fast, heavy or a topic",
    )


def sets_of(name: str) -> set[str]:
    """The sets a test file belongs to."""
    return {t for t, files in TOPICS.items() if name in files} | {
        "heavy" if name in HEAVY else "fast"
    }


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    chosen = set(config.getoption("--set") or [])
    if (
        not chosen
        or collection_path.suffix != ".py"
        or not collection_path.name.startswith("test_")
    ):
        return None
    return not (chosen & sets_of(collection_path.name))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        for name in sets_of(item.path.name):
            item.add_marker(getattr(pytest.mark, name))
