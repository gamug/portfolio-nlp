"""The before/after check of the matcher (PR #155, PLAN D38): a revision read from git, and the
self-check of a universe against its own names."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from news_nlp.triage import Company, CompanyResolver

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = _script("matcher_before_after_2026_11_09")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 -- a fixed git command in a temporary repository
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_a_revision_of_triage_is_read_from_git_and_not_imported_over_the_working_one(
    tmp_path: Path,
) -> None:
    (tmp_path / "src" / "news_nlp").mkdir(parents=True)
    triage_file = tmp_path / "src" / "news_nlp" / "triage.py"
    triage_file.write_text("VERSION = 'before'\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "before")
    before = _git(tmp_path, "rev-parse", "HEAD")
    triage_file.write_text("VERSION = 'after'\n", encoding="utf-8")
    _git(tmp_path, "commit", "-aq", "-m", "after")
    assert CHECK.triage_at(before, tmp_path).VERSION == "before"
    assert CHECK.triage_at("HEAD", tmp_path).VERSION == "after"
    with pytest.raises(subprocess.CalledProcessError):
        CHECK.triage_at("no-such-revision", tmp_path)


def test_the_self_check_lists_the_company_a_stop_word_erased() -> None:
    universe = [
        Company("CASY", "Casey's General Stores, Inc.", "Staples", "Food Retail"),
        Company("AAPL", "Apple Inc.", "Technology", "Hardware"),
        Company("FRT", "Federal Realty Investment Trust", "Real Estate", "REITs"),
    ]
    assert CHECK.self_check(CompanyResolver(universe), universe) == []
