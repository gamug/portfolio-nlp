"""The before/after check of the matcher (PR #155, PLAN D38): a revision read from git, and the
self-check of a universe against its own names."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from news_nlp import triage
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


def test_a_ticker_form_in_a_disclosure_or_a_list_is_told_from_one_in_a_sentence() -> None:
    footer = "Najarian Owns (INTC); Najarian Owns (GS) Calls; Terranova Is Short (F)."
    assert triage.in_disclosure(footer, footer.index("INTC"), footer.index("INTC") + 4)
    sentence = "Cramer said shares of AMZN rose on Tuesday after the quarter was reported."
    assert not triage.in_disclosure(sentence, sentence.index("AMZN"), sentence.index("AMZN") + 4)
    run = "short: SPY, XRT, QQQ, IWM, DIA, GLD, SLV. long: AAPL"
    assert triage.in_disclosure(run, run.index("XRT"), run.index("XRT") + 3)


def test_the_disclosures_report_counts_the_pairs_named_only_in_a_footer() -> None:
    universe = [
        Company("INTC", "Intel Corporation", "Technology", "Semiconductors"),
        Company("AAPL", "Apple Inc.", "Technology", "Hardware"),
    ]
    bodies = [
        "Apple rose. Najarian Owns (INTC); Najarian Owns (GS) Calls; Terranova Is Short (F).",
        "Intel (NYSE: INTC) fell and Apple rose.",
    ]
    lines = CHECK.disclosures(universe, bodies)
    assert lines[0].startswith("4 counted company-body pairs; 1 (25.0%)")
    assert lines[2].split()[:4] == ["INTC", "1", "/", "2"]


def test_the_labels_are_rendered_with_their_context_from_source(tmp_path: Path) -> None:
    import sqlite3  # noqa: PLC0415

    conn = sqlite3.connect(":memory:")
    conn.execute("ATTACH DATABASE ':memory:' AS source")
    conn.execute("CREATE TABLE source.articles (id INTEGER, body_text TEXT)")
    conn.execute("INSERT INTO source.articles VALUES (7, 'Mark Warner said Time Warner sold it.')")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "set,n,article_id,ticker,word,rule,offset,residue,seed,label,note\n"
        "demo,0,7,WBD,Warner,alias,5,13,1,not,a senator\n"
        "other,0,7,WBD,Warner,alias,25,13,1,not,\n",
        encoding="utf-8",
    )
    lines = CHECK.render_labels(conn, labels, only="demo", width=5)  # type: ignore[arg-type]
    assert lines == ["demo   0 not        WBD   Mark [[Warner]] said  (a senator)"]
