"""The count of the properly fetched corpus (T-159, PLAN D36): the pure part and a tiny SOURCE."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COUNT = _script("count_fetched_corpus_2026_11_08")


def _source(tmp_path: Path, extra: tuple[tuple[object, ...], ...] = ()) -> sqlite3.Connection:
    """A SOURCE with the two tables the count reads, attached as ``source``."""
    path = tmp_path / "source.db"
    with sqlite3.connect(path) as raw:
        raw.execute(
            "CREATE TABLE articles (id INTEGER, fetch_status TEXT, ticker TEXT, title TEXT, "
            "body_text TEXT, pub_date TEXT, word_count INTEGER)"
        )
        raw.execute("CREATE TABLE discovered_urls (id INTEGER, match_class TEXT)")
        rows = [
            # the same body under a right ticker (ok) and a wrong one (quarantined): one body
            (1, "ok", "AAA", "t", "Alpha rose  today", "2023-05-01", 3),
            (2, "quarantined", "BBB", "t", "Alpha rose today", "2023-05-01", 3),
            # a quarantined body that is not ok anywhere: the corpus the pipeline would gain
            (3, "quarantined", "BBB", "t", "Berkshire cash story " * 30, "2022-02-02", 90),
            (4, "quarantined", "CCC", "t", "An older story " * 40, "2019-01-01", 120),
            # thin content is counted apart; a row with no title or no body is not counted at all
            (5, "thin_content", "AAA", "t", "tiny", "2023-01-01", 1),
            (6, "ok", "AAA", "", "no title here", "2023-01-01", 3),
            (7, "ok", "AAA", "t", "", "2023-01-01", 0),
            (8, "ok", "AAA", "t", "A second ok story " * 20, None, 60),
        ]
        raw.executemany("INSERT INTO articles VALUES (?,?,?,?,?,?,?)", [*rows, *extra])
        raw.executemany(
            "INSERT INTO discovered_urls VALUES (?,?)",
            [(1, "accept"), (2, "reject"), (3, "review"), (4, "reject")],
        )
    conn = sqlite3.connect(":memory:")
    conn.execute("ATTACH DATABASE ? AS source", (str(path),))
    return conn


def test_a_body_stored_under_a_right_and_a_wrong_ticker_is_one_body(tmp_path: Path) -> None:
    result = COUNT.summarise(COUNT.read_rows(_source(tmp_path), "2021-12-31"))
    ub = result["unique_bodies"]
    # ok: bodies 1 and 8 (the whitespace-collapsed twin of 2 shares 1's hash); quarantined: 1's twin, 3, 4
    assert ub["ok"] == 2 and ub["quarantined"] == 3
    assert ub["quarantined_only"] == 2  # 3 and 4: not also ok under another ticker
    assert ub["properly_fetched"] == 4 and ub["thin_content_apart"] == 1


def test_a_row_without_a_title_or_a_body_is_not_counted_and_thin_content_is_apart(
    tmp_path: Path,
) -> None:
    result = COUNT.summarise(COUNT.read_rows(_source(tmp_path), "2021-12-31"))
    by_status = result["rows_by_fetch_status"]
    assert by_status == {"ok": 2, "quarantined": 3, "thin_content": 1}  # rows 6 and 7 are out
    classes = result["rows_by_status_and_pair_class"]
    assert classes["quarantined/reject"] == 2 and classes["quarantined/review"] == 1
    assert classes["ok/none"] == 1  # row 8 has no discovered_urls row


def test_the_window_the_years_the_words_and_the_companies_by_tag(tmp_path: Path) -> None:
    result = COUNT.summarise(COUNT.read_rows(_source(tmp_path), "2021-12-31"))
    window = result["unique_bodies_in_window"]
    assert window["ok"] == 1 and window["quarantined_only"] == 1  # row 8 has no date: out of it
    assert result["quarantined_only_by_year"] == {"2019": 1, "2022": 1}
    assert result["quarantined_only_words"] == {"median": 105.0, "under_100": 1, "under_300": 2}
    assert result["companies_by_tag"] == {"with_an_ok_body": 1, "only_quarantined": 2}


def test_the_report_names_the_corpus_of_every_figure(tmp_path: Path) -> None:
    result = COUNT.summarise(COUNT.read_rows(_source(tmp_path), "2021-12-31"))
    text = "\n".join(COUNT.report_lines(result, "2021-12-31"))
    for phrase in ("whole corpus", "the cut's window", "quarantined and not also ok", "apart"):
        assert phrase in text


def test_the_year_of_an_odd_date() -> None:
    assert COUNT.year_of("2023-05-01T10:00:00Z") == "2023"
    assert COUNT.year_of("") == "unknown" and COUNT.year_of(None) == "unknown"
    assert COUNT.year_of("May 2023") == "unknown"


def test_main_writes_the_result_as_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The entry point reads through ``connect_pipeline`` and writes ``--out`` (its parent is made)."""
    conn = _source(tmp_path)
    monkeypatch.setattr(COUNT.db.env, "load_env_file", lambda: None)
    monkeypatch.setattr(COUNT.db, "connect_pipeline", lambda: conn)
    out = tmp_path / "nested" / "count.json"
    monkeypatch.setattr(sys, "argv", ["count", "--since", "2021-12-31", "--out", str(out)])
    COUNT.main()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["since"] == "2021-12-31"
    assert written["unique_bodies"]["properly_fetched"] == 4
    assert "whole corpus" in capsys.readouterr().out


def test_the_rows_left_out_are_counted_by_what_they_lack(tmp_path: Path) -> None:
    result = COUNT.summarise(COUNT.read_rows(_source(tmp_path), "2021-12-31"))
    columns = result["rows_by_status_and_columns"]
    assert columns["ok"] == {
        "rows": 4,
        "with_title": 3,
        "with_body": 3,
        "with_both": 2,
    }  # rows 1, 6, 7, 8
    assert columns["quarantined"]["with_both"] == 3 and columns["thin_content"]["rows"] == 1
    assert "with a title / with a body" in "\n".join(COUNT.report_lines(result, "2021-12-31"))


def test_a_title_or_body_of_spaces_is_not_one(tmp_path: Path) -> None:
    extra = (
        (9, "ok", "AAA", "   ", "a body of words " * 10, "2023-01-01", 30),
        (10, "ok", "AAA", "t", "   \n ", "2023-01-01", 0),
    )
    result = COUNT.summarise(COUNT.read_rows(_source(tmp_path, extra), "2021-12-31"))
    ok = result["rows_by_status_and_columns"]["ok"]
    assert (
        ok["rows"] == 6 and ok["with_title"] == 4 and ok["with_body"] == 4 and ok["with_both"] == 2
    )
    assert result["unique_bodies"]["ok"] == 2  # no phantom body from the spaces
