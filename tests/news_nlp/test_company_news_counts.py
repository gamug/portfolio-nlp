"""The per-company news counts (T-160, PLAN D36): the reading of a body and the counts by corpus."""

from __future__ import annotations

import csv
import importlib.util
import sqlite3
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


COUNTS = _script("company_news_counts_2026_11_09")

UNIVERSE = [
    Company("ZEN", "Zenith Dynamics Inc", "Industrials", "Aerospace"),
    Company("QUO", "Quorra Holdings Corp", "Financials", "Banks"),
    Company("ZENB", "Zenith Dynamics Inc", "Industrials", "Aerospace"),  # a share class of ZEN
    Company("LON", "Lonely Valley Co", "Utilities", "Water"),  # named nowhere
]
RESOLVER = CompanyResolver(UNIVERSE)
TERMS = ["aerospace"]


def _row(i: int, status: str, title: str, body: str) -> tuple[int, str, str, str, str, str]:
    return (i, status, "ZEN", title, body, f"https://example.test/{i}.html")


def _counts(rows: list[tuple[int, str, str, str, str, str]]) -> list[dict[str, object]]:
    entries = COUNTS.read_corpus(rows, RESOLVER, TERMS, lambda _id: [])
    counted: list[dict[str, object]] = COUNTS.count_by_company(entries.values(), RESOLVER, UNIVERSE)
    return counted


def _by_ticker(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(r["ticker"]): r for r in rows}


ZENITH_STORY = "Zenith Dynamics raised its forecast. Zenith Dynamics also said demand is strong."
MIXED_STORY = (
    "Quorra Holdings said that Zenith Dynamics is a customer. Quorra Holdings reports on Friday."
)


def test_a_body_is_counted_once_in_its_corpus_and_in_all() -> None:
    rows = _counts(
        [
            _row(1, "ok", "Zenith up", ZENITH_STORY),
            _row(2, "quarantined", "Zenith up", ZENITH_STORY),  # the same body under a wrong ticker
            _row(3, "quarantined", "Zenith again", ZENITH_STORY + " A second story."),
        ]
    )
    zen = _by_ticker(rows)["ZEN"]
    assert zen["ok_bodies_subject"] == 1  # the twin of body 1 is the same body
    assert zen["quarantined_only_bodies_subject"] == 1  # body 3 is not also ok
    assert zen["all_bodies_subject"] == 2


def test_naming_subject_and_single_are_three_measures() -> None:
    rows = _counts(
        [
            _row(1, "ok", "Zenith up", ZENITH_STORY),  # Zenith twice: a subject, alone
            _row(
                2, "ok", "Quorra news", MIXED_STORY
            ),  # Quorra twice, Zenith once, not in the title
        ]
    )
    by = _by_ticker(rows)
    assert by["ZEN"]["ok_bodies_naming"] == 2
    assert by["ZEN"]["ok_bodies_subject"] == 1 and by["ZEN"]["ok_bodies_single"] == 1
    assert by["QUO"]["ok_bodies_naming"] == 1 and by["QUO"]["ok_bodies_subject"] == 1
    assert by["QUO"]["ok_bodies_single"] == 1  # Zenith was a passing name: Quorra is alone


def test_a_share_class_is_one_row_and_a_company_named_nowhere_is_a_zero_row() -> None:
    rows = _counts([_row(1, "ok", "Zenith up", ZENITH_STORY)])
    by = _by_ticker(rows)
    assert "ZENB" not in by and by["ZEN"]["also_covers"] == "ZENB"
    assert by["LON"]["all_bodies_naming"] == 0 and by["LON"]["all_bodies_subject"] == 0
    assert len(rows) == 3  # ZEN (with ZENB), QUO, LON: the whole universe, even when never named


def test_the_stored_spans_are_given_to_the_resolver() -> None:
    entries = COUNTS.read_corpus(
        [_row(1, "ok", "A story", "Nothing in the text names it.")],
        RESOLVER,
        TERMS,
        lambda _id: ["Quorra Holdings", "Quorra Holdings"],
    )
    (entry,) = entries.values()
    assert entry.reading.named == {"QUO"}


def test_the_csv_has_no_article_text_and_opens_in_excel(tmp_path: Path) -> None:
    rows = _counts([_row(1, "ok", "Zenith up", ZENITH_STORY)])
    path = tmp_path / "out" / "company_news_counts.csv"
    COUNTS.write_csv(path, rows)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # the byte-order mark Excel reads as UTF-8
    with path.open(encoding="utf-8-sig", newline="") as handle:
        read = list(csv.DictReader(handle))
    assert read[0].keys() == {
        "ticker", "company", "also_covers", "gics_sector",
        *(f"{c}_bodies_{m}" for c in COUNTS.CORPORA for m in COUNTS.MEASURES),
    }  # fmt: skip
    assert "Zenith Dynamics raised" not in path.read_text(encoding="utf-8-sig")


def test_the_readme_says_which_corpus_each_column_is_of_and_marks_a_smoke_run() -> None:
    entries = COUNTS.read_corpus(
        [_row(1, "ok", "Zenith up", ZENITH_STORY)], RESOLVER, TERMS, lambda _id: []
    )
    rows = COUNTS.count_by_company(entries.values(), RESOLVER, UNIVERSE)
    text = COUNTS.readme_text(
        entries,
        rows,
        universe_size=4,
        row_total=1,
        new_rows=1,
        limit=1,
        revision="abc",
        elapsed=60.0,
    )
    for phrase in ("ok_*", "quarantined_only_*", "all_*", "UNIQUE BODIES", "SMOKE RUN", "floor"):
        assert phrase in text


def _tiny_store(tmp_path: Path) -> sqlite3.Connection:
    """A SOURCE (``articles`` with the ``gics_*`` columns, ``discovered_urls``) attached as ``source``
    and an ``article_entities`` table in ``main``, as ``connect_pipeline`` gives them."""
    path = tmp_path / "source.db"
    with sqlite3.connect(path) as raw:
        raw.execute(
            "CREATE TABLE articles (id INTEGER, fetch_status TEXT, ticker TEXT, company TEXT, "
            "gics_sector TEXT, gics_sub_industry TEXT, title TEXT, body_text TEXT)"
        )
        raw.execute("CREATE TABLE discovered_urls (id INTEGER, url TEXT)")
        rows = [
            (
                1,
                "ok",
                "ZEN",
                "Zenith Dynamics Inc",
                "Industrials",
                "Aerospace",
                "Zenith up",
                ZENITH_STORY,
            ),
            (
                2,
                "quarantined",
                "QUO",
                "Quorra Holdings Corp",
                "Financials",
                "Banks",
                "Zenith up",
                ZENITH_STORY,
            ),
            (
                3,
                "quarantined",
                "QUO",
                "Quorra Holdings Corp",
                "Financials",
                "Banks",
                "Quorra",
                MIXED_STORY,
            ),
            (
                4,
                "quarantined",
                "QUO",
                "Quorra Holdings Corp",
                "Financials",
                "Banks",
                "",
                "no title",
            ),
            (
                5,
                "thin_content",
                "ZEN",
                "Zenith Dynamics Inc",
                "Industrials",
                "Aerospace",
                "t",
                "thin",
            ),
        ]
        raw.executemany("INSERT INTO articles VALUES (?,?,?,?,?,?,?,?)", rows)
        raw.executemany("INSERT INTO discovered_urls VALUES (?,?)", [(1, "https://e.test/a.html")])
    conn = sqlite3.connect(":memory:")
    conn.execute("ATTACH DATABASE ? AS source", (str(path),))
    conn.execute("CREATE TABLE article_entities (article_id INTEGER, entity_type TEXT, text TEXT)")
    return conn


def test_main_reads_the_store_and_writes_the_csv_and_the_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = _tiny_store(tmp_path)
    monkeypatch.setattr(COUNTS.db.env, "load_env_file", lambda: None)
    monkeypatch.setattr(COUNTS.db, "connect_pipeline", lambda: conn)
    out = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["counts", "--out", str(out)])
    COUNTS.main()
    with (out / "company_news_counts.csv").open(encoding="utf-8-sig", newline="") as handle:
        by = {r["ticker"]: r for r in csv.DictReader(handle)}
    assert set(by) == {"ZEN", "QUO"}  # the universe is the tickers of ok and quarantined rows
    assert by["ZEN"]["ok_bodies_subject"] == "1"  # rows 1 and 2 are one body, and it is ok
    assert by["ZEN"]["quarantined_only_bodies_subject"] == "0"
    assert by["QUO"]["quarantined_only_bodies_subject"] == "1"  # row 3; row 4 has no title
    readme = (out / "company_news_counts_README.txt").read_text(encoding="utf-8")
    assert "rows read 3" in readme and "unique bodies 2" in readme
    assert "universe" in capsys.readouterr().out


def test_a_limit_marks_the_readme_as_a_smoke_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _tiny_store(tmp_path)
    monkeypatch.setattr(COUNTS.db.env, "load_env_file", lambda: None)
    monkeypatch.setattr(COUNTS.db, "connect_pipeline", lambda: conn)
    out = tmp_path / "smoke"
    monkeypatch.setattr(sys, "argv", ["counts", "--limit", "1", "--out", str(out)])
    COUNTS.main()
    assert "SMOKE RUN: stopped after 1 rows" in (out / "company_news_counts_README.txt").read_text(
        encoding="utf-8"
    )


def _run(monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, out: Path, *flags: str) -> None:
    monkeypatch.setattr(COUNTS.db.env, "load_env_file", lambda: None)
    monkeypatch.setattr(COUNTS.db, "connect_pipeline", lambda: conn)
    monkeypatch.setattr(sys, "argv", ["counts", "--no-progress", "--out", str(out), *flags])
    COUNTS.main()


def _files(out: Path) -> tuple[bytes, str]:
    readme = (out / "company_news_counts_README.txt").read_text(encoding="utf-8")
    body = "\n".join(line for line in readme.splitlines() if not line.startswith("Made "))
    return (out / "company_news_counts.csv").read_bytes(), body


def test_a_stopped_run_continues_and_ends_as_a_run_that_never_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _tiny_store(tmp_path)
    _run(monkeypatch, conn, tmp_path / "whole")
    _run(monkeypatch, conn, tmp_path / "split", "--limit", "1")  # reads one row, saves, stops
    assert (tmp_path / "split" / "company_news_counts.checkpoint.json").exists()
    _run(monkeypatch, conn, tmp_path / "split")  # continues after the saved row
    assert _files(tmp_path / "split")[0] == _files(tmp_path / "whole")[0]


def test_running_again_on_an_unchanged_store_rewrites_the_same_csv_and_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = _tiny_store(tmp_path)
    out = tmp_path / "result"
    _run(monkeypatch, conn, out)
    first = _files(out)[0]
    capsys.readouterr()
    _run(monkeypatch, conn, out)
    assert _files(out)[0] == first
    assert "resuming after row id" in capsys.readouterr().out
    assert "0 in the run that wrote this file" in _files(out)[1]


def test_ctrl_c_saves_what_was_read_and_the_next_run_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _tiny_store(tmp_path)
    _run(monkeypatch, conn, tmp_path / "whole")
    real = COUNTS.stream_rows

    def interrupted(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        for n, row in enumerate(real(*args, **kwargs)):
            if n == 2:
                raise KeyboardInterrupt
            yield row

    out = tmp_path / "stopped"
    monkeypatch.setattr(COUNTS, "stream_rows", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run(monkeypatch, conn, out)
    assert (out / "company_news_counts.checkpoint.json").exists()
    assert not (out / "company_news_counts.csv").exists()  # no half-written result
    monkeypatch.setattr(COUNTS, "stream_rows", real)
    _run(monkeypatch, conn, out)
    assert _files(out)[0] == _files(tmp_path / "whole")[0]


def test_a_checkpoint_of_other_code_or_a_fresh_run_starts_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = _tiny_store(tmp_path)
    out = tmp_path / "result"
    _run(monkeypatch, conn, out)
    path = out / "company_news_counts.checkpoint.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"fingerprint":"', '"fingerprint":"x'))
    capsys.readouterr()
    _run(monkeypatch, conn, out)
    assert "the code or the universe changed" in capsys.readouterr().out
    _run(monkeypatch, conn, out, "--fresh")
    assert "--fresh was given" in capsys.readouterr().out
    path.write_text("not json")
    _run(monkeypatch, conn, out)
    assert "not readable" in capsys.readouterr().out


def test_a_file_is_written_whole_or_not_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "x.txt"
    COUNTS.atomic_write_text(target, "old")

    def failing(*_args: object) -> None:
        raise OSError("the disk went away")

    monkeypatch.setattr(COUNTS.os, "replace", failing)
    with pytest.raises(OSError, match="disk"):
        COUNTS.atomic_write_text(target, "new")
    assert target.read_text() == "old"  # the old file is untouched


def test_the_bar_shows_the_rows_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = _tiny_store(tmp_path)
    monkeypatch.setattr(COUNTS.db.env, "load_env_file", lambda: None)
    monkeypatch.setattr(COUNTS.db, "connect_pipeline", lambda: conn)
    monkeypatch.setattr(sys, "argv", ["counts", "--out", str(tmp_path / "bar")])  # the bar is on
    COUNTS.main()
    err = capsys.readouterr().err
    assert "reading" in err and "3/3" in err  # the three valid rows, all read
