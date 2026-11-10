"""The per-company news counts (T-160, PLAN D36): the reading of a body and the counts by corpus."""

from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

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


@pytest.fixture(autouse=True)
def _fixed_day(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(COUNTS, "today", lambda: "2026-11-09")


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
    entries = COUNTS.read_corpus(rows, RESOLVER, TERMS)
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


def test_the_csv_has_no_article_text_and_opens_in_excel(tmp_path: Path) -> None:
    rows = _counts([_row(1, "ok", "Zenith up", ZENITH_STORY)])
    path = tmp_path / "out" / "company_news_counts_2026-11-09.csv"
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
    entries = COUNTS.read_corpus([_row(1, "ok", "Zenith up", ZENITH_STORY)], RESOLVER, TERMS)
    rows = COUNTS.count_by_company(entries.values(), RESOLVER, UNIVERSE)
    text = COUNTS.readme_text(
        entries,
        rows,
        universe_size=4,
        row_total=1,
        new_rows=1,
        everything=2,
        day="2026-11-09",
        revision="abc",
        elapsed=60.0,
    )
    for phrase in (
        "ok_*",
        "quarantined_only_*",
        "all_*",
        "UNIQUE BODIES",
        "SMOKE RUN",
        "floor",
        "*_bodies_disclosure",
        "Trader A Owns (INTC)",
        "READING BY PATTERNS",
    ):
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
            (
                6,
                "ok",
                "ZEN",
                "Zenith Dynamics Inc",
                "Industrials",
                "Aerospace",
                "\t",
                ZENITH_STORY + " x",
            ),
            (7, "ok", "ZEN", "Zenith Dynamics Inc", "Industrials", "Aerospace", "t", "\n\n"),
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
    with (out / "company_news_counts_2026-11-09.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        by = {r["ticker"]: r for r in csv.DictReader(handle)}
    assert set(by) == {"ZEN", "QUO"}  # the universe is the tickers of ok and quarantined rows
    assert by["ZEN"]["ok_bodies_subject"] == "1"  # rows 1 and 2 are one body, and it is ok
    assert by["ZEN"]["quarantined_only_bodies_subject"] == "0"
    assert by["QUO"]["quarantined_only_bodies_subject"] == "1"  # row 3; row 4 has no title
    readme = (out / "company_news_counts_2026-11-09_README.txt").read_text(encoding="utf-8")
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
    assert "SMOKE RUN: 1 of 3 rows read" in (
        out / "company_news_counts_2026-11-09_README.txt"
    ).read_text(encoding="utf-8")


def _run(monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, out: Path, *flags: str) -> None:
    monkeypatch.setattr(COUNTS.db.env, "load_env_file", lambda: None)
    monkeypatch.setattr(COUNTS.db, "connect_pipeline", lambda: conn)
    monkeypatch.setattr(sys, "argv", ["counts", "--no-progress", "--out", str(out), *flags])
    COUNTS.main()


def _files(out: Path) -> tuple[bytes, str]:
    readme = (out / "company_news_counts_2026-11-09_README.txt").read_text(encoding="utf-8")
    body = "\n".join(line for line in readme.splitlines() if not line.startswith("Made "))
    return (out / "company_news_counts_2026-11-09.csv").read_bytes(), body


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
    assert not (out / "company_news_counts_2026-11-09.csv").exists()  # no half-written result
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


def test_a_file_open_in_excel_gives_a_message_and_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "x.csv"
    target.write_text("old")

    def locked(*_args: object) -> None:
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(COUNTS.os, "replace", locked)
    with pytest.raises(PermissionError, match="Close it and run the same command again"):
        COUNTS.atomic_write_text(target, "new")
    assert target.read_text() == "old" and not list(tmp_path.glob("*.tmp"))


def test_a_locked_csv_ends_the_run_with_the_message_and_the_checkpoint_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _tiny_store(tmp_path)
    real = COUNTS.os.replace

    def locked_csv(source: object, target: object) -> None:
        if str(target).endswith("company_news_counts_2026-11-09.csv"):
            raise PermissionError(13, "locked")
        real(source, target)

    monkeypatch.setattr(COUNTS.os, "replace", locked_csv)
    out = tmp_path / "result"
    with pytest.raises(SystemExit, match="open in another program"):
        _run(monkeypatch, conn, out)
    assert (out / "company_news_counts.checkpoint.json").exists()
    monkeypatch.setattr(COUNTS.os, "replace", real)
    _run(monkeypatch, conn, out)  # the same command finishes without reading a row
    assert (out / "company_news_counts_2026-11-09.csv").exists()


def test_the_rows_are_counted_once_and_a_complete_corpus_is_not_marked_as_a_smoke_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    conn = _tiny_store(tmp_path)
    out = tmp_path / "result"
    _run(monkeypatch, conn, out)
    calls: list[int] = []
    real = COUNTS.count_rows

    def counting(c: sqlite3.Connection, after: int = 0) -> int:
        calls.append(after)
        return int(real(c, after))

    monkeypatch.setattr(COUNTS, "count_rows", counting)
    capsys.readouterr()
    _run(monkeypatch, conn, out, "--limit", "1")  # the corpus is already complete in the checkpoint
    assert len(calls) == 1
    assert "counting the rows" in capsys.readouterr().out
    assert "SMOKE RUN" not in _files(out)[1]


def test_the_checkpoint_is_saved_every_so_many_rows_and_survives_a_run_that_dies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _tiny_store(tmp_path)
    _run(monkeypatch, conn, tmp_path / "whole")
    real_save, real_stream = COUNTS.save_checkpoint, COUNTS.stream_rows
    saves: list[int] = []

    def spy(path: Path, print_of: str, last_id: int, rows_read: int, entries: Any) -> None:
        if not saves:  # only the periodic save reaches the disk: the run dies before the last one
            real_save(path, print_of, last_id, rows_read, entries)
        saves.append(last_id)

    def dies(*args: Any, **kwargs: Any) -> Iterator[Any]:
        for n, row in enumerate(real_stream(*args, **kwargs)):
            if n == 2:
                raise KeyboardInterrupt
            yield row

    out = tmp_path / "died"
    monkeypatch.setattr(COUNTS, "CHECKPOINT_EVERY", 2)
    monkeypatch.setattr(COUNTS, "save_checkpoint", spy)
    monkeypatch.setattr(COUNTS, "stream_rows", dies)
    with pytest.raises(KeyboardInterrupt):
        _run(monkeypatch, conn, out)
    assert saves[0] == 2  # the periodic save, after the second row
    saved = json.loads((out / "company_news_counts.checkpoint.json").read_text(encoding="utf-8"))
    assert saved["last_id"] == 2 and saved["rows_read"] == 2
    monkeypatch.setattr(COUNTS, "save_checkpoint", real_save)
    monkeypatch.setattr(COUNTS, "stream_rows", real_stream)
    _run(monkeypatch, conn, out)
    assert _files(out)[0] == _files(tmp_path / "whole")[0]


def test_the_fingerprint_changes_with_the_sub_industry_the_rules_read() -> None:
    other = [UNIVERSE[0]._replace(gics_sub_industry="Defence"), *UNIVERSE[1:]]
    assert COUNTS.fingerprint(UNIVERSE) != COUNTS.fingerprint(other)
    assert COUNTS.fingerprint(UNIVERSE) == COUNTS.fingerprint(list(reversed(UNIVERSE)))


def test_the_readme_does_not_call_the_ok_column_the_cut_and_says_what_the_universe_is() -> None:
    entries = COUNTS.read_corpus([_row(1, "ok", "Zenith up", ZENITH_STORY)], RESOLVER, TERMS)
    rows = COUNTS.count_by_company(entries.values(), RESOLVER, UNIVERSE)
    text = COUNTS.readme_text(
        entries,
        rows,
        universe_size=4,
        row_total=1,
        new_rows=1,
        everything=1,
        day="2026-11-09",
        revision="a",
        elapsed=1.0,
    )
    assert "published from 2021-12-31" in text and "not the index's membership on a date" in text
    assert "which this run does not use" in text
    assert "SMOKE RUN" not in text


def _good_rows() -> list[dict[str, object]]:
    return _counts([_row(1, "ok", "Zenith up", ZENITH_STORY)])


def test_the_invariants_hold_for_a_real_count_and_a_broken_one_stops_the_run() -> None:
    COUNTS.check_invariants(_good_rows())  # no error
    broken = _good_rows()
    broken[0]["all_bodies_naming"] = 99  # all is not ok + quarantined_only
    with pytest.raises(COUNTS.CheckFailed, match="all_bodies_naming is not ok"):
        COUNTS.check_invariants(broken)
    worse = _good_rows()
    worse[0]["ok_bodies_single"] = worse[0]["ok_bodies_subject"] + 1  # type: ignore[operator]
    worse[0]["all_bodies_single"] = worse[0]["ok_bodies_single"]
    with pytest.raises(COUNTS.CheckFailed, match="single <= subject <= naming"):
        COUNTS.check_invariants(worse)


def test_a_failed_check_writes_no_file_and_says_not_to_send_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _tiny_store(tmp_path)

    def broken(*_a: object, **_k: object) -> None:
        raise COUNTS.CheckFailed("ZEN: all_bodies_naming is not ok + quarantined_only")

    monkeypatch.setattr(COUNTS, "check_invariants", broken)
    out = tmp_path / "result"
    with pytest.raises(SystemExit, match="Do not send results of this run"):
        _run(monkeypatch, conn, out)
    assert not list(out.glob("*.csv")) and not list(out.glob("*README*"))
    assert not (out / "company_news_counts.lock").exists()  # the lock is released


def test_the_readme_compares_with_t159_only_when_the_corpus_was_read_in_full() -> None:
    full = COUNTS.reconciliation(309_000, 81_900, 3, 3)
    assert "309,882 / 81,961 / 227,921" in full and "309,000 / 81,900 / 227,100" in full
    assert "compare them" in full and "not comparable" not in full
    assert "not comparable" in COUNTS.reconciliation(1, 1, 1, 3)


def test_a_second_run_on_the_same_folder_is_refused_and_the_lock_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _tiny_store(tmp_path)
    out = tmp_path / "result"
    lock = COUNTS.acquire_lock(out)  # a run is going
    with pytest.raises(SystemExit, match="another run is using"):
        _run(monkeypatch, conn, out)
    lock.unlink()
    _run(monkeypatch, conn, out)  # free: runs, and removes its own lock
    assert not lock.exists()


def test_the_lock_is_released_after_ctrl_c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _tiny_store(tmp_path)
    real = COUNTS.stream_rows

    def interrupted(*args: Any, **kwargs: Any) -> Iterator[Any]:
        for n, row in enumerate(real(*args, **kwargs)):
            if n == 1:
                raise KeyboardInterrupt
            yield row

    monkeypatch.setattr(COUNTS, "stream_rows", interrupted)
    out = tmp_path / "result"
    with pytest.raises(KeyboardInterrupt):
        _run(monkeypatch, conn, out)
    assert not (out / "company_news_counts.lock").exists()


def test_the_files_carry_the_run_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _tiny_store(tmp_path)
    out = tmp_path / "result"
    _run(monkeypatch, conn, out)
    names = sorted(p.name for p in out.iterdir())
    assert "company_news_counts_2026-11-09.csv" in names
    assert "company_news_counts_2026-11-09_README.txt" in names
    assert "company_news_counts_2026-11-09.csv" in (
        out / "company_news_counts_2026-11-09_README.txt"
    ).read_text(encoding="utf-8")


def test_a_title_or_body_of_tabs_and_new_lines_is_not_one_here_either(tmp_path: Path) -> None:
    """Rows 6 and 7 of the tiny store (a title of a tab, a body of new lines) are out, as in T-159."""
    conn = _tiny_store(tmp_path)
    assert COUNTS.count_rows(conn) == 3
    assert [row[0] for row in COUNTS.stream_rows(conn, 0)] == [1, 2, 3]


def test_the_stored_ner_spans_are_not_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _tiny_store(tmp_path)
    conn.executemany(
        "INSERT INTO article_entities VALUES (?, 'ORG', ?)",
        [(1, "Quorra Holdings"), (1, "Quorra Holdings"), (1, "Quorra Holdings")],
    )  # spans on article 1, a Zenith story: they would make Quorra a subject of it
    out = tmp_path / "result"
    _run(monkeypatch, conn, out)
    with (out / "company_news_counts_2026-11-09.csv").open(encoding="utf-8-sig", newline="") as fh:
        by = {r["ticker"]: r for r in csv.DictReader(fh)}
    assert by["QUO"]["ok_bodies_naming"] == "0"  # only the text counts


def test_the_universe_does_not_depend_on_the_order_of_the_rows(tmp_path: Path) -> None:
    def store(order: list[tuple[int, str]]) -> sqlite3.Connection:
        path = tmp_path / f"s{order[0][0]}.db"
        with sqlite3.connect(path) as raw:
            raw.execute(
                "CREATE TABLE articles (id INTEGER, fetch_status TEXT, ticker TEXT, company TEXT, "
                "gics_sector TEXT, gics_sub_industry TEXT, title TEXT, body_text TEXT)"
            )
            raw.executemany(
                "INSERT INTO articles VALUES (?, 'ok', 'ZEN', ?, 'Industrials', 'Aerospace', 't', 'b')",
                order,
            )
        conn = sqlite3.connect(":memory:")
        conn.execute("ATTACH DATABASE ? AS source", (str(path),))
        return conn

    rows = [(1, "Zenith Dynamics Inc"), (2, "Zenith Dynamics, Inc.")]
    assert COUNTS.load_universe(store(rows)) == COUNTS.load_universe(store(rows[::-1]))


def test_the_file_name_takes_the_date_of_the_machine_not_utc() -> None:
    fresh = _script("company_news_counts_2026_11_09")  # the real ``today``, not the patched one
    assert fresh.today() == f"{datetime.now().astimezone():%Y-%m-%d}"


def test_a_checkpoint_gives_back_the_bodies_it_was_given(tmp_path: Path) -> None:
    entries = COUNTS.read_corpus([_row(1, "ok", "Zenith up", ZENITH_STORY)], RESOLVER, TERMS)
    path = tmp_path / "c.json"
    COUNTS.save_checkpoint(path, "fp", 1, 1, entries)
    loaded = COUNTS.load_checkpoint(path, "fp")
    assert not isinstance(loaded, str)
    last, rows, back = loaded
    assert (last, rows) == (1, 1)
    (before,), (after,) = entries.values(), back.values()
    assert (before.reading, before.is_ok) == (after.reading, after.is_ok)
    assert COUNTS.load_checkpoint(path, "another") != loaded  # other code or universe: not used


DISCLOSURE_FOOTER = "Trader A Owns (ZEN); Trader A Owns (GS) Calls; Trader B Is Short (F)."


def test_a_company_named_only_in_a_disclosure_is_counted_as_named_and_as_disclosure() -> None:
    """The owner's decision of 2026-11-09 (PLAN D38): a separate column, the naming count unchanged."""
    universe = [Company("ZEN", "Zenith Dynamics Inc", "Industrials", "Aerospace")]
    resolver = CompanyResolver(universe)
    bodies = [
        _row(1, "ok", "Market notes", "Stocks rose on Friday. " + DISCLOSURE_FOOTER),
        _row(2, "ok", "Zenith up", ZENITH_STORY + " " + DISCLOSURE_FOOTER),  # named in the text too
        _row(3, "quarantined", "Market notes", "Stocks fell on Monday. " + DISCLOSURE_FOOTER),
    ]
    entries = COUNTS.read_corpus(bodies, resolver, TERMS)
    zen = COUNTS.count_by_company(entries.values(), resolver, universe)[0]
    assert zen["ok_bodies_naming"] == 2 and zen["ok_bodies_disclosure"] == 1
    assert (
        zen["quarantined_only_bodies_naming"] == 1
        and zen["quarantined_only_bodies_disclosure"] == 1
    )
    assert zen["all_bodies_naming"] == 3 and zen["all_bodies_disclosure"] == 2
    COUNTS.check_invariants([zen])


def test_the_disclosure_column_cannot_exceed_the_naming_column() -> None:
    broken = _good_rows()
    broken[0]["ok_bodies_disclosure"] = broken[0]["ok_bodies_naming"] + 1  # type: ignore[operator]
    broken[0]["all_bodies_disclosure"] = broken[0]["ok_bodies_disclosure"]
    with pytest.raises(COUNTS.CheckFailed, match="disclosure <= naming"):
        COUNTS.check_invariants(broken)


def test_a_body_that_names_the_company_in_its_text_has_no_disclosure_reading() -> None:
    entries = COUNTS.read_corpus([_row(1, "ok", "Zenith up", ZENITH_STORY)], RESOLVER, TERMS)
    assert next(iter(entries.values())).reading.disclosure == frozenset()


def test_the_readme_lists_exactly_the_measures_and_gives_the_runs_own_disclosure_share() -> None:
    entries = COUNTS.read_corpus(
        [
            _row(1, "ok", "Zenith up", ZENITH_STORY),
            _row(2, "ok", "Market notes", "Stocks rose on Friday. " + DISCLOSURE_FOOTER),
        ],
        RESOLVER,
        TERMS,
    )
    rows = COUNTS.count_by_company(entries.values(), RESOLVER, UNIVERSE)
    text = COUNTS.readme_text(
        entries,
        rows,
        universe_size=4,
        row_total=2,
        new_rows=2,
        everything=2,
        day="2026-11-09",
        revision="abc",
        elapsed=60.0,
    )
    measures = text.split("THE FOUR MEASURES")[1].split("THE TAG IS NOT USED")[0]
    for measure in COUNTS.MEASURES:
        assert f"*_bodies_{measure}" in measures
    assert measures.count("*_bodies_") == len(COUNTS.MEASURES)
    assert "THREE MEASURES" not in text and "Najarian" not in text and "fifth" not in text
    assert "(of them 1 only in a disclosure)" in text
    assert "naming 1 of 2 (50.0%)" in text and "subject 0 of 1 (0.0%)" in text


def test_the_disclosure_is_read_before_the_boilerplate_rules_cut_its_words() -> None:
    """``strip_boilerplate`` removes the sentence with "Disclosure:" and leaves tickers bare: the
    reading of the genre is on the body as it was."""
    universe = [Company("ZEN", "Zenith Dynamics Inc", "Industrials", "Aerospace")]
    resolver = CompanyResolver(universe)
    body = (
        "Stocks rose on Friday. Sign up for our newsletter. Trader A Owns (ZEN), (AAA), (BBB), "
        "(CCC), (DDD), (EEE), (FFF) Disclosure: Click here for more. See (ZEN)"
    )
    reading = COUNTS.read_body(resolver, "Market notes", body, "", TERMS)
    assert "ZEN" in reading.named
    assert "ZEN" in reading.disclosure
