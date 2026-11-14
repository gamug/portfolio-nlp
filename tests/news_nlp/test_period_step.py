"""Step 9, the period summary (T-196, D42): every 21 days from 2022-01-03 at 15:00 in BATCH_TIMEZONE, the counts of the snapshots per company and per
category, and the same counts summed per sector. The window, the order, what a period counts, what is late, what a reset clears. No model, no network."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sqlite3
import sys
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

import news_nlp as db_module
import news_period_stage as period
import news_snapshot_stage as snap
from news_nlp import steps, triage
from sector_stage import SECTOR_CODES, SECTOR_DESCRIPTIONS
from tests.news_nlp.conftest import ARTICLES_SCHEMA, seed_article, stamp_source

NY = ZoneInfo("America/New_York")
SNAP = "snapshot-test"
ROOT = Path(__file__).resolve().parents[2]
END0 = "2022-01-03T20:00:00Z"  # 15:00 EST
END1 = "2022-01-24T20:00:00Z"
END2 = "2022-02-14T20:00:00Z"
END3 = "2022-03-07T20:00:00Z"
AFTER_P3 = datetime(2022, 3, 8, tzinfo=UTC)  # periods 0 to 3 have passed
AFTER_P1 = datetime(2022, 1, 25, tzinfo=UTC)


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location(
        "news_nlp_steps_cli_period", ROOT / "cli" / "news_nlp_steps.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def conn(tmp_path: Path, results_db_path: Path) -> Iterator[db_module.NewsNlpDatabase]:
    """A universe of four companies (two in Information Technology, one in Energy, one with no sector) in SOURCE, and an empty RESULTS store."""
    source = tmp_path / "src.db"
    raw = sqlite3.connect(source)
    raw.executescript(ARTICLES_SCHEMA)
    stamp_source(raw)
    for i, (ticker, sector) in enumerate(
        [
            ("AAPL", "Information Technology"),
            ("MSFT", "Information Technology"),
            ("XOM", "Energy"),
            ("ZZZ", ""),
        ],
        start=1,
    ):
        seed_article(
            raw,
            id=i,
            ticker=ticker,
            company=ticker,
            gics_sector=sector,
            body_text=f"Body of {ticker}. " * 20,
            fetched_at="2026-08-20T10:00:00Z",  # crawled years after it was published: only the publication date puts a gap in window 1
            pub_date="2022-01-10T10:00:00Z",
        )
    raw.commit()
    raw.close()
    c = db_module.connect_pipeline(results_db=results_db_path, source_db=source)
    yield c
    db_module.detach_source(c)


def _add(
    conn: db_module.NewsNlpDatabase,
    item: str,
    ticker: str,
    fetched_at: str,
    category: str = "earnings_performance",
    sentiment: str = "positive",
    revision: str = SNAP,
    ticker_blank: bool = False,
) -> None:
    conn.execute(
        "INSERT INTO news_snapshot (item_kind, item_id, ticker, model_revision, fetched_at, text, text_source, category, sentiment, inputs, processed_at) "
        "VALUES ('body', ?, ?, ?, ?, 't', 'summary', ?, ?, '{}', ?)",
        (
            item,
            "" if ticker_blank else ticker,
            revision,
            fetched_at,
            category,
            sentiment,
            fetched_at,
        ),
    )
    conn.commit()


def _run(
    conn: db_module.NewsNlpDatabase, now: datetime = AFTER_P3, index: int | None = None
) -> dict[str, Any]:
    return period.run(conn, SNAP, tz=NY, now=now, index=index)


def _summary(conn: db_module.NewsNlpDatabase, end: str, kind: str) -> dict[str, dict[str, Any]]:
    return {
        str(r["key"]): json.loads(r["counts"])
        for r in conn.execute(
            "SELECT key, counts FROM period_summary WHERE period_end = ? AND kind = ?", (end, kind)
        )
    }


# --- the time rule ---------------------------------------------------------------------------------------------------


def test_the_cuts_are_15_00_new_york_in_utc_and_a_window_is_503_or_505_hours_across_a_clock_change() -> (
    None
):
    assert period.cut_utc(0, NY) == END0  # winter: UTC-5
    assert period.cut_utc(8, NY) == "2022-06-20T19:00:00Z"  # summer: UTC-4
    assert period.period_date(0) == date(2022, 1, 3) and period.period_date(1) == date(2022, 1, 24)
    hours = [
        (
            datetime.fromisoformat(period.cut_utc(k + 1, NY))
            - datetime.fromisoformat(period.cut_utc(k, NY))
        ).total_seconds()
        / 3600
        for k in range(40)
    ]
    assert set(hours) == {504.0, 503.0, 505.0} and 503.0 in hours and 505.0 in hours
    assert all(period.period_date(k).weekday() == 0 for k in range(40))  # every cut is a Monday
    assert period.cut_utc(0, ZoneInfo("America/Bogota")) == "2022-01-03T20:00:00Z"  # UTC-5 all year


def test_a_date_off_the_grid_or_before_the_first_period_is_refused() -> None:
    assert period.index_of(date(2022, 1, 3)) == 0 and period.index_of(date(2022, 1, 24)) == 1
    for bad in (date(2022, 1, 4), date(2021, 12, 13), date(2022, 1, 17)):
        with pytest.raises(period.PeriodRefused, match="not a period end"):
            period.index_of(bad)


def test_the_last_period_that_has_passed_follows_the_cut_not_the_date() -> None:
    assert period.last_index(datetime(2022, 1, 3, 19, 59, tzinfo=UTC), NY) == -1  # 14:59 New York
    assert period.last_index(datetime(2022, 1, 3, 20, 0, tzinfo=UTC), NY) == 0  # the cut itself
    assert period.last_index(datetime(2022, 1, 24, 20, 1, tzinfo=UTC), NY) == 1
    assert period.last_index(datetime(2021, 6, 1, tzinfo=UTC), NY) == -1


def test_an_invalid_time_zone_stops_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BATCH_TIMEZONE", "Mars/Olympus")
    with pytest.raises(ValueError, match="not a time zone name"):
        period.batch_timezone()
    monkeypatch.delenv("BATCH_TIMEZONE")
    assert str(period.batch_timezone()) == "America/New_York"
    monkeypatch.setenv("BATCH_TIMEZONE", "America/Bogota")
    assert str(period.batch_timezone()) == "America/Bogota"


# --- what a period counts --------------------------------------------------------------------------------------------


def test_the_window_is_the_previous_cut_excluded_to_this_cut_included_and_the_first_has_no_lower_bound(
    conn: db_module.NewsNlpDatabase,
) -> None:
    _add(conn, "a", "AAPL", "2021-06-01T10:00:00Z")  # long before: still the first window
    _add(conn, "b", "AAPL", END0)  # exactly at the cut: in period 0
    _add(conn, "c", "AAPL", "2022-01-03T20:00:01Z")  # one second after: period 1
    _add(conn, "d", "AAPL", END1)
    result = _run(conn, AFTER_P1)
    assert [p["counted"] for p in result["periods"]] == [2, 2]
    first = conn.execute(
        "SELECT period_start FROM period_run WHERE period_end = ?", (END0,)
    ).fetchone()
    assert first["period_start"] is None  # no lower bound
    assert (
        conn.execute("SELECT SUM(late) FROM period_item WHERE period_end = ?", (END0,)).fetchone()[
            0
        ]
        == 0
    )
    assert (
        conn.execute(
            "SELECT period_start FROM period_run WHERE period_end = ?", (END1,)
        ).fetchone()[0]
        == END0
    )


def test_a_company_line_counts_categories_and_the_sentiment_split_one_count_per_snapshot(
    conn: db_module.NewsNlpDatabase,
) -> None:
    for i, (category, sentiment) in enumerate(
        [
            ("labor", "negative"),
            ("labor", "negative"),
            ("legal_regulatory", "positive"),
            ("labor", "neutral"),
        ]
    ):
        _add(
            conn, f"s{i}", "AAPL", "2022-01-05T10:00:00Z", category, sentiment
        )  # four sub-news of one company count four
    _run(conn)
    assert _summary(conn, END1, "company")["AAPL"] == {
        "labor": {"total": 3, "neutral": 1, "positive": 0, "negative": 2},
        "legal_regulatory": {"total": 1, "neutral": 0, "positive": 1, "negative": 0},
    }


def test_a_sector_line_is_the_sum_of_its_companies_and_a_company_with_no_sector_is_unclassified(
    conn: db_module.NewsNlpDatabase,
) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z", "labor", "negative")
    _add(conn, "b", "MSFT", "2022-01-06T10:00:00Z", "labor", "positive")
    _add(conn, "c", "ZZZ", "2022-01-07T10:00:00Z", "legal_regulatory", "neutral")
    _add(
        conn, "d", "NOPE", "2022-01-08T10:00:00Z", "labor", "neutral"
    )  # a ticker the universe does not have
    _run(conn)
    sectors = _summary(conn, END1, "sector")
    assert sectors["45"] == {
        "labor": {"total": 2, "neutral": 0, "positive": 1, "negative": 1}
    }  # AAPL + MSFT
    assert (
        sectors[""]["legal_regulatory"]["total"] == 1 and sectors[""]["labor"]["total"] == 1
    )  # ZZZ and NOPE
    assert "10" not in sectors  # Energy has no news: no row
    companies = _summary(conn, END1, "company")
    assert "XOM" not in companies and set(companies) == {"AAPL", "MSFT", "ZZZ", "NOPE"}
    total = sum(c["total"] for cats in sectors.values() for c in cats.values())
    assert total == sum(c["total"] for cats in companies.values() for c in cats.values())


def test_a_sector_text_is_not_counted(conn: db_module.NewsNlpDatabase) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    _add(conn, "s", "", "2022-01-05T10:00:00Z", ticker_blank=True)
    assert _run(conn)["periods"][1]["counted"] == 1


def test_an_empty_window_in_the_middle_is_written_and_does_not_block_the_next(
    conn: db_module.NewsNlpDatabase,
) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")  # period 1
    _add(conn, "b", "AAPL", "2022-02-20T10:00:00Z")  # period 3; period 2 has nothing
    result = _run(conn)
    assert [p["counted"] for p in result["periods"]] == [0, 1, 0, 1]
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM period_summary WHERE period_end = ?", (END2,)
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute("SELECT COUNT(*) FROM period_run WHERE period_end = ?", (END2,)).fetchone()[0]
        == 1
    )


def test_a_snapshot_that_arrives_after_its_window_was_written_is_counted_next_and_marked_late(
    conn: db_module.NewsNlpDatabase,
) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    _run(conn, AFTER_P1)  # periods 0 and 1 written
    _add(conn, "late", "AAPL", "2022-01-20T10:00:00Z")  # crawled in window 1, snapshotted only now
    _add(conn, "new", "AAPL", "2022-02-01T10:00:00Z")
    _run(conn, AFTER_P3)
    lates = {
        (r["item_id"], r["period_end"]): r["late"]
        for r in conn.execute("SELECT item_id, period_end, late FROM period_item")
    }
    assert lates[("a", END1)] == 0 and lates[("late", END2)] == 1 and lates[("new", END2)] == 0
    assert (
        _summary(conn, END1, "company")["AAPL"]["earnings_performance"]["total"] == 1
    )  # the written period is frozen


def test_a_written_period_is_never_rewritten_and_a_rerun_adds_nothing(
    conn: db_module.NewsNlpDatabase,
) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    _run(conn)
    before = [
        tuple(r)
        for r in conn.execute("SELECT * FROM period_summary ORDER BY period_end, kind, key")
    ]
    items = conn.execute("SELECT COUNT(*) FROM period_item").fetchone()[0]
    again = _run(conn)
    assert again["to_do"] == 0 and again["periods"] == []
    assert _run(conn, index=1)["periods"] == []  # a period already written is skipped, not an error
    assert [
        tuple(r)
        for r in conn.execute("SELECT * FROM period_summary ORDER BY period_end, kind, key")
    ] == before
    assert conn.execute("SELECT COUNT(*) FROM period_item").fetchone()[0] == items


def test_a_period_is_refused_while_an_earlier_one_is_missing_or_its_cut_has_not_passed(
    conn: db_module.NewsNlpDatabase,
) -> None:
    with pytest.raises(period.PeriodRefused, match="are not written"):
        _run(conn, index=2)
    _run(conn, index=0)
    _run(conn, index=1)
    _run(conn, index=2)
    with pytest.raises(period.PeriodRefused, match="has not passed"):
        _run(conn, AFTER_P3, index=4)


def test_catch_up_runs_every_missing_period_in_order_up_to_the_last_cut_that_has_passed(
    conn: db_module.NewsNlpDatabase,
) -> None:
    result = _run(conn, AFTER_P3)
    assert result["to_do"] == 4
    assert [p["period_end"] for p in result["periods"]] == [END0, END1, END2, END3]
    _run(conn, index=0)  # nothing new
    assert (
        _run(conn, datetime(2022, 4, 1, tzinfo=UTC))["to_do"] == 1
    )  # period 4 (2022-03-28) has passed since


def test_the_not_counted_pairs_are_the_main_pairs_with_no_snapshot_on_rows_that_exist(
    conn: db_module.NewsNlpDatabase,
) -> None:
    body_a = "Body of AAPL. " * 20
    for kind_id, ticker in (
        (triage.body_hash(body_a), "AAPL"),
        (triage.body_hash("Body of MSFT. " * 20), "MSFT"),
    ):
        steps.record_companies(
            conn, steps.ITEM_BODY, kind_id, "companies-t", [steps.CompanyScore(ticker, 1, 0.9)], 0.5
        )
    conn.execute(
        "INSERT INTO news_snapshot (item_kind, item_id, ticker, model_revision, fetched_at, text, text_source, category, sentiment, inputs, processed_at) "
        "VALUES ('body', 'other', 'AAPL', ?, '2022-01-12T10:00:00Z', 't', 'summary', 'labor', 'positive', ?, 't')",
        (SNAP, json.dumps({"companies": "companies-t"})),
    )
    conn.commit()
    _run(conn)
    companies = {
        r["key"]: r["not_counted"]
        for r in conn.execute(
            "SELECT key, not_counted FROM period_summary WHERE period_end = ? AND kind = 'company'",
            (END1,),
        )
    }
    assert companies == {
        "AAPL": 1
    }  # the unsnapshotted AAPL pair; MSFT has no counted news, so no row for it
    sectors = {
        r["key"]: r["not_counted"]
        for r in conn.execute(
            "SELECT key, not_counted FROM period_summary WHERE period_end = ? AND kind = 'sector'",
            (END1,),
        )
    }
    assert sectors == {"45": 1}


# --- the revision and the sectors ------------------------------------------------------------------------------------


def test_the_revision_follows_the_sector_of_every_company_but_not_which_have_snapshots(
    conn: db_module.NewsNlpDatabase,
) -> None:
    period.seed_sectors(conn)
    mapping = period.sector_map(conn)
    base = period.revision(SNAP, mapping)
    assert mapping == {"AAPL": "45", "MSFT": "45", "XOM": "10"} and base.startswith("period-")
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    assert (
        period.revision(SNAP, period.sector_map(conn)) == base
    )  # a new snapshot, a new company with news: the same revision
    conn.execute("INSERT INTO company_sector (ticker, sector_code) VALUES ('ZZZ', '90')")
    conn.commit()
    assert period.revision(SNAP, period.sector_map(conn)) != base  # an override is another revision
    assert period.revision("snapshot-other", mapping) != base


def test_a_new_period_revision_starts_clean_and_keeps_the_old_rows(
    conn: db_module.NewsNlpDatabase,
) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    first = _run(conn)
    conn.execute("INSERT INTO company_sector (ticker, sector_code) VALUES ('AAPL', '90')")
    conn.execute("INSERT INTO sector (code, name, kind) VALUES ('90', 'Cryptocurrency', 'extra')")
    conn.commit()
    second = _run(conn)
    assert (
        second["period_revision"] != first["period_revision"] and second["to_do"] == 4
    )  # from period 0 again
    assert conn.execute("SELECT COUNT(DISTINCT model_revision) FROM period_run").fetchone()[0] == 2
    new = conn.execute(
        "SELECT key FROM period_summary WHERE kind = 'sector' AND model_revision = ?",
        (second["period_revision"],),
    ).fetchall()
    assert [r["key"] for r in new] == ["90"]  # AAPL moved to the extra sector


def test_the_sector_table_is_seeded_from_the_constants_and_keeps_a_row_the_owner_added(
    conn: db_module.NewsNlpDatabase,
) -> None:
    period.seed_sectors(conn)
    rows = {r["code"]: r for r in conn.execute("SELECT * FROM sector")}
    assert {r["name"]: code for code, r in rows.items()} == SECTOR_CODES
    assert all(r["kind"] == "gics" for r in rows.values())
    assert all(json.loads(rows[c]["descriptions"]) == SECTOR_DESCRIPTIONS.get(c, []) for c in rows)
    conn.execute("INSERT INTO sector (code, name, kind) VALUES ('90', 'Cryptocurrency', 'extra')")
    conn.execute("UPDATE sector SET name = 'Tech' WHERE code = '45'")
    conn.commit()
    period.seed_sectors(conn)  # INSERT OR IGNORE: nothing is overwritten
    assert conn.execute("SELECT name FROM sector WHERE code = '45'").fetchone()[0] == "Tech"
    assert conn.execute("SELECT COUNT(*) FROM sector").fetchone()[0] == len(SECTOR_CODES) + 1


def test_a_reset_of_the_snapshots_clears_the_periods_built_on_them(
    conn: db_module.NewsNlpDatabase,
) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z", revision=snap.revision())
    period.run(conn, snap.revision(), tz=NY, now=AFTER_P3)
    other = period.run(
        conn, "snapshot-other", tz=NY, now=AFTER_P3
    )  # a period of another snapshot revision stays
    assert conn.execute("SELECT COUNT(*) FROM period_run").fetchone()[0] == 8
    result = snap.reset(conn, sectors=False)
    assert result["periods"] == 4 and result["company_rows"] == 1
    conn.commit()
    assert [
        r["snapshot_revision"]
        for r in conn.execute("SELECT DISTINCT snapshot_revision FROM period_run")
    ] == ["snapshot-other"]
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM period_summary WHERE model_revision != ?",
            (other["period_revision"],),
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM period_item WHERE model_revision != ?",
            (other["period_revision"],),
        ).fetchone()[0]
        == 0
    )


# --- the text and the command ----------------------------------------------------------------------------------------


def test_the_text_of_a_period_lists_companies_by_total_then_sectors_with_the_owners_line_format(
    conn: db_module.NewsNlpDatabase,
) -> None:
    for i in range(3):
        _add(conn, f"a{i}", "AAPL", "2022-01-05T10:00:00Z", "labor", "negative")
    _add(conn, "m", "MSFT", "2022-01-06T10:00:00Z", "legal_regulatory", "positive")
    result = _run(conn)
    text = period.render(conn, END1, result["period_revision"])
    lines = text.splitlines()
    assert lines[0].startswith("Period summary of 2022-01-24")
    assert (
        lines.index("AAPL:") < lines.index("MSFT:") < lines.index("Information Technology:")
    )  # by total, companies first
    assert "  labor: 3 total-0 neutral-0 positive-3 negative" in lines
    assert (
        "  legal_regulatory: 1 total-0 neutral-1 positive-0 negative" in lines
    )  # MSFT: one positive
    with pytest.raises(period.PeriodRefused, match="not written"):
        period.render(conn, "2030-01-01T00:00:00Z", result["period_revision"])


def _args(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "status": False,
        "snapshot_revision": SNAP,
        "period_end": None,
        "catch_up": False,
        "render": False,
    }
    return argparse.Namespace(**{**base, **kw})


def test_the_command_writes_a_period_prints_it_and_refuses_an_unknown_snapshot_revision(
    conn: db_module.NewsNlpDatabase,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(period, "_now", lambda: AFTER_P3)
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    cli = _load_cli()
    cli.main_period(_args(catch_up=True), conn)
    out = capsys.readouterr()
    assert "4 periods to do" in out.err and json.loads(out.out)["to_do"] == 4
    cli.main_period(_args(period_end="2022-01-24", render=True), conn)
    assert "AAPL:" in capsys.readouterr().out
    cli.main_period(_args(status=True), conn)
    status = json.loads(capsys.readouterr().out)
    assert len(status["periods_written"]) == 4 and status["periods_written"][1]["counted"] == 1
    cli.main_period(_args(status=True), conn)  # a status changes nothing
    with pytest.raises(SystemExit, match="no company snapshots at snapshot-nope"):
        cli.main_period(_args(snapshot_revision="snapshot-nope", catch_up=True), conn)
    with pytest.raises(SystemExit, match="not a period end"):
        cli.main_period(_args(period_end="2022-01-04"), conn)


@pytest.mark.parametrize(
    "argv",
    [
        ["--step", "period", "--catch-up"],  # no snapshot revision
        ["--step", "period", "--snapshot-revision", "s"],  # neither a date nor --catch-up
        [
            "--step",
            "period",
            "--snapshot-revision",
            "s",
            "--catch-up",
            "--period-end",
            "2022-01-03",
        ],
        ["--step", "period", "--snapshot-revision", "s", "--render"],  # nothing to render
        ["--step", "period", "--snapshot-revision", "s", "--catch-up", "--limit", "5"],
        ["--step", "period", "--snapshot-revision", "s", "--catch-up", "--reset"],
        ["--step", "snapshot", "--status", "--period-end", "2022-01-03"],
        [
            "--step",
            "sentiment",
            "--catch-up",
            "--summary-revision",
            "s",
            "--companies-revision",
            "c",
        ],
    ],
)
def test_the_options_of_the_period_step_are_required_and_refused_elsewhere(argv: list[str]) -> None:
    cli = _load_cli()
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        cli.check_options(parser, parser.parse_args(argv))


def test_the_status_needs_no_revision() -> None:
    cli = _load_cli()
    parser = cli.build_parser()
    cli.check_options(parser, parser.parse_args(["--step", "period", "--status"]))
    cli.check_options(
        parser,
        parser.parse_args(
            ["--step", "period", "--snapshot-revision", "s", "--period-end", "2022-01-03"]
        ),
    )


def test_a_revision_with_only_sector_texts_is_refused(
    conn: db_module.NewsNlpDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(period, "_now", lambda: AFTER_P3)
    _add(conn, "s", "", "2022-01-05T10:00:00Z", ticker_blank=True)
    with pytest.raises(SystemExit, match="no company snapshots"):
        _load_cli().main_period(_args(catch_up=True), conn)


def test_the_refusal_for_a_date_off_the_grid_names_the_nearest_period_ends() -> None:
    with pytest.raises(period.PeriodRefused, match="the nearest: 2022-01-24 and 2022-02-14"):
        period.index_of(date(2022, 2, 1))
    with pytest.raises(period.PeriodRefused, match="the nearest: 2022-01-03 and 2022-01-24"):
        period.index_of(date(2021, 12, 1))


def test_the_report_date_is_the_cut_and_the_text_does_not_show_the_time_of_the_run(
    conn: db_module.NewsNlpDatabase,
) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    result = _run(conn)
    text = period.render(conn, END1, result["period_revision"])
    assert (
        text.splitlines()[0]
        == f"Period summary of 2022-01-24  (cut {END1}, window ({END0}, {END1}])"
    )
    assert "created" not in text


def test_the_status_and_the_text_read_only_and_a_sector_name_comes_from_the_constants(
    conn: db_module.NewsNlpDatabase,
) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    _add(conn, "z", "ZZZ", "2022-01-05T11:00:00Z")
    result = _run(conn)  # the run seeds the sector table
    conn.execute("DELETE FROM sector")  # the reading must not need it, and must not fill it
    conn.commit()
    text = period.render(conn, END1, result["period_revision"])
    assert "Information Technology:" in text and "unclassified:" in text
    status = period.status(conn, SNAP, tz=NY, now=AFTER_P3)
    assert (
        status["missing"] == [] and conn.execute("SELECT COUNT(*) FROM sector").fetchone()[0] == 0
    )


def test_the_status_lists_the_periods_still_missing(conn: db_module.NewsNlpDatabase) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    _run(conn, AFTER_P1)  # periods 0 and 1 written
    status = period.status(conn, SNAP, tz=NY, now=AFTER_P3)
    assert status["missing"] == ["2022-02-14", "2022-03-07"]
    assert [p["period_end"] for p in status["periods_written"]] == [END0, END1]  # type: ignore[attr-defined,index]


def test_a_run_in_another_zone_cuts_at_its_own_15_00(conn: db_module.NewsNlpDatabase) -> None:
    bogota = ZoneInfo("America/Bogota")
    sao_paulo = ZoneInfo("America/Sao_Paulo")  # UTC-3: the cut is 18:00 UTC
    _add(conn, "in", "AAPL", "2022-01-03T18:00:00Z")  # exactly at the cut in Sao Paulo
    _add(conn, "out", "AAPL", "2022-01-03T18:00:01Z")
    result = period.run(conn, SNAP, tz=sao_paulo, now=datetime(2022, 1, 25, tzinfo=UTC))
    assert (
        result["periods"][0]["period_end"] == "2022-01-03T18:00:00Z"
        and result["periods"][0]["counted"] == 1
    )
    assert result["periods"][1]["counted"] == 1
    assert period.cut_utc(0, bogota) == "2022-01-03T20:00:00Z"


def test_the_not_counted_line_is_printed_only_when_above_zero(
    conn: db_module.NewsNlpDatabase,
) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    result = _run(conn)
    assert "not counted yet" not in period.render(conn, END1, result["period_revision"])
    conn.execute("UPDATE period_summary SET not_counted = 2 WHERE kind = 'company'")
    conn.commit()
    assert "  not counted yet: 2" in period.render(conn, END1, result["period_revision"])


def test_a_sentiment_outside_the_three_is_refused_not_counted_in_no_split(
    conn: db_module.NewsNlpDatabase,
) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z", sentiment="mixed")
    with pytest.raises(period.PeriodRefused, match="sentiment other than"):
        _run(conn)


def test_the_revision_hashes_the_counting_rule_not_the_bookkeeping() -> None:
    code = inspect.getsource(period._code)
    for name in (
        "_tally",
        "_counts_of",
        "_load_rows",
        "_write",
        "_gaps",
        "snap.articles_of",
        "snap.crawl_date_ok",
        "snap._body_hashes",
    ):
        assert name in code
    assert "(run," not in code


def test_the_run_shows_a_progress_bar_with_the_time_left(
    conn: db_module.NewsNlpDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    _run(conn)
    err = capsys.readouterr().err
    assert "period rows:" in err and "1/1" in err and "<" in err  # the count and the time left


def test_the_number_of_periods_to_do_is_printed_before_the_first_is_written(
    conn: db_module.NewsNlpDatabase,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(period, "_now", lambda: AFTER_P3)
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    _load_cli().main_period(_args(catch_up=True), conn)
    err = capsys.readouterr().err
    assert "4 periods to do" in err
    assert err.index("4 periods to do") < err.index(
        "period rows:"
    )  # the plan comes before the first bar update
    assert "period revision period-" in err


def test_the_bar_counts_the_rows_of_the_periods_not_the_periods(
    conn: db_module.NewsNlpDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    for i in range(5):
        _add(
            conn, f"old{i}", "AAPL", "2021-06-01T10:00:00Z"
        )  # all in the first window, with no lower bound
    _add(conn, "new", "AAPL", "2022-01-30T10:00:00Z")
    _run(conn)
    err = capsys.readouterr().err
    assert "6/6" in err  # six rows, not four periods
    assert "0/6" in err  # it starts at zero and the first period alone moves it by five


def test_the_recipe_to_write_a_period_again_names_the_three_tables_and_the_later_periods() -> None:
    cli = _load_cli()
    sql = "\n".join(cli.REPERIOD_SQL)
    for table in ("period_item", "period_summary", "period_run"):
        assert f"DELETE FROM {table} WHERE model_revision" in sql  # noqa: S608 -- a test of the text
    assert sql.count("period_end >=") == 3  # this period and every later one
    assert (
        "period_run WHERE model_revision" in cli.RERUN_HELP and "every later one" in cli.RERUN_HELP
    )


def test_the_window_follows_the_publication_date_and_the_crawl_date_only_when_there_is_none(
    conn: db_module.NewsNlpDatabase,
) -> None:
    def add(item: str, published: str, crawled: str) -> None:
        conn.execute(
            "INSERT INTO news_snapshot (item_kind, item_id, ticker, model_revision, fetched_at, pub_date, text, text_source, category, sentiment, inputs, processed_at) "
            "VALUES ('body', ?, 'AAPL', ?, ?, ?, 't', 'summary', 'labor', 'positive', '{}', 't')",
            (item, SNAP, crawled, published),
        )

    add(
        "backlog", "2022-01-05T10:00:00Z", "2026-08-20T10:00:00Z"
    )  # published in window 1, crawled four years later
    add("no_pub", "", "2022-01-06T10:00:00Z")  # no publication date: the crawl date places it
    conn.commit()
    _run(conn)
    counted = {
        r["item_id"]: r["period_end"]
        for r in conn.execute("SELECT item_id, period_end FROM period_item")
    }
    assert counted == {"backlog": END1, "no_pub": END1}


def test_the_not_counted_pairs_are_dated_by_publication_not_by_crawl(
    conn: db_module.NewsNlpDatabase,
) -> None:
    steps.record_companies(
        conn,
        steps.ITEM_BODY,
        triage.body_hash("Body of AAPL. " * 20),
        "companies-t",
        [steps.CompanyScore("AAPL", 1, 0.9)],
        0.5,
    )
    conn.execute(
        "INSERT INTO news_snapshot (item_kind, item_id, ticker, model_revision, fetched_at, pub_date, text, text_source, category, sentiment, inputs, processed_at) "
        "VALUES ('body', 'other', 'AAPL', ?, '2026-08-20T10:00:00Z', '2022-01-12T10:00:00Z', 't', 'summary', 'labor', 'positive', ?, 't')",
        (SNAP, json.dumps({"companies": "companies-t"})),
    )
    conn.commit()
    _run(conn)
    missing = conn.execute(
        "SELECT not_counted FROM period_summary WHERE period_end = ? AND kind = 'company' AND key = 'AAPL'",
        (END1,),
    ).fetchone()[0]
    assert (
        missing == 1
    )  # the fixture article is published 2022-01-10 and crawled the same day: window 1 either way


def test_the_text_says_how_many_of_the_counted_snapshots_are_late(
    conn: db_module.NewsNlpDatabase,
) -> None:
    _add(conn, "a", "AAPL", "2022-01-05T10:00:00Z")
    _run(conn, AFTER_P1)
    _add(
        conn, "late", "AAPL", "2022-01-20T10:00:00Z"
    )  # dated in window 1, snapshotted after it was written
    result = _run(conn, AFTER_P3)
    text = period.render(conn, END2, result["period_revision"])
    assert (
        "1 snapshots counted, of which late (dated before the window, snapshotted after it was written): 1"
        in text
    )
    first = period.render(conn, END1, result["period_revision"])
    assert "1 snapshots counted, of which late" in first and first.rstrip().splitlines()[
        1
    ].endswith(": 0")
