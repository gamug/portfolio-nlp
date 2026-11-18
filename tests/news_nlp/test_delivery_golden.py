"""The golden check of the delivery note (T-220, PR A): its queries, run on a fixture, and the rule the depuration must keep.

``delivery_golden.digest`` is equal before and after a drop of the v1 tables and a prune of the revisions that are not delivered, and it changes if
a row the note reads goes. No model, no network."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest

import delivery_golden as golden
from news_nlp import schema as schema_module

SNAP = "snapshot-c3a753affa5d"
PERIOD = "period-ae014b56b055"  # the delivery note's. The owner decided (2026-11-18) that only period-3b07b83ecec3 stays, so after the prune and a new note the check runs with that one and `ae014` returns no rows (expected, not a failure)
OLD_SNAP = "snapshot-0000000000aa"
OLD_PERIOD = "period-0000000000bb"
END = "2026-08-31T19:00:00Z"


def _snapshot(
    conn: sqlite3.Connection,
    revision: str,
    ticker: str,
    item: str,
    pub: str,
    category: str = "earnings_performance",
) -> None:
    conn.execute(
        "INSERT INTO news_snapshot (item_kind, item_id, ticker, model_revision, fetched_at, pub_date, title, text, text_source, category, sentiment, "
        "sentiment_score, inputs, processed_at) VALUES ('sub_news', ?, ?, ?, '2026-01-20T00:00:00Z', ?, ?, 'a text', 'summary', ?, 'positive', 0.9, "
        "'{\"date\": \"published\"}', '2026-11-01T00:00:00Z')",
        (item, ticker, revision, pub, f"title {item}", category),
    )


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "results.db"
    raw = sqlite3.connect(path)
    raw.executescript(schema_module.SCHEMA)
    for revision, period in ((SNAP, PERIOD), (OLD_SNAP, OLD_PERIOD)):
        _snapshot(raw, revision, "AAPL", "1", "2026-01-05T00:00:00Z")
        _snapshot(raw, revision, "MSFT", "2", "2026-01-06T00:00:00Z", "other")
        raw.execute(
            "INSERT INTO period_run VALUES (?, ?, ?, NULL, '2026-09-01')", (END, period, revision)
        )
        counts = json.dumps(
            {"earnings_performance": {"total": 2, "positive": 2, "negative": 0, "neutral": 0}}
        )
        raw.execute(
            "INSERT INTO period_summary VALUES (?, 'company', 'AAPL', ?, ?, 0)",
            (END, period, counts),
        )
        raw.execute(
            "INSERT INTO period_summary VALUES (?, 'sector', '45', ?, ?, 0)", (END, period, counts)
        )
        raw.execute(
            "INSERT INTO period_item VALUES (?, ?, 'sub_news', '1', 'AAPL', 0)", (END, period)
        )
    raw.execute(
        "INSERT OR IGNORE INTO sector (code, name, kind) VALUES ('45', 'Information Technology', 'gics')"
    )
    raw.commit()
    return raw


def _digest(conn: sqlite3.Connection) -> str:
    return golden.digest(golden.run_all(conn, SNAP, PERIOD))


def test_every_query_of_the_note_returns_rows_on_the_fixture(conn: sqlite3.Connection) -> None:
    results = golden.run_all(conn, SNAP, PERIOD)
    assert set(results) == set(golden.QUERIES)
    assert all(rows for rows in results.values()), [n for n, rows in results.items() if not rows]


def test_the_queries_read_only_the_tables_the_depuration_keeps() -> None:
    read = {
        t
        for sql in golden.QUERIES.values()
        for t in re.findall(r"(?:FROM|JOIN)\s+([a-z_]+)", sql)
        if t != "json_each"
    }
    assert read <= set(golden.TABLES), read - set(golden.TABLES)


def test_the_kept_tables_exist_in_the_schema(conn: sqlite3.Connection) -> None:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert set(golden.TABLES) <= names


def test_dropping_the_v1_tables_does_not_move_what_the_upstream_reads(
    conn: sqlite3.Connection,
) -> None:
    before = _digest(conn)
    conn.executescript(
        "DROP TABLE article_sentiment; DROP TABLE article_category; DROP TABLE article_entities; DROP TABLE article_summary; DROP TABLE sector_summary;"
    )
    assert _digest(conn) == before


def test_pruning_the_revisions_that_are_not_delivered_does_not_move_it(
    conn: sqlite3.Connection,
) -> None:
    before = _digest(conn)
    for table in ("period_item", "period_summary", "period_run"):
        conn.execute(f"DELETE FROM {table} WHERE model_revision = ?", (OLD_PERIOD,))  # noqa: S608 -- fixed table names
    conn.execute("DELETE FROM news_snapshot WHERE model_revision = ?", (OLD_SNAP,))
    assert _digest(conn) == before


@pytest.mark.parametrize(
    "delete",
    [
        "DELETE FROM news_snapshot WHERE ticker = 'AAPL' AND model_revision = 'snapshot-c3a753affa5d'",
        "DELETE FROM period_summary WHERE model_revision = 'period-ae014b56b055' AND kind = 'sector'",
        "DELETE FROM period_run WHERE model_revision = 'period-ae014b56b055'",
        "DELETE FROM sector",
    ],
)
def test_losing_a_row_the_note_reads_changes_the_digest(
    conn: sqlite3.Connection, delete: str
) -> None:
    before = _digest(conn)
    conn.execute(delete)
    assert _digest(conn) != before


def test_the_command_prints_each_query_and_the_digest_and_writes_nothing(
    conn: sqlite3.Connection, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    conn.close()
    path = tmp_path / "results.db"
    before = path.stat().st_mtime_ns
    assert (
        golden.main(["--db", str(path), "--snapshot", SNAP, "--period", PERIOD, "--immutable"]) == 0
    )
    out = capsys.readouterr().out
    assert all(f"{name}: " in out for name in golden.QUERIES)
    assert "digest: " in out
    assert path.stat().st_mtime_ns == before  # read-only: the file was not touched
    assert not list(tmp_path.glob("results.db-*"))  # no -wal / -shm / -journal
