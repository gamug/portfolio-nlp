"""news_nlp.schema: the five result tables, `init_schema` idempotency, the
`sector_summary` self-heal migration, and the `fetch_pending_*` row shapes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from conftest import seed_article

import news_nlp as db
from news_nlp import schema

_RESULT_TABLES = {
    "article_sentiment",
    "article_entities",
    "article_summary",
    "sector_summary",
    "article_category",
}


def test_init_schema_creates_every_result_table(test_db_path: Path) -> None:
    conn = sqlite3.connect(test_db_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert names >= _RESULT_TABLES


def test_init_schema_is_idempotent(test_db_path: Path) -> None:
    conn = db.connect(test_db_path)
    db.init_schema(conn)  # second and third calls must not raise
    db.init_schema(conn)
    conn.close()


def test_migrate_sector_summary_schema_adds_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    # a pre-format_version sector_summary table, built the way the crawler /
    # an older release would have -- raw sqlite3 is fine for test setup.
    raw = sqlite3.connect(path)
    raw.executescript(
        """
        CREATE TABLE sector_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gics_sector TEXT NOT NULL, gics_sub_industry TEXT NOT NULL,
            week_start TEXT NOT NULL, week_end TEXT NOT NULL,
            summary_text TEXT NOT NULL, num_articles INTEGER NOT NULL,
            num_companies INTEGER NOT NULL, model_name TEXT NOT NULL,
            processed_at TEXT NOT NULL,
            UNIQUE (gics_sector, gics_sub_industry, week_start)
        );
        """
    )
    raw.commit()
    raw.close()

    conn = db.connect(path)
    schema._migrate_sector_summary_schema(conn)
    cols = set(conn.table_columns("sector_summary"))
    conn.close()
    assert {"format_version", "facts_json", "intro_text"} <= cols


def test_migrate_sector_summary_schema_noop_when_table_absent(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "empty.db")
    schema._migrate_sector_summary_schema(conn)  # must not raise (table absent)
    conn.close()


def test_fetch_pending_articles_unpacks_as_two_tuple(conn: sqlite3.Connection) -> None:
    seed_article(conn, id=1, body_text="Body text.")
    conn.commit()

    rows = db.fetch_pending_articles(conn, "article_sentiment")
    assert len(rows) == 1
    article_id, body_text = rows[0]
    assert article_id == 1
    assert body_text == "Body text."


def test_fetch_pending_category_articles_unpacks_as_three_tuple(conn: sqlite3.Connection) -> None:
    seed_article(conn, id=1, title="Test Title", body_text="Body text.")
    conn.commit()

    rows = db.fetch_pending_category_articles(conn)
    assert len(rows) == 1
    article_id, title, body_text = rows[0]
    assert article_id == 1
    assert title == "Test Title"
    assert body_text == "Body text."


def test_fetch_pending_articles_rejects_unknown_table(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="table must be one of"):
        db.fetch_pending_articles(conn, "article_category")
