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

# news_nlp.eval run-log tables (see docs/evaluation.md)
_EVAL_TABLES = {"eval_run", "eval_judgement"}


def test_init_schema_creates_every_result_table(test_db_path: Path) -> None:
    conn = sqlite3.connect(test_db_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert names >= _RESULT_TABLES
    assert names >= _EVAL_TABLES


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


def test_fetch_pending_articles_sample_seed_is_reproducible_subset(
    conn: sqlite3.Connection,
) -> None:
    """A `sample_seed` draw is a genuine subset of the population (not the
    first-`limit`-by-id backlog order) and reproducible for a repeated seed
    -- see docs/evaluation.md's 2026-09-12 NER follow-up / PLAN.md Work item
    3 T-025 for why this exists."""
    for i in range(1, 21):
        seed_article(conn, id=i, body_text=f"Body text {i}.")
    conn.commit()

    a = db.fetch_pending_articles(conn, "article_sentiment", limit=5, sample_seed=1)
    b = db.fetch_pending_articles(conn, "article_sentiment", limit=5, sample_seed=1)
    assert len(a) == 5
    assert [r[0] for r in a] == [r[0] for r in b]  # same seed -> same ids, same order
    assert set(r[0] for r in a) <= set(range(1, 21))  # a genuine subset of the population

    plain = db.fetch_pending_articles(conn, "article_sentiment", limit=5)
    assert [r[0] for r in plain] == [1, 2, 3, 4, 5]  # the backlog-order path, for contrast

    other_seed = db.fetch_pending_articles(conn, "article_sentiment", limit=5, sample_seed=2)
    assert [r[0] for r in a] != [r[0] for r in other_seed]  # a different seed draws differently


def test_fetch_pending_articles_sample_seed_requires_limit(conn: sqlite3.Connection) -> None:
    seed_article(conn, id=1, body_text="Body text.")
    conn.commit()
    with pytest.raises(ValueError, match="sample_seed requires a positive limit"):
        db.fetch_pending_articles(conn, "article_sentiment", sample_seed=1)


def test_fetch_pending_articles_sample_seed_caps_at_population(
    conn: sqlite3.Connection,
) -> None:
    for i in range(1, 4):
        seed_article(conn, id=i, body_text=f"Body text {i}.")
    conn.commit()

    rows = db.fetch_pending_articles(conn, "article_sentiment", limit=1000, sample_seed=1)
    assert len(rows) == 3  # fewer than `limit` only when the store holds too few
