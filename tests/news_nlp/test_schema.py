"""news_nlp.schema: the 20 tables `init_schema` creates (the lean `articles` among them, T-220 PR C2 round 6), its idempotency, that it brings no v1
table back (T-220), the fresh-file first lean-article row, and the `eval_run` additive migration.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import news_nlp as db
from news_nlp import db as news_db

#: Exactly what ``init_schema`` creates (T-220, PR C2): the lean ``articles`` table, the stepped pipeline's tables and the eval run-log tables. The v1 result tables are not created any more.
_KEPT_TABLES = {
    "articles",
    "body_scope",
    "body_sector",
    "sub_news",
    "news_summary",
    "news_summary_call",
    "news_company",
    "news_sentiment",
    "news_category",
    "news_snapshot",
    "period_run",
    "period_summary",
    "period_item",
    "sector",
    "company_sector",
    "pipeline_step_state",
    "eval_run",
    "eval_inference",
    "eval_verdict",
    "eval_confusion",
}
_V1_TABLES = {
    "article_sentiment",
    "article_entities",
    "article_category",
    "article_summary",
    "sector_summary",
    "eval_judgement",
}


def _tables(path: Path) -> set[str]:
    raw = sqlite3.connect(path)
    names = {
        r[0]
        for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if not r[0].startswith("sqlite_")
    }
    raw.close()
    return names


def test_init_schema_creates_exactly_the_kept_tables_on_an_empty_file(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "fresh.db")
    db.init_schema(conn)
    conn.close()
    assert _tables(tmp_path / "fresh.db") == _KEPT_TABLES


def test_init_schema_does_not_bring_the_v1_tables_back_on_a_purged_database(tmp_path: Path) -> None:
    """The first status run after the live purge re-created the six empty v1 tables (``init_schema`` ran the old DDL): this is what C2 stops."""
    path = tmp_path / "purged.db"
    conn = db.connect(path)
    for _ in range(3):
        db.init_schema(conn)
    conn.close()
    assert not _V1_TABLES & _tables(path)


def test_init_schema_leaves_a_v1_table_that_is_still_there_alone(tmp_path: Path) -> None:
    """It neither creates nor drops them: ``depurate_db`` is the only thing that drops."""
    path = tmp_path / "old.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE article_sentiment (article_id INTEGER, label TEXT)")
    raw.execute("INSERT INTO article_sentiment VALUES (1, 'positive')")
    raw.commit()
    raw.close()
    conn = db.connect(path)
    db.init_schema(conn)
    conn.close()
    kept = sqlite3.connect(path).execute("SELECT COUNT(*) FROM article_sentiment").fetchone()[0]
    assert kept == 1


def test_init_schema_is_idempotent(test_db_path: Path) -> None:
    conn = db.connect(test_db_path)
    db.init_schema(conn)  # second and third calls must not raise
    db.init_schema(conn)
    conn.close()


_PIPELINE_TABLES = {
    "body_scope",
    "sub_news",
    "news_summary",
    "news_company",
    "news_sentiment",
    "news_category",
    "news_snapshot",
    "pipeline_step_state",
}


def test_init_schema_creates_the_stepped_pipeline_tables(test_db_path: Path) -> None:
    conn = sqlite3.connect(test_db_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert names >= _PIPELINE_TABLES


def test_each_new_table_keeps_one_row_for_the_same_key(test_db_path: Path) -> None:
    conn = sqlite3.connect(test_db_path)
    rows = {
        "body_scope": "('h', 'r', 'single', '{}', 'title', 't')",
        "news_summary": "('body', 'h', 'r', 's', 'summary', 1, 't')",
        "news_company": "('body', 'h', 'AAPL', 'r', 0.9, 1, 3, 't')",
        "news_sentiment": "('body', 'h', '', 'r', 'positive', 0.9, 0.9, 0.05, 0.05, 'm', 0, 't')",
        "news_category": "('body', 'h', '', 'r', 'other', 0.5, '', 0.0, '{}', 'm', 't')",
        "news_snapshot": "('body', 'h', 'AAPL', 'r', '2022-01-02T09:00:00Z', '', 'T', 'x', 'summary', '', '[]', 'other', '', 'positive', 0.9, '{}', 't')",
        "pipeline_step_state": "('body', 'h', 'scope', 'r', 'done', 1, NULL, NULL, NULL)",
    }
    for table, values in rows.items():
        conn.execute(f"INSERT OR IGNORE INTO {table} VALUES {values}")  # noqa: S608 -- fixed test table names
        conn.execute(f"INSERT OR IGNORE INTO {table} VALUES {values}")  # noqa: S608 -- fixed test table names
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1, table  # noqa: S608
    conn.close()


def test_the_status_column_refuses_an_unknown_status(test_db_path: Path) -> None:
    conn = sqlite3.connect(test_db_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO pipeline_step_state VALUES ('body','h','scope','r','weird',0,NULL,NULL,NULL)"
        )
    conn.close()


def test_the_join_keys_are_indexed_and_the_flags_are_checked(test_db_path: Path) -> None:
    conn = sqlite3.connect(test_db_path)
    indexes = {r[1] for r in conn.execute("SELECT * FROM sqlite_master WHERE type='index'")}
    assert {
        "idx_sub_news_article_id",
        "idx_news_company_ticker",
    } <= indexes
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO news_company VALUES ('body','h','AAPL','r',0.9,2,1,'t')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO body_scope VALUES ('h','r','market','{}','title','t')")
    conn.close()


def _source_with_one_article(path: Path) -> Path:
    """A minimal SOURCE (the crawl database) that the pipeline connection accepts: schema version 5 and the columns it reads."""
    raw = sqlite3.connect(path)
    raw.execute(
        "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, body_text TEXT, pub_date TEXT, "
        "ticker TEXT, company TEXT, fetch_status TEXT, url TEXT, source TEXT, retrieved_at TEXT, extractor TEXT)"
    )
    raw.execute(
        "INSERT INTO articles VALUES (1, 't', 'body text', '2026-01-01', NULL, NULL, 'ok', NULL, NULL, NULL, NULL)"
    )
    raw.execute("PRAGMA user_version=5")
    raw.commit()
    raw.close()
    return path


def test_a_fresh_results_file_takes_its_first_lean_article_row(tmp_path: Path) -> None:
    source = _source_with_one_article(tmp_path / "urls.db")
    results = tmp_path / "nlp.db"
    conn = db.connect(results)
    db.init_schema(conn)
    conn.close()
    pipeline = db.connect_pipeline(results, source)
    try:
        news_db._ensure_article_row(pipeline, 1)
        assert pipeline.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1
    finally:
        pipeline.close()
