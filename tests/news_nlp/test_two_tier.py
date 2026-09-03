"""The two-tier DB contract: a read-only SOURCE database (holds
``articles.body_text``) ATTACHed into a writable RESULTS store (result tables +
a lean ``articles`` subset). See docs/db-topology.md.
"""

import hashlib
import sqlite3
from pathlib import Path

import pytest
from conftest import LEAN_ARTICLES_SCHEMA
from fastapi.testclient import TestClient

import db
import pipeline

_SENTIMENT = ("neutral", 0.5, 0.3, 0.3, 0.4, "fake-model")
_ENTITY = {
    "entity_type": "ORG",
    "text": "Acme Corp",
    "start_char": 0,
    "end_char": 9,
    "score": 0.95,
}


# --- connect_pipeline / attach_source ------------------------------------------


def test_distinct_paths_attach_source_read_only(two_tier_conn: sqlite3.Connection) -> None:
    assert two_tier_conn.articles_rel == "source"  # type: ignore[attr-defined]
    schemas = {row[1] for row in two_tier_conn.execute("PRAGMA database_list")}
    assert "source" in schemas


def test_same_path_skips_attach(source_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A single physical file that has both body_text and the result tables:
    # SOURCE_DATABASE_URL == DATABASE_URL -> no ATTACH, articles_rel stays "main".
    setup = db.connect(source_db_path)
    db.init_schema(setup)
    setup.close()

    monkeypatch.setattr(db, "DB_PATH", source_db_path)
    monkeypatch.setattr(db, "SOURCE_DB_PATH", source_db_path)
    conn = db.connect_pipeline()
    try:
        assert conn.articles_rel == "main"
        schemas = {row[1] for row in conn.execute("PRAGMA database_list")}
        assert "source" not in schemas
    finally:
        conn.close()


def test_connect_pipeline_requires_source(
    results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(db, "DB_PATH", results_db_path)
    monkeypatch.setattr(db, "SOURCE_DB_PATH", None)
    with pytest.raises(RuntimeError, match="SOURCE_DATABASE_URL"):
        db.connect_pipeline()


# --- cross-database "pending" joins read SOURCE.articles ---------------------


def test_pending_articles_join_reads_source_and_excludes_done(
    two_tier_conn: sqlite3.Connection,
) -> None:
    # source_db_path seeds ids 1, 2, 3; mark id 1 done in RESULTS.
    db.write_sentiment(two_tier_conn, 1, *_SENTIMENT)
    two_tier_conn.commit()

    pending = [r["id"] for r in db.fetch_pending_articles(two_tier_conn, "article_sentiment")]
    assert pending == [2, 3]


def test_pending_category_join_carries_source_body_text(
    two_tier_conn: sqlite3.Connection,
) -> None:
    rows = db.fetch_pending_category_articles(two_tier_conn)
    assert {r["id"] for r in rows} == {1, 2, 3}
    assert all("Body text for article" in r["body_text"] for r in rows)


def test_pending_company_summary_join_carries_source_body_text(
    two_tier_conn: sqlite3.Connection,
) -> None:
    db.write_sentiment(two_tier_conn, 1, "positive", 0.9, 0.9, 0.05, 0.05, "fake-model")
    db.write_entities(two_tier_conn, 1, [_ENTITY], "fake-model")
    two_tier_conn.commit()

    rows = db.fetch_pending_company_summaries(two_tier_conn)
    assert [r["article_id"] for r in rows] == [1]
    assert "Body text for article 1" in rows[0]["body_text"]


# --- lean-article upsert into RESULTS keeps it foreign-key consistent -------


def test_result_write_upserts_lean_article_row(
    two_tier_conn: sqlite3.Connection, results_db_path: Path
) -> None:
    db.write_sentiment(two_tier_conn, 2, *_SENTIMENT)
    two_tier_conn.commit()

    check = sqlite3.connect(results_db_path)
    row = check.execute("SELECT ticker, gics_sector FROM articles WHERE id = 2").fetchone()
    cols = {r[1] for r in check.execute("PRAGMA table_info(articles)")}
    check.close()

    assert row == ("AAPL", "Information Technology")  # metadata copied from source
    assert "body_text" not in cols  # ... but not the heavy text column


def test_results_store_foreign_keys_consistent_after_writes(
    two_tier_conn: sqlite3.Connection, results_db_path: Path
) -> None:
    for i in (1, 2, 3):
        db.write_sentiment(two_tier_conn, i, *_SENTIMENT)
        db.write_entities(two_tier_conn, i, [_ENTITY], "fake-model")
        db.write_company_summary(two_tier_conn, i, "a summary", 1, "fake-model")
    two_tier_conn.commit()

    check = sqlite3.connect(results_db_path)
    problems = check.execute("PRAGMA foreign_key_check").fetchall()
    article_ids = {r[0] for r in check.execute("SELECT id FROM articles")}
    sentiment_ids = {r[0] for r in check.execute("SELECT article_id FROM article_sentiment")}
    check.close()

    assert problems == []
    assert article_ids == {1, 2, 3}
    assert sentiment_ids <= article_ids


# --- the SOURCE database is never written ----------------------------------


def test_source_database_is_never_written(
    two_tier_conn: sqlite3.Connection, source_db_path: Path
) -> None:
    before = hashlib.sha256(source_db_path.read_bytes()).hexdigest()

    for i in (1, 2, 3):
        db.write_sentiment(two_tier_conn, i, *_SENTIMENT)
        db.write_entities(two_tier_conn, i, [_ENTITY], "fake-model")
    db.write_company_summary(two_tier_conn, 1, "a summary", 1, "fake-model")
    two_tier_conn.commit()

    assert hashlib.sha256(source_db_path.read_bytes()).hexdigest() == before
    src = sqlite3.connect(source_db_path)
    with pytest.raises(sqlite3.OperationalError):
        src.execute("SELECT 1 FROM article_sentiment")
    src.close()


# --- preflight: fail loudly instead of the old silent "0 pending" ----------


def test_run_pipeline_raises_without_source(
    results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(db, "DB_PATH", results_db_path)
    monkeypatch.setattr(db, "SOURCE_DB_PATH", None)
    with pytest.raises(RuntimeError, match="SOURCE_DATABASE_URL"):
        pipeline.run_pipeline()


def test_run_pipeline_raises_when_source_lacks_body_text(
    results_db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lean_source = tmp_path / "lean_source.db"
    setup = sqlite3.connect(lean_source)
    setup.executescript(LEAN_ARTICLES_SCHEMA)
    setup.commit()
    setup.close()

    monkeypatch.setattr(db, "DB_PATH", results_db_path)
    monkeypatch.setattr(db, "SOURCE_DB_PATH", lean_source)
    with pytest.raises(RuntimeError, match="body_text"):
        pipeline.run_pipeline()


# --- serving needs no SOURCE ----------------------------------------------------


def test_serving_endpoints_work_without_a_source(
    results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = db.connect(results_db_path)
    seed.execute(
        "INSERT INTO articles (id, ticker, company, title, pub_date, gics_sector) "
        "VALUES (1, 'MMM', '3M', 'A title', '2023-01-01', 'Industrials')"
    )
    db.write_sentiment(seed, 1, "positive", 0.9, 0.9, 0.05, 0.05, "fake-model")
    seed.commit()
    seed.close()

    monkeypatch.setattr(db, "DB_PATH", results_db_path)
    monkeypatch.setattr(db, "SOURCE_DB_PATH", None)
    from apps.news_nlp_api import app  # noqa: PLC0415

    api = TestClient(app)
    for path in ("/articles", "/articles/1", "/stats/sentiment", "/stats/categories"):
        assert api.get(path).status_code == 200, path
    assert api.get("/sectors/summary").status_code == 200
