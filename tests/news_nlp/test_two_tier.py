"""Two-tier DB behaviours that run through `pipeline.run_pipeline` / the FastAPI
app. The pure connection/attach/query cases live in test_db.py, alongside the
rest of the vendored news_nlp package's own unit tests.
"""

import sqlite3
from pathlib import Path

import pytest
from conftest import LEAN_ARTICLES_SCHEMA, seed_article, stamp_source
from fastapi.testclient import TestClient

import news_nlp as db
import pipeline
from news_nlp.taxonomy import CATEGORY_SLUGS


def test_run_pipeline_raises_without_source(
    results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.delenv("SOURCE_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="SOURCE_DATABASE_URL"):
        pipeline.run_pipeline()


def test_run_pipeline_raises_when_source_lacks_body_text(
    results_db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lean_source = tmp_path / "lean_source.db"
    setup = sqlite3.connect(lean_source)
    setup.executescript(LEAN_ARTICLES_SCHEMA)
    stamp_source(setup)
    setup.commit()
    setup.close()

    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.setenv("SOURCE_DATABASE_URL", str(lean_source))
    with pytest.raises(RuntimeError, match="body_text"):
        pipeline.run_pipeline()


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

    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.delenv("SOURCE_DATABASE_URL", raising=False)
    from apps.news_nlp_api import app  # noqa: PLC0415

    api = TestClient(app)
    for path in ("/articles", "/articles/1", "/stats/sentiment", "/stats/categories"):
        assert api.get(path).status_code == 200, path
    assert api.get("/sectors/summary").status_code == 200


def _results_query(results_db_path: Path, sql: str) -> list[tuple]:
    results = sqlite3.connect(results_db_path)
    try:
        return results.execute(sql).fetchall()
    finally:
        results.close()


def _processed(results_db_path: Path, table: str) -> set[int]:
    rows = _results_query(results_db_path, f"SELECT DISTINCT article_id FROM {table}")  # noqa: S608
    return {r[0] for r in rows}


def _row_count(results_db_path: Path, table: str) -> int:
    return int(_results_query(results_db_path, f"SELECT COUNT(*) FROM {table}")[0][0])  # noqa: S608


def _stub_stages(monkeypatch: pytest.MonkeyPatch, writer: sqlite3.Connection, late_id: int) -> None:
    """Replace the three model stages with model-free stubs that use the exact
    pending query its production stage calls (fetch_pending_sentiment_articles /
    fetch_pending_articles / fetch_pending_category_articles -- the real SOURCE
    read) and the real RESULTS writer. The collector (`writer`) commits a late
    article right after the sentiment stage read its pending list but before the
    later stages read theirs."""

    def sentiment_stage(conn: db.NewsNlpDatabase, **_: object) -> None:
        for row in db.fetch_pending_sentiment_articles(conn):
            db.write_sentiment(conn, row["id"], "neutral", 0.5, 0.3, 0.3, 0.4, "fake-model")
        conn.commit()
        if not writer.execute("SELECT 1 FROM articles WHERE id = ?", (late_id,)).fetchone():
            seed_article(writer, id=late_id, ticker="NEW", company="New Co", body_text="Late body.")
            writer.commit()

    entity = {"entity_type": "ORG", "text": "Acme", "start_char": 0, "end_char": 4, "score": 0.9}

    def ner_stage(conn: db.NewsNlpDatabase, **_: object) -> None:
        for row in db.fetch_pending_articles(conn, "article_entities"):
            db.write_entities(conn, row["id"], [entity], "fake-model")
        conn.commit()

    def category_stage(conn: db.NewsNlpDatabase, **_: object) -> None:
        scores = dict.fromkeys(CATEGORY_SLUGS, 0.0)
        for row in db.fetch_pending_category_articles(conn):
            db.write_category(conn, row["id"], "other", 0.5, scores, "fake-model")
        conn.commit()

    monkeypatch.setattr(pipeline, "run_sentiment_stage", sentiment_stage)
    monkeypatch.setattr(pipeline, "run_ner_stage", ner_stage)
    monkeypatch.setattr(pipeline, "run_category_stage", category_stage)


def test_run_pipeline_runs_against_a_live_writers_source(
    source_db_path: Path, results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The scenario the portfolio-common v1.2.2 bump exists for, end to end: a
    # collector still holds SOURCE open in WAL mode (non-empty -wal + live
    # -shm) while the whole pipeline runs (stages stubbed, see _stub_stages).
    writer = sqlite3.connect(source_db_path)
    assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.commit()
    articles_sql = "SELECT * FROM articles ORDER BY id"
    schema_sql = "SELECT type, name, sql FROM sqlite_master ORDER BY name"
    source_before = writer.execute(articles_sql).fetchall()
    schema_before = writer.execute(schema_sql).fetchall()
    late_id = 1000
    _stub_stages(monkeypatch, writer, late_id)
    monkeypatch.setenv("DATABASE_URL", str(results_db_path))
    monkeypatch.setenv("SOURCE_DATABASE_URL", str(source_db_path))

    try:
        assert Path(f"{source_db_path}-shm").exists()
        pipeline.run_pipeline()

        # The attach is NOT a frozen snapshot (observed SQLite connection
        # behaviour, not a contract this repo promises: wrapping a run in one
        # read transaction would make the late article invisible to every
        # stage and this assertion would then need to change). So the late
        # article reached the later stages but missed sentiment ...
        assert _processed(results_db_path, "article_sentiment") == {1, 2, 3}
        assert _processed(results_db_path, "article_entities") == {1, 2, 3, late_id}
        assert _processed(results_db_path, "article_category") == {1, 2, 3, late_id}

        # ... and the next run converges: sentiment picks it up, and the
        # finished stages write nothing again (no duplicate/rewritten rows).
        tables = ("article_entities", "article_category")
        counts = {t: _row_count(results_db_path, t) for t in tables}
        pipeline.run_pipeline()
        assert _processed(results_db_path, "article_sentiment") == {1, 2, 3, late_id}
        assert _row_count(results_db_path, "article_sentiment") == 4
        assert {t: _row_count(results_db_path, t) for t in tables} == counts

        # SOURCE is never written by the pipeline: only the collector's own row
        # was added, and its schema is untouched. (That the attach itself is
        # read-only is asserted separately below.)
        source_after = writer.execute(articles_sql).fetchall()
        assert source_after[: len(source_before)] == source_before
        assert [r[0] for r in source_after[len(source_before) :]] == [late_id]
        assert writer.execute(schema_sql).fetchall() == schema_before
    finally:
        writer.close()


def test_source_attach_is_read_only_while_a_writer_holds_it_open(
    source_db_path: Path, results_db_path: Path
) -> None:
    # The read-only guarantee itself, not inferred from "the pipeline never
    # wrote": the attach (mode=ro) refuses writes even with a live writer.
    writer = sqlite3.connect(source_db_path)
    assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.commit()
    try:
        conn = db.connect_pipeline(results_db=results_db_path, source_db=source_db_path)
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute("DELETE FROM source.articles")
        finally:
            db.detach_source(conn)
            conn.close()
    finally:
        writer.close()
