"""The news_nlp two-tier DB contract: a read-only SOURCE database (holds
``articles.body_text``) ATTACHed into a writable RESULTS store (result tables +
a lean ``articles`` subset). See docs/db-topology.md.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest
from conftest import LEAN_ARTICLES_SCHEMA, stamp_source

import news_nlp as db
from news_nlp import db as news_db
from news_nlp import steps

_SENTIMENT = ("neutral", 0.5, 0.3, 0.3, 0.4, "fake-model")
_ENTITY = {
    "entity_type": "ORG",
    "text": "Acme Corp",
    "start_char": 0,
    "end_char": 9,
    "score": 0.95,
}


# --- connect_pipeline / attach_source ------------------------------------------


def test_distinct_paths_attach_source_read_only(two_tier_conn: db.NewsNlpDatabase) -> None:
    assert two_tier_conn.articles_rel == "source"
    schemas = {row[1] for row in two_tier_conn.execute("PRAGMA database_list")}
    assert "source" in schemas


def test_same_path_skips_attach(source_db_path: Path) -> None:
    # A single physical file that has both body_text and the result tables:
    # SOURCE == RESULTS -> no ATTACH, articles_rel stays "main".
    setup = db.connect(source_db_path)
    db.init_schema(setup)
    setup.close()

    conn = db.connect_pipeline(results_db=source_db_path, source_db=source_db_path)
    try:
        assert conn.articles_rel == "main"
        schemas = {row[1] for row in conn.execute("PRAGMA database_list")}
        assert "source" not in schemas
    finally:
        conn.close()


def test_connect_pipeline_requires_source(
    results_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SOURCE_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="SOURCE_DATABASE_URL"):
        db.connect_pipeline(results_db=results_db_path)


# --- cross-database "pending" joins read SOURCE.articles ---------------------


# --- lean-article upsert into RESULTS keeps it foreign-key consistent -------


def test_cut_writes_the_lean_article_row_without_body_text(
    two_tier_conn: db.NewsNlpDatabase, results_db_path: Path
) -> None:
    body = steps.Body(body_hash="h2", article_id=2, title="t", text="x")
    steps.record_cut(two_tier_conn, body, "units-test", ["A sub-news."])

    check = sqlite3.connect(results_db_path)
    row = check.execute("SELECT ticker, gics_sector FROM articles WHERE id = 2").fetchone()
    cols = {r[1] for r in check.execute("PRAGMA table_info(articles)")}
    check.close()
    assert row == ("AAPL", "Information Technology")  # metadata copied from SOURCE
    assert "body_text" not in cols  # ... but not the heavy text column


def test_lean_article_row_is_not_duplicated_and_the_foreign_keys_hold(
    two_tier_conn: db.NewsNlpDatabase, results_db_path: Path
) -> None:
    for i in (1, 2, 3):
        body = steps.Body(body_hash=f"h{i}", article_id=i, title="t", text="x")
        steps.record_cut(two_tier_conn, body, "units-test", ["One.", "Two."])
        steps.record_cut(two_tier_conn, body, "units-other", ["Three."])  # same article again
    news_db._ensure_article_row(two_tier_conn, 1)  # a direct repeat is a no-op
    two_tier_conn.commit()

    check = sqlite3.connect(results_db_path)
    ids = [r[0] for r in check.execute("SELECT id FROM articles ORDER BY id")]
    problems = check.execute("PRAGMA foreign_key_check").fetchall()
    check.close()
    assert ids == [1, 2, 3]
    assert problems == []


def test_a_body_with_no_text_writes_no_article_row(
    two_tier_conn: db.NewsNlpDatabase, results_db_path: Path
) -> None:
    body = steps.Body(body_hash="h1", article_id=1, title="t", text="x")
    steps.record_cut(two_tier_conn, body, "units-test", [])
    two_tier_conn.commit()
    check = sqlite3.connect(results_db_path)
    count = check.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    check.close()
    assert count == 0


# --- the SOURCE database is never written ----------------------------------


def test_source_database_is_never_written(
    two_tier_conn: db.NewsNlpDatabase, source_db_path: Path
) -> None:
    before = hashlib.sha256(source_db_path.read_bytes()).hexdigest()
    for i in (1, 2, 3):
        two_tier_conn.execute(
            "INSERT INTO pipeline_step_state (item_kind, item_id, step, model_revision, status) VALUES ('body', ?, 'scope', 'r', 'done')",
            (str(i),),
        )
    two_tier_conn.commit()
    assert hashlib.sha256(source_db_path.read_bytes()).hexdigest() == before
    src = sqlite3.connect(source_db_path)
    with pytest.raises(sqlite3.OperationalError):
        src.execute("SELECT 1 FROM pipeline_step_state")
    src.close()


# --- require_source_text preflight ---------------------------------------------


def test_require_source_text_raises_when_source_lacks_body_text(
    results_db_path: Path, tmp_path: Path
) -> None:
    # A stamped SOURCE without `body_text` is now stopped by the schema guard at
    # connect_pipeline (the column guard); require_source_text still covers the
    # single-file case, where nothing is attached and the guard does not run.
    lean_source = tmp_path / "lean_source.db"
    setup = sqlite3.connect(lean_source)
    setup.executescript(LEAN_ARTICLES_SCHEMA)
    stamp_source(setup)
    setup.commit()
    setup.close()

    with pytest.raises(db.SourceSchemaError, match="body_text"):
        db.connect_pipeline(results_db=results_db_path, source_db=lean_source)

    single = db.connect(lean_source)
    try:
        with pytest.raises(RuntimeError, match="body_text"):
            db.require_source_text(single)
    finally:
        single.close()


def test_require_source_text_passes_on_a_populated_source(
    two_tier_conn: db.NewsNlpDatabase,
) -> None:
    db.require_source_text(two_tier_conn)  # source has non-empty body_text -> no raise


# --- orphaned-WAL preflight ---------------------------------------------------
#
# portfolio-common v1.2.2 narrowed `Database.attach(read_only=True)`'s preflight
# to reject only an *orphaned* WAL (non-empty -wal, no -shm) -- not a live
# writer's WAL. Before that fix, this exact scenario (a SOURCE db held open by
# another process, e.g. portfolio-data-mining's collector) made attach_source()
# raise for as long as the writer was running, which was the production bug
# this bump exists to pick up. See portfolio-common CHANGELOG v1.2.2.


def test_attach_source_allows_a_live_writers_wal(
    source_db_path: Path, results_db_path: Path
) -> None:
    # A real writer keeps the SOURCE db open in WAL mode with auto-checkpoint
    # off, so the -wal stays non-empty and the -shm stays present the whole
    # time -- the exact shape a concurrent portfolio-data-mining collector run
    # leaves behind. attach_source() must not be blocked by this.
    holder = sqlite3.connect(source_db_path)
    assert holder.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    holder.execute("PRAGMA wal_autocheckpoint=0")
    holder.execute("INSERT INTO articles (id, ticker, company) VALUES (999, 'ZZZ', 'Z Co')")
    holder.commit()
    try:
        assert Path(f"{source_db_path}-wal").stat().st_size > 0
        assert Path(f"{source_db_path}-shm").exists()

        conn = db.connect_pipeline(results_db=results_db_path, source_db=source_db_path)
        try:
            assert conn.articles_rel == "source"
            # The original bug was reads being blocked, not just ATTACH
            # raising -- prove a real cross-database read against `source`
            # actually works while the writer still holds it open.
            rows = conn.execute(
                "SELECT id, body_text FROM source.articles WHERE id < 999 ORDER BY id"
            ).fetchall()
            assert {r["id"] for r in rows} == {1, 2, 3}
            assert all("Body text for article" in r["body_text"] for r in rows)
        finally:
            db.detach_source(conn)
            conn.close()
    finally:
        holder.close()


def test_attach_source_rejects_an_orphaned_wal(
    source_db_path: Path, results_db_path: Path, tmp_path: Path
) -> None:
    # A -wal copied without its -shm, e.g. a backup/rsync that skipped the
    # (SQLite docs call it disposable) -shm file: no process has *this copy*
    # open, so its WAL can't be trusted as a consistent snapshot. Still
    # rejected -- the one real danger this preflight does catch.
    holder = sqlite3.connect(source_db_path)
    assert holder.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    holder.execute("PRAGMA wal_autocheckpoint=0")
    holder.execute("INSERT INTO articles (id, ticker, company) VALUES (999, 'ZZZ', 'Z Co')")
    holder.commit()
    try:
        assert Path(f"{source_db_path}-wal").stat().st_size > 0

        # Simulate the backup: copy the db + -wal while the writer is still
        # open, deliberately not -shm.
        orphaned = tmp_path / "orphaned_source.db"
        shutil.copyfile(source_db_path, orphaned)
        shutil.copyfile(f"{source_db_path}-wal", f"{orphaned}-wal")
        assert not Path(f"{orphaned}-shm").exists()

        with pytest.raises(RuntimeError, match="orphaned write-ahead log"):
            db.connect_pipeline(results_db=results_db_path, source_db=orphaned)
    finally:
        holder.close()


def test_documented_checkpoint_recipe_clears_an_orphaned_wal(
    source_db_path: Path, results_db_path: Path, tmp_path: Path
) -> None:
    # The operator loop docs/db-topology.md ("Working with a live SOURCE")
    # promises: an orphaned copy is rejected, the documented checkpoint
    # statement (run here exactly as the doc gives it) clears it, and the
    # attach then works and reads the writer's committed row. Guards that
    # recipe against rotting on a portfolio-common / SQLite change.
    holder = sqlite3.connect(source_db_path)
    assert holder.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    holder.execute("PRAGMA wal_autocheckpoint=0")
    holder.execute("INSERT INTO articles (id, ticker, company) VALUES (999, 'ZZZ', 'Z Co')")
    holder.commit()
    try:
        orphaned = tmp_path / "orphaned_source.db"
        shutil.copyfile(source_db_path, orphaned)
        shutil.copyfile(f"{source_db_path}-wal", f"{orphaned}-wal")
        assert not Path(f"{orphaned}-shm").exists()

        with pytest.raises(RuntimeError, match="orphaned write-ahead log"):
            db.connect_pipeline(results_db=results_db_path, source_db=orphaned)

        busy, _, _ = sqlite3.connect(orphaned).execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        assert busy == 0

        conn = db.connect_pipeline(results_db=results_db_path, source_db=orphaned)
        try:
            row = conn.execute("SELECT ticker FROM source.articles WHERE id = 999").fetchone()
            assert row is not None and row["ticker"] == "ZZZ"
        finally:
            db.detach_source(conn)
            conn.close()
    finally:
        holder.close()
