"""The purge of the results database (T-220, PR C): what it drops and deletes, the gates that stop it, and what it must not move. A fixture database; no model, no network."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from conftest import LEAN_ARTICLES_SCHEMA

import delivery_golden
import depurate_db as dd
from news_nlp import schema as schema_module

OLD = "-0000000000aa"
END = "2026-08-31T19:00:00Z"
KEEP_SNAP, KEEP_PERIOD = dd.GOLDEN


def _insert(conn: sqlite3.Connection, sql: str, *rows: tuple[object, ...]) -> None:
    conn.executemany(sql, rows)


def build(path: Path) -> Path:
    """A small corpus: every kept revision and an old one for each table, the v1 tables with a row, the untouched tables with a row."""
    c = sqlite3.connect(path)
    c.executescript(LEAN_ARTICLES_SCHEMA)
    c.executescript(schema_module.SCHEMA)
    _insert(c, "INSERT INTO articles (id, title) VALUES (?, ?)", (1, "a"), (2, "b"), (3, "c"))
    now = "2026-11-01T00:00:00Z"
    k = dict(
        zip(
            (
                "scope",
                "units",
                "summary",
                "companies",
                "sentiment",
                "category",
                "snapshot",
                "period",
                "sector",
            ),
            dd.KEEP,
            strict=True,
        )
    )
    _insert(
        c,
        "INSERT INTO sub_news (id, body_hash, article_id, segment_index, text, extractor_revision, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, "h1", 1, 0, "text one", k["units"], now),
        (2, "h2", 2, 0, "text two", "units-5f831c298bcb", now),
    )
    _insert(
        c,
        "INSERT INTO body_scope (body_hash, classifier_revision, scope, text_used, processed_at) VALUES (?, ?, 'single', 't', ?)",
        ("h1", k["scope"], now),
        ("h2", "scope-nli" + OLD, now),
    )
    _insert(
        c,
        "INSERT INTO body_sector (body_hash, classifier_revision, status, candidate, margin, processed_at) VALUES (?, ?, 'assigned', '45', 0.5, ?)",
        ("h1", k["sector"], now),
        ("h2", "sector" + OLD, now),
    )
    _insert(
        c,
        "INSERT INTO news_summary (item_kind, item_id, model_revision, summary_text, text_source, processed_at) VALUES ('sub_news', ?, ?, 's', 'summary', ?)",
        ("1", k["summary"], now),
        ("2", "summary" + OLD, now),
    )
    _insert(
        c,
        "INSERT INTO news_summary_call (item_kind, item_id, model_revision, api_model, model_asked, prompt_tokens, completion_tokens, called_at) "
        "VALUES ('sub_news', ?, ?, 'm', 'm', 1, 1, ?)",
        ("1", k["summary"], now),
        ("2", "summary" + OLD, now),
    )
    _insert(
        c,
        "INSERT INTO news_company (item_kind, item_id, ticker, model_revision, relevance_score, processed_at) VALUES ('sub_news', ?, 'AAPL', ?, 0.9, ?)",
        ("1", k["companies"], now),
        ("2", "companies" + OLD, now),
    )
    _insert(
        c,
        "INSERT INTO news_sentiment (item_kind, item_id, ticker, model_revision, label, score, positive, negative, neutral, model_name, processed_at) "
        "VALUES ('sub_news', ?, 'AAPL', ?, 'positive', 0.9, 0.9, 0.05, 0.05, 'm', ?)",
        ("1", k["sentiment"], now),
        ("2", "sentiment" + OLD, now),
    )
    _insert(
        c,
        "INSERT INTO news_category (item_kind, item_id, ticker, model_revision, label, score, model_name, processed_at) VALUES ('sub_news', ?, 'AAPL', ?, 'other', 0.9, 'm', ?)",
        ("1", k["category"], now),
        ("2", "category" + OLD, now),
    )
    inputs = json.dumps(
        {
            "summary": "summary" + OLD,
            "companies": k["companies"],
            "cut": k["units"],
            "category": k["category"],
            "sentiment": "sentiment" + OLD,
            "sector": "",
        }
    )
    for item, rev in (("1", k["snapshot"]), ("2", "snapshot" + OLD)):
        c.execute(
            "INSERT INTO news_snapshot (item_kind, item_id, ticker, model_revision, fetched_at, pub_date, title, text, text_source, category, sentiment, sentiment_score, inputs, "
            "processed_at) VALUES ('sub_news', ?, 'AAPL', ?, '2026-01-20T00:00:00Z', '2026-01-05T00:00:00Z', 't', 'x', 'summary', 'other', 'positive', 0.9, ?, ?)",
            (item, rev, inputs, now),
        )
    counts = json.dumps({"other": {"total": 1, "positive": 1, "negative": 0, "neutral": 0}})
    for period, snap in ((k["period"], k["snapshot"]), ("period" + OLD, "snapshot" + OLD)):
        c.execute(
            "INSERT INTO period_run (period_end, model_revision, snapshot_revision, period_start, created_at) VALUES (?, ?, ?, NULL, ?)",
            (END, period, snap, now),
        )
        c.execute(
            "INSERT INTO period_summary (period_end, kind, key, model_revision, counts, not_counted) VALUES (?, 'company', 'AAPL', ?, ?, 0)",
            (END, period, counts),
        )
        c.execute(
            "INSERT INTO period_summary (period_end, kind, key, model_revision, counts, not_counted) VALUES (?, 'sector', '45', ?, ?, 0)",
            (END, period, counts),
        )
        c.execute(
            "INSERT INTO period_item (period_end, model_revision, item_kind, item_id, ticker, late) VALUES (?, ?, 'sub_news', '1', 'AAPL', 0)",
            (END, period),
        )
    for rev in (*dd.KEEP, "units-5f831c298bcb", "summary" + OLD):
        c.execute(
            "INSERT INTO pipeline_step_state (item_kind, item_id, step, model_revision, status) VALUES ('sub_news', '1', 's', ?, 'done')",
            (rev,),
        )
    c.execute(
        "INSERT INTO article_sentiment VALUES (1, 'positive', 0.9, 0.9, 0.05, 0.05, 'm', ?)", (now,)
    )
    c.execute("INSERT INTO article_summary VALUES (1, 's', 1, 'm', ?)", (now,))
    c.execute(
        "INSERT OR IGNORE INTO sector (code, name, kind) VALUES ('45', 'Information Technology', 'gics')"
    )
    c.execute(
        "INSERT INTO eval_run (stage, started_at, sample_size, low_conf_n, random_n, judge_model, judge_url, code_version) VALUES ('category_step', ?, 1, 0, 1, 'm', 'u', 'v')",
        (now,),
    )
    c.commit()
    c.close()
    return path


def backup_of(db: Path, target: Path) -> Path:
    src, dst = sqlite3.connect(db), sqlite3.connect(target)
    src.backup(dst)
    src.close()
    dst.close()
    return target


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return build(tmp_path / "nlp.db")


def _plan(db: Path) -> dict[str, object]:
    conn = delivery_golden.open_read_only(db)
    try:
        return dd.make_plan(conn)
    finally:
        conn.close()


def _run(db: Path, *args: str) -> int:
    return dd.main(["--db", str(db), *args])


def test_a_dry_run_reports_and_changes_nothing(
    db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = db.read_bytes()
    assert _run(db) == 0
    out = capsys.readouterr().out
    assert "DROP (v1 tables)" in out and "article_sentiment: 1 rows" in out
    assert "DELETE news_snapshot.model_revision = snapshot-0000000000aa: 1" in out
    assert "keep  news_snapshot.model_revision = snapshot-c3a753affa5d: 1" in out
    assert "dry run: nothing changed" in out and "plan digest:" in out
    assert db.read_bytes() == before


def test_the_dry_run_names_the_kept_snapshot_rows_that_point_at_a_pruned_revision(
    db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(db)
    out = capsys.readouterr().out
    assert "inputs.summary = summary-0000000000aa: 1 rows" in out
    assert "inputs.sentiment = sentiment-0000000000aa: 1 rows" in out
    assert "inputs.category" not in out  # that one is a kept revision


def test_the_plan_digest_is_the_same_twice_and_moves_with_the_data(db: Path) -> None:
    first = _plan(db)["digest"]
    assert _plan(db)["digest"] == first
    c = sqlite3.connect(db)
    c.execute(
        "INSERT INTO pipeline_step_state (item_kind, item_id, step, model_revision, status) VALUES ('sub_news', '9', 's', 'summary-new', 'done')"
    )
    c.commit()
    c.close()
    assert _plan(db)["digest"] != first


def test_apply_needs_a_backup_and_the_plan(db: Path) -> None:
    with pytest.raises(SystemExit):
        _run(db, "--apply")
    with pytest.raises(SystemExit):
        _run(db, "--apply", "--backup", str(db.with_name("b.db")))


def test_apply_refuses_a_plan_that_is_not_the_one_read(
    db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    backup = backup_of(db, tmp_path / "backup.db")
    assert _run(db, "--apply", "--backup", str(backup), "--plan", "0123456789abcdef") == 2
    assert "REFUSED" in capsys.readouterr().err
    assert "article_sentiment" in dd.tables_of(sqlite3.connect(db))  # nothing changed


def test_apply_refuses_a_backup_that_is_not_a_copy(
    db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    backup = backup_of(db, tmp_path / "backup.db")
    c = sqlite3.connect(backup)
    c.execute("DELETE FROM sub_news WHERE id = 2")
    c.commit()
    c.close()
    assert _run(db, "--apply", "--backup", str(backup), "--plan", str(_plan(db)["digest"])) == 2
    assert "not a copy of the database: sub_news" in capsys.readouterr().err


def test_apply_refuses_the_database_as_its_own_backup(
    db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(db, "--apply", "--backup", str(db), "--plan", str(_plan(db)["digest"])) == 2
    assert "the database itself" in capsys.readouterr().err


@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
def test_apply_refuses_when_something_holds_the_database(
    db: Path, tmp_path: Path, suffix: str, capsys: pytest.CaptureFixture[str]
) -> None:
    backup = backup_of(db, tmp_path / "backup.db")
    plan = str(_plan(db)["digest"])
    Path(f"{db}{suffix}").write_bytes(b"x")
    assert _run(db, "--apply", "--backup", str(backup), "--plan", plan) == 2
    assert suffix in capsys.readouterr().err


def test_apply_refuses_a_kept_row_that_belongs_to_a_pruned_cut(
    db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    c = sqlite3.connect(db)
    c.execute(
        "UPDATE news_company SET item_id = '2' WHERE model_revision = ?", (dd.KEEP[3],)
    )  # a kept row, on the sub-news of the old cut
    c.commit()
    c.close()
    backup = backup_of(db, tmp_path / "backup.db")
    assert _run(db, "--apply", "--backup", str(backup), "--plan", str(_plan(db)["digest"])) == 2
    assert "belong to a sub-news of a cut that would be pruned" in capsys.readouterr().err
    assert _run(db) == 1  # the dry run says it too


def test_apply_refuses_a_kept_period_built_on_a_pruned_snapshot(
    db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    c = sqlite3.connect(db)
    c.execute(
        "UPDATE period_run SET snapshot_revision = ? WHERE model_revision = ?",
        ("snapshot" + OLD, KEEP_PERIOD),
    )
    c.commit()
    c.close()
    backup = backup_of(db, tmp_path / "backup.db")
    assert _run(db, "--apply", "--backup", str(backup), "--plan", str(_plan(db)["digest"])) == 2
    assert "built on a snapshot revision that would be pruned" in capsys.readouterr().err


def test_a_kept_revision_with_no_rows_is_refused(
    db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    c = sqlite3.connect(db)
    c.execute("DELETE FROM news_category WHERE model_revision = ?", (dd.KEEP[5],))
    c.commit()
    c.close()
    assert _run(db) == 1
    assert f"the kept revision {dd.KEEP[5]} has no row in news_category" in capsys.readouterr().out


def test_a_table_the_script_does_not_know_is_refused(
    db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE somebody_elses (x)")
    c.commit()
    c.close()
    assert _run(db) == 1
    assert "somebody_elses" in capsys.readouterr().out


def test_apply_drops_the_v1_tables_prunes_the_old_revisions_and_moves_nothing_that_stays(
    db: Path, tmp_path: Path
) -> None:
    backup = backup_of(db, tmp_path / "backup.db")
    conn = delivery_golden.open_read_only(db)
    kept_before, golden_before = (
        dd.kept_counts(conn),
        delivery_golden.digest(delivery_golden.run_all(conn, *dd.GOLDEN)),
    )
    conn.close()
    report = tmp_path / "report.json"
    assert (
        _run(
            db,
            "--apply",
            "--backup",
            str(backup),
            "--plan",
            str(_plan(db)["digest"]),
            "--report",
            str(report),
        )
        == 0
    )
    rep = json.loads(report.read_text())
    assert rep["ok"] is True and rep["foreign_key_violations"] == 0 and rep["quick_check"] == ["ok"]
    assert rep["golden_before"] == rep["golden_after"] == golden_before
    c = sqlite3.connect(db)
    assert not set(dd.DROP) & dd.tables_of(c)  # the v1 tables are gone
    for table, column in dd.PRUNE:
        assert set(dd.revision_counts(c, table, column)) <= set(dd.KEEP), (
            table
        )  # only kept revisions remain
    assert (
        dd.kept_counts(c) == kept_before
    )  # the kept rows, the lean articles, the eval tables: the same
    c.close()
    backup_tables = dd.tables_of(sqlite3.connect(backup))
    assert set(dd.DROP) <= backup_tables  # the backup still has everything


def test_a_second_run_has_nothing_left_to_do(
    db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    backup = backup_of(db, tmp_path / "b1.db")
    assert _run(db, "--apply", "--backup", str(backup), "--plan", str(_plan(db)["digest"])) == 0
    capsys.readouterr()
    assert _run(db) == 0
    out = capsys.readouterr().out
    assert "nothing: already dropped" in out and "DELETE" not in out


def test_the_golden_digest_of_the_kept_period_is_what_the_note_reads(db: Path) -> None:
    conn = delivery_golden.open_read_only(db)
    results = delivery_golden.run_all(conn, *dd.GOLDEN)
    conn.close()
    assert all(
        rows for rows in results.values()
    )  # the fixture has a row for each of the seven queries


def test_the_empty_wal_and_shm_left_by_a_dry_run_do_not_stop_the_apply(
    db: Path, tmp_path: Path
) -> None:
    """The rehearsal on a copy of the backup found it: a dry run opens the file and leaves an empty -wal and a -shm, and the apply must not mistake them for a writer."""
    backup = backup_of(db, tmp_path / "backup.db")
    plan = str(_plan(db)["digest"])
    Path(f"{db}-wal").write_bytes(b"")
    Path(f"{db}-shm").write_bytes(b"\0" * 32)
    assert _run(db, "--apply", "--backup", str(backup), "--plan", plan) == 0
