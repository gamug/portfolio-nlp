"""The purge of the results database (T-220, PR C): what it drops and deletes, the gates that stop it, and what it must not move. A fixture database; no model, no network."""

from __future__ import annotations

import json
import os
import shutil
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


# --- review of #215: the race, the unsafe place, the backup gate, the room, the failure paths, the swap -----------------------------------


def _quiet(*_args: object) -> None:
    return None


def _add_summary(path: Path, revision: str, item: str = "9") -> None:
    other = sqlite3.connect(path)
    other.execute(
        "INSERT INTO news_summary (item_kind, item_id, model_revision, summary_text, text_source, processed_at) VALUES ('sub_news', ?, ?, 's', 'summary', 't')",
        (item, revision),
    )
    other.commit()
    other.close()


def test_a_revision_that_appears_during_the_apply_is_left_alone(db: Path) -> None:
    """The plan was read before; a step run that starts meanwhile writes a new revision. It was not in the plan, so it is not deleted (the old ``NOT IN`` would have)."""
    plan = str(_plan(db)["digest"])
    report = dd.apply(
        db,
        plan,
        vacuum=False,
        log=_quiet,
        hook=lambda table: _add_summary(db, "summary-new") if table == "news_summary" else None,
    )
    assert report["ok"] is True and report["left_alone"] == {"news_summary": ["summary-new"]}
    c = sqlite3.connect(db)
    assert (
        c.execute(
            "SELECT COUNT(*) FROM news_summary WHERE model_revision = 'summary-new'"
        ).fetchone()[0]
        == 1
    )
    assert (
        c.execute(
            "SELECT COUNT(*) FROM news_summary WHERE model_revision = ?", ("summary" + OLD,)
        ).fetchone()[0]
        == 0
    )  # the planned one is gone


def test_a_planned_revision_that_changes_during_the_apply_refuses_that_table_and_says_what_was_done(
    db: Path,
) -> None:
    plan = str(_plan(db)["digest"])
    with pytest.raises(
        dd.Refused, match=r"news_summary: summary-0000000000aa has 2 rows now, the plan read 1"
    ) as info:
        dd.apply(
            db,
            plan,
            vacuum=False,
            log=_quiet,
            hook=lambda table: (
                _add_summary(db, "summary" + OLD, "8") if table == "news_summary" else None
            ),
        )
    assert "news_snapshot" in info.value.report["deleted"]  # the tables before it were pruned
    c = sqlite3.connect(db)
    assert (
        c.execute(
            "SELECT COUNT(*) FROM news_summary WHERE model_revision = ?", ("summary" + OLD,)
        ).fetchone()[0]
        == 2
    )  # that table was not touched
    assert "article_sentiment" in dd.tables_of(c)  # and nothing was dropped yet


def test_an_unexpected_failure_names_the_table_and_the_report_lists_the_tables_already_pruned(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = dd._prune_table

    def failing(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        planned: dict[str, int],
        report: dict[str, object],
        log: object,
    ) -> None:
        if table == "news_company":
            raise RuntimeError("disk I/O error")
        real(conn, table, column, planned, report, log)  # type: ignore[arg-type]

    monkeypatch.setattr(dd, "_prune_table", failing)
    with pytest.raises(
        dd.Refused,
        match=r"news_company: RuntimeError: disk I/O error \(the tables before it were already changed",
    ) as info:
        dd.apply(db, str(_plan(db)["digest"]), vacuum=False, log=_quiet)
    assert {"period_item", "news_snapshot", "news_sentiment"} <= set(info.value.report["deleted"])
    assert "news_company" not in info.value.report["deleted"]


def test_a_failed_post_check_exits_2_and_the_report_file_says_what_was_done(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    backup = backup_of(db, tmp_path / "backup.db")
    digests = iter(["before", "after"])
    monkeypatch.setattr(dd, "_golden", lambda _conn: next(digests))
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
            "--no-vacuum",
        )
        == 2
    )
    assert "the checks after the purge failed (restore the backup)" in capsys.readouterr().err
    written = json.loads(report.read_text())
    assert (
        written["ok"] is False
        and written["golden_before"] == "before"
        and written["golden_after"] == "after"
        and written["deleted"]
        and "error" in written
    )


def test_apply_is_refused_on_a_windows_drive_or_on_the_configured_store_unless_in_place(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    backup = backup_of(db, tmp_path / "backup.db")
    plan = str(_plan(db)["digest"])
    monkeypatch.setattr(dd, "_on_windows_drive", lambda _path: True)
    assert _run(db, "--apply", "--backup", str(backup), "--plan", plan, "--no-vacuum") == 2
    assert "Windows drive" in capsys.readouterr().err and "--swap-into" in str(
        dd.refuse_unsafe_place.__doc__
    )
    monkeypatch.setattr(dd, "_on_windows_drive", lambda _path: False)
    monkeypatch.setattr(dd.delivery_golden, "configured_store_paths", lambda: {db.resolve()})
    assert _run(db, "--apply", "--backup", str(backup), "--plan", plan, "--no-vacuum") == 2
    assert "DATABASE_URL" in capsys.readouterr().err
    assert (
        _run(db, "--apply", "--backup", str(backup), "--plan", plan, "--no-vacuum", "--in-place")
        == 0
    )


def test_the_identical_backup_gate_compares_sha256(
    db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    same = tmp_path / "same.db"
    shutil.copyfile(db, same)
    plan = str(_plan(db)["digest"])
    dd.verify_backup(db, same, identical=True)  # a byte-for-byte copy passes
    other = backup_of(db, tmp_path / "other.db")
    c = sqlite3.connect(other)
    c.execute(
        "UPDATE sub_news SET text = 'changed' WHERE id = 1"
    )  # same tables, same counts, same rowids: only the strong gate sees it
    c.commit()
    c.close()
    dd.verify_backup(db, other)  # the fingerprint gate does not see a changed value...
    assert (
        _run(
            db,
            "--apply",
            "--backup",
            str(other),
            "--plan",
            plan,
            "--no-vacuum",
            "--require-identical-backup",
        )
        == 2
    )  # ...the strong one does
    assert "different SHA-256" in capsys.readouterr().err


def test_the_fingerprint_catches_a_backup_with_the_same_counts_and_other_rows(
    db: Path, tmp_path: Path
) -> None:
    other = backup_of(db, tmp_path / "other.db")
    c = sqlite3.connect(other)
    c.execute("UPDATE sub_news SET id = 99 WHERE id = 2")  # the same number of rows, another rowid
    c.commit()
    c.close()
    with pytest.raises(
        dd.Refused, match=r"the backup is not a copy of the database: sub_news reads"
    ):
        dd.verify_backup(db, other)


def test_not_enough_room_for_vacuum_is_reported_and_refused(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    backup = backup_of(db, tmp_path / "backup.db")
    plan = str(_plan(db)["digest"])
    monkeypatch.setattr(dd.shutil, "disk_usage", lambda _path: shutil._ntuple_diskusage(100, 99, 1))
    assert _run(db) == 0
    assert (
        "free space for VACUUM: the volume of the database has 1 bytes free"
        in capsys.readouterr().out
    )
    assert _run(db, "--apply", "--backup", str(backup), "--plan", plan) == 2
    assert "VACUUM needs about" in capsys.readouterr().err
    assert (
        _run(db, "--apply", "--backup", str(backup), "--plan", plan, "--no-vacuum") == 0
    )  # without VACUUM it does not need the room


def _files(tmp_path: Path, db: Path) -> tuple[Path, Path, Path]:
    live = tmp_path / "live.db"
    shutil.copyfile(db, live)
    backup = tmp_path / "live_backup.db"
    shutil.copyfile(db, backup)
    purged = tmp_path / "purged.db"
    shutil.copyfile(db, purged)
    dd.apply(purged, str(_plan(purged)["digest"]), vacuum=False, log=_quiet)
    return purged, live, backup


def test_swap_replaces_the_live_file_with_the_verified_purged_copy(
    db: Path, tmp_path: Path
) -> None:
    purged, live, backup = _files(tmp_path, db)
    sha = dd.sha256_file(purged)
    report = dd.swap(purged, live, expect_sha=sha, backup=backup, log=_quiet)
    assert report["ok"] is True and dd.sha256_file(live) == sha
    assert not live.with_name("live.db.new").exists()
    assert set(dd.DROP).isdisjoint(
        dd.tables_of(sqlite3.connect(live))
    )  # what is live now is the purged one
    assert set(dd.DROP) <= dd.tables_of(
        sqlite3.connect(backup)
    )  # and the backup still has everything


def test_swap_refuses_a_wrong_sha_a_live_file_that_is_not_the_backup_and_a_holder(
    db: Path, tmp_path: Path
) -> None:
    purged, live, backup = _files(tmp_path, db)
    live_sha = dd.sha256_file(live)
    with pytest.raises(dd.Refused, match="--expect-sha says"):
        dd.swap(purged, live, expect_sha="0" * 64, log=_quiet)
    c = sqlite3.connect(backup)
    c.execute("DELETE FROM article_sentiment")
    c.commit()
    c.close()
    with pytest.raises(dd.Refused, match="not byte-identical to --backup"):
        dd.swap(purged, live, backup=backup, log=_quiet)
    Path(f"{live}-journal").write_bytes(b"x")
    with pytest.raises(dd.Refused, match="someone holds the live file"):
        dd.swap(purged, live, log=_quiet)
    assert dd.sha256_file(live) == live_sha  # the live file was never touched


def test_swap_refuses_when_the_live_file_changes_while_the_copy_is_made_and_removes_the_copy(
    db: Path, tmp_path: Path
) -> None:
    purged, live, _backup = _files(tmp_path, db)
    live_sha = dd.sha256_file(live)
    with pytest.raises(dd.Refused, match=r"changed .* while the copy was made"):
        dd.swap(purged, live, log=_quiet, hook=lambda: os.utime(live, ns=(1, 1)))
    assert dd.sha256_file(live) == live_sha and not live.with_name("live.db.new").exists()


def test_the_swap_command(db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    purged, live, backup = _files(tmp_path, db)
    assert (
        _run(
            purged,
            "--swap-into",
            str(live),
            "--backup",
            str(backup),
            "--expect-sha",
            dd.sha256_file(purged),
        )
        == 0
    )
    assert '"ok": true' in capsys.readouterr().out
    assert _run(purged, "--swap-into", str(live), "--expect-sha", "bad") == 2
    assert "REFUSED" in capsys.readouterr().err
