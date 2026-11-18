"""Depurate the NLP results database (T-220, PR C): drop the v1 tables and prune the revisions that are not delivered.

**Dry run by default.** It opens the file read-only, prints what it would drop and delete (per table and per revision) and a *plan digest*. ``--apply`` changes the file and
only when **every gate** holds:

* ``--backup`` names a verified copy of the same database (``quick_check`` ok, the same tables, the same row count in each: the owner makes it with the SQLite backup API);
* ``--plan`` is the digest a dry run printed (so the apply deletes exactly what the owner read, not what the file holds today);
* nobody else holds the file (no ``-wal`` / ``-shm`` / ``-journal`` beside it; an exclusive lock is taken);
* the refusal checks pass: no kept row refers to a pruned cut, no kept period is built on a pruned snapshot, every kept revision exists.

Then, per table, one transaction: ``DELETE`` the rows of the revisions not in ``KEEP``; ``DROP TABLE`` the v1 tables; then ``foreign_key_check``, ``quick_check``, the counts of the
kept rows (must equal the "before") and the golden digest of the delivery note's queries (``delivery_golden``, must equal the "before"); then ``VACUUM``.

**``KEEP`` is a decision of 2026-11-17/18, hard-coded.** A revision made after it is listed as DELETE with a warning (the dry run shows the newest row of every revision): read the plan, do not assume.

What stays untouched: ``articles`` (lean), ``sector``, ``company_sector``, ``eval_run``, ``eval_inference``, ``eval_verdict``, ``eval_confusion``. The rows of a kept snapshot whose ``inputs``
name a pruned revision are **reported, not refused** (the owner accepted it, 2026-11-18): they keep their text, category and sentiment.

Run: ``PYTHONPATH=src uv run python -m depurate_db --db <file> [--backup <file> --plan <digest> --apply]``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import delivery_golden

#: The revisions that stay (owner, 2026-11-17 and 2026-11-18; ``docs/db_depuration_plan.md``).
KEEP: tuple[str, ...] = (
    "scope-nli-8ff21e50796b",
    "units5-d3e2cbfa830e",
    "summary-llm-70cb6bb07292",
    "companies-f349536cf701",
    "sentiment-59863d15fc42",
    "category-a23569dba86f",
    "snapshot-c3a753affa5d",
    "period-3b07b83ecec3",
    "sector-417a93feb789",
)
#: The delivered pair the golden digest is computed for (the delivery note's snapshot and the kept period).
GOLDEN = ("snapshot-c3a753affa5d", "period-3b07b83ecec3")

#: Each pruned table and the column that holds its revision, in the order the rows are deleted (what refers to what first).
PRUNE: tuple[tuple[str, str], ...] = (
    ("period_item", "model_revision"),
    ("period_summary", "model_revision"),
    ("period_run", "model_revision"),
    ("news_snapshot", "model_revision"),
    ("news_category", "model_revision"),
    ("news_sentiment", "model_revision"),
    ("news_company", "model_revision"),
    ("news_summary_call", "model_revision"),
    ("news_summary", "model_revision"),
    ("sub_news", "extractor_revision"),
    ("body_sector", "classifier_revision"),
    ("body_scope", "classifier_revision"),
    ("pipeline_step_state", "model_revision"),
)
#: The v1 tables (and the legacy ``eval_judgement``, superseded 2026-09-18; the verified backup keeps its rows).
DROP: tuple[str, ...] = (
    "article_sentiment",
    "article_entities",
    "article_category",
    "article_summary",
    "sector_summary",
    "eval_judgement",
)
#: Tables that stay and are never written.
UNTOUCHED: tuple[str, ...] = (
    "articles",
    "sector",
    "company_sector",
    "eval_run",
    "eval_inference",
    "eval_verdict",
    "eval_confusion",
)
#: Where each kept revision must exist (a typo in ``KEEP`` would delete everything of that step).
MUST_EXIST: dict[str, str] = {
    "scope-nli-8ff21e50796b": "body_scope",
    "sector-417a93feb789": "body_sector",
    "units5-d3e2cbfa830e": "sub_news",
    "summary-llm-70cb6bb07292": "news_summary",
    "companies-f349536cf701": "news_company",
    "sentiment-59863d15fc42": "news_sentiment",
    "category-a23569dba86f": "news_category",
    "snapshot-c3a753affa5d": "news_snapshot",
    "period-3b07b83ecec3": "period_run",
}
#: The ``news_*`` tables whose ``sub_news`` items must belong to a kept cut.
ITEM_TABLES = ("news_summary", "news_company", "news_sentiment", "news_category", "news_snapshot")
#: The keys of ``news_snapshot.inputs`` that name a revision.
INPUT_KEYS = ("summary", "companies", "cut", "category", "sentiment", "sector")


class Refused(Exception):
    """A gate or a refusal check failed. ``report`` says what had been done before it (empty when nothing was changed)."""

    def __init__(self, message: str, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report: dict[str, Any] = report or {}


def _marks(values: Sequence[str]) -> str:
    return ",".join("?" for _ in values)


def tables_of(conn: sqlite3.Connection) -> set[str]:
    return {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        if not r[0].startswith("sqlite_")
    }


def count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])  # noqa: S608 -- the names come from this module's constants or sqlite_master


#: The day of the keep decision (owner, 2026-11-17 and 2026-11-18). ``KEEP`` is that decision, hard-coded: a revision whose rows are **newer** than this was made after it, and the dry run
#: warns about it if it is listed as DELETE. Compared as text with the ISO timestamps of the tables.
KEEP_DECIDED = "2026-11-18T23:59:59Z"

#: The column of each pruned table that says when a row was written (``COALESCE`` of two for the step state), used only to show the age of a revision. ``period_item`` and
#: ``period_summary`` have none: their age is that of the ``period_run`` of the same revision.
TIMESTAMP: dict[str, str] = {
    "period_run": "created_at",
    "news_snapshot": "processed_at",
    "news_category": "processed_at",
    "news_sentiment": "processed_at",
    "news_company": "processed_at",
    "news_summary_call": "called_at",
    "news_summary": "processed_at",
    "sub_news": "processed_at",
    "body_sector": "processed_at",
    "body_scope": "processed_at",
    "pipeline_step_state": "COALESCE(finished_at, started_at)",
}


def revision_newest(conn: sqlite3.Connection, table: str, column: str) -> dict[str, str]:
    """The newest timestamp of the rows of each revision of *table* (``{}`` for the tables that have none)."""
    stamp = TIMESTAMP.get(table)
    if not stamp:
        return {}
    rows = conn.execute(f'SELECT "{column}", MAX({stamp}) FROM "{table}" GROUP BY 1').fetchall()  # noqa: S608 -- constants
    return {str(r[0]): str(r[1]) for r in rows if r[1] is not None}


def revision_counts(conn: sqlite3.Connection, table: str, column: str) -> dict[str, int]:
    rows = conn.execute(f'SELECT "{column}", COUNT(*) FROM "{table}" GROUP BY 1').fetchall()  # noqa: S608 -- constants
    return {str(r[0]): int(r[1]) for r in rows}


def make_plan(conn: sqlite3.Connection) -> dict[str, Any]:
    """What would be dropped and deleted, and the digest that binds an ``--apply`` to it. Reads only."""
    present = tables_of(conn)
    known = set(UNTOUCHED) | set(DROP) | {t for t, _ in PRUNE}
    plan: dict[str, Any] = {
        "unknown_tables": sorted(present - known),
        "missing_tables": sorted({t for t, _ in PRUNE} - present),
        "drop": [],
        "prune": [],
    }
    for table in DROP:
        if table in present:
            plan["drop"].append({"table": table, "rows": count(conn, table)})
    for table, column in PRUNE:
        if table not in present:
            continue
        newest = revision_newest(conn, table, column)
        for revision, rows in sorted(revision_counts(conn, table, column).items()):
            plan["prune"].append(
                {
                    "table": table,
                    "column": column,
                    "revision": revision,
                    "rows": rows,
                    "keep": revision in KEEP,
                    "newest": newest.get(revision),
                }
            )
    key = json.dumps(
        [[d["table"], d["rows"]] for d in plan["drop"]]
        + [[p["table"], p["revision"], p["rows"], p["keep"]] for p in plan["prune"]],
        sort_keys=True,
    )
    plan["digest"] = hashlib.sha256(key.encode()).hexdigest()[:16]
    return plan


def _kept_revisions_missing(plan: dict[str, Any]) -> list[str]:
    kept_rows = {(p["table"], p["revision"]): p["rows"] for p in plan["prune"] if p["keep"]}
    return [
        f"the kept revision {revision} has no row in {table}: a typo in the keep list would delete the step"
        for revision, table in MUST_EXIST.items()
        if kept_rows.get((table, revision), 0) == 0
    ]


def _pruned_cut_references(conn: sqlite3.Connection, present: set[str]) -> list[str]:
    """Kept ``news_*`` rows whose item is a sub-news of a cut that would be pruned."""
    problems: list[str] = []
    for table in ITEM_TABLES if "sub_news" in present else ():
        if table not in present:
            continue
        bad = conn.execute(
            f"SELECT COUNT(*) FROM {table} t JOIN sub_news s ON t.item_kind = 'sub_news' AND s.id = CAST(t.item_id AS INTEGER) "  # noqa: S608 -- constants
            f"WHERE t.model_revision IN ({_marks(KEEP)}) AND s.extractor_revision NOT IN ({_marks(KEEP)})",
            (*KEEP, *KEEP),
        ).fetchone()[0]
        if bad:
            problems.append(
                f"{bad} kept row(s) of {table} belong to a sub-news of a cut that would be pruned"
            )
    return problems


def _period_snapshot_references(conn: sqlite3.Connection, present: set[str]) -> list[str]:
    if "period_run" not in present:
        return []
    bad = conn.execute(
        f"SELECT COUNT(*) FROM period_run WHERE model_revision IN ({_marks(KEEP)}) AND snapshot_revision NOT IN ({_marks(KEEP)})",  # noqa: S608 -- constants
        (*KEEP, *KEEP),
    ).fetchone()[0]
    return (
        [f"{bad} kept period(s) are built on a snapshot revision that would be pruned"]
        if bad
        else []
    )


def refusals(conn: sqlite3.Connection, plan: dict[str, Any]) -> list[str]:
    """Why the purge must not run (empty list: it can). Reads only."""
    problems: list[str] = []
    if plan["unknown_tables"]:
        problems.append(
            f"tables this script does not know (it will not guess): {plan['unknown_tables']}"
        )
    if plan["missing_tables"]:
        problems.append(f"tables of the stepped pipeline are missing: {plan['missing_tables']}")
    present = tables_of(conn)
    return (
        problems
        + _kept_revisions_missing(plan)
        + _pruned_cut_references(conn, present)
        + _period_snapshot_references(conn, present)
    )


def dangling_provenance(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """Kept snapshot rows whose ``inputs`` name a revision that would be pruned: per key, the revision and the number of rows. Reported, never refused."""
    out: dict[str, dict[str, int]] = {}
    if "news_snapshot" not in tables_of(conn):
        return out
    for key in INPUT_KEYS:
        rows = conn.execute(
            f"SELECT json_extract(inputs, '$.{key}') AS r, COUNT(*) FROM news_snapshot WHERE model_revision IN ({_marks(KEEP)}) "  # noqa: S608 -- constants
            f"AND r IS NOT NULL AND r != '' AND r NOT IN ({_marks(KEEP)}) GROUP BY 1",
            (*KEEP, *KEEP),
        ).fetchall()
        if rows:
            out[key] = {str(r[0]): int(r[1]) for r in rows}
    return out


def kept_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Rows per kept table and revision, and the untouched tables: what must be equal before and after."""
    counts: dict[str, int] = {}
    present = tables_of(conn)
    for table, column in PRUNE:
        if table in present:
            for revision, rows in revision_counts(conn, table, column).items():
                if revision in KEEP:
                    counts[f"{table}:{revision}"] = rows
    for table in UNTOUCHED:
        if table in present:
            counts[table] = count(conn, table)
    return counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(conn: sqlite3.Connection, table: str) -> tuple[int, int, int, int]:
    """A cheap content proxy of a table: the row count and the minimum, maximum and sum of ``rowid``. Equal counts with other rows (another database of the same shape,
    an older copy) differ here; it is not a proof of identical content (``--require-identical-backup`` is)."""
    row = conn.execute(
        f'SELECT COUNT(*), COALESCE(MIN(rowid), 0), COALESCE(MAX(rowid), 0), COALESCE(SUM(rowid), 0) FROM "{table}"'  # noqa: S608 -- names from sqlite_master
    ).fetchone()
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]))


def verify_backup(db: Path, backup: Path, *, full: bool = False, identical: bool = False) -> None:
    """The backup is a readable copy of the same database: same tables, the same count and rowid fingerprint in each, ``quick_check`` (or ``integrity_check``) ok; with
    ``identical`` also the same SHA-256 (the strong gate: valid while the database has not been written since the copy). Raises ``Refused``."""
    if backup.resolve() == db.resolve():
        raise Refused("--backup is the database itself")
    if not backup.is_file():
        raise Refused(f"--backup {backup} does not exist")
    if identical and sha256_file(db) != sha256_file(backup):
        raise Refused(
            "--require-identical-backup: the backup and the database have different SHA-256"
        )
    b = delivery_golden.open_read_only(backup)
    d = delivery_golden.open_read_only(db)
    try:
        check = b.execute("PRAGMA integrity_check" if full else "PRAGMA quick_check").fetchall()
        if check != [("ok",)]:
            raise Refused(
                f"the backup fails its {'integrity' if full else 'quick'} check: {check[:3]}"
            )
        tb, td = tables_of(b), tables_of(d)
        if tb != td:
            raise Refused(f"the backup has other tables than the database: {sorted(tb ^ td)}")
        for table in sorted(td):
            fb, fd = fingerprint(b, table), fingerprint(d, table)
            if fb != fd:
                raise Refused(
                    f"the backup is not a copy of the database: {table} reads {fb} (rows, min, max, sum of rowid), the database {fd}"
                )
    finally:
        b.close()
        d.close()


_MOUNT_FIELDS = 3  # /proc/mounts: device, mount point, file system type, ...


def _on_windows_drive(path: Path) -> bool:
    """True when *path* is on a Windows drive mounted into Linux (``9p`` / ``drvfs``): SQLite is slow there and its locks are not trustworthy for a long write."""
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    target, best, fstype = str(path.resolve()), "", ""
    for line in mounts:
        parts = line.split()
        if len(parts) >= _MOUNT_FIELDS:
            point = parts[1].replace("\\040", " ")
            if (target == point or target.startswith(point.rstrip("/") + "/")) and len(point) > len(
                best
            ):
                best, fstype = point, parts[2]
    return fstype in {"9p", "drvfs", "drvfs2"}


def refuse_unsafe_place(db: Path, *, in_place: bool) -> None:
    """``--apply`` on the live store or on a Windows drive is refused unless ``--in-place``: the safe shape is a purge of a local copy and ``--swap-into``."""
    if in_place:
        return
    if db.resolve() in delivery_golden.configured_store_paths():
        raise Refused(
            f"{db} is the store DATABASE_URL / SOURCE_DATABASE_URL names: purge a local copy and swap it in (--swap-into), or pass --in-place"
        )
    if _on_windows_drive(db):
        raise Refused(
            f"{db} is on a Windows drive (slow, and its locks are not trustworthy for a long write): purge a local copy and swap it in (--swap-into), or pass --in-place"
        )


def check_free_space(db: Path) -> list[str]:
    """Why ``VACUUM`` would run out of room (empty list: enough): it needs about the size of the file free on the volume of the database and in the temp folder."""
    need = int(db.stat().st_size * 1.2)
    problems = []
    for label, folder in (
        ("the volume of the database", db.parent),
        ("the temp folder", Path(tempfile.gettempdir())),
    ):
        free = shutil.disk_usage(folder).free
        if free < need:
            problems.append(f"{label} has {free:,} bytes free, VACUUM needs about {need:,}")
    return problems


def _golden(conn: sqlite3.Connection) -> str:
    return delivery_golden.digest(delivery_golden.run_all(conn, *GOLDEN))


def _hold(db: Path) -> sqlite3.Connection:
    """Open the database for writing and prove nobody else holds it (no writer file beside it, an exclusive lock taken then released)."""
    held = delivery_golden.holder_file(db)
    if held:
        raise Refused(
            f"{held} exists: someone holds the database or was writing; stop the writers and let the last connection close first"
        )
    conn = sqlite3.connect(db, isolation_level=None, timeout=3)
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN EXCLUSIVE")
    except sqlite3.OperationalError as exc:
        conn.close()
        raise Refused(f"the database is locked by another process: {exc}") from exc
    return conn


def _planned(plan: dict[str, Any]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """What the dry run listed as DELETE (per table: revision -> rows) and as DROP (table -> rows). The apply touches nothing else."""
    prune: dict[str, dict[str, int]] = {}
    for p in plan["prune"]:
        if not p["keep"]:
            prune.setdefault(p["table"], {})[p["revision"]] = p["rows"]
    return prune, {d["table"]: d["rows"] for d in plan["drop"]}


def _expect_revision_rows(table: str, revision: str, now: int, planned: int) -> None:
    if now != planned:
        raise Refused(
            f"{table}: {revision} has {now} rows now, the plan read {planned}: a writer changed it; run the dry run again"
        )


def _expect_plan(plan: dict[str, Any], plan_digest: str, problems: list[str]) -> None:
    if plan["digest"] != plan_digest:
        raise Refused(
            f"the database no longer matches the plan: it is {plan['digest']}, --plan says {plan_digest}"
        )
    if problems:
        raise Refused("; ".join(problems))


#: The refusal checks that protect a table, run again **inside its lock**: a writer that adds a kept row pointing at what this table is about to lose, after the start-of-run check, is caught.
_TABLE_GUARDS = {"sub_news": _pruned_cut_references, "news_snapshot": _period_snapshot_references}


def _expect_no_new_references(conn: sqlite3.Connection, table: str) -> None:
    guard = _TABLE_GUARDS.get(table)
    problems = guard(conn, tables_of(conn)) if guard else []
    if problems:
        raise Refused(
            f"{table}: {'; '.join(problems)} (a writer added them after the start-of-run check)"
        )


def _exclusive(conn: sqlite3.Connection, table: str) -> None:
    try:
        conn.execute("BEGIN EXCLUSIVE")
    except sqlite3.OperationalError as exc:
        raise Refused(f"{table}: the database is locked by another process: {exc}") from exc


def _prune_table(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    planned: dict[str, int],
    report: dict[str, Any],
    log: Any,
) -> None:
    """One transaction holding the exclusive lock from the check to the commit: delete **only the revisions the plan lists**, and only if each still has the rows the plan read."""
    started = time.monotonic()
    _exclusive(conn, table)
    try:
        now = revision_counts(conn, table, column)
        for revision, rows in planned.items():
            _expect_revision_rows(table, revision, now.get(revision, 0), rows)
        _expect_no_new_references(conn, table)
        delete = f'DELETE FROM "{table}" WHERE "{column}" IN ({_marks(list(planned))})'  # noqa: S608 -- names from the constants
        cur = conn.execute(delete, tuple(planned))
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    left = sorted(r for r in now if r not in KEEP and r not in planned)
    if left:
        report["left_alone"][table] = left
    report["deleted"][table] = cur.rowcount
    report["seconds"][table] = round(time.monotonic() - started, 1)
    log(
        f"pruned {table}: {cur.rowcount} rows ({report['seconds'][table]} s)"
        + (f"; left alone (not in the plan): {left}" if left else "")
    )


def _drop_table(
    conn: sqlite3.Connection, table: str, planned_rows: int, report: dict[str, Any], log: Any
) -> None:
    started = time.monotonic()
    _exclusive(conn, table)
    try:
        rows = count(conn, table)
        _expect_revision_rows(table, "(the whole table)", rows, planned_rows)
        conn.execute(f'DROP TABLE "{table}"')
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    report["dropped"][table] = rows
    log(f"dropped {table}: {rows} rows ({round(time.monotonic() - started, 1)} s)")


def _purge(
    conn: sqlite3.Connection, plan: dict[str, Any], report: dict[str, Any], log: Any, hook: Any
) -> None:
    prune, drop = _planned(plan)
    steps: list[tuple[str, Any]] = [
        (t, lambda t=t, c=c: _prune_table(conn, t, c, prune[t], report, log))
        for t, c in PRUNE
        if t in prune
    ]
    steps += [
        (t, lambda t=t: _drop_table(conn, t, drop[t], report, log)) for t in DROP if t in drop
    ]
    for table, step in steps:
        if hook:
            hook(table)
        try:
            step()
        except Refused:
            raise
        except Exception as exc:
            raise Refused(
                f"{table}: {type(exc).__name__}: {exc} (the tables before it were already changed: restore the backup)",
                report,
            ) from exc


def _expect_ok(report: dict[str, Any]) -> None:
    if not report["ok"]:
        raise Refused("the checks after the purge failed (restore the backup)")


def _vacuum(conn: sqlite3.Connection, report: dict[str, Any], log: Any) -> None:
    started = time.monotonic()
    conn.execute("VACUUM")
    report["seconds"]["vacuum"] = round(time.monotonic() - started, 1)
    log(f"vacuum ({report['seconds']['vacuum']} s)")


def _after_checks(
    conn: sqlite3.Connection, report: dict[str, Any], before_counts: dict[str, int]
) -> None:
    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    quick = conn.execute("PRAGMA quick_check").fetchall()
    report.update(
        foreign_key_violations=len(fk),
        quick_check=[r[0] for r in quick],
        golden_after=_golden(conn),
        counts_equal=before_counts == kept_counts(conn),
    )
    report["ok"] = (
        not fk
        and quick == [("ok",)]
        and report["counts_equal"]
        and report["golden_before"] == report["golden_after"]
    )
    _expect_ok(report)


def apply(
    db: Path, plan_digest: str, *, vacuum: bool = True, log: Any = print, hook: Any = None
) -> dict[str, Any]:
    """Run the purge on ``db``. The gates of ``main`` are checked by the caller; this re-checks the plan once at the start (under an exclusive lock that it then **releases**: it is a
    start-of-run check, not a lock held during the purge) and then holds the exclusive lock **table by table**, from the check of that table to its commit. ``hook(table)`` is called
    before each table (the tests use it to play a writer). On a refusal or a failure, ``Refused.report`` says what had already been done."""
    conn = _hold(db)
    report: dict[str, Any] = {
        "ok": False,
        "plan": plan_digest,
        "deleted": {},
        "dropped": {},
        "left_alone": {},
        "seconds": {},
    }
    try:
        plan = make_plan(conn)
        _expect_plan(plan, plan_digest, refusals(conn, plan))
        conn.execute("ROLLBACK")
        before_counts = kept_counts(conn)
        log("golden digest before ...")
        report["golden_before"] = _golden(conn)
        _purge(conn, plan, report, log, hook)
        _after_checks(conn, report, before_counts)
        if vacuum:
            _vacuum(conn, report, log)
    except Refused as exc:
        exc.report = report
        raise
    except Exception as exc:
        raise Refused(
            f"the purge failed: {type(exc).__name__}: {exc} (restore the backup)", report
        ) from exc
    else:
        return report
    finally:
        conn.close()


def _sidecars(path: Path) -> dict[str, bool]:
    return {suffix: Path(f"{path}{suffix}").exists() for suffix in ("-wal", "-shm")}


def _remove_probe_sidecars(live: Path, existed: dict[str, bool], log: Any) -> None:
    """The probe opens the live file read-write; in WAL mode that leaves an empty ``-wal`` and a ``-shm`` beside it (on a Windows drive they are not always removed when the probe
    closes). Remove **only** the ones the probe created (absent before; an empty ``-wal``; the ``-shm``). A leftover that was there before, or a ``-wal`` with data, is not touched.
    The probe proved no *writer* holds the file; it cannot prove no *reader* does, which is why the runbook says to close every tool."""
    for suffix, was_there in existed.items():
        path = Path(f"{live}{suffix}")
        if was_there or not path.exists() or (suffix == "-wal" and path.stat().st_size > 0):
            continue
        path.unlink()
        log(f"removed {path.name}: the probe had created it")


def _probe_writer(live: Path, log: Any = print) -> None:
    """Refuse when another connection holds a write lock on *live*: take ``BEGIN EXCLUSIVE`` and roll it back. It detects a writer; it **cannot** detect a reader (a DB browser), so the
    runbook says to close every tool that has the file open. The sidecars it creates are removed (``_remove_probe_sidecars``)."""
    existed = _sidecars(live)
    probe = sqlite3.connect(live, isolation_level=None, timeout=1)
    try:
        probe.execute("BEGIN EXCLUSIVE")
        probe.execute("ROLLBACK")
    except sqlite3.OperationalError as exc:
        raise Refused(
            f"the live file is locked by another process ({exc}): stop every writer and close every tool that has it open"
        ) from exc
    finally:
        probe.close()
        _remove_probe_sidecars(live, existed, log)


def journal_mode(path: Path) -> str:
    """The journal mode recorded in the file (``wal`` or ``delete``...), read without changing it."""
    conn = delivery_golden.open_read_only(path)
    try:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()


def _expect_purged(purged: Path, live: Path) -> None:
    """The file being swapped in is a good purged copy of **this** live file: readable, ``quick_check`` ok, nothing left for a purge to do, and the same kept rows (kept revisions and
    the untouched tables) as the file it replaces."""
    try:
        p, lv = delivery_golden.open_read_only(purged), delivery_golden.open_read_only(live)
    except sqlite3.Error as exc:
        raise Refused(f"cannot open the files to compare them: {exc}") from exc
    try:
        _compare_purged(p, lv)
    except sqlite3.Error as exc:
        raise Refused(
            f"cannot read the purged or the live file ({exc}): a process may hold the live file, or the purged file is not a database"
        ) from exc
    finally:
        p.close()
        lv.close()


def _kept_rows_differ(p: sqlite3.Connection, lv: sqlite3.Connection) -> str:
    a, b = kept_counts(p), kept_counts(lv)
    first = next((k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)), "?")
    return (
        f"the kept rows differ ({first}: purged {a.get(first)}, live {b.get(first)}): either the live file changed since the copy was made (a step ran on it: start again "
        "from a fresh copy and a fresh backup), or this file was made from another database"
    )


def _compare_purged(p: sqlite3.Connection, lv: sqlite3.Connection) -> None:
    quick = p.execute("PRAGMA quick_check").fetchall()
    if quick != [("ok",)]:
        raise Refused(f"the purged file fails its quick_check: {quick[:3]}")
    plan = make_plan(p)
    left = [d["table"] for d in plan["drop"]] + [
        f"{x['table']}.{x['revision']}" for x in plan["prune"] if not x["keep"]
    ]
    if left or refusals(p, plan):
        raise Refused(
            f"the purged file is not a purged database: a purge would still drop or delete {left[:5]}"
        )
    if kept_counts(p) != kept_counts(lv):
        raise Refused(_kept_rows_differ(p, lv))


def _swap_preconditions(
    purged: Path, live: Path, expect_sha: str, backup: Path | None, log: Any
) -> str:
    """Everything the swap checks before it copies a byte; returns the purged file's SHA-256."""
    for label, path in (("purged", purged), ("live", live)):
        if not path.is_file():
            raise Refused(f"the {label} file {path} does not exist")
        held = delivery_golden.holder_file(path)
        if held:
            raise Refused(
                f"{held} exists: someone holds the {label} file; stop the writers and let the last connection close first (a non-empty -wal: open it once with the SQLite tool to checkpoint it)"
            )
    purged_sha = sha256_file(purged)
    log(f"purged {purged}: {purged.stat().st_size:,} bytes, SHA-256 {purged_sha}")
    if purged_sha != expect_sha:
        raise Refused(f"the purged file has SHA-256 {purged_sha}, the expected one is {expect_sha}")
    _expect_purged(purged, live)
    if backup is not None:
        live_sha, backup_sha = sha256_file(live), sha256_file(backup)
        log(
            f"live {live}: {live.stat().st_size:,} bytes, SHA-256 {live_sha}; backup SHA-256 {backup_sha}"
        )
        if live_sha != backup_sha:
            raise Refused(
                "the live file is not byte-identical to --backup: it changed since the backup was made; start again from a fresh backup"
            )
    _probe_writer(live, log)
    return purged_sha


def _check_journal_modes(purged: Path, live: Path, allow: bool, log: Any) -> dict[str, str]:
    """Log the journal mode of both files and refuse a change (the owner's tools behave differently in WAL and in rollback mode) unless ``allow``."""
    modes = {"purged": journal_mode(purged), "live": journal_mode(live)}
    log(f"journal mode: purged {modes['purged']}, live {modes['live']}")
    if modes["purged"] != modes["live"] and not allow:
        raise Refused(
            f"the purged file is in {modes['purged']} mode and the live file in {modes['live']} mode: pass --allow-journal-mode-change if that is intended"
        )
    return modes


def swap(
    purged: Path,
    live: Path,
    *,
    expect_sha: str,
    backup: Path | None = None,
    log: Any = print,
    hook: Any = None,
    allow_journal_mode_change: bool = False,
) -> dict[str, Any]:
    """Replace the live file with a verified purged copy: the shape that protects a production file (the purge ran on a local copy, never in place).

    Checks (``_swap_preconditions``): neither file has a holder (a ``-journal`` or a non-empty ``-wal``); the purged file has ``expect_sha`` (the ``sha256`` of the apply's report); it is a
    good purged copy of this live file; with ``backup`` the live file **is** the backup byte for byte; nobody holds a write lock on the live file; both files have the same journal mode
    (unless ``allow_journal_mode_change``). Then the copy next to the live file must have the same SHA-256, and the live file must not change (size, mtime), gain a holder or get locked
    while it is made; then one atomic rename. On any refusal the live file is untouched and the copy is removed. What it compared is logged."""
    purged_sha = _swap_preconditions(purged, live, expect_sha, backup, log)
    modes = _check_journal_modes(purged, live, allow_journal_mode_change, log)
    before = (live.stat().st_size, live.stat().st_mtime_ns)
    new = live.with_name(live.name + ".new")
    shutil.copyfile(purged, new)
    try:
        if sha256_file(new) != purged_sha:
            raise Refused("the copy next to the live file differs from the purged file")
        if hook:
            hook()
        if (live.stat().st_size, live.stat().st_mtime_ns) != before or delivery_golden.holder_file(
            live
        ):
            raise Refused(
                "the live file changed (or gained a holder) while the copy was made: nothing was replaced"
            )
        _probe_writer(live, log)
        os.replace(new, live)
    finally:
        if new.exists():
            new.unlink()
    log(f"replaced {live} with the purged copy (SHA-256 {purged_sha})")
    return {"ok": True, "sha256": purged_sha, "live": str(live), "journal_mode": modes}


def newer_than_the_decision(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """The revisions the plan would DELETE although their newest row is later than ``KEEP_DECIDED``: made after the keep list was decided."""
    return [
        p for p in plan["prune"] if not p["keep"] and p.get("newest") and p["newest"] > KEEP_DECIDED
    ]


def format_plan(
    plan: dict[str, Any], problems: list[str], dangling: dict[str, dict[str, int]]
) -> str:
    lines = ["DROP (v1 tables):"]
    lines += [f"  {d['table']}: {d['rows']:,} rows" for d in plan["drop"]] or [
        "  nothing: already dropped"
    ]
    lines.append(
        "PRUNE (rows of revisions not in the keep list; `keep` rows stay; the date is the newest row of the revision):"
    )
    for p in plan["prune"]:
        age = f"  (newest {p['newest']})" if p.get("newest") else ""
        lines.append(
            f"  {'keep ' if p['keep'] else 'DELETE'} {p['table']}.{p['column']} = {p['revision']}: {p['rows']:,}{age}"
        )
    fresh = newer_than_the_decision(plan)
    if fresh:
        lines.append(
            f"WARNING: the keep list was decided on {KEEP_DECIDED[:10]}; these revisions to be DELETED are newer than that. Is it meant to prune them?"
        )
        lines += [
            f"  {p['table']}.{p['column']} = {p['revision']}: newest row {p['newest']}"
            for p in fresh
        ]
    if dangling:
        lines.append(
            "KEPT snapshot rows whose `inputs` name a revision that would be pruned (reported, not refused):"
        )
        for key, revisions in dangling.items():
            lines += [f"  inputs.{key} = {rev}: {rows:,} rows" for rev, rows in revisions.items()]
    lines.append("REFUSALS: " + ("none" if not problems else ""))
    lines += [f"  - {p}" for p in problems]
    lines.append(f"plan digest: {plan['digest']}")
    return "\n".join(lines)


def _run_gated(
    args: argparse.Namespace, plan: dict[str, Any], problems: list[str]
) -> dict[str, Any]:
    if problems:
        raise Refused("; ".join(problems))
    if args.plan != plan["digest"]:
        raise Refused(
            f"--plan {args.plan} is not the digest of this database today ({plan['digest']}): run the dry run again and read it"
        )
    refuse_unsafe_place(args.db, in_place=args.in_place)
    room = check_free_space(args.db)
    if room and not args.no_vacuum:
        raise Refused("; ".join(room))
    verify_backup(
        args.db, args.backup, full=args.full_backup_check, identical=args.require_identical_backup
    )
    report = apply(args.db, args.plan, vacuum=not args.no_vacuum)
    report.update(
        sha256=sha256_file(args.db), size=args.db.stat().st_size
    )  # what --swap-into expects
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="depurate_db", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="the database to purge (a local copy), or with --swap-into the purged copy",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        help="a verified copy of --db (SQLite backup API); required by --apply",
    )
    parser.add_argument("--plan", help="the plan digest a dry run printed; required by --apply")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="change the database (default: a dry run that reads only)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="allow --apply on the store DATABASE_URL names or on a Windows drive (not recommended: purge a copy, then --swap-into)",
    )
    parser.add_argument(
        "--require-identical-backup",
        action="store_true",
        help="the backup must have the same SHA-256 as --db (the strong gate)",
    )
    parser.add_argument(
        "--full-backup-check",
        action="store_true",
        help="integrity_check of the backup instead of quick_check (slow on a 5 GB file)",
    )
    parser.add_argument("--no-vacuum", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        help="write the report of an --apply (also a failed one) here as JSON",
    )
    parser.add_argument(
        "--swap-into",
        type=Path,
        help="replace this live file with --db (a verified purged copy); with --backup the live file must equal it byte for byte",
    )
    parser.add_argument(
        "--expect-sha",
        help="with --swap-into: the SHA-256 the purged copy must have (the `sha256` of the apply's report)",
    )
    parser.add_argument(
        "--allow-journal-mode-change",
        action="store_true",
        help="with --swap-into: allow the purged file and the live file to be in different journal modes",
    )
    parser.add_argument(
        "--from-report",
        type=Path,
        help="with --swap-into: the --report of the apply that made the purged copy (its `sha256` is the expected one; it must say ok)",
    )
    return parser


def _write_report(path: Path | None, report: dict[str, Any], error: str | None = None) -> None:
    if path:
        path.write_text(
            json.dumps({**report, **({"error": error} if error else {})}, indent=2),
            encoding="utf-8",
        )


def _expected_sha(args: argparse.Namespace) -> str:
    if args.from_report:
        written = json.loads(args.from_report.read_text(encoding="utf-8"))
        if not written.get("ok") or not written.get("sha256"):
            raise Refused(
                f"{args.from_report} is not the report of a successful apply (ok and sha256 are needed)"
            )
        return str(written["sha256"])
    if not args.expect_sha:
        raise Refused(
            "--swap-into needs --expect-sha <sha256> or --from-report <report of the apply>"
        )
    return str(args.expect_sha)


def _main_swap(args: argparse.Namespace) -> int:
    try:
        report = swap(
            args.db,
            args.swap_into,
            expect_sha=_expected_sha(args),
            backup=args.backup,
            allow_journal_mode_change=args.allow_journal_mode_change,
        )
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.db.is_file():
        parser.error(f"{args.db} does not exist")
    if args.swap_into:
        return _main_swap(args)
    held = delivery_golden.holder_file(args.db)
    if held and args.apply:
        print(
            f"REFUSED: {held} exists: someone holds the database or was writing; stop the writers and let the last connection close first",
            file=sys.stderr,
        )
        return 2
    conn = delivery_golden.open_read_only(args.db)
    try:
        plan = make_plan(conn)
        problems = refusals(conn, plan)
        dangling = dangling_provenance(conn)
    except sqlite3.Error as exc:
        print(
            f"REFUSED: the database cannot be read ({exc})"
            + (f"; {held} sits beside it" if held else ""),
            file=sys.stderr,
        )
        return 2
    finally:
        conn.close()
    print(format_plan(plan, problems, dangling))
    room = check_free_space(args.db)
    print("free space for VACUUM: " + ("enough" if not room else "; ".join(room)))
    if not args.apply:
        print("dry run: nothing changed")
        return 1 if problems else 0
    if not args.backup or not args.plan:
        parser.error("--apply needs --backup <verified copy> and --plan <digest of a dry run>")
    try:
        report = _run_gated(args, plan, problems)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        _write_report(args.report, exc.report, str(exc))
        return 2
    print(json.dumps(report, indent=2))
    _write_report(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
