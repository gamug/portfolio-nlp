"""Depurate the NLP results database (T-220, PR C): drop the v1 tables and prune the revisions that are not delivered.

**Dry run by default.** It opens the file read-only, prints what it would drop and delete (per table and per revision) and a *plan digest*. ``--apply`` changes the file and
only when **every gate** holds:

* ``--backup`` names a verified copy of the same database (``quick_check`` ok, the same tables, the same row count in each: the owner makes it with the SQLite backup API);
* ``--plan`` is the digest a dry run printed (so the apply deletes exactly what the owner read, not what the file holds today);
* nobody else holds the file (no ``-wal`` / ``-shm`` / ``-journal`` beside it; an exclusive lock is taken);
* the refusal checks pass: no kept row refers to a pruned cut, no kept period is built on a pruned snapshot, every kept revision exists.

Then, per table, one transaction: ``DELETE`` the rows of the revisions not in ``KEEP``; ``DROP TABLE`` the v1 tables; then ``foreign_key_check``, ``quick_check``, the counts of the
kept rows (must equal the "before") and the golden digest of the delivery note's queries (``delivery_golden``, must equal the "before"); then ``VACUUM``.

What stays untouched: ``articles`` (lean), ``sector``, ``company_sector``, ``eval_run``, ``eval_inference``, ``eval_verdict``, ``eval_confusion``. The rows of a kept snapshot whose ``inputs``
name a pruned revision are **reported, not refused** (the owner accepted it, 2026-11-18): they keep their text, category and sentiment.

Run: ``PYTHONPATH=src uv run python -m depurate_db --db <file> [--backup <file> --plan <digest> --apply]``."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
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
    """A gate or a refusal check failed: nothing was changed."""


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
        for revision, rows in sorted(revision_counts(conn, table, column).items()):
            plan["prune"].append(
                {
                    "table": table,
                    "column": column,
                    "revision": revision,
                    "rows": rows,
                    "keep": revision in KEEP,
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


def verify_backup(db: Path, backup: Path, *, full: bool = False) -> None:
    """The backup is a readable copy of the same database: same tables, same row counts, ``quick_check`` (or ``integrity_check``) ok. Raises ``Refused``."""
    if backup.resolve() == db.resolve():
        raise Refused("--backup is the database itself")
    if not backup.is_file():
        raise Refused(f"--backup {backup} does not exist")
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
            nb, nd = count(b, table), count(d, table)
            if nb != nd:
                raise Refused(
                    f"the backup is not a copy of the database: {table} has {nb} rows, the database {nd}"
                )
    finally:
        b.close()
        d.close()


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


def _prune_and_drop(conn: sqlite3.Connection, report: dict[str, Any], log: Any) -> None:
    for table, column in PRUNE:
        if table not in tables_of(conn):
            continue
        started = time.monotonic()
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(f'DELETE FROM "{table}" WHERE "{column}" NOT IN ({_marks(KEEP)})', KEEP)  # noqa: S608 -- constants
        conn.execute("COMMIT")
        report["deleted"][table] = cur.rowcount
        report["seconds"][table] = round(time.monotonic() - started, 1)
        log(f"pruned {table}: {cur.rowcount} rows ({report['seconds'][table]} s)")
    for table in DROP:
        if table in tables_of(conn):
            started = time.monotonic()
            rows = count(conn, table)
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(f'DROP TABLE "{table}"')
            conn.execute("COMMIT")
            report["dropped"][table] = rows
            log(f"dropped {table}: {rows} rows ({round(time.monotonic() - started, 1)} s)")


def apply(db: Path, plan_digest: str, *, vacuum: bool = True, log: Any = print) -> dict[str, Any]:
    """Run the purge on ``db`` (the gates of ``main`` are checked by the caller; this re-checks the plan and the lock). Returns the report."""
    conn = _hold(db)
    try:
        plan = make_plan(conn)
        if plan["digest"] != plan_digest:
            raise Refused(
                f"the database no longer matches the plan: it is {plan['digest']}, --plan says {plan_digest}"
            )
        problems = refusals(conn, plan)
        if problems:
            raise Refused("; ".join(problems))
        conn.execute("ROLLBACK")
        before_counts = kept_counts(conn)
        log("golden digest before ...")
        report: dict[str, Any] = {
            "plan": plan["digest"],
            "golden_before": _golden(conn),
            "deleted": {},
            "dropped": {},
            "seconds": {},
        }
        _prune_and_drop(conn, report, log)
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
        if not report["ok"]:
            raise Refused(f"the checks after the purge failed (restore the backup): {report}")
        if vacuum:
            started = time.monotonic()
            conn.execute("VACUUM")
            report["seconds"]["vacuum"] = round(time.monotonic() - started, 1)
            log(f"vacuum ({report['seconds']['vacuum']} s)")
        return report
    finally:
        conn.close()


def format_plan(
    plan: dict[str, Any], problems: list[str], dangling: dict[str, dict[str, int]]
) -> str:
    lines = ["DROP (v1 tables):"]
    lines += [f"  {d['table']}: {d['rows']:,} rows" for d in plan["drop"]] or [
        "  nothing: already dropped"
    ]
    lines.append("PRUNE (rows of revisions not in the keep list; `keep` rows stay):")
    for p in plan["prune"]:
        lines.append(
            f"  {'keep ' if p['keep'] else 'DELETE'} {p['table']}.{p['column']} = {p['revision']}: {p['rows']:,}"
        )
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
    verify_backup(args.db, args.backup, full=args.full_backup_check)
    return apply(args.db, args.plan, vacuum=not args.no_vacuum)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="depurate_db", description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, required=True)
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
        "--full-backup-check",
        action="store_true",
        help="integrity_check of the backup instead of quick_check (slow on a 5 GB file)",
    )
    parser.add_argument("--no-vacuum", action="store_true")
    parser.add_argument("--report", type=Path, help="write the report of an --apply here as JSON")
    args = parser.parse_args(argv)
    if not args.db.is_file():
        parser.error(f"{args.db} does not exist")
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
    if not args.apply:
        print("dry run: nothing changed")
        return 1 if problems else 0
    if not args.backup or not args.plan:
        parser.error("--apply needs --backup <verified copy> and --plan <digest of a dry run>")
    try:
        report = _run_gated(args, plan, problems)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
