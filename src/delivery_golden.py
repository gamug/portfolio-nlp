"""The queries of the delivery note (``snapshow_delivery.md``) as code, so a database can be checked against what the upstream team reads (T-220, PR A).

The note is untracked (the owner's choice), so its SQL is copied here, once, with the two revisions as parameters. ``run_all`` runs every query on a
connection and returns the rows sorted; ``digest`` is their hash. The depuration compares the digest on the backup copy before and after each
destructive step: it must not change (a drop of a v1 table, or a prune of old revisions, may not move one row the upstream reads).

Read only: it opens nothing and writes nothing. A query that fails (a table the depuration should have kept) raises, which is the signal."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: name -> SQL, with ``{snapshot}`` and ``{period}`` for the two revisions the note names. The text of each is the note's own.
QUERIES: dict[str, str] = {
    "snapshot_date_sources": """
        SELECT json_extract(inputs, '$.date') AS date_source, COUNT(*)
        FROM news_snapshot
        WHERE model_revision = '{snapshot}' AND ticker != ''
        GROUP BY 1""",
    "company_news_in_a_range": """
        SELECT d AS date, title, category, sentiment, sentiment_score, text
        FROM (
          SELECT *, CASE WHEN pub_date != '' THEN pub_date ELSE fetched_at END AS d
          FROM news_snapshot
          WHERE model_revision = '{snapshot}' AND ticker = 'AAPL'
        )
        WHERE d >= '2026-01-01T00:00:00Z' AND d < '2026-02-01T00:00:00Z'
        ORDER BY d""",
    "periods": """
        SELECT period_end, period_start FROM period_run
        WHERE model_revision = '{period}'
        ORDER BY period_end""",
    "counts_per_company_category_period": """
        SELECT s.period_end, s.key AS ticker, c.key AS category,
               json_extract(c.value, '$.total')    AS total,
               json_extract(c.value, '$.positive') AS positive,
               json_extract(c.value, '$.negative') AS negative,
               json_extract(c.value, '$.neutral')  AS neutral
        FROM period_summary s, json_each(s.counts) c
        WHERE s.model_revision = '{period}' AND s.kind = 'company'
        ORDER BY s.period_end, s.key, category""",
    "net_sentiment_per_company_period": """
        SELECT s.period_end, s.key AS ticker,
               SUM(json_extract(c.value, '$.total'))    AS news,
               SUM(json_extract(c.value, '$.positive')) AS positive,
               SUM(json_extract(c.value, '$.negative')) AS negative,
               1.0 * (SUM(json_extract(c.value, '$.positive')) - SUM(json_extract(c.value, '$.negative')))
                   / NULLIF(SUM(json_extract(c.value, '$.total')), 0) AS net_sentiment
        FROM period_summary s, json_each(s.counts) c
        WHERE s.model_revision = '{period}' AND s.kind = 'company'
        GROUP BY s.period_end, s.key""",
    "counts_per_sector": """
        SELECT s.period_end, COALESCE(n.name, 'unclassified') AS sector, c.key AS category,
               json_extract(c.value, '$.total') AS total
        FROM period_summary s
        JOIN json_each(s.counts) c
        LEFT JOIN sector n ON n.code = s.key
        WHERE s.model_revision = '{period}' AND s.kind = 'sector'
        ORDER BY s.period_end, sector, category""",
    "news_a_period_counted": """
        SELECT i.period_end, i.ticker, i.late, n.title, n.category, n.sentiment
        FROM period_item i
        JOIN news_snapshot n
          ON n.item_kind = i.item_kind AND n.item_id = i.item_id AND n.ticker = i.ticker
         AND n.model_revision = '{snapshot}'
        WHERE i.model_revision = '{period}' AND i.period_end = '2026-08-31T19:00:00Z'""",
}

#: The queries that read a period revision: after that period is pruned they return nothing, the two that read only the snapshot do not change.
PERIOD_QUERIES = (
    "periods",
    "counts_per_company_category_period",
    "net_sentiment_per_company_period",
    "counts_per_sector",
    "news_a_period_counted",
)

#: The tables those queries read: the ones the depuration must keep for the upstream team.
TABLES = ("news_snapshot", "period_run", "period_summary", "period_item", "sector")


def run_all(
    conn: sqlite3.Connection, snapshot: str, period: str
) -> dict[str, list[tuple[Any, ...]]]:
    """Every query of the note on ``conn``, the rows of each sorted (a query without ``ORDER BY`` is not deterministic)."""
    out: dict[str, list[tuple[Any, ...]]] = {}
    for name, sql in QUERIES.items():
        rows = conn.execute(sql.format(snapshot=snapshot, period=period)).fetchall()
        out[name] = sorted((tuple(r) for r in rows), key=repr)
    return out


def digest(results: dict[str, list[tuple[Any, ...]]]) -> str:
    """One hash of all the results: equal before and after a depuration step, or the step moved what the upstream reads."""
    return hashlib.sha256(json.dumps(results, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]


def holder_file(path: Path) -> str | None:
    """The file beside *path* that says someone holds it (or was cut off while writing), or ``None``: a ``-journal`` (an unfinished rollback-mode transaction) or a **non-empty**
    ``-wal`` (written pages not yet in the main file). An empty ``-wal`` and a ``-shm`` are left by any reader, this tool's own dry run included, and mean nothing."""
    journal = Path(f"{path}-journal")
    wal = Path(f"{path}-wal")
    if journal.exists():
        return str(journal)
    if wal.exists() and wal.stat().st_size > 0:
        return str(wal)
    return None


def _refuse_if_live(path: Path) -> None:
    """``immutable=1`` on a database that someone writes can return a half-written state with no error, so it is refused when the file looks live: a ``-journal`` or a non-empty
    ``-wal`` beside it (``holder_file``), or the path is the one ``DATABASE_URL`` / ``SOURCE_DATABASE_URL`` names (the store the pipeline writes or reads)."""
    held = holder_file(path)
    if held:
        raise ValueError(
            f"{held} sits beside {path}: it looks live, --immutable is refused (copy it with the SQLite backup API first)"
        )
    from news_nlp import env  # noqa: PLC0415 -- only the guard needs it

    configured = {
        p.resolve() for p in (env.results_db_path(), env.source_db_path()) if p is not None
    }
    if path.resolve() in configured:
        raise ValueError(
            f"{path} is the store named by DATABASE_URL or SOURCE_DATABASE_URL: --immutable is refused on a live store; run it on a copy"
        )


def open_read_only(path: Path, *, immutable: bool = False) -> sqlite3.Connection:
    """A read-only connection. ``immutable`` tells SQLite the file cannot change (a backup copy): it takes no lock and needs no ``-wal`` / ``-shm`` file. It is refused when
    the file looks live (``_refuse_if_live``)."""
    if immutable:
        _refuse_if_live(path)
    uri = f"file:{path.resolve()}?mode=ro" + ("&immutable=1" if immutable else "")
    return sqlite3.connect(uri, uri=True)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m delivery_golden --db <file> --snapshot <rev> --period <rev>``: the row count and the time of each query of the note, and the digest. Writes nothing. The depuration
    compares the digest before and after each destructive step (T-220)."""
    parser = argparse.ArgumentParser(prog="delivery_golden", description=main.__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--snapshot", required=True, help="e.g. snapshot-c3a753affa5d")
    parser.add_argument("--period", required=True, help="e.g. period-3b07b83ecec3")
    parser.add_argument(
        "--immutable",
        action="store_true",
        help="a backup copy that no one writes (no lock, no -wal/-shm)",
    )
    args = parser.parse_args(argv)
    try:
        conn = open_read_only(args.db, immutable=args.immutable)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        results: dict[str, list[tuple[Any, ...]]] = {}
        for name, sql in QUERIES.items():
            started = time.monotonic()
            rows = conn.execute(sql.format(snapshot=args.snapshot, period=args.period)).fetchall()
            results[name] = sorted((tuple(r) for r in rows), key=repr)
            print(f"{name}: {len(rows)} rows, {time.monotonic() - started:.1f}s", flush=True)
    finally:
        conn.close()
    print(f"digest: {digest(results)}")
    emptied = {name: ([] if name in PERIOD_QUERIES else rows) for name, rows in results.items()}
    print(
        f"digest if --period is pruned (its five period queries empty, the snapshot ones unchanged): {digest(emptied)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
