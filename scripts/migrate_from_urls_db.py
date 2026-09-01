#!/usr/bin/env python
"""One-shot copy of the NLP data out of the legacy shared `urls.db` into this
repo's dedicated `nlp.db`.

Copies the five result tables (`article_sentiment`, `article_category`,
`article_summary`, `article_entities`, `sector_summary`) and a lean `articles`
subset -- every column except `body_text`, only for rows a copied result row
references. `discovered_urls` / `discovery_progress` are left behind.

Idempotent (INSERT OR IGNORE). The source is attached read-only and never
written. Usage:

    uv run python scripts/migrate_from_urls_db.py \
        --source D:/thesis/data/urls.db --dest D:/thesis/data/nlp.db
    uv run python scripts/migrate_from_urls_db.py --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db as nlp_db

DEFAULT_SOURCE = Path("D:/thesis/data/urls.db")
DEFAULT_DEST = Path("D:/thesis/data/nlp.db")

# copied in this order; every one except sector_summary is filtered to
# article_id present in src.articles
RESULT_TABLES = (
    "article_sentiment",
    "article_category",
    "article_summary",
    "article_entities",
    "sector_summary",
)
ENTITIES_INDEX = "idx_article_entities_article_id"


def _connect_dest(dest: Path) -> sqlite3.Connection:
    """Open the destination with the same pragma contract as
    ``db.connect`` (busy_timeout + foreign_keys), but with
    ``uri=True`` so the read-only ``file:...?mode=ro`` ATTACH of the source is
    honored -- SQLite only interprets an ATTACH argument as a URI when the main
    connection was opened with URI filenames enabled.
    """
    conn = sqlite3.connect(f"file:{dest.as_posix()}", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={nlp_db.BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _lean_article_columns(src: sqlite3.Connection) -> list[str]:
    cols = [row[1] for row in src.execute("PRAGMA table_info(articles)")]
    return [c for c in cols if c != "body_text"]


def _copy_result_table(conn: sqlite3.Connection, t: str) -> int:
    """Copy one result table from `src.<t>` into dest, returning rows added.

    Copies by an explicit, symmetric column list taken from the *dest* schema
    -- never `SELECT *` -- so the copy is immune to the physical column order
    differing between src and dest. `sector_summary` in particular carries
    `format_version` / `facts_json` / `intro_text` in a different position in
    dest (they arrive via `db._migrate_sector_summary_schema`'s
    `ALTER TABLE ADD COLUMN`) than in the legacy src, and every column is
    NOT NULL, so a positional copy would silently shuffle values with no error.
    Every table except `sector_summary` is filtered to article ids present in
    `src.articles`.
    """
    before = _table_count(conn, t)
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
    col_list = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO {t} ({col_list}) SELECT {col_list} FROM src.{t}"  # noqa: S608
    if t != "sector_summary":
        sql += " WHERE article_id IN (SELECT id FROM src.articles)"
    conn.execute(sql)
    return _table_count(conn, t) - before


def _referenced_article_ids_sql() -> str:
    parts = [
        f"SELECT article_id FROM src.{t}"  # noqa: S608
        for t in RESULT_TABLES
        if t != "sector_summary"
    ]
    return " UNION ".join(parts)


def migrate(source: Path, dest: Path, *, dry_run: bool = False) -> dict[str, int]:
    source, dest = Path(source), Path(dest)
    if not source.exists():
        raise FileNotFoundError(f"source database not found: {source}")

    src_ro = f"file:{source.as_posix()}?mode=ro"

    if dry_run:
        probe = sqlite3.connect(":memory:", uri=True)
        probe.execute("ATTACH DATABASE ? AS src", (src_ro,))
        counts: dict[str, int] = {}
        for t in RESULT_TABLES:
            if t == "sector_summary":
                n = probe.execute("SELECT COUNT(*) FROM src.sector_summary").fetchone()[0]
            else:
                n = probe.execute(
                    f"SELECT COUNT(*) FROM src.{t} "  # noqa: S608
                    f"WHERE article_id IN (SELECT id FROM src.articles)"
                ).fetchone()[0]
            counts[t] = n
        counts["articles"] = probe.execute(
            f"SELECT COUNT(*) FROM src.articles WHERE id IN ({_referenced_article_ids_sql()})"  # noqa: S608
        ).fetchone()[0]
        probe.close()
        _print_report(counts, dry_run=True)
        return counts

    dest.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect_dest(dest)  # nlp_db.connect's pragma contract + uri=True
    nlp_db.init_schema(conn)  # creates result tables + entities index
    _ensure_lean_articles_table(conn, source, src_ro)

    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("ATTACH DATABASE ? AS src", (src_ro,))

    counts = {}
    src_probe = sqlite3.connect(src_ro, uri=True)
    art_cols = _lean_article_columns(src_probe)
    src_probe.close()
    col_list = ", ".join(art_cols)
    before = _table_count(conn, "articles")
    conn.execute(
        f"INSERT OR IGNORE INTO articles ({col_list}) "  # noqa: S608
        f"SELECT {col_list} FROM src.articles "
        f"WHERE id IN ({_referenced_article_ids_sql()})"
    )
    counts["articles"] = _table_count(conn, "articles") - before

    conn.execute(f"DROP INDEX IF EXISTS {ENTITIES_INDEX}")
    for t in RESULT_TABLES:
        counts[t] = _copy_result_table(conn, t)
    conn.execute(f"CREATE INDEX IF NOT EXISTS {ENTITIES_INDEX} ON article_entities(article_id)")

    conn.commit()
    conn.execute("DETACH DATABASE src")

    fk_problems = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk_problems:
        for row in fk_problems:
            print(f"  FK violation: {row}", file=sys.stderr)
        conn.close()
        print("foreign_key_check failed -- aborting", file=sys.stderr)
        sys.exit(2)

    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("ANALYZE")
    conn.close()
    _print_report(counts, dry_run=False)
    return counts


def _ensure_lean_articles_table(conn: sqlite3.Connection, source: Path, src_ro: str) -> None:
    if conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
    ).fetchone():
        return
    src = sqlite3.connect(src_ro, uri=True)
    cols = [
        (row[1], row[2])
        for row in src.execute("PRAGMA table_info(articles)")
        if row[1] != "body_text"
    ]
    src.close()
    defs = ", ".join(
        f"{name} {ctype or 'TEXT'}{' PRIMARY KEY' if name == 'id' else ''}" for name, ctype in cols
    )
    conn.execute(f"CREATE TABLE articles ({defs})")


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    result: int = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
    return result


def _print_report(counts: dict[str, int], *, dry_run: bool) -> None:
    verb = "would copy" if dry_run else "copied"
    width = max(len(k) for k in counts)
    print(f"\n{'table':<{width}}  rows {verb}")
    print("-" * (width + 6 + len(verb)))
    for name, n in counts.items():
        print(f"{name:<{width}}  {n:>12,}")
    print(f"{'TOTAL':<{width}}  {sum(counts.values()):>12,}\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    migrate(args.source, args.dest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
