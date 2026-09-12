#!/usr/bin/env python
"""One-shot: version the pre-fix `article_entities` table, then reprocess a
random sample of it under the fixed (2026-09-10 word-boundary /
2026-09-12 uncapped-eval-sampling) NER code.

Context: docs/evaluation.md's 2026-09-12 NER follow-up / PLAN.md Work item 3
T-025. `article_entities` hadn't run since 2026-08-19 (pre-fix), and there
was no backlog of un-processed articles for a plain pipeline re-run to pick
up -- every article already had a pre-fix row. Rather than delete existing
rows (destroys the pre-fix baseline) or reprocess the full 459K-article
corpus (T-022's larger, still-open question), this:

1. Renames the existing table to `article_entities_v1` (a durable, queryable
   snapshot of the pre-fix model's output -- nothing is deleted).
2. Recreates a fresh, empty `article_entities` via `news_nlp.init_schema`
   (`CREATE TABLE IF NOT EXISTS`, so this is the only place a truly empty
   table lands).
3. Runs the NER stage over a `--sample-seed`-reproducible RANDOM sample of
   `--limit` articles (default 20,000) under the current, fixed code
   (`db.fetch_pending_articles`'s `sample_seed` param, `PLAN.md` Work item
   3 / this script's own commit).

One-shot by design, matching this repo's precedent for this kind of
maintenance script (`docs/migration-2026-09-01.md`'s
`migrate_from_urls_db.py`, removed after its one-time use, recoverable from
git history) -- not meant to become a permanent CLI flag. Safe to remove
once run; keep it in git history rather than deleting it here so a future
similar resample has a starting point.

Usage:
    uv run python scripts/resample_ner_2026_09_12.py \\
        --results-db /path/to/nlp.db --source-db /path/to/urls.db \\
        --sample-size 20000 --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import news_nlp as db
from pipeline import run_ner_stage

_OLD_TABLE = "article_entities"
_VERSIONED_TABLE = "article_entities_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-db", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def version_ner_table(results_db: Path) -> None:
    """Rename the pre-fix `article_entities` to `article_entities_v1`, then
    recreate a fresh empty `article_entities`. Refuses to run twice (a
    pre-existing `article_entities_v1` means this already happened)."""
    conn = db.connect(results_db)
    try:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?)",
                (_OLD_TABLE, _VERSIONED_TABLE),
            ).fetchall()
        }
        if _VERSIONED_TABLE in existing:
            raise RuntimeError(
                f"{_VERSIONED_TABLE!r} already exists -- this script already ran "
                "against this store. Not overwriting; investigate manually."
            )
        if _OLD_TABLE not in existing:
            raise RuntimeError(f"{_OLD_TABLE!r} not found -- nothing to version.")

        before_count = conn.execute(f"SELECT COUNT(*) FROM {_OLD_TABLE}").fetchone()[0]  # noqa: S608
        print(f"Renaming {_OLD_TABLE!r} ({before_count:,} rows) -> {_VERSIONED_TABLE!r} ...")
        conn.execute(f"ALTER TABLE {_OLD_TABLE} RENAME TO {_VERSIONED_TABLE}")

        # SQLite quirk verified before running this against real data: ALTER
        # TABLE RENAME does NOT rename a table's indexes -- they stay bound to
        # the renamed table under their OLD name. Left alone,
        # `init_schema`'s `CREATE INDEX IF NOT EXISTS idx_article_entities_
        # article_id ON article_entities(...)` would silently no-op (an index
        # of that name already exists, just on `article_entities_v1` now) and
        # the fresh `article_entities` would end up with NO index at all.
        # Rename the index onto its new home explicitly first, freeing the
        # canonical name for init_schema to recreate on the fresh table.
        conn.execute("DROP INDEX IF EXISTS idx_article_entities_article_id")
        conn.execute(
            f"CREATE INDEX idx_{_VERSIONED_TABLE}_article_id ON {_VERSIONED_TABLE}(article_id)"
        )
        conn.commit()

        db.init_schema(conn)  # recreates an empty article_entities + its index
        after_count = conn.execute(f"SELECT COUNT(*) FROM {_OLD_TABLE}").fetchone()[0]  # noqa: S608
        assert after_count == 0, f"expected a fresh empty {_OLD_TABLE!r}, got {after_count} rows"
        has_index = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' "
            "AND name='idx_article_entities_article_id' AND tbl_name=?",
            (_OLD_TABLE,),
        ).fetchone()
        assert has_index, f"fresh {_OLD_TABLE!r} is missing its article_id index"
        print(
            f"{_OLD_TABLE!r} recreated empty (indexed); {_VERSIONED_TABLE!r} holds the pre-fix snapshot."
        )
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    version_ner_table(args.results_db)

    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    try:
        run_ner_stage(conn, limit=args.sample_size, sample_seed=args.seed)
    finally:
        db.detach_source(conn)
        conn.close()

    verify = db.connect(args.results_db)
    try:
        n = verify.execute(
            f"SELECT COUNT(DISTINCT article_id) FROM {_OLD_TABLE}"  # noqa: S608
        ).fetchone()[0]
        print(f"Done: {n:,} distinct articles now have post-fix rows in {_OLD_TABLE!r}.")
    finally:
        verify.close()


if __name__ == "__main__":
    main()
