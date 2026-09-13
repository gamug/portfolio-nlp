#!/usr/bin/env python
"""One-shot: version the current `article_sentiment` table (whatever it
holds -- pre-change scores, or an unmerged entity-scoped prototype's
output), then reprocess a random sample under the new title-only design.

Context: `PLAN.md` Work item 4 (chosen 2026-09-13, in preference to an
entity-scoped chunk-weighting design prototyped and real-data-validated
the same week -- `docs/evaluation.md`'s 2026-09-13 follow-up has that
comparison). `run_sentiment_stage` now scores each article's `title`
directly instead of chunking/aggregating `body_text`, sidestepping the
multi-sentence-aggregation problem entirely. This is future-runs-only by
design (same precedent as every other model/aggregation change in this
project) -- `run_sentiment_stage` only processes articles missing from
`article_sentiment`, so a plain pipeline re-run picks up nothing new.

This script:

1. Renames the existing `article_sentiment` out of the way (whatever name
   is given via `--versioned-table-name` -- there is no single canonical
   "pre-change" name here, since the table may currently hold results from
   an earlier, unmerged design iteration rather than the original
   pre-any-change baseline). Nothing is deleted.
2. Recreates a fresh, empty `article_sentiment` via `news_nlp.init_schema`.
3. Runs the sentiment stage over a `--sample-seed`-reproducible RANDOM
   sample of `--limit` articles under the new title-only code
   (`db.fetch_pending_sentiment_titles`'s `sample_seed` param).

One-shot by design, matching this repo's precedent for this kind of
maintenance script -- not meant to become a permanent CLI flag. Safe to
remove once run; keep it in git history so a future similar resample has a
starting point.

Resumable: if the target versioned-table name already exists, re-running
skips straight to reprocessing instead of erroring.

Usage:
    uv run python scripts/resample_sentiment_title_only_2026_09_13.py \\
        --results-db /path/to/nlp.db --source-db /path/to/urls.db \\
        --versioned-table-name article_sentiment_chunk_level_unmerged_pr42 \\
        --sample-size 2000 --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import news_nlp as db
from pipeline import run_sentiment_stage

_TABLE = "article_sentiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-db", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument(
        "--versioned-table-name",
        required=True,
        help="Name to rename the current article_sentiment to before recreating it empty.",
    )
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def version_sentiment_table(results_db: Path, versioned_table: str) -> None:
    """Rename the current `article_sentiment` to `versioned_table`, then
    recreate a fresh empty `article_sentiment`. Skips (does not error) if
    `versioned_table` already exists -- resuming after this step already
    ran."""
    conn = db.connect(results_db)
    try:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?)",
                (_TABLE, versioned_table),
            ).fetchall()
        }
        if versioned_table in existing:
            print(f"{versioned_table!r} already exists -- versioning already done, resuming.")
            return
        if _TABLE not in existing:
            raise RuntimeError(f"{_TABLE!r} not found -- nothing to version.")

        before_count = conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]  # noqa: S608
        print(f"Renaming {_TABLE!r} ({before_count:,} rows) -> {versioned_table!r} ...")
        conn.execute(f"ALTER TABLE {_TABLE} RENAME TO {versioned_table}")
        conn.commit()

        db.init_schema(conn)  # recreates an empty article_sentiment
        after_count = conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]  # noqa: S608
        assert after_count == 0, f"expected a fresh empty {_TABLE!r}, got {after_count} rows"
        print(f"{_TABLE!r} recreated empty; {versioned_table!r} holds the prior snapshot.")
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    version_sentiment_table(args.results_db, args.versioned_table_name)

    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    try:
        run_sentiment_stage(conn, limit=args.sample_size, sample_seed=args.seed)
    finally:
        db.detach_source(conn)
        conn.close()

    verify = db.connect(args.results_db)
    try:
        n = verify.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]  # noqa: S608
        print(f"Done: {n:,} articles now have title-only rows in {_TABLE!r}.")
    finally:
        verify.close()


if __name__ == "__main__":
    main()
