#!/usr/bin/env python
"""One-shot: version the pre-entity-scoping `article_sentiment` table, then
reprocess a random sample of it under the new entity-scoped, sentence-level
aggregation.

Context: `PLAN.md` Work item 4 / `TASKS.md` T-030/T-034. `run_sentiment_stage`
was rewritten (2026-09-12) to score each sentence individually and weight
sentences naming the article's own company/ticker over everything else,
instead of a plain token-count-weighted mean across ~510-token chunks. That
change is future-runs-only by design (same precedent as the NER
subword-fragmentation fix, `scripts/resample_ner_2026_09_12.py`) -- the
existing `article_sentiment` table holds only pre-change scores, and
`run_sentiment_stage` only processes articles missing from its own results
table, so a plain pipeline re-run picks up nothing new. This script:

1. Renames the existing table to `article_sentiment_v1` (a durable, queryable
   snapshot of the pre-change model's output -- nothing is deleted). Unlike
   `article_entities`, `article_sentiment.article_id` is itself the
   `INTEGER PRIMARY KEY` (the SQLite rowid alias, not a separate secondary
   index) -- so, verified against a scratch in-memory DB before running this
   against real data, `ALTER TABLE ... RENAME TO` carries the primary key
   along for free and there is no separate index to re-home (the gotcha the
   NER script had to work around does not apply here).
2. Recreates a fresh, empty `article_sentiment` via `news_nlp.init_schema`
   (`CREATE TABLE IF NOT EXISTS`, so this is the only place a truly empty
   table lands).
3. Runs the sentiment stage over a `--sample-seed`-reproducible RANDOM
   sample of `--limit` articles (default 10,000 -- smaller than NER's
   20,000: per-sentence scoring means more forward passes per article than
   NER's per-chunk approach, so this is deliberately more conservative
   pending a real throughput measurement) under the current, entity-scoped
   code (`db.fetch_pending_sentiment_articles`'s `sample_seed` param).

One-shot by design, matching this repo's precedent for this kind of
maintenance script (`docs/migration-2026-09-01.md`'s
`migrate_from_urls_db.py`, `scripts/resample_ner_2026_09_12.py`) -- not
meant to become a permanent CLI flag. Safe to remove once run; keep it in
git history rather than deleting it here so a future similar resample has a
starting point.

Resumable: if `article_sentiment_v1` already exists (step 1 already ran --
interactively, or a prior invocation of this script got interrupted before
step 3), re-running skips straight to reprocessing instead of erroring.

Usage:
    uv run python scripts/resample_sentiment_2026_09_12.py \\
        --results-db /path/to/nlp.db --source-db /path/to/urls.db \\
        --sample-size 10000 --seed 1

Then evaluate the freshly-reprocessed sample against the LLM judge:
    uv run cli/news_nlp_eval.py --stage sentiment --seed 1 --sample-size 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import news_nlp as db
from pipeline import run_sentiment_stage

_OLD_TABLE = "article_sentiment"
_VERSIONED_TABLE = "article_sentiment_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-db", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def version_sentiment_table(results_db: Path) -> None:
    """Rename the pre-entity-scoping `article_sentiment` to
    `article_sentiment_v1`, then recreate a fresh empty `article_sentiment`.
    Refuses to run twice (a pre-existing `article_sentiment_v1` means this
    already happened)."""
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
        conn.commit()

        db.init_schema(conn)  # recreates an empty article_sentiment
        after_count = conn.execute(f"SELECT COUNT(*) FROM {_OLD_TABLE}").fetchone()[0]  # noqa: S608
        assert after_count == 0, f"expected a fresh empty {_OLD_TABLE!r}, got {after_count} rows"
        print(
            f"{_OLD_TABLE!r} recreated empty; {_VERSIONED_TABLE!r} holds the pre-change snapshot."
        )
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    try:
        version_sentiment_table(args.results_db)
    except RuntimeError as exc:
        if _VERSIONED_TABLE not in str(exc):
            raise  # a different problem (e.g. _OLD_TABLE missing entirely) -- don't mask it
        # Versioning already happened (e.g. run interactively, or this script
        # was re-run after an interruption between versioning and
        # reprocessing) -- safe to resume straight into reprocessing rather
        # than treat this as an error.
        print(f"{_VERSIONED_TABLE!r} already exists -- versioning already done, resuming.")

    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    try:
        run_sentiment_stage(conn, limit=args.sample_size, sample_seed=args.seed)
    finally:
        db.detach_source(conn)
        conn.close()

    verify = db.connect(args.results_db)
    try:
        n = verify.execute(f"SELECT COUNT(*) FROM {_OLD_TABLE}").fetchone()[0]  # noqa: S608
        print(f"Done: {n:,} articles now have post-change rows in {_OLD_TABLE!r}.")
    finally:
        verify.close()


if __name__ == "__main__":
    main()
