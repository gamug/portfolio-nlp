#!/usr/bin/env python
"""One-shot: score a real-article sample with v4 (the class-weighted
sentiment retrain, PLAN.md Work item 9), so the LLM-judge eval harness can
measure its downstream/production-pipeline behavior -- the number that
actually validated v2 (docs/evaluation.md's 2026-09-13 four-candidate
comparison), and the one still missing for every retrained candidate so
far.

Same mechanics as scripts/resample_sentiment_2026_09_12.py (version the
current `article_sentiment`, recreate it empty, reprocess a fresh sample
under the code being evaluated) -- except the "code change" here is a
*candidate model swap*, done via monkeypatching `pipeline.SENTIMENT_MODEL`
/ `pipeline.MODEL_REVISIONS` in-process for the duration of this script
only. `src/pipeline.py` itself is never edited -- the file on disk, and
therefore the actually-pinned production model, is untouched throughout.

**This is a temporary, reversible experiment, not an adoption.** The
current table is preserved under `article_sentiment_v2_published_snapshot_2026_09_15`
(exact production state before this script ran), not overwritten.
Run scripts/restore_sentiment_after_v4_eval_2026_09_15.py afterward --
once the eval judge run below has completed and its results are captured
-- to put production back exactly as it was; do not leave the v4-scored
table in place as `article_sentiment` past that point.

Usage:
    uv run python scripts/resample_sentiment_v4_2026_09_15.py \\
        --results-db /path/to/nlp_.db --source-db /path/to/urls_.db \\
        --sample-size 2500 --seed 1

Then evaluate the freshly-scored sample against the LLM judge, at the
recommended sentiment sample-size floor (docs/evaluation.md's "Sample-size
floor" section):
    uv run cli/news_nlp_eval.py --stage sentiment --sample-size 2000 --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import news_nlp as db
import pipeline

_OLD_TABLE = "article_sentiment"
_VERSIONED_TABLE = "article_sentiment_v2_published_snapshot_2026_09_15"
_V4_CHECKPOINT = str(
    Path(__file__).resolve().parent.parent / "models" / "finbert-financial-news-weighted"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-db", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def version_sentiment_table(results_db: Path) -> None:
    """Rename the current (v2-scored, production) `article_sentiment` to a
    dated snapshot, then recreate a fresh empty `article_sentiment`. Refuses
    to run twice (a pre-existing snapshot means this already happened)."""
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
        print(
            f"Renaming {_OLD_TABLE!r} ({before_count:,} rows, the live v2-scored "
            f"production data) -> {_VERSIONED_TABLE!r} ..."
        )
        conn.execute(f"ALTER TABLE {_OLD_TABLE} RENAME TO {_VERSIONED_TABLE}")
        conn.commit()

        db.init_schema(conn)  # recreates an empty article_sentiment
        after_count = conn.execute(f"SELECT COUNT(*) FROM {_OLD_TABLE}").fetchone()[0]  # noqa: S608
        assert after_count == 0, f"expected a fresh empty {_OLD_TABLE!r}, got {after_count} rows"
        print(
            f"{_OLD_TABLE!r} recreated empty; {_VERSIONED_TABLE!r} holds the v2 snapshot "
            "(restore from this once the v4 eval is done)."
        )
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    try:
        version_sentiment_table(args.results_db)
    except RuntimeError as exc:
        if _VERSIONED_TABLE not in str(exc):
            raise
        print(f"{_VERSIONED_TABLE!r} already exists -- versioning already done, resuming.")

    # Point run_sentiment_stage at v4's local checkpoint instead of the
    # pinned Hub model, in-process only -- src/pipeline.py on disk is never
    # touched. "local" as the revision is a harmless placeholder --
    # from_pretrained ignores `revision` entirely for a local directory path,
    # it's only meaningful for a Hub repo id.
    pipeline.SENTIMENT_MODEL = _V4_CHECKPOINT
    pipeline.MODEL_REVISIONS[_V4_CHECKPOINT] = "local"
    print(f"Scoring with v4 (class-weighted retrain): {_V4_CHECKPOINT}")

    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    try:
        pipeline.run_sentiment_stage(conn, limit=args.sample_size, sample_seed=args.seed)
    finally:
        db.detach_source(conn)
        conn.close()

    verify = db.connect(args.results_db)
    try:
        n = verify.execute(f"SELECT COUNT(*) FROM {_OLD_TABLE}").fetchone()[0]  # noqa: S608
        print(f"Done: {n:,} articles now have v4-scored rows in {_OLD_TABLE!r}.")
        print("Next: uv run cli/news_nlp_eval.py --stage sentiment --sample-size 2000 --seed 1")
        print(
            "Then: uv run python scripts/restore_sentiment_after_v4_eval_2026_09_15.py "
            "to put production back."
        )
    finally:
        verify.close()


if __name__ == "__main__":
    main()
