#!/usr/bin/env python
"""One-shot: restore production `article_sentiment` after the v4 downstream
eval experiment (scripts/resample_sentiment_v4_2026_09_15.py).

Run this only after `uv run cli/news_nlp_eval.py --stage sentiment
--sample-size 2000 --seed 1` has completed against the v4-scored table and
its results are captured (MLflow + the `eval_run`/`eval_judgement` rows,
which live independently of `article_sentiment` and are unaffected by this
restore).

Renames the current (v4-scored, experimental) `article_sentiment` to a
dated snapshot -- kept, not dropped, matching this project's "nothing is
deleted" convention, in case the exact judged rows need re-inspecting --
then renames `article_sentiment_v2_published_snapshot_2026_09_15` (the
real production data, preserved untouched by the resample script) back to
`article_sentiment`. Production ends this script in exactly the state it
was in before the v4 experiment started.

Usage:
    uv run python scripts/restore_sentiment_after_v4_eval_2026_09_15.py \\
        --results-db /path/to/nlp_.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import news_nlp as db

_CURRENT_TABLE = "article_sentiment"
_V2_SNAPSHOT = "article_sentiment_v2_published_snapshot_2026_09_15"
_V4_EXPERIMENT_SNAPSHOT = "article_sentiment_v4_experiment_2026_09_15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-db", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = db.connect(args.results_db)
    try:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?, ?)",
                (_CURRENT_TABLE, _V2_SNAPSHOT, _V4_EXPERIMENT_SNAPSHOT),
            ).fetchall()
        }
        if _V4_EXPERIMENT_SNAPSHOT in existing:
            raise RuntimeError(
                f"{_V4_EXPERIMENT_SNAPSHOT!r} already exists -- this restore already ran."
            )
        if _V2_SNAPSHOT not in existing:
            raise RuntimeError(
                f"{_V2_SNAPSHOT!r} not found -- nothing to restore from. Did "
                "scripts/resample_sentiment_v4_2026_09_15.py actually run against this store?"
            )
        if _CURRENT_TABLE not in existing:
            raise RuntimeError(f"{_CURRENT_TABLE!r} not found -- unexpected state, stop.")

        v4_count = conn.execute(f"SELECT COUNT(*) FROM {_CURRENT_TABLE}").fetchone()[0]  # noqa: S608
        v2_count = conn.execute(f"SELECT COUNT(*) FROM {_V2_SNAPSHOT}").fetchone()[0]  # noqa: S608
        print(
            f"Archiving v4-scored {_CURRENT_TABLE!r} ({v4_count:,} rows) -> "
            f"{_V4_EXPERIMENT_SNAPSHOT!r} ..."
        )
        conn.execute(f"ALTER TABLE {_CURRENT_TABLE} RENAME TO {_V4_EXPERIMENT_SNAPSHOT}")
        conn.commit()

        print(f"Restoring {_V2_SNAPSHOT!r} ({v2_count:,} rows) -> {_CURRENT_TABLE!r} ...")
        conn.execute(f"ALTER TABLE {_V2_SNAPSHOT} RENAME TO {_CURRENT_TABLE}")
        conn.commit()

        final_count = conn.execute(f"SELECT COUNT(*) FROM {_CURRENT_TABLE}").fetchone()[0]  # noqa: S608
        assert final_count == v2_count, (
            f"expected {v2_count:,} rows restored, got {final_count:,} -- stop, investigate"
        )
        print(
            f"Done: {_CURRENT_TABLE!r} restored to its pre-experiment state "
            f"({final_count:,} rows, all v2-scored). The v4-scored experimental rows "
            f"are preserved at {_V4_EXPERIMENT_SNAPSHOT!r}, not deleted."
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
