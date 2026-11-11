#!/usr/bin/env python
"""CLI entrypoint of the stepped pipeline: one step at a time, never the whole chain (D39 (vii), D40; T-162, T-163).

Usage:
    uv run cli/news_nlp_steps.py --step scope --limit 500     # classify up to 500 unique bodies not yet done
    uv run cli/news_nlp_steps.py --step scope --status        # counts by state and by scope, no model loaded

A rerun adds nothing: the bodies already done at the model's revision are skipped, failed ones are retried. A new model
revision starts a new set of rows and leaves the old ones. ``--model-dir`` is the folder written by
``scripts/train_scope_final_2026_11_11.py`` (default ``models/scope_final``).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tqdm import tqdm

import news_nlp as db
from news_nlp import steps
from news_nlp.db import exit_on_schema_error

STEPS = (steps.STEP_SCOPE,)
BATCH = 16


def run_scope(conn: db.NewsNlpDatabase, model_dir: Path, limit: int | None) -> dict[str, int]:
    """Classify up to ``limit`` unique bodies not yet done at the model's revision; return what was done."""
    from scope_stage import ScopeClassifier  # noqa: PLC0415 -- loads torch

    classifier = ScopeClassifier(model_dir)
    todo = list(steps.pending_bodies(conn, steps.STEP_SCOPE, classifier.revision, limit))
    counts: dict[str, int] = {"done": 0, "failed": 0}
    bar = tqdm(total=len(todo), desc="scope", unit="body")  # progress and time left
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        try:
            results = classifier.classify([(b.title, b.text) for b in batch])
        except Exception as exc:
            for body in batch:
                steps.record_failure(conn, body, steps.STEP_SCOPE, classifier.revision, repr(exc))
            counts["failed"] += len(batch)
            bar.update(len(batch))
            continue
        for body, result in zip(batch, results, strict=True):
            steps.record_scope(
                conn,
                body,
                classifier.revision,
                result.scope,
                result.scores,
                f"title+{classifier.chars}",
            )
            counts["done"] += 1
            counts[result.scope] = counts.get(result.scope, 0) + 1
        bar.update(len(batch))
    bar.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", choices=STEPS, required=True)
    parser.add_argument("--limit", type=int, default=None, help="touch at most this many bodies")
    parser.add_argument("--status", action="store_true", help="print the counts and exit")
    parser.add_argument("--model-dir", type=Path, default=Path("models/scope_final"))
    parser.add_argument("--source-db", default=None)
    parser.add_argument("--results-db", default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    db.env.load_env_file()
    with exit_on_schema_error():
        conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    db.init_schema(conn)
    revision = None
    if (args.model_dir / "training_record.json").exists():
        from scope_stage import revision_of  # noqa: PLC0415 -- no torch import here

        revision = revision_of(args.model_dir)
    if args.status:
        print(json.dumps(steps.status(conn, args.step, revision), indent=2))
        return
    if revision is None:
        sys.exit(f"{args.model_dir}/training_record.json is missing: train the models first")
    print(json.dumps(run_scope(conn, args.model_dir, args.limit), indent=2))
    print(json.dumps(steps.status(conn, args.step, revision), indent=2))


if __name__ == "__main__":
    main()
