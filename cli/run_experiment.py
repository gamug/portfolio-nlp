#!/usr/bin/env python
"""CLI entrypoint: run one JSON-described model experiment end to end.

The single command PLAN.md Work item 11 / SPEC.md FR-017 exists for:
loads and strictly validates an `ExperimentSpec` JSON file (`src/
experiment.py`), then calls `run_experiment` -- train a new checkpoint if
the spec asks for one, evaluate it (or, for an eval-only spec, whatever
the spec already names) via `news_nlp.eval.runner.run_eval`, and write a
git-tracked `experiments/results/<name>.result.json`. Publishing a
trained checkpoint to the Hub stays a separate, explicit, manually-run
step (the existing `scripts/publish_*.py` pattern) -- this command never
does it, even when `spec.publish.enabled` is true.

An invalid spec (unpinned `base_model`, `pretrain` for category/
`c_summary`, an unknown `hyperparameters` key, ...) fails with pydantic's
own validation message, naming exactly what's wrong, before any
train/eval work starts -- nothing here wraps or re-words that.

"Exit 1 if regressed" (TASKS.md T-100's own acceptance criterion) needs
no code of its own: `run_experiment` reuses `run_eval` verbatim, which
already raises `SystemExit(1)` -- uncaught, same as `cli/news_nlp_eval.py`
already has for `--check-regression` -- the moment a stage's headline
metric drops past tolerance, before `run_experiment` can even return. This
module doesn't duplicate that check.

Usage:
    uv run cli/run_experiment.py --config experiments/sentiment_v4_class_weighted.json
    uv run cli/run_experiment.py --config experiments/ner_production.json --source-db urls.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment import ExperimentSpec, run_experiment
from news_nlp.eval.runner import summary_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to an ExperimentSpec JSON file."
    )
    parser.add_argument(
        "--source-db", type=Path, default=None, help="Override $SOURCE_DATABASE_URL."
    )
    parser.add_argument("--results-db", type=Path, default=None, help="Override $DATABASE_URL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ExperimentSpec.model_validate_json(args.config.read_text())
    print(f"experiment: name={spec.name!r} stage={spec.stage} pretrain={spec.pretrain.enabled}")

    result = run_experiment(
        spec,
        source_db=str(args.source_db) if args.source_db else None,
        results_db=str(args.results_db) if args.results_db else None,
    )

    if result.train_output_dir:
        print(f"trained: {result.train_output_dir} (metrics: {result.train_metrics})")
    print(summary_table(result.eval_results))
    print(f"result written to {result.result_path}")


if __name__ == "__main__":
    main()
