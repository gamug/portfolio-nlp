#!/usr/bin/env python
"""CLI entrypoint: LLM-as-judge accuracy evaluation of the news-NLP stages.

Samples a stratified low_conf / target_<x> / representative slice of the
stored sentiment / category / NER / c_summary predictions (soft-probability
class-targeted for sentiment/category, num_chunks-tiered for c_summary), has
an LLM judge (a strands-agents agent over an OpenAI-compatible endpoint)
score each one against the source article text, and writes aggregate
metrics + per-row verdicts to MLflow and to the eval_run / eval_inference /
eval_verdict tables (plus eval_confusion for sentiment/category) in the
RESULTS store. See docs/evaluation.md.

sector_summary is judged differently: its population is small enough
(thousands, not hundreds of thousands, of rows) to score in full every run,
so --sample-size/--seed/--low-conf-frac/--target-frac are silently ignored
for it -- see news_nlp.eval.sampling's module docstring.

The judge is itself a model, so the numbers are agreement-with-a-judge, not
ground truth. Needs LLM_API_KEY / LLM_MODEL / LLM_URL in the environment
(or .env); view runs with `uv run mlflow ui`.

Usage:
    uv run cli/news_nlp_eval.py --stage all --sample-size 80
    uv run cli/news_nlp_eval.py --stage sentiment --stage category --seed 1
    uv run cli/news_nlp_eval.py --stage all --check-regression   # exit 1 on a metric drop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from news_nlp.eval.config import EvalSettings
from news_nlp.eval.runner import run_eval, summary_table
from news_nlp.eval.sampling import STAGES

_STAGE_CHOICES = (*STAGES, "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="append",
        choices=_STAGE_CHOICES,
        dest="stages",
        help="Stage to evaluate; repeatable. 'all' (the default) runs every stage.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Rows to judge per stage (default 80). See docs/evaluation.md for the "
        "recommended floor on a regression-tracked run.",
    )
    parser.add_argument(
        "--low-conf-frac",
        type=float,
        default=None,
        help="Share of --sample-size spent on the deterministic worst-case "
        "low_conf bucket (default 0.2). Diagnostic-only -- excluded from every "
        "headline/population-estimate metric.",
    )
    parser.add_argument(
        "--target-frac",
        type=float,
        default=None,
        help="Share of the post-low_conf budget spent on soft-probability-targeted "
        "strata (default 0.6; no-op for ner, which has none). See docs/evaluation.md.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (reproducible runs).")
    parser.add_argument(
        "--max-workers", type=int, default=None, help="Concurrent judge calls (default 4)."
    )
    parser.add_argument(
        "--source-db", type=Path, default=None, help="Override $SOURCE_DATABASE_URL."
    )
    parser.add_argument("--results-db", type=Path, default=None, help="Override $DATABASE_URL.")
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help="Override $MLFLOW_TRACKING_URI (default ./mlruns).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Label this run in the MLflow UI's run list (e.g. 'v5-sec-bert-base'), instead "
        "of MLflow's auto-generated name. Cosmetic only -- doesn't change which experiment "
        "the run lands in (still news_nlp_eval/<stage>) or --check-regression's comparison. "
        "Applied to every stage in this invocation.",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Exit 1 if a stage's headline metric dropped past --regression-tolerance "
        "versus the previous MLflow run.",
    )
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=0.05,
        help="Allowed headline-metric drop before --check-regression fails (default 0.05).",
    )
    parser.add_argument(
        "--candidate-model",
        default=None,
        help="Score this stage with a candidate model/checkpoint (Hub repo id or local "
        "path) instead of reading the production model's already-stored predictions -- "
        "via that stage's own FTI Inference subclass, against a throwaway scratch RESULTS "
        "file that's discarded afterward. Requires exactly one --stage (a candidate swap "
        "is inherently stage-specific). Replaces the old scripts/resample_sentiment_v* "
        "scratch-DB-copy workaround.",
    )
    parser.add_argument(
        "--candidate-revision",
        default=None,
        help="Required with --candidate-model. Revision/commit SHA for a Hub repo id; any "
        "placeholder (e.g. 'local') for a local checkpoint path, since from_pretrained "
        "ignores revision for a local directory entirely.",
    )
    parser.add_argument(
        "--candidate-prescore-size",
        type=int,
        default=None,
        help="How many SOURCE articles to pre-score with --candidate-model before sampling "
        "from them (default: --sample-size). Larger than --sample-size on purpose, so "
        "stratification has a real population to draw worst-case/near-miss rows from.",
    )
    args = parser.parse_args()
    if args.candidate_model:
        stages = args.stages or []
        if len(stages) != 1 or "all" in stages:
            parser.error("--candidate-model requires exactly one --stage (not 'all' or several)")
        if not args.candidate_revision:
            parser.error("--candidate-model requires --candidate-revision")
    return args


def main() -> None:
    args = parse_args()
    requested = args.stages or ["all"]
    stages = list(STAGES) if "all" in requested else list(dict.fromkeys(requested))

    settings = EvalSettings.load(
        mlflow_tracking_uri=args.mlflow_uri,
        sample_size=args.sample_size,
        low_conf_frac=args.low_conf_frac,
        target_frac=args.target_frac,
        seed=args.seed,
        max_workers=args.max_workers,
        run_name=args.run_name,
        candidate_model=args.candidate_model,
        candidate_revision=args.candidate_revision,
        candidate_prescore_size=args.candidate_prescore_size,
    )
    print(
        f"eval: stages={stages} sample_size={settings.sample_size} "
        f"judge={settings.llm_model} mlflow={settings.mlflow_tracking_uri}"
    )
    results = run_eval(
        stages,
        settings=settings,
        source_db=str(args.source_db) if args.source_db else None,
        results_db=str(args.results_db) if args.results_db else None,
        check_regression=args.check_regression,
        regression_tolerance=args.regression_tolerance,
    )
    print(summary_table(results))


if __name__ == "__main__":
    main()
