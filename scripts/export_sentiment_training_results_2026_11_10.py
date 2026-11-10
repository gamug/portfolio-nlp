"""Export the metrics of the MLflow runs cited in docs/sentiment_training.md to a tracked CSV
(docs/sentiment_training_results.csv), so the figures can be checked without the local, git-ignored `mlruns/`.
One row per (run, metric): experiment, run name, start time (UTC), run id, metric, value. A run that was made
twice (the relevance experiment was rerun on 2026-11-10 and reproduced the figures exactly) appears twice, with
its own start time and a `recorded` column: `yes` for the earliest run of each name (the one
docs/sentiment_training.md quotes), `no` for a later rerun or for a run tagged `superseded` (its model was replaced). Usage: `python scripts/export_sentiment_training_results_2026_11_10.py <mlruns> [out.csv]`."""

import csv
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow

OUT_ARG = 2  # the position of the optional output path
EXPERIMENTS = (
    "sentiment_pair_company_gate",
    "sentiment_pair_company_yes_no",
    "sentiment_pair_neutral_threshold",
    "sentiment_pair_learning_curve",
    "sentiment_pair_more_data",
    "sentiment_pair_cv_all_data",
    "sentiment_pair_cv_all_data_replay809",
    "sentiment_pair_cv_all_data_base_finbert",
    "sentiment_pair_final_all",
    "sentiment_pair_cv_threshold",
    "sentiment_pair_company_cues_cv",
    "company_relevance_zeroshot",
    "sentiment_pair_relevance_gate",
)


def main() -> None:
    mlflow.set_tracking_uri(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > OUT_ARG else "docs/sentiment_training_results.csv")
    rows = []
    for name in EXPERIMENTS:
        runs = mlflow.search_runs(experiment_names=[name])
        for _, run in runs.iterrows():
            started = datetime.fromtimestamp(run["start_time"].timestamp(), UTC)
            for column, value in run.items():
                if str(column).startswith("metrics.") and not math.isnan(float(value)):
                    rows.append(
                        {
                            "experiment": name,
                            "run_name": run.get("tags.mlflow.runName", ""),
                            "superseded": run.get("tags.superseded") == "true",
                            "started_utc": started.strftime("%Y-%m-%d %H:%M"),
                            "run_id": run["run_id"],
                            "metric": str(column).removeprefix("metrics."),
                            "value": f"{value:.6g}",
                        }
                    )
    rows.sort(key=lambda r: (r["experiment"], r["started_utc"], r["run_name"], r["metric"]))
    first: dict[
        tuple[str, str], str
    ] = {}  # the earliest run of each (experiment, run name) is the recorded one
    for row in rows:
        first.setdefault((row["experiment"], row["run_name"]), row["run_id"])
    for row in rows:
        is_first = first[(row["experiment"], row["run_name"])] == row["run_id"]
        row["recorded"] = "yes" if is_first and not row["superseded"] else "no"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            ["experiment", "run_name", "started_utc", "run_id", "recorded", "metric", "value"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"{out}: {len(rows)} rows from {len(EXPERIMENTS)} experiments")


if __name__ == "__main__":
    main()
