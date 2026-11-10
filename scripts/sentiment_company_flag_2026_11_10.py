# ruff: noqa: E402  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
"""Experiment `sentiment_pair_company_yes_no`: the PR #154 recipe (D35 pair, 2 epochs, lr 3e-5, seeds 1-4,
same train/validation marks, same replay) with one change: segment A says whether the company is named in
the text, so the model can learn the prior of the not-named texts instead of a hard gate forcing neutral.

    segment A = "<company> | company named in text"   /   "<company> | company not named in text"

The flag is on `Example.company`, so training (`encode_batch`) and scoring (`predict_probs(pair=True)`) both
carry it on every chunk. The flag is `sentiment_stage._text_mentions_subject(text, company, ticker)` on the
summary; the replay sentences are all "named" by construction (the replay keeps only those that name the
company). Run from the main checkout with PYTHONPATH=<PR #154 worktree>/src; reads data, writes only to
data/sentiment_pair_company_yes_no/, models/sentiment_pair_company_yes_no/ (deleted after each run) and MLflow.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

WT = Path(sys.argv[1])
MLRUNS = sys.argv[2]
sys.path[:0] = [str(WT / "src"), str(WT / "scripts")]

spec = importlib.util.spec_from_file_location(
    "pair_script", WT / "scripts/train_sentiment_pair_2026_11_09.py"
)
assert spec and spec.loader
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)

import mlflow

import sentiment_pair_train as spt
from news_nlp import env
from sentiment_stage import _text_mentions_subject

EXPERIMENT = "sentiment_pair_company_yes_no"
SETTING = (2, 3e-5)  # the grid's choice (D35 tie rule)
SEEDS = (1, 2, 3, 4)  # 1-3 are the grid cell's seeds, 4 the final run's
pair.OUT = Path("data/sentiment_pair_company_yes_no")
pair.MODELS = Path("models/sentiment_pair_company_yes_no")

env.load_env_file()
conn = pair.connect_source()
marked, replay, _ = pair.build_data(conn)
marks = pair.read_csv(pair.MARKS)
ticker = {m["key"]: m["ticker"] for m in marks}


def flagged(e: spt.Example, named: bool) -> spt.Example:
    suffix = "company named in text" if named else "company not named in text"
    return e._replace(company=f"{e.company} | {suffix}")


def flag_summary(e: spt.Example) -> spt.Example:
    return flagged(e, _text_mentions_subject(e.text, e.company, ticker[e.key]))


train = [flag_summary(e) for e in marked.train] + [flagged(e, True) for e in replay]
validation = [flag_summary(e) for e in marked.validation]
n_not = sum("not named" in e.company for e in train if e.kind == "summary")
print(
    f"train {len(train)}: summaries not named {n_not}; validation {len(validation)}: not named "
    f"{sum('not named' in e.company for e in validation)}"
)

scorer = pair.ValidationScorer(validation, marks)
mlflow.set_tracking_uri(MLRUNS)
mlflow.set_experiment(EXPERIMENT)
rows = []
for seed in SEEDS:
    settings = spt.RunSettings(*SETTING, seed=seed)
    row = pair.one_run(settings, train, scorer, mlflow_on=False, tag="flag")
    rows.append(row)
    with mlflow.start_run(run_name=f"flag__{settings.name}"):
        mlflow.set_tags(
            {
                "stage": "sentiment",
                "recipe": "D35 pair + company named yes/no in segment A",
                "kind": "flag",
            }
        )
        mlflow.log_params(
            {
                "epochs": settings.epochs,
                "learning_rate": settings.learning_rate,
                "seed": seed,
                "batch_size": spt.BATCH_SIZE,
                "weight_decay": spt.WEIGHT_DECAY,
                "n_train": len(train),
                "base_model": "{}@{}".format(*pair.base_model()),
                "flag": "segment A: '<company> | company [not] named in text'",
            }
        )
        mlflow.log_metric("validation_score", row["validation_score"])
        for label, figures in (row.get("validation_detail") or {}).items():
            for metric, value in figures.items():
                if value is not None:
                    mlflow.log_metric(f"{metric}_{label}", value)
        mlflow.log_metric("train_seconds", row["train_seconds"])
        for epoch, loss in enumerate(row["epoch_losses"], 1):
            mlflow.log_metric("train_loss", loss, step=epoch)
    print(f"seed {seed}: validation score {row['validation_score']:.4f}", flush=True)

# the same setting without the flag, from the PR's own results file (same data, same replay, same seeds)
old = json.loads(Path("data/sentiment_pair/grid_results.json").read_text(encoding="utf-8"))
old_rows = old["rows"]
base = {
    r["name"]: r["validation_score"]
    for r in old_rows
    if r.get("settings", {}).get("epochs") == SETTING[0]
    and r.get("settings", {}).get("learning_rate") == SETTING[1]
}
print("\nwithout the flag (PR #154):", {k: round(v, 4) for k, v in sorted(base.items())})
print("with the flag:             ", {r["name"]: round(r["validation_score"], 4) for r in rows})
