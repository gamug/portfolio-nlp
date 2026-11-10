# ruff: noqa: E402, PLR2004, S311  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
"""Experiment `sentiment_pair_learning_curve`: the PR #154 recipe (2 epochs, lr 3e-5) on 25/50/75% of the
training data (summaries and replay both, stratified by class), seeds 1-3, same validation slice. 100% is the
PR's own grid cell (seeds 1-3), read from its results file."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

WT = Path(sys.argv[1])
MLRUNS = sys.argv[2]
sys.path[:0] = [str(WT / "src"), str(WT / "scripts")]
spec = importlib.util.spec_from_file_location(
    "pair_script", WT / "scripts/train_sentiment_pair_2026_11_09.py"
)
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)
import mlflow

import sentiment_pair_train as spt
from news_nlp import env

pair.OUT = Path("data/sentiment_pair_learning_curve")
pair.MODELS = Path("models/sentiment_pair_learning_curve")
env.load_env_file()
marked, replay, _ = pair.build_data(pair.connect_source())
marks = pair.read_csv(pair.MARKS)


def subsample(examples, frac, seed):
    by = defaultdict(list)
    for e in examples:
        by[e.label].append(e)
    rng = random.Random(seed)
    out = []
    for label in sorted(by):
        rows = by[label][:]
        rng.shuffle(rows)
        out += rows[: round(len(rows) * frac)]
    return out


scorer = pair.ValidationScorer(marked.validation, marks)
mlflow.set_tracking_uri(MLRUNS)
mlflow.set_experiment("sentiment_pair_learning_curve")
res = {}
for frac in (0.25, 0.5, 0.75):
    for seed in (1, 2, 3):
        train = subsample(marked.train, frac, seed) + subsample(replay, frac, seed)
        s = spt.RunSettings(2, 3e-5, seed)
        row = pair.one_run(s, train, scorer, tag=f"frac{int(frac * 100)}")
        res.setdefault(frac, []).append(row["validation_score"])
        with mlflow.start_run(run_name=f"frac{int(frac * 100)}__{s.name}"):
            mlflow.set_tags({"stage": "sentiment", "kind": "learning_curve", "fraction": str(frac)})
            mlflow.log_params(
                {
                    "fraction": frac,
                    "epochs": 2,
                    "learning_rate": 3e-5,
                    "seed": seed,
                    "n_train": len(train),
                }
            )
            mlflow.log_metric("validation_score", row["validation_score"])
            for label, figs in (row.get("validation_detail") or {}).items():
                for k, v in figs.items():
                    if v is not None:
                        mlflow.log_metric(f"{k}_{label}", v)
        print(f"frac {frac} seed {seed} n {len(train)}: {row['validation_score']:.4f}", flush=True)
old = json.loads(Path("data/sentiment_pair/grid_results.json").read_text())["rows"]
res[1.0] = [
    r["validation_score"]
    for r in old
    if r["tag"] == "grid"
    and r["settings"]["epochs"] == 2
    and r["settings"]["learning_rate"] == 3e-5
]
import statistics as st

for f in sorted(res):
    print(
        f"RESULT {f:4.0%}: mean {st.mean(res[f]):.3f} sd {st.stdev(res[f]):.3f}  {[round(x, 3) for x in res[f]]}"
    )
