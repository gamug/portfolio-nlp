# ruff: noqa: PLR2004, PLR0124, PLW2901  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
"""Company-mention split of the 450 sentiment marks + the "no mention -> neutral" gate on the 83
validation texts. Read-only on data; logs to MLflow experiment `sentiment_pair_company_gate`.

Run from the repository root: `<root> <mlruns>` as arguments, like the other scripts of this set.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import mlflow

import sentiment_pair_report as rep
from sentiment_stage import _text_mentions_subject

MARKS = Path("gold/sentiment_train.csv")
IDS = Path("data/sentiment_train_ids.csv")
PRED = Path("data/sentiment_pair/predictions")
EXPERIMENT = "sentiment_pair_company_gate"
CLASSES = ("positive", "negative", "neutral")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


marks = read(MARKS)
ids = {r["key"]: r for r in read(IDS)}
for m in marks:
    r = ids[m["key"]]
    m["text"], m["title"] = r["text"], r["title"]
    # three readings of "the company is named": in the summary the pair model reads; in the
    # headline; in either (what a gate on the article could see)
    m["in_summary"] = _text_mentions_subject(m["text"], m["company"], m["ticker"])
    m["in_title"] = _text_mentions_subject(m["title"], m["company"], m["ticker"])
    m["in_either"] = m["in_summary"] or m["in_title"]

out: dict[str, object] = {}


def dist(rows: list[dict[str, str]]) -> dict[str, int]:
    c = Counter(r["sentiment"] for r in rows)
    return {k: c.get(k, 0) for k in CLASSES}


print("== gold labels by whether the company is named (all 450 marks) ==")
for reading in ("in_summary", "in_title", "in_either"):
    yes = [m for m in marks if m[reading]]
    no = [m for m in marks if not m[reading]]
    print(f"{reading:11} named: n={len(yes):3} {dist(yes)}   not named: n={len(no):3} {dist(no)}")
    out[reading] = {"named": dist(yes), "not_named": dist(no)}

val_marks = [m for m in marks if m["slice"] == "validation"]
print("\n== same, validation slice only (83) ==")
for reading in ("in_summary", "in_either"):
    yes = [m for m in val_marks if m[reading]]
    no = [m for m in val_marks if not m[reading]]
    print(f"{reading:11} named: n={len(yes):3} {dist(yes)}   not named: n={len(no):3} {dist(no)}")

# --- the models on each group, and the gate ---------------------------------------------------
named = {m["key"]: m["in_summary"] for m in marks}
readings = {
    "final": read(PRED / "final__e2_lr3e-05_s4.csv"),
    "pinned_text": read(PRED / "baseline_text_only.csv"),
    "pinned_pair": read(PRED / "baseline_pair.csv"),
}


def gated(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Not named -> neutral (probabilities set to a certain neutral); named -> the model's own."""
    g = []
    for r in rows:
        r = dict(r)
        if not named[r["key"]]:
            r.update(predicted="neutral", p_positive="0.0", p_negative="0.0", p_neutral="1.0")
        g.append(r)
    return g


def acc(rows: list[dict[str, str]]) -> str:
    return f"{sum(r['expected'] == r['predicted'] for r in rows)}/{len(rows)}"


print("\n== raw hits of each reading on the validation texts, by group ==")
for name, rows in readings.items():
    yes = [r for r in rows if named[r["key"]]]
    no = [r for r in rows if not named[r["key"]]]
    print(
        f"{name:12} named {acc(yes):>6}  not named {acc(no):>6}  all {acc(rows):>6}  | gated all {acc(gated(rows))}"
    )

print("\n== weighted precision / recall / ROC OVR (PLAN D37), without and with the gate ==")
results: dict[str, dict[str, float]] = {}
for name, rows in readings.items():
    for variant, rr in (("plain", rows), ("gated", gated(rows))):
        scored, pops = rep.scored_from_rows(rr, marks)
        stats = rep.statistics_of(scored, rep.row_weights(scored, pops))
        results[f"{name}/{variant}"] = stats
        print(
            f"{name:12} {variant:6} score {stats['score']:.3f}  macro P {stats['macro precision']:.3f} "
            f"R {stats['macro recall']:.3f} ROC {stats['macro roc_ovr']:.3f} | neutral P "
            f"{stats['neutral precision']:.3f} R {stats['neutral recall']:.3f}"
        )

# --- MLflow ------------------------------------------------------------------------------------
uri = sys.argv[2] if len(sys.argv) > 2 else "mlruns"
mlflow.set_tracking_uri(uri)
mlflow.set_experiment(EXPERIMENT)
for key, stats in results.items():
    name, variant = key.split("/")
    with mlflow.start_run(run_name=f"{name}__{variant}"):
        mlflow.set_tags(
            {"reading": name, "gate": variant, "named_rule": "in_summary", "n_validation": "83"}
        )
        mlflow.log_params(
            {
                "rule": "company/ticker named in the summary (sentiment_stage._text_mentions_subject)",
                "gate": "not named -> neutral" if variant == "gated" else "none",
            }
        )
        mlflow.log_metrics(
            {k.replace(" ", "_"): v for k, v in stats.items() if v == v}  # drops NaN
        )
with mlflow.start_run(run_name="split_of_450_marks"):
    mlflow.set_tag("kind", "split")
    mlflow.log_dict(out, "gold_by_company_named.json")
print(f"\nlogged to {uri} / {EXPERIMENT}")
