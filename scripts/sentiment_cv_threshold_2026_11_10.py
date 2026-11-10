# ruff: noqa: B905, E731, PLR2004, S311, SIM115  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
"""Experiment `sentiment_pair_cv_threshold`: a decision rule label = argmax(p_c * w_c) (w_positive = 1; w_negative, w_neutral
searched) over the OUT-OF-FOLD predictions of the 809-text cross-validation (data/sentiment_pair_cv_all/predictions). ROC OVR is
threshold-free, so only precision and recall move; the search maximises (macro precision + macro recall) / 2, which is the part
of the validation score the rule can change. Two rules: 'free' (no constraint) and 'bar' (only weights that keep the
fitting texts' weighted negative recall >= 0.70, D32's bar). The honest figure is NESTED: weights fitted on 4/5 of the texts,
applied to the other 1/5, pooled, 10 outer repeats for each of the 2 cross-validation repeats."""

import csv
import glob
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1] + "/src")
import mlflow

import sentiment_pair_report as rep

C = rep.CLASSES  # positive, negative, neutral
GRID = [
    (wn, wu)
    for wn in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25)
    for wu in (1.0, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6)
]
BAR = 0.70

meta = {}
for d, path in (("old", "gold/sentiment_train.csv"), ("new", "gold/sentiment_train_2.csv")):
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        meta[r["key"]] = (d, r["predicted"], int(r["pool_class_size"]))
oof = []
for rpt in (1, 2):
    rows = {}
    for p in sorted(glob.glob(f"data/sentiment_pair_cv_all/predictions/cv_r{rpt}_f*__*.csv")):
        for r in csv.DictReader(open(p)):
            rows[r["key"]] = r
    oof.append(rows)
keys = sorted(set(oof[0]) & set(oof[1]))
cell = {k: meta[k][:2] for k in keys}
pool = {meta[k][:2]: meta[k][2] for k in keys}
n_cell = defaultdict(int)
for k in keys:
    n_cell[cell[k]] += 1
W = np.array([pool[cell[k]] / n_cell[cell[k]] for k in keys])
gold = np.array([C.index(oof[0][k]["expected"]) for k in keys])
P = [np.array([[float(oof[r][k][f"p_{c}"]) for c in C] for k in keys]) for r in (0, 1)]
print("texts:", len(keys))


def decide(Pm, wn, wu):
    return np.argmax(Pm * np.array([1.0, wn, wu]), axis=1)


def pr(dec, mask):
    """weighted per-class precision / recall on mask; undefined counts as 0 in the macro (as the PR does)."""
    prec, rec = [], []
    for c in range(3):
        hit = W[mask & (dec == c) & (gold == c)].sum()
        pred = W[mask & (dec == c)].sum()
        act = W[mask & (gold == c)].sum()
        prec.append(hit / pred if pred else 0.0)
        rec.append(hit / act if act else 0.0)
    return prec, rec


def fit(Pm, mask, constrained):
    best, best_key = (1.0, 1.0), -1
    for wn, wu in GRID:
        prec, rec = pr(decide(Pm, wn, wu), mask)
        if constrained and rec[1] < BAR:
            continue
        s = (sum(prec) + sum(rec)) / 6
        if s > best_key + 1e-12 or (
            abs(s - best_key) <= 1e-12
            and abs(wn - 1) + abs(wu - 1) < abs(best[0] - 1) + abs(best[1] - 1)
        ):
            best, best_key = (wn, wu), s
    return best


def folds(rng, k=5):
    by = defaultdict(list)
    for i, key in enumerate(keys):
        by[cell[key]].append(i)
    out = [[] for _ in range(k)]
    for idx in by.values():
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            out[j % k].append(i)
    return out


def full_stats(Pm, dec):
    sc = [
        rep.Scored(k, cell[k][1], C[gold[i]], C[dec[i]], dict(zip(C, Pm[i])))
        for i, k in enumerate(keys)
    ]
    return rep.statistics_of(sc, list(W))


SHOW = (
    "score",
    "macro precision",
    "macro recall",
    "neutral precision",
    "neutral recall",
    "negative precision",
    "negative recall",
    "positive precision",
    "positive recall",
)
base = [full_stats(P[r], P[r].argmax(1)) for r in (0, 1)]
base = {n: st.mean(b[n] for b in base) for n in SHOW}
res, chosen, nested_dec = {}, defaultdict(list), {}
for rule, constrained in (("free", False), ("bar", True)):
    stats_runs, dec_first = [], []
    for r in (0, 1):
        for rep_i in range(10):
            rng = random.Random(500 * r + rep_i)
            dec = np.zeros(len(keys), int)
            for held in folds(rng):
                mask = np.ones(len(keys), bool)
                mask[held] = False
                g = fit(P[r], mask, constrained)
                chosen[rule].append(g)
                dec[held] = decide(P[r][held], *g)
            stats_runs.append(full_stats(P[r], dec))
            if rep_i == 0:
                dec_first.append(dec)
    res[rule] = {n: st.mean(s[n] for s in stats_runs) for n in SHOW}
    nested_dec[rule] = dec_first
    allm = np.ones(len(keys), bool)
    ins = [fit(P[r], allm, constrained) for r in (0, 1)]
    res[rule]["in_sample_weights"] = ins
print("\n== nested (held-out) figures, weighted; mean of 2 CV repeats x 10 outer repeats ==")
print(f"{'':20} {'no rule':>8} {'free':>8} {'bar':>8}")
for n in SHOW:
    print(f"{n:20} {base[n]:8.3f} {res['free'][n]:8.3f} {res['bar'][n]:8.3f}")
for rule in ("free", "bar"):
    c = chosen[rule]
    print(
        f"\nweights chosen per outer fold, rule {rule}: w_neg median {st.median(g[0] for g in c)}, w_neu median {st.median(g[1] for g in c)}; "
        f"w_neu range {min(g[1] for g in c)}-{max(g[1] for g in c)}; share at grid edge (w_neu=6 or w_neg=0.5): {np.mean([g[1] == 6 or g[0] == 0.5 for g in c]):.0%}; in-sample on all 809: {res[rule]['in_sample_weights']}"
    )


# bootstrap of the nested gain over no rule (first outer repeat of each CV repeat)
def bstats(decs, ks_idx):
    out = []
    for r in (0, 1):
        sub = np.array(ks_idx)
        sc = [
            rep.Scored(keys[i], cell[keys[i]][1], C[gold[i]], C[decs[r][i]], dict(zip(C, P[r][i])))
            for i in sub
        ]
        out.append(rep.statistics_of(sc, [W[i] for i in sub]))
    return {n: st.mean(o[n] for o in out) for n in SHOW}


rng = random.Random(11)
byc = defaultdict(list)
for i, k in enumerate(keys):
    byc[cell[k]].append(i)
base_dec = [P[0].argmax(1), P[1].argmax(1)]
gains = {rule: defaultdict(list) for rule in ("free", "bar")}
for _ in range(200):
    idx = [rng.choice(v) for v in byc.values() for _ in v]
    b = bstats(base_dec, idx)
    for rule in gains:
        a = bstats(nested_dec[rule], idx)
        for n in SHOW:
            gains[rule][n].append(a[n] - b[n])
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print("\n== paired gain over no rule, 95% bootstrap interval (200 resamples) ==")
for rule in gains:
    print(rule)
    for n in (
        "score",
        "neutral recall",
        "neutral precision",
        "negative recall",
        "negative precision",
        "positive recall",
    ):
        g = gains[rule][n]
        print(f"  {n:20} {res[rule][n] - base[n]:+.3f} [{q(g, 0.025):+.3f}, {q(g, 0.975):+.3f}]")
# the weights for the final model: fit on all 809 (both CV repeats' probabilities pooled)
final = {}
for rule, constrained in (("free", False), ("bar", True)):
    best, best_s = (1.0, 1.0), -1
    for wn, wu in GRID:
        ss = []
        for r in (0, 1):
            prec, rec = pr(decide(P[r], wn, wu), np.ones(len(keys), bool))
            if constrained and rec[1] < BAR:
                ss = None
                break
            ss.append((sum(prec) + sum(rec)) / 6)
        if ss and st.mean(ss) > best_s:
            best, best_s = (wn, wu), st.mean(ss)
    final[rule] = {"w_positive": 1.0, "w_negative": best[0], "w_neutral": best[1]}
print("\nweights fitted on all 809 out-of-fold texts:", final)
Path("data/sentiment_pair_cv_all/decision_weights.json").write_text(
    json.dumps(
        {
            "rule": "label = argmax(p_c * w_c)",
            "fitted_on": "809 out-of-fold texts of the cv",
            "weights": final,
        },
        indent=1,
    )
)
mlflow.set_tracking_uri(sys.argv[2])
mlflow.set_experiment("sentiment_pair_cv_threshold")
for rule in ("free", "bar"):
    with mlflow.start_run(run_name=f"threshold_{rule}"):
        mlflow.set_tags({"stage": "sentiment", "kind": "decision_rule", "rule": rule})
        mlflow.log_params(
            {
                **{k: v for k, v in final[rule].items()},
                "constraint": "none" if rule == "free" else f"negative recall >= {BAR}",
                "evaluation": "nested 5-fold x10 x2 cv repeats",
            }
        )
        mlflow.log_metrics({f"nested_{n.replace(' ', '_')}": res[rule][n] for n in SHOW})
        mlflow.log_metrics({f"base_{n.replace(' ', '_')}": base[n] for n in SHOW})
