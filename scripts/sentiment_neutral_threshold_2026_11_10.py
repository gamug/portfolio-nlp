# ruff: noqa: B023, E731, S311, SIM115  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
"""Experiment `sentiment_pair_neutral_threshold`: re-decide the final-recipe predictions (2 ep, lr 3e-5, seeds 1-4,
no flag) with class multipliers on the probabilities: label = argmax(p_c * w_c), w_positive = 1, w_neutral and
w_negative searched. ROC OVR is threshold-free, so only precision/recall move. The weights are chosen on the
validation texts, so the honest figure is OUT-OF-FOLD: 5-fold (stratified by draw stratum), weights fitted on 4/5,
applied to the held-out 1/5, pooled, 10 repeats; the in-sample optimum is shown only as the optimistic bound."""

import csv
import itertools
import random
import statistics as st
import sys
from collections import defaultdict
from dataclasses import replace

sys.path.insert(0, sys.argv[1] + "/src")
import mlflow

import sentiment_pair_report as rep

MARKS = list(csv.DictReader(open("gold/sentiment_train.csv", encoding="utf-8-sig")))
RUNS = {
    1: "grid__e2_lr3e-05_s1",
    2: "grid__e2_lr3e-05_s2",
    3: "grid__e2_lr3e-05_s3",
    4: "final__e2_lr3e-05_s4",
}
GRID = list(
    itertools.product((0.25, 0.4, 0.6, 0.8, 1.0, 1.25), (1, 1.5, 2, 3, 4, 6, 8))
)  # (w_negative, w_neutral)
C = rep.CLASSES


def redo(scored, wn, wu):
    w = {"positive": 1.0, "negative": wn, "neutral": wu}
    return [replace(s, predicted=max(C, key=lambda c: s.probs[c] * w[c])) for s in scored]


def score(scored, pops):
    return rep.selection_score(rep.class_metrics(scored, rep.row_weights(scored, pops)))


def best(scored, pops):
    return max(GRID, key=lambda g: (score(redo(scored, *g), pops), -abs(g[0] - 1) - abs(g[1] - 1)))


def folds(scored, rng, k=5):
    by = defaultdict(list)
    for i, s in enumerate(scored):
        by[s.stratum].append(i)
    out = [[] for _ in range(k)]
    for idx in by.values():
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            out[j % k].append(i)
    return out


res = defaultdict(list)
mlflow.set_tracking_uri(sys.argv[2])
mlflow.set_experiment("sentiment_pair_neutral_threshold")
for seed, name in RUNS.items():
    rows = list(csv.DictReader(open(f"data/sentiment_pair/predictions/{name}.csv")))
    scored, pops = rep.scored_from_rows(rows, MARKS)
    base = score(scored, pops)
    g_in = best(scored, pops)
    s_in = score(redo(scored, *g_in), pops)
    oofs, chosen = [], []
    for rpt in range(10):
        rng = random.Random(1000 * seed + rpt)
        fl = folds(scored, rng)
        pred = {}
        for held in fl:
            tr = [scored[i] for f in fl if f is not held for i in f]
            g = best(tr, pops)
            chosen.append(g)
            for i in held:
                pred[i] = redo([scored[i]], *g)[0]
        oof = [pred[i] for i in range(len(scored))]
        oofs.append(oof)
    oof_scores = [score(o, pops) for o in oofs]
    gain = rep.weighted_gains(oofs[0], scored, pops)  # paired, repeat 0
    st_base = rep.statistics_of(scored, rep.row_weights(scored, pops))
    st_oof = [rep.statistics_of(o, rep.row_weights(o, pops)) for o in oofs]
    m = lambda k: st.mean(s[k] for s in st_oof)
    print(
        f"seed {seed}: base {base:.3f} | in-sample best (w_neg,w_neu)={g_in} {s_in:.3f} | out-of-fold {st.mean(oof_scores):.3f} "
        f"(gain {st.mean(oof_scores) - base:+.3f}; paired, repeat 0: {gain['score'][0]} {gain['score'][1]}) | "
        f"neutral P {st_base['neutral precision']:.2f}->{m('neutral precision'):.2f} R {st_base['neutral recall']:.2f}->{m('neutral recall'):.2f} | "
        f"negative P {st_base['negative precision']:.2f}->{m('negative precision'):.2f} R {st_base['negative recall']:.2f}->{m('negative recall'):.2f}",
        flush=True,
    )
    res["base"].append(base)
    res["oof"].append(st.mean(oof_scores))
    res["in"].append(s_in)
    with mlflow.start_run(run_name=f"threshold__{name}"):
        mlflow.set_tags({"stage": "sentiment", "kind": "decision_rule", "source_run": name})
        mlflow.log_params(
            {
                "rule": "argmax(p*w)",
                "w_neg_neu_in_sample": str(g_in),
                "folds": "5x10 stratified by stratum",
            }
        )
        mlflow.log_metrics(
            {
                "score_base": base,
                "score_in_sample": s_in,
                "score_out_of_fold": st.mean(oof_scores),
                "neutral_precision_oof": m("neutral precision"),
                "neutral_recall_oof": m("neutral recall"),
                "negative_precision_oof": m("negative precision"),
                "negative_recall_oof": m("negative recall"),
            }
        )
print(
    f"MEAN over seeds: base {st.mean(res['base']):.3f}  in-sample {st.mean(res['in']):.3f}  out-of-fold {st.mean(res['oof']):.3f}"
)
