# ruff: noqa: B905, E402, E731, PLR0124, S311, SIM115  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
"""Experiment `sentiment_pair_relevance_gate`: (1) fine-tune the category stage's NLI model (deberta-v3-base-zeroshot-v2.0)
on "is the story about the company": premise = summary, hypothesis = "This text is mainly about <company>.",
entailment = the owner marked a direction, not_entailment = neutral. Same 5-fold splits and seeds as the sentiment
cross-validation (cv_all.py), so every text gets an out-of-fold relevance score AND an out-of-fold sentiment reading.
(2) Combine them, evaluated NESTED (rule fitted on 4/5 of the out-of-fold rows, applied to the other 1/5):
  gate   label = argmax(p) unless relevance < t, then neutral          (t fitted)
  stack  multinomial logistic regression on [log p_pos, log p_neg, log p_neu, logit(relevance)]
Weighted precision / recall / ROC OVR as in the PR; paired bootstrap against the sentiment model alone."""

import csv
import glob
import importlib.util
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

WT = sys.argv[1]
sys.path[:0] = [WT + "/src", WT + "/scripts"]
spec = importlib.util.spec_from_file_location(
    "pair_script", WT + "/scripts/train_sentiment_pair_2026_11_09.py"
)
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)
import mlflow
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import sentiment_pair_report as rep

cfg = json.load(open("config/pipeline_models.json"))["category"]
sets = {d: (str(m), str(i)) for d, (m, i) in pair.MARK_SETS.items()}
marks, ids = {}, {}
for d, (mp, ip) in sets.items():
    for r in csv.DictReader(open(mp, encoding="utf-8-sig")):
        marks[r["key"]] = {**r, "draw": d}
    for r in csv.DictReader(open(ip, encoding="utf-8-sig")):
        ids[r["key"]] = r
oof = []
for rpt in (1, 2):
    rows = {}
    for p in glob.glob(f"data/sentiment_pair_cv_all/predictions/cv_r{rpt}_f*__*.csv"):
        for r in csv.DictReader(open(p)):
            rows[r["key"]] = r
    oof.append(rows)
keys = sorted(set(oof[0]) & set(oof[1]))
C = list(rep.CLASSES)  # positive, negative, neutral
cell = {k: (marks[k]["draw"], marks[k]["predicted"]) for k in keys}
pool = {cell[k]: int(marks[k]["pool_class_size"]) for k in keys}
ncell = defaultdict(int)
for k in keys:
    ncell[cell[k]] += 1
W = np.array([pool[cell[k]] / ncell[cell[k]] for k in keys])
gold = np.array([C.index(marks[k]["sentiment"]) for k in keys])
P = [np.array([[float(o[k][f"p_{c}"]) for c in C] for k in keys]) for o in oof]
rel = np.array([marks[k]["sentiment"] != "neutral" for k in keys])
idx_of = {k: i for i, k in enumerate(keys)}


def folds(rng, k=5):  # identical to cv_all.py: sorted keys, grouped by (draw, stratum)
    by = defaultdict(list)
    for key in keys:
        by[cell[key]].append(key)
    out = [[] for _ in range(k)]
    for ks in by.values():
        rng.shuffle(ks)
        for j, key in enumerate(ks):
            out[j % k].append(key)
    return out


tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])


def hyp(k):
    return f"This text is mainly about {marks[k]['company'].split(' (')[0]}."


def enc(ks):
    return tok(
        [ids[k]["text"] for k in ks],
        [hyp(k) for k in ks],
        truncation="only_first",
        max_length=320,
        padding=True,
        return_tensors="pt",
    )


def finetune_and_score(train_keys, test_keys, seed, epochs=3, lr=2e-5, bs=16):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], revision=cfg["revision"], dtype=torch.float32
    ).cuda()  # transformers 5 loads fp16 by default: AdamW underflows to NaN
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train_keys) + bs - 1) // bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    for _ in range(epochs):
        ks = train_keys[:]
        rng.shuffle(ks)
        model.train()
        for i in range(0, len(ks), bs):
            b = ks[i : i + bs]
            x = enc(b).to("cuda")
            y = torch.tensor(
                [0 if rel[idx_of[k]] else 1 for k in b], device="cuda"
            )  # 0 = entailment
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(**x).logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
    model.eval()
    out = {}
    with torch.no_grad():
        for i in range(0, len(test_keys), 32):
            b = test_keys[i : i + 32]
            x = enc(b).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr = torch.softmax(model(**x).logits.float(), -1)[:, 0].cpu().tolist()
            out.update(dict(zip(b, pr)))
    del model
    torch.cuda.empty_cache()
    return out


#: CONTROL=1 reuses the saved out-of-fold relevance scores and shuffles them across the texts: the same nested
#: procedure then has no relevance information left, so any gain over the sentiment model alone is the procedure's
#: own (the stacking leak of docs/sentiment_training.md), not the relevance model's.
CONTROL = os.environ.get("CONTROL") == "1"
SAVED = Path("data/sentiment_pair_relevance_cv/oof_relevance.json")
if CONTROL:
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    assert saved["keys"] == list(keys), "the saved scores are for other texts"
    perm = np.random.default_rng(31).permutation(len(keys))
    R = [np.array(r_)[perm] for r_ in saved["R"]]
else:
    R = []  # per repeat: out-of-fold relevance score per key index
    for rpt in (1, 2):
        fl = folds(random.Random(20261109 + rpt))
        score = np.zeros(len(keys))
        for f, held in enumerate(fl, 1):
            hs = set(held)
            tr = [k for k in keys if k not in hs]
            got = finetune_and_score(tr, held, rpt)
            for k, v in got.items():
                score[idx_of[k]] = v
            print(f"repeat {rpt} fold {f} done", flush=True)
        R.append(score)
    SAVED.parent.mkdir(parents=True, exist_ok=True)
    SAVED.write_text(
        json.dumps({"keys": list(keys), "R": [r_.tolist() for r_ in R]}), encoding="utf-8"
    )


def wauc(s, pos):
    return rep.weighted_roc_auc(list(s), list(pos), list(W))


print(
    f"\nrelevance AUC (weighted), fine-tuned out-of-fold: {[round(wauc(r, rel), 3) for r in R]}  (zero-shot alone 0.743, sentiment model 1-p_neutral 0.795)"
)


def pr_fig(dec, mask):
    prec, rec = [], []
    for c in range(3):
        hit = W[mask & (dec == c) & (gold == c)].sum()
        pred = W[mask & (dec == c)].sum()
        act = W[mask & (gold == c)].sum()
        prec.append(hit / pred if pred else 0.0)
        rec.append(hit / act if act else 0.0)
    return (sum(prec) + sum(rec)) / 6


def gate_dec(Pm, Rm, t):
    d = Pm.argmax(1)
    d[Rm < t] = 2
    return d


TS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def feats(Pm, Rm):
    return np.column_stack(
        [
            np.log(np.clip(Pm, 1e-6, 1)),
            np.log(np.clip(Rm, 1e-6, 1 - 1e-6) / (1 - np.clip(Rm, 1e-6, 1 - 1e-6))),
        ]
    )


def full_stats(Pm, dec):
    sc = [
        rep.Scored(keys[i], cell[keys[i]][1], C[gold[i]], C[dec[i]], dict(zip(C, Pm[i])))
        for i in range(len(keys))
    ]
    return rep.statistics_of(sc, list(W))


SHOW = (
    "score",
    "macro precision",
    "macro recall",
    "macro roc_ovr",
    "neutral precision",
    "neutral recall",
    "negative precision",
    "negative recall",
    "positive precision",
    "positive recall",
)


def nested(Rr, outer=10):
    res = {"none": [], "gate": [], "stack": []}
    dec_keep = {"none": [], "gate": [], "stack": []}
    prob_keep = {"none": [], "gate": [], "stack": []}
    chosen_t = []
    for r in (0, 1):
        res["none"].append(full_stats(P[r], P[r].argmax(1)))
        dec_keep["none"].append(P[r].argmax(1))
        prob_keep["none"].append(P[r])
        for o in range(outer):
            rng = random.Random(900 * r + o)
            fl = folds(rng)
            dg = np.zeros(len(keys), int)
            ds = np.zeros(len(keys), int)
            Ps = np.zeros_like(P[r])
            for held in fl:
                hi = np.array([idx_of[k] for k in held])
                mask = np.ones(len(keys), bool)
                mask[hi] = False
                t = max(TS, key=lambda t: (pr_fig(gate_dec(P[r], Rr[r], t), mask), -abs(t - 0.5)))
                chosen_t.append(t)
                dg[hi] = gate_dec(P[r][hi], Rr[r][hi], t)
                X = feats(P[r], Rr[r])
                lr = LogisticRegression(C=1.0, max_iter=500).fit(
                    X[mask], gold[mask], sample_weight=W[mask] / W[mask].mean()
                )
                Ps[hi] = lr.predict_proba(X[hi])
                ds[hi] = Ps[hi].argmax(1)
            if o == 0:
                dec_keep["gate"].append(dg)
                dec_keep["stack"].append(ds)
                prob_keep["gate"].append(P[r])
                prob_keep["stack"].append(Ps)
            res["gate"].append(full_stats(P[r], dg))
            res["stack"].append(full_stats(Ps, ds))
    return res, dec_keep, prob_keep, chosen_t


res, dec_keep, prob_keep, chosen_t = nested(R)
SEEDS = int(os.environ.get("CONTROL_SEEDS", "0"))
if (
    SEEDS
):  # many shuffles of the saved scores: the spread of the gain a signal-free relevance score gives
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    assert saved["keys"] == list(keys), "the saved scores are for other texts"
    gains_by_seed = {m: {n: [] for n in SHOW} for m in ("gate", "stack")}
    for seed in range(SEEDS):
        perm = np.random.default_rng(1000 + seed).permutation(len(keys))
        rs, *_ = nested([np.array(r_)[perm] for r_ in saved["R"]], outer=2)
        base = {n: st.mean(x[n] for x in rs["none"]) for n in SHOW}
        for m in gains_by_seed:
            for n in SHOW:
                gains_by_seed[m][n].append(st.mean(x[n] for x in rs[m]) - base[n])
        print(
            f"shuffle {seed + 1}/{SEEDS}: stack score {gains_by_seed['stack']['score'][-1]:+.3f}",
            flush=True,
        )
    print(
        f"\n== gain over the sentiment model alone with SHUFFLED relevance, {SEEDS} shuffles (2 outer x 2 CV repeats each) =="
    )
    for m, per in gains_by_seed.items():
        for n in (
            "score",
            "neutral recall",
            "neutral precision",
            "negative recall",
            "negative precision",
        ):
            v = per[n]
            print(
                f"{m:6} {n:20} mean {st.mean(v):+.3f}  sd {st.stdev(v):.3f}  range [{min(v):+.3f}, {max(v):+.3f}]"
            )
    mlflow.set_tracking_uri(sys.argv[2])
    mlflow.set_experiment("sentiment_pair_relevance_gate")
    with mlflow.start_run(run_name=f"control_shuffled_{SEEDS}_seeds"):
        mlflow.set_tags({"stage": "sentiment", "kind": "relevance", "control": "True"})
        for m, per in gains_by_seed.items():
            for n in SHOW:
                mlflow.log_metric(f"gain_{m}_{n.replace(' ', '_')}_mean", st.mean(per[n]))
                mlflow.log_metric(f"gain_{m}_{n.replace(' ', '_')}_sd", st.stdev(per[n]))
    sys.exit(0)
avg = {m: {n: st.mean(s[n] for s in v) for n in SHOW} for m, v in res.items()}
print(
    f"\n== nested held-out, weighted (809 texts; 2 CV repeats x 10 outer repeats) ==   gate threshold chosen: median {st.median(chosen_t)}, range {min(chosen_t)}-{max(chosen_t)}"
)
print(f"{'':20} {'sentiment only':>15} {'gate':>8} {'stack':>8}")
for n in SHOW:
    print(f"{n:20} {avg['none'][n]:15.3f} {avg['gate'][n]:8.3f} {avg['stack'][n]:8.3f}")
rng = random.Random(11)
byc = defaultdict(list)
for i, k in enumerate(keys):
    byc[cell[k]].append(i)


def bst(m, idx):
    out = []
    for r in (0, 1):
        sc = [
            rep.Scored(
                keys[i],
                cell[keys[i]][1],
                C[gold[i]],
                C[dec_keep[m][r][i]],
                dict(zip(C, prob_keep[m][r][i])),
            )
            for i in idx
        ]
        out.append(rep.statistics_of(sc, [W[i] for i in idx]))
    return {n: st.mean(o[n] for o in out) for n in SHOW}


gains = {m: defaultdict(list) for m in ("gate", "stack")}
for _ in range(200):
    idx = [rng.choice(v) for v in byc.values() for _ in v]
    b = bst("none", idx)
    for m in gains:
        a = bst(m, idx)
        for n in SHOW:
            gains[m][n].append(a[n] - b[n])
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print(
    "\n== paired gain over the sentiment model alone, 95% bootstrap interval (first outer repeat) =="
)
for m in gains:
    print(m)
    for n in (
        "score",
        "neutral recall",
        "neutral precision",
        "negative recall",
        "negative precision",
        "macro roc_ovr",
    ):
        g = gains[m][n]
        print(
            f"  {n:20} {avg[m][n] - avg['none'][n]:+.3f} [{q(g, 0.025):+.3f}, {q(g, 0.975):+.3f}]"
        )
mlflow.set_tracking_uri(sys.argv[2])
mlflow.set_experiment("sentiment_pair_relevance_gate")
for m in res:
    with mlflow.start_run(run_name=f"{'control_shuffled_' if CONTROL else ''}nested_{m}"):
        mlflow.set_tags(
            {
                "stage": "sentiment",
                "kind": "relevance",
                "combination": m,
                "control": str(CONTROL),
            }
        )
        mlflow.log_params(
            {
                "relevance_model": f"{cfg['model']}@{cfg['revision'][:8]}",
                "hypothesis": "This text is mainly about <company>.",
                "epochs": 3,
                "lr": 2e-5,
                "evaluation": "nested 5-fold x10 x2",
            }
        )
        mlflow.log_metrics({n.replace(" ", "_"): v for n, v in avg[m].items() if v == v})
        if m == "none":
            mlflow.log_metric("relevance_auc", st.mean(wauc(r_, rel) for r_ in R))
