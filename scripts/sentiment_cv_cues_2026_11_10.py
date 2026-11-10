# ruff: noqa: E402, E731, E741, PLC0206, PLR0124, PLR2004, S311, SIM115  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
"""Experiment `sentiment_pair_company_cues_cv`: the 809-text cross-validation of sentiment_cv_all_2026_11_10.py (same folds, same seeds, same
replay) with cues about the company written into segment A, computed alias-aware by the stage's own CompanyResolver:
  V1  "<company> | company named in text" / "not named in text"
  V2  "<company> | <not named | named only as the source of an opinion | named once | named 2+ times> | other companies: <none | 1-2 | 3+>"
Compared with the no-flag cross-validation (data/sentiment_pair_cv_all/predictions) by a paired weighted bootstrap."""

import csv
import glob
import importlib.util
import json
import random
import statistics as st
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
import size_triage_2026_10_28 as size_triage

import sentiment_pair_report as rep
import sentiment_pair_train as spt
from news_nlp import env, triage
from sentiment_stage import _text_mentions_subject

pair.OUT = Path("data/sentiment_pair_cues_cv")
pair.MODELS = Path("models/sentiment_pair_cues_cv")
env.load_env_file()
conn = pair.connect_source()
resolver = triage.CompanyResolver(size_triage.load_universe(conn))
sets = {
    "old": (pair.read_csv(pair.MARKS), pair.read_csv(pair.IDS)),
    "new": (
        pair.read_csv(Path("gold/sentiment_train_2.csv")),
        pair.read_csv(Path("data/sentiment_eval_ids.csv")),
    ),
}
marked = {d: spt.load_marked_summaries(m, i) for d, (m, i) in sets.items()}
meta = {
    r["key"]: (d, r["predicted"], int(r["pool_class_size"]), r["ticker"])
    for d, (m, _) in sets.items()
    for r in m
}
allex = [e for d in marked.values() for e in d.train + d.validation]
bodies = marked["old"].body_hashes | marked["new"].body_hashes
ids_all = sets["old"][1] + sets["new"][1]
titles_m = {k for r in ids_all if r["body_hash"] in bodies and (k := pair.headline_key(r["title"]))}
hashes, titles = pair.exam_keys(conn, bodies, titles_m)
sentences = [json.loads(l) for l in pair.REPLAY_POOL.read_text(encoding="utf-8").splitlines()]
infos = pair.article_infos(conn, {int(s["article_id"]) for s in sentences})
replay, _ = spt.replay_examples(
    sentences,
    infos.get,
    hashes,
    _text_mentions_subject,
    n=len(marked["old"].train),
    seed=pair.REPLAY_SEED,
    never_titles=titles,
)
ticker_of = {
    **{k: v[3] for k, v in meta.items()},
    **{e.key: infos[int(e.key.split("#")[0])].ticker for e in replay},
}


def cues(text, ticker):
    own = resolver.by_ticker.get(ticker.upper(), ticker)
    plain, quoted = resolver._text_mentions(text, True)
    others = len({t for t in set(plain) | set(quoted) if t != own})
    return plain.get(own, 0), quoted.get(own, 0), others


def v1(e):
    n, q, _ = cues(e.text, ticker_of[e.key])
    return e._replace(company=f"{e.company} | company {'named' if n + q else 'not named'} in text")


def v2(e):
    n, q, o = cues(e.text, ticker_of[e.key])
    level = (
        "not named"
        if n + q == 0
        else "named only as the source of an opinion"
        if n == 0
        else "named once"
        if n == 1
        else "named 2+ times"
    )
    return e._replace(
        company=f"{e.company} | {level} | other companies: {'none' if o == 0 else '1-2' if o <= 2 else '3+'}"
    )


from collections import Counter

for name, f in (("V1", v1), ("V2", v2)):
    print(name, "summaries:", dict(Counter(f(e).company.split(" | ", 1)[1] for e in allex)))

label = {e.key: spt.ID2LABEL[e.label] for e in allex}
byk = {e.key: e for e in allex}
cell = {k: meta[k][:2] for k in label}
pool = {meta[k][:2]: meta[k][2] for k in label}
n_cell = defaultdict(int)
for k in label:
    n_cell[cell[k]] += 1
weight = {k: pool[cell[k]] / n_cell[cell[k]] for k in label}


class CVScorer(pair.ValidationScorer):
    def measure(self, predictions):
        return {}, 0.0


def folds(keys, rng, k=5):
    by = defaultdict(list)
    for key in keys:
        by[cell[key]].append(key)
    out = [[] for _ in range(k)]
    for ks in by.values():
        rng.shuffle(ks)
        for j, key in enumerate(ks):
            out[j % k].append(key)
    return out


allmarks = [m for d in sets.values() for m in d[0]]
mlflow.set_tracking_uri(MLRUNS)
mlflow.set_experiment("sentiment_pair_company_cues_cv")
base = []
for rpt in (1, 2):
    rows = {}
    for p in glob.glob(f"data/sentiment_pair_cv_all/predictions/cv_r{rpt}_f*__*.csv"):
        for r in csv.DictReader(open(p)):
            rows[r["key"]] = r
    base.append(rows)
oofs = {"none": base}
for vname, f in (("V1", v1), ("V2", v2)):
    fx = [f(e) for e in allex]
    fr = [f(e) for e in replay]
    fbyk = {e.key: e for e in fx}
    runs = []
    for rpt in (1, 2):
        rng = random.Random(20261109 + rpt)
        fl = folds(sorted(label), rng)
        pred = {}
        for fi, held in enumerate(fl, 1):
            hs = set(held)
            train = [e for e in fx if e.key not in hs] + fr
            sc = CVScorer([fbyk[k] for k in held], allmarks)
            pair.one_run(spt.RunSettings(2, 3e-5, rpt), train, sc, tag=f"{vname}_r{rpt}_f{fi}")
            for p in sc.predictions:
                pred[p["key"]] = p
        runs.append(pred)
        print(vname, "repeat", rpt, "done", flush=True)
    oofs[vname] = runs
keys = sorted(set.intersection(*[set(r) for v in oofs.values() for r in v]))


def scored(pred, ks):
    return [
        rep.Scored(
            k,
            cell[k][1],
            label[k],
            pred[k]["predicted"],
            {c: float(pred[k][f"p_{c}"]) for c in rep.CLASSES},
        )
        for k in ks
    ]


def stats_of(preds, ks):
    w = [weight[k] for k in ks]
    ss = [rep.statistics_of(scored(p, ks), w) for p in preds]
    return {n: st.mean(s[n] for s in ss) for n in ss[0]}


SHOW = (
    "score",
    "macro precision",
    "macro recall",
    "macro roc_ovr",
    "neutral precision",
    "neutral recall",
    "negative precision",
    "negative recall",
)
pt = {v: stats_of(r, keys) for v, r in oofs.items()}
rng = random.Random(7)
byc = defaultdict(list)
for k in keys:
    byc[cell[k]].append(k)
gain = {v: defaultdict(list) for v in ("V1", "V2")}
for _ in range(200):
    res = [rng.choice(x) for x in byc.values() for _ in x]
    b = stats_of(oofs["none"], res)
    for v in gain:
        a = stats_of(oofs[v], res)
        for n in SHOW:
            gain[v][n].append(a[n] - b[n])
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print("\n== out-of-fold, weighted (809 texts, mean of 2 repeats) ==")
print(f"{'':20} {'no cue':>8} {'V1 yes/no':>10} {'V2 levels':>10}")
for n in SHOW:
    print(f"{n:20} {pt['none'][n]:8.3f} {pt['V1'][n]:10.3f} {pt['V2'][n]:10.3f}")
print("\n== paired gain over no cue, 95% bootstrap interval ==")
for v in ("V1", "V2"):
    print(v)
    for n in SHOW:
        g = gain[v][n]
        print(f"  {n:20} {pt[v][n] - pt['none'][n]:+.3f} [{q(g, 0.025):+.3f}, {q(g, 0.975):+.3f}]")
for v in pt:
    with mlflow.start_run(run_name=f"cv_{v}"):
        mlflow.set_tags({"stage": "sentiment", "kind": "cv", "cue": v})
        mlflow.log_params(
            {
                "epochs": 2,
                "learning_rate": 3e-5,
                "n_rows": len(keys),
                "cue": {
                    "none": "no cue",
                    "V1": "alias-aware named yes/no",
                    "V2": "named level (4) + other companies (3)",
                }[v],
            }
        )
        mlflow.log_metrics({n.replace(" ", "_"): x for n, x in pt[v].items() if x == x})
