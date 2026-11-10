# ruff: noqa: E402, E731, E741, PLR0124, S311  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
"""Experiment `sentiment_pair_cv_all_data`: 5-fold cross-validation (2 repeats, different splits) of the PR #154 recipe
(2 epochs, lr 3e-5) over ALL usable marked summaries (gold/sentiment_train.csv + gold/sentiment_train_2.csv), the same 341
replay sentences in every training set (they exclude every marked body). Out-of-fold predictions -> weighted precision,
recall, ROC OVR per class (each text weighs pool size / drawn count of its draw and predicted-class stratum), bootstrap
intervals, and the paired gain over the pinned text-only model on the same texts. The model on all rows is
`train_sentiment_pair_2026_11_09.py --final-all` (no held-out score of its own: the CV figures are its estimate).
REPLAY_N=<n> sets the number of replay sentences (default: as many as the first file's training rows, 341, about
30% of the mix; the grid's mix was 50%), and then names the outputs `..._replay<n>`."""

import importlib.util
import json
import os
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
import draw_sentiment_gold_2026_11_05 as gold_draw
import mlflow

import sentiment_pair_report as rep
import sentiment_pair_train as spt
from news_nlp import env
from sentiment_stage import _text_mentions_subject

SUFFIX = f"_replay{os.environ['REPLAY_N']}" if os.environ.get("REPLAY_N") else ""
if os.environ.get("BASE_MODEL"):  # another starting point, e.g. ProsusAI/finbert@<revision>
    SUFFIX += "_base_" + os.environ["BASE_MODEL"].split("@")[0].split("/")[-1].lower()
pair.OUT = Path(f"data/sentiment_pair_cv_all{SUFFIX}")
pair.MODELS = Path(f"models/sentiment_pair_all{SUFFIX}")
env.load_env_file()
conn = pair.connect_source()
sets = {d: (pair.read_csv(marks), pair.read_csv(ids)) for d, (marks, ids) in pair.MARK_SETS.items()}
marked = {d: spt.load_marked_summaries(m, i) for d, (m, i) in sets.items()}
meta = {
    r["key"]: (d, r["predicted"], int(r["pool_class_size"]))
    for d, (m, _) in sets.items()
    for r in m
}
allex = [e for d in marked.values() for e in d.train + d.validation]
print("rows used:", len(allex), {d: len(v.train) + len(v.validation) for d, v in marked.items()})
# replay: 341 as built, excluding the bodies/headlines of both sets
bodies = marked["old"].body_hashes | marked["new"].body_hashes
ids_all = sets["old"][1] + sets["new"][1]
titles_m = {k for r in ids_all if r["body_hash"] in bodies and (k := pair.headline_key(r["title"]))}
hashes, titles = pair.exam_keys(conn, bodies, titles_m)
sentences = [json.loads(l) for l in pair.REPLAY_POOL.read_text(encoding="utf-8").splitlines()]
infos = pair.article_infos(conn, {int(s["article_id"]) for s in sentences})
replay, report = spt.replay_examples(
    sentences,
    infos.get,
    hashes,
    _text_mentions_subject,
    n=int(os.environ.get("REPLAY_N") or len(marked["old"].train)),
    seed=pair.REPLAY_SEED,
    never_titles=titles,
)
print(report.lines()[-1] if report.lines() else "")
label = {e.key: spt.ID2LABEL[e.label] for e in allex}
text = {e.key: e for e in allex}
cell = {k: (meta[k][0], meta[k][1]) for k in label}  # (draw, stratum)
cell_pool = {(d, s): p for (d, s, p) in meta.values()}
cell_n = defaultdict(int)
for k in label:
    cell_n[cell[k]] += 1
weight = {k: cell_pool[cell[k]] / cell_n[cell[k]] for k in label}


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


mlflow.set_tracking_uri(MLRUNS)
mlflow.set_experiment(f"sentiment_pair_cv_all_data{SUFFIX}")
oof = []  # one dict key -> prediction row per repeat
for rpt in (1, 2):
    rng = random.Random(20261109 + rpt)
    fl = folds(sorted(label), rng)
    pred = {}
    for f, held in enumerate(fl, 1):
        held_set = set(held)
        train = [e for e in allex if e.key not in held_set] + replay
        scorer = CVScorer([text[k] for k in held], [m for d in sets.values() for m in d[0]])
        s = spt.RunSettings(2, 3e-5, rpt)
        pair.one_run(s, train, scorer, tag=f"cv_r{rpt}_f{f}")
        for p in scorer.predictions:
            pred[p["key"]] = p
        print(f"repeat {rpt} fold {f}: held out {len(held)}, trained on {len(train)}", flush=True)
    oof.append(pred)

# pinned text-only model on the same texts
items = [{"key": e.key, "text": e.text, "company": e.company} for e in allex]
pin = {
    t["key"]: {
        "key": t["key"],
        "expected": label[t["key"]],
        "predicted": t["predicted"],
        **{f"p_{c}": repr(float(t["probs"][c])) for c in rep.CLASSES},
    }
    for t in gold_draw.predict_probs(items, None, pair=False)
}
keys = sorted(set(pin) & set(oof[0]) & set(oof[1]))
print("texts scored by all:", len(keys), "of", len(label))


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
    sc = [scored(p, ks) for p in preds]
    w = [weight[k] for k in ks]
    ss = [rep.statistics_of(x, w) for x in sc]
    return {n: st.mean(s[n] for s in ss) for n in ss[0]}


def boot(ks, n=400, seed=7):
    rng = random.Random(seed)
    by = defaultdict(list)
    for k in ks:
        by[cell[k]].append(k)
    cand, gain, base = defaultdict(list), defaultdict(list), defaultdict(list)
    for _ in range(n):
        res = [rng.choice(v) for v in by.values() for _ in v]
        # a resample keeps each cell's size, so the cell weights are unchanged: weight by the original key's cell
        a, b = stats_of(oof, res), stats_of([pin], res)
        for name in a:
            cand[name].append(a[name])
            base[name].append(b[name])
            gain[name].append(a[name] - b[name])
    return cand, base, gain


point, pinp = stats_of(oof, keys), stats_of([pin], keys)
cand, base, gain = boot(keys)
q = lambda v, p: sorted(x for x in v if x == x)[int(p * (len([x for x in v if x == x]) - 1))]
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
print(
    "\n== out-of-fold, weighted; 95% bootstrap interval; paired gain over the pinned text-only model =="
)
for n in SHOW:
    print(
        f"{n:20} cv {point[n]:.3f} [{q(cand[n], 0.025):.3f}, {q(cand[n], 0.975):.3f}]   pinned {pinp[n]:.3f}   gain {point[n] - pinp[n]:+.3f} [{q(gain[n], 0.025):+.3f}, {q(gain[n], 0.975):+.3f}]"
    )
with mlflow.start_run(run_name="cv_summary"):
    mlflow.set_tags({"stage": "sentiment", "kind": "cv", "folds": "5 x 2 repeats"})
    mlflow.log_params(
        {"epochs": 2, "learning_rate": 3e-5, "n_rows": len(keys), "replay": len(replay)}
    )
    mlflow.log_metrics({f"cv_{n.replace(' ', '_')}": v for n, v in point.items() if v == v})
    mlflow.log_metrics({f"pinned_{n.replace(' ', '_')}": v for n, v in pinp.items() if v == v})
    mlflow.log_metrics(
        {
            f"gain_{n.replace(' ', '_')}": point[n] - pinp[n]
            for n in point
            if point[n] == point[n] and pinp[n] == pinp[n]
        }
    )

# the model on all rows is made by `train_sentiment_pair_2026_11_09.py --final-all` (one implementation).
