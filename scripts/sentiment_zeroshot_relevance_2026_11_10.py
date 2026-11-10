# ruff: noqa: E402, E731, E741, RUF015, SIM115  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
"""Does the zero-shot NLI model of the category stage (MoritzLaurer/deberta-v3-base-zeroshot-v2.0, pinned revision) tell
"the story is about the company's own business" from "it is not"? Score = P(entailment) of hypothesis H given the premise.
Target: non-neutral (the owner marked a direction, so the story is about the company) vs neutral (about 70% not about it).
Metric: weighted ROC AUC (each text weighs pool size / drawn of its draw and stratum, as in the cross-validation), against
the no-cue cross-validated model's own ranking (1 - p_neutral, out-of-fold, mean of 2 repeats) and a rank average of both."""

import csv
import glob
import importlib.util
import json
import sys

import numpy as np
import torch

WT = sys.argv[1]
sys.path[:0] = [WT + "/src", WT + "/scripts"]
spec = importlib.util.spec_from_file_location(
    "pair_script", WT + "/scripts/train_sentiment_pair_2026_11_09.py"
)
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)
import mlflow
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sentiment_pair_report as rep
from news_nlp import env

env.load_env_file()
conn = pair.connect_source()
cfg = json.load(open("config/pipeline_models.json"))["category"]
sets = {
    "old": ("gold/sentiment_train.csv", "data/sentiment_train_ids.csv"),
    "new": ("gold/sentiment_train_2.csv", "data/sentiment_eval_ids.csv"),
}
marks, ids = {}, {}
for d, (mp, ip) in sets.items():
    for r in csv.DictReader(open(mp, encoding="utf-8-sig")):
        marks[r["key"]] = {**r, "draw": d}
    for r in csv.DictReader(open(ip, encoding="utf-8-sig")):
        ids[r["key"]] = r
# the 809 pair-readable texts of the cross-validation, and the cv model's out-of-fold p_neutral
oof = []
for rpt in (1, 2):
    rows = {}
    for p in glob.glob(f"data/sentiment_pair_cv_all/predictions/cv_r{rpt}_f*__*.csv"):
        for r in csv.DictReader(open(p)):
            rows[r["key"]] = r
    oof.append(rows)
keys = sorted(set(oof[0]) & set(oof[1]))
cell = {k: (marks[k]["draw"], marks[k]["predicted"]) for k in keys}
pool = {cell[k]: int(marks[k]["pool_class_size"]) for k in keys}
n_cell = {}
for k in keys:
    n_cell[cell[k]] = n_cell.get(cell[k], 0) + 1
w = [pool[cell[k]] / n_cell[cell[k]] for k in keys]
relevant = [marks[k]["sentiment"] != "neutral" for k in keys]
print(
    f"texts {len(keys)}; relevant (non-neutral) {sum(relevant)}, neutral {len(keys) - sum(relevant)}"
)
bodies = {}
for k in keys:
    r = conn.execute(
        "SELECT body_text FROM source.articles WHERE id=?", (int(marks[k]["article_id"]),)
    ).fetchone()
    bodies[k] = " ".join((r[0] or "").split())[:2500] if r else ""
tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
model = (
    AutoModelForSequenceClassification.from_pretrained(cfg["model"], revision=cfg["revision"])
    .eval()
    .cuda()
)
ent = [i for i, l in model.config.id2label.items() if l.lower() == "entailment"][0]
print("labels:", model.config.id2label)


@torch.no_grad()
def entail(premises, hyps, bs=16):
    out = []
    for i in range(0, len(premises), bs):
        enc = tok(
            premises[i : i + bs],
            hyps[i : i + bs],
            truncation="only_first",
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        out += torch.softmax(model(**enc).logits, -1)[:, ent].cpu().tolist()
    return np.array(out)


HYP = {
    "H1 own business": "This news is about {c}'s own business, financial performance or prospects.",
    "H2 about company": "This text is mainly about {c}.",
    "H3 affects company": "This news has a direct effect on {c}.",
}
PREM = {
    "summary": lambda k: ids[k]["text"],
    "title+summary": lambda k: f"{ids[k]['title']}. {ids[k]['text']}",
    "title+body": lambda k: f"{ids[k]['title']}. {bodies[k]}",
}


def wauc(scores, pos):
    return rep.weighted_roc_auc(list(scores), list(pos), w)


cv_rel = np.mean([[1 - float(o[k]["p_neutral"]) for k in keys] for o in oof], axis=0)
print(
    f"\nweighted AUC, relevant vs neutral:   the cross-validated sentiment model (1 - p_neutral): {wauc(cv_rel, relevant):.3f}"
)
rank = lambda x: np.argsort(np.argsort(x)) / (len(x) - 1)
results, scores = {}, {}
for pn, pf in PREM.items():
    for hn, h in HYP.items():
        s = entail(
            [pf(k) for k in keys], [h.format(c=marks[k]["company"].split(" (")[0]) for k in keys]
        )
        scores[(pn, hn)] = s
        a = wauc(s, relevant)
        both = wauc(rank(s) + rank(cv_rel), relevant)
        results[f"{pn} | {hn}"] = (a, both)
        print(
            f"  {pn:14} {hn:18} zero-shot alone {a:.3f}   rank-average with the cv model {both:.3f}",
            flush=True,
        )
best = max(results, key=lambda k: results[k][0])
print(f"\nbest alone: {best} {results[best][0]:.3f}")
# the owner's own wording of the groups: neutral rows that carry a note
notes = {k: marks[k]["notes"].strip() for k in keys}
mlflow.set_tracking_uri(sys.argv[2])
mlflow.set_experiment("company_relevance_zeroshot")
with mlflow.start_run(run_name="zeroshot_relevance_auc"):
    mlflow.set_tags(
        {"model": f"{cfg['model']}@{cfg['revision'][:8]}", "target": "non-neutral vs neutral"}
    )
    mlflow.log_metric("auc_cv_sentiment_model", wauc(cv_rel, relevant))
    for k, (a, b) in results.items():
        mlflow.log_metric("auc_zs_" + k.replace(" | ", "_").replace(" ", "_").replace("+", "_"), a)
        mlflow.log_metric("auc_avg_" + k.replace(" | ", "_").replace(" ", "_").replace("+", "_"), b)
