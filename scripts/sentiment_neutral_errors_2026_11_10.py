# ruff: noqa: B007, E402, S311, SIM115  (exploratory experiment script, run once; kept as the record of the figures in docs/sentiment_training.md)
import collections
import csv
import glob
import importlib.util
import random
import sys

import numpy as np

WT = sys.argv[1]
sys.path[:0] = [WT + "/src", WT + "/scripts"]
spec = importlib.util.spec_from_file_location(
    "pair_script", WT + "/scripts/train_sentiment_pair_2026_11_09.py"
)
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)
from news_nlp import env
from sentiment_stage import _text_mentions_subject as named

env.load_env_file()
conn = pair.connect_source()
marks = {}
ids = {}
for d, mp, ip in (
    ("old", "gold/sentiment_train.csv", "data/sentiment_train_ids.csv"),
    ("new", "gold/sentiment_train_2.csv", "data/sentiment_eval_ids.csv"),
):
    for r in csv.DictReader(open(mp, encoding="utf-8-sig")):
        marks[r["key"]] = r
    for r in csv.DictReader(open(ip, encoding="utf-8-sig")):
        ids[r["key"]] = r
C = ["positive", "negative", "neutral"]
oof = []
for rpt in (1, 2):
    rows = {}
    for p in glob.glob(f"data/sentiment_pair_cv_all/predictions/cv_r{rpt}_f*__*.csv"):
        for r in csv.DictReader(open(p)):
            rows[r["key"]] = r
    oof.append(rows)
rec = []
for k in oof[0]:
    pr = np.mean([[float(o[k][f"p_{c}"]) for c in C] for o in oof], axis=0)
    m, i = marks[k], ids[k]
    body = conn.execute(
        "SELECT body_text FROM source.articles WHERE id=?", (int(m["article_id"]),)
    ).fetchone()
    body = (body[0] or "") if body else ""
    rec.append(
        dict(
            key=k,
            gold=m["sentiment"],
            pred=C[int(pr.argmax())],
            pn=pr[2],
            text=i["text"],
            title=i["title"],
            company=m["company"],
            notes=m["notes"],
            in_s=named(i["text"], m["company"], m["ticker"]),
            in_t=named(i["title"], m["company"], m["ticker"]),
            in_b=named(body, m["company"], m["ticker"]),
        )
    )
neu = [r for r in rec if r["gold"] == "neutral"]
print(
    f"gold neutral: {len(neu)} of {len(rec)}; recalled {sum(r['pred'] == 'neutral' for r in neu)} ({sum(r['pred'] == 'neutral' for r in neu) / len(neu):.0%}); missed as",
    dict(collections.Counter(r["pred"] for r in neu if r["pred"] != "neutral")),
)
print("\n== neutral recall by where the company is named (raw counts, not weighted) ==")
for name, f in (
    ("summary names company", lambda r: r["in_s"]),
    ("summary does NOT, title does", lambda r: not r["in_s"] and r["in_t"]),
    (
        "neither summary nor title, body does",
        lambda r: not r["in_s"] and not r["in_t"] and r["in_b"],
    ),
    ("not even the body (string match)", lambda r: not r["in_b"]),
):
    g = [r for r in neu if f(r)]
    h = sum(r["pred"] == "neutral" for r in g)
    allg = [r for r in rec if f(r)]
    print(
        f"  {name:38} neutral gold {len(g):3}, recalled {h:3} ({h / max(len(g), 1):.0%}) | of all {len(allg)} texts in this group, {len(g) / max(len(allg), 1):.0%} are neutral"
    )
print("\n== precision side: predicted neutral ==")
pn = [r for r in rec if r["pred"] == "neutral"]
print(
    f"  predicted neutral {len(pn)}; owner says", dict(collections.Counter(r["gold"] for r in pn))
)
print("\n== owner notes on neutral rows ==")
nn = [r for r in neu if r["notes"].strip()]
print(
    f"  {len(nn)} of {len(neu)} neutral rows have a note; recalled {sum(r['pred'] == 'neutral' for r in nn)}"
)
miss = [r for r in neu if r["pred"] != "neutral"]
random.Random(3).shuffle(miss)
print(f"\n== 30 of the {len(miss)} neutral texts the cross-validation missed ==")
for r in miss[:30]:
    print(
        f"\n[{r['key']}] gold neutral -> {r['pred']} (p_neutral {r['pn']:.2f}) | company {r['company']} | named: summary {r['in_s']}, title {r['in_t']}, body {r['in_b']}"
    )
    print("  TITLE:", r["title"][:110])
    print("  TEXT: ", r["text"][:300])
    if r["notes"].strip():
        print("  NOTE: ", r["notes"][:150])

print(
    "\n\n===== SAME SITUATION, TAGGED NON-NEUTRAL: neither summary nor title names the company ====="
)
other = [r for r in rec if not r["in_s"] and not r["in_t"] and r["gold"] != "neutral"]
print(
    len(other),
    "texts;",
    dict(collections.Counter(r["gold"] for r in other)),
    "| the model's reading:",
    dict(collections.Counter(r["pred"] for r in other)),
)
for r in other:
    print(
        f"\n[{r['key']}] owner {r['gold']} | model {r['pred']} | company {r['company']} | body names it: {r['in_b']}"
    )
    print("  TITLE:", r["title"][:100])
    print("  TEXT: ", r["text"][:230])
    if r["notes"].strip():
        print("  NOTE: ", r["notes"][:150])
