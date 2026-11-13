"""Second half of the groups mock: embed the rows, cluster the labels. See category_groups_mock."""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sector_stage import BgeEmbedder

OUT = ROOT / "data" / "category_groups"
cfg = json.loads((ROOT / "config/sector_step.json").read_text())["embedder"]
emb = BgeEmbedder(cfg["model"], cfg["revision"], cfg["max_length"], cfg["batch_size"])
df = pd.read_csv(OUT / "rows.csv")
labels = [l for l in df.label.unique()]
X = emb(df.text.tolist())
np.save(OUT / "rows_emb.npy", X)

# definitions (docs/category-taxonomy.md, "The text is about")
DEFS = {
    "earnings_performance": "results, guidance, revenue or profit figures",
    "mergers_acquisitions": "deals, takeovers, divestitures",
    "leadership_governance": "executive appointments and departures, board actions, an executive's insider trade",
    "legal_regulatory": "litigation, regulatory action, sanctions, bankruptcy",
    "product_innovation": "product launches, R&D, technology",
    "capital_shareholder_returns": "dividends, buybacks, debt and credit actions of the company",
    "labor_human_capital": "layoffs, hiring, workforce and labour relations",
    "market_analyst_sentiment": "analyst ratings and price targets, stock price moves, an investor's bullish or bearish view",
    "partnerships_business_dev": "strategic alliances, joint ventures, distribution deals",
    "reputational_harm": "an event that damages or repairs the company's public standing: apologies, service failures, controversies, boycotts",
    "competitive_actions": "the company's own move or statement against or about a competitor: share fights, pricing pressure",
    "investor_activity": "a fund or famous investor buys or sells the stock, an activist position, 13F filings, block trades",
    "peer_comparison": "a comparison between companies: which is the better stock, who is ahead",
    "other": "the event is clear and about the company but fits none of the other labels",
}
D = emb([DEFS[l] for l in labels])
C = np.stack([X[(df.label == l).values].mean(0) for l in labels])
C /= np.linalg.norm(C, axis=1, keepdims=True)


def show(name, S):
    print(f"\n== {name} ==")
    print(
        pd.DataFrame(S, index=[l[:14] for l in labels], columns=[l[:5] for l in labels])
        .round(2)
        .to_string()
    )


def groups(S, ks=(4, 5, 6)):
    dist = 1 - S
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), "average")
    for k in ks:
        g = fcluster(Z, k, "maxclust")
        print(f"\n-- {k} groups --")
        for c in sorted(set(g)):
            print(f"  G{c}:", ", ".join(l for l, x in zip(labels, g, strict=False) if x == c))


S1 = C @ C.T
S2 = D @ D.T
show("centroid similarity of the training texts", S1)
groups(S1)
show("similarity of the label definitions", S2)
groups(S2)
# nearest-centroid confusion: where texts of one label sit closest to another label's centroid
pred = np.argmax(X @ C.T, axis=1)
conf = collections.Counter()
for t, p in zip(df.label, pred, strict=False):
    if labels[p] != t:
        conf[(t, labels[p])] += 1
print("\nmost confused (true -> nearest other centroid):")
for (a, b), n in conf.most_common(10):
    print(f"  {a} -> {b}: {n}")
# co-occurrence of two labels on one (article, company) in all 284 gold bodies
g = pd.read_csv(ROOT / "gold/category_gold_v2.csv")
co = collections.Counter()
per = collections.defaultdict(set)
for r in g.itertuples():
    if isinstance(r.company_labels, str):
        for part in r.company_labels.split(", "):
            if ":" in part:
                n, l = part.rsplit(":", 1)
                per[(r.article_id, n)].add(l)
for s in per.values():
    for a in s:
        for b in s:
            if a < b:
                co[(a, b)] += 1
print("\nlabels given together to one company in one article:", co.most_common(8))

# ruff: noqa: E402, E741  -- one-shot research script
