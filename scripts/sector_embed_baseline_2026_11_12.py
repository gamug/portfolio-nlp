"""Sector identification by embeddings: options a (summary) and b (chunks), scored on the owner's tags (2026-11-12).

Reads data/sector_rollup.csv (the 71 single-sector bodies), the texts in gold/texts/, and scores each body against
11 GICS sector descriptions with a small embedding model. Option a embeds the body as the summary step would
leave it (a summary over 2,000 characters, the text itself otherwise); option b embeds ~120-word chunks and
aggregates the chunk scores (max, mean of the top 3). The descriptions are in DESCRIPTIONS below and were written
before any score was seen.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from news_summary_stage import Summarizer, input_of_body, needs_summary  # noqa: E402

MODEL = "BAAI/bge-small-en-v1.5"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"

DESCRIPTIONS: dict[str, list[str]] = {
    "10": [
        "Oil and gas producers, crude oil prices, OPEC output decisions, drilling and exploration, shale",
        "Refiners, pipelines, LNG exports, oilfield services, rig counts, coal producers",
        "Energy stocks move with Brent and WTI crude, natural gas demand and petroleum inventories",
    ],
    "15": [
        "Chemicals makers, fertilizers, industrial gases and specialty chemicals demand",
        "Mining companies, copper, gold, silver, aluminum and steel prices, metals and ore output",
        "Packaging, paper and forest products, construction materials such as cement and aggregates",
    ],
    "20": [
        "Aerospace and defense contractors, aircraft orders, military contracts, shipbuilding",
        "Airlines, railroads, trucking, freight and logistics, shipping rates",
        "Industrial machinery, construction and engineering, electrical equipment, building products, staffing and business services",
    ],
    "25": [
        "Automakers, car sales, electric vehicles, auto parts and dealers",
        "Retailers, department stores, e-commerce, apparel and footwear, consumer spending on goods",
        "Hotels, restaurants, casinos, cruise lines, travel and leisure, homebuilders and household durables",
    ],
    "30": [
        "Packaged food, beverage and soft drink makers, brewers, distillers, tobacco",
        "Grocery chains, supermarkets, drugstores, warehouse clubs and food distributors",
        "Household and personal products, cosmetics, consumer staples pricing and volumes",
    ],
    "35": [
        "Drugmakers and biotechnology, clinical trials, FDA approvals, drug pricing and patents",
        "Medical devices, diagnostics, life sciences tools, hospitals and health care providers",
        "Health insurers, managed care, Medicare and Medicaid, health care costs",
    ],
    "40": [
        "Banks, lending, net interest margins, deposits, loan losses, credit quality and capital rules",
        "Insurers, reinsurance, asset managers, brokerages, investment banking fees, private equity",
        "Payment networks, consumer finance, credit cards, mortgage lenders, financial regulation",
    ],
    "45": [
        "Software and cloud computing, enterprise software spending, SaaS, cybersecurity, artificial intelligence",
        "Semiconductors, chipmakers, memory chips, chip equipment, data center demand",
        "Computer hardware, smartphones, networking gear, electronic components, IT services and consulting",
    ],
    "50": [
        "Telecom carriers, wireless subscribers, broadband, cable and satellite operators",
        "Media and entertainment, film studios, streaming services, broadcasters, publishers, video games",
        "Social media, internet platforms and advertising, search engines, digital advertising revenue",
    ],
    "55": [
        "Electric utilities, power generation, rate cases, regulators and electricity prices",
        "Natural gas and water utilities, nuclear plants, renewable power producers, wind and solar",
        "Power grids, transmission, independent power producers, utility earnings and dividends",
    ],
    "60": [
        "Real estate investment trusts, office, retail and industrial properties, occupancy and rents",
        "Data center and cell tower REITs, apartment and housing REITs, healthcare properties",
        "Commercial real estate, property values, mortgage rates and real estate services",
    ],
}


class Embedder:
    def __init__(self) -> None:
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL).eval()
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.dev)

    @torch.no_grad()
    def __call__(self, texts: list[str], bs: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(
                texts[i : i + bs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.dev)
            h = self.model(**b).last_hidden_state[:, 0]  # CLS pooling, as the model card
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.vstack(out)


def chunks_of(text: str, words: int = 120) -> list[str]:
    w = text.split()
    return [" ".join(w[i : i + words]) for i in range(0, max(len(w), 1), words)] or [""]


def load_inputs() -> tuple[list[dict[str, str]], dict[str, tuple[str, str]]]:
    """The single-sector bodies of the rollup and the texts of the tracked gold."""
    texts: dict[str, tuple[str, str]] = {}
    for p in (ROOT / "gold" / "texts").glob("*_texts.csv"):
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                texts.setdefault(r["article_id"], (r["title"], r["text"]))
    with open(ROOT / "data/sector_rollup.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok" and r["article_id"] in texts]
    return rows, texts


def texts_of_option_a(rows: list[dict[str, str]], texts: dict[str, tuple[str, str]]) -> list[str]:
    """What the summary step leaves of each body: a summary over 2,000 characters, the text itself otherwise."""
    inputs = [input_of_body(*texts[r["article_id"]]) for r in rows]
    need = [needs_summary(x) for x in inputs]
    cache = SCRATCH / "sector_summaries.json"
    summ = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [
        (r["article_id"], x)
        for r, x, n in zip(rows, inputs, need, strict=True)
        if n and r["article_id"] not in summ
    ]
    if todo:
        done = Summarizer().summarize([x for _, x in todo])
        for (aid, _), (sm, _) in zip(todo, done, strict=True):
            summ[aid] = sm
        cache.write_text(json.dumps(summ))
    return [
        summ[r["article_id"]] if n else texts[r["article_id"]][1]
        for r, n in zip(rows, need, strict=True)
    ]


def report(
    methods: dict[str, np.ndarray], truth: list[str], sectors: list[str]
) -> dict[str, list[str]]:
    """Print accuracy (all, the two halves by position, top 2), per true sector and by margin; return the predictions."""
    half = np.array([i % 2 for i in range(len(truth))])  # 0 = dev, 1 = held-out by position
    print(f"{'method':10s} {'all':>6s} {'half0':>6s} {'half1':>6s}  top2")
    preds: dict[str, list[str]] = {}
    for name, score in methods.items():
        pred = [sectors[i] for i in score.argmax(1)]
        ok = np.array([p == t for p, t in zip(pred, truth, strict=True)])
        top2 = np.mean(
            [t in {sectors[j] for j in np.argsort(-score[i])[:2]} for i, t in enumerate(truth)]
        )
        print(
            f"{name:10s} {ok.mean():6.1%} {ok[half == 0].mean():6.1%} {ok[half == 1].mean():6.1%}  {top2:.1%}"
        )
        preds[name] = pred
    for name, score in methods.items():
        pred = preds[name]
        print("\n", name, "per true sector (n, correct):")
        for s in sectors:
            idx = [i for i, t in enumerate(truth) if t == s]
            if idx:
                print("  ", s, len(idx), sum(pred[i] == s for i in idx))
        srt = np.sort(score, axis=1)
        marg = srt[:, -1] - srt[:, -2]
        ok = np.array([p == t for p, t in zip(pred, truth, strict=True)])
        q = np.quantile(marg, [0.25, 0.5])
        for lo, lab in ((0, "all"), (q[0], "top 75% margin"), (q[1], "top 50% margin")):
            m = marg >= lo
            print(f"   {lab}: n={m.sum()}, accuracy {ok[m].mean():.1%}")
    return preds


def main() -> None:
    rows, texts = load_inputs()
    print(len(rows), "bodies with text")
    sectors = list(DESCRIPTIONS)
    emb = Embedder()
    desc = emb([d for s in sectors for d in DESCRIPTIONS[s]])  # sectors x 3 descriptions, flattened

    def scores(vecs: np.ndarray) -> np.ndarray:
        """Cosine to each sector's best description: rows x sectors."""
        return (vecs @ desc.T).reshape(len(vecs), len(sectors), 3).max(-1)

    score_a = scores(emb(texts_of_option_a(rows, texts)))
    b_max, b_top3 = [], []
    for r in rows:
        chunk_scores = scores(emb(chunks_of(texts[r["article_id"]][1])))  # chunks x sectors
        b_max.append(chunk_scores.max(0))
        b_top3.append(np.sort(chunk_scores, axis=0)[-3:].mean(0))
    methods = {"a_summary": score_a, "b_max": np.array(b_max), "b_top3": np.array(b_top3)}
    truth = [r["sectors"] for r in rows]
    preds = report(methods, truth, sectors)
    out = SCRATCH / "sector_embed_baseline_rows.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["article_id", "truth"] + [f"{n}_pred" for n in preds])
        for i, r in enumerate(rows):
            w.writerow([r["article_id"], truth[i]] + [preds[n][i] for n in preds])


if __name__ == "__main__":
    main()
