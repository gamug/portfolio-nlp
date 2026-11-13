"""Score the owner's yes/no review of the sector step and test the improvements it suggests (2026-11-13).

    uv run scripts/sector_review_eval_2026_11_13.py SCRATCH            # the review, the labels it gives, the grid on the TUNE half
    uv run scripts/sector_review_eval_2026_11_13.py SCRATCH --final "<variant>"   # the TEST half, for the one variant chosen

Labels from the review (data/sector_yesno_to_review.csv + data/sector_yesno_key.csv, 100 real rows, the 10 controls apart):
``yes`` -> the sector the model showed; ``no`` with a ``right_sector`` -> that sector; ``no`` whose note says the body is not a
sector body (scope error: market_economy, investing_general, unrelated, multi ...) -> ``reject``; the rest -> unknown (kept
out of the sector accuracy, counted as wrong in the precision). Three ``right_sector`` cells of the retail articles (15, the
Materials code) read as a typo of 25 and are unknown too.

Split: the 70 tagged single-sector bodies and the 100 reviewed rows are each split by position (even = TUNE, odd = TEST).
Descriptions v3 are written from the GICS structure (one line per industry group) and from the owner's notes, before any score
of them was seen; the TEST halves are printed once, with --final. The owner's notes cover all 100 rows, so the reviewed TEST
half is not pristine: the clean check is a second, short review of the rows whose answer changes.
"""

from __future__ import annotations

import csv
import itertools
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sector_embed_baseline_2026_11_12 as base
import sector_embed_tune_2026_11_12 as tune

ROOT = Path(__file__).resolve().parent.parent
ALPHA = 0.5
RETAIL_TYPO = {
    "451722",
    "90842",
    "476212",
}  # right_sector 15 on a retail article: read as 25, unknown
DESC_V3: dict[str, list[str]] = {
    "10": [
        "Oil and gas exploration and production, crude oil prices, Brent and WTI, shale drilling, OPEC output",
        "Integrated oil majors, oil company earnings, production growth, buybacks in the oil patch",
        "Oil refining and marketing, pipelines, storage and transportation, LNG export terminals, natural gas prices",
        "Oilfield services and equipment, offshore drilling, rig counts, drillers' spending",
        "Coal and consumable fuels, uranium miners, energy stocks upgrades and downgrades",
    ],
    "15": [
        "Chemical makers: commodity and specialty chemicals, industrial gases, fertilizers and agricultural chemicals",
        "Chemical companies face lawsuits, contamination and environmental liability over their products",
        "Metals and mining: steel producers, iron ore, aluminum, copper, gold, silver, lithium and diversified miners",
        "Steel mills, steel tariffs and imports, trade war in steel, mill capacity and steelmakers' earnings",
        "Paper and forest products, timber, lumber and pulp, packaging and containers, cement and aggregates",
        "Commodity prices for metals and minerals, mine output, mining stocks",
    ],
    "20": [
        "Aerospace and defense contractors, aircraft makers, defense budgets and Pentagon contracts",
        "Airlines and air freight, parcel delivery and logistics, FedEx, UPS and DHL, shipping and holiday volumes",
        "Railroads, trucking, marine shipping, container rates, ports and freight transportation",
        "Industrial machinery, heavy equipment, construction and engineering, electrical equipment, factory orders",
        "Building products, industrial conglomerates, trading companies and distributors",
        "Commercial and professional services: waste management, staffing, consulting, security and office services",
    ],
    "25": [
        "Automakers and auto parts, car sales, electric vehicle makers and EV sales, auto dealers, tariffs on cars",
        "Online retail and e-commerce, Amazon, Alibaba and Chinese internet marketplaces, consumer internet stocks",
        "Retailers and specialty retail: department stores, apparel and footwear chains, home improvement, retail sales and prices",
        "Restaurants and fast food chains, hotels, casinos, cruise lines, theme parks and leisure travel",
        "Homebuilders and household durables, furniture, appliances, leisure products, textiles, apparel and luxury brands",
        "Consumer discretionary stocks, consumer spending on non-essential goods, holiday shopping season",
    ],
    "30": [
        "Food and staples retailing: grocery chains, supermarkets, warehouse clubs, drugstore chains, discount retailers",
        "Beverage makers: beer, wine, spirits, soft drinks, bottlers and energy drinks",
        "Packaged food and meat producers, snacks, candy, agricultural products, food prices and volumes",
        "Tobacco and cigarette makers, smoking decline, vaping and nicotine products",
        "Household products and personal care: detergent, diapers, cosmetics, razors and consumer staples brands",
    ],
    "35": [
        "Pharmaceutical companies, drug approvals by the FDA, drug pricing and patents, vaccines, obesity drugs",
        "Biotechnology, clinical trial results, gene therapy and drug pipelines",
        "Medical devices and supplies, surgical robots, diagnostics, life sciences tools and lab equipment",
        "Hospitals and health care providers, health services, nursing homes, labs and clinics",
        "Health insurers and managed care, Medicare and Medicaid, health care costs, health care technology",
    ],
    "40": [
        "Banks: net interest margin, loans, deposits, credit quality and loan losses, stress tests, bank earnings",
        "Investment banks, capital markets, asset managers, brokerages, private equity, exchanges and fees",
        "Insurance companies, insurers and reinsurers, premiums, claims and underwriting",
        "Consumer finance, credit cards, payment networks, fintech lenders, mortgage finance",
        "Financial stocks and financial regulation, bank mergers, financials sector rally",
    ],
    "45": [
        "Software and cloud computing, enterprise software, SaaS, cybersecurity, IT services and consulting",
        "Semiconductors and chip equipment, chipmakers, memory chips, GPUs, foundries",
        "Artificial intelligence spending, data centers, servers and hyperscaler capital expenditure",
        "Technology hardware: computers, smartphones, networking and communications equipment, storage, electronic components",
        "Technology stocks, big tech valuations, tech earnings, crypto mining hardware and blockchain technology",
    ],
    "50": [
        "Telecom carriers and wireless operators, broadband, 5G, cable and satellite providers",
        "Media and entertainment: film studios, streaming services, subscribers, music and video games",
        "Social media, search engines and interactive media platforms, digital advertising revenue",
        "Broadcasting, television networks, publishing and newspapers, advertising agencies",
    ],
    "55": [
        "Electric utilities, power generation and electricity prices, rate cases and state regulators",
        "Gas utilities, water utilities and multi-utilities, regulated returns and customers",
        "Independent power producers, renewable electricity, wind and solar developers, nuclear plants",
        "Power grids, transmission, wildfire liability, utility dividends, electricity demand from data centers",
    ],
    "60": [
        "Real estate investment trusts, office, retail, residential and industrial properties, occupancy and rents",
        "Shopping malls, apartment REITs, health care properties, hotel REITs",
        "Data center, cell tower, warehouse and self-storage REITs, funds from operations",
        "Housing market, home prices and home sales, mortgage rates, housing starts",
        "Commercial real estate, office vacancy, real estate services, brokers and property developers",
    ],
}
# not a sector body: what the owner's notes name (scope errors of the 33 "no")
REJECT: dict[str, list[str]] = {
    "market_economy": [
        "Jobs report, private payrolls, unemployment, GDP, inflation, consumer prices and economic data below expectations",
        "Federal Reserve, interest rates, tariffs and trade policy, the economy, bitcoin and commodity prices",
    ],
    "investing_general": [
        "CEO interview about leadership and business advice, a billionaire investor on how to build a business",
        "Stock picks and portfolio strategy, market outlook, how to invest, strategists' targets and predictions",
    ],
    "investment_activity": [
        "Fund managers and hedge funds buying, ETF flows, small cap rally, investors positioning and funds bets",
    ],
    "unrelated": [
        "Politics, lawsuits, social issues, inauguration, harassment, charity, scams and news not about any industry",
    ],
}


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def read_review() -> list[dict[str, str]]:
    rows = []
    with open(ROOT / "data/sector_yesno_to_review.csv", encoding="utf-8-sig", newline="") as f:
        rev = list(csv.DictReader(f))
    with open(ROOT / "data/sector_yesno_key.csv", encoding="utf-8-sig", newline="") as f:
        key = {k["review_order"]: k for k in csv.DictReader(f)}
    for r in rev:
        k = key[r["review_order"]]
        if k["control"] == "yes":
            continue
        rows.append({**r, "margin": k["margin"], "sure": k["sure"]})
    return rows


def label_of(r: dict[str, str]) -> str | None:
    """sector code (2 digits), "reject" (not a sector body) or None (unknown)."""
    if r["correct"].strip().lower() == "yes":
        return r["model_sector_code"][:2]
    right = re.sub(r"\D", "", r["right_sector"])
    if right:
        if r["article_id"] in RETAIL_TYPO:
            return None
        return "15" if right == "5" else right
    note = r["notes"].lower()
    scope_words = (
        "scope",
        "misslabel",
        "unrelated",
        "multi",
        "laboral",
        "investment_activity",
        "market_economy",
    )
    return "reject" if any(w in note for w in scope_words) else None


class Scorer:
    """Chunk embeddings are computed once; the descriptions and the options are applied to them."""

    def __init__(self, items: list[tuple[str, str]]) -> None:
        self.emb = base.Embedder()
        self.noise = self.emb([tune.NOISE])[0]
        self.vecs = []
        for title, text in items:
            ch = base.chunks_of(text)
            v = self.emb([f"{title}. {ch[0]}", *ch[1:]])
            self.vecs.append(unit(v - ALPHA * (v @ self.noise)[:, None] * self.noise[None, :]))

    def scores(self, desc: dict[str, list[str]], k: int = 3) -> np.ndarray:
        names = list(desc)
        flat = self.emb([d for n in names for d in desc[n]])
        owner = np.repeat(np.arange(len(names)), [len(desc[n]) for n in names])
        out = np.zeros((len(self.vecs), len(names)))
        for i, v in enumerate(self.vecs):
            s = v @ flat.T  # chunks x descriptions
            per = np.stack([s[:, owner == j].max(1) for j in range(len(names))], 1)
            out[i] = np.sort(per, axis=0)[-k:].mean(0)
        return out


def evaluate(
    scores: np.ndarray, names: list[str], labels: list[str | None], margin_thr: float
) -> dict[str, float]:
    """precision of the sectors assigned (as the yes/no review counts it), coverage and the pieces."""
    pred = np.array(names)[scores.argmax(1)]
    srt = np.sort(scores, axis=1)
    marg = srt[:, -1] - srt[:, -2]
    is_sector = np.array([p in tune.DESC_V2 for p in pred])
    lab = np.array([lb or "?" for lb in labels])
    right = is_sector & (pred == lab)
    known = np.array([lb not in (None, "reject") for lb in labels])
    assigned = is_sector & (marg >= margin_thr)
    return {
        "n": len(labels),
        "assigned %": float(assigned.mean()),
        "precision all": float(right[is_sector].mean()) if is_sector.any() else 0.0,
        "precision assigned": float(right[assigned].mean()) if assigned.any() else 0.0,
        "accuracy known sector": float((pred == lab)[known].mean()) if known.any() else 0.0,
        "rejects caught": float(
            np.mean(
                [
                    p not in tune.DESC_V2
                    for p, lb in zip(pred, labels, strict=True)
                    if lb == "reject"
                ]
                or [0]
            )
        ),
        "true sector rejected": float(
            np.mean(
                [
                    p not in tune.DESC_V2
                    for p, lb in zip(pred, labels, strict=True)
                    if lb not in (None, "reject")
                ]
                or [0]
            )
        ),
    }


def main() -> None:
    args = sys.argv[1:]
    final = args[args.index("--final") + 1] if "--final" in args else None
    base.SCRATCH = Path(args[0])
    a_rows, a_texts = base.load_inputs()
    b_rows = read_review()
    items = [a_texts[r["article_id"]] for r in a_rows] + [(r["title"], r["text"]) for r in b_rows]
    a_lab: list[str | None] = [r["sectors"] for r in a_rows]
    b_lab = [label_of(r) for r in b_rows]
    labels = a_lab + b_lab
    na = len(a_rows)
    print(
        f"A: {na} tagged bodies; B: {len(b_rows)} reviewed rows; B labels:",
        {k: b_lab.count(k) for k in set(b_lab)},
    )
    scorer = Scorer(items)
    tune_idx = np.array([i for i in range(len(items)) if (i if i < na else i - na) % 2 == 0])
    test_idx = np.array([i for i in range(len(items)) if i not in set(tune_idx)])
    variants: dict[str, tuple[dict[str, list[str]], bool, int]] = {}
    union = {k: [*tune.DESC_V2[k], *DESC_V3[k]] for k in DESC_V3}
    for dname, d in (("v2", tune.DESC_V2), ("v3", DESC_V3), ("v2+v3", union)):
        for rej, k in itertools.product((False, True), (1, 3)):
            variants[f"{dname}|{'reject' if rej else 'noreject'}|top{k}"] = (
                {**d, **REJECT} if rej else d,
                rej,
                k,
            )
    cache = {n: scorer.scores(d, k) for n, (d, _, k) in variants.items()}
    names_of = {n: list(d) for n, (d, _, _) in variants.items()}
    for part, idx in (("TUNE", tune_idx),) if final is None else (("TEST", test_idx),):
        for n in variants if final is None else [final]:
            s = cache[n][idx]
            lab = [labels[i] for i in idx]
            thr = float(
                np.quantile(
                    np.sort(cache[n][tune_idx], 1)[:, -1] - np.sort(cache[n][tune_idx], 1)[:, -2],
                    0.25,
                )
            )
            for lbl, sel in (("A", idx < na), ("B", idx >= na)):
                m = evaluate(
                    s[sel], names_of[n], [x for x, k in zip(lab, sel, strict=True) if k], thr
                )
                print(
                    f"{part} {n:24s} {lbl}: "
                    + "  ".join(
                        f"{k} {v:.1%}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()
                    )
                )


if __name__ == "__main__":
    main()
