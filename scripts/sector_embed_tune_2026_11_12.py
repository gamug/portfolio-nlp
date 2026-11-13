"""Tune the sector embedding step on the dev half and score the held-out half once (2026-11-12).

    uv run scripts/sector_embed_tune_2026_11_12.py SCRATCH            # dev half only: the grid of variants
    uv run scripts/sector_embed_tune_2026_11_12.py SCRATCH --final "<variant name>"   # the held-out half, for that one variant

The 70 single-sector bodies are split by position (even = dev, odd = held-out). Everything that is chosen (descriptions v2,
the noise direction, the title, the method) is chosen on the dev half; the held-out half is printed only with --final.
Descriptions v2 were written from the sectors' vocabulary and from what the dev errors of v1 showed (investing-style words,
bank names as sources, thin vocabulary); they are not copies of any text.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import sector_embed_baseline_2026_11_12 as base

DESC_V2: dict[str, list[str]] = {
    "10": [
        "Energy stocks: oil and gas producers, crude oil prices, Brent and WTI, OPEC output cuts, shale drilling",
        "Midstream pipeline operators, LNG export terminals, natural gas prices and storage, refiners and refining margins",
        "Oilfield services and drillers, rig counts, offshore drilling, exploration and production spending",
        "Oil majors' earnings, production growth, capital discipline and buybacks in the oil patch, coal producers",
        "Analysts weigh in on the energy sector as oil prices swing, upgrades and downgrades of energy stocks",
    ],
    "15": [
        "Chemicals makers, fertilizers, industrial gases and specialty chemicals, pricing and demand",
        "Mining companies, copper, gold, silver, lithium, iron ore and aluminum prices, metal output",
        "Steel producers, tariffs on steel and aluminum, mill capacity and steel demand",
        "Packaging, paper and forest products, construction materials such as cement, aggregates and lumber",
        "Gold and copper miners' production guidance, commodity prices and mine costs",
    ],
    "20": [
        "Aerospace and defense contractors, aircraft production and orders, Pentagon contracts, shipbuilding",
        "Airlines, fares, capacity and fuel costs, travel demand, government aid to carriers, pilots and unions",
        "Railroads, trucking, freight and logistics, parcel delivery, shipping rates and container volumes",
        "Industrial machinery, construction and engineering, electrical equipment, building products, heavy equipment orders",
        "Waste services, staffing and business services, conglomerates, industrial stocks and factory orders",
    ],
    "25": [
        "Automakers, car sales and auto profits, electric vehicles, tariffs on cars, auto parts and dealers",
        "Retailers, department stores, e-commerce platforms, apparel and footwear brands, same-store sales, holiday shopping",
        "Restaurant chains and fast food, same-store sales, menu prices, burgers, coffee and pizza chains",
        "Hotels, casinos, cruise lines, airlines' leisure travel, theme parks, homebuilders and household durables",
        "Chinese internet platforms and online marketplaces, food delivery and consumer e-commerce stocks",
    ],
    "30": [
        "Tobacco and cigarette makers, cigarette sales decline, smoking, vaping and nicotine products",
        "Beer, wine, spirits and liquor makers, soft drinks, bottlers, energy drinks and beverage volumes",
        "Packaged food makers, snacks, cereal and candy, food prices, commodity costs and volumes",
        "Grocery chains, supermarkets, warehouse clubs, drugstore chains, discount retailers and food distributors",
        "Household and personal care products, cosmetics, diapers, detergent, consumer staples pricing power",
    ],
    "35": [
        "Drugmakers and biotechnology, clinical trial results, FDA approvals, drug patents and generics",
        "Medical device makers, diagnostics, life sciences tools, surgical robots and lab equipment",
        "Hospitals, health care providers, nursing homes, physician groups and health services",
        "Health insurers, managed care, Medicare and Medicaid, health care costs and premiums",
        "Drug pricing policy, vaccines, obesity drugs, pharmaceutical sales and pipelines",
    ],
    "40": [
        "Banks: net interest margin, loans, deposits, loan losses, credit quality, stress tests and capital rules",
        "Insurers and reinsurers, premiums, claims, catastrophe losses and insurance underwriting",
        "Asset managers, exchanges, brokerages, private equity firms, fees and assets under management",
        "Payment networks, credit cards, consumer finance, fintech lenders and mortgage lenders",
        "Bank earnings season, regional bank deposits, financial regulation and bank mergers",
    ],
    "45": [
        "Software and cloud computing stocks, enterprise software spending, SaaS subscriptions, cybersecurity",
        "Semiconductor and chip stocks, chipmakers, memory chips, chip equipment, data center demand",
        "Artificial intelligence spending, AI chips and servers, hyperscalers' capital expenditure",
        "Computer hardware, smartphones, networking gear, storage, electronic components and tech distributors",
        "IT services and consulting, technology stocks rally or selloff, tech valuations and earnings",
    ],
    "50": [
        "Telecom carriers, wireless subscribers, broadband, 5G, cable and satellite operators",
        "Media and entertainment companies, film studios, streaming services, subscribers and content spending",
        "Social media and internet platforms, search engines, digital advertising revenue",
        "Video game publishers, music, publishing, newspapers and broadcasters",
        "Advertising agencies, television networks, ratings and media mergers",
    ],
    "55": [
        "Electric utilities, power generation, rate cases, state regulators and electricity prices",
        "Natural gas utilities and water utilities, rate increases, customers and regulated returns",
        "Nuclear plants, renewable power producers, wind and solar developers, independent power producers",
        "Power grids, transmission lines, wildfire liability, utility earnings and dividends",
        "Power demand growth from data centers, utility capital plans and electricity load",
    ],
    "60": [
        "Real estate investment trusts, office, retail and industrial properties, occupancy and rents",
        "Shopping malls and mall owners, apartment and housing REITs, healthcare properties",
        "Data center, cell tower, warehouse and self-storage REITs, funds from operations",
        "Housing market recovery, home prices, home sales, mortgage rates and housing starts",
        "Commercial real estate, property values, office vacancy, real estate services and brokers",
    ],
}
NOISE = "An investment bank analyst upgrades the stock, raises the price target and says investors should buy the shares"


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main() -> None:
    args = sys.argv[1:]
    final = args[args.index("--final") + 1] if "--final" in args else None
    base.SCRATCH = Path(args[0])
    rows, texts = base.load_inputs()
    sectors = list(base.DESCRIPTIONS)
    emb = base.Embedder()
    n_desc = {1: 3, 2: 5}
    desc = {
        1: emb([d for s in sectors for d in base.DESCRIPTIONS[s]]),
        2: emb([d for s in sectors for d in DESC_V2[s]]),
    }
    noise = emb([NOISE])[0]
    a_texts = base.texts_of_option_a(rows, texts)
    titles = [texts[r["article_id"]][0] for r in rows]
    a_plain = emb(a_texts)
    a_title = emb([f"{t}. {x}" for t, x in zip(titles, a_texts, strict=True)])
    chunks = [base.chunks_of(texts[r["article_id"]][1]) for r in rows]
    c_plain = [emb(c) for c in chunks]
    c_title = [emb([f"{t}. {c[0]}", *c[1:]]) for t, c in zip(titles, chunks, strict=True)]
    truth = np.array([r["sectors"] for r in rows])
    n = len(rows)
    idx_dev = np.arange(0, n, 2)

    def score(vecs: np.ndarray, v: int, alpha: float) -> np.ndarray:
        if alpha:
            vecs = unit(vecs - alpha * (vecs @ noise)[:, None] * noise[None, :])
        return (vecs @ desc[v].T).reshape(len(vecs), len(sectors), n_desc[v]).max(-1)

    def matrix(method: str, v: int, alpha: float, title: bool) -> np.ndarray:
        a = score(a_title if title else a_plain, v, alpha)
        cs = [score(c, v, alpha) for c in (c_title if title else c_plain)]
        b_max = np.array([c.max(0) for c in cs])
        b_top3 = np.array([np.sort(c, axis=0)[-3:].mean(0) for c in cs])
        return {"a": a, "b_max": b_max, "b_top3": b_top3, "a+b_max": (a + b_max) / 2}[method]

    def acc(s: np.ndarray, idx: np.ndarray) -> float:
        return float(np.mean(np.array(sectors)[s[idx].argmax(1)] == truth[idx]))

    grid = list(
        itertools.product(
            (1, 2), (0.0, 0.5, 1.0), (False, True), ("a", "b_max", "b_top3", "a+b_max")
        )
    )
    name = lambda g: f"desc{g[0]}|noise{g[1]}|{'title' if g[2] else 'notitle'}|{g[3]}"  # noqa: E731
    if final is None:
        res = sorted(
            ((acc(matrix(g[3], g[0], g[1], g[2]), idx_dev), name(g)) for g in grid), reverse=True
        )
        print(f"dev half: {len(idx_dev)} bodies; {len(grid)} variants")
        for a, nm in res[:12]:
            print(f"  {a:6.1%}  {nm}")
        print("  ...")
        for a, nm in res[-3:]:
            print(f"  {a:6.1%}  {nm}")
        print(
            "baseline v1 (desc1|noise0.0|notitle|a, b_max):",
            [f"{acc(matrix(m, 1, 0.0, False), idx_dev):.1%}" for m in ("a", "b_max")],
        )
        return
    g = next(g for g in grid if name(g) == final)
    print_final(final, matrix(g[3], g[0], g[1], g[2]), sectors, truth, rows, titles)


def print_final(
    final: str,
    s: np.ndarray,
    sectors: list[str],
    truth: np.ndarray,
    rows: list[dict[str, str]],
    titles: list[str],
) -> None:
    """The held-out half for the one variant chosen on the dev half."""
    idx_dev, idx_held = np.arange(0, len(rows), 2), np.arange(1, len(rows), 2)

    def acc(idx: np.ndarray) -> float:
        return float(np.mean(np.array(sectors)[s[idx].argmax(1)] == truth[idx]))

    pred = np.array(sectors)[s.argmax(1)]
    srt = np.sort(s, axis=1)
    marg = srt[:, -1] - srt[:, -2]
    print(f"FINAL {final}")
    print(
        f"  dev accuracy  {acc(idx_dev):.1%} (n={len(idx_dev)})   held-out accuracy {acc(idx_held):.1%} (n={len(idx_held)})"
    )
    top2 = np.mean([truth[i] in {sectors[j] for j in np.argsort(-s[i])[:2]} for i in idx_held])
    print(f"  held-out top-2 {top2:.1%}")
    thr = np.quantile(marg[idx_dev], [0.25, 0.5])
    for lab, t in (
        ("all", 0.0),
        ("margin >= dev 25th pct", thr[0]),
        ("margin >= dev median", thr[1]),
    ):
        m = marg[idx_held] >= t
        ok = pred[idx_held][m] == truth[idx_held][m]
        print(
            f"  held-out {lab}: n={int(m.sum())} ({m.mean():.0%} of bodies), accuracy {ok.mean():.1%}"
        )
    print("  held-out errors (true -> predicted):")
    for i in idx_held:
        if pred[i] != truth[i]:
            print(
                f"    #{rows[i]['article_id']} {truth[i]} -> {pred[i]}  margin {marg[i]:.3f}  {titles[i][:70]}"
            )


if __name__ == "__main__":
    main()
