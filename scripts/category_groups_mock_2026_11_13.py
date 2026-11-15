"""Mock of the category groups (owner request 2026-11-13): build the per-company training rows of the
284 train bodies (held-out 200 untouched), embed them, and show how the 13 labels cluster.

Rows: a `single` body -> the whole body for its company; a `multi` body written as one segment per
company ("structured") -> the company's own segments; a narrative `multi` body is excluded.
Run: uv run scripts/category_groups_mock_2026_11_13.py [--out DIR]
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from news_nlp import company_texts
from news_nlp.triage import Company, CompanyResolver

LABELS = [
    "earnings_performance",
    "mergers_acquisitions",
    "leadership_governance",
    "legal_regulatory",
    "product_innovation",
    "capital_shareholder_returns",
    "labor_human_capital",
    "market_analyst_sentiment",
    "partnerships_business_dev",
    "reputational_harm",
    "competitive_actions",
    "investor_activity",
    "peer_comparison",
    "other",
]  # `uncertain` is a label-level abstention, not a class to learn


def parse_labels(cell: str) -> list[tuple[str, str]]:
    out = []
    for part in str(cell).split(", "):
        if ":" in part:
            name, label = part.rsplit(":", 1)
            out.append((name.strip(), label.strip()))
    return out


def build_rows(gold: pd.DataFrame, texts: pd.DataFrame, resolver: CompanyResolver):
    text_of = dict(zip(texts.article_id, texts.text, strict=False))
    rows, dropped = [], collections.Counter()
    for r in gold.itertuples():
        if r.scope not in ("single", "multi") or not isinstance(r.company_labels, str):
            continue
        body = text_of[r.article_id]
        pairs = parse_labels(r.company_labels)
        if r.scope == "single":
            for name, label in pairs:
                rows.append((r.article_id, "single", name, label, body))
            continue
        split = company_texts.split_company_texts(body, resolver)
        if split is None:
            dropped["narrative multi body"] += 1
            continue
        for name, label in pairs:
            ticker = next(iter(resolver.resolve(name, skip_attribution=False)), None)
            if ticker is None or ticker not in split:
                dropped["label whose company has no segment"] += 1
                continue
            rows.append((r.article_id, "multi_split", name, label, split[ticker]))
    cols = ["article_id", "kind", "company", "label", "text"]
    return pd.DataFrame(rows, columns=cols), dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "category_groups")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    uni = pd.read_csv(ROOT / "gold/universe.csv").fillna("")
    resolver = CompanyResolver(
        [
            Company(t, c, s, "")
            for t, c, s in zip(uni.ticker, uni.company, uni.gics_sector, strict=False)
        ]
    )
    gold = pd.read_csv(ROOT / "gold/category_gold_v2.csv")
    texts = pd.read_csv(ROOT / "gold/texts/category_gold_v2_texts.csv")
    df, dropped = build_rows(gold, texts, resolver)
    df.to_csv(args.out / "rows.csv", index=False)
    print("scopes in the 284:", gold.scope.value_counts().to_dict())
    print("rows:", len(df), df.kind.value_counts().to_dict(), "dropped:", dict(dropped))
    print(df.label.value_counts().to_string())


if __name__ == "__main__":
    main()

# ruff: noqa: E402  -- one-shot research script
