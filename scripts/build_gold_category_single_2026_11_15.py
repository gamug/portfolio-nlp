# ruff: noqa: E402  -- one-shot script
"""One tag per (article, company): the owner's multi-tag sheet after the review of 2026-11-15 (T-187).

    uv run scripts/build_gold_category_single_2026_11_15.py [--force]

Reads ``data/category_multi_to_tag_claude_review.csv`` (the owner's sheet + a ``claude_review`` column). The tag of a row is the one the
review suggests ("SUGGEST: <tag>"), else the owner's single tag. ``competitive_actions`` and ``peer_comparison`` are read as ``other``
(owner, 2026-11-14), so the labels are the 12 of ``category_multilabel.merge_labels``. Row 166 is left out: its text has a gap where
the topic should be. Writes ``gold/category_single_gold.csv`` (``label`` the tag, ``label_owner`` the owner's cell, ``review`` the comment of the review, so a change can be audited without the git-ignored copy) and ``gold/texts/category_single_gold_texts.csv`` (refuses to overwrite
without ``--force``) and adds a ``final_tag`` column to the review copy of the sheet.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import category_multilabel as ml
from news_category_stage import LABELS

SHEET = Path("/workspaces/portfolio-nlp/data/category_multi_to_tag_claude_review.csv")
LEFT_OUT = {"166"}
TEXT_COLUMNS = [
    "review_order",
    "article_id",
    "ticker",
    "company",
    "pub_date",
    "source_domain",
    "title",
    "text",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet", type=Path, default=SHEET)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    gold_f = ROOT / "gold" / "category_single_gold.csv"
    text_f = ROOT / "gold" / "texts" / "category_single_gold_texts.csv"
    if (gold_f.exists() or text_f.exists()) and not a.force:
        sys.exit(f"{gold_f} exists; pass --force to overwrite")
    df = pd.read_csv(a.sheet, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    keep12 = ml.merge_labels(tuple(LABELS), ml.FOLD_INTO_OTHER)
    final = []
    for _, r in df.iterrows():
        m = re.match(r"SUGGEST: (\w+)", r.claude_review)
        tag = m.group(1) if m else r.label.strip()
        tag = ml.FOLD_INTO_OTHER.get(tag, tag)
        if tag not in keep12:
            sys.exit(
                f"row {r.review_order}: '{tag}' is not one of the 12 labels (a suggestion is missing)"
            )
        final.append(tag)
    df["final_tag"] = final
    with a.sheet.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(df.columns)
        w.writerows(df.values.tolist())
    use = df[~df.review_order.isin(LEFT_OUT)]
    gold_f.parent.mkdir(parents=True, exist_ok=True)
    text_f.parent.mkdir(parents=True, exist_ok=True)
    use.assign(label_owner=use.label, label=use.final_tag, review=use.claude_review)[
        ["review_order", "article_id", "ticker", "company", "label", "label_owner", "review"]
    ].to_csv(gold_f, index=False, lineterminator="\n")
    use[TEXT_COLUMNS].to_csv(text_f, index=False, lineterminator="\n")
    print(
        f"{len(use)} rows ({len(df) - len(use)} left out); changed from the owner's tag: {(use.final_tag != use.label).sum()}"
    )
    print(use.final_tag.value_counts().to_string())


if __name__ == "__main__":
    main()
