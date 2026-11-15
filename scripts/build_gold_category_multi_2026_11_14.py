"""Build the tracked set of the owner's multi-label category tags (T-187, 2026-11-14) from the finished working file.

    uv run scripts/build_gold_category_multi_2026_11_14.py [--data data] [--gold gold] [--force]

Reads ``<data>/category_multi_to_tag.csv`` (the owner's file, never written). Writes ``<gold>/category_multi_gold.csv`` (the tags:
``review_order, article_id, ticker, company, labels``) and ``<gold>/texts/category_multi_gold_texts.csv`` (the texts, without the Spanish
aid, docs/gold_datasets.md). A bad cell stops the build and names the row; a row of only ``uncertain`` is left out and listed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import category_multilabel as ml  # noqa: E402
from news_category_stage import LABELS  # noqa: E402


def write(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--gold", type=Path, default=ROOT / "gold")
    ap.add_argument("--force", action="store_true", help="overwrite the tracked files")
    a = ap.parse_args()
    gold_path = a.gold / "category_multi_gold.csv"
    texts_path = a.gold / "texts" / "category_multi_gold_texts.csv"
    if not a.force and (gold_path.exists() or texts_path.exists()):
        sys.exit(f"{gold_path} or {texts_path} exists: use --force to rewrite it")
    with (a.data / "category_multi_to_tag.csv").open(newline="", encoding="utf-8-sig") as f:
        sheet = list(csv.DictReader(f))
    try:
        gold, texts, left_out = ml.gold_rows(sheet, LABELS)
    except ValueError as e:
        sys.exit(f"the sheet has a bad row: {e}")
    write(gold_path, ml.GOLD_COLUMNS, gold)
    write(texts_path, ml.TEXT_COLUMNS, texts)
    print(
        f"{len(gold)} rows -> {gold_path}; texts -> {texts_path}; left out (only uncertain): {left_out}"
    )


if __name__ == "__main__":
    main()
