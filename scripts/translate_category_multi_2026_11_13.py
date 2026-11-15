"""Add spanish_title / spanish_text to data/category_multi_to_tag.csv (owner 2026-11-13: translate).
    uv run scripts/translate_category_multi_2026_11_13.py [--yes] [--max-new 1000]
Without --yes it prints the estimate and buys nothing. The original is kept in data/backup/ first."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import retag_gold_2026_10_29 as cli

import news_nlp as db


def main() -> None:
    db.env.load_env_file()
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--max-new", type=int, default=cli.DEFAULT_MAX_NEW)
    a = ap.parse_args()
    out = ROOT / "data"
    path = out / "category_multi_to_tag.csv"
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if "spanish_text" in rows[0]:
        sys.exit("already translated")
    # the cache is keyed by article_id: rows of one body need their own key (article_id_ticker) or they share one translation
    keyed = [r | {"article_id": f"{r['article_id']}_{r['ticker']}"} for r in rows]
    done = cli.add_spanish(out, keyed, cli._translator(), yes=a.yes, max_new=a.max_new)
    rows = [d | {"article_id": o["article_id"]} for d, o in zip(done, rows, strict=False)]
    (out / "backup").mkdir(exist_ok=True)
    shutil.copy(
        path, out / "backup" / f"category_multi_to_tag_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows with Spanish -> {path}")


if __name__ == "__main__":
    main()

# ruff: noqa: E402  -- one-shot research script
