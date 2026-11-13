"""Fix of translate_category_multi: the translation cache is keyed by article_id, so the rows of one body shared one
Spanish text. Re-translates the rows of the bodies with more than one row under a per-row id (article_id_ticker).
    uv run scripts/translate_category_multi_fix_2026_11_13.py [--yes]"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from collections import Counter
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
    a = ap.parse_args()
    out = ROOT / "data"
    path = out / "category_multi_to_tag.csv"
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    n = Counter(r["article_id"] for r in rows)
    redo = [
        {k: v for k, v in r.items() if not k.startswith("spanish_")}
        | {"article_id": f"{r['article_id']}_{r['ticker']}"}
        for r in rows
        if n[r["article_id"]] > 1
    ]
    print(len(redo), "rows to re-translate", flush=True)
    done = {
        r["article_id"]: r
        for r in cli.add_spanish(out, redo, cli._translator(), yes=a.yes, max_new=1000)
    }
    shutil.copy(
        path, out / "backup" / f"category_multi_to_tag_prefix_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    for r in rows:
        k = f"{r['article_id']}_{r['ticker']}"
        if k in done:
            r["spanish_title"], r["spanish_text"] = (
                done[k]["spanish_title"],
                done[k]["spanish_text"],
            )
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("fixed ->", path)


if __name__ == "__main__":
    main()

# ruff: noqa: E402  -- one-shot research script
