"""Add the Spanish aid (``spanish_title`` / ``spanish_text`` after ``text``) to the sector yes/no file (owner, 2026-11-12: spend approved).

    uv run scripts/translate_sector_yesno_2026_11_12.py [--out data] [--max-new 800]

Same translator and cache as the other tagging files (``retag_gold_2026_10_29.add_spanish``): an article already in the cache is not paid
for again. The file is copied to ``<out>/backup/`` first and every other cell is kept as it was.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import retag_gold_2026_10_29 as retag_cli

import news_nlp as db

NAME = "sector_yesno_to_review.csv"


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    args = parser.parse_args()
    source: Path = args.out / NAME
    with source.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if "spanish_text" in rows[0]:
        sys.exit(f"{source} already has the Spanish columns")
    result = retag_cli.add_spanish(
        args.out, rows, retag_cli._translator(), yes=True, max_new=args.max_new
    )
    backup = args.out / "backup" / f"{source.stem}_before_translate.csv"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, backup)
    with source.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f"{len(result)} rows -> {source} (backup: {backup})")


if __name__ == "__main__":
    main()
