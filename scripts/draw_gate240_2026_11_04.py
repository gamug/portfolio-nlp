"""Draw the fixed 240-body set of the prompt gate (Work item 18, T-146c) from the review file.

    uv run scripts/draw_gate240_2026_11_04.py [--out data] [--file NAME] [--seed 20261104]

Writes ``data/gate240_ids.csv`` (the 240, with the stratum and whether the owner has tagged it)
and ``data/gate240_to_tag.csv`` (the ones still to tag, in file order). **The set is drawn once**:
the script refuses to run if ``gate240_ids.csv`` exists. The review file is only read.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import retag

IDS_NAME = "gate240_ids.csv"
TO_TAG_NAME = "gate240_to_tag.csv"


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--file", default=retag.REVIEW_NAME)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    if (args.out / IDS_NAME).exists():
        sys.exit(f"{args.out / IDS_NAME} exists: the gate set is drawn once")
    rows = retag.read_tag_file(args.out / args.file)
    chosen = retag.draw_gate(rows, seed=args.seed)
    _write(args.out / IDS_NAME, chosen)
    to_tag = [r for r in chosen if r["already_tagged"] == "no"]
    if to_tag:
        _write(args.out / TO_TAG_NAME, to_tag)
    print(f"{len(chosen)} bodies -> {args.out / IDS_NAME}; {len(to_tag)} still to tag")
    print("by stratum:", dict(Counter(r["stratum"] for r in chosen)))
    print("still to tag:", dict(Counter(r["stratum"] for r in to_tag)))


if __name__ == "__main__":
    main()
