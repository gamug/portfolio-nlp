"""Draw the fixed 240-body set of the prompt gate (Work item 18, T-146c) from the review file.

    uv run scripts/draw_gate240_2026_11_04.py [draw] [--out data] [--file NAME] [--seed 20261104]
    uv run scripts/draw_gate240_2026_11_04.py merge [--out data] [--file NAME]

``draw`` writes ``data/gate240_ids.csv`` (the 240 with their stratum; **drawn once**, a second run
does not draw again) and ``data/gate240_to_tag.csv``: a tagging file of its own with the 240 rows
in the same columns as the review file (title, text, the Spanish ``spanish_title`` /
``spanish_text`` after ``text``, the v1 columns and the columns to tag), the tags the owner has
already made included. That file is never overwritten once it exists. ``merge`` copies the tags
made in it into the review file as a new dated file (nothing is replaced, a row with different
tags in both is listed). The review file is only read.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime
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


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def cmd_draw(out: Path, file: str, seed: int) -> None:
    rows = retag.read_tag_file(out / file)
    if (out / IDS_NAME).exists():  # the set is fixed
        chosen = _read(out / IDS_NAME)
        print(f"{out / IDS_NAME} exists: the set is not drawn again")
    else:
        chosen = retag.draw_gate(rows, seed=seed)
        _write(out / IDS_NAME, chosen)
    target = out / TO_TAG_NAME
    if target.exists() and "scope" in _read(target)[0]:
        done = sum(1 for r in _read(target) if (r.get("scope") or "").strip())
        sys.exit(
            f"{target} exists and is your tagging file ({done} tagged): it is not overwritten. "
            "Use 'merge' to bring its tags into the review file."
        )
    gate = retag.gate_rows(chosen, rows)
    try:
        _write(target, gate)
    except PermissionError:
        sys.exit(
            f"cannot write {target}: it is open in another program (Excel?); close it and run again"
        )
    tagged = sum(1 for r in gate if (r.get("scope") or "").strip())
    print(f"{len(gate)} bodies -> {target} ({tagged} already tagged, {len(gate) - tagged} to tag)")
    print("by stratum:", dict(Counter(r["stratum"] for r in chosen)))


def cmd_merge(out: Path, file: str) -> None:
    """Copy the tags made in the gate file into the review file, as a NEW file (dated name): neither
    input is touched, a review row that already holds different tags is not overwritten."""
    review = retag.read_tag_file(out / file)
    gate = retag.read_tag_file(out / TO_TAG_NAME)
    merged, copied, conflicts = retag.merge_tags(review, gate)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    target = out / f"{Path(file).stem}_merged_{stamp}.csv"
    _write(target, merged)
    print(f"{len(merged)} rows -> {target}: {copied} tagged rows copied from {TO_TAG_NAME}")
    if conflicts:
        print(f"  not overwritten (different tags in both files): {', '.join(conflicts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("draw", "merge"), default="draw")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--file", default=retag.REVIEW_NAME)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    if args.command == "merge":
        cmd_merge(args.out, args.file)
    else:
        cmd_draw(args.out, args.file, args.seed)


if __name__ == "__main__":
    main()
