"""Score the owner's answers on the fresh sector test (2026-11-13; see ``draw_sector_fresh_2026_11_13.py`` for how it was drawn).

    uv run scripts/score_sector_fresh_2026_11_13.py [--review data/sector_fresh_to_review_es.csv] [--key data/sector_fresh_key.csv]

Prints, from the owner's ``correct`` / ``right_sector`` answers and the key file (which rows were controls, the margin the step saw):

* the precision of the sectors the step assigned (controls apart), with its Wilson interval, split by what let the body through
  (the margin, or only the companies' vote) and by sector;
* the controls (a ``yes`` to a control means a reviewer that agrees too soon);
* the bodies the gate sent out_of_scope: how many the owner agrees to drop, and how many had a sector the step had guessed right.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

THRESHOLD = 0.0293  # config/sector_step.json margin_threshold
Z = 1.96


def wilson(hits: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return centre - half, centre + half


def line(label: str, rows: list[dict[str, str]]) -> None:
    n = len(rows)
    yes = sum(r["correct"].strip().lower() == "yes" for r in rows)
    if n == 0:
        print(f"{label:40s} n=0")
        return
    lo, hi = wilson(yes, n)
    print(f"{label:40s} n={n:3d} yes={yes:3d} {yes / n:6.1%}  (95%: {lo:.0%} to {hi:.0%})")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review", type=Path, default=Path("data/sector_fresh_to_review_es.csv"))
    parser.add_argument("--key", type=Path, default=Path("data/sector_fresh_key.csv"))
    args = parser.parse_args()
    key = {k["review_order"]: k for k in read(args.key)}
    rows = [
        {**r, **{"_" + a: b for a, b in key[r["review_order"]].items()}} for r in read(args.review)
    ]
    assigned = [r for r in rows if r["_status"] == "assigned" and r["_control"] == "no"]
    controls = [r for r in rows if r["_control"] == "yes"]
    dropped = [r for r in rows if r["_status"] == "out_of_scope"]
    line("assigned (precision)", assigned)
    by_margin = [r for r in assigned if float(r["_margin"]) >= THRESHOLD]
    line("  let through by the margin", by_margin)
    line("  let through by the vote only", [r for r in assigned if r not in by_margin])
    for code in sorted({r["model_sector_code"] for r in assigned}):
        line(f"  sector {code}", [r for r in assigned if r["model_sector_code"] == code])
    line("controls (a yes is a reviewer too fast)", controls)
    line("out_of_scope: the owner agrees", dropped)
    wrong = [r for r in dropped if r["correct"].strip().lower() != "yes"]
    guessed = [r for r in wrong if r["right_sector"].strip() == r["_candidate"]]
    print(
        f"out_of_scope the owner would have given a sector: {len(wrong)}; the step's best guess was that sector in {len(guessed)}"
    )
    print("assigned and wrong:")
    for r in assigned:
        if r["correct"].strip().lower() != "yes":
            print(
                f"  #{r['article_id']} {r['model_verdict']} | right: {r['right_sector'] or '-'} | vote {r['_vote_sector'] or '-'} | {r['notes'][:80]}"
            )


if __name__ == "__main__":
    main()
