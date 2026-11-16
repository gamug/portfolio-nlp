# ruff: noqa: PLR2004  (one-off export and scoring script: thresholds are the owner's decisions, the draw is not cryptographic)
"""Score the owner's partial main-company marks (T-170) against the relevance model of step 3.

    uv run scripts/score_main_company_marks_2026_11_16.py [--dir data/main_company]

Reads ``to_tag.csv`` (the owner's ``es_principal`` marks, ``yes`` or ``no``) and ``to_tag.strata.json`` (the relevance score of every (item, ticker),
and the relevance band of each item). A pair is called main when its score is at the threshold or over it. Nothing is written; the figures are
printed unweighted (the six bands were drawn evenly, so they are not corpus rates). Used for ``docs/main_company_training.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(x: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "[-]"
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / 2 / n) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"[{100 * (c - h):.1f}, {100 * (c + h):.1f}]"


def load(folder: Path) -> list[tuple[str, str, bool, float, int]]:
    strata = json.loads((folder / "to_tag.strata.json").read_text(encoding="utf-8"))["items"]
    rows = []
    with (folder / "to_tag.csv").open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            mark = (r["es_principal"] or "").strip().lower()
            if mark not in ("yes", "no", "si"):
                continue
            item = strata[r["item_id"]]
            rows.append(
                (r["item_id"], r["ticker"], mark != "no", item["scores"][r["ticker"]], item["band"])
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("data/main_company"))
    parser.add_argument(
        "--exclude", default="335155,343149,260642", help="items to leave out in the last line"
    )
    args = parser.parse_args()
    rows = load(args.dir)
    yes = sum(r[2] for r in rows)
    print(
        f"marked rows {len(rows)} of {len({r[0] for r in rows})} items: {yes} yes, {len(rows) - yes} no"
    )
    for t in (0.3, 0.5, 0.7):
        tp = sum(1 for r in rows if r[2] and r[3] >= t)
        called = sum(1 for r in rows if r[3] >= t)
        tn = sum(1 for r in rows if not r[2] and r[3] < t)
        not_called = sum(1 for r in rows if r[3] < t)
        print(
            f"threshold {t}: main P {tp}/{called} = {100 * tp / called:.1f}% {wilson(tp, called)}, R {tp}/{yes} = {100 * tp / yes:.1f}% {wilson(tp, yes)}; "
            f"not main P {tn}/{not_called} = {100 * tn / not_called:.1f}%, R {tn}/{len(rows) - yes} = {100 * tn / (len(rows) - yes):.1f}%"
        )
    items = defaultdict(list)
    for r in rows:
        items[r[0]].append(r)
    with_main = [k for k, v in items.items() if any(r[2] for r in v)]
    found = sum(1 for k in with_main if any(r[2] and r[3] >= 0.5 for r in items[k]))
    called = [k for k, v in items.items() if any(r[3] >= 0.5 for r in v)]
    agreed = sum(1 for k in called if any(r[2] and r[3] >= 0.5 for r in items[k]))
    print(
        f"items {len(items)}: with a main company {len(with_main)}, found by the model {found}; model calls main on {len(called)}, agreed {agreed}"
    )
    print(
        "yes by band:",
        {
            b: f"{sum(1 for r in rows if r[4] == b and r[2])}/{sum(1 for r in rows if r[4] == b)}"
            for b in range(6)
        },
    )
    big = set(args.exclude.split(","))
    rest = [r for r in rows if r[0] not in big]
    y2 = sum(r[2] for r in rest)
    tp = sum(1 for r in rest if r[2] and r[3] >= 0.5)
    called2 = sum(1 for r in rest if r[3] >= 0.5)
    print(
        f"without the items {sorted(big)} (lists of tickers and picks, every row `yes`): {len(rest)} rows, main P {tp}/{called2}, R {tp}/{y2}"
    )


if __name__ == "__main__":
    main()
