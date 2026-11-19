"""Score the owner's partial main-company marks (T-170) against the relevance model of step 3.

    uv run scripts/score_main_company_marks_2026_11_16.py [--marks gold/main_company_marks_2026_11_19.csv] [--exclude 335155,343149,260642]

Reads the tracked marks (the owner's ``es_principal`` ``yes`` or ``no``, the relevance score of the pair and the relevance band of its item; the
2026-11-16 version is ``gold/main_company_marks.csv``, the 2026-11-19 one ``gold/main_company_marks_2026_11_19.csv``, built by
``build_main_company_marks_2026_11_19.py``). A pair is called main when its score is at the threshold or over it. Nothing is written. The pair and
piece figures are unweighted (the six bands were drawn evenly, so they are not corpus rates), with the SHA-256 of the file scored; the last block
re-weights the pieces by the size of their band in the corpus (``POPULATION``) and says what the owner's removals do to it.
Used for ``docs/main_company_training.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import NamedTuple

THRESHOLDS = (0.3, 0.5, 0.7)
#: Pieces of the ``units5`` cut with a candidate company in each relevance band of the piece's highest score (``to_tag.strata.json``, ``population``).
POPULATION = {0: 13756, 1: 9724, 2: 4994, 3: 4980, 4: 12221, 5: 105945}
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261119
DEFAULT_MARKS = Path("gold/main_company_marks_2026_11_19.csv")
DEFAULT_EXCLUDE = ("335155", "343149", "260642")  # lists of tickers and picks, every row `yes`


class Mark(NamedTuple):
    item: str
    ticker: str
    main: bool
    score: float
    band: int


def wilson(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / 2 / n) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(path: Path) -> list[Mark]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [
            Mark(
                r["item_id"],
                r["ticker"],
                r["es_principal"] == "yes",
                float(r["relevance_score"]),
                int(r["band"]),
            )
            for r in csv.DictReader(fh)
        ]


def at_threshold(rows: list[Mark], threshold: float) -> dict[str, int]:
    """Counts of the one-vs-rest table at a threshold: called main, true main, and the same for not main."""
    yes = sum(r.main for r in rows)
    return {
        "rows": len(rows),
        "yes": yes,
        "no": len(rows) - yes,
        "called_main": sum(1 for r in rows if r.score >= threshold),
        "tp_main": sum(1 for r in rows if r.main and r.score >= threshold),
        "called_not": sum(1 for r in rows if r.score < threshold),
        "tp_not": sum(1 for r in rows if not r.main and r.score < threshold),
    }


def by_item(rows: list[Mark], threshold: float = 0.5) -> dict[str, int]:
    items: dict[str, list[Mark]] = defaultdict(list)
    for r in rows:
        items[r.item].append(r)
    with_main = [k for k, v in items.items() if any(r.main for r in v)]
    called = [k for k, v in items.items() if any(r.score >= threshold for r in v)]
    return {
        "items": len(items),
        "with_main": len(with_main),
        "found": sum(
            1 for k in with_main if any(r.main and r.score >= threshold for r in items[k])
        ),
        "called": len(called),
        "agreed": sum(1 for k in called if any(r.main and r.score >= threshold for r in items[k])),
    }


def by_band(rows: list[Mark]) -> dict[int, tuple[int, int]]:
    return {
        b: (sum(1 for r in rows if r.band == b and r.main), sum(1 for r in rows if r.band == b))
        for b in range(6)
    }


def auc(rows: list[Mark]) -> float:
    """ROC AUC of the relevance score for the owner's `yes` against `no` (ties count half); unweighted."""
    yes = [r.score for r in rows if r.main]
    no = [r.score for r in rows if not r.main]
    wins = sum(1.0 if y > n else 0.5 if y == n else 0.0 for y in yes for n in no)
    return wins / (len(yes) * len(no))


def group(rows: list[Mark]) -> dict[str, list[Mark]]:
    items: dict[str, list[Mark]] = defaultdict(list)
    for r in rows:
        items[r.item].append(r)
    return items


def interval(
    rows: list[Mark], stat: Callable[[list[Mark]], float], level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of ``stat`` by resampling whole pieces (the rows of a piece are not independent)."""
    items = list(group(rows).values())
    rng = random.Random(BOOTSTRAP_SEED)  # noqa: S311  (a seeded resample, not a secret)
    values = sorted(
        stat([r for it in rng.choices(items, k=len(items)) for r in it]) for _ in range(BOOTSTRAP)
    )
    lo = int(BOOTSTRAP * (1 - level) / 2)
    return values[lo], values[BOOTSTRAP - 1 - lo]


def recall_rows(rows: list[Mark], threshold: float = 0.5) -> float:
    yes = [r for r in rows if r.main]
    return sum(r.score >= threshold for r in yes) / len(yes) if yes else 0.0


def hit_rate(rows: list[Mark], threshold: float = 0.5) -> float:
    c = by_item(rows, threshold)
    return c["found"] / c["with_main"] if c["with_main"] else 0.0


def weighted(rows: list[Mark], threshold: float) -> dict[str, float]:
    """Piece-level recall and precision with each band weighted by its corpus size (a band with no piece is skipped)."""
    items = group(rows)
    band = {k: v[0].band for k, v in items.items()}
    with_main = found = called = agreed = 0.0
    for b, size in POPULATION.items():
        its = [k for k in items if band[k] == b]
        if not its:
            continue
        w = size / len(its)
        for k in its:
            rs = items[k]
            has = any(r.main for r in rs)
            hit = any(r.main and r.score >= threshold for r in rs)
            with_main += w * has
            found += w * hit
            called += w * any(r.score >= threshold for r in rs)
            agreed += w * hit
    return {
        "with_main_share": with_main / sum(POPULATION.values()),
        "recall": found / with_main,
        "precision": agreed / called,
    }


def by_owner_mains(rows: list[Mark], threshold: float = 0.5) -> dict[str, tuple[int, int]]:
    """Pieces with one owner main against pieces with two or more: (found, pieces) for each."""
    out = {"one": [0, 0], "several": [0, 0]}
    for rs in group(rows).values():
        n = sum(r.main for r in rs)
        if n == 0:
            continue
        k = "one" if n == 1 else "several"
        out[k][1] += 1
        out[k][0] += any(r.main and r.score >= threshold for r in rs)
    return {k: (v[0], v[1]) for k, v in out.items()}


def pct(x: int, n: int) -> str:
    return f"{100 * x / n:.1f}%" if n else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--marks", type=Path, default=DEFAULT_MARKS)
    parser.add_argument(
        "--exclude", default=",".join(DEFAULT_EXCLUDE), help="items to leave out in the last line"
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in THRESHOLDS),
        help="thresholds to score, comma separated",
    )
    args = parser.parse_args()
    thresholds = tuple(float(t) for t in args.thresholds.split(","))
    rows = load(args.marks)
    print(f"{args.marks}: sha256 {hashlib.sha256(args.marks.read_bytes()).hexdigest()}")
    print(
        f"marked rows {len(rows)} of {len({r.item for r in rows})} items: {sum(r.main for r in rows)} yes, {sum(not r.main for r in rows)} no"
    )
    for t in thresholds:
        c = at_threshold(rows, t)
        lo, hi = wilson(c["tp_main"], c["called_main"])
        rlo, rhi = wilson(c["tp_main"], c["yes"])
        print(
            f"threshold {t}: main P {c['tp_main']}/{c['called_main']} = {pct(c['tp_main'], c['called_main'])} [{100 * lo:.1f}, {100 * hi:.1f}], "
            f"R {c['tp_main']}/{c['yes']} = {pct(c['tp_main'], c['yes'])} [{100 * rlo:.1f}, {100 * rhi:.1f}]; "
            f"not main P {c['tp_not']}/{c['called_not']} = {pct(c['tp_not'], c['called_not'])}, R {c['tp_not']}/{c['no']} = {pct(c['tp_not'], c['no'])}"
        )
    alo, ahi = interval(rows, auc)
    print(
        f"AUC of the score for yes against no: {auc(rows):.3f} [{alo:.3f}, {ahi:.3f}] (resampling pieces; the relevance model's out-of-fold AUC on summaries: 0.896)"
    )
    for t in thresholds:
        i = by_item(rows, t)
        flo, fhi = wilson(i["found"], i["with_main"])
        rlo, rhi = interval(rows, partial(hit_rate, threshold=t))
        print(
            f"pieces at {t}: {i['items']}, with a main company {i['with_main']}, found by the model {i['found']} ({pct(i['found'], i['with_main'])}) "
            f"[Wilson {100 * flo:.1f}, {100 * fhi:.1f}; resampling {100 * rlo:.1f}, {100 * rhi:.1f}]; "
            f"model calls main on {i['called']}, agreed {i['agreed']} ({pct(i['agreed'], i['called'])})"
        )
    for k, (f, n) in by_owner_mains(rows).items():
        print(f"pieces with {k} owner main at 0.5: found {f}/{n} ({pct(f, n)})")
    print("pieces per band:", {b: len({r.item for r in rows if r.band == b}) for b in range(6)})
    for t in thresholds:
        w = weighted(rows, t)
        print(
            f"re-weighted by band size at {t}: share of pieces with an owner main {100 * w['with_main_share']:.1f}%, "
            f"piece recall {100 * w['recall']:.1f}%, piece precision {100 * w['precision']:.1f}%"
        )
    print("yes by band:", {b: f"{y}/{n}" for b, (y, n) in by_band(rows).items()})
    skip = set(args.exclude.split(","))
    rest = [r for r in rows if r.item not in skip]
    c = at_threshold(rest, 0.5)
    print(
        f"without the items {sorted(skip)}: {c['rows']} rows ({c['yes']} yes), main P {c['tp_main']}/{c['called_main']} = {pct(c['tp_main'], c['called_main'])}, "
        f"R {c['tp_main']}/{c['yes']} = {pct(c['tp_main'], c['yes'])}; not main P {c['tp_not']}/{c['called_not']} = {pct(c['tp_not'], c['called_not'])}, "
        f"R {c['tp_not']}/{c['no']} = {pct(c['tp_not'], c['no'])}"
    )


if __name__ == "__main__":
    main()
