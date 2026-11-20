"""Score the sentiment step on the 190 pieces the owner tagged (PR #201).

    uv run scripts/score_sentiment_val190_2026_11_16.py [--gold gold/sentiment_val190.csv]

Reads the tracked marks (``gold/sentiment_val190.csv``, built by ``build_val_gold_2026_11_16.py``: the owner's ``sentiment`` of each piece and the
model's ``predicted`` class) and prints the one-vs-rest tables of ``docs/sentiment_training.md`` ("Held-out check on 190 pieces"): the pooled 190, each
draw alone, and the weighted figures. The first draw (``batch`` 90) took 30 pieces per predicted class from a pool of 57,826; the second (``batch`` 100)
is a simple random draw from a stricter pool. **Recall weighted** gives each predicted class its pool size over the pieces drawn from it; it treats the two
draws as samples of the one pool of the first draw (an approximation, said in the doc). Nothing is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import NamedTuple

CLASSES = ("positive", "negative", "neutral")
POOL = {"positive": 38_559, "negative": 18_847, "neutral": 420}


class Piece(NamedTuple):
    batch: str
    predicted: str
    tag: str


def wilson(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / 2 / n) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(path: Path) -> list[Piece]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [Piece(r["batch"], r["predicted"], r["sentiment"]) for r in csv.DictReader(fh)]


def class_table(rows: list[Piece], weighted: bool = False) -> dict[str, dict[str, float]]:
    """Per class: precision, recall in the sample, and (when asked) the recall and F1 weighted by the pool sizes."""
    called = {q: sum(1 for r in rows if r.predicted == q) for q in CLASSES}
    out: dict[str, dict[str, float]] = {}
    for c in CLASSES:
        tp = sum(1 for r in rows if r.predicted == c == r.tag)
        tagged = sum(1 for r in rows if r.tag == c)
        d: dict[str, float] = {"tp": tp, "called": called[c], "tagged": tagged}
        d["precision"] = tp / called[c] if called[c] else float("nan")
        d["recall"] = tp / tagged if tagged else float("nan")
        if weighted:
            est = {
                q: (
                    POOL[q] * sum(1 for r in rows if r.predicted == q and r.tag == c) / called[q]
                    if called[q]
                    else 0.0
                )
                for q in CLASSES
            }
            d["recall_weighted"] = est[c] / sum(est.values()) if sum(est.values()) else float("nan")
        out[c] = d
    return out


def accuracy(rows: list[Piece]) -> tuple[int, int]:
    return sum(1 for r in rows if r.predicted == r.tag), len(rows)


def accuracy_weighted(rows: list[Piece]) -> float:
    total = sum(POOL.values())
    return (
        sum(
            POOL[q] * sum(1 for r in rows if r.predicted == q == r.tag) / n
            for q in CLASSES
            if (n := sum(1 for r in rows if r.predicted == q))
        )
        / total
    )


def confusion(rows: list[Piece]) -> dict[str, list[int]]:
    return {
        p: [sum(1 for r in rows if r.predicted == p and r.tag == g) for g in CLASSES]
        for p in CLASSES
    }


def _pct(v: float) -> str:
    return "n/a" if math.isnan(v) else f"{100 * v:.1f}%"


def report(rows: list[Piece], label: str, weighted: bool) -> None:
    ok, n = accuracy(rows)
    lo, hi = wilson(ok, n)
    extra = f"; weighted to the pool {100 * accuracy_weighted(rows):.1f}%" if weighted else ""
    print(
        f"== {label}: {n} pieces, tags {[sum(1 for r in rows if r.tag == c) for c in CLASSES]}, accuracy {ok}/{n} = {100 * ok / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]{extra}"
    )
    for c, d in class_table(rows, weighted).items():
        plo, phi = wilson(int(d["tp"]), int(d["called"]))
        rlo, rhi = wilson(int(d["tp"]), int(d["tagged"]))
        line = (
            f"   {c}: P {_pct(d['precision'])} [{100 * plo:.1f}, {100 * phi:.1f}] ({int(d['tp'])}/{int(d['called'])}), "
            f"R {_pct(d['recall'])} [{100 * rlo:.1f}, {100 * rhi:.1f}] ({int(d['tp'])}/{int(d['tagged'])})"
        )
        if weighted:
            p, r = d["precision"], d["recall"]
            rw = d["recall_weighted"]
            line += f", R weighted {_pct(rw)}, F1 {_pct(2 * p * r / (p + r))} / weighted {_pct(2 * p * rw / (p + rw))}"
        print(line)
    print("   confusion (rows model, columns tag; positive / negative / neutral):", confusion(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gold", type=Path, default=Path("gold/sentiment_val190_2026_11_20_v2.csv")
    )
    args = parser.parse_args()
    rows = load(args.gold)
    print(f"{args.gold}: sha256 {hashlib.sha256(args.gold.read_bytes()).hexdigest()}")
    report(rows, "all 190 pooled (weighted to the pool of 57,826)", True)
    report(
        [r for r in rows if r.batch == "90"], "first draw (90, stratified by the prediction)", True
    )
    report(
        [r for r in rows if r.batch == "100"],
        "second draw (100, random: no weighting needed)",
        False,
    )


if __name__ == "__main__":
    main()
