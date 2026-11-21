"""Score the sentiment step on the 190 pieces the owner tagged (PR #201; the version 2 re-tag of 2026-11-20).

    uv run scripts/score_sentiment_val190_2026_11_16.py [--gold gold/sentiment_val190_2026_11_20_v2.csv]

Reads the tracked marks (the owner's ``sentiment`` of each piece and the model's ``predicted`` class) and prints the one-vs-rest tables of
``docs/sentiment_training.md`` ("Held-out check on 190 pieces"): the pooled 190, and each draw alone. Precision is exact within each predicted
class. Recall is the share of the tagged class found **in the sample**. The sample is stratified by the model's prediction, so the sample recall
describes the sampled pieces; it is not an estimate of the recall over the corpus, and no figure here is weighted to the pool. Nothing is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import NamedTuple

CLASSES = ("positive", "negative", "neutral")


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
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [Piece(r["batch"], r["predicted"], r["sentiment"]) for r in csv.DictReader(fh)]


def class_table(rows: list[Piece]) -> dict[str, dict[str, float]]:
    """Per class: the precision within the predicted class, and the recall within the sampled tagged class."""
    called = {q: sum(1 for r in rows if r.predicted == q) for q in CLASSES}
    out: dict[str, dict[str, float]] = {}
    for c in CLASSES:
        tp = sum(1 for r in rows if r.predicted == c == r.tag)
        tagged = sum(1 for r in rows if r.tag == c)
        d: dict[str, float] = {"tp": tp, "called": called[c], "tagged": tagged}
        d["precision"] = tp / called[c] if called[c] else float("nan")
        d["recall"] = tp / tagged if tagged else float("nan")
        out[c] = d
    return out


def accuracy(rows: list[Piece]) -> tuple[int, int]:
    return sum(1 for r in rows if r.predicted == r.tag), len(rows)


def confusion(rows: list[Piece]) -> dict[str, list[int]]:
    return {
        p: [sum(1 for r in rows if r.predicted == p and r.tag == g) for g in CLASSES]
        for p in CLASSES
    }


def _pct(v: float) -> str:
    return "n/a" if math.isnan(v) else f"{100 * v:.1f}%"


def report(rows: list[Piece], label: str) -> None:
    ok, n = accuracy(rows)
    lo, hi = wilson(ok, n)
    print(
        f"== {label}: {n} pieces, tags {[sum(1 for r in rows if r.tag == c) for c in CLASSES]}, "
        f"accuracy {ok}/{n} = {100 * ok / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
    )
    for c, d in class_table(rows).items():
        plo, phi = wilson(int(d["tp"]), int(d["called"]))
        rlo, rhi = wilson(int(d["tp"]), int(d["tagged"]))
        print(
            f"   {c}: P {_pct(d['precision'])} [{100 * plo:.1f}, {100 * phi:.1f}] ({int(d['tp'])}/{int(d['called'])}), "
            f"R {_pct(d['recall'])} [{100 * rlo:.1f}, {100 * rhi:.1f}] ({int(d['tp'])}/{int(d['tagged'])})"
        )
    print("   confusion (rows model, columns tag; positive / negative / neutral):", confusion(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gold", type=Path, default=Path("gold/sentiment_val190_2026_11_20_v2.csv")
    )
    args = parser.parse_args()
    rows = load(args.gold)
    print(f"{args.gold}: sha256 {hashlib.sha256(args.gold.read_bytes()).hexdigest()}")
    report(rows, "all 190 pooled")
    report([r for r in rows if r.batch == "90"], "first draw (90, stratified by the prediction)")
    report([r for r in rows if r.batch == "100"], "second draw (100, random)")


if __name__ == "__main__":
    main()
