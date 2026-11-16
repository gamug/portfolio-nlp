# ruff: noqa: E402  -- src on the path first
"""The category metrics for the planned definition: four labels folded into `other`, which is not measured (owner, 2026-11-16).

    uv run scripts/category_planned_8_2026_11_16.py [--out docs/category_results_2026_11_15/planned_8_labels.json]

Not a retrained model: the predictions of the 12-label models and the tags are read through the fold. The accuracy and the macro F1 (the 7
measured labels) are over the rows whose tag is one of the 7 measured labels; a row of those predicted as `other` is wrong, the rows tagged
`other` are not scored. The macro F1 is over all rows, one versus rest on the 7 measured labels, like the per-label
precision and recall of the reports (a measured label predicted on a row tagged `other` is a false positive). Intervals are Wilson. Reads only committed files, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import category_multilabel as ml

R15 = ROOT / "docs" / "category_results_2026_11_15"
R13 = ROOT / "docs" / "category_results_2026_11_13"
FOLDED = (
    "product_innovation",
    "leadership_governance",
    "capital_shareholder_returns",
    "reputational_harm",
)
MEASURED = (
    "earnings_performance",
    "market_analyst_sentiment",
    "mergers_acquisitions",
    "legal_regulatory",
    "labor_human_capital",
    "partnerships_business_dev",
    "investor_activity",
)
L14 = ml.LABELS14
L12 = ml.merge_labels(tuple(L14), ml.FOLD_INTO_OTHER)


def fold(label: str) -> str:
    return "other" if label in FOLDED else label


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(100 * (c - r) / d, 1), round(100 * (c + r) / d, 1)


def macro_f1(gold: list[str], pred: list[str], labels: tuple[str, ...]) -> float:
    out = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred, strict=True))
        out.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(out) / len(out)


def measure(gold: list[str], pred: list[str]) -> dict[str, object]:
    """12-label accuracy as it is, and the planned figures over the rows with a measured tag."""
    acc12 = sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold)
    g8 = [fold(g) for g in gold]
    p8 = [fold(p) for p in pred]
    keep = [i for i, g in enumerate(g8) if g in MEASURED]
    k = sum(g8[i] == p8[i] for i in keep)
    lo, hi = wilson(k, len(keep))
    return {
        "rows": len(gold),
        "accuracy_12": round(100 * acc12, 1),
        "rows_measured": len(keep),
        "correct": k,
        "accuracy": round(100 * k / len(keep), 1),
        "ci": [lo, hi],
        "macro_f1": round(macro_f1(g8, p8, MEASURED), 3),
    }


def out_of_fold() -> dict[str, object]:
    sel = pd.read_csv(R15 / "single_predictions.csv", dtype=str)
    res: dict[str, object] = {"selected": measure(list(sel.gold), list(sel.top))}
    rows = pd.read_csv(ml.scores_dir(ROOT, "category_single") / "rows_single.csv", dtype=str)
    old = pd.read_csv(ml.scores_dir(ROOT, "category_multilabel") / "rows.csv", dtype=str)
    pos: dict[tuple[str, str], int] = {}
    for i, k in enumerate(zip(old.article_id, old.company, strict=True)):
        pos.setdefault(k, i)
    base = ml.merge_scores(
        np.load(ml.scores_dir(ROOT, "category_multilabel") / "oof_base.npz")["scores"],
        tuple(L14),
        ml.FOLD_INTO_OTHER,
    )
    gold, pred = [], []
    for k, g in zip(zip(rows.article_id, rows.company, strict=True), rows.gold, strict=True):
        if k in pos:
            gold.append(g)
            pred.append(L12[int(base[pos[k]].argmax())])
    res["untouched"] = measure(gold, pred)
    return res


def held_out() -> dict[str, object]:
    sel = pd.read_csv(R15 / "heldout_selected.csv", dtype=str)
    res: dict[str, object] = {"selected": measure(list(sel.gold), list(sel.pred))}
    base = pd.read_csv(R13 / "baseline_heldout.csv", dtype=str)
    pred = {
        k: p
        for k, p in zip(zip(base.article_id, base.company, strict=True), base.pred, strict=True)
    }
    keys = list(zip(sel.article_id, sel.company, strict=True))
    res["untouched"] = measure(
        list(sel.gold), ["other" if pred[k] in ml.FOLD_INTO_OTHER else pred[k] for k in keys]
    )
    return res


def real_sample() -> dict[str, object]:
    r = pd.read_csv(R15 / "real_sample.csv", dtype=str).dropna(subset=["tag"])
    return {
        "selected": measure(list(r.tag), list(r.selected)),
        "untouched": measure(list(r.tag), list(r.untouched)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=R15 / "planned_8_labels.json")
    a = ap.parse_args()
    result = {
        "folded": list(FOLDED),
        "measured": list(MEASURED),
        "out_of_fold": out_of_fold(),
        "held_out": held_out(),
        "real_sub_news": real_sample(),
    }
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
