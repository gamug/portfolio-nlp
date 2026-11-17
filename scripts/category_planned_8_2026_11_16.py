# ruff: noqa: E402  -- src on the path first
"""The category metrics for the planned definition: four labels folded into `other`, which is not measured (owner, 2026-11-16).

    uv run scripts/category_planned_8_2026_11_16.py [--out docs/category_results_2026_11_15/planned_8_labels.json]

Not a retrained model: the predictions of the 12-label models and the tags are read through the fold. The accuracy and the macro F1 (the 7
measured labels) are over the rows whose tag is one of the 7 measured labels; a row of those predicted as `other` is wrong, the rows tagged
`other` are not scored. The macro F1 is over all rows, one versus rest on the 7 measured labels, like the per-label
precision and recall of the reports (a measured label predicted on a row tagged `other` is a false positive). `accuracy_8` scores **every**
row with `other` as an eighth label (both sides folded): it is higher because the four weak labels no longer count as errors among themselves.
`judge_step7` reads the blind judge on the 300 real sub-news through the fold, lenient (official) and strict. Intervals are Wilson. Reads only committed files, no GPU.
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
    k8 = sum(a == b for a, b in zip(g8, p8, strict=True))
    lo8, hi8 = wilson(k8, len(g8))
    return {
        "rows": len(gold),
        "accuracy_12": round(100 * acc12, 1),
        "rows_measured": len(keep),
        "correct": k,
        "accuracy": round(100 * k / len(keep), 1),
        "ci": [lo, hi],
        "macro_f1": round(macro_f1(g8, p8, MEASURED), 3),
        "correct_8": k8,
        "accuracy_8": round(100 * k8 / len(g8), 1),
        "ci_8": [lo8, hi8],
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


def judge_step7() -> dict[str, object]:
    """The blind LLM judge on the 300 real sub-news (T-209), read through the fold. The OFFICIAL reading (owner, 2026-11-17) is the
    lenient one: the model agrees when its label is the judge's first label **or its second**. Strict (first label only) is kept beside it.
    Four views per model: 12 labels, the 8 labels (the four folded into `other`, `other` scored as a label), the 7 measured labels
    (rows the judge puts in `other` after the fold are not scored)."""
    rows = pd.read_csv(R15 / "judge_step7_n300.csv", dtype=str, keep_default_na=False).to_dict(
        "records"
    )
    rows = [r for r in rows if r["parse_failed"] != "True"]
    out: dict[str, object] = {"rows": len(rows)}

    def view(model: str, kind: str, strict: bool, keep: list[dict[str, str]]) -> dict[str, object]:
        def ok(r: dict[str, str]) -> bool:
            conv = (lambda x: x) if kind == "12" else fold
            first = conv(r[model]) == conv(r["judge"])
            second = bool(r["second"]) and conv(r[model]) == conv(r["second"])
            return first or (second and not strict)

        k = sum(ok(r) for r in keep)
        lo, hi = wilson(k, len(keep))
        return {
            "rows": len(keep),
            "correct": k,
            "accuracy": round(100 * k / len(keep), 1),
            "ci": [lo, hi],
        }

    for model in ("selected", "untouched", "tag"):
        res: dict[str, object] = {}
        for kind, keep in (
            ("12", rows),
            ("8", rows),
            ("7", [r for r in rows if fold(r["judge"]) != "other"]),
        ):
            res[f"lenient_{kind}"] = view(model, kind, False, keep)
            res[f"strict_{kind}"] = view(model, kind, True, keep)
        out[model] = res
    return out


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
        "judge_step7": judge_step7(),
    }
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
