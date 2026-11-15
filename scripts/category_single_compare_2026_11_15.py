# ruff: noqa: E402  -- one-shot research script
"""Compares the two single-tag entailment fine-tunes of 2026-11-15 and saves the predictions without text (T-187).

    uv run scripts/category_single_compare_2026_11_15.py [--out docs/category_results_2026_11_15]

``data/category_single_v1/`` is the first run (the sheet's 999 rows re-tagged, the 225 older rows with their old tags: 1,224 rows);
``data/category_single/`` is the second (the older rows read again with the same rules, 8 dropped: 1,216 rows). The rows both runs
have are matched by (article, company). Tables, per class, are "first -> second": precision, recall, ROC AUC (by ranks), for the
top label of every row, out of fold. Views: each run on its own tags; both runs on the NEW tags (does retraining help, apart from the
cleaner tags?); the older 225 rows and the sheet's 999 rows apart. Also writes ``single_predictions.csv`` (no text).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import category_multilabel as ml

L14 = ml.LABELS14

L12 = list(ml.merge_labels(tuple(L14), ml.FOLD_INTO_OTHER))


def load(folder: str) -> tuple[pd.DataFrame, np.ndarray]:
    rows = pd.read_csv(ml.scores_dir(ROOT, folder) / "rows_single.csv", dtype=str)
    rows["gold"] = rows.gold.str.split(",").str[0]
    return rows, np.load(ml.scores_dir(ROOT, folder) / "oof_single.npz")["scores"]


def measure(
    S: np.ndarray, g: np.ndarray
) -> tuple[float, float, dict[str, tuple[float, float, float, int]]]:
    pred = np.array(L12)[S.argmax(1)]
    out, f1 = {}, []
    for j, lab in enumerate(L12):
        tp = int(((pred == lab) & (g == lab)).sum())
        p, r = tp / max(int((pred == lab).sum()), 1), tp / max(int((g == lab).sum()), 1)
        auc = (
            ml.roc_auc(S[:, j], [x == lab for x in g])
            if (g == lab).any() and (g != lab).any()
            else float("nan")
        )
        out[lab] = (p, r, auc, int((g == lab).sum()))
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return float((pred == g).mean()), float(np.mean(f1)), out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    r2, s2 = load("category_single")
    r1, s1 = load("category_single_v1")
    # a pair that carried two tags in the first run keeps one row in the second: match it to the first-run row with the same tag when
    # there is one (an unchanged row is then not counted as changed), else to the first row of the pair
    by_key: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(zip(r1.article_id, r1.company, strict=True)):
        by_key.setdefault(key, []).append(i)
    i1, i2 = [], []
    for j, (key, tag) in enumerate(
        zip(zip(r2.article_id, r2.company, strict=True), r2.gold, strict=True)
    ):
        found = by_key.get(key)
        if found:
            i1.append(next((i for i in found if r1.gold[i] == tag), found[0]))
            i2.append(j)
    S1, S2 = s1[np.array(i1)], s2[np.array(i2)]
    g1, g2 = r1.gold.to_numpy()[i1], r2.gold.to_numpy()[i2]
    src = r2.source.to_numpy()[i2]
    m = i2
    out = [
        f"# First run -> second run: {len(m)} rows both have ({len(r1)} and {len(r2)} rows in all)\n"
    ]

    def table(title: str, mask: np.ndarray, ga: np.ndarray, gb: np.ndarray) -> None:
        a1, f1, o1 = measure(S1[mask], ga[mask])
        a2, f2, o2 = measure(S2[mask], gb[mask])
        out.append(
            f"\n## {title} (n = {int(mask.sum())})\n\naccuracy {a1:.1%} -> {a2:.1%}, macro F1 {f1:.3f} -> {f2:.3f}\n"
        )
        out.append(
            "| class | rows (second tags) | precision | recall | ROC AUC |\n|---|---|---|---|---|"
        )
        for lab in L12:
            x, y = o1[lab], o2[lab]
            out.append(
                f"| {lab} | {y[3]} | {x[0]:.2f} -> {y[0]:.2f} | {x[1]:.2f} -> {y[1]:.2f} | {x[2]:.2f} -> {y[2]:.2f} |"
            )

    every = np.ones(len(m), bool)
    table("Each run on its own tags", every, g1, g2)
    table("Both runs on the second tags", every, g2, g2)
    table("The older rows only, each run on its own tags", src == "v2", g1, g2)
    table("The older rows only, both on the second tags", src == "v2", g2, g2)
    table("The sheet's rows only (same tags in both runs)", src == "single", g2, g2)
    out.append(
        f"\nOlder rows whose tag changed between the runs: {int((g1 != g2)[src == 'v2'].sum())}.\n"
    )
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "single_comparison.md").write_text(text, encoding="utf-8")
        pred = pd.DataFrame(S2, columns=[f"score_{x}" for x in L12]).round(4)
        keep = r2[["article_id", "company", "source", "gold"]].reset_index(drop=True)
        keep["top"] = np.array(L12)[s2.argmax(1)]
        pd.concat([keep, pred], axis=1).to_csv(a.out / "single_predictions.csv", index=False)


if __name__ == "__main__":
    main()
