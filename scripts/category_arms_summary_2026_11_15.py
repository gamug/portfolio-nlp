# ruff: noqa: E402, PLR2004  -- one-shot research script
"""The four arms of the multi-label experiments on the OLD tags, one line each (T-187, owner 2026-11-15).

    uv run scripts/category_arms_summary_2026_11_15.py [--out docs/category_results_2026_11_15]

Reads the out-of-fold scores of ``data/category_multilabel/`` (1,222 rows, 14 labels read as 12 through the merge; the plain
DeBERTa head already has 12). A row is read three ways: the top label against the FIRST gold label, the top label against ANY gold
label (the multi-tag rows count as right when the top label is one of their tags), and the same top-vs-first reading on the rows with a
single tag only. The old tags are the ones before the review of 2026-11-15; nothing here uses the new tags.
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

L12 = ml.merge_labels(tuple(L14), ml.FOLD_INTO_OTHER)
DATA = ml.scores_dir(ROOT, "category_multilabel")
ARMS = {
    "untouched zero-shot": "base",
    "entailment fine-tune, old recipe (first label the only positive)": "first",
    "entailment fine-tune, every tag a positive": "multi",
    "head on the zero-shot encoder (12 sigmoid outputs)": "head",
    "plain DeBERTa-v3-base, 12 sigmoid outputs": "deb",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = pd.read_csv(DATA / "rows.csv", dtype=str)
    gold = [sorted({ml.FOLD_INTO_OTHER.get(x, x) for x in g.split(",")}) for g in rows.gold]
    first = [ml.FOLD_INTO_OTHER.get(g.split(",")[0], g.split(",")[0]) for g in rows.gold]
    single = np.array([len(g) == 1 for g in gold])
    out = [
        f"# The arms on the old tags: {len(rows)} rows, {rows.article_id.nunique()} articles, {int(single.sum())} rows with one tag\n",
        "| arm | file | top = first tag | top is any tag | single-tag rows: accuracy | macro F1 |\n|---|---|---|---|---|---|",
    ]
    for name, key in ARMS.items():
        f = DATA / f"oof_{key}.npz"
        if not f.exists():
            continue  # the first-label arm was stopped and has no file
        S = np.load(f)["scores"]
        if S.shape[1] == 14:
            S = ml.merge_scores(S, tuple(L14), ml.FOLD_INTO_OTHER)
        top = [L12[j] for j in S.argmax(1)]
        acc_first = float(np.mean([t == g for t, g in zip(top, first, strict=True)]))
        acc_any = float(np.mean([t in g for t, g in zip(top, gold, strict=True)]))
        acc_single = float(
            np.mean([t == g for t, g, s in zip(top, first, single, strict=True) if s])
        )
        pl = ml.per_label([[g] for g in first], [[t] for t in top], L12)
        f1 = []
        for lab in L12:
            p, r = pl[lab]["precision"] or 0.0, pl[lab]["recall"] or 0.0
            f1.append(2 * p * r / (p + r) if p + r else 0.0)
        out.append(
            f"| {name} | `oof_{key}.npz` | {acc_first:.1%} | {acc_any:.1%} | {acc_single:.1%} | {np.mean(f1):.3f} |"
        )
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "arms_old_tags.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
