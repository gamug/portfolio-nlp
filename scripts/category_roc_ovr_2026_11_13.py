"""ROC AUC one-vs-rest per class (D37: Precision, Recall and ROC OVR are the recorded metrics), from scores a run saved.

    uv run scripts/category_roc_ovr_2026_11_13.py            # the held-out scoring of the mixed recipe (docs/category_results_2026_11_13/)
    uv run scripts/category_roc_ovr_2026_11_13.py --cv mixed # the out-of-fold scores of a CV run (oof_<mode>_scores.npz)

The AUC of a class is the probability that a row of the class scores higher than a row of another class (ties count half); a
class with no row, or with every row, has none (undefined, D34 (a)).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from news_category_stage import GROUP_OF, GROUPS, LABELS  # noqa: E402

OUT = ROOT / "data" / "category_groups"  # local runs (git-ignored)
RECORD = (
    ROOT / "docs" / "category_results_2026_11_13"
)  # the tracked record of the runs reported in docs/category_step.md


def auc(positive: np.ndarray, score: np.ndarray) -> float | None:
    """ROC AUC of one class against the rest, by ranks (Mann-Whitney); ``None`` when undefined."""
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = pd.Series(score).rank(method="average").to_numpy()
    return float((order[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def table(truth: pd.Series, scores: np.ndarray, classes: list[str]) -> pd.DataFrame:
    rows = [
        (c, int((truth == c).sum()), auc((truth == c).to_numpy(), scores[:, j]))
        for j, c in enumerate(classes)
    ]
    return pd.DataFrame(rows, columns=["class", "rows", "roc_auc_ovr"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cv", choices=["flat", "mixed"], default=None)
    args = ap.parse_args()
    if args.cv:
        data = np.load(OUT / f"oof_{args.cv}_scores.npz")
        labels_s, groups_s = data["labels"], data["groups"]
        truth = pd.read_csv(OUT / f"oof_{args.cv}.csv").label
    else:
        record = pd.read_csv(RECORD / "heldout_mixed_scores.csv")
        labels_s = record[[f"label_{label}" for label in LABELS]].to_numpy()
        groups_s = record[[f"group_{group}" for group in GROUPS]].to_numpy()
        truth = record.label
    pd.options.display.float_format = "{:.3f}".format
    print(f"labels (n={len(truth)}):")
    labels = table(truth, labels_s, list(LABELS))
    print(
        labels.to_string(index=False), f"\nmacro (defined classes): {labels.roc_auc_ovr.mean():.3f}"
    )
    print("\ngroups (the group of the owner's label, the group hypotheses of the model):")
    groups = table(truth.map(GROUP_OF), groups_s, list(GROUPS))
    print(
        groups.to_string(index=False), f"\nmacro (defined classes): {groups.roc_auc_ovr.mean():.3f}"
    )


if __name__ == "__main__":
    main()
