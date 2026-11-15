# ruff: noqa: E402  -- one-shot research script
"""One-versus-rest precision and recall per category from the out-of-fold scores (T-187, owner 2026-11-14).

    uv run scripts/category_ovr_report_2026_11_14.py --mode multi|head|base|deb [--no-merge] [--out docs/category_results_2026_11_14]
        [--threshold 0.4 | --thresholds cuts.json | (default: a cut per label fitted on the other folds)] [--max-labels 3]

Three tables, all with 95% intervals that resample ARTICLES (a body gives up to three rows): the output set with every row counted once,
the same with every article counted once (a row weighs 1 over the rows of its article), and the pairs (one row per gold tag, the top label
as the output). For the probability models (``head``, ``deb``) the labels of a row are the ones over their cut: ``--threshold`` is one cut for
every label, ``--thresholds`` a JSON file ``{"label": cut}``, and by default each label's cut is fitted on the other folds (and written to
``<out>/thresholds_<mode>.json``, fitted on all rows, to be reused). By default ``competitive_actions`` and ``peer_comparison`` are read as ``other`` (owner, 2026-11-14).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import category_multilabel as ml

LABELS = ml.LABELS14

OUT = ROOT / "data" / "category_multilabel"


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def cell(v: float | None, ci: tuple[float, float] | None) -> str:
    return "-" if v is None else f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{v:.2f}"


def table(
    gold: list[list[str]],
    pred: list[list[str]],
    labels: tuple[str, ...],
    arts: list[str],
    weights: list[float] | None,
    boot: int,
    scores: np.ndarray | None = None,
) -> pd.DataFrame:
    pl = ml.per_label(gold, pred, labels, weights)
    ci = ml.per_label_ci(gold, pred, labels, arts, boot=boot, weights=weights)
    rows = []
    for j, label in enumerate(labels):
        row = {
            "label": label,
            "rows": round(pl[label]["support"] or 0.0, 1),
            "predicted": round(pl[label]["predicted"] or 0.0, 1),
            "precision [95%]": cell(pl[label]["precision"], ci[label]["precision"]),
            "recall [95%]": cell(pl[label]["recall"], ci[label]["recall"]),
        }
        if scores is not None:
            auc = ml.roc_auc(scores[:, j], [int(label in g) for g in gold])
            row["ROC AUC"] = fmt(auc)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["base", "multi", "head", "deb"], default="multi")
    ap.add_argument("--threshold", type=float, default=None, help="one cut for every label")
    ap.add_argument("--thresholds", type=Path, default=None, help='JSON {"label": cut}')
    ap.add_argument("--max-labels", type=int, default=3)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    df = pd.read_csv(OUT / "rows.csv", dtype=str)
    gold = [x.split(",") for x in df.gold]
    arts = list(df.article_id)
    z = np.load(OUT / f"oof_{a.mode}.npz")
    S, fold = z["scores"], z["fold"]
    labels: tuple[str, ...] = tuple(LABELS)
    if S.shape[1] == len(ml.merge_labels(LABELS, ml.FOLD_INTO_OTHER)):
        gold = ml.merge_gold(gold, ml.FOLD_INTO_OTHER)  # a model trained on the 12 labels
        labels = ml.merge_labels(LABELS, ml.FOLD_INTO_OTHER)
    elif not a.no_merge:
        gold = ml.merge_gold(gold, ml.FOLD_INTO_OTHER)
        S = ml.merge_scores(S, LABELS, ml.FOLD_INTO_OTHER)
        labels = ml.merge_labels(LABELS, ml.FOLD_INTO_OTHER)
    top = [labels[int(j)] for j in S.argmax(1)]
    fitted: dict[str, float] | None = None
    if a.mode in ("head", "deb"):
        if a.thresholds:
            cuts = json.loads(a.thresholds.read_text(encoding="utf-8"))
            sets = [ml.set_by_thresholds(s, labels, cuts, a.max_labels) for s in S]
        elif a.threshold is not None:
            cuts = ml.global_thresholds(labels, a.threshold)
            sets = [ml.set_by_thresholds(s, labels, cuts, a.max_labels) for s in S]
        else:
            sets = ml.cross_fitted_thresholds(S, gold, labels, fold, a.max_labels)
            fitted = ml.fit_thresholds(S, gold, labels)
    else:
        sets = ml.cross_fitted_sets(S, gold, labels, fold)[1]
    pairs = ml.explode_pairs(gold, top)
    pg = [[t] for _, t, _ in pairs]
    pp = [[o] for *_, o in pairs]
    parts = [arts[i] for i, _, _ in pairs]
    w = ml.article_weights(arts)
    text = f"# One versus rest, mode `{a.mode}`, {len(labels)} labels, {len(gold)} rows from {len(set(arts))} articles\n"
    for title, t in (
        ("Output set, every row counts once", table(gold, sets, labels, arts, None, a.boot, S)),
        ("Output set, every article counts once", table(gold, sets, labels, arts, w, a.boot)),
        (
            "Pairs: one row per gold tag, the top label as the output (a record with two tags is two rows with one output)",
            table(pg, pp, labels, parts, None, a.boot),
        ),
    ):
        text += f"\n## {title}\n\n{ml.markdown_table(list(t.columns), t.values.tolist())}\n"
    print(text)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        if fitted is not None:
            (a.out / f"thresholds_{a.mode}.json").write_text(
                json.dumps(fitted, indent=1), encoding="utf-8"
            )
        (a.out / f"ovr_{a.mode}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
