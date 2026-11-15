# ruff: noqa: E402, PLR0915, C901, PLC0206  -- one-shot research script
"""One-versus-rest precision, recall and ROC AUC per class of the single-tag entailment fine-tune (T-187, owner 2026-11-15).

    uv run scripts/category_single_report_2026_11_15.py [--boot 1000] [--out docs/category_results_2026_11_15]

Out-of-fold scores of ``data/category_single/oof_single.npz`` (12 labels) against the single tags. The output of a row is its top label.
For the comparison on the same rows and the same tags, the untouched model and the earlier entailment fine-tune (trained on the old tags,
14 labels, scores read through the merge) are scored too. Intervals resample articles.
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
OLD = ml.scores_dir(ROOT, "category_multilabel")
NEW = ml.scores_dir(ROOT, "category_single")


def cell(v: float | None, ci: tuple[float, float] | None) -> str:
    return "-" if v is None else f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = pd.read_csv(
        NEW / ("rows_single.csv" if (NEW / "rows_single.csv").exists() else "rows_single_prov.csv"),
        dtype=str,
    )
    key = list(zip(rows.article_id, rows.company, strict=True))
    gold = [g for g in rows.gold]
    arts = list(rows.article_id)
    models: dict[str, np.ndarray] = {}
    if (NEW / "oof_single.npz").exists():  # else a provisional report of the earlier models only
        models["entailment, new tags"] = np.load(NEW / "oof_single.npz")["scores"]
    old_rows = pd.read_csv(OLD / "rows.csv", dtype=str)
    pos = {}
    for i, k in enumerate(zip(old_rows.article_id, old_rows.company, strict=True)):
        pos.setdefault(k, i)
    idx = [pos.get(k, -1) for k in key]
    have = np.array([i >= 0 for i in idx])
    for name, f in (("untouched", "oof_base"), ("entailment, old tags", "oof_multi")):
        S = ml.merge_scores(np.load(OLD / f"{f}.npz")["scores"], tuple(L14), ml.FOLD_INTO_OTHER)
        M = np.full((len(rows), len(L12)), np.nan)
        M[have] = S[[i for i in idx if i >= 0]]
        models[name] = M
    out = [
        f"# One versus rest per class, 12 labels, {len(rows)} rows from {len(set(arts))} articles (out of fold)\n"
    ]
    res = {}
    for name, S in models.items():
        ok = ~np.isnan(S).any(axis=1)
        g = [[gold[i]] for i in np.flatnonzero(ok)]
        top = [[L12[j]] for j in S[ok].argmax(1)]
        ar = [arts[i] for i in np.flatnonzero(ok)]
        pl = ml.per_label(g, top, L12)
        ci = ml.per_label_ci(g, top, L12, ar, boot=a.boot)
        auc = {lab: ml.roc_auc(S[ok][:, j], [x[0] == lab for x in g]) for j, lab in enumerate(L12)}
        acc = float(np.mean([x[0] == y[0] for x, y in zip(g, top, strict=True)]))
        f1 = []
        for lab in L12:
            p, r = pl[lab]["precision"] or 0.0, pl[lab]["recall"] or 0.0
            f1.append(2 * p * r / (p + r) if p + r else 0.0)
        res[name] = (pl, ci, auc, acc, float(np.mean(f1)), int(ok.sum()))
    for name, (_, _, _, acc, mf1, n) in res.items():
        out.append(f"- **{name}**: {n} rows, accuracy {acc:.1%}, macro F1 {mf1:.3f}")
    first = next(iter(res))
    out.append(f"\n## {first}: precision, recall and ROC AUC per class\n")
    out.append(
        "| class | rows | predicted | precision [95%] | recall [95%] | ROC AUC |\n|---|---|---|---|---|---|"
    )
    pl, ci, auc, *_ = res[first]
    for lab in L12:
        out.append(
            f"| {lab} | {int(pl[lab]['support'] or 0)} | {int(pl[lab]['predicted'] or 0)} | "
            f"{cell(pl[lab]['precision'], ci[lab]['precision'])} | {cell(pl[lab]['recall'], ci[lab]['recall'])} | "
            f"{'-' if auc[lab] is None else f'{auc[lab]:.2f}'} |"
        )
    out.append(
        "\n## The three models on the same rows and the same (new) tags: precision / recall / ROC AUC\n"
    )
    out.append("| class | rows | " + " | ".join(res) + " |\n|---|---|" + "---|" * len(res))
    for lab in L12:
        cells = []
        for name in res:
            p, _, au, *_ = res[name]
            pr = "-" if p[lab]["precision"] is None else f"{p[lab]['precision']:.2f}"
            rc = "-" if p[lab]["recall"] is None else f"{p[lab]['recall']:.2f}"
            cells.append(f"{pr} / {rc} / {'-' if au[lab] is None else f'{au[lab]:.2f}'}")
        out.append(
            f"| {lab} | {int(res[first][0][lab]['support'] or 0)} | " + " | ".join(cells) + " |"
        )
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "ovr_single.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
