# ruff: noqa: E402, PLR0915  -- one-shot research script
"""The held-out 200 scored ONCE with the selected category model (T-201, owner's word 2026-11-15).

    uv run scripts/category_heldout_2026_11_15.py

Refits the selected recipe (``config/category_model.json``: the entailment fine-tune, one tag per row, 3 epochs, lr 2e-5, 4 negatives)
on ALL 1,216 training rows (no fold held out, seed 0), keeps the weights in the git-ignored ``models/category_single_all/`` and scores the
held-out rows of ``gold/category_gold_v2_test.csv`` (built as ``category_baseline_2026_11_13.py`` built them: one row per company with a
label; narrative bodies are not rows). Nothing about the model is chosen on these rows. The script **refuses to run if its output exists**
and has no ``--again``: a second scoring is a decision of the owner, taken by deleting the output by hand.

The held-out tags are the owner's of 2026-11-12, **before** the one-tag review: an analyst rating may be tagged ``investor_activity`` there.
They are read as they are, with ``competitive_actions`` and ``peer_comparison`` read as ``other`` (the same merge as the training rows).
Writes ``docs/category_results_2026_11_15/heldout_selected.csv`` (no text) and ``heldout_selected.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import category_groups_mock_2026_11_13 as mock
import category_multilabel_cv_2026_11_14 as cvs
import category_single_cv_2026_11_15 as scv

import category_multilabel as ml
from news_category_stage import GROUP_OF
from news_nlp.triage import Company, CompanyResolver

L14 = ml.LABELS14
L12 = scv.L12
OUT = ROOT / "docs" / "category_results_2026_11_15"
WEIGHTS = ROOT / "models" / "category_single_all"
BAR = 0.80  # D24: agreement with the owner's labels on the held-out single rows


def ci_mean(ok: np.ndarray, articles: list[str]) -> tuple[float, float]:
    return ml.bootstrap_ci(ok.astype(float), articles, boot=2000)


def cell(v: float | None, ci: tuple[float, float] | None) -> str:
    return "-" if v is None else f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{v:.2f}"


def main() -> None:
    record = OUT / "heldout_selected.csv"
    if record.exists():
        sys.exit(
            f"{record} exists: the held-out set was scored once with this model and is not scored again"
        )
    uni = pd.read_csv(ROOT / "gold/universe.csv").fillna("")
    resolver = CompanyResolver(
        [
            Company(t, c, s, "")
            for t, c, s in zip(uni.ticker, uni.company, uni.gics_sector, strict=False)
        ]
    )
    test, dropped = mock.build_rows(
        pd.read_csv(ROOT / "gold/category_gold_v2_test.csv"),
        pd.read_csv(ROOT / "gold/texts/category_gold_v2_test_texts.csv"),
        resolver,
    )
    test = test[test.label.isin(L14)].reset_index(drop=True)
    test["gold"] = [ml.FOLD_INTO_OTHER.get(x, x) for x in test.label]
    print(
        f"[heldout] {len(test)} rows {test.kind.value_counts().to_dict()}, left out {dict(dropped)}",
        flush=True,
    )

    # --- the refit on all training rows: the recipe of the pin, no fold held out
    args = argparse.Namespace(
        folds=5, epochs=3, lr=2e-5, neg=4, bs=4, accum=4, max_len=512, seed=0, smoke=False
    )
    df = scv.rows()
    assert all(g[0] in L12 for g in df.gold)
    print(f"[heldout] refit on {len(df)} rows, {df.article_id.nunique()} articles", flush=True)
    tok, model, ent = cvs.train(df, True, args, args.seed)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(WEIGHTS)
    tok.save_pretrained(WEIGHTS)

    # --- scored ONCE
    S = cvs.score(tok, model, ent, test, args.max_len)
    pred = [L12[j] for j in S.argmax(1)]
    out = pd.DataFrame(
        {
            "article_id": test.article_id,
            "company": test.company,
            "kind": test.kind,
            "label_owner": test.label,
            "gold": test.gold,
            "pred": pred,
        }
    )
    for j, lab in enumerate(L12):
        out[f"score_{lab}"] = S[:, j].round(4)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(record, index=False)
    print(f"[heldout] saved {record}", flush=True)

    # --- the figures
    articles = [str(a) for a in test.article_id]
    ok = np.array([p == g for p, g in zip(pred, test.gold, strict=True)])
    okg = np.array([GROUP_OF[p] == GROUP_OF[g] for p, g in zip(pred, test.gold, strict=True)])
    lines = [
        f"# The held-out 200 scored once with the selected model ({len(test)} rows, refit on {len(df)} rows)\n"
    ]
    a, g = ci_mean(ok, articles), ci_mean(okg, articles)
    lines.append(
        f"- **label agreement {ok.mean():.1%} [{a[0]:.1%}, {a[1]:.1%}]**, group agreement {okg.mean():.1%} [{g[0]:.1%}, {g[1]:.1%}]; bar of D24: {BAR:.0%} on the held-out single rows: **{'passed' if ok[(test.kind == 'single').to_numpy()].mean() >= BAR else 'NOT passed'}**"
    )
    for kind in sorted(test.kind.unique()):
        m = (test.kind == kind).to_numpy()
        lo, hi = ci_mean(ok[m], [x for x, k in zip(articles, m, strict=True) if k])
        lines.append(
            f"- rows of kind `{kind}`: {int(m.sum())}, label agreement {ok[m].mean():.1%} [{lo:.1%}, {hi:.1%}]"
        )
    gold = [[x] for x in test.gold]
    top = [[p] for p in pred]
    per = ml.per_label(gold, top, L12)
    ci = ml.per_label_ci(gold, top, L12, articles, boot=1000)
    with_rows = [lab for lab in L12 if per[lab]["support"]]
    f1_all, f1_rows = [], []
    for lab in L12:
        p, r = per[lab]["precision"] or 0.0, per[lab]["recall"] or 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        f1_all.append(f)
        if lab in with_rows:
            f1_rows.append(f)
    lines.append(
        f"- macro F1 over the {len(with_rows)} labels that have held-out rows {np.mean(f1_rows):.3f}; over all 12 (a label with no row or never predicted counts 0) {np.mean(f1_all):.3f}"
    )
    lines.append(
        "\n## Per class, one versus rest (the top label of each row), 95% intervals by article\n"
    )
    lines.append(
        "| class | rows | predicted | precision [95%] | recall [95%] | ROC AUC |\n|---|---|---|---|---|---|"
    )
    for j, lab in enumerate(L12):
        auc = ml.roc_auc(S[:, j], [x == lab for x in test.gold]) if per[lab]["support"] else None
        lines.append(
            f"| {lab} | {int(per[lab]['support'] or 0)} | {int(per[lab]['predicted'] or 0)} | "
            f"{cell(per[lab]['precision'], ci[lab]['precision'])} | {cell(per[lab]['recall'], ci[lab]['recall'])} | {'-' if auc is None else f'{auc:.2f}'} |"
        )
    # --- the same rows, the earlier models (scored once each on 2026-11-13; the merge is read on their labels)
    single = test.kind == "single"
    base = pd.read_csv(ROOT / "docs/category_results_2026_11_13/baseline_heldout.csv", dtype=str)
    key = dict(zip(zip(base.article_id, base.company, strict=True), base.pred, strict=True))
    bp = [
        ml.FOLD_INTO_OTHER.get(key.get((str(a), c), ""), key.get((str(a), c), ""))
        for a, c in zip(test.article_id, test.company, strict=True)
    ]
    okb = np.array([b == g for b, g in zip(bp, test.gold, strict=True)])
    lines.append(
        "\n## The same held-out rows, the earlier models (12 labels, `competitive_actions` and `peer_comparison` read as `other`)\n"
    )
    lines.append("| model | rows | label agreement |\n|---|---|---|")
    lines.append(
        f"| untouched zero-shot (the step today) | {int(single.sum())} | {okb[single.to_numpy()].mean():.1%} |"
    )
    lines.append(
        f"| **selected recipe, refit on 1,216 rows** | {int(single.sum())} | **{ok[single.to_numpy()].mean():.1%}** |"
    )
    # --- where it fails: the biggest confusions, and the tags of the convention that changed
    conf = pd.crosstab(test.gold, pd.Series(pred, name="pred"))
    pairs = [
        (g_, p_, int(conf.loc[g_, p_]))
        for g_ in conf.index
        for p_ in conf.columns
        if g_ != p_ and conf.loc[g_, p_] > 0
    ]
    pairs.sort(key=lambda x: -x[2])
    lines.append("\n## The largest confusions (gold -> predicted, rows)\n")
    lines += [f"- `{g_}` -> `{p_}`: {n}" for g_, p_, n in pairs[:8]]
    ia = [(g_, p_) for g_, p_ in zip(test.gold, pred, strict=True) if g_ == "investor_activity"]
    lines.append(
        f"\nHeld-out tags are of the earlier convention: {len(ia)} rows are tagged `investor_activity`, {sum(p_ == 'market_analyst_sentiment' for _, p_ in ia)} of them predicted `market_analyst_sentiment` (the review of 2026-11-15 moved analyst ratings out of `investor_activity` in the training rows; the held-out tags were not reviewed)."
    )
    text = "\n".join(lines) + "\n"
    (OUT / "heldout_selected.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
