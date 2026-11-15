# ruff: noqa: E402  -- one-shot research script, a reproducible draw
"""The entailment fine-tune on ONE tag per (article, company), 12 labels, cross-validated (T-187, owner 2026-11-15).

    uv run scripts/category_single_cv_2026_11_15.py cv [--smoke]

Rows: ``gold/category_single_gold.csv`` (the owner's sheet after the review: one tag per row, ``competitive_actions`` and
``peer_comparison`` read as ``other``) and the 225 single-label train rows of ``gold/category_gold_v2.csv`` with the same merge. The
held-out 200 is never read. The recipe is the one of ``category_multilabel_cv_2026_11_14.py`` (pairs: the tag a positive, 4 negatives
from the other labels; 3 epochs, lr 2e-5), but the model is trained on the 12 labels, so the merge is a real training choice. Folds
are grouped by article. Saves ``data/category_single/oof_single.npz`` (scores of the 12 labels, fold) and ``rows.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import category_multilabel_cv_2026_11_14 as cvs

import category_multilabel as ml
from news_category_stage import LABELS as L14

L12 = ml.merge_labels(tuple(L14), ml.FOLD_INTO_OTHER)
cvs.LABELS = L12  # train() and score() read the module's label list
OUT = ROOT / "data" / "category_single"


def rows_old() -> pd.DataFrame:
    """The 225 older single-label rows with the tags they had before the 2026-11-15 review (read through the merge)."""
    cvs.LABELS = tuple(L14)  # the old rows are read with the 14 labels
    old = cvs.rows()
    cvs.LABELS = L12
    old = old[old.source == "v2"].reset_index(drop=True)
    old["gold"] = [[ml.FOLD_INTO_OTHER.get(g[0], g[0])] for g in old.gold]
    return old


def rows() -> pd.DataFrame:
    old = rows_old()
    rev = pd.read_csv(
        ROOT / "gold/category_v2_single.csv", dtype=str, keep_default_na=False
    )  # the review of the older rows
    assert len(rev) == len(old) and all(rev.label_old == [g[0] for g in old.gold]), (
        "the review does not match the older rows"
    )
    old = old.assign(gold=[[x] for x in rev.label])[rev.label.to_numpy() != ""].reset_index(
        drop=True
    )
    tags = pd.read_csv(ROOT / "gold/category_single_gold.csv", dtype=str)
    texts = pd.read_csv(ROOT / "gold/texts/category_single_gold_texts.csv", dtype=str)
    m = tags.merge(texts[["review_order", "text"]], on="review_order")
    new = pd.DataFrame(
        {
            "article_id": m.article_id,
            "company": m.company,
            "text": m.text,
            "gold": [[x] for x in m.label],
            "source": "single",
        }
    )
    return pd.concat([old, new], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["cv"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--neg", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    df = rows()
    assert all(g[0] in L12 for g in df.gold), "a label outside the 12"
    if args.smoke:
        df = df.sample(40, random_state=0).reset_index(drop=True)
        args.epochs, args.folds = 1, 2
    fold = np.array(ml.folds_of(list(df.article_id), args.folds))
    S = np.zeros((len(df), len(L12)))
    print(
        f"[single] {len(df)} rows, {df.article_id.nunique()} articles, {len(L12)} labels",
        flush=True,
    )
    for k in range(args.folds):
        tr, va = df[fold != k].reset_index(drop=True), df[fold == k].reset_index(drop=True)
        print(f"[single] fold {k}: train {len(tr)}, validate {len(va)}", flush=True)
        tok, model, ent = cvs.train(tr, True, args, args.seed + k)
        S[fold == k] = cvs.score(tok, model, ent, va, args.max_len)
        del model
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "single_smoke" if args.smoke else "single"
    np.savez(OUT / f"oof_{tag}.npz", scores=S, fold=fold)
    df.drop(columns="text").assign(gold=df.gold.map(",".join)).to_csv(
        OUT / f"rows_{tag}.csv", index=False
    )
    print(f"saved {OUT / f'oof_{tag}.npz'}", flush=True)


if __name__ == "__main__":
    main()
