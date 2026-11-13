"""Held-out scoring, once (owner decision 2026-11-13: recipe = mixed; narrative bodies scored later).
Trains the mixed recipe on all 225 train rows, scores the 135 scorable held-out rows, writes the scores.
Primary reading fixed before the run: two-step (best mixed reading in the CV); the others are reported too.
Run: uv run scripts/category_heldout_2026_11_13.py
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
import category_baseline_2026_11_13 as b
import category_groups_mock_2026_11_13 as mock
import category_train_cv_2026_11_13 as t

from news_nlp.triage import Company, CompanyResolver


def main():
    args = argparse.Namespace(epochs=3, lr=2e-5, neg=4, bs=4, accum=4, max_len=512)
    uni = pd.read_csv(ROOT / "gold/universe.csv").fillna("")
    res = CompanyResolver(
        [
            Company(x, c, s, "")
            for x, c, s in zip(uni.ticker, uni.company, uni.gics_sector, strict=False)
        ]
    )

    def load(g, tx):
        r, d = mock.build_rows(pd.read_csv(ROOT / g), pd.read_csv(ROOT / tx), res)
        return r[r.label.isin(b.LABELS)].reset_index(drop=True), d

    train, _ = load("gold/category_gold_v2.csv", "gold/texts/category_gold_v2_texts.csv")
    test, dropped = load(
        "gold/category_gold_v2_test.csv", "gold/texts/category_gold_v2_test_texts.csv"
    )
    print(
        f"train {len(train)} rows, held-out {len(test)} rows, left out {dict(dropped)}", flush=True
    )
    tok, model, ent = t.train(train, "mixed", args, 0)
    SL, SG = t.score(tok, model, ent, test, args.max_len)
    GN = t.GN
    flat = [b.LABELS[j] for j in SL.argmax(1)]
    gi = [GN.index(b.GROUP_OF[x]) for x in b.LABELS]
    prod = [b.LABELS[int((SL[i] * SG[i][gi]).argmax())] for i in range(len(test))]
    pg = [GN[j] for j in SG.argmax(1)]
    two = []
    for i, g in enumerate(pg):
        idx = [b.LABELS.index(x) for x in b.GROUPS[g]]
        two.append(b.LABELS[idx[int(SL[i, idx].argmax())]])
    out = ROOT / "data/category_groups"
    pd.DataFrame(
        {
            "article_id": test.article_id,
            "company": test.company,
            "label": test.label,
            "flat": flat,
            "prod": prod,
            "two_step": two,
            "group_pred": pg,
        }
    ).to_csv(out / "heldout_mixed.csv", index=False)
    np.save(out / "heldout_mixed_SL.npy", SL)
    np.save(out / "heldout_mixed_SG.npy", SG)
    print("done")


if __name__ == "__main__":
    main()

# ruff: noqa: E402  -- one-shot research script
