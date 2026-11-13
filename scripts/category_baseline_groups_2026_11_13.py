"""Zero-shot baseline at group level and two-step (group, then label inside it). Same model, wording
fixed to 'company+topic' (picked on the train rows in category_baseline_2026_11_13). Option B groups."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import category_baseline_2026_11_13 as b
import category_groups_mock_2026_11_13 as mock

from news_nlp.triage import Company, CompanyResolver

GROUP_DEF = {g: "; or ".join(b.DEFS[l] for l in ls) for g, ls in b.GROUPS.items()}
GROUP_DEF["other"] = b.DEFS["other"]
GN = list(b.GROUPS)


def group_scores(rows):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(b.MODEL, revision=b.REVISION)
    m = (
        AutoModelForSequenceClassification.from_pretrained(b.MODEL, revision=b.REVISION)
        .to(dev)
        .eval()
    )
    ent = m.config.label2id["entailment"]
    out = np.zeros((len(rows), len(GN)))
    for i, r in enumerate(rows.itertuples()):
        p = f"{r.company}: {r.text[: b.PREMISE_CHARS]}"
        hs = [f"The news about {r.company} is about {GROUP_DEF[g]}." for g in GN]
        enc = tok(
            [p] * len(hs),
            hs,
            truncation="only_first",
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(dev)
        with torch.no_grad():
            out[i] = torch.softmax(m(**enc).logits, -1)[:, ent].cpu().numpy()
    return out


def evaluate(name, rows, SL):
    SG = group_scores(rows)
    true_g = np.array([b.GROUP_OF[t] for t in rows.label])
    pg = np.array([GN[j] for j in SG.argmax(1)])
    two = []
    for i, g in enumerate(pg):
        idx = [b.LABELS.index(l) for l in b.GROUPS[g]]
        two.append(b.LABELS[idx[int(SL[i, idx].argmax())]])
    flat = [b.LABELS[j] for j in SL.argmax(1)]
    f = lambda a, c: np.mean([x == y for x, y in zip(a, c, strict=False)])
    print(
        f"{name:10s} n={len(rows)}  group (own hypotheses) {f(pg, true_g):.1%}  | group via flat label {f([b.GROUP_OF[x] for x in flat], true_g):.1%}"
        f"  | label: flat {f(flat, rows.label):.1%}, two-step {f(two, rows.label):.1%}"
    )
    per = {
        g: np.mean([x == y for x, y, t in zip(pg, true_g, true_g, strict=False) if t == g])
        for g in GN
    }
    print(
        "   group recall:",
        {g: f"{v:.0%}" for g, v in per.items()},
        " rows:",
        {g: int((true_g == g).sum()) for g in GN},
    )


def main():
    uni = pd.read_csv(ROOT / "gold/universe.csv").fillna("")
    res = CompanyResolver(
        [
            Company(t, c, s, "")
            for t, c, s in zip(uni.ticker, uni.company, uni.gics_sector, strict=False)
        ]
    )
    for name, g, t in [
        ("train", "category_gold_v2", "category_gold_v2_texts"),
        ("held-out", "category_gold_v2_test", "category_gold_v2_test_texts"),
    ]:
        rows, _ = mock.build_rows(
            pd.read_csv(ROOT / f"gold/{g}.csv"), pd.read_csv(ROOT / f"gold/texts/{t}.csv"), res
        )
        rows = rows[rows.label.isin(b.LABELS)].reset_index(drop=True)
        evaluate(name, rows, b.scores(rows, "company+topic"))


if __name__ == "__main__":
    main()

# ruff: noqa: E402, E731, E741, PLC0415  -- one-shot research script
