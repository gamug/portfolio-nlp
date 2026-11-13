"""Baseline of the category model before any training (owner request 2026-11-13): the pinned
zero-shot model, untouched, scoring one hypothesis per label.

Variants are compared on the 225 train rows only; the held-out rows are scored once, with the
variant picked on the train rows. Rows are built as in category_groups_mock (single bodies and
structured multi bodies split per company; narrative multi bodies are not scored here).
Run: uv run scripts/category_baseline_2026_11_13.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import category_groups_mock_2026_11_13 as mock

from news_category_stage import DEFINITIONS
from news_category_stage import GROUPS as _STEP_GROUPS
from news_nlp.triage import Company, CompanyResolver

MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
REVISION = "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8"  # category_stage.MODEL_REVISIONS
PREMISE_CHARS = 2000  # the size of the summaries step 7 reads (owner 2026-11-12)

DEFS = dict(
    DEFINITIONS
)  # one source of truth: the step (src/news_category_stage.py) and the training scripts read the same definitions
LABELS = list(DEFS)
GROUPS = {g: list(v) for g, v in _STEP_GROUPS.items()}  # option B, owner 2026-11-13
GROUP_OF = {label: g for g, ls in GROUPS.items() for label in ls}
VARIANTS = {
    "topic": ("{text}", "This text is about {d}."),
    "company+topic": ("{company}: {text}", "The news about {company} is about {d}."),
    "company+event": ("{company}: {text}", "The main event for {company} is {d}."),
}


def scores(rows: pd.DataFrame, variant: str) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    prem, hyp = VARIANTS[variant]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = (
        AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION).to(dev).eval()
    )
    ent = model.config.label2id["entailment"]
    out = np.zeros((len(rows), len(LABELS)))
    for i, r in enumerate(rows.itertuples()):
        p = prem.format(text=r.text[:PREMISE_CHARS], company=r.company)
        hs = [hyp.format(d=DEFS[label], company=r.company) for label in LABELS]
        enc = tok(
            [p] * len(hs),
            hs,
            truncation="only_first",
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(dev)
        with torch.no_grad():
            logits = model(**enc).logits
        out[i] = torch.softmax(logits, -1)[:, ent].cpu().numpy()
    return out


def report(name: str, rows: pd.DataFrame, S: np.ndarray, boot: int = 2000) -> dict:
    pred = [LABELS[j] for j in S.argmax(1)]
    ok = np.array([p == t for p, t in zip(pred, rows.label, strict=False)])
    okg = np.array([GROUP_OF[p] == GROUP_OF[t] for p, t in zip(pred, rows.label, strict=False)])
    rng = np.random.default_rng(0)
    ci = lambda v: np.percentile(
        [v[rng.integers(0, len(v), len(v))].mean() for _ in range(boot)], [2.5, 97.5]
    )
    f1 = []
    for label in set(rows.label):
        tp = sum(p == label and t == label for p, t in zip(pred, rows.label, strict=False))
        fp = sum(p == label and t != label for p, t in zip(pred, rows.label, strict=False))
        fn = sum(p != label and t == label for p, t in zip(pred, rows.label, strict=False))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    a, g = ci(ok), ci(okg)
    print(
        f"{name:34s} n={len(rows):3d}  label {ok.mean():.1%} [{a[0]:.1%},{a[1]:.1%}]  "
        f"group {okg.mean():.1%} [{g[0]:.1%},{g[1]:.1%}]  macro-F1 {np.mean(f1):.2f}"
    )
    return {"pred": pred, "ok": ok}


def main() -> None:
    held_out_record = ROOT / "docs" / "category_results_2026_11_13" / "baseline_heldout.csv"
    if held_out_record.exists() and "--again" not in sys.argv:
        sys.exit(
            f"{held_out_record} exists: the held-out set was scored once and is not scored again without --again"
        )
    uni = pd.read_csv(ROOT / "gold/universe.csv").fillna("")
    resolver = CompanyResolver(
        [
            Company(t, c, s, "")
            for t, c, s in zip(uni.ticker, uni.company, uni.gics_sector, strict=False)
        ]
    )
    train, _ = mock.build_rows(
        pd.read_csv(ROOT / "gold/category_gold_v2.csv"),
        pd.read_csv(ROOT / "gold/texts/category_gold_v2_texts.csv"),
        resolver,
    )
    train = train[train.label.isin(LABELS)].reset_index(drop=True)
    out = ROOT / "data" / "category_groups"
    best, best_acc, cache = None, -1.0, {}
    print("== variants, train rows (225) ==")
    for v in VARIANTS:
        S = scores(train, v)
        cache[v] = S
        r = report(v, train, S)
        if r["ok"].mean() > best_acc:
            best, best_acc = v, r["ok"].mean()
    print("picked on the train rows:", best)
    S = cache[best]
    pred = [LABELS[j] for j in S.argmax(1)]
    print("\nper label (train rows): recall / predicted")
    for label in LABELS:
        m = (train.label == label).values
        print(
            f"  {label:30s} rows {m.sum():3d}  recall {np.mean([p == label for p, k in zip(pred, m, strict=False) if k]):.0%}  predicted {pred.count(label)}"
        )
    pd.DataFrame(
        {
            "article_id": train.article_id,
            "company": train.company,
            "label": train.label,
            "pred": pred,
        }
    ).to_csv(out / "baseline_train.csv", index=False)
    # held-out: scored once, with the variant picked above
    test, dropped = mock.build_rows(
        pd.read_csv(ROOT / "gold/category_gold_v2_test.csv"),
        pd.read_csv(ROOT / "gold/texts/category_gold_v2_test_texts.csv"),
        resolver,
    )
    test = test[test.label.isin(LABELS)].reset_index(drop=True)
    print(
        "\n== held-out 200, scored once ==  rows:",
        len(test),
        test.kind.value_counts().to_dict(),
        "left out:",
        dict(dropped),
    )
    St = scores(test, best)
    report(f"held-out ({best})", test, St)
    for k in test.kind.unique():
        m = (test.kind == k).values
        report(f"  held-out {k}", test[m], St[m])
    pt = [LABELS[j] for j in St.argmax(1)]
    pd.DataFrame(
        {
            "article_id": test.article_id,
            "company": test.company,
            "kind": test.kind,
            "label": test.label,
            "pred": pt,
        }
    ).to_csv(out / "baseline_heldout.csv", index=False)
    print("held-out label counts:", test.label.value_counts().to_dict())


if __name__ == "__main__":
    main()

# ruff: noqa: E402, E731, PLC0415  -- one-shot research script
