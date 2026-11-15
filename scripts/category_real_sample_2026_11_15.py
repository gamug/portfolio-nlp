# ruff: noqa: E402, PLR2004, S608  -- one-shot research script (the SQL has no user input)
"""The category model on real data: a sample of what the step reads (T-187, owner 2026-11-15).

    uv run scripts/category_real_sample_2026_11_15.py draw  [--db PATH] [--n 150] [--seed 20261115]
    uv run scripts/category_real_sample_2026_11_15.py score [--db PATH]        # after ``gold/category_real_sample.csv`` has the tags

``draw``: a seeded random sample of (sub-news, main company) pairs of the first corpus run of the category step (revision
``category-65f33b55b26b``, cut ``units5-d3e2cbfa830e``) whose **summarized text is under 1,000 characters** (the text the step reads is
``news_summary.summary_text``, not the raw text; under 2,000 characters it is kept as it is). A sub-news from an article or a body used in
any tagged set (the training rows, the older rows, the held-out 200) is never drawn. Writes ``gold/texts/category_real_sample_texts.csv``
(index, article, ticker, company, the text the step reads) and, apart, ``docs/category_results_2026_11_15/real_sample_untouched_key.csv`` (what the untouched model said:
read only by ``score``, so the tagger does not see it).

``score``: reads the tags, scores the sample with the selected model (``models/category_single_all/``, the refit of the held-out run) and
reads the untouched model's labels and scores from the database; writes ``docs/category_results_2026_11_15/real_sample.md``.
The tags are NOT the owner's: they are made by the assistant, blind to both models' output, with the rules of ``docs/category-taxonomy.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import category_multilabel as ml
from news_category_stage import GROUP_OF

L14 = ml.LABELS14
L12 = ml.merge_labels(tuple(L14), ml.FOLD_INTO_OTHER)
DB = Path("/workspaces/thesis/data/nlp.db")
CAT_REV = "category-65f33b55b26b"
CUT_REV = "units5-d3e2cbfa830e"
TEXTS = ROOT / "gold" / "texts" / "category_real_sample_texts.csv"
TAGS = ROOT / "gold" / "category_real_sample.csv"
KEY = ROOT / "docs" / "category_results_2026_11_15" / "real_sample_untouched_key.csv"
WEIGHTS = ROOT / "models" / "category_single_all"
OUT = ROOT / "docs" / "category_results_2026_11_15" / "real_sample.md"


def used_articles() -> set[str]:
    ids: set[str] = set()
    for f, col in [
        ("gold/category_single_gold.csv", "article_id"),
        ("gold/category_multi_gold.csv", "article_id"),
        ("gold/category_v2_single.csv", "article_id"),
    ]:
        ids |= set(pd.read_csv(ROOT / f, dtype=str)[col])
    for f in ("gold/category_gold_v2.csv", "gold/category_gold_v2_test.csv"):
        ids |= {str(x) for x in pd.read_csv(ROOT / f).article_id}
    return ids


def draw(a: argparse.Namespace) -> None:
    if TEXTS.exists() and not a.more:
        sys.exit(
            f"{TEXTS} exists: the sample is drawn once; --more adds a second batch that shares no article or body with it"
        )
    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    used = used_articles()
    before_texts = before_key = None
    taken: set[str] = set()
    if a.more:
        before_texts = pd.read_csv(TEXTS, dtype={"article_id": str})
        before_key = pd.read_csv(KEY, dtype={"item_id": str})
        used |= set(before_texts.article_id)
        ph0 = ",".join("?" * len(before_key))
        taken = {
            r[0]
            for r in c.execute(
                f"SELECT DISTINCT body_hash FROM sub_news WHERE id IN ({ph0})",
                [int(x) for x in before_key.item_id],
            )
        }
        print(
            f"[draw] the first batch has {len(before_key)} rows, {len(taken)} bodies: none of them is drawn again"
        )
    ph = ",".join("?" * len(used))
    bad_hash = {
        r[0]
        for r in c.execute(
            f"SELECT DISTINCT body_hash FROM sub_news WHERE article_id IN ({ph})", list(used)
        )
    } | taken
    q = """SELECT k.item_id, k.ticker, n.article_id, n.body_hash, s.summary_text, s.text_source, k.label AS model_label, k.score, k.slug_scores
            FROM news_category k
            JOIN news_summary s ON s.item_kind = k.item_kind AND s.item_id = k.item_id
            JOIN sub_news n ON k.item_kind = 'sub_news' AND n.id = CAST(k.item_id AS INTEGER)
            WHERE k.model_revision = ? AND n.extractor_revision = ? AND length(s.summary_text) < 1000"""
    cols = [
        "item_id",
        "ticker",
        "article_id",
        "body_hash",
        "text",
        "text_source",
        "model_label",
        "model_score",
        "slug_scores",
    ]
    df = pd.DataFrame(c.execute(q, (CAT_REV, CUT_REV)).fetchall(), columns=cols)
    n0 = len(df)
    df = df[~df.article_id.astype(str).isin(used) & ~df.body_hash.isin(bad_hash)]
    print(
        f"[draw] pairs under 1,000 chars: {n0}; after leaving out the articles and bodies of every tagged set{' and of the first batch' if a.more else ''}: {len(df)}"
    )
    df = df.drop_duplicates("body_hash")  # one text per body: no two rows of one round-up
    sample = df.sample(a.n, random_state=a.seed).reset_index(drop=True)
    uni = pd.read_csv(ROOT / "gold/universe.csv").fillna("")
    name = dict(zip(uni.ticker, uni.company, strict=True))
    sample["company"] = [name.get(t, t) for t in sample.ticker]
    start = 0 if before_texts is None else len(before_texts)
    sample.insert(0, "idx", range(start, start + len(sample)))
    new_texts = sample[["idx", "article_id", "ticker", "company", "text_source", "text"]]
    new_key = sample[["idx", "item_id", "ticker", "model_label", "model_score", "slug_scores"]]
    if before_texts is not None and before_key is not None:
        new_texts, new_key = pd.concat([before_texts, new_texts]), pd.concat([before_key, new_key])
    TEXTS.parent.mkdir(parents=True, exist_ok=True)
    new_texts.to_csv(TEXTS, index=False)
    KEY.parent.mkdir(parents=True, exist_ok=True)
    new_key.to_csv(KEY, index=False)
    print(
        f"[draw] {len(sample)} new rows (idx {start} to {start + len(sample) - 1}); text_source {sample.text_source.value_counts().to_dict()}; median {int(sample.text.str.len().median())} chars; tickers {sample.ticker.nunique()}"
    )


def cell(v: float | None, ci: tuple[float, float] | None) -> str:
    return "-" if v is None else f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{v:.2f}"


def metrics(
    name: str, gold: list[str], pred: list[str], S: np.ndarray, articles: list[str]
) -> list[str]:
    ok = np.array([p == g for p, g in zip(pred, gold, strict=True)])
    okg = np.array([GROUP_OF[p] == GROUP_OF[g] for p, g in zip(pred, gold, strict=True)])
    lo, hi = ml.bootstrap_ci(ok.astype(float), articles, boot=2000)
    glo, ghi = ml.bootstrap_ci(okg.astype(float), articles, boot=2000)
    per = ml.per_label([[g] for g in gold], [[p] for p in pred], L12)
    ci = ml.per_label_ci([[g] for g in gold], [[p] for p in pred], L12, articles, boot=1000)
    f1 = []
    for lab in L12:
        p, r = per[lab]["precision"] or 0.0, per[lab]["recall"] or 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    out = [
        f"\n## {name}\n",
        f"- **label agreement {ok.mean():.1%} [{lo:.1%}, {hi:.1%}]**, group agreement {okg.mean():.1%} [{glo:.1%}, {ghi:.1%}], macro F1 over the 12 labels {np.mean(f1):.3f}\n",
    ]
    out.append(
        "| class | rows | predicted | precision [95%] | recall [95%] | ROC AUC |\n|---|---|---|---|---|---|"
    )
    for j, lab in enumerate(L12):
        auc = ml.roc_auc(S[:, j], [g == lab for g in gold]) if per[lab]["support"] else None
        out.append(
            f"| {lab} | {int(per[lab]['support'] or 0)} | {int(per[lab]['predicted'] or 0)} | {cell(per[lab]['precision'], ci[lab]['precision'])} | {cell(per[lab]['recall'], ci[lab]['recall'])} | {'-' if auc is None else f'{auc:.2f}'} |"
        )
    return out


def score(a: argparse.Namespace) -> None:
    import category_multilabel_cv_2026_11_14 as cvs  # noqa: PLC0415
    import category_single_cv_2026_11_15 as scv  # noqa: PLC0415 -- sets the labels of cvs to the 12
    import torch  # noqa: PLC0415 -- only the scoring needs it
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    assert tuple(cvs.LABELS) == tuple(scv.L12)
    texts = pd.read_csv(TEXTS, dtype={"article_id": str})
    tags = pd.read_csv(TAGS)
    df = texts.merge(tags[["idx", "label", "flag"]], on="idx")
    assert len(df) == len(texts) and df.label.isin(L12).all(), (
        "every row needs one of the 12 labels"
    )
    key = pd.read_csv(KEY)
    df = df.merge(key[["idx", "model_label", "slug_scores"]], on="idx")
    tok = AutoTokenizer.from_pretrained(WEIGHTS)
    model = AutoModelForSequenceClassification.from_pretrained(WEIGHTS).to(cvs.DEV)
    ent = next(i for lab, i in model.config.label2id.items() if lab.lower().startswith("entail"))
    S_sel = cvs.score(tok, model, ent, df, 512)
    del model
    torch.cuda.empty_cache()
    pred_sel = [L12[j] for j in S_sel.argmax(1)]
    base14 = np.array([[json.loads(s)[lab] for lab in L14] for s in df.slug_scores])
    S_base = ml.merge_scores(base14, tuple(L14), ml.FOLD_INTO_OTHER)
    pred_base = [L12[j] for j in S_base.argmax(1)]
    assert [ml.FOLD_INTO_OTHER.get(x, x) for x in df.model_label] == pred_base, (
        "the database label is the top of its scores"
    )
    gold, articles = list(df.label), [str(x) for x in df.article_id]
    out = [
        f"# The category model on real data: {len(df)} sub-news pairs under 1,000 characters (the text the step reads)\n"
    ]
    out.append(
        f"Drawn from the first corpus run (`{CAT_REV}`, cut `{CUT_REV}`), seeded, none from a tagged set. The tags are the assistant's (blind to both models), not the owner's. Tag counts: {dict(pd.Series(gold).value_counts())}. Rows the tagger flagged unclear: {int(df.flag.notna().sum())}."
    )
    out += metrics("Selected recipe (refit on 1,216 rows)", gold, pred_sel, S_sel, articles)
    out += metrics(
        "Untouched zero-shot model (what the step ran), 14 labels read as 12",
        gold,
        pred_base,
        S_base,
        articles,
    )
    ok_sel = np.array([p == g for p, g in zip(pred_sel, gold, strict=True)])
    ok_base = np.array([p == g for p, g in zip(pred_base, gold, strict=True)])
    unclear = df.flag.notna().to_numpy()
    idx = df.idx.to_numpy()
    out.append("\n## Label agreement by part of the sample\n")
    out.append("| rows | n | selected | untouched |\n|---|---|---|---|")
    parts = {
        "first batch (idx < 150)": idx < 150,
        "second batch (idx >= 150)": idx >= 150,
        "all": np.ones(len(df), bool),
        "tagger sure": ~unclear,
        "tagger unsure (flagged)": unclear,
    }
    for name_, m in parts.items():
        if m.any():
            out.append(
                f"| {name_} | {int(m.sum())} | {ok_sel[m].mean():.1%} | {ok_base[m].mean():.1%} |"
            )
    wins, losses = int((ok_sel & ~ok_base).sum()), int((~ok_sel & ok_base).sum())
    n_d = wins + losses
    p_sign = (
        min(1.0, 2 * sum(math.comb(n_d, k) for k in range(min(wins, losses) + 1)) / 2**n_d)
        if n_d
        else 1.0
    )
    out.append(
        f"\nPaired: the selected model is right where the untouched one is wrong on {wins} rows and the reverse on {losses} (two-sided sign test p = {p_sign:.3f})."
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "idx": df.idx,
            "article_id": df.article_id,
            "ticker": df.ticker,
            "tag": df.label,
            "selected": pred_sel,
            "untouched": pred_base,
        }
    ).to_csv(OUT.with_suffix(".csv"), index=False)
    print("\n".join(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["draw", "score"])
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument(
        "--more",
        action="store_true",
        help="draw a further batch (use another --seed): no article or body of the first one",
    )
    a = ap.parse_args()
    draw(a) if a.cmd == "draw" else score(a)


if __name__ == "__main__":
    main()
