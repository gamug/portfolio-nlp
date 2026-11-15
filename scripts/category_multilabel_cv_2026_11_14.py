# ruff: noqa: E402, S311  -- one-shot research script, a reproducible draw
"""Does fine-tuning the category model on the owner's multi-label tags pay off? (T-187, owner 2026-11-14)

    uv run scripts/category_multilabel_cv_2026_11_14.py cv --mode base|first|multi [--smoke]
    uv run scripts/category_multilabel_cv_2026_11_14.py report [--out docs/category_results_2026_11_14]

Rows: the 997 tagged rows of ``gold/category_multi_gold.csv`` and the 225 single-label train rows of ``gold/category_gold_v2.csv`` (the
same rows as the CV of 2026-11-13). The held-out 200 (``category_gold_v2_test.csv``) is NEVER read here (owner 2026-11-14).
Folds are grouped by article and the same for every mode. Modes: ``base`` the untouched model; ``first`` fine-tuned with the first
label of a row as the only positive (the old recipe); ``multi`` fine-tuned with every label of the row as a positive. In both
fine-tunes a negative is never a gold label, so they differ in the positives only. ``cv`` saves the out-of-fold scores of the 14
labels to ``data/category_multilabel/oof_<mode>.npz``; ``report`` picks the rule of the set (which extra labels join the top one)
on the other folds only, and prints the figures with intervals by article.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import category_baseline_2026_11_13 as b
import category_groups_mock_2026_11_13 as mock

import category_multilabel as ml
from news_nlp.triage import Company, CompanyResolver

LABELS = tuple(b.LABELS)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = ROOT / "data" / "category_multilabel"


def hyp(company: str, label: str) -> str:
    return f"The news about {company} is about {b.DEFS[label]}."


def premise(company: str, text: str) -> str:
    return f"{company}: {text[: b.PREMISE_CHARS]}"


def rows() -> pd.DataFrame:
    """The rows of the CV: article_id, company, text, gold (a list of labels)."""
    uni = pd.read_csv(ROOT / "gold/universe.csv").fillna("")
    res = CompanyResolver(
        [
            Company(t, c, s, "")
            for t, c, s in zip(uni.ticker, uni.company, uni.gics_sector, strict=False)
        ]
    )
    old, _ = mock.build_rows(
        pd.read_csv(ROOT / "gold/category_gold_v2.csv"),
        pd.read_csv(ROOT / "gold/texts/category_gold_v2_texts.csv"),
        res,
    )
    old = old[old.label.isin(LABELS)]
    a = pd.DataFrame(
        {
            "article_id": old.article_id.astype(str),
            "company": old.company,
            "text": old.text,
            "gold": [[label] for label in old.label],
            "source": "v2",
        }
    )
    tags = pd.read_csv(ROOT / "gold/category_multi_gold.csv", dtype=str)
    texts = pd.read_csv(ROOT / "gold/texts/category_multi_gold_texts.csv", dtype=str)
    m = tags.merge(texts[["review_order", "text"]], on="review_order")
    c = pd.DataFrame(
        {
            "article_id": m.article_id,
            "company": m.company,
            "text": m.text,
            "gold": [ml.parse_labels(x, LABELS) for x in m.labels],
            "source": "multi",
        }
    )
    return pd.concat([a, c], ignore_index=True)


def load() -> tuple[AutoTokenizer, AutoModelForSequenceClassification, int]:
    tok = AutoTokenizer.from_pretrained(b.MODEL, revision=b.REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        b.MODEL, revision=b.REVISION, dtype=torch.float32
    ).to(DEV)
    return tok, model, model.config.label2id["entailment"]


def train(tr: pd.DataFrame, first_only: bool, args: argparse.Namespace, seed: int):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok, model, ent = load()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    gold = list(tr.gold)
    n_pairs = len(ml.label_pairs(gold, LABELS, args.neg, random.Random(seed), first_only))
    steps = args.epochs * (n_pairs // (args.bs * args.accum) + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps)
    )
    scaler = torch.amp.GradScaler()
    model.train()
    prem = [premise(c, t) for c, t in zip(tr.company, tr.text, strict=True)]
    comp = list(tr.company)
    for epoch in range(args.epochs):
        pairs = ml.label_pairs(gold, LABELS, args.neg, rng, first_only)
        tot = 0.0
        for i in range(0, len(pairs), args.bs):
            chunk = pairs[i : i + args.bs]
            enc = tok(
                [prem[r] for r, _, _ in chunk],
                [hyp(comp[r], label) for r, label, _ in chunk],
                truncation="only_first",
                max_length=args.max_len,
                padding=True,
                return_tensors="pt",
            ).to(DEV)
            y = torch.tensor([ent if e else 1 - ent for _, _, e in chunk], device=DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                loss = (
                    torch.nn.functional.cross_entropy(model(**enc).logits.float(), y) / args.accum
                )
            scaler.scale(loss).backward()
            tot += loss.item() * args.accum
            if (i // args.bs + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
        print(f"    epoch {epoch + 1}: loss {tot / (len(pairs) / args.bs):.3f}", flush=True)
    return tok, model, ent


def score(tok, model, ent, df: pd.DataFrame, max_len: int) -> np.ndarray:
    model.eval()
    out = np.zeros((len(df), len(LABELS)))
    with torch.no_grad():
        for i, (c, t) in enumerate(zip(df.company, df.text, strict=True)):
            hs = [hyp(c, label) for label in LABELS]
            enc = tok(
                [premise(c, t)] * len(hs),
                hs,
                truncation="only_first",
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                out[i] = torch.softmax(model(**enc).logits.float(), -1)[:, ent].cpu().numpy()
    return out


def cmd_cv(args: argparse.Namespace) -> None:
    df = rows()
    if args.smoke:
        df = df.sample(40, random_state=0).reset_index(drop=True)
        args.epochs, args.folds = 1, 2
    fold = np.array(ml.folds_of(list(df.article_id), args.folds))
    S = np.zeros((len(df), len(LABELS)))
    print(f"[{args.mode}] {len(df)} rows ({(df.source == 'multi').sum()} multi sheet)", flush=True)
    if args.mode == "base":
        tok, model, ent = load()
        S[:] = score(tok, model, ent, df, args.max_len)
    else:
        for k in range(args.folds):
            tr, va = df[fold != k].reset_index(drop=True), df[fold == k].reset_index(drop=True)
            print(f"[{args.mode}] fold {k}: train {len(tr)}, validate {len(va)}", flush=True)
            tok, model, ent = train(tr, args.mode == "first", args, args.seed + k)
            S[fold == k] = score(tok, model, ent, va, args.max_len)
            del model
            torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}{'_smoke' if args.smoke else ''}"
    np.savez(OUT / f"oof_{tag}.npz", scores=S, fold=fold)
    df.drop(columns="text").assign(gold=df.gold.map(",".join)).to_csv(
        OUT / f"rows{'_smoke' if args.smoke else ''}.csv", index=False
    )
    print(f"saved {OUT / f'oof_{tag}.npz'}")


def cmd_report(args: argparse.Namespace) -> None:
    df = pd.read_csv(OUT / "rows.csv", dtype=str)
    gold = [x.split(",") for x in df.gold]
    arts = list(df.article_id)
    lines = [
        "| mode | rows | top label in the gold | exact set | micro F1 | macro F1 | labels per row |",
        "|---|---|---|---|---|---|---|",
    ]
    detail = {}
    for mode in ("base", "first", "multi"):
        f = OUT / f"oof_{mode}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        S, fold = z["scores"], z["fold"]
        top = [[LABELS[int(j)]] for j in S.argmax(1)]
        rules, sets = ml.cross_fitted_sets(S, gold, LABELS, fold)
        for name, pred in (("top only", top), ("top + rule", sets)):
            fig = ml.figures(gold, pred, LABELS)
            hit = np.array([p[0] in g for g, p in zip(gold, pred, strict=True)], dtype=float)
            lo, hi = ml.bootstrap_ci(hit, arts)
            ex = np.array([set(g) == set(p) for g, p in zip(gold, pred, strict=True)], dtype=float)
            elo, ehi = ml.bootstrap_ci(ex, arts)
            lines.append(
                f"| {mode} {name} | {int(fig['n'])} | {fig['hit_at_1']:.1%} [{lo:.1%}, {hi:.1%}] | {fig['exact']:.1%} [{elo:.1%}, {ehi:.1%}] | "
                f"{fig['micro_f1']:.3f} | {fig['macro_f1']:.3f} | {fig['mean_set_size']:.2f} |"
            )
        detail[mode] = {str(k): v for k, v in rules.items()}
        multi = np.array([len(g) > 1 for g in gold])
        for name, mask in (("single-label rows", ~multi), ("multi-label rows", multi)):
            idx = np.flatnonzero(mask)
            fig = ml.figures([gold[i] for i in idx], [sets[i] for i in idx], LABELS)
            lines.append(
                f"| {mode} top + rule, {name} | {int(fig['n'])} | {fig['hit_at_1']:.1%} | {fig['exact']:.1%} | {fig['micro_f1']:.3f} | {fig['macro_f1']:.3f} | {fig['mean_set_size']:.2f} |"
            )
    text = (
        "\n".join(lines)
        + "\n\nRule per fold (min score, ratio), picked on the other folds: "
        + str(detail)
        + "\n"
    )
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "figures.md").write_text(text, encoding="utf-8")
        for mode in ("base", "first", "multi"):
            f = OUT / f"oof_{mode}.npz"
            if f.exists():
                z = np.load(f)
                with (args.out / f"oof_{mode}.csv").open("w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh, lineterminator="\n")
                    w.writerow(["article_id", "company", "gold", "fold", *LABELS])
                    for i in range(len(df)):
                        w.writerow(
                            [
                                df.article_id[i],
                                df.company[i],
                                df.gold[i],
                                int(z["fold"][i]),
                                *[f"{v:.4f}" for v in z["scores"][i]],
                            ]
                        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cv = sub.add_parser("cv")
    cv.add_argument("--mode", choices=["base", "first", "multi"], required=True)
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--epochs", type=int, default=3)
    cv.add_argument("--lr", type=float, default=2e-5)
    cv.add_argument("--neg", type=int, default=4)
    cv.add_argument("--bs", type=int, default=4)
    cv.add_argument("--accum", type=int, default=4)
    cv.add_argument("--max-len", type=int, default=512)
    cv.add_argument("--seed", type=int, default=0)
    cv.add_argument("--smoke", action="store_true", help="40 rows, 1 epoch, 2 folds: the plumbing")
    cv.set_defaults(fn=cmd_cv)
    rp = sub.add_parser("report")
    rp.add_argument("--out", type=Path, default=None)
    rp.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
