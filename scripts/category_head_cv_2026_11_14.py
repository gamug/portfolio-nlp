# ruff: noqa: E402, S311  -- one-shot research script, a reproducible draw
"""A multi-label head on the same encoder, cross-validated like the entailment fine-tune (T-187, owner 2026-11-14).

    uv run scripts/category_head_cv_2026_11_14.py cv [--smoke]
    uv run scripts/category_head_cv_2026_11_14.py report [--out docs/category_results_2026_11_14]

The encoder is the pinned zero-shot model's (``MoritzLaurer/deberta-v3-base-zeroshot-v2.0``); its 3-way entailment head is replaced by 14
sigmoid outputs trained with binary cross-entropy on the premise ``<company>: <text>`` alone: one forward pass per row, not one per
(row, label) pair. Same rows and same folds as ``category_multilabel_cv_2026_11_14.py`` (the held-out 200 is never read). ``cv`` saves the
out-of-fold probabilities to ``data/category_multilabel/oof_head.npz``; ``report`` fits one cut per label on the other folds only and
prints the figures next to the untouched model and the entailment fine-tune, and the one-versus-rest tables.
"""

from __future__ import annotations

import argparse
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
import category_multilabel_cv_2026_11_14 as cvs

import category_multilabel as ml

LABELS = cvs.LABELS
DEV = cvs.DEV
OUT = cvs.OUT


def load() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tok = AutoTokenizer.from_pretrained(b.MODEL, revision=b.REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        b.MODEL,
        revision=b.REVISION,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
        dtype=torch.float32,
    ).to(DEV)
    return tok, model


def targets(gold: list[list[str]]) -> torch.Tensor:
    y = torch.zeros(len(gold), len(LABELS))
    for i, g in enumerate(gold):
        for label in g:
            y[i, LABELS.index(label)] = 1.0
    return y


def train(tr: pd.DataFrame, args: argparse.Namespace, seed: int):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok, model = load()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr) // (args.bs * args.accum) + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps)
    )
    scaler = torch.amp.GradScaler()
    prem = [cvs.premise(c, t) for c, t in zip(tr.company, tr.text, strict=True)]
    y_all = targets(list(tr.gold))
    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(tr)))
        rng.shuffle(order)
        tot = 0.0
        for i in range(0, len(order), args.bs):
            idx = order[i : i + args.bs]
            enc = tok(
                [prem[j] for j in idx],
                truncation=True,
                max_length=args.max_len,
                padding=True,
                return_tensors="pt",
            ).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                logits = model(**enc).logits.float()
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(logits, y_all[idx].to(DEV))
                / args.accum
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
        print(f"    epoch {epoch + 1}: loss {tot / (len(order) / args.bs):.4f}", flush=True)
    return tok, model


def score(tok, model, df: pd.DataFrame, max_len: int, bs: int = 16) -> np.ndarray:
    model.eval()
    out = np.zeros((len(df), len(LABELS)))
    prem = [cvs.premise(c, t) for c, t in zip(df.company, df.text, strict=True)]
    with torch.no_grad():
        for i in range(0, len(df), bs):
            enc = tok(
                prem[i : i + bs],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                out[i : i + bs] = torch.sigmoid(model(**enc).logits.float()).cpu().numpy()
    return out


def cmd_cv(args: argparse.Namespace) -> None:
    df = cvs.rows()
    if args.smoke:
        df = df.sample(40, random_state=0).reset_index(drop=True)
        args.epochs, args.folds = 1, 2
    fold = np.array(ml.folds_of(list(df.article_id), args.folds))
    S = np.zeros((len(df), len(LABELS)))
    print(f"[head] {len(df)} rows", flush=True)
    for k in range(args.folds):
        tr, va = df[fold != k].reset_index(drop=True), df[fold == k].reset_index(drop=True)
        print(f"[head] fold {k}: train {len(tr)}, validate {len(va)}", flush=True)
        tok, model = train(tr, args, args.seed + k)
        S[fold == k] = score(tok, model, va, args.max_len)
        del model
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    name = "oof_head_smoke.npz" if args.smoke else "oof_head.npz"
    np.savez(OUT / name, scores=S, fold=fold)
    print(f"saved {OUT / name}")


def ovr_table(gold: list[list[str]], pred: list[list[str]], S: np.ndarray) -> pd.DataFrame:
    pl = ml.per_label(gold, pred, LABELS)
    rows = []
    for j, label in enumerate(LABELS):
        auc = ml.roc_auc(S[:, j], [int(label in g) for g in gold])
        rows.append(
            (
                label,
                int(pl[label]["support"] or 0),
                int(pl[label]["predicted"] or 0),
                pl[label]["precision"],
                pl[label]["recall"],
                pl[label]["f1"],
                auc,
            )
        )
    t = pd.DataFrame(
        rows, columns=["label", "gold rows", "predicted", "precision", "recall", "F1", "ROC AUC"]
    )
    return t.round(3)


def cmd_report(args: argparse.Namespace) -> None:
    df = pd.read_csv(OUT / "rows.csv", dtype=str)
    gold = [x.split(",") for x in df.gold]
    arts = list(df.article_id)
    lines = [
        "| model, output | records | top label in the gold | exact set | micro F1 | macro F1 | labels per record | pair accuracy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    tables = {}
    for mode, label in (
        ("base", "untouched"),
        ("multi", "entailment fine-tune"),
        ("head", "multi-label head"),
    ):
        f = OUT / f"oof_{mode}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        S, fold = z["scores"], z["fold"]
        top = [[LABELS[int(j)]] for j in S.argmax(1)]
        if mode == "head":
            sets = ml.cross_fitted_thresholds(S, gold, LABELS, fold)
        else:
            _, sets = ml.cross_fitted_sets(S, gold, LABELS, fold)
        for name, pred in (("top label", top), ("set", sets)):
            fig = ml.figures(gold, pred, LABELS)
            hit = np.array([p[0] in g for g, p in zip(gold, pred, strict=True)], dtype=float)
            lo, hi = ml.bootstrap_ci(hit, arts)
            pairs = ml.explode_pairs(gold, [p[0] for p in pred])
            pacc = float(np.mean([a == o for _, a, o in pairs]))
            lines.append(
                f"| {label}, {name} | {int(fig['n'])} | {fig['hit_at_1']:.1%} [{lo:.1%}, {hi:.1%}] | {fig['exact']:.1%} | "
                f"{fig['micro_f1']:.3f} | {fig['macro_f1']:.3f} | {fig['mean_set_size']:.2f} | {pacc:.1%} |"
            )
        tables[label] = (ovr_table(gold, sets, S), ovr_table_paired(gold, top, S))
    text = (
        "\n".join(lines)
        + f"\n\nPair ceiling (one output per record): {ml.pair_ceiling(gold):.1%}\n"
    )
    for label, (sets_t, paired_t) in tables.items():
        text += f"\n### {label}: one versus rest on the output set\n\n{ml.markdown_table(list(sets_t.columns), sets_t.values.tolist())}\n"
        text += f"\n### {label}: one versus rest on the pairs (one row per gold tag, the top label as the output)\n\n{ml.markdown_table(list(paired_t.columns), paired_t.values.tolist())}\n"
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "figures_head.md").write_text(text, encoding="utf-8")


def ovr_table_paired(gold: list[list[str]], top: list[list[str]], S: np.ndarray) -> pd.DataFrame:
    pairs = ml.explode_pairs(gold, [p[0] for p in top])
    g1 = [[t] for _, t, _ in pairs]
    p1 = [[o] for *_, o in pairs]
    pl = ml.per_label(g1, p1, LABELS)
    return pd.DataFrame(
        [
            (
                label,
                int(pl[label]["support"] or 0),
                int(pl[label]["predicted"] or 0),
                pl[label]["precision"],
                pl[label]["recall"],
                pl[label]["f1"],
            )
            for label in LABELS
        ],
        columns=["label", "pairs with the tag", "pairs predicted", "precision", "recall", "F1"],
    ).round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cv = sub.add_parser("cv")
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--epochs", type=int, default=5)
    cv.add_argument("--lr", type=float, default=3e-5)
    cv.add_argument("--bs", type=int, default=8)
    cv.add_argument("--accum", type=int, default=2)
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
