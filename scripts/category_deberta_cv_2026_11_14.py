# ruff: noqa: E402, S311  -- one-shot research script, a reproducible draw
"""Plain DeBERTa-v3-base, 12 sigmoid outputs, cross-validated (T-187, owner 2026-11-14).

    uv run scripts/category_deberta_cv_2026_11_14.py cv [--smoke] [--epochs 10] [--head-lr 5e-4] [--no-pos-weight]

The encoder is ``microsoft/deberta-v3-base`` (not the zero-shot NLI model; its tokenizer is the zero-shot repo's, whose spm.model is byte-identical) and a new head gives one probability per label; the 14 labels of the
taxonomy are read as 12 (``competitive_actions`` and ``peer_comparison`` into ``other``, owner 2026-11-14). The probabilities are the product:
the cuts that turn them into the labels of a row are parameters of the report (``category_ovr_report_2026_11_14.py``: one global cut, a cut per
label read from a file, or one per label fitted on the other folds). Same rows and folds as ``category_multilabel_cv_2026_11_14.py``; the
held-out 200 is never read. Saves ``data/category_multilabel/oof_deb.npz`` (out-of-fold probabilities, folds) and ``oof_deb.meta.json``.
"""

from __future__ import annotations

import argparse
import json
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
import category_multilabel_cv_2026_11_14 as cvs

import category_multilabel as ml

LABELS12 = ml.merge_labels(cvs.LABELS, ml.FOLD_INTO_OTHER)
DEV = cvs.DEV
OUT = cvs.OUT


def load(model_id: str, revision: str | None):
    # The tokenizer of the pinned zero-shot model: its spm.model is byte-identical to microsoft/deberta-v3-base's (sha256 c679fbf9...),
    # and it ships a ready tokenizer.json, so the conversion that needs sentencepiece/tiktoken is not run.
    tok = AutoTokenizer.from_pretrained(cvs.b.MODEL, revision=cvs.b.REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        num_labels=len(LABELS12),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS12)),
        label2id={label: i for i, label in enumerate(LABELS12)},
        dtype=torch.float32,
    ).to(DEV)
    return tok, model


def targets(gold: list[list[str]]) -> torch.Tensor:
    y = torch.zeros(len(gold), len(LABELS12))
    for i, g in enumerate(gold):
        for label in g:
            y[i, LABELS12.index(label)] = 1.0
    return y


def train(tr: pd.DataFrame, args: argparse.Namespace, seed: int):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok, model = load(args.model, args.revision)
    head = [p for n, p in model.named_parameters() if n.startswith(("classifier", "pooler"))]
    body = [p for n, p in model.named_parameters() if not n.startswith(("classifier", "pooler"))]
    opt = torch.optim.AdamW(
        [{"params": body, "lr": args.lr}, {"params": head, "lr": args.head_lr}], weight_decay=0.01
    )
    prem = [cvs.premise(c, t) for c, t in zip(tr.company, tr.text, strict=True)]
    lengths = [len(x) for x in tok(prem, truncation=True, max_length=args.max_len)["input_ids"]]
    y_all = targets(list(tr.gold))
    pw = None
    if not args.no_pos_weight:
        w = ml.pos_weights(list(tr.gold), LABELS12)
        pw = torch.tensor([w[label] for label in LABELS12], device=DEV)
    steps = args.epochs * (len(tr) // (args.bs * args.accum) + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps)
    )
    scaler = torch.amp.GradScaler()
    model.train()
    for epoch in range(args.epochs):
        batches = ml.length_batches(lengths, args.bs, rng)
        tot = 0.0
        for n, idx in enumerate(batches, 1):
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
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y_all[idx].to(DEV), pos_weight=pw
                )
                / args.accum
            )
            scaler.scale(loss).backward()
            tot += loss.item() * args.accum
            if n % args.accum == 0 or n == len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
        print(f"    epoch {epoch + 1}: loss {tot / len(batches):.4f}", flush=True)
    return tok, model


def score(tok, model, df: pd.DataFrame, max_len: int, bs: int = 16) -> np.ndarray:
    model.eval()
    prem = [cvs.premise(c, t) for c, t in zip(df.company, df.text, strict=True)]
    out = np.zeros((len(df), len(LABELS12)))
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["cv"])
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument(
        "--revision", default=None, help="a commit of the model; the one used is recorded"
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pos-weight", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="40 rows, 1 epoch, 2 folds: the plumbing")
    args = ap.parse_args()
    df = cvs.rows()
    df["gold"] = ml.merge_gold(list(df.gold), ml.FOLD_INTO_OTHER)
    if args.smoke:
        df = df.sample(40, random_state=0).reset_index(drop=True)
        args.epochs, args.folds = 1, 2
    fold = np.array(ml.folds_of(list(df.article_id), args.folds))
    S = np.zeros((len(df), len(LABELS12)))
    sha = ""
    print(f"[deb] {len(df)} rows, {len(LABELS12)} labels, {args.model}", flush=True)
    for k in range(args.folds):
        tr, va = df[fold != k].reset_index(drop=True), df[fold == k].reset_index(drop=True)
        print(f"[deb] fold {k}: train {len(tr)}, validate {len(va)}", flush=True)
        tok, model = train(tr, args, args.seed + k)
        sha = str(getattr(model.config, "_commit_hash", "") or "")
        S[fold == k] = score(tok, model, va, args.max_len)
        del model
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "deb_smoke" if args.smoke else "deb"
    np.savez(OUT / f"oof_{tag}.npz", scores=S, fold=fold)
    meta = {**vars(args), "labels": list(LABELS12), "model_commit": sha, "rows": len(df)}
    (OUT / f"oof_{tag}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"saved {OUT / f'oof_{tag}.npz'} (model commit {sha})")


if __name__ == "__main__":
    main()
