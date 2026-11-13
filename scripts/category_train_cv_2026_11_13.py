"""Cross-validated fine-tune of the zero-shot NLI model on the 225 category rows (owner 2026-11-13).

--mode flat  : one (premise, label hypothesis) pair per label.
--mode mixed : the same pairs plus (premise, group hypothesis) pairs, one model for both levels.
Folds are grouped by article and identical across modes (seeded). The held-out set is never read here.
Out-of-fold scores are saved to data/category_groups/oof_<mode>.csv.
Run: uv run scripts/category_train_cv_2026_11_13.py --mode mixed
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import category_baseline_2026_11_13 as b
import category_groups_mock_2026_11_13 as mock

from news_nlp.triage import Company, CompanyResolver

GN = list(b.GROUPS)
GROUP_DEF = {g: "; or ".join(b.DEFS[label] for label in ls) for g, ls in b.GROUPS.items()}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hyp_label(company, label):
    return f"The news about {company} is about {b.DEFS[label]}."


def hyp_group(company, group):
    return f"The news about {company} is about {GROUP_DEF[group]}."


def premise(r):
    return f"{r.company}: {r.text[: b.PREMISE_CHARS]}"


def folds_of(rows, k, seed=0):
    ids = sorted(rows.article_id.unique())
    random.Random(seed).shuffle(ids)
    fold = {a: i % k for i, a in enumerate(ids)}
    return rows.article_id.map(fold).values


def make_pairs(rows, mode, neg, rng):
    """(premise, hypothesis, entails) for one epoch; negatives are re-drawn each epoch."""
    out = []
    for r in rows.itertuples():
        p = premise(r)
        others = [label for label in b.LABELS if label != r.label]
        out.append((p, hyp_label(r.company, r.label), 1))
        out += [(p, hyp_label(r.company, o), 0) for o in rng.sample(others, neg)]
        if mode == "mixed":
            g = b.GROUP_OF[r.label]
            out.append((p, hyp_group(r.company, g), 1))
            out += [
                (p, hyp_group(r.company, o), 0)
                for o in rng.sample([x for x in GN if x != g], min(2, len(GN) - 1))
            ]
    rng.shuffle(out)
    return out


def train(rows, mode, args, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(b.MODEL, revision=b.REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        b.MODEL, revision=b.REVISION, dtype=torch.float32
    ).to(DEV)
    ent = model.config.label2id["entailment"]
    non = 1 - ent
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (
        len(make_pairs(rows, mode, args.neg, random.Random(seed))) // (args.bs * args.accum) + 1
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps)
    )
    scaler = torch.amp.GradScaler()
    model.train()
    step = 0
    for epoch in range(args.epochs):
        pairs = make_pairs(rows, mode, args.neg, rng)
        tot = 0.0
        for i in range(0, len(pairs), args.bs):
            chunk = pairs[i : i + args.bs]
            enc = tok(
                [c[0] for c in chunk],
                [c[1] for c in chunk],
                truncation="only_first",
                max_length=args.max_len,
                padding=True,
                return_tensors="pt",
            ).to(DEV)
            y = torch.tensor([ent if c[2] else non for c in chunk], device=DEV)
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
                step += 1
        print(f"    epoch {epoch + 1}: loss {tot / (len(pairs) / args.bs):.3f}", flush=True)
    return tok, model, ent


def score(tok, model, ent, rows, max_len):
    model.eval()
    SL = np.zeros((len(rows), len(b.LABELS)))
    SG = np.zeros((len(rows), len(GN)))
    with torch.no_grad():
        for i, r in enumerate(rows.itertuples()):
            hs = [hyp_label(r.company, label) for label in b.LABELS] + [
                hyp_group(r.company, g) for g in GN
            ]
            enc = tok(
                [premise(r)] * len(hs),
                hs,
                truncation="only_first",
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                p = torch.softmax(model(**enc).logits.float(), -1)[:, ent].cpu().numpy()
            SL[i], SG[i] = p[: len(b.LABELS)], p[len(b.LABELS) :]
    return SL, SG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["flat", "mixed"], required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--neg", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--smoke", action="store_true", help="30 rows, 1 epoch, 2 folds: checks the plumbing"
    )
    args = ap.parse_args()
    uni = pd.read_csv(ROOT / "gold/universe.csv").fillna("")
    res = CompanyResolver(
        [
            Company(t, c, s, "")
            for t, c, s in zip(uni.ticker, uni.company, uni.gics_sector, strict=False)
        ]
    )
    rows, _ = mock.build_rows(
        pd.read_csv(ROOT / "gold/category_gold_v2.csv"),
        pd.read_csv(ROOT / "gold/texts/category_gold_v2_texts.csv"),
        res,
    )
    rows = rows[rows.label.isin(b.LABELS)].reset_index(drop=True)
    if args.smoke:
        rows = rows.sample(30, random_state=0).reset_index(drop=True)
        args.epochs, args.folds = 1, 2
    fold = folds_of(rows, args.folds)
    SL = np.zeros((len(rows), len(b.LABELS)))
    SG = np.zeros((len(rows), len(GN)))
    for k in range(args.folds):
        tr, va = rows[fold != k].reset_index(drop=True), rows[fold == k].reset_index(drop=True)
        print(f"[{args.mode}] fold {k}: train {len(tr)} rows, validate {len(va)}", flush=True)
        tok, model, ent = train(tr, args.mode, args, args.seed + k)
        SL[fold == k], SG[fold == k] = score(tok, model, ent, va, args.max_len)
        del model
        torch.cuda.empty_cache()
    flat = [b.LABELS[j] for j in SL.argmax(1)]
    gi = [GN.index(b.GROUP_OF[label]) for label in b.LABELS]
    prod = [b.LABELS[int((SL[i] * SG[i][gi]).argmax())] for i in range(len(rows))]
    pg = [GN[j] for j in SG.argmax(1)]
    tg = [b.GROUP_OF[t] for t in rows.label]
    two = []
    for i, g in enumerate(pg):
        idx = [b.LABELS.index(label) for label in b.GROUPS[g]]
        two.append(b.LABELS[idx[int(SL[i, idx].argmax())]])
    acc = lambda a, c: np.mean([x == y for x, y in zip(a, c, strict=False)])
    print(
        f"\n[{args.mode}] out-of-fold n={len(rows)}: label flat {acc(flat, rows.label):.1%} | "
        f"group x label {acc(prod, rows.label):.1%} | two-step {acc(two, rows.label):.1%} | "
        f"group {acc(pg, tg):.1%} (group via flat label {acc([b.GROUP_OF[x] for x in flat], tg):.1%})"
    )
    out = ROOT / "data" / "category_groups"
    scores_file = f"oof_{args.mode}{'_smoke' if args.smoke else ''}_scores.npz"
    np.savez(out / scores_file, labels=SL, groups=SG)  # D37: ROC OVR needs the scores
    pd.DataFrame(
        {
            "article_id": rows.article_id,
            "company": rows.company,
            "label": rows.label,
            "fold": fold,
            "flat": flat,
            "prod": prod,
            "two_step": two,
            "group_pred": pg,
        }
    ).to_csv(out / f"oof_{args.mode}{'_smoke' if args.smoke else ''}.csv", index=False)


if __name__ == "__main__":
    main()

# ruff: noqa: E402, E731, S311  -- one-shot research script
