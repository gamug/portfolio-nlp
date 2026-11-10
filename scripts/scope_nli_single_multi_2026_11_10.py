"""T-147, the owner's idea (2026-11-10): single or multi among the company bodies, with the NLI model.

    uv run python scripts/scope_nli_single_multi_2026_11_10.py [--features data/scope_relevance_features.json]

Only the bodies that really are company (single or multi: 170 of the 284, 165 of the 200). The category stage's NLI model
(``MoritzLaurer/deberta-v3-base-zeroshot-v2.0``, pinned) reads the title + first ``--chars`` characters (800 and 2,000 are tried:
a round-up shows its companies late) with two hypotheses. Four stage-B variants, scored on the 165 company bodies of the
held-out 200 (plain counts: the weights of the strata are not defined inside a class):
``current`` (the relevance features + logistic regression of ``scope_two_stage``), ``zero-shot`` (the higher entailment),
``fine-tuned`` (the NLI model trained on the 170, the relevance model's recipe) and ``nli + relevance features`` (the
fine-tuned score added to the relevance features in the logistic regression).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import scope_relevance_features_2026_11_10 as feat
import scope_title_baseline_2026_11_10 as base

CLASSES = ("single", "multi")
HYPOTHESES = {
    "single": "This text is about the news of one single company.",
    "multi": "This text covers the news of several different companies.",
}
RECIPE = {"epochs": 3, "lr": 2e-5, "batch": 16, "max_length": 512}


def scores(model, tok, texts: list[str]) -> np.ndarray:
    import torch  # noqa: PLC0415 -- heavy import only where the model runs

    out = np.zeros((len(texts), 2))
    model.eval()
    for j, name in enumerate(CLASSES):
        for i in range(0, len(texts), 16):
            chunk = texts[i : i + 16]
            x = tok(
                chunk,
                [HYPOTHESES[name]] * len(chunk),
                truncation="only_first",
                max_length=RECIPE["max_length"],
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**{k: v.cuda() for k, v in x.items()}).logits.float()
            out[i : i + 16, j] = (logits[:, 0] - logits[:, 1]).cpu().numpy()
    return out


def fine_tune(model, tok, texts: list[str], labels: list[str], seed: int = 1) -> None:
    import torch  # noqa: PLC0415
    from transformers import get_linear_schedule_with_warmup  # noqa: PLC0415

    torch.manual_seed(seed)
    rng = random.Random(seed)  # noqa: S311 -- a reproducible shuffle, not security
    pairs = [
        (t, HYPOTHESES[h], 0 if h == labels[i] else 1) for i, t in enumerate(texts) for h in CLASSES
    ]
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=0.01)
    steps = RECIPE["epochs"] * ((len(pairs) + 7) // 8)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    for _ in range(RECIPE["epochs"]):
        rng.shuffle(pairs)
        model.train()
        for i in range(0, len(pairs), 8):
            batch = pairs[i : i + 8]
            x = tok(
                [p[0] for p in batch],
                [p[1] for p in batch],
                truncation="only_first",
                max_length=RECIPE["max_length"],
                padding=True,
                return_tensors="pt",
            )
            y = torch.tensor([p[2] for p in batch], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(
                    model(**{k: v.cuda() for k, v in x.items()}).logits.float(), y
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()


def show(name: str, gold: list[str], pred: list[str]) -> None:
    right = {c: sum(g == p == c for g, p in zip(gold, pred, strict=True)) for c in CLASSES}
    total = {c: gold.count(c) for c in CLASSES}
    called = {c: pred.count(c) for c in CLASSES}
    print(
        f"{name:34} accuracy {np.mean([g == p for g, p in zip(gold, pred, strict=True)]):.3f} | "
        f"single recall {right['single']}/{total['single']} precision {right['single']}/{called['single']} | "
        f"multi recall {right['multi']}/{total['multi']} precision {right['multi']}/{called['multi']}"
    )


def main() -> None:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=Path("data/scope_relevance_features.json"))
    args = parser.parse_args()
    sets = {}
    for name, marks, texts in (
        ("train", "category_gold_v2.csv", "category_gold_v2_texts.csv"),
        ("test", "category_gold_v2_test.csv", "category_gold_v2_test_texts.csv"),
    ):
        text_of = {
            (r["article_id"], r["ticker"]): r for r in base.read(feat.GOLD / "texts" / texts)
        }
        rows = [
            {**r, **text_of[(r["article_id"], r["ticker"])]} for r in base.read(feat.GOLD / marks)
        ]
        sets[name] = rows
    cached = json.loads(args.features.read_text(encoding="utf-8"))["features"]
    keep = {n: [i for i, r in enumerate(sets[n]) if r["scope"] in CLASSES] for n in sets}
    rel = {n: np.array([[cached[n][i][k] for k in feat.FEATURES] for i in keep[n]]) for n in sets}
    y = {n: [sets[n][i]["scope"] for i in keep[n]] for n in sets}
    print(
        f"company bodies: train {len(y['train'])} ({y['train'].count('multi')} multi), test {len(y['test'])} ({y['test'].count('multi')} multi)"
    )
    cfg = json.loads(Path("config/pipeline_models.json").read_text(encoding="utf-8"))["category"]
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])

    def load():
        return AutoModelForSequenceClassification.from_pretrained(
            cfg["model"], revision=cfg["revision"], dtype=torch.float32
        ).cuda()

    scaler = StandardScaler().fit(rel["train"])
    cur = LogisticRegression(max_iter=2000).fit(scaler.transform(rel["train"]), y["train"])
    show(
        "current (relevance features)",
        y["test"],
        [str(p) for p in cur.predict(scaler.transform(rel["test"]))],
    )
    for chars in (800, 2000):
        x = {n: [feat.fragment(sets[n][i], chars) for i in keep[n]] for n in sets}
        model = load()
        zs_te = scores(model, tok, x["test"])
        show(f"zero-shot, {chars} characters", y["test"], [CLASSES[i] for i in zs_te.argmax(1)])
        fine_tune(model, tok, x["train"], y["train"])
        ft_te = scores(model, tok, x["test"])
        show(f"fine-tuned, {chars} characters", y["test"], [CLASSES[i] for i in ft_te.argmax(1)])
        # the fine-tuned score on its own training rows is over-fitted: the stack is fitted on the zero-shot score
        # of the untouched model instead, which has seen no label (cross-validated would need a model per fold)
        zero = load()
        zs_tr = scores(zero, tok, x["train"])
        lr = LogisticRegression(max_iter=2000).fit(
            scaler.transform(rel["train"])
            if False
            else np.hstack([scaler.transform(rel["train"]), zs_tr[:, :1] - zs_tr[:, 1:]]),
            y["train"],
        )
        show(
            f"zero-shot + relevance feat., {chars}",
            y["test"],
            [
                str(p)
                for p in lr.predict(
                    np.hstack([scaler.transform(rel["test"]), zs_te[:, :1] - zs_te[:, 1:]])
                )
            ],
        )
        del model, zero
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
