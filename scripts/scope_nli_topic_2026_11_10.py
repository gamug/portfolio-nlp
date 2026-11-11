"""T-147, the owner's idea (2026-11-10): the topic stage (company / sector / not_related) with the NLI model, no embeddings.

    uv run python scripts/scope_nli_topic_2026_11_10.py [--features data/scope_relevance_features.json] [--chars 800]

The category stage's NLI model (``MoritzLaurer/deberta-v3-base-zeroshot-v2.0`` at the pinned revision) reads the title + first
``--chars`` characters as the premise and one hypothesis per topic. Three ways, all scored on the 200 held-out the way
``scope_two_stage_2026_11_10.py`` scores its stage A (stage B, single or multi among the companies, is the same):
``zero-shot`` (the highest entailment), ``zero-shot scores + LR`` (the three entailment logits into a logistic regression
trained on the 284) and ``fine-tuned`` (the model trained on the 284: each text gives three pairs, entailment for its own
topic; the relevance model's recipe: 3 epochs, lr 2e-5, batch 16, float32 weights, bf16 autocast).
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
from scope_two_stage_2026_11_10 import topic

TOPICS = ("company", "sector", "not_related")
HYPOTHESES = {
    "company": "This text is about one specific company or a few companies.",
    "sector": "This text is about an industry or sector as a whole.",
    "not_related": "This text is about the general market, the economy or investing advice, or it is not about companies.",
}
RECIPE = {"epochs": 3, "lr": 2e-5, "batch": 16, "max_length": 320}


def scores(model, tok, texts: list[str]) -> np.ndarray:
    import torch  # noqa: PLC0415 -- heavy import only where the model runs

    out = np.zeros((len(texts), len(TOPICS)))
    model.eval()
    for j, name in enumerate(TOPICS):
        for i in range(0, len(texts), 32):
            x = tok(
                texts[i : i + 32],
                [HYPOTHESES[name]] * len(texts[i : i + 32]),
                truncation="only_first",
                max_length=RECIPE["max_length"],
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**{k: v.cuda() for k, v in x.items()}).logits.float()
            out[i : i + 32, j] = (
                (logits[:, 0] - logits[:, 1]).cpu().numpy()
            )  # entailment against not_entailment
    return out


def fine_tune(model, tok, texts: list[str], topics: list[str], seed: int = 1) -> None:
    import torch  # noqa: PLC0415
    from transformers import get_linear_schedule_with_warmup  # noqa: PLC0415

    torch.manual_seed(seed)
    rng = random.Random(seed)  # noqa: S311 -- a reproducible shuffle, not security
    pairs = [
        (t, HYPOTHESES[h], 0 if h == topics[i] else 1) for i, t in enumerate(texts) for h in TOPICS
    ]
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=0.01)
    steps = RECIPE["epochs"] * ((len(pairs) + RECIPE["batch"] - 1) // RECIPE["batch"])
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    for _ in range(RECIPE["epochs"]):
        rng.shuffle(pairs)
        model.train()
        for i in range(0, len(pairs), RECIPE["batch"]):
            batch = pairs[i : i + RECIPE["batch"]]
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


def main() -> None:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=Path("data/scope_relevance_features.json"))
    parser.add_argument("--chars", type=int, default=800)
    args = parser.parse_args()
    sets = {}
    for name, marks, texts in (
        ("train", "category_gold_v2.csv", "category_gold_v2_texts.csv"),
        ("test", "category_gold_v2_test.csv", "category_gold_v2_test_texts.csv"),
    ):
        text_of = {
            (r["article_id"], r["ticker"]): r for r in base.read(feat.GOLD / "texts" / texts)
        }
        sets[name] = [
            {**r, **text_of[(r["article_id"], r["ticker"])]} for r in base.read(feat.GOLD / marks)
        ]
    train, test = sets["train"], sets["test"]
    cached = json.loads(args.features.read_text(encoding="utf-8"))["features"]
    f_tr = np.array([[f[n] for n in feat.FEATURES] for f in cached["train"]])
    f_te = np.array([[f[n] for n in feat.FEATURES] for f in cached["test"]])
    x_tr = [feat.fragment(r, args.chars) for r in train]
    x_te = [feat.fragment(r, args.chars) for r in test]
    y_tr = [topic(r["scope"]) for r in train]
    cfg = json.loads(Path("config/pipeline_models.json").read_text(encoding="utf-8"))["category"]
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])

    def load():
        return AutoModelForSequenceClassification.from_pretrained(
            cfg["model"], revision=cfg["revision"], dtype=torch.float32
        ).cuda()

    # stage B: single or multi, trained on the company bodies only (as in scope_two_stage)
    company = np.array([t == "company" for t in y_tr])
    scaler = StandardScaler().fit(f_tr[company])
    b_model = LogisticRegression(max_iter=2000).fit(
        scaler.transform(f_tr[company]),
        [r["scope"] for r, c in zip(train, company, strict=True) if c],
    )
    b_pred = [str(p) for p in b_model.predict(scaler.transform(f_te))]
    gold_topic = [topic(r["scope"]) for r in test]

    def report(name: str, a_pred: list[str]) -> None:
        pred = [
            b if a == "company" else ("sector" if a == "sector" else "out_of_scope")
            for a, b in zip(a_pred, b_pred, strict=True)
        ]
        print(
            f"\n== {name}: topic accuracy {np.mean([g == a for g, a in zip(gold_topic, a_pred, strict=True)]):.3f}"
        )
        print("\n".join(base.lines("end to end", test, pred)))
        print(
            "   topic confusion (company / sector / not_related):",
            [
                [
                    sum(1 for g, a in zip(gold_topic, a_pred, strict=True) if g == gg and a == aa)
                    for aa in TOPICS
                ]
                for gg in TOPICS
            ],
        )

    model = load()
    s_tr, s_te = scores(model, tok, x_tr), scores(model, tok, x_te)
    report("zero-shot (highest entailment)", [TOPICS[i] for i in s_te.argmax(1)])
    lr = LogisticRegression(max_iter=2000).fit(s_tr, y_tr)
    report("zero-shot scores + logistic regression", [str(p) for p in lr.predict(s_te)])
    fine_tune(model, tok, x_tr, y_tr)
    report("fine-tuned on the 284", [TOPICS[i] for i in scores(model, tok, x_te).argmax(1)])


if __name__ == "__main__":
    main()
