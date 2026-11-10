"""The final relevance model and the stack's coefficients (T-169; owner 2026-11-10, PLAN D35/D40).

    uv run python scripts/final_models_2026_11_10.py relevance [--out models/relevance_final] [--seed 1] [--order-of OOF.json]
    uv run python scripts/final_models_2026_11_10.py stack [--cv data/sentiment_pair_cv_all] \
        [--oof data/sentiment_pair_relevance_cv/oof_relevance.json] [--out config/sentiment_stack.json] [--verify]

``relevance`` fine-tunes the category stage's NLI model (``MoritzLaurer/deberta-v3-base-zeroshot-v2.0`` at the pinned
revision of ``config/pipeline_models.json``) on **every** one of the 809 pair rows of ``gold/sentiment_train.csv`` and
``gold/sentiment_train_2.csv`` (the texts are in ``gold/texts/``): premise = the text, hypothesis ``This text is mainly
about <company>.``, entailment when the owner marked a direction, not-entailment when the mark is neutral. The recipe is
the cross-validated one (3 epochs, lr 2e-5, batch 16, float32 weights, bf16 autocast, max length 320), which scored an
out-of-fold weighted AUC of 0.896. The model is not evaluated on its own training rows. **A GPU is required** (the run
stops with a message when CUDA is missing). The 809 rows are the tracked rows whose ``rule_scope`` is ``single`` or
``multi_dominated`` (the 46 company-less scopes are left out), so the command needs only tracked files; ``--order-of`` takes
the key order of an out-of-fold file, which the published run used: with it the run was repeated on 2026-11-10 and gave
the same weights, bit for bit (sha256 of ``model.safetensors`` and ``final_loss`` 0.06313314288854599). Without it the
order is the file order, so the shuffle differs and the weights are not those published (how much, not measured).

``stack`` fits the multinomial logistic regression of the nested evaluation (features: the log of the three sentiment
probabilities and the logit of the relevance score; C = 1, rows weighted by ``pool size / drawn``) on the **out-of-fold**
scores of the cross-validation (both repeats), and writes the coefficients to a JSON next to the code. The out-of-fold
files are in the git-ignored ``data/``; the coefficients are the tracked result.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

csv.field_size_limit(sys.maxsize)
CLASSES = (
    "positive",
    "negative",
    "neutral",
)  # the order of the model's labels and of the stack's features
SETS = (
    ("gold/sentiment_train.csv", "gold/texts/sentiment_train_texts.csv"),
    ("gold/sentiment_train_2.csv", "gold/texts/sentiment_train_2_texts.csv"),
)
HYPOTHESIS = "This text is mainly about {}."
RECIPE = {"epochs": 3, "lr": 2e-5, "batch": 16, "max_length": 320}
PINNED = Path("config/pipeline_models.json")
PAIR_SCOPES = (
    "single",
    "multi_dominated",
)  # rows that read as a pair (company-less scopes are not)


def read(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def company_name(company: str) -> str:
    return company.split(" (", maxsplit=1)[0]


def pair_rows(keys: list[str] | None = None) -> list[dict[str, str]]:
    """The marked rows with their text. With *keys* (the 809 rows of the cross-validation) only those, in that order.

    Without *keys*: every row that reads as a pair (``rule_scope`` in ``PAIR_SCOPES``), in file order.
    """
    rows: dict[str, dict[str, str]] = {}
    for draw, (marks, texts) in zip(("old", "new"), SETS, strict=True):
        text_of = {r["key"]: r for r in read(texts)}
        for r in read(marks):
            rows[r["key"]] = {**r, "draw": draw, "text": text_of[r["key"]]["text"]}
    if keys is None:
        return [r for r in rows.values() if r["rule_scope"] in PAIR_SCOPES]
    return [rows[k] for k in keys]


def feats(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """The stack's four features: the log of the three sentiment probabilities and the logit of the relevance score."""
    r = np.clip(r, 1e-6, 1 - 1e-6)
    return np.column_stack([np.log(np.clip(p, 1e-6, 1)), np.log(r / (1 - r))])


def apply_stack(coefficients: dict[str, Any], p: list[float], r: float) -> dict[str, float]:
    """The stack's probabilities for one text (*p*: the sentiment model's, in ``CLASSES`` order; *r*: the relevance score)."""
    z = np.array(coefficients["coef"]) @ feats(np.array([p]), np.array([r]))[0]
    z = z + np.array(coefficients["intercept"])
    e = np.exp(z - z.max())
    return dict(zip(coefficients["classes"], (e / e.sum()).tolist(), strict=True))


def relevance(out: Path, seed: int, order_of: Path | None = None) -> None:
    import torch  # noqa: PLC0415 -- heavy imports only for the command that needs them
    from transformers import (  # noqa: PLC0415
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    cfg = json.loads(PINNED.read_text(encoding="utf-8"))["category"]
    if not torch.cuda.is_available():
        raise SystemExit("relevance needs a GPU (CUDA is not available)")
    keys = json.loads(order_of.read_text(encoding="utf-8"))["keys"] if order_of else None
    rows = pair_rows(keys)  # the 809 rows that read as a pair (no company-less scope)
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    torch.manual_seed(seed)
    rng = random.Random(seed)  # noqa: S311 -- a reproducible shuffle, not security
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], revision=cfg["revision"], dtype=torch.float32
    ).cuda()  # transformers 5 loads fp16: AdamW gives NaN
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=0.01)
    steps = RECIPE["epochs"] * ((len(rows) + RECIPE["batch"] - 1) // RECIPE["batch"])
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    def enc(batch: list[dict[str, str]]) -> Any:
        return tok(
            [r["text"] for r in batch],
            [HYPOTHESIS.format(company_name(r["company"])) for r in batch],
            truncation="only_first",
            max_length=RECIPE["max_length"],
            padding=True,
            return_tensors="pt",
        )

    losses: list[float] = []
    for epoch in range(RECIPE["epochs"]):
        order = rows[:]
        rng.shuffle(order)
        model.train()
        for i in range(0, len(order), RECIPE["batch"]):
            batch = order[i : i + RECIPE["batch"]]
            x = enc(batch).to("cuda")
            y = torch.tensor(
                [0 if r["sentiment"] != "neutral" else 1 for r in batch], device="cuda"
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(**x).logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(float(loss))
        print(
            f"epoch {epoch + 1}: mean loss {np.mean(losses[-(len(order) // RECIPE['batch']) :]):.4f}",
            flush=True,
        )
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "training_record.json").write_text(
        json.dumps(
            {
                "base": f"{cfg['model']}@{cfg['revision']}",
                "rows": len(rows),
                "relevant": sum(r["sentiment"] != "neutral" for r in rows),
                "seed": seed,
                "recipe": RECIPE,
                "final_loss": losses[-1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {out} ({len(rows)} rows)")


def input_hashes(cv_dir: Path, oof: Path) -> tuple[str, str]:
    """sha256 of the out-of-fold file and of the ten prediction files (``cv_r[12]_f*__*.csv``) concatenated in sorted name order."""
    files = sorted(glob.glob(f"{cv_dir}/predictions/cv_r[12]_f*__*.csv"))
    joined = hashlib.sha256()
    for name in files:
        joined.update(Path(name).read_bytes())
    return hashlib.sha256(oof.read_bytes()).hexdigest(), joined.hexdigest()


def stack(cv_dir: Path, oof: Path, out: Path) -> None:
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    saved = json.loads(oof.read_text(encoding="utf-8"))
    keys = saved["keys"]
    rows = {r["key"]: r for r in pair_rows(keys)}
    cell = {k: (rows[k]["draw"], rows[k]["predicted"]) for k in keys}
    pool = {cell[k]: int(rows[k]["pool_class_size"]) for k in keys}
    count: dict[tuple[str, str], int] = defaultdict(int)
    for k in keys:
        count[cell[k]] += 1
    weight = np.array([pool[cell[k]] / count[cell[k]] for k in keys])
    gold = np.array([CLASSES.index(rows[k]["sentiment"]) for k in keys])
    probs = []
    for rpt in (1, 2):
        got: dict[str, dict[str, str]] = {}
        for path in glob.glob(f"{cv_dir}/predictions/cv_r{rpt}_f*__*.csv"):
            for r in read(path):
                got[r["key"]] = r
        probs.append(np.array([[float(got[k][f"p_{c}"]) for c in CLASSES] for k in keys]))

    x = np.vstack([feats(probs[i], np.array(saved["R"][i])) for i in (0, 1)])
    y = np.concatenate([gold, gold])
    w = np.concatenate([weight, weight])
    lr = LogisticRegression(C=1.0, max_iter=500).fit(x, y, sample_weight=w / w.mean())
    record = {
        "what": "stack of the sentiment pair model and the relevance model (T-169, docs/sentiment_training.md)",
        "classes": list(CLASSES),
        "features": [f"log p_{c}" for c in CLASSES] + ["logit relevance"],
        "coef": lr.coef_.tolist(),
        "intercept": lr.intercept_.tolist(),
        "fitted_on": "out-of-fold scores of the 5-fold x 2 cross-validation over 809 rows, both repeats, weighted by pool size / drawn",
        "rows": len(keys),
        "regularisation_C": 1.0,
        "class_order_of_coef": [CLASSES[int(c)] for c in lr.classes_],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: coef {np.round(lr.coef_, 3).tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("relevance")
    a.add_argument("--out", type=Path, default=Path("models/relevance_final"))
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--order-of", type=Path, default=None)
    b = sub.add_parser("stack")
    b.add_argument("--cv", type=Path, default=Path("data/sentiment_pair_cv_all"))
    b.add_argument(
        "--oof", type=Path, default=Path("data/sentiment_pair_relevance_cv/oof_relevance.json")
    )
    b.add_argument("--out", type=Path, default=Path("config/sentiment_stack.json"))
    b.add_argument(
        "--verify", action="store_true", help="print the sha256 of the two inputs and stop"
    )
    args = parser.parse_args()
    if args.cmd == "relevance":
        relevance(args.out, args.seed, args.order_of)
    elif args.verify:
        oof_hash, predictions_hash = input_hashes(args.cv, args.oof)
        print(f"oof file {oof_hash}\npredictions {predictions_hash}")
    else:
        stack(args.cv, args.oof, args.out)


if __name__ == "__main__":
    main()
