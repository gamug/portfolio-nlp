"""Train the attribution model of step 4 (Work item 18, T-148b3, D27).

    uv run scripts/train_attribution_2026_11_08.py [--out data/models/attribution_2026_11_08]
        [--epochs 2] [--max-pairs 80000] [--seed 20261108] [--limit-bodies N]

The training set is ``gold/company_segments_train.csv`` (positions and tickers; the text is rebuilt
from SOURCE and each body's hash is checked). **Trains on the ``lead`` paragraphs only** (D27): the
run stops unless the owner's label check (``gold/segment_check_gold.csv``) gives
``gold_metrics.train_on_lead_only`` true, so the rule is applied, not remembered. The rows used are
counted and printed. ``val`` bodies (1 in 10, by hash) are never trained on; the model is reported
there against the mention-and-recency baseline (the candidate's name in the previous sentence) on
the same pairs, for 2022 on and before apart, and for the first sentence of a paragraph (where the
company changes) apart. The figure that decides is the owner's faithfulness check, not this one.

Writes the model and ``metrics.json`` in ``--out``. It publishes nothing and pins nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import attribution_model as am
from news_nlp import company_texts, gold_metrics, triage
from news_nlp.triage import CompanyResolver

GOLD = Path("gold/company_segments_train.csv")
CHECK = Path("gold/segment_check_gold.csv")
LATE_FROM = 2022


def require_lead_only(check: Path = CHECK) -> None:
    """D27's rule as a condition of the run."""
    rows = gold_metrics.read_marked(
        check, "correct", ("yes", "no"), required=("article_id", "year", "kind")
    )
    if not gold_metrics.train_on_lead_only(gold_metrics.label_check_report(rows)):
        sys.exit(
            "the label check does not give 'train on lead only' (D27): this script trains on lead rows only"
        )


def read_rows(path: Path = GOLD) -> dict[int, list[dict[str, str]]]:
    by_article: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            by_article[int(row["article_id"])].append(row)
    return by_article


def examples_of(
    by_article: dict[int, list[dict[str, str]]],
    split: str,
    resolver: CompanyResolver,
    names: dict[str, str],
    seed: int,
    limit: int | None,
) -> tuple[list[am.Example], dict[str, int]]:
    conn = db.connect_pipeline()
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    out: list[am.Example] = []
    counts = {"bodies": 0, "lead paragraphs": 0, "hash mismatch": 0}
    for article_id in sorted(by_article):
        rows = by_article[article_id]
        if rows[0]["split"] != split:
            continue
        if limit is not None and counts["bodies"] >= limit:
            break
        src = conn.execute(
            "SELECT body_text FROM source.articles WHERE id = ?", (article_id,)
        ).fetchone()
        body = (src["body_text"] if src else "") or ""
        if triage.body_hash(body) != rows[0]["body_hash"]:
            counts["hash mismatch"] += 1
            continue
        labels = [
            company_texts.SegmentLabel(int(r["segment_index"]), r["ticker"], r["kind"])
            for r in sorted(rows, key=lambda r: int(r["segment_index"]))
        ]
        counts["bodies"] += 1
        counts["lead paragraphs"] += sum(label.kind == "lead" for label in labels)
        out += am.training_examples(
            body,
            labels,
            article_id=article_id,
            year=int(rows[0]["year"]),
            resolver=resolver,
            names=names,
            rng=rng,
        )
    return out, counts


def evaluate(examples: list[am.Example], scores: list[float]) -> dict[str, dict[str, float]]:
    """The model and the baseline on the same pairs: all, 2022 on, before 2022, first sentences."""
    base = [am.named_in_context(e.pair) for e in examples]
    views = {
        "all": lambda e: True,
        "2022+": lambda e: e.year >= LATE_FROM,
        "before 2022": lambda e: e.year < LATE_FROM,
        "first sentences": lambda e: e.first,
    }
    out: dict[str, dict[str, float]] = {}
    for name, keep in views.items():
        idx = [i for i, e in enumerate(examples) if keep(e)]
        sub = [examples[i] for i in idx]
        for who, values in (("model", scores), ("baseline", base)):
            for metric, value in am.pair_accuracy(sub, [values[i] for i in idx]).items():
                out.setdefault(f"{name} {who}", {})[metric] = value
    return out


def fit(
    train: list[am.Example], config: dict[str, str], args: argparse.Namespace
) -> tuple[am.TransformerScorer, object, object, float]:
    """Fine-tune the pinned base on the pairs; the scorer wraps the trained model."""
    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"],
        revision=config["revision"],
        num_labels=2,
        id2label={0: "other", 1: "about"},
        label2id={"other": 0, "about": 1},
        ignore_mismatched_sizes=True,
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    schedule = get_linear_schedule_with_warmup(optimiser, 0, steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    scorer = am.TransformerScorer.__new__(am.TransformerScorer)
    scorer.torch, scorer.device, scorer.tokenizer, scorer.model = torch, device, tok, model
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        random.Random(args.seed + epoch).shuffle(train)  # noqa: S311
        loss_sum = 0.0
        for start in range(0, len(train), args.batch_size):
            chunk = train[start : start + args.batch_size]
            batch = {k: v.to(device) for k, v in scorer.encode([e.pair for e in chunk]).items()}
            batch["labels"] = torch.tensor([e.label for e in chunk], device=device)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            optimiser.zero_grad()
            loss_sum += loss.item() * len(chunk)
        model.eval()
        print(
            f"epoch {epoch + 1}: loss {loss_sum / len(train):.4f}, {time.time() - started:.0f}s",
            flush=True,
        )

    return scorer, tok, model, time.time() - started


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/models/attribution_2026_11_08"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--limit-bodies", type=int, default=None)
    args = parser.parse_args()
    require_lead_only()

    config = json.loads(Path("config/pipeline_models.json").read_text(encoding="utf-8"))[
        "sentiment"
    ]
    companies = retag_cli.companies()
    resolver = CompanyResolver(companies)
    names = {c.ticker: c.name for c in companies}
    by_article = read_rows()
    train, tcount = examples_of(by_article, "train", resolver, names, args.seed, args.limit_bodies)
    val, vcount = examples_of(by_article, "val", resolver, names, args.seed + 1, args.limit_bodies)
    if tcount["hash mismatch"] or vcount["hash mismatch"]:
        sys.exit(f"bodies whose text changed since the set was built: {tcount}, {vcount}")
    random.Random(args.seed).shuffle(train)  # noqa: S311
    total = len(train)
    train = train[: args.max_pairs]
    print(
        f"train: {tcount['bodies']} bodies, {tcount['lead paragraphs']} lead paragraphs (continuation rows are not used, D27)"
    )
    print(
        f"       {total} pairs, {len(train)} used (--max-pairs); val: {vcount['bodies']} bodies, {len(val)} pairs"
    )

    scorer, tok, model, elapsed = fit(train, config, args)
    scores = scorer.score([e.pair for e in val])
    report = evaluate(val, scores)
    for name, figures in report.items():
        print(
            f"{name:>28}: ranking {figures['ranking']:.3f}  at 0.5 {figures['threshold']:.3f}  ({int(figures['pairs'])} pairs)"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "base": config,
                "args": {k: str(v) for k, v in vars(args).items()},
                "train": tcount,
                "pairs_used": len(train),
                "val": vcount,
                "report": report,
                "seconds": round(elapsed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("model and metrics.json written to", args.out)


if __name__ == "__main__":
    main()
