"""Train the attribution model of step 4 (Work item 18, T-148b3, D27).

    uv run scripts/train_attribution_2026_11_08.py [--out data/models/attribution_2026_11_08]
        [--epochs 2] [--max-pairs 80000] [--seed 20261108] [--limit-bodies N]
    uv run scripts/train_attribution_2026_11_08.py --eval-only data/models/attribution_2026_11_08
    uv run scripts/train_attribution_2026_11_08.py --abstention data/models/attribution_2026_11_08

(The date in the names is the one the run was planned for; the code and the first model were
built on 2026-11-07.) Paths are relative to the repository root, wherever the script is run from.

The training set is ``gold/company_segments_train.csv`` (positions and tickers; the text is
rebuilt from SOURCE and each body's hash is checked). **Trains on the ``lead`` paragraphs only**
(D27): the run stops unless the owner's label check (``gold/segment_check_gold.csv``) gives
``gold_metrics.train_on_lead_only`` true, so the rule is applied, not remembered. The rows used
are counted and printed. ``val`` bodies (1 in 10, by hash) are never trained on. The model is
reported there on pairs (a sentence's right company against one wrong one) against the
mention-and-recency baseline (the candidate mentioned in the previous sentence, read by the
resolver; also the weaker exact legal name): all pairs, 2022 on and before apart, first sentences
of a paragraph (where the company changes) apart, and the sentences that name no universe company
(the only ones scored at run time, and without the ``[COMPANY]`` mask). It is also reported on
the run-time decision for those sentences: every company of the body as candidate, the share
given to the right company, to a wrong one and to nobody, against the baseline's.

A ``val`` body is left out when 80% of a sample of its 8-word shingles are in one train body
(copies of a story, wherever the headline or the ending differ). A copy rewritten throughout is
not caught. A key on the start and the end of a body was tried first, then 30% of the shingles
of all the train bodies together: what they matched were footers, a shared closing paragraph and
stories of one template (a Zacks earnings note, a "Fast Money" disclosure), not copies.

``--abstention DIR`` (no training, no ``val``) runs the model and the baseline over the 30
faithfulness bodies (``--bodies``) and prints, for the sentences that name no universe company,
how many each gives to nobody, one company or several, with a seeded sample of the model's choices.
``--eval-only DIR`` scores an already trained model on ``val`` and trains nothing; it keeps the
training-time report in ``metrics.json`` and adds the evaluation, with the hash of the files it
scored. The figure that decides is the owner's faithfulness check, not this one.

Writes the model and ``metrics.json`` in ``--out``. It publishes nothing and pins nothing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import attribution_baseline, company_texts, gold_metrics, retag, triage
from news_nlp import attribution_model as am
from news_nlp.triage import CompanyResolver

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "gold/company_segments_train.csv"
CHECK = ROOT / "gold/segment_check_gold.csv"
PIPELINE_MODELS = ROOT / "config/pipeline_models.json"
LATE_FROM = 2022


def require_lead_only(check: Path = CHECK) -> float:
    """D27's rule as a condition of the run; the share of ``continuation`` labels right."""
    rows = gold_metrics.read_marked(
        check, "correct", ("yes", "no"), required=("article_id", "year", "kind")
    )
    report = gold_metrics.label_check_report(rows)
    if not gold_metrics.train_on_lead_only(report):
        sys.exit(
            "the label check does not give 'train on lead only' (D27): this script trains on lead rows only"
        )
    return report["continuation"].point


SHINGLE = 8  # words
SAMPLE = 8  # one shingle in this many is kept
COPY_SHARE = 0.8  # of a body's kept shingles found in ONE train body: a copy


def shingles(body: str) -> set[int]:
    """A sample of the 8-word shingles of a body (by hash, so the same ones are kept in every
    body): two copies of a story share most of them wherever the headline or the ending differ,
    and boilerplate that two stories share (a footer, a disclosure) is a few of many. A body of
    fewer than :data:`SHINGLE` words has none, so it is never taken for a copy."""
    words = re.sub(r"[^a-z0-9 ]+", " ", body.lower()).split()
    hashes = (
        zlib.crc32(" ".join(words[i : i + SHINGLE]).encode())
        for i in range(len(words) - SHINGLE + 1)
    )
    return {h for h in hashes if h % SAMPLE == 0}


def is_copy(own: set[int], train: list[set[int]]) -> bool:
    """Whether :data:`COPY_SHARE` of the sampled shingles of a body are in a single train body.
    One body, not all of them together: stories of one template (a market wrap, a Zacks earnings
    note, a trader's disclosure) share a third of their shingles with the union of many."""
    return bool(own) and any(len(own & other) / len(own) >= COPY_SHARE for other in train)


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
    keys: list[set[int]],
    runtime: list[am.Decision] | None = None,
) -> tuple[list[am.Example], dict[str, int]]:
    """The examples of the bodies of *split*; with *runtime*, the run-time decisions of those
    bodies are appended to it (:func:`news_nlp.attribution_model.decisions`). *keys* are the :func:`shingles` of each train body: a train body adds its own, a ``val``
    body that :func:`is_copy` of them is left out and counted."""
    conn = db.connect_pipeline()
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    out: list[am.Example] = []
    counts = {"bodies": 0, "lead paragraphs": 0, "hash mismatch": 0, "near-duplicate": 0}
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
        own = shingles(body)
        if split == "train":
            keys.append(own)
        elif is_copy(own, keys):
            counts["near-duplicate"] += 1
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
        if runtime is not None:
            runtime += am.decisions(body, labels, resolver=resolver, names=names)
    return out, counts


def evaluate(
    examples: list[am.Example], scores: list[float], resolver: CompanyResolver
) -> dict[str, dict[str, float]]:
    """The model and two baselines on the same pairs: all, 2022 on, before 2022, first sentences,
    and the sentences that name no universe company (the run-time population)."""
    base = [am.named_in_context(e.pair, resolver) for e in examples]
    exact = [am.exact_name_in_context(e.pair) for e in examples]
    views = {
        "all": lambda e: True,
        "2022+": lambda e: e.year >= LATE_FROM,
        "before 2022": lambda e: e.year < LATE_FROM,
        "first sentences": lambda e: e.first,
        "unnamed sentences": lambda e: not e.named,
    }
    out: dict[str, dict[str, float]] = {}
    for name, keep in views.items():
        idx = [i for i, e in enumerate(examples) if keep(e)]
        sub = [examples[i] for i in idx]
        for who, values in (("model", scores), ("baseline", base), ("exact name", exact)):
            for metric, value in am.pair_accuracy(sub, [values[i] for i in idx]).items():
                out.setdefault(f"{name} {who}", {})[metric] = value
    return out


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def provenance(continuation: float) -> dict[str, object]:
    """What a pinned model needs to be reproduced: the code, the training set, the label check."""
    return {
        "code": retag.code_revision(*am.CODE_MODULES),
        "training_set": digest(GOLD),
        "label_check": {
            "file": digest(CHECK),
            "continuation": continuation,
            "train_on_lead_only": True,
        },
    }


def runtime_decision(scorer: am.Scorer, runtime: list[am.Decision]) -> dict[str, float]:
    """The run-time decision on the unnamed sentences of ``val``, printed and returned."""
    flat = scorer.score([p for d in runtime for p in d.pairs])
    rows: list[list[float]] = []
    at = 0
    for d in runtime:
        rows.append(flat[at : at + len(d.pairs)])
        at += len(d.pairs)
    decision = am.decision_report(runtime, rows)
    print("run-time decision on the unnamed sentences of val:")
    for name, value in decision.items():
        print(f"{name:>30}: {value:.3f}")
    return decision


def add_evaluation(
    out: Path,
    vcount: dict[str, int],
    report: dict[str, dict[str, float]],
    decision: dict[str, float],
    continuation: float,
) -> None:
    """Keep the training-time figures of ``metrics.json`` and add this evaluation, tied to the
    hash of the model files it scored."""
    path = out / "metrics.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved.setdefault("report_at_training", {"val": saved["val"], "report": saved["report"]})
    saved["eval"] = {
        "model_files": am.directory_identity(out),
        "val": vcount,
        "report": report,
        "decision": decision,
        "provenance": provenance(continuation),
    }
    path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print("evaluation added to", path)


SCORE_BANDS = (
    (0.1, "0 to 0.1"),
    (0.3, "0.1 to 0.3"),
    (0.5, "0.3 to 0.5 (nobody)"),
    (0.6, "0.5 to 0.6 (given)"),
    (0.8, "0.6 to 0.8 (given)"),
    (1.01, "0.8 to 1 (given)"),
)


def band(score: float | None) -> str:
    """The band of :data:`SCORE_BANDS` a best score falls in."""
    if score is None:
        return "no candidate"
    return next(label for top, label in SCORE_BANDS if score < top)


def abstention(
    bodies: dict[int, str],
    resolver: CompanyResolver,
    names: dict[str, str],
    scorer: am.Scorer,
    seed: int = 1,
    sample: int = 5,
) -> dict[str, object]:
    """Who each extractor gives the sentences that name no universe company to, on *bodies* (the
    faithfulness bodies): counts of sentences given to nobody, one company or several, for the
    baseline and the model, the model's best score for those sentences in bands (``scores``) and
    the share of the ones it gave to a company whose chosen company is named in the previous
    sentence, and a seeded sample of its choices (``article``, ``sentence``, the previous
    sentence, the companies it chose, their ``score`` and the companies the body names). The same
    *seed* gives the same sample; the command's default is the training seed."""
    counts = {f"{who} {k}": 0 for who in ("baseline", "model") for k in ("none", "one", "several")}
    counts["sentences"] = counts["named"] = 0
    nobody: list[dict[str, object]] = []
    given: list[dict[str, object]] = []
    bands = {label: 0 for _, label in SCORE_BANDS}
    bands["no candidate"] = 0
    recent = 0

    def kind(tickers: tuple[str, ...]) -> str:
        return "none" if not tickers else "one" if len(tickers) == 1 else "several"

    for article, body in sorted(bodies.items()):
        base = attribution_baseline.attribute_sentences(body, resolver)
        model = am.scored_sentences(body, resolver, names, scorer)
        companies = sorted(resolver.resolve(body, skip_attribution=False))
        for at, ((sentence, from_base), (_, chosen, score)) in enumerate(
            zip(base, model, strict=True)
        ):
            counts["sentences"] += 1
            if resolver.resolve(sentence, skip_attribution=False):
                counts["named"] += 1
                continue
            counts[f"baseline {kind(from_base)}"] += 1
            counts[f"model {kind(chosen)}"] += 1
            previous = base[at - 1][0] if at else ""
            bands[band(score)] += 1
            if chosen and set(chosen) & set(resolver.resolve(previous, skip_attribution=False)):
                recent += 1
            row = {
                "article": article,
                "sentence": sentence,
                "previous": previous,
                "chosen": chosen,
                "score": None if score is None else round(score, 3),
                "body names": companies,
            }
            (given if chosen else nobody).append(row)
    rng = random.Random(seed)  # noqa: S311 -- a reproducible sample
    return {
        "counts": counts,
        "scores": bands,
        "given, chosen company named in the previous sentence": recent,
        "nobody": rng.sample(nobody, min(sample, len(nobody))),
        "given": rng.sample(given, min(sample, len(given))),
    }


def run_abstention(args: argparse.Namespace) -> None:
    """``--abstention DIR``: print :func:`abstention` for a trained model on the faithfulness bodies."""
    companies = retag_cli.companies()
    names = {c.ticker: c.name for c in companies}
    ids = [
        int(row["article_id"])
        for row in csv.DictReader(args.bodies.open(encoding="utf-8-sig", newline=""))
    ]
    conn = db.connect_pipeline()
    bodies = {}
    for article in ids:
        src = conn.execute("SELECT body_text FROM source.articles WHERE id = ?", (article,))
        row = src.fetchone()
        bodies[article] = (row["body_text"] if row else "") or ""
    scorer = am.TransformerScorer(str(args.abstention))
    result = abstention(bodies, CompanyResolver(companies), names, scorer, args.seed)
    for key in ("counts", "scores"):
        print(key, json.dumps(result[key], indent=2))
    print(
        "given, chosen company named in the previous sentence:",
        result["given, chosen company named in the previous sentence"],
    )
    for label in ("nobody", "given"):
        print(f"--- a sample of the sentences the model gave to: {label}")
        for item in result[label]:  # type: ignore[attr-defined]
            print(json.dumps(item, ensure_ascii=False))


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
    parser.add_argument("--out", type=Path, default=ROOT / "data/models/attribution_2026_11_08")
    parser.add_argument("--eval-only", type=Path, metavar="DIR", help="score this trained model")
    parser.add_argument(
        "--abstention",
        type=Path,
        metavar="DIR",
        help="who the model gives the unnamed sentences of",
    )
    parser.add_argument("--bodies", type=Path, default=ROOT / "data/faithfulness_bodies.csv")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--limit-bodies", type=int, default=None)
    args = parser.parse_args()
    if args.abstention:
        run_abstention(args)
        return
    continuation = require_lead_only()

    config = json.loads(PIPELINE_MODELS.read_text(encoding="utf-8"))["sentiment"]
    companies = retag_cli.companies()
    resolver = CompanyResolver(companies)
    names = {c.ticker: c.name for c in companies}
    by_article = read_rows()
    keys: list[set[int]] = []
    train, tcount = examples_of(
        by_article, "train", resolver, names, args.seed, args.limit_bodies, keys
    )
    runtime: list[am.Decision] = []
    val, vcount = examples_of(
        by_article, "val", resolver, names, args.seed + 1, args.limit_bodies, keys, runtime
    )
    if tcount["hash mismatch"] or vcount["hash mismatch"]:
        sys.exit(f"bodies whose text changed since the set was built: {tcount}, {vcount}")
    random.Random(args.seed).shuffle(train)  # noqa: S311
    total = len(train)
    train = train[: args.max_pairs]
    print(
        f"train: {tcount['bodies']} bodies, {tcount['lead paragraphs']} lead paragraphs (continuation rows are not used, D27)"
    )
    print(
        f"       {total} pairs, {len(train)} used (--max-pairs); val: {vcount['bodies']} bodies, "
        f"{len(val)} pairs, {vcount['near-duplicate']} bodies left out as near-duplicates of train"
    )

    if args.eval_only:
        out = args.eval_only if args.eval_only.is_absolute() else ROOT / args.eval_only
        scorer = am.TransformerScorer(str(out))
        elapsed = 0.0
    else:
        out = args.out
        scorer, tok, model, elapsed = fit(train, config, args)
    scores = scorer.score([e.pair for e in val])
    report = evaluate(val, scores, resolver)
    for name, figures in report.items():
        print(
            f"{name:>30}: ranking {figures['ranking']:.3f}  at 0.5 {figures['threshold']:.3f}  "
            f"({int(figures['pairs'])} pairs)"
        )
    decision = runtime_decision(scorer, runtime)
    if args.eval_only:
        add_evaluation(out, vcount, report, decision, continuation)
        return
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "base": config,
                "args": {k: str(v) for k, v in vars(args).items()},
                "train": tcount,
                "pairs_used": len(train),
                "val": vcount,
                "report": report,
                "decision": decision,
                "model_files": am.directory_identity(out),
                "seconds": round(elapsed),
                "provenance": provenance(continuation),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("model and metrics.json written to", out)


if __name__ == "__main__":
    main()
