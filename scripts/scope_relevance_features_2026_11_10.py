"""T-147, the owner's route (2026-11-10): the scope from the companies at the top of the article and how relevant each is.

    uv run python scripts/scope_relevance_features_2026_11_10.py [--chars 800] [--relevance models/relevance_final] \
        [--ner gamug/sec-bert-finer-ord-ner] [--cache data/scope_relevance_features.json]

For every body of ``gold/category_gold_v2.csv`` (284, train) and ``gold/category_gold_v2_test.csv`` (200, held-out): take
the title and the first ``--chars`` characters, run the NER model on that fragment, match the spans to universe tickers
(``gold/universe.csv``, ``news_nlp.triage``), score the (at most 6) most mentioned companies and the ticker the crawl
attached with the relevance model, and keep the top 4 scores. A small classifier on those numbers (logistic regression,
fixed in advance) predicts ``single`` / ``multi`` / ``sector`` / ``out_of_scope``. It is trained on the 284 and scored on the
200 the way ``scope_title_baseline_2026_11_10.py`` scores its models (weighted to the corpus, same four classes), next to the
rules and the title model, and the seconds per body of each stage are printed (the cost at scale).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import scope_title_baseline_2026_11_10 as base

from news_nlp import triage
from news_nlp.triage import Company

csv.field_size_limit(sys.maxsize)
GOLD = Path("gold")
TOP = 4  # scores kept
CANDIDATES = 6  # companies scored per body (the most mentioned)
FEATURES = (
    *[f"top{i}" for i in range(1, TOP + 1)],
    "gap12",
    "n_candidates",
    "n_over_half",
    "mentions",
    "n_org",
    "crawl_rel",
    "crawl_in_candidates",
)


def fragment(row: dict[str, str], chars: int) -> str:
    return f"{row['title']}. {row['text'][:chars]}"


def run_models(
    rows: list[dict[str, str]], chars: int, ner_repo: str, rel_path: str
) -> tuple[list[dict[str, float]], dict[str, float]]:
    import torch  # noqa: PLC0415 -- the heavy imports only where the models run
    from transformers import (  # noqa: PLC0415
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    from ner_stage import merge_bio_predictions  # noqa: PLC0415

    pinned = json.loads(Path("config/pipeline_models.json").read_text(encoding="utf-8"))["ner"]
    ner_tok = AutoTokenizer.from_pretrained(ner_repo, revision=pinned["revision"])
    ner = (
        AutoModelForTokenClassification.from_pretrained(ner_repo, revision=pinned["revision"])
        .cuda()
        .eval()
    )
    rel_tok = AutoTokenizer.from_pretrained(rel_path)
    rel = (
        AutoModelForSequenceClassification.from_pretrained(rel_path, dtype=torch.float32)
        .cuda()
        .eval()
    )
    universe = base.read(GOLD / "universe.csv")
    names = {r["ticker"]: r["company"] for r in universe}
    resolver = triage.CompanyResolver(
        [Company(r["ticker"], r["company"], r["gics_sector"], "") for r in universe]
    )
    id2label = {int(k): v for k, v in ner.config.id2label.items()}
    out: list[dict[str, float]] = []
    seconds = {"ner": 0.0, "relevance": 0.0}
    for row in rows:
        text = fragment(row, chars)
        t0 = time.perf_counter()
        enc = ner_tok(
            text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        word_ids = enc.word_ids(0)
        with torch.no_grad():
            probs = torch.softmax(ner(**{k: v.cuda() for k, v in enc.items()}).logits, -1)[0].cpu()
        spans = [
            text[e["start_char"] : e["end_char"]]
            for e in merge_bio_predictions(
                probs.argmax(-1).tolist(), word_ids, offsets, probs.tolist(), id2label
            )
            if e["entity_type"] == "ORG"
        ]
        mentions = resolver.resolve(text, spans)
        seconds["ner"] += time.perf_counter() - t0
        crawl = row["ticker"]
        cands = [t for t, _ in mentions.most_common(CANDIDATES)]
        t0 = time.perf_counter()
        scores: dict[str, float] = {}
        for ticker in dict.fromkeys([*cands, crawl]):
            name = names.get(ticker)
            if not name:
                continue
            x = rel_tok(
                text,
                f"This text is mainly about {name}.",
                truncation="only_first",
                max_length=320,
                return_tensors="pt",
            )
            with torch.no_grad():
                scores[ticker] = torch.softmax(
                    rel(**{k: v.cuda() for k, v in x.items()}).logits, -1
                )[0, 0].item()
        seconds["relevance"] += time.perf_counter() - t0
        top = sorted((scores[t] for t in cands if t in scores), reverse=True)
        padded = [*top[:TOP], *[0.0] * (TOP - len(top[:TOP]))]
        out.append(
            {
                **{f"top{i + 1}": padded[i] for i in range(TOP)},
                "gap12": padded[0] - padded[1],
                "n_candidates": float(len(cands)),
                "n_over_half": float(sum(s > 0.5 for s in top)),  # noqa: PLR2004 -- the score's midpoint
                "mentions": float(sum(mentions.values())),
                "n_org": float(len(spans)),
                "crawl_rel": scores.get(crawl, 0.0),
                "crawl_in_candidates": float(crawl in cands),
            }
        )
    seconds = {k: v / len(rows) for k, v in seconds.items()}
    return out, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chars", type=int, default=800)
    parser.add_argument("--relevance", default="models/relevance_final")
    parser.add_argument("--ner", default="gamug/sec-bert-finer-ord-ner")
    parser.add_argument("--cache", type=Path, default=Path("data/scope_relevance_features.json"))
    args = parser.parse_args()
    sets = {}
    for name, marks, texts in (
        ("train", "category_gold_v2.csv", "category_gold_v2_texts.csv"),
        ("test", "category_gold_v2_test.csv", "category_gold_v2_test_texts.csv"),
    ):
        text_of = {(r["article_id"], r["ticker"]): r for r in base.read(GOLD / "texts" / texts)}
        sets[name] = [
            {**r, **text_of[(r["article_id"], r["ticker"])]} for r in base.read(GOLD / marks)
        ]
    key = f"{args.chars}|{args.relevance}"
    if args.cache.exists() and json.loads(args.cache.read_text(encoding="utf-8")).get("key") == key:
        cached = json.loads(args.cache.read_text(encoding="utf-8"))
        feats, seconds = cached["features"], cached["seconds"]
    else:
        rows = sets["train"] + sets["test"]
        got, seconds = run_models(rows, args.chars, args.ner, args.relevance)
        feats = {"train": got[: len(sets["train"])], "test": got[len(sets["train"]) :]}
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(
            json.dumps({"key": key, "features": feats, "seconds": seconds}), encoding="utf-8"
        )
    print(
        f"seconds per body: NER {seconds['ner']:.3f}, relevance {seconds['relevance']:.3f} (chars {args.chars})"
    )
    x_tr = np.array([[f[n] for n in FEATURES] for f in feats["train"]])
    x_te = np.array([[f[n] for n in FEATURES] for f in feats["test"]])
    y_tr = [base.merged(r["scope"]) for r in sets["train"]]
    scaler = StandardScaler().fit(x_tr)
    model = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(x_tr), y_tr)
    pred = [str(p) for p in model.predict(scaler.transform(x_te))]
    test = sets["test"]
    rules = [base.merged(triage.RULE_SCOPE_TO_GOLD[r["stratum"]]) for r in test]
    for name, p in (("relevance features (LR)", pred), ("rules triage", rules)):
        print("\n".join(base.lines(name, test, p)), end="\n\n")
    labels = base.LABELS
    count = Counter((base.merged(r["scope"]), p) for r, p in zip(test, pred, strict=True))
    print("confusion matrix (plain counts), rows = gold, columns = predicted", labels)
    for gold in labels:
        print(f"{gold:>14}", [count[(gold, p)] for p in labels])


if __name__ == "__main__":
    main()
