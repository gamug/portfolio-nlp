"""T-147 end to end (owner 2026-11-10): the scope from two fine-tuned NLI models, measured once on the held-out 200.

    uv run python scripts/scope_end_to_end_2026_11_10.py [--chars 800] [--out data/scope_end_to_end_predictions.csv]

Two modes. Default (the first measurement, 2026-11-10): the models are trained on the 284 only and the 51 rare-scope bodies
(``gold/scope_enrich.csv``) are scored apart. ``--train-on-enrich`` (the **selected model**, whose figures are in
``docs/splitting_training.md``): the same models trained on 365 bodies (the 284 + the 51 + the 30 of ``gold/scope_hook.csv``);
the 200 stay held out and are scored, the 51 and the 30 are not. Needs a GPU.

Stage A is the category model (``deberta-v3-base-zeroshot-v2.0``, pinned) fine-tuned as an NLI over three topics (company /
sector / not_related); stage B is the same model fine-tuned on the company bodies as an NLI over single / multi; both read the
title + the first 800 characters, the relevance model's recipe (3 epochs, lr 2e-5, float32 weights, bf16), seed 1. Scored on the
held-out 200 the way the other scope scripts are (weighted to the corpus, four classes) with a paired gain over the rules triage.
The choices were fixed before the first run; nothing is tuned on the 200 here (they were seen in earlier experiments: see the
limitations in the doc). The weights are not saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import scope_nli_single_multi_2026_11_10 as sm
import scope_nli_topic_2026_11_10 as tp
import scope_relevance_features_2026_11_10 as feat
import scope_title_baseline_2026_11_10 as base

from news_nlp import gold_metrics, triage


def topic_of_scope(row: dict[str, str]) -> str:
    if row["scope"] in sm.CLASSES:
        return "company"
    return "sector" if row["scope"] == "sector" else "not_related"


def load_sets() -> dict[str, list[dict[str, str]]]:
    sets = {}
    for name, marks, texts in (
        ("train", "category_gold_v2.csv", "category_gold_v2_texts.csv"),
        ("test", "category_gold_v2_test.csv", "category_gold_v2_test_texts.csv"),
        ("enrich", "scope_enrich.csv", "scope_enrich_texts.csv"),
        ("hook", "scope_hook.csv", "scope_hook_texts.csv"),
    ):
        text_of = {
            (r["article_id"], r["ticker"]): r for r in base.read(feat.GOLD / "texts" / texts)
        }
        sets[name] = [
            {**r, **text_of[(r["article_id"], r["ticker"])]} for r in base.read(feat.GOLD / marks)
        ]
    return sets


def main() -> None:  # noqa: PLR0915 -- one linear measurement script
    import torch  # noqa: PLC0415 -- heavy imports only where the models run
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chars", type=int, default=800)
    parser.add_argument("--out", type=Path, default=Path("data/scope_end_to_end_predictions.csv"))
    parser.add_argument(
        "--train-on-enrich",
        action="store_true",
        help="add the 51 rare-scope and the 30 company-as-hook tags to the training set (the 200 stay held out)",
    )
    args = parser.parse_args()
    sets = load_sets()
    cfg = json.loads(Path("config/pipeline_models.json").read_text(encoding="utf-8"))["category"]
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])

    def load():
        return AutoModelForSequenceClassification.from_pretrained(
            cfg["model"], revision=cfg["revision"], dtype=torch.float32
        ).cuda()

    text = {n: [feat.fragment(r, args.chars) for r in rows] for n, rows in sets.items()}
    extra = (
        ("enrich", "hook") if args.train_on_enrich else ()
    )  # the 51 and the 30 company-as-hook tags
    train = sets["train"] + [r for n in extra for r in sets[n]]
    text["train"] = text["train"] + [t for n in extra for t in text[n]]
    topic_of = topic_of_scope
    # stage A
    model_a = load()
    tp.fine_tune(model_a, tok, text["train"], [topic_of(r) for r in train])
    a_pred = {
        n: [tp.TOPICS[i] for i in tp.scores(model_a, tok, text[n]).argmax(1)]
        for n in ("test", "enrich")
    }
    del model_a
    torch.cuda.empty_cache()
    # stage B: only the company bodies of the 284
    own = [i for i, r in enumerate(train) if r["scope"] in sm.CLASSES]
    model_b = load()
    sm.fine_tune(model_b, tok, [text["train"][i] for i in own], [train[i]["scope"] for i in own])
    b_pred = {
        n: [sm.CLASSES[i] for i in sm.scores(model_b, tok, text[n]).argmax(1)]
        for n in ("test", "enrich")
    }
    pred = {
        n: [
            b if a == "company" else ("sector" if a == "sector" else "out_of_scope")
            for a, b in zip(a_pred[n], b_pred[n], strict=True)
        ]
        for n in a_pred
    }
    # the held-out 200, weighted, next to the rules
    test = sets["test"]
    rules = [base.merged(triage.RULE_SCOPE_TO_GOLD[r["stratum"]]) for r in test]
    print("\n".join(base.lines("two fine-tuned NLI models", test, pred["test"])), end="\n\n")
    print("\n".join(base.lines("rules triage", test, rules)), end="\n\n")
    populations: dict[str, int] = {}
    for r in test:
        populations.setdefault(r["stratum"], int(r["frame"]))
    rows = sorted(
        (
            {
                "article_id": r["article_id"],
                "stratum": r["stratum"],
                "gold": base.merged(r["scope"]),
                "predicted": p,
                "baseline": q,
            }
            for r, p, q in zip(test, pred["test"], rules, strict=True)
        ),
        key=lambda r: int(r["article_id"]),
    )
    gain = gold_metrics.paired_stratified_gain(
        rows, populations, metric="agreement", stratum_key="stratum"
    )
    print(
        f"paired gain in weighted agreement over the rules: {gain} -> {gold_metrics.gain_verdict(gain)}"
    )
    from collections import Counter  # noqa: PLC0415

    count = Counter((base.merged(r["scope"]), p) for r, p in zip(test, pred["test"], strict=True))
    print(
        "confusion matrix of the held-out 200 (plain), rows = gold, columns = predicted",
        base.LABELS,
    )
    for g in base.LABELS:
        print(f"{g:>14}", [count[(g, p)] for p in base.LABELS])
    # the 51 rare-scope bodies: class-conditional recall, unweighted
    enrich = sets["enrich"]
    if args.train_on_enrich:
        print(
            f"\ntrained on {len(train)} bodies (284 + the 51 + the 30 hook tags); they are not scored. predictions -> {args.out}"
        )
        _write(args.out, sets, pred)
        return
    print("\nthe 51 bodies drawn for the rare scopes (class-conditional recall, unweighted):")
    for label in base.LABELS:
        got = [
            p
            for r, p in zip(enrich, pred["enrich"], strict=True)
            if base.merged(r["scope"]) == label
        ]
        print(f"{label:>14}: {sum(p == label for p in got)}/{len(got)}")
    # sector and out_of_scope pooled with the held-out 200: the 40 gold bodies of D33
    for label in ("sector", "out_of_scope"):
        golds = [(base.merged(r["scope"]), p) for r, p in zip(test, pred["test"], strict=True)] + [
            (base.merged(r["scope"]), p) for r, p in zip(enrich, pred["enrich"], strict=True)
        ]
        hit = sum(g == p == label for g, p in golds)
        total = sum(g == label for g, _ in golds)
        print(
            f"{label:>14} pooled (200 + 51, unweighted): {hit}/{total} = {hit / total:.1%} ({total} gold bodies)"
        )
    _write(args.out, sets, pred)
    print(f"predictions -> {args.out}")


def _write(out: Path, sets: dict[str, list[dict[str, str]]], pred: dict[str, list[str]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["set", "article_id", "gold", "predicted"])
        for n in ("test", "enrich"):
            for r, p in zip(sets[n], pred[n], strict=True):
                writer.writerow([n, r["article_id"], base.merged(r["scope"]), p])


if __name__ == "__main__":
    main()
