#!/usr/bin/env python
"""One-shot: evaluate all three sentiment candidates (v2 published, v3
downsampled retrain, v4 class-weighted retrain) with the same, complete
metric set -- accuracy_ovr/precision/recall/F1 per class, plus overall
accuracy/macro F1 -- so every model in the comparison has a real, computed
number in every cell instead of "not recorded" gaps for the ones whose
original publish/training run predates a metric being added.

Evaluation-only, no training: loads each checkpoint, runs it over its own
already-established test set (v2/v4 share the same n=579 unbalanced-pool
test split; v3 uses its own n=386 rebalanced-pool test split) and the
shared n=100 idiom_probe, and reports train_sentiment.make_compute_metrics()'s
full metric set for each.

Requires network access to fetch v2 (gamug/FinBERT-financial-news) from the
Hub at its pinned revision (pipeline.MODEL_REVISIONS). v3/v4 are read from
their local saved checkpoints (models/finbert-financial-news,
models/finbert-financial-news-weighted) -- both must already exist (run
train_sentiment.py and train_sentiment.py --weighted first).

Usage:
    uv run python scripts/evaluate_sentiment_candidates_2026_09_15.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from pipeline import MODEL_REVISIONS, SENTIMENT_MODEL
from train_sentiment import (
    BALANCED_DATA_PATH,
    DATA_PATH,
    IDIOM_AUGMENT_PATH,
    IDIOM_PROBE_PATH,
    OUTPUT_DIR,
    OUTPUT_DIR_WEIGHTED,
    load_labeled_sentences,
    make_compute_metrics,
    stratified_split,
)

load_dotenv()


def evaluate_checkpoint(
    model_name_or_path: str,
    revision: str | None,
    test_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, revision=revision
    )

    def tokenize(batch: dict[str, Any]) -> Any:
        return tokenizer(batch["text"], truncation=True, max_length=128)

    test_ds = Dataset.from_list(test_rows).map(tokenize, batched=True, remove_columns=["text"])
    probe_ds = Dataset.from_list(probe_rows).map(tokenize, batched=True, remove_columns=["text"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir=tmp_dir, per_device_eval_batch_size=32, report_to=[]),
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            processing_class=tokenizer,
            compute_metrics=make_compute_metrics(),
        )
        test_metrics = trainer.evaluate(test_ds)
        probe_metrics = trainer.evaluate(probe_ds, metric_key_prefix="idiom_probe")
    return {"test_metrics": test_metrics, "idiom_probe_metrics": probe_metrics}


def main() -> None:
    # v2 / v4's shared test set + the idiom probe every candidate shares.
    unbalanced_rows = load_labeled_sentences(DATA_PATH) + load_labeled_sentences(IDIOM_AUGMENT_PATH)
    unbalanced_splits = stratified_split(unbalanced_rows)
    probe_rows = load_labeled_sentences(IDIOM_PROBE_PATH)

    # v3's own (smaller, rebalanced-pool) test set.
    balanced_rows = load_labeled_sentences(BALANCED_DATA_PATH)
    balanced_splits = stratified_split(balanced_rows)

    results: dict[str, Any] = {}

    print(f"=== v2 (published): {SENTIMENT_MODEL} @ {MODEL_REVISIONS[SENTIMENT_MODEL]} ===")
    results["v2_published"] = evaluate_checkpoint(
        SENTIMENT_MODEL, MODEL_REVISIONS[SENTIMENT_MODEL], unbalanced_splits["test"], probe_rows
    )

    print(f"\n=== v3 (downsampled retrain): {OUTPUT_DIR} ===")
    results["v3_downsampled"] = evaluate_checkpoint(
        OUTPUT_DIR, None, balanced_splits["test"], probe_rows
    )

    print(f"\n=== v4 (class-weighted retrain): {OUTPUT_DIR_WEIGHTED} ===")
    results["v4_weighted"] = evaluate_checkpoint(
        OUTPUT_DIR_WEIGHTED, None, unbalanced_splits["test"], probe_rows
    )

    out_path = Path("data/sentiment_finetune/candidates_comparison_2026_09_15.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved full comparison to {out_path}")

    for name, r in results.items():
        print(f"\n--- {name} ---")
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
