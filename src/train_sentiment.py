"""One-time fine-tuning: continue-training ProsusAI/finbert on a fresh,
in-domain sentence-level dataset labeled by an LLM from this project's own
real financial-news corpus (`scripts/label_sentiment_sentences_2026_09_13.py`)
-- not FinBERT's original 2014 Financial-PhraseBank-only training data
(Nordic-listed companies, no crypto/modern-instrument vocabulary; see
docs/evaluation.md's 2026-09-13 sentiment follow-up for the real-data
evidence this gap causes: FinBERT missing "crushed" as a positive idiom,
missing that import restrictions can be bullish for the specific company
they favor, etc.).

Continues fine-tuning from the ProsusAI/finbert checkpoint itself (not
vanilla bert-base) -- this is domain *refresh*, not a from-scratch
retrain: the label space (positive/negative/neutral, investor/price-impact
framing) is unchanged, only the training sentences are new and more
current/diverse.

Run once, offline, before publishing to the Hugging Face Hub (see
scripts/publish_finbert_financial_news_2026_09_13.py). Not part of
run_pipeline.py.

Known limitation, disclosed on the resulting model's card: the training
(and held-out test) labels are LLM-generated (silver-standard), not
human-annotated ground truth -- see the labeling script's own docstring.
"""

import json
import random
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "ProsusAI/finbert"
DATA_PATH = Path("data/sentiment_finetune/labeled_sentences.jsonl")
OUTPUT_DIR = "models/finbert-financial-news"
METRICS_OUTPUT = Path("data/sentiment_finetune/test_metrics.json")

# Matches ProsusAI/finbert's own config.id2label/label2id exactly -- this is
# a continued fine-tune, not a fresh label space.
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# 80/10/10, stratified per label so validation/test aren't accidentally
# skewed by whatever class happened to be more common in this draw.
_VAL_FRAC = 0.1
_TEST_FRAC = 0.1
_SEED = 42


def load_labeled_sentences(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rows.append({"text": row["sentence"], "label": LABEL2ID[row["label"]]})
    return rows


def stratified_split(
    rows: list[dict[str, Any]], seed: int = _SEED
) -> dict[str, list[dict[str, Any]]]:
    """Split rows into train/validation/test, stratified per label so each
    split's class balance matches the full dataset's -- important here since
    real financial news skews neutral/positive and a plain random split
    could leave the test set with very few negative examples."""
    rng = random.Random(seed)  # noqa: S311 -- split assignment, not cryptography
    by_label: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for label_rows in by_label.values():
        shuffled = label_rows[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(n * _TEST_FRAC))
        n_val = max(1, int(n * _VAL_FRAC))
        splits["test"].extend(shuffled[:n_test])
        splits["validation"].extend(shuffled[n_test : n_test + n_val])
        splits["train"].extend(shuffled[n_test + n_val :])

    for name, split_rows in splits.items():
        rng.shuffle(split_rows)
        print(f"{name}: {len(split_rows)} rows")
    return splits


def make_compute_metrics() -> Any:
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        macro_f1 = f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
        per_class_f1 = f1.compute(predictions=predictions, references=labels, average=None)["f1"]
        per_class_p = precision.compute(predictions=predictions, references=labels, average=None)[
            "precision"
        ]
        per_class_r = recall.compute(predictions=predictions, references=labels, average=None)[
            "recall"
        ]
        result = {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "macro_f1": macro_f1,
        }
        for label_id, label_name in ID2LABEL.items():
            result[f"f1_{label_name}"] = per_class_f1[label_id]
            result[f"precision_{label_name}"] = per_class_p[label_id]
            result[f"recall_{label_name}"] = per_class_r[label_id]
        return result

    return compute_metrics


def main() -> None:
    rows = load_labeled_sentences(DATA_PATH)
    print(f"Loaded {len(rows)} labeled sentences from {DATA_PATH}")
    splits = stratified_split(rows)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, id2label=ID2LABEL, label2id=LABEL2ID
    )

    def tokenize(batch: dict[str, Any]) -> Any:
        return tokenizer(batch["text"], truncation=True, max_length=128)

    ds = {name: Dataset.from_list(split_rows) for name, split_rows in splits.items()}
    tokenized_ds = {
        name: split.map(tokenize, batched=True, remove_columns=["text"])
        for name, split in ds.items()
    }

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=2,
        logging_steps=50,
        report_to=[],
        seed=_SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(),
    )

    trainer.train()

    print("\n=== Test set evaluation ===")
    test_metrics = trainer.evaluate(tokenized_ds["test"])
    print(test_metrics)

    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(
        json.dumps(
            {
                "test_metrics": test_metrics,
                "dataset_sizes": {name: len(split_rows) for name, split_rows in splits.items()},
                "base_model": MODEL_NAME,
            },
            indent=2,
        )
    )
    print(f"Saved test metrics to {METRICS_OUTPUT}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved fine-tuned model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
