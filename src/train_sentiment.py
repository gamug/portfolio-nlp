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

If `data/sentiment_finetune/labeled_sentences_balanced_2026_09_14.jsonl`
exists (produced by the 2026-09-14 follow-up
`scripts/rebalance_sentiment_data_2026_09_14.py`, PLAN.md Work item 9 --
the base draw + idiom-augment pool was 56.1% neutral / 22.8% negative /
21.1% positive, never a deliberate target, so neutral was downsampled to
the larger minority class's size via TF-IDF-centroid representative
selection), it is used as the **entire** training pool as-is -- it already
has idiom_augment.jsonl merged in, so DATA_PATH/IDIOM_AUGMENT_PATH below
are *not* also loaded in that case (that would double-count the idiom rows).

Otherwise (that file absent), the original, unbalanced pool is used: if
`data/sentiment_finetune/idiom_augment.jsonl` exists (produced by the
2026-09-13 follow-up `scripts/mine_idiom_sentences_2026_09_13.py`, mined
after a spot-check of the first fine-tune found it still mislabeled
"crushed earnings" idioms), those rows are merged into the training pool
before the stratified split.

If `data/sentiment_finetune/idiom_probe.jsonl` exists (a held-out slice of
the idiom-mining pass, never trained on and untouched by the 2026-09-14
rebalance either way), the final model is *also* evaluated on it
separately and the result recorded alongside the regular test metrics --
a direct, targeted measurement of whether the idiom fix actually worked,
not just an aggregate-metric inference.

`--weighted` (PLAN.md Work item 9, second experiment): trains on the
*original, unbalanced* pool (DATA_PATH + IDIOM_AUGMENT_PATH merged --
BALANCED_DATA_PATH is ignored even if present) with an inverse-class-
frequency-weighted cross-entropy loss instead of downsampling neutral.
Unlike the downsample approach, no training sentence is discarded -- every
neutral example the model could have learned from is still seen, just
weighted down in the loss so the model isn't rewarded for defaulting to
the majority class. Weights are computed once from the *train* split's
own label counts after stratified_split (not the full pool's), so they
match what the model actually trains on. Writes to OUTPUT_DIR_WEIGHTED /
METRICS_OUTPUT_WEIGHTED, not the default paths -- doesn't overwrite the
2026-09-14 rebalanced-retrain artifacts, so both experiments' results stay
on disk side by side.

`--base-model` (2026-09-15, third experiment): swap the base checkpoint
that gets continue-fine-tuned, instead of another data-side intervention.
Motivated by precision_negative being stuck in a narrow ~0.50-0.51 band
across every FinBERT-based candidate tried so far (base, v1, v2, v4 --
docs/evaluation.md's 2026-09-15 follow-ups), and by the 2026-09-13
follow-up's confidence/margin-threshold finding: only 7% of predictions
have a thin top1-vs-top2 margin, median margin 0.90 on errors -- the
model is *confidently* wrong, not *hesitantly* wrong, so a threshold
gate was already ruled out there. That points at the base checkpoint's
vocabulary/pretraining rather than at training-data balance.
`nlpaueb/sec-bert-base` (already this project's
NER base checkpoint, `src/train_ner.py`; Loukas et al. 2022,
arXiv:2203.06482 -- SEC-BERT outperformed FinBERT on that paper's own
financial NER task) is domain-pretrained on 260,773 real SEC 10-K filings
with its own 30k financial-vocabulary WordPiece tokenizer, rather than
FinBERT's generic-BERT-derived vocabulary -- a real candidate fix if
precision_negative's stuck-ness traces to subword fragmentation on
financial entity names/terms rather than to the aggregation-level
multi-company misattribution already identified as a likely cause.
Same procedure/hyperparameters/data (defaults to the *original* unbalanced
pool, matching v2 exactly, for a clean base-model-only comparison --
combine with `--weighted` for the rebalanced-data variant once the base
model itself is assessed). Output paths are derived from the base model
name so this can't collide with any FinBERT run's saved artifacts.

Run once, offline, before publishing to the Hugging Face Hub (see
scripts/publish_finbert_financial_news_2026_09_13.py). Not part of
run_pipeline.py.

Known limitation, disclosed on the resulting model's card: the training
(and held-out test) labels are LLM-generated (silver-standard), not
human-annotated ground truth -- see the labeling script's own docstring.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
from datasets import Dataset
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from fti import TrainConfig, TrainedArtifact
from fti import Trainer as FtiTrainer

MODEL_NAME = "ProsusAI/finbert"
DATA_PATH = Path("data/sentiment_finetune/labeled_sentences.jsonl")
# 2026-09-13 crushed-earnings-idiom follow-up (optional -- merged in / probed
# only if present, see module docstring).
IDIOM_AUGMENT_PATH = Path("data/sentiment_finetune/idiom_augment.jsonl")
IDIOM_PROBE_PATH = Path("data/sentiment_finetune/idiom_probe.jsonl")
# 2026-09-14 rebalance follow-up (PLAN.md Work item 9): already-merged,
# already-rebalanced training pool -- preferred over DATA_PATH +
# IDIOM_AUGMENT_PATH when present, see module docstring.
BALANCED_DATA_PATH = Path("data/sentiment_finetune/labeled_sentences_balanced_2026_09_14.jsonl")
OUTPUT_DIR = "models/finbert-financial-news"
METRICS_OUTPUT = Path("data/sentiment_finetune/test_metrics.json")
# --weighted experiment (2026-09-14, PLAN.md Work item 9): separate output
# paths so this doesn't clobber the rebalanced-retrain's saved model/metrics.
OUTPUT_DIR_WEIGHTED = "models/finbert-financial-news-weighted"
METRICS_OUTPUT_WEIGHTED = Path("data/sentiment_finetune/test_metrics_weighted.json")

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
    rows: list[dict[str, Any]],
    seed: int = _SEED,
    test_frac: float = _TEST_FRAC,
    val_frac: float = _VAL_FRAC,
) -> dict[str, list[dict[str, Any]]]:
    """Split rows into train/validation/test, stratified per label so each
    split's class balance matches the full dataset's -- important here since
    real financial news skews neutral/positive and a plain random split
    could leave the test set with very few negative examples.

    ``test_frac``/``val_frac`` (added 2026-09-18, TASKS.md T-096,
    PLAN.md Work item 11) default to this module's own pre-existing 10/10
    split -- every existing call site's behavior is unchanged unless it
    now explicitly overrides them (e.g. via ``SentimentTrainConfig``)."""
    rng = random.Random(seed)  # noqa: S311 -- split assignment, not cryptography
    by_label: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for label_rows in by_label.values():
        shuffled = label_rows[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        splits["test"].extend(shuffled[:n_test])
        splits["validation"].extend(shuffled[n_test : n_test + n_val])
        splits["train"].extend(shuffled[n_test + n_val :])

    for name, split_rows in splits.items():
        rng.shuffle(split_rows)
        print(f"{name}: {len(split_rows)} rows")
    return splits


def compute_class_weights(train_rows: list[dict[str, Any]]) -> torch.Tensor:
    """Inverse-class-frequency weights from the train split's own label counts:
    weight_c = total / (num_classes * count_c). A class with half the average
    count gets ~2x the weight, so a mistake on it costs the loss ~2x as much --
    the same effect downsampling achieves by removing data, but without
    discarding any training example. Indexed by LABEL2ID's integer ids so it
    lines up with the model's logits/CrossEntropyLoss ordering directly."""
    counts = [0] * len(LABEL2ID)
    for row in train_rows:
        counts[row["label"]] += 1
    total = len(train_rows)
    num_classes = len(LABEL2ID)
    weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies compute_class_weights' per-class weights to
    the cross-entropy loss, instead of the plain unweighted loss Trainer uses
    by default. Nothing else about training changes."""

    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, len(LABEL2ID)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


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
            # One-vs-rest binary accuracy: "is this <label> or not", collapsing
            # the other two labels into a single negative class -- same
            # formula/naming as news_nlp.eval.metrics.aggregate_category's
            # accuracy_ovr_<slug>, for consistency across this project's model
            # evaluations (constitution.md AI behavior #12). Skews high when a
            # label is rare (dominated by true negatives) -- read alongside
            # precision/recall above, not instead of them.
            ovr_hits = [
                1.0 if (int(t) == label_id) == (int(p) == label_id) else 0.0
                for t, p in zip(labels, predictions, strict=True)
            ]
            result[f"accuracy_ovr_{label_name}"] = sum(ovr_hits) / len(ovr_hits)
        return result

    return compute_metrics


def load_training_pool(
    base_model: str, weighted: bool
) -> tuple[list[dict[str, Any]], str, str, Path]:
    """Load the right training pool for this run and the output paths that go
    with it.

    --weighted uses the original, unbalanced pool (DATA_PATH +
    IDIOM_AUGMENT_PATH) regardless of base_model. Otherwise, for the default
    base model (MODEL_NAME, i.e. no --base-model override) only, prefer
    BALANCED_DATA_PATH when present (see module docstring) -- a non-default
    base model always starts from the plain unbalanced pool unless --weighted
    is also passed, so a --base-model run defaults to the same data v2 used,
    for a clean base-model-only comparison; layering rebalancing on top is a
    deliberate, separate --weighted combination, not an accidental pickup of
    a file that happens to exist on disk from a different base model's run."""
    output_dir, metrics_output = paths_for_base_model(base_model, weighted)

    if weighted:
        rows = load_labeled_sentences(DATA_PATH)
        print(f"Loaded {len(rows)} labeled sentences from {DATA_PATH}")
        if IDIOM_AUGMENT_PATH.exists():
            augment_rows = load_labeled_sentences(IDIOM_AUGMENT_PATH)
            print(f"Merging {len(augment_rows)} idiom-augment sentences from {IDIOM_AUGMENT_PATH}")
            rows = rows + augment_rows
        print("--weighted: training on the ORIGINAL unbalanced pool with class-weighted loss")
        return rows, str(DATA_PATH), output_dir, metrics_output

    if base_model == MODEL_NAME and BALANCED_DATA_PATH.exists():
        # Already has idiom_augment.jsonl merged in and neutral downsampled --
        # load it whole, don't also merge IDIOM_AUGMENT_PATH (would double-count).
        rows = load_labeled_sentences(BALANCED_DATA_PATH)
        print(f"Loaded {len(rows)} labeled sentences from {BALANCED_DATA_PATH} (rebalanced pool)")
        return rows, str(BALANCED_DATA_PATH), output_dir, metrics_output

    rows = load_labeled_sentences(DATA_PATH)
    print(f"Loaded {len(rows)} labeled sentences from {DATA_PATH}")
    if IDIOM_AUGMENT_PATH.exists():
        augment_rows = load_labeled_sentences(IDIOM_AUGMENT_PATH)
        print(f"Merging {len(augment_rows)} idiom-augment sentences from {IDIOM_AUGMENT_PATH}")
        rows = rows + augment_rows
    return rows, str(DATA_PATH), output_dir, metrics_output


def paths_for_base_model(base_model: str, weighted: bool) -> tuple[str, Path]:
    """Output paths for a given base checkpoint. The default checkpoint
    (MODEL_NAME, i.e. no --base-model override) keeps the exact pre-existing
    OUTPUT_DIR/METRICS_OUTPUT (or their _WEIGHTED counterparts) so this is a
    no-op for every run this project has already made. Any other base model
    gets its own derived paths (models/<slug>-financial-sentiment[-weighted],
    data/sentiment_finetune/test_metrics_<slug>[_weighted].json) so it can
    never collide with a FinBERT run's saved artifacts."""
    if base_model == MODEL_NAME:
        return (
            (OUTPUT_DIR_WEIGHTED, METRICS_OUTPUT_WEIGHTED)
            if weighted
            else (OUTPUT_DIR, METRICS_OUTPUT)
        )
    slug = base_model.rsplit("/", maxsplit=1)[-1]
    suffix = "-weighted" if weighted else ""
    output_dir = str(Path("models") / f"{slug}-financial-sentiment{suffix}")
    metrics_output = (
        Path("data/sentiment_finetune")
        / f"test_metrics_{slug.replace('-', '_')}{suffix.replace('-', '_')}.json"
    )
    return output_dir, metrics_output


@dataclass(frozen=True)
class SentimentTrainConfig(TrainConfig):
    """`--weighted`/`--base-model` (module docstring) as a config object
    instead of argparse's `args_ns` -- same two knobs, same defaults.

    `split_seed`/`test_frac`/`val_frac` (added 2026-09-18, TASKS.md T-096,
    PLAN.md Work item 11) make `stratified_split`'s train/test setup
    genuinely config-driven -- previously `stratified_split(rows)` was
    always called with no arguments, silently relying on this module's own
    hardcoded `_SEED`/`_TEST_FRAC`/`_VAL_FRAC` constants with no way to
    override them from a config or CLI flag. Defaults match those exact
    constants, so every existing invocation's behavior is unchanged."""

    weighted: bool = False
    base_model: str = MODEL_NAME
    split_seed: int = _SEED
    test_frac: float = _TEST_FRAC
    val_frac: float = _VAL_FRAC


class SentimentTrainer(FtiTrainer[SentimentTrainConfig]):
    """Wraps this module's previous `main()` body (PLAN.md Work item 10 /
    TASKS.md T-083) -- not `sentiment_stage.SentimentFeature`, since
    training operates on raw labeled *sentences* (`tokenizer(batch["text"],
    truncation=True, max_length=128)` below), a fundamentally different
    input shape from inference's full-article chunking; reusing that
    `Feature` here would be forcing the wrong abstraction onto a shape it
    doesn't fit."""

    def train(self, config: SentimentTrainConfig) -> TrainedArtifact:
        base_model = config.base_model
        rows, training_data_path, output_dir, metrics_output = load_training_pool(
            base_model, config.weighted
        )
        splits = stratified_split(
            rows, seed=config.split_seed, test_frac=config.test_frac, val_frac=config.val_frac
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, id2label=ID2LABEL, label2id=LABEL2ID
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
            output_dir=output_dir,
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

        class_weights = None
        if config.weighted:
            class_weights = compute_class_weights(splits["train"])
            print(
                "Class weights (inverse train-split frequency, "
                f"{ {ID2LABEL[i]: round(w.item(), 3) for i, w in enumerate(class_weights)} }):"
            )
            trainer: Trainer = WeightedLossTrainer(
                model=model,
                args=args,
                train_dataset=tokenized_ds["train"],
                eval_dataset=tokenized_ds["validation"],
                data_collator=data_collator,
                processing_class=tokenizer,
                compute_metrics=make_compute_metrics(),
                class_weights=class_weights,
            )
        else:
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

        metrics_payload: dict[str, Any] = {
            "test_metrics": test_metrics,
            "dataset_sizes": {name: len(split_rows) for name, split_rows in splits.items()},
            "base_model": base_model,
            "training_data_path": training_data_path,
        }
        if class_weights is not None:
            metrics_payload["class_weights"] = {
                ID2LABEL[i]: w.item() for i, w in enumerate(class_weights)
            }

        if IDIOM_PROBE_PATH.exists():
            print(f"\n=== Idiom probe evaluation ({IDIOM_PROBE_PATH}, held out of training) ===")
            probe_rows = load_labeled_sentences(IDIOM_PROBE_PATH)
            probe_ds = Dataset.from_list(probe_rows).map(
                tokenize, batched=True, remove_columns=["text"]
            )
            probe_metrics = trainer.evaluate(probe_ds, metric_key_prefix="idiom_probe")
            print(probe_metrics)
            metrics_payload["idiom_probe_metrics"] = probe_metrics
            metrics_payload["idiom_probe_size"] = len(probe_rows)

        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_output.write_text(json.dumps(metrics_payload, indent=2))
        print(f"Saved test metrics to {metrics_output}")

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\nSaved fine-tuned model to {output_dir}")

        return TrainedArtifact(output_dir=output_dir, metrics=metrics_payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weighted",
        action="store_true",
        help=(
            "Train on the original, unbalanced pool with inverse-class-frequency "
            "weighted loss instead of the rebalanced (downsampled) pool. Ignores "
            "BALANCED_DATA_PATH even if present. Writes to separate output paths."
        ),
    )
    parser.add_argument(
        "--base-model",
        default=MODEL_NAME,
        help=(
            "Base checkpoint to continue-fine-tune (default: %(default)s). A "
            "non-default value defaults to the ORIGINAL unbalanced training pool "
            "(matching v2's data exactly) unless --weighted is also passed, and "
            "writes to derived output paths -- see module docstring."
        ),
    )
    args_ns = parser.parse_args()
    config = SentimentTrainConfig(weighted=args_ns.weighted, base_model=args_ns.base_model)
    SentimentTrainer().train(config)


if __name__ == "__main__":
    main()
