#!/usr/bin/env python
"""One-shot: rebalance the sentiment fine-tuning data (PLAN.md Work item 9 /
SPEC.md §13 item 14).

The merged training pool (data/sentiment_finetune/labeled_sentences.jsonl +
idiom_augment.jsonl, 5,800 sentences -- exactly what train_sentiment.py's
stratified_split() partitions) is 3,256 neutral (56.1%) / 1,321 negative
(22.8%) / 1,223 positive (21.1%). That ratio was never a deliberate target:
it fell out of drawing sentences from the eval harness's confidence-
stratified sampling pool (stratified on prediction confidence, not label
ratio), on top of real financial news skewing neutral/factual.

This script downsamples neutral to 1,321 (the size of the larger minority
class, negative) -- every negative/positive sentence is kept untouched, only
neutral is trimmed, landing close to a genuine three-way balance (1,321 /
1,321 / 1,223) without discarding any minority-class data. Which 1,321 of
the 3,256 neutral sentences survive is not a random cut: each is ranked by
cosine similarity to the neutral class's own TF-IDF centroid (scikit-learn,
already a transitive dependency -- no new one added), and the most
representative (closest to centroid) 1,321 are kept, the most atypical/
outlier ones dropped.

idiom_probe.jsonl is untouched -- it stays exactly as published, per its
own disclosed role (held out of training entirely, used only to measure the
idiom-family fix directly against real, unfiltered traffic).

Writes data/sentiment_finetune/labeled_sentences_balanced_2026_09_14.jsonl
-- the full, already-merged, already-rebalanced training pool.
train_sentiment.py prefers this file over labeled_sentences.jsonl +
idiom_augment.jsonl when it exists (see that module's docstring).

Run once, before re-running train_sentiment.py and before republishing
gamug/FinBERT-financial-news-data.

Usage:
    uv run python scripts/rebalance_sentiment_data_2026_09_14.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from train_sentiment import DATA_PATH, IDIOM_AUGMENT_PATH

OUTPUT_PATH = Path("data/sentiment_finetune/labeled_sentences_balanced_2026_09_14.jsonl")


def load_full_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def most_representative(rows: list[dict[str, object]], keep_n: int) -> list[dict[str, object]]:
    """Rank rows by cosine similarity of their TF-IDF vector to the group's own
    centroid (mean vector), and keep the keep_n closest -- the most prototypical
    examples of this class, not a random subsample."""
    if keep_n >= len(rows):
        return rows

    sentences = [str(r["sentence"]) for r in rows]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    matrix = vectorizer.fit_transform(sentences)
    # matrix.mean(axis=0) returns a np.matrix (sparse-matrix legacy type);
    # cosine_similarity requires a plain ndarray, hence asarray + reshape.
    centroid = np.asarray(matrix.mean(axis=0)).reshape(1, -1)
    similarities = cosine_similarity(matrix, centroid).ravel()

    ranked_idx = similarities.argsort()[::-1]  # most similar to centroid first
    keep_idx = sorted(ranked_idx[:keep_n])  # restore original row order
    return [rows[i] for i in keep_idx]


def main() -> None:
    print(f"Loading {DATA_PATH} ...")
    rows = load_full_rows(DATA_PATH)
    print(f"Loaded {len(rows)} base sentences")

    print(f"Merging {IDIOM_AUGMENT_PATH} ...")
    augment_rows = load_full_rows(IDIOM_AUGMENT_PATH)
    rows = rows + augment_rows
    print(f"Merged pool: {len(rows)} sentences")

    by_label: dict[str, list[dict[str, object]]] = {"positive": [], "negative": [], "neutral": []}
    for r in rows:
        by_label[str(r["label"])].append(r)

    before = {label: len(group) for label, group in by_label.items()}
    print(f"Before: {before}")

    target_n = max(len(by_label["positive"]), len(by_label["negative"]))
    print(f"Target neutral count (= larger minority class): {target_n}")

    neutral_kept = most_representative(by_label["neutral"], target_n)
    print(
        f"Neutral kept: {len(neutral_kept)} / {len(by_label['neutral'])} "
        f"(dropped {len(by_label['neutral']) - len(neutral_kept)} least-representative)"
    )

    balanced_rows = by_label["positive"] + by_label["negative"] + neutral_kept
    after = {
        "positive": len(by_label["positive"]),
        "negative": len(by_label["negative"]),
        "neutral": len(neutral_kept),
    }
    total = sum(after.values())
    print(f"After: {after} (total {total})")
    for label, n in after.items():
        print(f"  {label}: {n} ({100 * n / total:.1f}%)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        for r in balanced_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(balanced_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
