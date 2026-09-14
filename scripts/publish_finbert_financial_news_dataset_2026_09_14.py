#!/usr/bin/env python
"""One-shot: push the sentence-level sentiment training data
(src/train_sentiment.py's input, data/sentiment_finetune/*.jsonl) to the
Hugging Face Hub as a dataset, gamug/FinBERT-financial-news-data --
the counterpart to the already-published gamug/FinBERT-financial-news
model.

Only this project's own sentiment fine-tuning data gets a dataset repo.
NER's fine-tuning data (gtfintechlab/finer-ord) is already a public HF
dataset this project didn't create -- nothing to publish there.

Splits are produced by train_sentiment.stratified_split() itself (same
seed=42, same row order: base draw then idiom-augment merged in), not
reimplemented here -- so the published train/validation/test split is
exactly what the model was actually trained and evaluated on, not a
fresh split that only looks similar. That function only reads
row["label"] to stratify, so passing it the full {sentence, label,
article_id} rows (instead of train_sentiment's own {text, label}-only
encoding) reproduces the identical partition while keeping every field.
idiom_probe.jsonl is published as its own split, labeled "held out of
training entirely" to match its actual, disclosed role.

Requires HF_TOKEN in the environment/.env (write scope).

Usage:
    uv run python scripts/publish_finbert_financial_news_dataset_2026_09_14.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from train_sentiment import (
    DATA_PATH,
    IDIOM_AUGMENT_PATH,
    IDIOM_PROBE_PATH,
    stratified_split,
)

load_dotenv()

REPO_ID = "gamug/FinBERT-financial-news-data"

DATASET_CARD = """---
license: cc-by-nc-4.0
language:
- en
task_categories:
- text-classification
tags:
- finance
- financial
- sentiment
- sentiment-analysis
- llm-labeled
pretty_name: FinBERT-financial-news training data
---

# FinBERT-financial-news-data

The sentence-level sentiment training data behind
[`gamug/FinBERT-financial-news`](https://huggingface.co/gamug/FinBERT-financial-news) --
5,900 sentences from real, English-language financial-news articles (2010s-2020s), each
labeled `positive`/`negative`/`neutral` from an investor/price-impact perspective by an LLM
(DeepSeek `deepseek-chat`, temperature 0, one sentence at a time, in isolation).

## Why this exists

Published as the direct counterpart to the model it trains, so the model's own claims about
its training data are independently checkable rather than only asserted in a model card.

## Splits

- **`train`** (4,642 rows) / **`validation`** (579) / **`test`** (579) -- the exact 80/10/10,
  per-label-stratified split (seed 42) `src/train_sentiment.py` actually trained and evaluated
  on, reproduced here via that script's own split function, not re-derived. `train` already
  includes the idiom-augmentation round (below) merged in, matching what the published model
  was actually trained on.
- **`idiom_probe`** (100 rows) -- **held out of training entirely**, used only to measure a
  specific fix directly (see the model card's "Idiom probe" evaluation table). Never appears
  in `train`/`validation`/`test`.

## Fields

- `sentence` -- the labeled sentence, extracted from a real financial-news article's body text.
- `label` -- `positive` / `negative` / `neutral`.
- `article_id` -- this project's own internal source-article id (`news_nlp` `articles.id`),
  included for provenance/traceability; meaningless outside that project's own database, not a
  public identifier.

## Composition

- **Base draw (5,000 sentences, in `train`/`validation`/`test`)**: split from real
  financial-news article bodies already sampled across this project's own LLM-as-judge
  sentiment evaluation runs (stratified sampling, not a fresh uniform draw). Class
  distribution: 938 positive (18.8%), 893 negative (17.9%), 3,169 neutral (63.4%) -- real
  financial news skews neutral/factual.
- **Idiom-family augmentation round (900 sentences: 800 merged into `train`/`validation`/
  `test`, 100 held out as `idiom_probe`)**: added after a spot-check of the first published
  model version found it still mislabeled sentences like "Amazon and Alphabet crushed
  earnings" as negative. A targeted mining pass scanned the full ~480k-article source corpus
  for the same idiom family (crushed/smashed/trounced/clobbered/routed/walloped/demolished/
  hammered + earnings/estimate/guidance/consensus/stock context) and labeled 900 more
  sentences the same way -- covering both directions explicitly, since the idiom family is
  genuinely ambiguous ("Nvidia stock **got crushed**" is negative; "Meta **crushed** its
  earnings estimates" is positive).

**Important limitation**: these labels are LLM-generated (silver-standard), not
human-annotated ground truth -- not independently verified against a human-labeled reference
set, beyond a manual spot-check of a 20-sentence sample of the idiom probe. Treat this as
"one specific LLM's sentence-level judgment," not an absolute-truth label set -- the same
caveat this project applies to every LLM-as-judge number in its own evaluation docs.

**Known overlap**: 7 of the 900 idiom-round `article_id`s also appear (via different
sentences) in the base 5,000-sentence draw -- the mining pass drew from the same overall
corpus, so a handful of source articles contributed sentences to both. No sentence itself is
duplicated across any two splits (verified exactly, not sampled), and `idiom_probe`'s own
100 sentences never appear in `train`/`validation`/`test`.

## Source and license

Sentences are excerpts from real, third-party financial-news article bodies; the underlying
articles themselves are not redistributed here, only these derived per-sentence excerpts and
their LLM-generated labels, for research/reproducibility -- the same scope the model card
built on this data discloses for its own training-data license note.

Released under **CC-BY-NC-4.0**, matching the model this data trains
([`gamug/FinBERT-financial-news`](https://huggingface.co/gamug/FinBERT-financial-news)):
attribution required, non-commercial use only.

## Related

- Model: [`gamug/FinBERT-financial-news`](https://huggingface.co/gamug/FinBERT-financial-news)
- Base checkpoint: [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert)
- Pipeline this model serves: [`portfolio-nlp`](https://github.com/gamug/portfolio-nlp)
"""


def load_full_rows(path: Path) -> list[dict[str, object]]:
    """Like train_sentiment.load_labeled_sentences, but keeps every original field
    (sentence text, article_id) instead of discarding everything but {text, label}."""
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    print(f"Loading {DATA_PATH} ...")
    rows = load_full_rows(DATA_PATH)
    print(f"Loaded {len(rows)} base sentences")

    if IDIOM_AUGMENT_PATH.exists():
        augment_rows = load_full_rows(IDIOM_AUGMENT_PATH)
        print(f"Merging {len(augment_rows)} idiom-augment sentences from {IDIOM_AUGMENT_PATH}")
        rows = rows + augment_rows

    # train_sentiment.stratified_split only reads row["label"] to stratify -- passing it
    # these full-field rows (same order the real training run used: base draw, then
    # idiom-augment merged in) reproduces the identical train/validation/test partition
    # while keeping "sentence"/"article_id", which the model's own {text, label}-only
    # encoding discards after tokenization.
    splits = stratified_split(rows)

    dataset_dict: dict[str, Dataset] = {}
    for split_name, split_rows in splits.items():
        dataset_dict[split_name] = Dataset.from_list(
            [
                {"sentence": r["sentence"], "label": r["label"], "article_id": r["article_id"]}
                for r in split_rows
            ]
        )
        print(f"{split_name}: {len(split_rows)} rows")

    if IDIOM_PROBE_PATH.exists():
        probe_rows = load_full_rows(IDIOM_PROBE_PATH)
        dataset_dict["idiom_probe"] = Dataset.from_list(
            [
                {"sentence": r["sentence"], "label": r["label"], "article_id": r["article_id"]}
                for r in probe_rows
            ]
        )
        print(f"idiom_probe: {len(probe_rows)} rows")

    ds = DatasetDict(dataset_dict)

    print(f"Creating/confirming repo {REPO_ID} ...")
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    print("Pushing dataset ...")
    ds.push_to_hub(REPO_ID, token=token)

    print("Uploading dataset card ...")
    commit_info = api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Add dataset card documenting sourcing, splits, and known caveats",
    )
    print("commit url:", commit_info.commit_url)


if __name__ == "__main__":
    main()
