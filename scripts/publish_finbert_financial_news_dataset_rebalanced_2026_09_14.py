#!/usr/bin/env python
"""One-shot: republish gamug/FinBERT-financial-news-data with the rebalanced
train/validation/test splits (PLAN.md Work item 9 / SPEC.md §13 item 14).

Run scripts/rebalance_sentiment_data_2026_09_14.py first -- this script only
publishes its output
(data/sentiment_finetune/labeled_sentences_balanced_2026_09_14.jsonl).

The original v1 dataset (published by
scripts/publish_finbert_financial_news_dataset_2026_09_14.py, same day) was
56.1% neutral / 22.8% negative / 21.1% positive -- never a deliberate
target. This version downsamples neutral to the larger minority class's
size (1,321) via TF-IDF-centroid representative selection, landing at
34.2% / 34.2% / 31.6%. `idiom_probe` is republished unchanged (same 100
rows, straight from idiom_probe.jsonl) -- its whole role is measuring
against real, unfiltered idiom-family traffic, not a class-balance concern,
and push_to_hub replaces every split in one commit, so it has to be
included even though nothing about it changed.

train/validation/test are produced by train_sentiment.stratified_split()
itself against the balanced pool -- the exact same split the retrained
model (src/train_sentiment.py, BALANCED_DATA_PATH branch) was actually
trained/evaluated on.

Requires HF_TOKEN in the environment/.env (write scope).

Usage:
    uv run python scripts/publish_finbert_financial_news_dataset_rebalanced_2026_09_14.py
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

from train_sentiment import BALANCED_DATA_PATH, IDIOM_PROBE_PATH, stratified_split

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
5,865 sentences from real, English-language financial-news articles (2010s-2020s), each
labeled `positive`/`negative`/`neutral` from an investor/price-impact perspective by an LLM
(DeepSeek `deepseek-chat`, temperature 0, one sentence at a time, in isolation).

## Why this exists

Published as the direct counterpart to the model it trains, so the model's own claims about
its training data are independently checkable rather than only asserted in a model card.

## v2 (2026-09-14): rebalanced

**v1 of this dataset (same day) was 56.1% neutral / 22.8% negative / 21.1% positive across
`train`/`validation`/`test` combined -- never a deliberate target.** It fell out of drawing
sentences from this project's own LLM-as-judge eval harness's sampling pool (stratified on
prediction *confidence*, not on label ratio), on top of real financial news skewing
neutral/factual. Nothing in the original training procedure corrected for it: model
selection used macro F1 (equal per-class weight, but only at evaluation time, not in the
training data itself), and the train/validation/test split was stratified to *match* the
source distribution, not rebalance it.

**This version downsamples `neutral`** to 1,321 -- the size of the larger minority class
(`negative`) -- keeping every `positive`/`negative` sentence untouched. Which 1,321 of the
original 3,256 `neutral` sentences survive is not a random cut: each was ranked by cosine
similarity to the `neutral` class's own TF-IDF centroid, and the most representative
(closest to centroid) were kept, the most atypical/outlier ones dropped
(`scripts/rebalance_sentiment_data_2026_09_14.py`).

| | v1 (2026-09-14, this same day) | **v2 (this version)** |
|---|---|---|
| neutral | 3,256 (56.1%) | **1,321 (34.2%)** |
| negative | 1,321 (22.8%) | 1,321 (34.2%) |
| positive | 1,223 (21.1%) | 1,223 (31.6%) |
| total (train+val+test) | 5,800 | 3,865 |

`idiom_probe` (100 rows, held out of training) is unchanged in both versions.

## Splits

- **`train`** (3,093) / **`validation`** (386) / **`test`** (386) -- the exact 80/10/10,
  per-label-stratified split (seed 42) `src/train_sentiment.py` actually trained and evaluated
  on for this version, reproduced here via that script's own split function, not re-derived.
- **`idiom_probe`** (100 rows) -- **held out of training entirely**, used only to measure a
  specific fix directly (see the model card's "Idiom probe" evaluation table). Never appears
  in `train`/`validation`/`test`.

## Fields

- `sentence` -- the labeled sentence, extracted from a real financial-news article's body text.
- `label` -- `positive` / `negative` / `neutral`.
- `article_id` -- this project's own internal source-article id (`news_nlp` `articles.id`),
  included for provenance/traceability; meaningless outside that project's own database, not a
  public identifier.

## Composition (pre-rebalance sourcing, still applies to every row kept)

- **Base draw**: split from real financial-news article bodies already sampled across this
  project's own LLM-as-judge sentiment evaluation runs (stratified sampling, not a fresh
  uniform draw).
- **Idiom-family augmentation round**: added after a spot-check of the first published model
  version found it still mislabeled sentences like "Amazon and Alphabet crushed earnings" as
  negative. A targeted mining pass scanned the full ~480k-article source corpus for the same
  idiom family (crushed/smashed/trounced/clobbered/routed/walloped/demolished/hammered +
  earnings/estimate/guidance/consensus/stock context) and labeled more sentences the same way
  -- covering both directions explicitly, since the idiom family is genuinely ambiguous
  ("Nvidia stock **got crushed**" is negative; "Meta **crushed** its earnings estimates" is
  positive).

**Important limitation**: these labels are LLM-generated (silver-standard), not
human-annotated ground truth -- not independently verified against a human-labeled reference
set, beyond a manual spot-check of a 20-sentence sample of the idiom probe. Treat this as
"one specific LLM's sentence-level judgment," not an absolute-truth label set -- the same
caveat this project applies to every LLM-as-judge number in its own evaluation docs.

**Selection caveat**: the TF-IDF-centroid method selects for *lexical* typicality (surface
wording close to the class's average), not necessarily semantic/topical diversity -- it could
in principle concentrate the kept `neutral` sentences around one common phrasing pattern
rather than spreading across the full range of neutral financial-news content. Not measured
directly here; the retrained model's own per-class metrics (linked below) are the real test
of whether this mattered.

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
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    print(f"Loading {BALANCED_DATA_PATH} ...")
    rows = load_full_rows(BALANCED_DATA_PATH)
    print(f"Loaded {len(rows)} rebalanced sentences")

    # train_sentiment.stratified_split only reads row["label"] to stratify -- passing it
    # these full-field rows reproduces the identical partition the actual retrain used
    # while keeping "sentence"/"article_id".
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

    probe_rows = load_full_rows(IDIOM_PROBE_PATH)
    dataset_dict["idiom_probe"] = Dataset.from_list(
        [
            {"sentence": r["sentence"], "label": r["label"], "article_id": r["article_id"]}
            for r in probe_rows
        ]
    )
    print(f"idiom_probe: {len(probe_rows)} rows (unchanged)")

    ds = DatasetDict(dataset_dict)

    print(f"Pushing rebalanced dataset to {REPO_ID} ...")
    ds.push_to_hub(REPO_ID, token=token)

    print("Uploading updated dataset card ...")
    commit_info = api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Rebalance: downsample neutral via TF-IDF-centroid representative selection (Work item 9)",
    )
    print("commit url:", commit_info.commit_url)


if __name__ == "__main__":
    main()
