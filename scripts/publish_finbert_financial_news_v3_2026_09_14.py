#!/usr/bin/env python
"""One-shot: push the rebalanced-data retrain (src/train_sentiment.py's
output using BALANCED_DATA_PATH, models/finbert-financial-news/) to the
Hugging Face Hub as gamug/FinBERT-financial-news v3, with a model card
documenting the rebalance and a full, honest before/after comparison
against v2 (PLAN.md Work item 9 / SPEC.md §13 item 14).

This publishes a new commit to the existing repo -- it does NOT change
src/pipeline.py's MODEL_REVISIONS pin (still the v2 SHA), so the
production pipeline keeps using v2 until/unless that pin is deliberately
updated. That's a separate, not-yet-made decision: v3 fixes the disclosed
class imbalance but introduces its own disclosed regression (idiom-probe
neutral F1 collapses to 0.0, see below) -- not a strict improvement to
silently adopt.

Requires HF_TOKEN in the environment/.env (write scope). Run once, after
src/train_sentiment.py (BALANCED_DATA_PATH branch) and after
scripts/publish_finbert_financial_news_dataset_rebalanced_2026_09_14.py.

Usage:
    uv run python scripts/publish_finbert_financial_news_v3_2026_09_14.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

MODEL_DIR = Path("models/finbert-financial-news")
METRICS_PATH = Path("data/sentiment_finetune/test_metrics.json")
REPO_ID = "gamug/FinBERT-financial-news"

MODEL_CARD = """---
license: cc-by-nc-4.0
base_model: ProsusAI/finbert
language:
- en
pipeline_tag: text-classification
tags:
- finance
- financial
- sentiment
- sentiment-analysis
- bert
- finbert
metrics:
- accuracy
- f1
- precision
- recall
---

# FinBERT-financial-news

A continued fine-tune of [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) on
sentences from real English-language financial news articles (2010s-2020s), labeled by an LLM
(DeepSeek `deepseek-chat`) from an investor/price-impact perspective — not a from-scratch
retrain: the label space (`positive`/`negative`/`neutral`) and task framing are unchanged from
the base model, only the training sentences (and, as of v3, their class balance) differ.

## v3 (2026-09-14): rebalanced training data — read this before using v3 over v2

**v1/v2's training data was 56.1% neutral / 22.8% negative / 21.1% positive — never a
deliberate target.** It fell out of drawing sentences from this project's own LLM-as-judge
eval harness's sampling pool (stratified on prediction *confidence*, not label ratio), on top
of real financial news skewing neutral/factual. Nothing in v1/v2's training procedure
corrected for it — model selection used macro F1 (equal per-class weight, but only at
*evaluation* time), and the train/validation/test split was stratified to match the source
distribution, not rebalance it.

v3 downsamples `neutral` to 1,321 (the size of the larger minority class, `negative`),
keeping every `positive`/`negative` sentence untouched — selected by cosine similarity to the
`neutral` class's own TF-IDF centroid (most representative kept, most atypical dropped), not a
random cut. Full sourcing/methodology and the exact rebalanced data:
[`gamug/FinBERT-financial-news-data`](https://huggingface.co/datasets/gamug/FinBERT-financial-news-data)
(v2 of that dataset).

**Honest result — a trade, not a strict improvement**: rebalancing measurably fixed what it
targeted, and measurably cost something else. Read both tables below before choosing v2 or v3.

### Held-out sentence-level test set

| metric | v2 (published, unbalanced data, n=579) | **v3 (this version, rebalanced data, n=386)** |
|---|---|---|
| Accuracy | 0.798 | 0.777 |
| Macro F1 | 0.779 | 0.774 |
| F1 — positive | 0.775 | **0.778** |
| Precision — positive | — | 0.795 |
| Recall — positive | — | 0.762 |
| F1 — negative | 0.726 | **0.829** |
| Precision — negative | — | 0.756 |
| Recall — negative | — | **0.917** |
| F1 — neutral | **0.838** | 0.714 |
| Precision — neutral | — | 0.789 |
| Recall — neutral | — | 0.652 |

Negative F1 improves substantially (0.726→0.829, the class that was *not* touched by
rebalancing but benefits from the model no longer being pulled toward the now-shrunk neutral
majority). Neutral F1 drops (0.838→0.714) — an expected, direct cost of training on fewer
neutral examples (1,321 vs. the original 3,256), not a surprise.

### Idiom probe (n=100, held out of training entirely, unchanged between v2/v3)

The idiom-family regression probe from the v1→v2 fix — **this is where v3's real cost shows
up**:

| metric | v2 (published, pre-rebalance) | **v3 (this version)** |
|---|---|---|
| Accuracy | 0.870 | 0.830 |
| Macro F1 | 0.759 | **0.583** |
| F1 — positive | 0.889 | 0.848 |
| F1 — negative | 0.917 | 0.902 |
| F1 — neutral | 0.47 | **0.0** |

**Neutral F1 on this probe collapses to 0.0 (precision and recall both 0.0) in v3.** This
probe is only 10% neutral by design (10/100 rows — it targets the crushed/smashed/hammered
idiom family, which this project's own labeling found skews negative/positive, not neutral),
so it's a small-n reading, not a broad claim about v3's neutral performance generally — but
it is a real, measured, disclosed regression, not glossed over: v3 apparently stopped
predicting `neutral` at all on this specific low-neutral-count, idiom-heavy slice, consistent
with training on 40% fewer neutral examples overall.

**Not yet measured for v3**: the downstream, production-pipeline evaluation (entity-scoped,
chunk-level aggregation against real article traffic, LLM-judge) that v2's card reports below
— that requires a full `--stage sentiment` eval run against live production data, a separate,
larger step from this retrain. Until that's run, v3's real-traffic behavior (as opposed to
its held-out-sentence behavior above) is not established.

**This repo's `src/pipeline.py` still pins v2's commit SHA** (`MODEL_REVISIONS`,
`portfolio-nlp`) — v3 is published here as an available checkpoint, not silently adopted into
the production pipeline. Given the disclosed neutral/idiom-probe regression above, that's a
deliberate choice to make with the downstream numbers in hand, not something this publish
decides on its own.

## Why this exists (v1, unchanged)

`ProsusAI/finbert` was fine-tuned on [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
— ~4,840 sentences from **2014 English-language news about OMX Helsinki (Nordic) listed
companies**, via LexisNexis. That's a real, measurable domain/vocabulary gap for a pipeline
scoring 2010s-2020s English-language financial news on globally-listed companies: real
disagreement cases found during evaluation included the base model missing "crushed" as a
positive idiom ("Amazon and Alphabet crushed earnings"), and terms/instruments (e.g.
cryptocurrency) that didn't meaningfully exist in the base model's training window.
This model targets that specific gap — a vocabulary/domain refresh, not an architecture or
label-space change.

## Training data

Full sourcing (base draw, idiom-augmentation round, v3's rebalance) documented in the
companion dataset:
[`gamug/FinBERT-financial-news-data`](https://huggingface.co/datasets/gamug/FinBERT-financial-news-data).
v3 trains on that dataset's `train` split (3,093 rows: 1,057 negative, 1,057 neutral, 979
positive — 34.2%/34.2%/31.6%), validates on `validation` (386), tests on `test` (386), and is
additionally evaluated (never trained) on `idiom_probe` (100).

**Important limitation, unchanged from v1/v2**: these labels are LLM-generated
(silver-standard), not human-annotated ground truth. Treat the metrics above as "agreement
with this specific LLM's sentence-level judgment," not an absolute accuracy figure — the same
caveat this project applies to every LLM-as-judge number in its own evaluation docs.

## Training procedure

Identical to v2's procedure, only the training data changed (rebalanced, see above) — via
Hugging Face `Trainer`:

- learning rate: 2e-5
- batch size: 16 (train) / 32 (eval)
- epochs: 4
- weight decay: 0.01
- mixed precision (fp16)
- `load_best_model_at_end=True`, selected by validation macro F1
- seed: 42

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("gamug/FinBERT-financial-news")
model = AutoModelForSequenceClassification.from_pretrained("gamug/FinBERT-financial-news")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
classifier("Acme Corp reported record profit and raised its full-year guidance.")
```

**Note**: this loads whatever is currently the latest published commit (v3, as of this
publish). To use the same checkpoint `portfolio-nlp`'s production pipeline actually runs
(v2, pinned), pass `revision="072712344f1f82e54391e6721b0b39e7b944e898"` to both
`from_pretrained` calls.

Recommended pre/post-processing: same as the base model — chunk long documents rather than
truncating (this checkpoint keeps BERT-base's 512-token limit), and if aggregating multiple
chunks/sentences per document yourself, consider scoping/weighting toward the actual subject
of the document rather than a plain average (see the `portfolio-nlp` repo above for one
worked design and its measured trade-offs).

## License

Derivative of two upstream works with different licenses:

- Base model `ProsusAI/finbert`.
- Training data here: LLM-generated from real news article text; the underlying articles are
  not redistributed, only derived per-sentence sentiment labels used for training.

Released under **`CC-BY-NC-4.0`**, matching the base model's own license: attribution
required, **non-commercial use only**.

## Related

- Training data: [`gamug/FinBERT-financial-news-data`](https://huggingface.co/datasets/gamug/FinBERT-financial-news-data)
- Pipeline this model serves: [`portfolio-nlp`](https://github.com/gamug/portfolio-nlp)
"""


def main() -> None:
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    print(f"Creating/confirming repo {REPO_ID} ...")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    print(f"Uploading model files from {MODEL_DIR} ...")
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(MODEL_DIR),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "vocab.txt"],
        ignore_patterns=["checkpoint-*/**", "checkpoint-*"],
        commit_message="v3: retrain on rebalanced training data (Work item 9)",
    )

    card_path = MODEL_DIR / "README.md"
    card_path.write_text(MODEL_CARD)
    print("Uploading model card ...")
    commit_info = api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="v3 model card: rebalance rationale + honest before/after comparison",
    )
    print("commit url:", commit_info.commit_url)

    if METRICS_PATH.exists():
        print("Local test metrics on file:", json.loads(METRICS_PATH.read_text())["test_metrics"])


if __name__ == "__main__":
    main()
