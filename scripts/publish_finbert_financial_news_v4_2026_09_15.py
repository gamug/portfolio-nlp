#!/usr/bin/env python
"""One-shot: push the class-weighted retrain (src/train_sentiment.py --weighted,
models/finbert-financial-news-weighted/) to the Hugging Face Hub as
gamug/FinBERT-financial-news v4, with a model card documenting the
class-weighting rationale and a full, honest before/after comparison
against v2 -- both the sentence-level/idiom-probe numbers and the
downstream, production-pipeline LLM-judge numbers (the ones that actually
validated v2 in the first place).

Unlike v3 (scripts/publish_finbert_financial_news_v3_2026_09_14.py), this
publish IS adopted into production: src/pipeline.py's MODEL_REVISIONS pin
moves to this commit's SHA in the same PR (a deliberate user decision --
v4 trades some agreement_rate/mean_severity for a real recall_negative
gain, this pipeline's stated priority metric; see docs/evaluation.md's
2026-09-15 follow-ups for the full numbers this decision was made from,
including v5's base-checkpoint-swap experiment that was tried and not
chosen).

Requires HF_TOKEN in the environment/.env (write scope). Run once, after
src/train_sentiment.py --weighted and after the downstream v4 eval
(scripts/resample_sentiment_v4_2026_09_15.py + cli/news_nlp_eval.py).

Usage:
    uv run python scripts/publish_finbert_financial_news_v4_2026_09_15.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

MODEL_DIR = Path("models/finbert-financial-news-weighted")
METRICS_PATH = Path("data/sentiment_finetune/test_metrics_weighted.json")
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
the base model, only the training sentences (and, as of v4, the loss weighting) differ.

## v4 (2026-09-15): class-weighted loss — this is the version this repo's pipeline runs

**v1/v2's training data was 56.1% neutral / 22.8% negative / 21.1% positive — never a
deliberate target.** Two fixes were tried for it: v3 downsampled `neutral` (a real trade — see
that version's own notes, still published here as an available checkpoint but not adopted);
v4 instead keeps the full original pool and applies an inverse-class-frequency-weighted
`CrossEntropyLoss` (`train_sentiment.py --weighted`) so no training sentence is discarded, only
reweighted. A third approach — swapping the base checkpoint to `nlpaueb/sec-bert-base` instead
of a data-side fix — was also tried and rejected before a downstream eval, since it lost to v2
on every sentence-level and idiom-probe metric (see the `portfolio-nlp` repo's
`docs/evaluation.md`, 2026-09-15).

**This version (v4) is the one `portfolio-nlp`'s production pipeline actually pins** —
`src/pipeline.py`'s `MODEL_REVISIONS` was updated to this commit in the same change that
published it, a deliberate adoption decision, not a default. Read the honest trade below before
assuming "newest = strictly better."

### One-vs-rest accuracy per class — held-out sentence-level test set (n=579, same split as v2)

| class | v2 (published) | **v4 (this version)** |
|---|---|---|
| Positive | `accuracy_ovr` **0.902** | 0.900 |
| Negative | `accuracy_ovr` 0.872 | **0.876** |
| Neutral | `accuracy_ovr` **0.822** | 0.817 |
| **Overall** | **Accuracy / Macro F1 0.798 / 0.779** | 0.796 / 0.778 |

Unlike v3, v4 doesn't meaningfully move any class here — every number sits within ~0.01 of v2.

### One-vs-rest accuracy per class — idiom probe (n=100, held out of training)

| class | v2 (published) | **v4 (this version)** |
|---|---|---|
| Positive | `accuracy_ovr` **0.93** | 0.92 |
| Negative | `accuracy_ovr` **0.90** | 0.87 |
| Neutral | `accuracy_ovr` 0.91 | 0.91 |
| **Overall** | **Accuracy / Macro F1 0.870 / 0.759** | 0.850 / 0.745 |

**The number that matters most here**: v4's idiom-probe neutral `accuracy_ovr` (0.91) lands
exactly on v2's (0.91) — the catastrophic collapse to 0.0 (measured in F1) that made v3 a real
regression simply doesn't happen with class weighting, since no neutral training sentence is
ever discarded.

### One-vs-rest accuracy per class — downstream, real-traffic production-pipeline eval (n=2000, LLM-judge)

The evaluation that actually validated v2 in the first place (entity-scoped, chunk-level
aggregation — the real `run_sentiment_stage` code path, not sentence-level scoring in
isolation). Full per-class precision/recall/F1 breakdown (not just `accuracy_ovr`) is in the
`portfolio-nlp` repo's `docs/evaluation.md`, 2026-09-15 follow-up:

| class | v2 (published) | **v4 (this version)** |
|---|---|---|
| Positive | `accuracy_ovr` **0.878** | 0.870 |
| Negative | `accuracy_ovr` **0.882** | 0.877 |
| Neutral | `accuracy_ovr` **0.811** | 0.803 |
| **Overall** | `agreement_rate` **0.701** | 0.674 |
| **Overall** | `macro_f1_vs_judge` **0.731** | 0.724 |
| **Overall** | `mean_severity` (lower is better) **0.341** | 0.369 |

**Why v4**: `recall_negative` is this pipeline's stated priority metric — missing a real
negative-sentiment article is a worse failure mode than a false alarm for this use case — and
v4 delivers a real gain there (0.808→0.832, full precision/recall/F1 breakdown in
`docs/evaluation.md`). The cost is real too, not hidden: every other per-class cell in that
breakdown moves slightly in v2's favor, and both `agreement_rate` and `mean_severity` above
get worse. This is a deliberate, disclosed trade, not a strict improvement — made with the
full breakdown in hand, not before it.

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

Full sourcing (base draw, idiom-augmentation round) documented in the companion dataset:
[`gamug/FinBERT-financial-news-data`](https://huggingface.co/datasets/gamug/FinBERT-financial-news-data).
v4 trains on the **original, unbalanced** `train` split (4,642 rows: 56.1% neutral / 22.8%
negative / 21.1% positive — same pool as v1/v2, not v3's rebalanced one), validates on
`validation` (579), tests on `test` (579), and is additionally evaluated (never trained) on
`idiom_probe` (100). Class imbalance is corrected at the loss level instead (inverse-frequency
weights: neutral 0.594, negative 1.464, positive 1.581 — computed from the train split's own
label counts), not by discarding data.

**Important limitation, unchanged from v1/v2/v3**: these labels are LLM-generated
(silver-standard), not human-annotated ground truth. Treat the metrics above as "agreement
with this specific LLM's sentence-level judgment" (offline) or "agreement with the downstream
LLM judge" (production-pipeline table), not an absolute accuracy figure — the same caveat this
project applies to every LLM-as-judge number in its own evaluation docs.

## Training procedure

Identical to v2's procedure, except the loss function — via Hugging Face `Trainer` subclassed
to apply the class weights above to `CrossEntropyLoss`:

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

**Note**: this loads whatever is currently the latest published commit (v4, as of this
publish, and the commit `portfolio-nlp`'s production pipeline actually pins). v2/v3 remain
available at their own earlier commits in this repo's history if you need to reproduce an
older comparison.

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
        commit_message="v4: retrain with class-weighted loss (Work item 9), adopted into production",
    )

    card_path = MODEL_DIR / "README.md"
    card_path.write_text(MODEL_CARD)
    print("Uploading model card ...")
    commit_info = api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="v4 model card: class-weighting rationale + full offline/downstream comparison",
    )
    print("commit url:", commit_info.commit_url)
    print("commit sha:", commit_info.oid)

    if METRICS_PATH.exists():
        print("Local test metrics on file:", json.loads(METRICS_PATH.read_text())["test_metrics"])


if __name__ == "__main__":
    main()
