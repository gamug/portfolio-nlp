"""Publish the final relevance and sentiment models to the Hugging Face Hub, **private** (T-169; owner's word, 2026-11-10).

    uv run python scripts/publish_final_models_2026_11_10.py [--sentiment-dir models/sentiment_pair_all/final] \
        [--relevance-dir models/relevance_final] [--dry-run]

Two repositories under ``gamug/``: ``FinBERT-company-sentiment`` (the pair model + ``sentiment_stack.json``) and
``deberta-v3-company-relevance``. Each gets a model card that says what the model is trained for, on which data, with
which recipe, which figures were measured and how, and what is not known. The figures are those of
``docs/sentiment_training.md`` (cross-validation over the owner's tags); **the 180-text gold has not been scored** and the
cards say so. Nothing is pinned: ``config/pipeline_models.json`` is not touched (the pin is decided on the gold, D35).
Needs ``HF_TOKEN`` (write scope) in the environment or ``.env``. The result, with the commit SHAs, is printed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SENTIMENT_REPO = "gamug/FinBERT-company-sentiment"
RELEVANCE_REPO = "gamug/deberta-v3-company-relevance"
ENDPOINT = "gamug/FinBERT-financial-news@93863fcb7252874e7c0339081b34f691f9e17ff6"
NLI = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0@8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8"

DATA = """\
**Data.** 809 rows, each a *text* (a short summary of a financial-news article, about 330 characters on average, English,
published 2022 to 2026; written by `sshleifer/distilbart-cnn-12-6`) and a *company* (228 companies of the S&P 500
universe). The labels were made by one person, the project owner, who read each text **for that company**: the sentiment
an investor would read on the company's price (positive 394, negative 213, neutral 202). The rule: if the company is in
the facts the text presents, the mark is the sentiment about it; if the text gives no fact about the company (it is absent,
mentioned in passing, or the facts are about another company or the market), the mark is **neutral**. The rows come from two
draws balanced by what an earlier model predicted (not by the owner's classes), so every figure below is *weighted* back
to the pool each draw came from. The marks and their texts are tracked in the project repository `gamug/portfolio-nlp`
(`gold/sentiment_train.csv`, `gold/sentiment_train_2.csv`, `gold/texts/`); this model repository holds no news text."""

SENTIMENT_CARD = f"""---
license: cc-by-nc-4.0
base_model: gamug/FinBERT-financial-news
language:
- en
pipeline_tag: text-classification
tags:
- finance
- sentiment-analysis
- company-level
- bert
- finbert
---
# FinBERT-company-sentiment (private)

**What it is for.** The sentiment of a short financial-news text **about one named company**: the input is the pair
(company name, text) and the output is `positive`, `negative` or `neutral` for that company. "Neutral" also means *the text
says nothing about this company*. It is the sentiment model of step 6 of the project's news pipeline (PLAN D39/D40), meant to be
used **together with the relevance model** (`{RELEVANCE_REPO}`) through the stack in `sentiment_stack.json`.

**Not for:** raw article bodies (it was trained on summaries of a few sentences), texts with no company (sector, market,
economy: they are read by another model), languages other than English, or any use as investment advice.

## How it was trained
- Start: `{ENDPOINT}` (itself a continued fine-tune of `ProsusAI/finbert` on a balanced pool of 3,865 sentences labelled one by one by an LLM, not tied to a company).
- Recipe: input = company name as segment A, text as segment B (only the text is truncated, 512 tokens); 2 epochs, learning
  rate 3e-5, seed 4, no validation slice (every row is used for training).
- Training rows: the 809 owner-tagged pairs below **plus 341 replay sentences** from the 3,865 (sentences that name their
  article's company, read as a pair with it; their labels are an LLM's sentence-level reading, not the owner's rule) to keep the
  sentence knowledge: 1,150 examples.

{DATA}

## Figures (cross-validation, not an exam)
Measured **before** this final model existed, on the same recipe: 5-fold cross-validation, twice, over the 809 rows (each text
scored by a model that did not train on it, "out-of-fold"), weighted as above. Selection metric (owner, D37): the mean of nine
figures: precision, recall and ROC (one class against the other two) per class.

| | sentiment model alone | **stacked with the relevance model** |
|---|---|---|
| score (mean of the nine) | 0.737 | **0.814** (nested, below) |
| macro precision / recall | 0.692 / 0.674 | 0.780 / 0.763 |
| macro ROC (one vs rest) | 0.845 | 0.898 |
| neutral precision / recall | 0.634 / 0.442 | 0.787 / 0.686 |
| negative precision / recall | 0.680 / 0.745 | 0.747 / 0.748 |
| positive precision / recall | 0.761 / 0.834 | 0.807 / 0.855 |

For comparison, the text-only model that is pinned in the pipeline scores 0.648 on the same texts. The stack was evaluated
*nested* (its coefficients fitted on 4/5 of the out-of-fold rows and applied to the other 1/5, 10 repeats x 2); gains over the
sentiment model alone are paired and bootstrapped (400 resamples): stack score +0.077 [+0.057, +0.095].

**The 180-text gold has not been scored.** That is the exam of the project (D32/D35); it is run once, on this published
revision, and decides the pin. Until then these figures are estimates of the *recipe*, not of this model.

## Assumptions and limits
- One annotator, one rule; agreement between annotators was not measured. The rule was given after the first 450 marks.
- The classes are not balanced (394/213/202) and the weights correct for the sampling design, not for the corpus.
- The final model saw **all** the rows, so there is no held-out score of *this* model; the figures are for the recipe.
- The stack's coefficients were fitted on out-of-fold scores; this final model is slightly stronger than the fold models, so
  its probabilities are a little more confident than the stack saw (the effect is not measured).
- Summaries, English, 2022 to 2026; the labels have no date cut-off (look-ahead in a backtest is not excluded).
- "Neutral because the company is absent" is learned from the pair; three quarters of neutral texts do name the company, so the
  model cannot rely on the name alone.

## Use
```python
import json, numpy as np, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{SENTIMENT_REPO}")
model = AutoModelForSequenceClassification.from_pretrained("{SENTIMENT_REPO}").eval()
x = tok("Apple Inc.", "Apple shares fell 5% after weak iPhone sales.", truncation="only_second", max_length=512, return_tensors="pt")
p = torch.softmax(model(**x).logits, -1)[0].tolist()          # order: positive, negative, neutral
# stack with the relevance model's score r (probability of 'entailment', see its card):
s = json.load(open(hf_hub_download("{SENTIMENT_REPO}", "sentiment_stack.json")))
r = 0.9
f = np.array([*np.log(np.clip(p, 1e-6, 1)), np.log(np.clip(r, 1e-6, 1 - 1e-6) / (1 - np.clip(r, 1e-6, 1 - 1e-6)))])
z = np.array(s["coef"]) @ f + np.array(s["intercept"]); probs = np.exp(z - z.max()); probs /= probs.sum()   # s["classes"] order
```
"""

RELEVANCE_CARD = f"""---
license: mit
base_model: MoritzLaurer/deberta-v3-base-zeroshot-v2.0
language:
- en
pipeline_tag: text-classification
tags:
- finance
- relevance
- natural-language-inference
- deberta-v3
---
# deberta-v3-company-relevance (private)

**What it is for.** Whether a short financial-news text **is about a given company** (the company is in the facts the text
presents) or not (absent, a passing mention, or the facts are about another company or the market). It is a natural-language
inference classifier: premise = the text, hypothesis = `This text is mainly about <company>.`; the probability of class 0
(`entailment`) is the **relevance score**. In the project's pipeline it (1) picks the main company of a summary or a sub-news
among the companies found by NER (every company over a threshold is main, PLAN D39 (i)) and (2) is stacked with the sentiment
model (`{SENTIMENT_REPO}`) to fix its weak class (neutral).

**Not for:** raw article bodies, texts without a company name to test, other languages, or as a sentiment model.

## How it was trained
- Start: `{NLI}` (the project's zero-shot category model).
- Labels derived from the sentiment tags: **relevant** = the owner marked a direction (positive or negative), **not relevant**
  = the owner marked neutral. So "not relevant" carries the same mixture as neutral: it includes texts that *do* name the
  company but state no fact about its price. This is a limit of the labels, not a separate annotation.
- Recipe: 3 epochs, learning rate 2e-5, batch 16, weight decay 0.01, 10% warm-up, max length 320, float32 weights with bf16
  autocast, seed 1; trained on **all 809 rows** (no validation slice). Cross-entropy on two classes.

{DATA.replace("(positive 394, negative 213, neutral 202)", "(relevant = positive or negative: 607; not relevant = neutral: 202)")}

## Figures (cross-validation, not an exam)
5-fold x 2 cross-validation over the same 809 rows, every text scored by a model that did not train on it, weighted:
**AUC of the relevance score for "relevant" = 0.902 and 0.890 (mean 0.896)**. For comparison: the same model zero-shot 0.743
(the best of nine wordings, chosen on these same texts), and the sentiment model's own `1 - p(neutral)` 0.795.
Stacked with the sentiment model the score goes from 0.737 to 0.814 (see `{SENTIMENT_REPO}`).

**The 180-text gold has not been scored.** The relevance score was never evaluated on texts the project did not tag, and not on
sub-news (shorter pieces cut by the extractor): it was trained on summaries, so its use there is an open risk (task T-170).

## Assumptions and limits
- The labels are derived (a direction means relevant) from one annotator; neutral-but-about-the-company rows teach the model
  that "not mainly about" can include "about, no direction".
- Weighted figures estimate the pools of two sampling designs, not the corpus.
- The threshold for "main company" is **not set** (task T-169/T-172): it is chosen from the out-of-fold scores before use.
- The final model saw all rows; the figures are for the recipe.

## Use
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{RELEVANCE_REPO}")
model = AutoModelForSequenceClassification.from_pretrained("{RELEVANCE_REPO}").eval()
x = tok("Apple shares fell 5% after weak iPhone sales.", "This text is mainly about Apple.", truncation="only_first", max_length=320, return_tensors="pt")
relevance = torch.softmax(model(**x).logits, -1)[0, 0].item()      # class 0 = entailment = relevant
```
"""


def publish(
    token: str, repo: str, folder: Path, card: str, extra: dict[str, Path], dry: bool
) -> str:
    import tempfile  # noqa: PLC0415 -- only the upload needs these

    from huggingface_hub import HfApi  # noqa: PLC0415

    if dry:
        print(f"[dry run] {repo}: {folder} + {sorted(extra)} + README.md ({len(card)} characters)")
        return "dry-run"
    api = HfApi(token=token)
    api.create_repo(repo, private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for item in folder.iterdir():
            if item.name != "training_record.json" or repo == RELEVANCE_REPO:
                (stage / item.name).symlink_to(item.resolve())
        for name, path in extra.items():
            (stage / name).write_bytes(path.read_bytes())
        (stage / "README.md").write_text(card, encoding="utf-8")
        info = api.upload_folder(
            repo_id=repo, folder_path=str(stage), commit_message="final model (T-169)"
        )
    sha = getattr(info, "oid", "")
    print(f"{repo} (private): {sha}")
    return str(sha)


def main() -> None:
    from dotenv import load_dotenv  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sentiment-dir", type=Path, default=Path("models/sentiment_pair_all/final")
    )
    parser.add_argument("--relevance-dir", type=Path, default=Path("models/relevance_final"))
    parser.add_argument("--stack", type=Path, default=Path("config/sentiment_stack.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    token = os.environ.get("HF_TOKEN", "")
    if not token and not args.dry_run:
        raise SystemExit("HF_TOKEN (write scope) is not set")
    out = {
        "sentiment": publish(
            token,
            SENTIMENT_REPO,
            args.sentiment_dir,
            SENTIMENT_CARD,
            {"sentiment_stack.json": args.stack},
            args.dry_run,
        ),
        "relevance": publish(
            token, RELEVANCE_REPO, args.relevance_dir, RELEVANCE_CARD, {}, args.dry_run
        ),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
