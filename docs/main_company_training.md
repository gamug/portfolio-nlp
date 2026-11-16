# Main company: the relevance model and its measures (T-165, T-170)

**Status: written 2026-11-16.** What decides "the main company" of a text in step 3 (`docs/companies_step.md`), how the relevance model behind it was
trained and chosen, and every measure of it that exists, with the owner's partial marks on sub-news (T-170) as the only check on the texts the step
really reads. The step's own command, the revision and the dry run are in `docs/companies_step.md`; the labels the model learned from were made for the
sentiment step (`docs/sentiment_training.md`, "A relevance model"). Percentages are of rows unless said otherwise.

## What the step does

1. **Candidates**: NER (`gamug/sec-bert-finer-ord-ner@ba7b9e43`) finds organisation spans and the matcher turns each into a universe ticker; with
   `--candidates ner+text` the matcher also runs over the text (without lower-case matches and without matches inside disclosure lines).
2. **The relevance model** (`gamug/deberta-v3-company-relevance@8812c8bc`, private) scores each (text, ticker) pair: premise `<text>`, hypothesis
   `This text is mainly about <company>.`, 320 tokens.
3. **A ticker is main when its score is at the threshold or over it**: 0.5 (owner, 2026-11-13; another value is a new revision). Rows go to
   `news_company` (`relevance_score`, `is_main`, `mention_count`).

## Training data and model choice

| Item | Value |
|---|---|
| Labels | 809 (text, company) pairs from 855 tagged summaries (two draws, 450 and 405); relevant = the owner marked a direction (positive or negative), not relevant = the owner marked neutral |
| Why these labels | the model is built from the sentiment tags, so it learns which texts the owner found neutral for the company; it is not independent evidence about the sentiment |
| Zero-shot start | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`: weighted AUC 0.743 alone, the best of nine variants (three hypotheses x three premises: 0.618 to 0.743); the sentiment model's own `1 - p_neutral` reaches 0.795 |
| Fine-tuned | 3 epochs, learning rate 2e-5, batch 16, float32, 5 folds x 2 splits: out-of-fold weighted AUC **0.902 / 0.890** (mean 0.896) |
| In the sentiment stack | stack score 0.814 against 0.737 for the sentiment model alone (nested cross-validation, `docs/sentiment_training.md`) |
| Threshold | chosen on the out-of-fold scores of the 809 summaries (table below), at 0.5 |

Out-of-fold, unweighted, on the 809 marked summaries:

| Threshold | Relevant rows kept (recall of main) | Not-relevant rows called main | Precision of main |
|---|---|---|---|
| 0.3 | 96.2% | 33.2% | 89.7% |
| **0.5 (the step)** | **94.7%** | **31.7%** | **90.0%** |
| 0.7 | 92.3% | 27.2% | 91.1% |
| 0.9 | 87.3% | 23.8% | 91.7% |

On summaries the threshold hardly matters (the scores sit near 0 or 1), and no value separates the mentions well: between a quarter and a third of
the rows the owner marked neutral are called main. These are cross-validation figures on tags the recipe was tuned on: **not held out**, and the
model never saw a sub-news.

## What the model does on the texts the step reads

Dry run of 2026-11-13, a random sample of 200 sub-news (threshold 0.5, `ner+text`, seed 20261113), no marks:

| Threshold | 0.1 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 |
|---|---|---|---|---|---|---|
| Sub-news with a main company | 84.0% | 80.0% | 76.5% | **73.0%** | 70.5% | 65.5% |

`ner+text` found a candidate company in 93.5% of the sub-news and a main one in 73.0% (`ner` alone: 89.0% and 70.5%). The share is what the model
outputs, not what is right; nothing in the dry run says which threshold is correct.

## The owner's marks on sub-news (T-170)

**Dataset.** `data/main_company/to_tag.csv` (local, git-ignored; relevance scores and bands in `to_tag.strata.json`): 290 rows over 153 sub-news of the
cut `units5-d3e2cbfa830e` (companies `companies-f349536cf701`, seed 20261113), six relevance bands of the model's score (0 to 0.1, 0.1 to 0.3,
0.3 to 0.5, 0.5 to 0.7, 0.7 to 0.9, 0.9 to 1), drawn evenly, so every figure below is unweighted and none is a corpus rate. Columns: `item_id`, `texto`,
`texto_es`, `ticker`, `empresa`, `es_principal` (is this company the main one of the piece: `yes` or `no`), `falta_la_principal` (is the main company missing from the list),
`notas`. **Marked: 119 rows of 59 pieces (100 `yes`, 19 `no`); 171 rows are not marked.** The model's call is its relevance score at the threshold.
(An earlier version of the file had 353 rows over 180 pieces; the strata file still lists the 180.)

**Results** (`scripts/score_main_company_marks_2026_11_16.py`; Wilson 95% intervals):

| Class | Rows | Precision | Recall |
|---|---|---|---|
| Main (`yes`) at 0.5 | 100 | 90.9% [76.4, 96.9] (30/33) | 30.0% [21.9, 39.6] (30/100) |
| Not main (`no`) at 0.5 | 19 | 18.6% (16/86) | 84.2% (16/19) |
| Main at 0.3 | 100 | 90.0% [78.6, 95.7] (45/50) | 45.0% [35.6, 54.8] (45/100) |
| Main at 0.7 | 100 | 85.0% [64.0, 94.8] (17/20) | 17.0% [10.9, 25.5] (17/100) |

- **By piece (59):** 53 have a company the owner marked main; the model finds at least one of them in 30 (56.6%). The model calls a main company on 32
  pieces, and the owner agrees on at least one in 30 (93.8%).
- **By band** (share of `yes` rows; band 0 has the lowest scores): band 0, 19 of 21; band 1, 19 of 25; band 2, 20 of 22; band 3, 20 of 20; band 4, 10 of 11; band 5, 12 of 20.
- **Without three pieces** (335155 and 343149 are lists of tickers, 260642 a list of picks: 21 rows, every one `yes`, model scores of 0.01 to 0.16 on the
  first two): 98 rows (79 `yes`), main precision 90.9% (30/33), recall 38.0% (30/79), not-main precision 24.6% (16/65), recall 84.2% (16/19).
- `falta_la_principal` is filled on the 119 marked rows (114 `no`, 5 `yes`) and agrees with `es_principal` except on one row; it is not scored here.

**What it says.**

- **When the step names a main company it is almost always one the owner accepts** (precision 90.9% at 0.5, 85.0% at 0.7).
- **It misses most of what the owner calls main** (recall 30.0%; 45.0% at 0.3): 19 of the 21 rows of the lowest band are `yes`. The lists of tickers explain
  part of it, not all (recall 38.0% without them).
- **The owner's rule may be wider than the model's**: a piece with several companies gets several `yes`, while the model was trained to pick the one the
  text is mainly about (the 809 summaries). The marks do not say which reading is wanted; that is a decision for the owner, not an error of the marking.
- On these marks the 0.3 threshold gains 15 points of recall at the same precision (90.0%), where on summaries the threshold hardly mattered; the marks are
  too few to move the threshold, which is a new revision (T-170).

## Limits

- 59 pieces, one reader, 41% of the rows marked; the intervals are wide and the figures are not corpus rates (the bands were drawn evenly).
- The model has not been shown to pick the owner's main company on sub-news; this is the open question of T-170. The owner's marks on the rest of
  the file would give the first figure with an interval narrow enough to act on.
- The out-of-fold figures of the 809 summaries are not held out, and are not about sub-news.
- No second reader, and no check of the share of pieces with no company named (the split between "no universe company" and "named in a form the matcher does not know" was not measured).
- The 119 marks are the owner's judgement of "main" under the file's definition, not a count of facts.

## Reproduce

```bash
uv run scripts/score_main_company_marks_2026_11_16.py            # the table above, from data/main_company
uv run scripts/score_main_company_marks_2026_11_16.py --exclude 335155,343149,260642
```

The out-of-fold figures and the threshold table: `scripts/relevance_threshold_candidates_2026_11_13.py` and `docs/sentiment_training.md`, "A relevance model".
