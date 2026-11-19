# Main company: the relevance model and its measures (T-165, T-170)

**Status: written 2026-11-16 (PR #201); the owner's marks of 2026-11-19 added (185 rows of 104 pieces).** What decides "the main company" of a text in step 3 (`docs/companies_step.md`), how the relevance model behind it was
trained and chosen, and every measure of it that exists, with the owner's partial marks on sub-news (T-170) as the only check on the texts the step
really reads. The step's own command, the revision and the dry run are in `docs/companies_step.md`; the labels the model learned from were made for the
sentiment step (`docs/sentiment_training.md`, "A relevance model"). Percentages are of rows unless said otherwise (the same marks are quoted in percent in `pipeline_metrics_report.md`, section 3.3).

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
model never saw a sub-news. **Its held-out validation is the owner's marks on sub-news, below.**

## What the model does on the texts the step reads

Dry run of 2026-11-13, a random sample of 200 sub-news (threshold 0.5, `ner+text`, seed 20261113), no marks:

| Threshold | 0.1 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 |
|---|---|---|---|---|---|---|
| Sub-news with a main company | 84.0% | 80.0% | 76.5% | **73.0%** | 70.5% | 65.5% |

`ner+text` found a candidate company in 93.5% of the sub-news and a main one in 73.0% (`ner` alone: 89.0% and 70.5%). The share is what the model
outputs, not what is right; nothing in the dry run says which threshold is correct.

## The owner's marks on sub-news (T-170)

Two versions of the marks are tracked. **The current one is the owner's of 2026-11-19**; the figures of 2026-11-16 stay below it as the earlier version, because a new set of marks is a new build and not an edit of the old one. The model is the same in both (`gamug/deberta-v3-company-relevance@8812c8bc`, companies `companies-f349536cf701`), so every difference between them comes from the marks and from which pieces are in the file.

### Version of 2026-11-19 (current)

**Dataset.** `gold/main_company_marks_2026_11_19.csv` (key `item_id#ticker`, band, relevance score, `text_hash`, the mark) and `gold/texts/main_company_marks_texts_2026_11_19.csv`, built by `scripts/build_main_company_marks_2026_11_19.py` from the owner's working file `data/main_company/to_tag.csv` (local, git-ignored; scores and bands in `to_tag.strata.json`): **185 marked rows of 104 pieces of the `units5` cut (166 `yes`, 19 `no`)**, plus one row of the file still unmarked (`282096#AIG`, not scored). The six relevance bands (0 to 0.1, 0.1 to 0.3, 0.3 to 0.5, 0.5 to 0.7, 0.7 to 0.9, 0.9 to 1, by the highest score of the piece) were drawn with 30 pieces each, seed 20261113. SHA-256: `to_tag.csv` `c4b59a20992912133b84914e4f22a35e8c43c3a877e88a7c3650781bb86a1ed1`, `to_tag.strata.json` `6e7f8d81d9feb6529528e4df0585bbd1ed88df32776715ea82e282ac2a9a3032` (unchanged), `gold/main_company_marks_2026_11_19.csv` `2e7e368d82c93c79802e8d542cf5d1a227a34a65440664f557b775efd1321687`; the scorer prints the hash of the file it reads.

**What changed in the file since 2026-11-16** (`build_main_company_marks_2026_11_19.py` prints it): 80 rows of the earlier 119 are still marked, 2 of them with another answer (`320508#WBD` and `320508#PSKY`, `no` before, `yes` now); 39 rows of 15 earlier pieces are no longer in the file; 105 marked rows of 60 pieces are new (100 `yes`, 5 `no`). Of the 180 pieces first drawn, 76 are no longer in the file (167 rows, 44 of the 55 rows of band 0 among them). The pieces left are therefore **not 30 per band any more**: 8, 14, 15, 16, 22 and 29 pieces from band 0 to band 5. The reason for each removal is the owner's and is not recorded here.

**Results by company** (`scripts/score_main_company_marks_2026_11_16.py`, which reads the tracked file; Wilson 95% intervals; unweighted):

| Class | Rows | Precision | Recall |
|---|---|---|---|
| Main (`yes`) at 0.5 | 166 | 96.1% [89.2, 98.7] (74/77) | 44.6% [37.2, 52.2] (74/166) |
| Not main (`no`) at 0.5 | 19 | 14.8% (16/108) | 84.2% (16/19) |
| Main at 0.1 | 166 | 95.6% [90.6, 97.9] (129/135) | 77.7% [70.8, 83.4] (129/166) |
| Main at 0.2 | 166 | 96.5% [91.3, 98.6] (110/114) | 66.3% [58.8, 73.0] (110/166) |
| Main at 0.3 | 166 | 96.1% [90.3, 98.5] (98/102) | 59.0% [51.4, 66.2] (98/166) |
| Main at 0.7 | 166 | 94.9% [86.1, 98.3] (56/59) | 33.7% [27.0, 41.2] (56/166) |

Not-main recall at the other thresholds: 68.4% (13/19) at 0.1, 78.9% (15/19) at 0.2 and at 0.3, 84.2% (16/19) at 0.7.

**Results by piece: the recall that does not depend on how many companies a piece has.** A piece counts as found when the model calls main at least one of the companies the owner marked main, so a piece with three owner mains needs one hit, not three. The model's calls are checked the same way: a piece it calls is agreed when one of its main companies is one the owner accepts.

| Threshold | Pieces with an owner main | Found by the model | Pieces the model calls main | Agreed |
|---|---|---|---|---|
| 0.1 | 99 | 92 (92.9% [86.1, 96.5]) | 96 | 92 (95.8%) |
| 0.2 | 99 | 84 (84.8% [76.5, 90.6]) | 87 | 84 (96.6%) |
| 0.3 | 99 | 79 (79.8% [70.8, 86.5]) | 82 | 79 (96.3%) |
| **0.5 (the step)** | 99 | **65 (65.7% [55.9, 74.3])** | 67 | 65 (97.0%) |
| 0.7 | 99 | 49 (49.5% [39.9, 59.2]) | 51 | 49 (96.1%) |

(Of the 104 pieces, 99 have a main company for the owner and 5 have none.)

**Re-weighted by the size of each band.** The bands were drawn evenly, and the corpus is not even: of the 151,620 pieces of the cut with a candidate company, 105,945 are in band 5 and 28,474 in bands 0 to 2. Giving each band its corpus weight (`POPULATION` in the scorer), the share of pieces with an owner main is 97.5% and the piece recall and precision are:

| Threshold | Piece recall | Piece precision |
|---|---|---|
| 0.3 | 85.7% | 98.9% |
| **0.5** | **82.6%** | **99.1%** |
| 0.7 | 79.2% | 99.1% |

The weights correct for the band sizes, not for the owner's removals: band 0 has 8 pieces of the 30 drawn and band 1 has 14, so their share of the estimate rests on few pieces.

| Band | Pieces | With an owner main | Found at 0.5 | Corpus pieces |
|---|---|---|---|---|
| 0 (0 to 0.1) | 8 | 7 | 0 | 13,756 |
| 1 (0.1 to 0.3) | 14 | 13 | 0 | 9,724 |
| 2 (0.3 to 0.5) | 15 | 14 | 0 | 4,994 |
| 3 (0.5 to 0.7) | 16 | 16 | 16 | 4,980 |
| 4 (0.7 to 0.9) | 22 | 20 | 20 | 12,221 |
| 5 (0.9 to 1) | 29 | 29 | 29 | 105,945 |

- **The gap is in the pieces where the model calls nothing, not in the number of mains a piece has.** At 0.5 the model names an owner main in all 65 pieces of bands 3 to 5 that have one, and in none of bands 0 to 2 (their highest score is under 0.5 by definition), where the owner finds a main company in 34 of 37 pieces. Pieces with one owner main are found 38 of 57 times (66.7%), and pieces with two or more 27 of 42 (64.3%): the one-company limit of the training does not explain the gap. It does show at the level of rows: in the 42 pieces with several owner mains the model names 36 of the 109 `yes` rows.
- **Precision stays high at every threshold** (94.9% to 96.5% of rows, 95.8% to 97.0% of pieces). Lowering the threshold costs little precision and finds more: at 0.1, 77.7% of the `yes` rows and 92.9% of the pieces, with 13 of the 19 `no` rows still kept out (68.4%), against 16 of 19 at 0.5. Only 19 rows are `no`, so that price is known to within a wide interval; the 0.1 to 0.5 choice is a new revision of step 3 and the owner's (T-170).
- **AUC of the score for `yes` against `no`: 0.789 [0.635, 0.909]** (resampling pieces; unweighted), against 0.896 out of fold on summaries. The point estimate is 10.7 points under, so the rule of T-170 (more than 5 points under: sub-news get their own threshold) still applies; the interval is wide and contains 0.896 because the `no` rows did not grow (19 in both versions). It is the least settled figure of this page.
- The same picture on the 60 pieces new to the marks alone (105 rows): main precision 98.0% (49/50), recall 49.0% (49/100), pieces found 40 of 58 (69.0%), AUC 0.788.
- Without the list of picks 260642, the only one of the three list pieces of the earlier version still in the file (4 rows, every one `yes`): 181 rows (162 `yes`), main precision 96.1% (74/77), recall 45.7% (74/162).
- `falta_la_principal` is `no` on 175 of the 185 marked rows and `yes` on 10; it is not scored here.

### Version of 2026-11-16 (earlier)

**Dataset.** `gold/main_company_marks.csv` and `gold/texts/main_company_marks_texts.csv`, built by `scripts/build_val_gold_2026_11_16.py` from the working file as it was then: 290 rows over 153 sub-news of the
cut `units5-d3e2cbfa830e` (companies `companies-f349536cf701`, seed 20261113), six relevance bands drawn evenly. Columns of the working file: `item_id`, `texto`,
`texto_es`, `ticker`, `empresa`, `es_principal` (is this company the main one of the piece: `yes` or `no`), `falta_la_principal` (is the main company missing from the list),
`notas`. **Marked: 119 rows of 59 pieces (100 `yes`, 19 `no`); 171 rows were not marked.** (The first version of the file had 353 rows over 180 pieces.) Files of that version: `to_tag.csv` sha256 `f3bd3030254cdad060ca19fbf9537bebdb6e749bbd3cf965fc2793f4c2533089`, `gold/main_company_marks.csv` `413a4bae73be2a297ab58e84b2f5adfa5a2145c0509d7c51ab12dfb8601667ad`.

**Results** (`--marks gold/main_company_marks.csv`; Wilson 95% intervals):

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
- **AUC of the score for `yes` against `no`: 0.663** (unweighted), against 0.896 out of fold on summaries.
- `falta_la_principal` was filled on the 119 marked rows (114 `no`, 5 `yes`).

**What the two versions say together.**

- **When the step names a main company it is almost always one the owner accepts** (precision 90.9% in the first version, 96.1% in the second, at 0.5).
- **At 0.5 it misses much of what the owner calls main**: 30.0% of the `yes` rows in the first version, 44.6% in the second. The second is higher mostly because the pieces left in the file are of the higher bands (band 0 has 8 pieces instead of 30), not because the model changed; the re-weighted piece recall of the second version is 82.6%.
- **The owner's rule may be wider than the model's**: a piece with several companies gets several `yes`, while the model was trained to pick the one the
  text is mainly about (the 809 summaries). The piece-level recall above measures the model without that limit, and shows it is not the main cause.
- On both sets a lower threshold gains recall at the same precision (at 0.3: 45.0% against 30.0% in the first, 59.0% against 44.6% in the second); on summaries the threshold hardly mattered.

## Decision this evidence is for

The rule of T-170 says that when the model's AUC on sub-news is more than 5 points under its 0.90 on summaries, sub-news get their own threshold chosen on the marked sub-news. The AUC on the marks is 0.789 (0.663 in the first version), so the rule applies; **the owner decides** (a new revision of step 3, `docs/companies_step.md`) the threshold, or whether the model should be retrained on pieces. What the 185 rows add to the choice: precision does not fall as the threshold goes down to 0.1 (95.6%), the piece recall goes from 65.7% at 0.5 to 79.8% at 0.3 and 92.9% at 0.1, and the cost is the `no` rows called main (3 of 19 at 0.5, 6 of 19 at 0.1), which only 19 rows measure. The sentiment step reads `news_company.is_main`, so this choice moves its input too.

## Limits

- 104 pieces, one reader, 19 `no` rows: the intervals are wide (the AUC especially) and the unweighted figures are not corpus rates.
- 76 of the 180 pieces first drawn, and 15 of the pieces marked on 2026-11-16, are no longer in the file. The re-weighting corrects the band sizes, not the reasons for those removals; a different set of pieces gives different row-level figures (30.0% against 44.6% of `yes` rows at 0.5 with the same model).
- The marks measure whether the model finds the owner's main company on sub-news; they do not say which threshold is right, which is the owner's decision above.
- The out-of-fold figures of the 809 summaries are not held out, and are not about sub-news.
- No second reader, and no check of the share of pieces with no company named (the split between "no universe company" and "named in a form the matcher does not know" was not measured).
- The marks are the owner's judgement of "main" under the file's definition, not a count of facts.

## Reproduce

```bash
uv run scripts/build_main_company_marks_2026_11_19.py --data data   # gold/main_company_marks_2026_11_19.csv and its texts, from the local working file (refuses to overwrite); prints the differences with the 2026-11-16 marks
uv run scripts/score_main_company_marks_2026_11_16.py                # the tables of the current version, from gold/main_company_marks_2026_11_19.csv
uv run scripts/score_main_company_marks_2026_11_16.py --thresholds 0.1,0.2,0.3,0.5,0.7
uv run scripts/score_main_company_marks_2026_11_16.py --marks gold/main_company_marks.csv   # the 2026-11-16 version
```

`tests/news_nlp/test_val_gold_2026_11_16.py` checks that the figures of this page are the scored ones.

The out-of-fold figures and the threshold table: `scripts/relevance_threshold_candidates_2026_11_13.py` and `docs/sentiment_training.md`, "A relevance model".
