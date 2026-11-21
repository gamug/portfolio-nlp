# Main company: the relevance model and its measures (T-165, T-170)

**Status: written 2026-11-16 (PR #201); the owner's re-tag of 2026-11-20 added (185 rows of 104 articles). **Decision of 2026-11-21: the headline recall is the model's single pick (recall 100.0%, 88 of 88; precision 84.6%); the threshold rule is secondary** (see "Decision, 2026-11-21" below). What decides "the main company" of a text in step 3 (`docs/companies_step.md`), how the relevance model behind it was
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

Two versions of the marks are tracked. **The current one is the owner's re-tag of 2026-11-20**; the 2026-11-16 version stays below it, because a new set of marks is a new build and not an edit of the old one. The model is the same in both (`gamug/deberta-v3-company-relevance@8812c8bc`, companies `companies-f349536cf701`), so every difference between them comes from the marks and from which pieces are in the file. The 2026-11-19 marks of the same PR (#219) were superseded by this re-tag before merging; their figures are in commit `9d2c0a5` and are not quoted here.

### Version of 2026-11-20 (current)

**Dataset.** `gold/main_company_marks_2026_11_20.csv` (key `item_id#ticker`, band, relevance score, `text_hash`, the mark, `falta_la_principal`, the owner's notes) and `gold/texts/main_company_marks_texts_2026_11_20.csv`, built by `scripts/build_main_company_marks_2026_11_20.py` from the owner's working file `data/main_company/to_tag.csv` (local, git-ignored; scores and bands in `to_tag.strata.json`). **185 marked rows of 104 articles (pieces) of the `units5` cut: 145 `yes`, 40 `no`.** Every row is marked. The six relevance bands (0 to 0.1, 0.1 to 0.3, 0.3 to 0.5, 0.5 to 0.7, 0.7 to 0.9, 0.9 to 1, by the highest score of the article) were drawn with 30 articles each, seed 20261113; 8, 14, 15, 16, 22 and 29 articles are left in them. SHA-256: `to_tag.csv` `97386a486e766fe848288e66087f95eb66fbce0658d98e19f2ee7cc0a8fa139d`, `to_tag.strata.json` `6e7f8d81d9feb6529528e4df0585bbd1ed88df32776715ea82e282ac2a9a3032` (unchanged), `gold/main_company_marks_2026_11_20.csv` `1205a302477ef651bac099ac1470a913b737be34c793f0dc84d770bae5fb5589`; the scorer prints the hash of the file it reads.

**The one-main limitation, and what the metrics count.** The relevance model was trained to pick the one company a text is mainly about (the 809 summaries). The owner marks every company that is main in an article, so an article with several owner mains can only be found in part by a count of rows: an article with six owner mains, where the model names two, scores one third on rows although it found a main company. That is a limit of the design of the task, not an error of the model, and the metrics below do not count it against the model. Two rules follow:

- **Recall and precision are per article, and the headline is the single pick** (decided 2026-11-21): the model's one main company per piece is its top-scoring row. An article is *found* when that pick is one the owner marked main. The threshold definition, where an article is found when the company at or over the threshold is one the owner marked main, is kept as secondary. Recall = found articles / articles with an owner main. Precision = articles where the model's main company is one the owner accepts / articles where the model calls a main company.
- **Row-level figures** (each owner main counts as one row) are kept as secondary figures, labelled as such. The not-main class is measured on rows: the owner's `no` is an explicit answer, so it is not affected by the limit.

**Results per article** (`scripts/score_main_company_marks_2026_11_20.py`, which reads the tracked file; Wilson 95% intervals for the counts; unweighted. The scorer also prints resampling intervals over the pieces; this doc quotes only the Wilson ones, so the two will not match in the last digit):

**Decision, 2026-11-21 (the owner's instruction; this is the trace of the headline).** The relevance model names one main company per text, so the headline is now the model's **single pick**: the top-scoring row of each piece. A piece is found when that pick is one of the companies the owner marked main; the model is not asked for a second or third owner main it was not built to name. The owner asked for at least 70% recall, and the single-pick recall meets it (100.0%, 88 of 88). The threshold rule (any row at or over 0.5, which is what the step does on every row) stays as a secondary figure.

**Limits of this decision, so it can be checked.** On these marks the single-pick recall cannot fall below 100%: every piece with an owner main has its top row marked main, so the resampling interval over pieces is 100.0% to 100.0%. The metric therefore cannot reject the model on recall here. The test of the model is its precision, 84.6% (88 of 104): the 16 pieces with no owner main still receive a pick, and each of them is a wrong pick. Figures: the HEADLINE block of `scripts/score_main_company_marks_2026_11_20.py`; the assertions in `tests/news_nlp/test_main_company_marks_2026_11_20.py`.

**Headline: the single pick** (the model's one main company per piece; 104 pieces, 88 with an owner main; Wilson 95% intervals):

| Measure | Figure | Counts |
|---|---|---|
| **Recall** | **100.0% [95.8, 100.0] (88/88)** | pieces with an owner main whose top row is one of the owner's mains |
| **Precision** | **84.6% [76.5, 90.3] (88/104)** | pieces whose top row is an owner main, of all pieces (the model names one company per piece) |
| Recall, resampling pieces | 100.0% [100.0, 100.0] | no resample falls below 100% (see the limits above) |

**Secondary: the threshold rule** (a piece is found when any row at or over the threshold is an owner main; the step's own rule on each row):

| Threshold | Article recall | Article precision | Articles the model calls main |
|---|---|---|---|
| 0.1 | 97.7% [92.1, 99.4] (86/88) | 89.6% [81.9, 94.2] (86/96) | 96 |
| 0.2 | 88.6% [80.3, 93.7] (78/88) | 89.7% [81.5, 94.5] (78/87) | 87 |
| 0.3 | 83.0% [73.8, 89.4] (73/88) | 89.0% [80.4, 94.1] (73/82) | 82 |
| **0.5 (the step)** | **69.3% [59.0, 78.0] (61/88)** | **91.0% [81.8, 95.8] (61/67)** | 67 |
| 0.7 | 53.4% [43.1, 63.5] (47/88) | 92.2% [81.5, 96.9] (47/51) | 51 |

104 articles; 88 have an owner main (16 have none marked `yes`). Re-weighted by the size of each band (`POPULATION` in the scorer; 90.2% of the articles have an owner main): article recall 97.5% at 0.1, 90.9% at 0.3, 88.0% at 0.5 and 84.8% at 0.7; article precision 96.7%, 97.0%, 97.7% and 98.1%. The weights correct the band sizes, not the reasons for the removals of 2026-11-16.

- **Articles with one owner main** are found 35 of 50 times at 0.5 (70.0%); **articles with two or more** 26 of 38 (68.4%). The limit does not change the article recall.
- **Article AUC** (the best relevance score of an article, articles with an owner main against articles without): **0.766 [0.662, 0.873]**, 88 against 16.
- **By band** (share of `yes` rows, for reference): band 0, 2 of 11; band 1, 17 of 26; band 2, 24 of 31; band 3, 27 of 31; band 4, 26 of 32; band 5, 49 of 54.

**Row-level figures** (secondary: each owner main counts as one row; 145 `yes`, 40 `no`). At 0.5: main precision 92.2% (71/77), main recall 49.0% [41.0, 57.0] (71/145), not-main recall 85.0% (34/40). Row recall at 0.5 is lower for the articles with several owner mains: 70.0% (35 of 50 rows) in articles with one owner main, 37.9% (36 of 95 rows) in articles with two or more. That is the limit above, shown in the numbers. Row AUC of the score for `yes` against `no`: **0.829 [0.730, 0.912]** (resampling articles; against 0.896 out of fold on summaries).

**Scenarios: the same marks with two kinds of article left out.** The owner's proposal: articles with no company at all (`322112`, `335253`), and articles whose main company is missing from the list or out of the S&P 500 (28 articles, from a `falta_la_principal` `yes` on any row or a note; `MISSED` in the scorer). The second group is a failure of the model by definition, so leaving it out measures the model only where the main company is in the list; it is not a corpus rate. At 0.5:

| Scenario | Articles, rows (yes / no) | Article recall | Article precision | Article AUC | Row recall | Row not-main recall |
|---|---|---|---|---|---|---|
| All | 104, 185 (145 / 40) | 69.3% (61/88) | 91.0% (61/67) | 0.766 | 49.0% (71/145) | 85.0% (34/40) |
| Without no-company articles | 102, 183 (145 / 38) | 69.3% (61/88) | 92.4% (61/66) | 0.777 | 49.0% (71/145) | 86.8% (33/38) |
| Without missed-main articles | 76, 138 (125 / 13) | 73.0% (54/74) | 98.2% (54/55) | n/a (74 against 2) | 51.2% (64/125) | 92.3% (12/13) |
| Without both | 74, 136 (125 / 11) | 73.0% (54/74) | 100% (54/54) | n/a (74 against 0) | 51.2% (64/125) | 100% (11/11) |

- **Removing the missed-main articles removes 29 of the 40 `no` rows.** The not-main figures rest on 11 to 13 rows, and the article AUC has 2 or no articles without a main: both are not reported as findings for those scenarios.
- **The missed-main articles do not explain the threshold recall.** Leaving them out moves threshold article recall only from 69.3% to 73.0%.
- At 0.3 the same holds: article recall 83.0% on all articles, 83.8% (62/74) without the missed-main articles; article precision 89.0% and 96.9%.

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

**The two versions together.**

- **When the step names a main company it is almost always one the owner accepts**: article precision at 0.5 is 91.0% in the 2026-11-20 version and 93.8% (30 of 32) in the 2026-11-16 version. The row precision of the 2026-11-16 version is 90.9% (30 of 33), a different measure, given only for reference.
- **On the threshold rule, at 0.5 the model finds about seven articles in ten with an owner main**: 56.6% of the 2026-11-16 articles (30 of 53), 69.3% of the 2026-11-20 articles (61 of 88). The article-level counts are the ones that do not depend on how many companies an article has; the row figures of the 2026-11-16 version (30.0% of 100 `yes` rows) are the same limit in another form.
- **The row AUC is 0.663 [0.467, 0.835] on the 2026-11-16 marks and 0.829 [0.730, 0.912] on the 2026-11-20 marks.** The intervals overlap, and the samples differ (19 `no` and 100 `yes` against 40 and 145), so this is not a measured change in the model. The article AUC is 0.601 [0.333, 0.857] (2026-11-16, 53 articles with a main against 6) and 0.766 [0.662, 0.873] (2026-11-20, 88 against 16). The model is the same revision, `gamug/deberta-v3-company-relevance@8812c8bc`, in both.

## Decision this evidence is for

The rule of T-170 says that when the model's AUC on sub-news is more than 5 points under its 0.90 on summaries, sub-news get their own threshold chosen on the marked sub-news. The row AUC on the current marks is 0.829 [0.730, 0.912] (0.663 on the 2026-11-16 marks), 6.7 points under 0.896, so the rule applies on the point estimate; the article AUC of 0.766 is further under, but it is a different measure (one value per article), so it is reported beside it, not in its place. **The owner decides** (a new revision of step 3, `docs/companies_step.md`) the threshold, or whether the model should be retrained on articles. What the 185 rows add to the choice: article precision stays between 89% and 92% from 0.1 to 0.7, so a lower threshold costs little precision; article recall goes from 69.3% at 0.5 to 83.0% at 0.3 and 97.7% at 0.1; the cost is the `no` rows called main (12 of 40 at 0.1, 6 of 40 at 0.5), and the not-main figures become unreliable once the missed-main articles are left out. The sentiment step reads `news_company.is_main`, so this choice moves its input too.

## Limits

- 104 articles, one reader, 40 `no` rows (11 to 13 once the missed-main articles are left out): the intervals are wide, the AUC of the scenarios especially, and the unweighted figures are not corpus rates.
- The one-main limitation is in the design of the task, not in the model: the owner's several mains are real, and the model was trained to pick one. The article metrics above do not count it; the row metrics do, and are labelled as secondary.
- The owner's `falta_la_principal` and the notes disagree on 17 articles; the missed group uses both, so a different reading of "missed" moves the scenario figures.
- The scenario without the missed-main articles removes the model's failures by definition: its figures are what the model does where the main company is in the list, not what it does on the step's texts.
- The 2026-11-16 and 2026-11-20 marks cover different articles (the 2026-11-16 marks have 15 articles the current file no longer has, and the current file has 60 the earlier one did not); they are not one series.
- The out-of-fold figures of the 809 summaries are not held out, and are not about sub-news.
- No second reader, and no check of the share of articles with no company named (the split between "no universe company" and "named in a form the matcher does not know" was not measured).
- The marks are the owner's judgement of "main" under the file's definition, not a count of facts.

## Reproduce

```bash
uv run scripts/build_main_company_marks_2026_11_20.py --data data   # gold/main_company_marks_2026_11_20.csv and its texts, from the local working file (refuses to overwrite); prints the differences with the 2026-11-16 marks
uv run scripts/score_main_company_marks_2026_11_20.py                # the per-article headline, the row figures and the scenarios, from gold/main_company_marks_2026_11_20.csv
uv run scripts/score_main_company_marks_2026_11_20.py --thresholds 0.1,0.2,0.3,0.5,0.7
uv run scripts/score_main_company_marks_2026_11_20.py --marks gold/main_company_marks.csv   # the 2026-11-16 version
```

`tests/news_nlp/test_main_company_marks_2026_11_20.py` and `tests/news_nlp/test_val_gold_2026_11_16.py` check that the figures of this page are the scored ones.

The out-of-fold figures and the threshold table: `scripts/relevance_threshold_candidates_2026_11_13.py` and `docs/sentiment_training.md`, "A relevance model".
