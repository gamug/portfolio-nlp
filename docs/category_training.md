# Training the category model (T-187, T-174): datasets, experiments, results and the selected model

This is the record of how the category model was studied and why one recipe is **selected** (2026-11-15). It is the base for later
documentation: the step that runs a category is `docs/category_step.md`; the taxonomy the owner tags with is
`docs/category-taxonomy.md`; the tagging sets are listed in `docs/gold_datasets.md`; the figures of every run are in
`docs/category_results_2026_11_13/` (the first measures) and `docs/category_results_2026_11_15/` (the final ones). The decision is
`PLAN.md` D43; the task is `TASKS.md` T-187; the pin is `config/category_model.json`.

**Read this first.**

* **Selected, not wired.** `--step category` still runs the *untouched* zero-shot model. The selected model is a recipe plus the data it
  is measured on (`config/category_model.json`). No weights are fitted for it, none are published, and the step does not read it.
* **The bar is not shown.** The bar of D24 is 80% agreement on the held-out 200. This model has **not** scored them (the owner's rule:
  not without the owner's word). Its figures are out-of-fold on the training rows: 87.8% accuracy, macro F1 0.748, on tags that were
  reviewed after the first results were seen (section 8). Read them as "this recipe learns the tags well", not as "the step is at 80%".
* **Wiring is another revision** of the step, and needs one refit on all rows (a fixed seed) and the held-out scored with those weights (section 9).

## 1. The result in one table

Five-fold cross-validation grouped by article, out-of-fold, the top label of each row against its single tag. Rows and tags are the final
ones (1,216 rows, 12 labels). "Untouched" is the model the step runs today; the first fine-tune was trained on the tags of the first review.

| | accuracy | macro F1 |
|---|---|---|
| untouched zero-shot model | 75.5% | 0.525 |
| entailment fine-tune, first run (older rows not yet reviewed) | 85.5% | 0.708 |
| **entailment fine-tune, second run (selected recipe, 1,216 rows)** | **87.8%** | **0.748** |

(Both runs on the 1,216 rows they share; the first run has 1,224 rows in all, 85.0% and 0.701 on them (E8). **The untouched model is scored on 1,214 of the 1,216 rows**: the earlier cross-validation never saw the two sheet rows that carried only `uncertain` (review order 37, Best Buy; 568, Fastenal), which the review later tagged; they are not a join error. `docs/category_results_2026_11_15/single_comparison.md`, `ovr_single.md`.)

## 2. What a category is here

* **12 labels, one per (article, company) row.** The taxonomy has 14 labels (D15); on 2026-11-14 the owner folded
  `competitive_actions` and `peer_comparison` into `other` (they were almost never found and their texts read as `other`), and on
  2026-11-15 chose **one tag per row**. The labels: `earnings_performance`, `mergers_acquisitions`, `leadership_governance`,
  `legal_regulatory`, `product_innovation`, `capital_shareholder_returns`, `labor_human_capital`, `market_analyst_sentiment`,
  `partnerships_business_dev`, `reputational_harm`, `investor_activity`, `other`.
* **A row is a company's segment of a text.** A body that talks about several companies gives one row per company; the tag is what the
  text says about that company.
* **The reading is entailment.** The model gets a premise `<company>: <text, first 2,000 characters>` and, for each label, a
  hypothesis `The news about <company> is about <definition>.`; the entailment probability of each hypothesis is that label's score,
  independent of the others. The output of a row is the label with the best score. The definitions are the short ones of
  `docs/category_step.md` (not the taxonomy's full rules).
* **Why this matters for the metrics:** the score of a label does not depend on how many labels there are, so the number of classes is not what
  makes a class hard (section 7).

## 3. The datasets

### 3.1 The sets and what they are used for

| set | file | rows | role | state |
|---|---|---|---|---|
| older single-label rows (set `v2`, train) | `gold/category_gold_v2.csv` → reviewed in `gold/category_v2_single.csv` | 225, 217 kept | training and cross-validation | read again 2026-11-15 (3.4) |
| the 1,000-row sheet (multi-label by design) | `gold/category_multi_gold.csv` (the owner's tags) → `gold/category_single_gold.csv` (one tag per row) | 997 tagged, 999 in the single file | training and cross-validation | tagged by the owner 2026-11-14, reviewed 2026-11-15 (3.3) |
| held-out | `gold/category_gold_v2_test.csv` | 200 (135 single rows score; 44 narrative bodies do not) | the bar of D24 | **never read by these experiments**; **already scored once**, on 2026-11-13, with the older mixed fine-tune (48.9%); a score of the selected model is the second look (T-201) |

Texts are in `gold/texts/` (`category_gold_v2_texts.csv`, `category_multi_gold_texts.csv`, `category_single_gold_texts.csv`). The
held-out is not scored again without the owner's word (it has been scored once; the next score is the second look). The training rows of the selected recipe are the sheet's 999 plus the 217 kept
older rows: **1,216 rows, 1,049 articles**.

### 3.2 The record of changes, in order

| date | change | files |
|---|---|---|
| 2026-11-12/13 | the older set `v2` (284 tagged rows; 225 single-label rows train) and the held-out 200 (`gold/category_gold_v2_test.csv`) are fixed; 14-label taxonomy, groups of option B | `gold/category_gold_v2*.csv`, `docs/category_results_2026_11_13/` |
| 2026-11-13 | the 1,000-row sheet is drawn from structured multi bodies (`scripts/draw_category_multi_2026_11_13.py`) and translated for the owner | `data/category_multi_to_tag.csv` (git-ignored) |
| 2026-11-14 | the owner tags the sheet, one to three labels per row (774 with one label, 213 with two, 10 with three: 997 rows); 3 more rows hold only `uncertain` and are left out of the multi file (so section 3.3 counts 777 rows with one tag, 774 + 3, and 223 with several); the owner folds two labels into `other` and decides on **one tag per row** | `gold/category_multi_gold.csv` (997 rows) |
| 2026-11-15 | a `claude_review` column (comments and a suggested single tag; 355 of the 1,000 rows have a comment) is added to a copy of the sheet; the owner reads it and accepts the suggestions; the single tag of a row is the suggestion, or the owner's tag when there is none. Three rows had only `uncertain`: the review tagged them `market_analyst_sentiment`; two (orders 37 and 568) are in the set, order 166, whose text has a gap where the topic should be, is left out | `data/category_multi_to_tag_claude_review.csv` (git-ignored, the full copy), `gold/category_single_gold.csv` (999 rows; **`review` keeps the comment of each row that has one**, so every change can be audited without the ignored copy) |
| 2026-11-15 | the 225 older rows are read again with the same rules: 47 tags change and 8 rows are dropped (an (article, company) pair that carried two tags keeps one); 170 are unchanged | `gold/category_v2_single.csv` (one line per row: old tag, new tag, reason) |

The original `data/category_multi_to_tag.csv` is unchanged (it could not be written; the review is a separate copy). The review copy has two
added columns, `claude_review` and `final_tag`. Scripts that build the files refuse to overwrite without `--force`.

### 3.3 What the review of the sheet changed (1,000 rows)

227 of the 1,000 rows end with a tag other than the owner's first tag (114 of the 777 rows that had one tag, 113 of the 223 that had several: the
best of several was chosen). **In the 999-row file the count is 226**: the 227th is row 166, left out. 226 is the number the audit of section 8 draws from. The largest moves:

| owner's first tag | final tag | rows |
|---|---|---|
| `investor_activity` | `market_analyst_sentiment` | 124 |
| `earnings_performance` | `market_analyst_sentiment` | 28 |
| `market_analyst_sentiment` | `earnings_performance` | 7 |
| `capital_shareholder_returns` | `mergers_acquisitions` | 6 |
| `legal_regulatory` | `product_innovation` | 4 |

Class sizes before (owner's first tag) and after: `investor_activity` 145 → 17, `market_analyst_sentiment` 54 → 200,
`earnings_performance` 488 → 469, the others within a few rows. **The convention that changed is the one that matters**: an analyst's rating or
price target is `market_analyst_sentiment`; `investor_activity` is for a fund or investor moving a stake. The taxonomy already said so
(`docs/category-taxonomy.md`); the sheet had tagged analyst-rating items as `investor_activity`: 124 rows whose first tag was `investor_activity` ended as `market_analyst_sentiment` (table above).

### 3.4 The rules of the review (the same for the sheet and the older rows)

The nine rules are kept in `docs/category-taxonomy.md`, "One tag per row", because that page is the authority for whoever tags (they are not repeated here, so there is one text to
keep). In short: an analyst rating or price target is `market_analyst_sentiment` and a stake moving is `investor_activity`; results with guidance, a delay or a share move are
`earnings_performance`; a regulator acting on the company is `legal_regulatory`; a launch built on a partner is `partnerships_business_dev` when the partner is the story and
`product_innovation` when the product is; an index change, a rate rise, a discount or a macro release is `other`; a proxy fight is `leadership_governance`; a pair gets one row.

The review of the older rows (47 retagged and 8 dropped: 8 rows of article 166583 from `market_analyst_sentiment` to `earnings_performance`; analyst
upgrades and downgrades into `market_analyst_sentiment`; the SEC rumour on Coinbase and the Tesla investigation into `legal_regulatory`;
and so on) is in `gold/category_v2_single.csv` with a reason on every line. A handful are close calls (Microsoft dismissing protesting
engineers, `labor_human_capital`; the Treasury selling Citigroup warrants, `investor_activity`); the reason column says why.

### 3.5 The classes, final (1,216 rows)

`earnings_performance` 547, `market_analyst_sentiment` 214, `product_innovation` 90, `mergers_acquisitions` 81, `legal_regulatory` 45,
`leadership_governance` 40, `capital_shareholder_returns` 40, `reputational_harm` 39, `labor_human_capital` 36, `partnerships_business_dev` 32,
`other` 32, `investor_activity` 20. **Eight classes have 45 rows or fewer; their intervals are wide (section 6).**

## 4. How a run is measured

* **Cross-validation.** Five folds, grouped by article (a body can give several rows, and no article is in train and validation together). The same
  folds for every arm of an experiment. Scores are **out of fold**: every row is scored by a model that never saw it.
* **Per class, one versus rest.** For each label: precision (of the rows predicted as it, how many have it as the tag), recall (of the rows that have
  it as the tag, how many are predicted as it), and ROC AUC (how well its score ranks the rows that have it above those that do not, computed by ranks
  from the scores of all rows; it does not depend on the top-label choice). Precision and recall use the top label of each row.
* **Intervals.** 95% intervals by bootstrap over **articles**, not rows (`category_multilabel.per_label_ci`), because rows of one article are not independent.
* **Views of the multi-label experiments (old tags).** The top label against the *first* tag; against *any* tag (a multi-tag row is right when the top label is
  one of its tags); on the *single-tag rows only* (the cleanest test of whether a class is learnable); and the set view, where extra labels join
  the top one by a rule picked on the other folds (`cross_fitted_sets`).
* **Hardware and time.** One RTX 4050 (6 GB, shared), fp16. One cross-validation of the entailment fine-tune takes about an hour.

## 5. The experiments

All on 5 folds. E1 to E7 use the **old tags** (the owner's sheet as tagged, plus the 225 older rows; 1,222 rows read as 12 labels,
`docs/category_results_2026_11_15/arms_old_tags.md`); E8 onwards the new ones. Out-of-fold scores of E1 to E7 are in
`data/category_multilabel/` (git-ignored); the final run's predictions, without text, are in `docs/category_results_2026_11_15/single_predictions.csv`.

### E0. The untouched model and the first fine-tune (2026-11-13)

The pinned zero-shot model, no training: 56.4% on the 225 train rows and 43.0% on the 135 held-out single rows (group 62.7% / 55.6%). A fine-tune
of the entailment head with one tag per row (the "mixed" recipe): 62.2% out of fold, 48.9% on the held-out single rows. Under the bar, not published.
(`docs/category_step.md`, `docs/category_results_2026_11_13/`.) This is what the 1,000-row sheet was drawn to improve on.

### E1. Untouched model on the 1,222 old-tag rows

The top label against the first tag: 59.7%; against any tag: 66.0%; on single-tag rows: 65.0%; macro F1 0.429. `market_analyst_sentiment`
precision 0.15 (it takes every stock-move line), `other` 0.14.

### E2. Entailment fine-tune, first tag the only positive (the old recipe): **no result**

Pairs: the first tag a positive, four labels that are *not* gold tags as negatives. **Stopped before it finished; no scores were kept**, so it has no
figures and is not an arm of any table. The 2026-11-13 "mixed" fine-tune (E0) is the earlier figure of the same recipe, and E3 reads the same rows with every tag a positive. The number is kept so that E3 to E9 keep theirs.

### E3. Entailment fine-tune, every tag a positive

Same pairs, but **every** tag of a row is a positive and a negative is never a gold tag, so a multi-tag row teaches both labels. Top label: 79.4% against
the first tag, 82.7% against any tag, 82.2% on the single-tag rows; macro F1 0.654. By far the best arm on the old tags (+20 points over the untouched model).

### E4. E3 with a rule for extra labels (set view)

The top label always; a second or third label (cap 3) when its score is at least `min_score` and at least `ratio` times the top, the pair chosen on the
other folds. It lifts recall on the labels that are often second (`legal_regulatory` 0.79, `labor_human_capital` 0.79, `reputational_harm` 0.75) at some cost in
precision. `market_analyst_sentiment` stayed low (precision 0.64, recall 0.37, ROC AUC 0.86) and `other` near zero (recall 0.06): the next experiments say why.
**This rule is not part of the selected recipe**: with one tag per row the output is the top label.

### E5. A multi-label head on the zero-shot encoder

The zero-shot model's encoder with its 3-way head replaced by 12 sigmoid outputs (BCE), reading `<company>: <text>` once per row. **A failure**: 65.0%
against the first tag, macro F1 0.267; `leadership_governance`, `labor_human_capital`, `partnerships_business_dev` and `other` are never predicted. A likely reason (not tested): the encoder
of an NLI model is tuned to read a pair, and ~1,200 rows are too few to train a new head on it.

### E6. Plain DeBERTa-v3-base, 12 sigmoid outputs

`microsoft/deberta-v3-base` (its tokenizer is the zero-shot repo's; `spm.model` is byte-identical), 12 outputs, BCE with class weights, 10 epochs, lr 3e-5,
head lr 5e-4. 72.5% against the first tag, 77.5% against any, macro F1 0.585. Good for `investor_activity` (the class the old sheet over-tagged) and
`mergers_acquisitions`, weak for `market_analyst_sentiment` (recall 0.25), `partnerships_business_dev` (0.21) and `other` (0.13). Not retrained on the new tags.

| arm (old tags, 1,222 rows) | top = first tag | top is any tag | single-tag rows | macro F1 |
|---|---|---|---|---|
| untouched zero-shot | 59.7% | 66.0% | 65.0% | 0.429 |
| E3 entailment fine-tune, every tag a positive | **79.4%** | **82.7%** | **82.2%** | **0.654** |
| E5 head on the zero-shot encoder | 65.0% | 66.6% | 65.2% | 0.267 |
| E6 plain DeBERTa-v3-base | 72.5% | 77.5% | 76.8% | 0.585 |

### E7. Why the per-class numbers were low (the owner's challenge, 2026-11-14/15)

The owner would not accept low precision and recall on classes that are "very clear" and asked whether the number of classes was the cause. What was checked:

1. **Top-only scoring penalises multi-tag rows.** A row with two tags can match only one with a single top label. Reading against the first tag or on the single-tag rows only
   lifted every arm that was tried (E3: 79.4% against the first tag, 82.2% on the single-tag rows).
2. **The number of classes is not the cause.** Each hypothesis is scored independently; adding labels does not change another label's score. The classes that stayed weak
   (`market_analyst_sentiment`, `partnerships_business_dev`, `other`) were the ones whose *tags* disagreed with the taxonomy.
3. **The tags had a convention problem.** The tag audit (ids returned to the owner, no edits) found the analyst-rating rows tagged `investor_activity` (3.3). The entailment model had,
   correctly, learned to score "upgraded to buy" high for `market_analyst_sentiment` and was punished for it: the old-tag model on the *new* tags has `investor_activity` precision 0.11.

### E8. First run on the new tags (2026-11-15, 04:04)

The sheet's 999 rows with their single tags plus the 225 older rows still with their old tags (1,224 rows, 12 labels), entailment fine-tune as E3 with one positive per row,
3 epochs, lr 2e-5. Accuracy **85.0%**, macro F1 **0.701** on its 1,224 rows (85.5% and 0.708 on the 1,216 rows the second run also has; the comparison tables use the latter). On the same rows and the same tags the untouched model scores 73.9% (it was 66.0% on the old tags: a part of the
gap was the tags) and the E3 model trained on the old tags 73.6% (`investor_activity` precision 0.11, `market_analyst_sentiment` recall 0.16). Of the older rows, article 166583 (ten
companies, all tagged `market_analyst_sentiment`, the text is earnings reactions) was still wrong in training.

### E9. Second run: the older rows read again (2026-11-15, 13:58) — the selected recipe

The 225 older rows read with the one-tag rules of the taxonomy (47 retagged, 8 dropped): 1,216 rows. **87.8% accuracy, macro F1 0.748.**

| view | first run | second run |
|---|---|---|
| each run on its own tags | 85.5% / 0.708 | **87.8% / 0.748** |
| both runs on the second tags (is it the training or the tags?) | 87.9% / 0.738 | 87.8% / 0.748 |
| the older rows only (217 matched), each on its own tags | 70.0% / 0.565 | 81.6% / 0.714 |
| the older rows only, both on the second tags | 83.4% / 0.719 | 81.6% / 0.714 |
| the sheet's 999 rows only (tags unchanged) | 88.9% / 0.748 | 89.2% / 0.759 |

Retraining on the cleaned older rows changed almost nothing for the model: the sheet's 999 rows had already taught it the boundaries, and the older rows' old tags
simply disagreed with it (`market_analyst_sentiment` on those rows: precision 0.35, recall 0.30, ROC AUC 0.69 on the old tags; 0.67, 0.86, 0.95 on the new). **The gain
is mostly the cleaner measure**, not a better model; the second run is selected because it is trained on data that agrees with the rules.

## 6. The selected model, per class

Out of fold, 1,216 rows, 1,049 articles, 95% intervals by article (`docs/category_results_2026_11_15/ovr_single.md`):

| class | rows | predicted | precision [95%] | recall [95%] | ROC AUC |
|---|---|---|---|---|---|
| earnings_performance | 547 | 552 | 0.97 [0.95, 0.98] | 0.97 [0.96, 0.99] | 1.00 |
| mergers_acquisitions | 81 | 90 | 0.80 [0.71, 0.88] | 0.89 [0.81, 0.95] | 0.99 |
| leadership_governance | 40 | 41 | 0.88 [0.77, 0.97] | 0.90 [0.80, 0.98] | 1.00 |
| legal_regulatory | 45 | 46 | 0.72 [0.58, 0.85] | 0.73 [0.60, 0.85] | 0.98 |
| product_innovation | 90 | 89 | 0.87 [0.79, 0.94] | 0.86 [0.78, 0.92] | 0.99 |
| capital_shareholder_returns | 40 | 36 | 0.78 [0.63, 0.92] | 0.70 [0.56, 0.84] | 0.96 |
| labor_human_capital | 36 | 37 | 0.84 [0.71, 0.94] | 0.86 [0.75, 0.97] | 1.00 |
| market_analyst_sentiment | 214 | 212 | 0.89 [0.85, 0.93] | 0.88 [0.84, 0.92] | 0.98 |
| partnerships_business_dev | 32 | 31 | 0.77 [0.62, 0.90] | 0.75 [0.60, 0.90] | 0.97 |
| reputational_harm | 39 | 41 | 0.71 [0.57, 0.85] | 0.74 [0.61, 0.88] | 0.98 |
| investor_activity | 20 | 20 | 0.50 [0.26, 0.73] | 0.50 [0.27, 0.73] | 0.93 |
| other | 32 | 21 | 0.29 [0.11, 0.50] | 0.19 [0.06, 0.36] | 0.80 |

**Strong:** `earnings_performance`, `market_analyst_sentiment`, `leadership_governance`, `labor_human_capital`, `product_innovation`, `mergers_acquisitions`: precision and recall of 0.80 to 0.97 and ROC AUC 0.98 or more.
**Middle:** `legal_regulatory`, `capital_shareholder_returns`, `reputational_harm`, `partnerships_business_dev`: 0.70 to 0.78, ROC AUC 0.96 to 0.98; intervals of ±0.12 to ±0.15.
**Weak:** `investor_activity` (20 rows; recall 0.50 with an interval of 0.27 to 0.73) and `other` (32 rows; precision 0.29, recall 0.19, ROC AUC 0.80).

### 6.1 What the weak classes show

* **`other` is a definition problem, not a model one.** It is "an event that fits none of the other topics"; its rows are one-line blurbs inside "stocks making the biggest moves" articles (an index
  addition, a rate rise at a competitor, a discount) and the model often reads them as a price move: 7 of its 26 missed rows go to `market_analyst_sentiment`, the rest are spread (5 to `earnings_performance`, 3 to `capital_shareholder_returns`, 2 to `product_innovation`). On the older rows its precision and recall are 0.14 and 0.06. The rule is written (the taxonomy, "One tag per row", rule 7),
  but 32 rows do not teach it.
* **`investor_activity` is small now.** After the review only 20 rows are a fund or investor moving a stake (145 before). The misses go to `market_analyst_sentiment` with `investor_activity` second.
* **`market_analyst_sentiment` and `partnerships_business_dev` errors are related to the label the model gives.** In the selected run, 13 of the 23 false positives of `market_analyst_sentiment` are gold `other` (7) or
  `investor_activity` (6), and its 25 misses are spread thinly; `partnerships_business_dev` has 8 misses (3 to `mergers_acquisitions`, 2 to `product_innovation`, 2 to `other`, 1 to `market_analyst_sentiment`) and 7 false
  positives (3 gold `product_innovation`). In the **first** run the same class lost 6 of 18 to `product_innovation`, launches built on a partner (Coinbase and Cloudflare's x402 Foundation, Coinbase prediction markets
  "powered by Kalshi", an American Express plan for merchants), with `partnerships_business_dev` ranked second; rule 5 of the taxonomy settles them, and 2 remain in the second run. Both tags defend themselves in a few rows.
* **The intervals do not get narrower by cleaning.** Their width follows the number of rows. Mean interval width of precision: 0.27 (first run) and 0.24 (second); recall 0.23 and 0.22. A class with 20 to 45 rows stays at ±0.12 to ±0.15 (a
  0.25 to 0.30 range), whatever the model. **A difference of a few hundredths in a small class is not a finding**: partnerships precision 0.66 → 0.77 and `investor_activity` recall 0.35 → 0.50 between the runs sit inside their intervals. Only
  `earnings_performance`, `market_analyst_sentiment`, `mergers_acquisitions` and `product_innovation` are tight enough to compare runs by.

## 7. Findings

1. **Fine-tuning the entailment head pays**: +20 points on the old tags (59.7% → 79.4%), and on the final tags 75.5% → 87.8% (macro F1 0.525 → 0.748).
2. **The reading that works is the entailment reading.** A new head on the same encoder (E5) and a plain encoder with sigmoid outputs (E6) are both worse with ~1,200 rows.
3. **Most of what looked like model weakness was in the tags.** Reading analyst ratings as `market_analyst_sentiment` took the untouched model from 66.0% to 73.9% on the same rows, and the fine-tune from 73.6% (trained on the old tags) to 85.0%.
4. **One tag per row** is what the owner decided and what the model is trained on; the multi-label set view is not used.
5. **Two classes are not solved by this data**: `other` (a definition) and `investor_activity` (20 rows).

## 8. Caveats (read before quoting a figure)

* **The tags were reviewed after seeing results.** The review rules (the taxonomy, "One tag per row") were written after the first per-class tables showed which classes failed, and the single tag of a multi-tag row was chosen by
  the reviewer (the owner accepted it). 87.8% is a measure of how well the recipe learns *these* tags; it is not a clean held-out figure and not comparable with the 48.9% of 2026-11-13.
* **The bar of D24 (80% on the held-out single rows) is not shown.** The held-out 200 was tagged under the earlier convention (analyst ratings partly as `investor_activity`); scoring it with this model would
  need the owner's word, and the held-out tags may need the same review first.
* **Wide intervals** for 8 of the 12 classes (6.1).
* **Rows of one body.** Folds are grouped by `article_id`, and about a third of the `ok` bodies of the corpus are shared by several tickers (`docs/category_step.md`), so the grouping might put two copies of one body in two folds. **Measured** with `scripts/category_body_overlap_2026_11_15.py` (`--source-db` the crawl database; output `docs/category_results_2026_11_15/body_overlap.md`, from the committed folds): the 1,216 rows are 1,049 articles and 1,048 distinct bodies (`triage.body_hash`); **one** body is the text of two articles (116568 and 439320) and they sit in two folds: 16 rows (1.3%), all 16 predicted right; without them the accuracy is 87.7% (1,200 rows), against 87.8%. The leak is not a source of the figure, so the cross-validation was not repeated with a body grouping. Why the sheet has so few shared bodies was not investigated.
* **Long texts.** The fine-tune reads at most 512 tokens (the first 2,000 characters of the text), as the step does; nothing was measured on what a longer text would change.
* **The review of the sheet, the single tags and the older-rows tags are mine, accepted by the owner** (the sheet) or **not yet read by the owner** (the older rows' `gold/category_v2_single.csv`). **No second reader has checked the reviewer's changes** (226 tags of the sheet changed, 47 of the older 225; section 3.3). T-200 asks the owner to read a blind sheet of **75 rows**: 50 changed rows (the review's tag against the owner's first) and 25 unchanged rows (the row's tag against the label the model scored highest among the others), drawn at random (`scripts/draw_category_review_audit_2026_11_15.py`, seed 1) and mixed. Read the figures as: the share of picks of the review's tag on the changed rows, and of the row's tag on the unchanged ones; with 50 and 25 rows the 95% intervals are about ±0.13 and ±0.2. The rule for what they mean is a table by outcome (changed rows low, mid or high; unchanged rows low, mid or high) in `TASKS.md` T-200, written before the sheet was read: it says which rows the owner re-reads in each case, and the figure of 87.8% is quoted with the two agreements beside it. The reading is not fully blind: on a changed row one option is the owner's own earlier tag, and the owner may lean to it.

## 9. What is pinned and what is not

`config/category_model.json` records the **selected recipe**: the start model and revision (the pin of `config/pipeline_models.json`), the 12 labels, the pair recipe and hyper-parameters, the data files with their
SHA-256, and the evidence. A test (`tests/news_nlp/test_category_selection.py`) fails when a tag file changes without the pin changing, so these figures cannot silently describe other data.

**Not done, in order, each on the owner's word:**

1. **Refit once on all 1,216 rows, with a fixed seed, and keep the weights**: `uv run scripts/category_single_cv_2026_11_15.py refit` (GPU, the owner's word). The seed (`recipe.refit_seed` = 100, not the seed of a fold, 0 to 4) and the hyper-parameters come from `recipe` in the pin and from nowhere else: the command refuses a flag that differs, `--seed`, `--smoke` and `--folds`, a data file that is not the pinned one, and an output folder that is not empty. It saves the weights and the tokenizer to the git-ignored `data/models/category_2026_11_15/` with `refit_record.json` (seed, recipe, data hashes, git commit, device, library versions, sha256 of `model.safetensors`). A second refit is not expected to give the same weights (GPU fp16, no deterministic mode, not measured): the hash of the kept folder is its identity, as the sentiment pair model is identified (`config/sentiment_step.json`).
   A private Hub copy is a record (`config/published_models.json`), uploaded by a script like `scripts/publish_final_models_2026_11_10.py`, of that folder; it needs `HF_TOKEN` with write scope and the owner's word.
2. **Score the held-out 200 with those same weights** (`scripts/category_score_2026_11_15.py --heldout --why "..."`: it refuses a folder with no refit record or with other weights, needs the reason and the tag set, and refuses a repeat), the **second** look at them (the first, on 2026-11-13, was the older mixed fine-tune, 48.9%; both scores stay in the record), after deciding whether its single rows get the one-tag review first; the bar is 80%. The weights that are scored are the ones that get wired.
3. **Wire it into `--step category`.** This is a new revision of the step: the step's tables (14 labels, groups, definitions) are pinned by a test and have to change to 12 labels or map the 12 back to the 14; `LEGACY_REVISIONS`
   is retired in that change; `docs/category_step.md` and FR-018 change; the old rows of the untouched revision stay.
4. **More rows for the small classes.** About 100 rows per class would bring the intervals to ±0.08. The cheapest source is the rows the model already scores low on for `investor_activity`, `other`,
   `partnerships_business_dev`, `legal_regulatory` and `capital_shareholder_returns`.
5. **Settle `other`** (a written definition, or an abstention) and **partner versus product** (rule 5 of the taxonomy).
6. Baselines not run: embeddings with a logistic regression, and an LLM few-shot reader (paid; confirm the spend first).

## 10. Reproduce

The out-of-fold scores of every run are **committed without text** in `docs/category_results_2026_11_15/scores/` (`category_multilabel/` the four arms of the old tags and their rows, `category_single_v1/` the first run, `category_single/` the second, selected run; about 1 MB). The three report scripts read `data/<name>/` when a run left it there and the committed copy otherwise (`category_multilabel.scores_dir`), so **the three report commands below run from a clean checkout** and give the committed tables. Only the cross-validation itself (the first command, about an hour on the GPU) needs the gold files and writes to the ignored `data/`. The owner's sheet and its review copy (`data/category_multi_to_tag*.csv`) stay local; the review comments of every row are in `gold/category_single_gold.csv` (`review`).

```
uv run scripts/category_single_cv_2026_11_15.py cv            # the selected recipe, 5 folds -> data/category_single/oof_single.npz (~1 h on the GPU)
uv run scripts/category_body_overlap_2026_11_15.py --source-db <urls.db>   # shared bodies in two folds (committed folds; reads SOURCE)
uv run scripts/category_single_report_2026_11_15.py --out docs/category_results_2026_11_15   # per-class precision, recall, ROC AUC, intervals
uv run scripts/category_single_compare_2026_11_15.py --out docs/category_results_2026_11_15  # first run -> second run; predictions without text
uv run scripts/category_arms_summary_2026_11_15.py --out docs/category_results_2026_11_15    # the arms on the old tags
```

Rebuilding the data: `scripts/build_gold_category_multi_2026_11_14.py` (the owner's tags), `scripts/build_gold_category_single_2026_11_15.py` (one tag per row from the review copy),
`scripts/build_gold_category_v2_single_2026_11_15.py` (the older rows read again; the table of changes is in the script). The arms on the old tags:
`scripts/category_multilabel_cv_2026_11_14.py` (modes `base`, `first`, `multi`), `scripts/category_head_cv_2026_11_14.py`, `scripts/category_deberta_cv_2026_11_14.py`,
`scripts/category_ovr_report_2026_11_14.py` (these four need the local `data/category_multilabel/`; their scores are in the committed folder). The torch-free helpers (parsing labels, the set rule, folds, per-class figures, intervals, ROC AUC, the merge) are `src/category_multilabel.py`
(`tests/news_nlp/test_category_multilabel.py`).

## 11. File index

| what | where |
|---|---|
| the pin (selected recipe) | `config/category_model.json` |
| the tests of the pin and the tag files (hashes; the accuracy and macro F1 of the pin against the committed scores; the untouched model's rows; the older-rows table against the rows of the earlier runs; a review comment for every changed tag; the hash of the definitions, hypothesis and premise) | `tests/news_nlp/test_category_selection.py` |
| final results | `docs/category_results_2026_11_15/`: `ovr_single.md` (second run, per class, with intervals, and the three models), `ovr_single_first_run.md`, `single_comparison.md`, `arms_old_tags.md`, `single_predictions.csv` (no text), `scores/` (the out-of-fold scores of every run, no text) |
| the first results | `docs/category_results_2026_11_13/` |
| tag files | `gold/category_multi_gold.csv`, `gold/category_single_gold.csv`, `gold/category_v2_single.csv`, texts in `gold/texts/` (list: `docs/gold_datasets.md`, rows 13 to 15; the sheet's `review` column holds the comment behind each change) |
| the refit, the scoring of its weights, the body overlap and the audit sheet | `scripts/category_single_cv_2026_11_15.py refit`, `scripts/category_score_2026_11_15.py`, `scripts/category_body_overlap_2026_11_15.py` (output `docs/category_results_2026_11_15/body_overlap.md`, its figures tested), `scripts/draw_category_review_audit_2026_11_15.py` (tests: `tests/news_nlp/test_category_refit.py`, `test_category_review_audit.py`) |
| the tagging rules ("One tag per row"), the definitions | `docs/category-taxonomy.md`, `docs/category_step.md` |
| the decision and the task | `PLAN.md` D43, `TASKS.md` T-187 |
