# Precision and recall (one-vs-rest) of the pipeline steps

Date: 2026-11-16 (the figures of sections 5.1 and 3.3 are of 2026-11-16 and 2026-11-15; the rest, of the dates each section gives). The number in a heading is the pipeline's own step number; the sections follow the order the steps run. "The owner" is the project's term for the person who marks and decides. Steps covered, in pipeline order: scope (with the sector step), cut, NER and the company pick, summarization, sentiment, category.

Every figure below is taken from the repository's documents and tracked result files, or recomputed from tracked files (the category
and sector tables were recomputed for this report from `docs/category_results_2026_11_13/*.csv` and `gold/sector_validation.csv`; the category
candidate's figures are the files of PR #196, `docs/category_results_2026_11_15/`, and its MLflow runs, section 6). **The figures measured on the real corpus run are
the category real-data sample (6.4), the main-company marks on pieces (3.3) and the sentiment check on 190 pieces (5.1)**; the category sample was read from the RESULTS store (read-only: `news_category`, `news_summary`, `sub_news`); all others are from tracked or
result files.

**One-vs-rest (OVR)** means: for each class, precision = of what the step called that class, the share that is right; recall = of the true members of
that class, the share the step found. Where a step has no classes of that kind, or the owner never measured them, the table says so instead of
inventing a figure.

**Which metrics are kept.** For every model trained in the project (the scope classifiers, the cut's pair classifier, the relevance model, the sentiment pair model and stack, the category candidate) this report keeps two kinds of metrics only: **training-validation** (cross-validation, out-of-fold and validation-split scores) and **held-out** (scores on texts the model was neither trained nor tuned on).
What was removed in this version is every other measure: the share of a run's items that a step labelled (a dry-run output share) and the counts of a corpus run, which say what a model produces, not whether it is right. The dataset descriptions, the model-selection narratives and the limitations are kept. Models not trained here (summarization, the category step's zero-shot model, the rules) keep the figures they had.
The removed shares and counts are still in the step documents (`docs/companies_step.md`, `docs/main_company_training.md`, `docs/summary_step.md`), which are not pipeline-wide reports and keep them. Dataset descriptions keep the label-quality figures of their sets (for example the label check of the cut's training labels): they describe the data, not a model.

## Summary

| Step | What is measured | Has per-class OVR precision and recall? | Headline | Held out? |
|---|---|---|---|---|
| Scope | one of 4 classes per text (`single`, `multi`, `sector`, `out_of_scope`) | **Yes**, four classes (section 1.1) | held-out agreement 83.6% [78.1, 88.9]; `out_of_scope` recall 53%; quarantined precision 90.8% | Yes (200 texts, seen many times), plus 225 read by the owner |
| Sector | the 2-digit GICS sector assigned to a `sector` text | **Precision only** (per sector, 2 to 5 texts each); recall not measured | 87.5% of the assigned sectors right (35 of 40) | Yes (a draw made after the step was built), one reviewer |
| Cut | the pieces ("sub-news") the cut makes from a multi-company text | **No classes.** Three yes/no questions per piece (precision-like) | 84.6% of pieces faithful; 75.5% clean | Yes (28 texts, 143 pieces) |
| Company: NER | the organisation spans NER finds | Not per class on the current texts; one older judged run (v1, whole articles) | recall 0.84, precision 0.66 (older run) | Judged by an LLM, not a gold |
| Company: relevance (which company is main) | is a (text, company) pair "main" | Yes, for the two classes, out-of-fold | main-company precision 90.0%, recall 94.7% at threshold 0.5 (cross-validation on summaries); **on 119 marked rows of 59 pieces, the texts the step reads: precision 90.9%, recall 30.0%** | **Yes**: the owner's marks on pieces the model never saw (section 3.3, partial: 119 rows of 59 pieces); the 94.7% is cross-validation on 809 marked summaries, not held out |
| Summarization | the summary of a text over about 2,000 characters | **No classes.** LLM-judge scores (1 to 5) | **DeepSeek, the step's summariser since T-206:** faithfulness 4.99, coverage 4.54, hallucination 0.7% (300 articles); the earlier distilbart step: 4.60, 3.24, 15.0% | The judge is DeepSeek too (it graded its own summaries); not a gold; the corpus run is not measured |
| Sentiment | positive, negative, neutral for a (text, company) pair | **Yes**, three classes | score 0.814; negative P/R 0.747 / 0.748 (cross-validation); on 190 real pieces: accuracy 73.7%, negative precision 69.4% | **No** for the 0.814 (nested cross-validation, 809 rows); **yes** for the 190 pieces (section 5.1) |
| Category | 14 labels (6 groups) for a (text, company) pair; the candidate reads 12 | **Yes**, per label (precision, recall and ROC) | the step's model: 43.0% held-out. **Candidate (not wired): 54.1% held-out, 71.3% on 300 real sub-news (assistant's tags), 87.8% out of fold (reviewed tags)** | Yes (135 single rows, scored once); the bar of 80% is **not met** |

**Read the last column first.** Only the scope, the cut, the category, the sector clean test and the two checks on pieces of the real run (the owner's main-company marks, section 3.3, and the 190-piece sentiment check, section 5.1) are measures on texts the model or the rules did not choose
their settings on; the cross-validation figures of the relevance model and of the sentiment stack are out-of-fold and not held out. None of the steps has been scored by a second
reader. The bar the owner set for the labels is 80% (D24 for category; 70% for sentiment negative precision and recall, D32).

## 1. Scope (step 1, `docs/scope_training.md`)

Two fine-tuned NLI classifiers on `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (stage A: company / sector / not related; stage B: single / multi) give each text one of four classes:
`single`, `multi`, `sector`, `out_of_scope`. The model reads the title and the first 800 characters. `out_of_scope` stops the text; nothing else is computed for it.

### Dataset description

All sets are texts of financial news (articles from the crawl; the held-out sample is 2007 to 2026, 193 of 200 from one news site, 6 from a finance portal, 1 from a research site). Each text is one unique body; the model reads its title and first 800 characters.
Tags are one of six scopes (`single`, `multi`, `sector`, `market_economy`, `investing_general`, `unrelated`); the classifier merges the last three into `out_of_scope`, giving four classes.

| Set | Texts | Class counts | Columns | How it was drawn |
|---|---|---|---|---|
| Train 1 (`category_gold_v2.csv`) | 284 | single 99, multi 78, sector 43, unrelated 33, market_economy 17, investing_general 14 | `article_id`, `session`, `ticker`, `scope`, `company_labels`, `mentions`, `sub_industry`, `notes` | 240 bodies fixed for a prompt gate (stratified on the v1 scope) and 44 sector bodies |
| Train 2 (`scope_enrich.csv`) | 51 | mostly sector (26), single 14, market_economy 7, the rest 1 or 2 each | as above plus `rule` | drawn by rules to hold more `sector` and `out_of_scope`; easier than the corpus |
| Train 3 (`scope_hook.csv`) | 30 | single 18, market_economy 7, investing_general 3, sector 2 | as above | drawn by rule: one universe company in the title and at least three market or macro terms in the first 800 characters |
| **Held-out** (`category_gold_v2_test.csv`) | 200 | single 116, multi 44, market_economy 14, unrelated 10, investing_general 9, sector 7 | as above plus `stratum` and `frame` | drawn from a 1,000-text sample of the corpus; `stratum` and `frame` give the weight of each text to the corpus (the frame sizes sum to 1,000) |
| Quarantined validation (`scope_quarantined_validation.csv`) | 225 | 100 predicted `out_of_scope`, 50 `single`, 50 `multi`, 25 `sector` | `draw`, `review_order`, `article_id`, `ticker`, `predicted_scope`, `validator_first`, `correct_first`, `validator`, `correct_scope`, `changed`, `notes` | drawn per predicted class from the texts the step set apart as `quarantined`; none in a gold file; a yes or no per text, with the scope it should be when no |

The texts of the held-out carry `article_id`, `ticker`, `company`, `pub_date`, `source_domain`, `title` and `text`; the text is 2,557 characters on average (median 2,260, maximum 6,020; the training texts average 3,123, median 2,899).
The held-out was re-reviewed once against written rules (14 scope changes) after its first results were seen, so two keys exist and figures must name the key.

**The sector step (1.2) has its own sets.** Labelled data: 70 single-sector texts of the tagged gold rolled up to the 2-digit GICS sector, and 100 real rows of a first yes/no review (a sector shown, `correct` yes or no), each split by position (even rows tune, odd rows test).
The clean test (`sector_validation.csv`, `review` = `fresh`) is 57 rows drawn with the final configuration, by quota: 40 sector assigned, 12 sent `out_of_scope`, 5 controls (the least-similar sector shown, `no` expected). Columns: `review`, `review_order`, `article_id`, `shown_sector`,
`control`, `correct`, `right_sector`, `notes`, `status`, `candidate`, `margin`, `vote_sector`, `sure`, `step_revision`. Its 167 rows (110 first review, 57 clean) are a measure of the step, never training data.

### Model selection

The task: say `single`, `multi`, `sector` or `out_of_scope` from the title and the opening, with the bar at 80% agreement (lowered from 90% on 2026-11-14). Weighted agreement on the held-out (the earlier key):

| Approach | Agreement | What it taught |
|---|---|---|
| Always `single` | 58.2% | the floor |
| Current rules (`size_triage`) | 68.2% | good on `single` (91%), almost blind to `sector` and `out_of_scope` |
| TF-IDF and logistic regression, title | 66.4% | a bag of words cannot count companies |
| Title plus matcher features | 68.1% | no gain over the rules |
| NER plus relevance features | 71.1% | a real gain, weak on `sector` and `out_of_scope`; 4.6 h for the corpus |
| Two stages on sentence embeddings (bge-base) | 74.5% | better balanced; the size was picked after seeing three on the same 200 |
| Two NLI stages, zero-shot and fine-tuned on 284 | 72.4% and 73.4%; stage B alone 90.9% among company texts | no embedding step needed |
| **Two NLI stages, both fine-tuned, end to end** | 79.9%, then 83.5% with 51 more rare-class tags | best; `sector` recall 82% |
| Same, plus 30 hook tags as first tagged | 80.4% | contradictory tags found; tags reviewed against written rules |
| **Selected: the pinned model after the tag review** | **83.6% [78.1, 88.9]** | paired gain over the rules +16.5 [9.6, 23.4] |

The selected model is two fine-tuned copies of the pinned `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (stage A company / sector / not related; stage B single / multi). It was chosen for its paired gain over the rules, because it reuses a model already pinned (no new model, no embedding step), and because
it costs two short forward passes per text. A known trade-off was accepted on purpose: `out_of_scope` recall is about half while `single` recall is 92%, because a `single` read as `out_of_scope` stops at step 1 and never reaches the company views, and the reverse only costs compute.
The sector step chose the small embedder `BAAI/bge-small-en-v1.5` with sector descriptions v2, four "not a sector" classes and a gate (margin or company votes agree, a tie is not agreement); larger embedders gained 11 to 15 points of precision on one half and lost 12 to 18 on the other, and were not adopted.

### 1.1 Scope classifier

**Held-out 200 texts** (never trained on). One-vs-rest, recomputed from the saved predictions (`data/scope_end_to_end_with81b.csv`, the key of the first fit: 26 `out_of_scope`).
Weighted means weighted to the corpus by stratum.

| Class (gold n) | Precision | Recall | Weighted precision | Weighted recall |
|---|---|---|---|---|
| `single` (118) | 85.7% | 91.5% | 86.9% | 91.5% |
| `multi` (47) | 97.4% | 78.7% | 97.2% | 79.5% |
| `sector` (9) | 46.2% | 66.7% | 47.0% | 82.4% |
| `out_of_scope` (26) | 65.2% | 57.7% | 55.2% | 49.3% |
| Overall agreement | 83.0% | | 83.5% | |

The pinned weights, scored by the doc on the new key (33 `out_of_scope`), give **83.6% [78.1, 88.9]** weighted agreement (the rules reach 67.2%; paired gain +16.5 [9.6, 23.4]) and `out_of_scope` recall 53% [32, 74];
the bar the owner set is 80%. Those per-class figures were not recomputed here (the key differs), so the table above is the first-fit run, not the pinned weights.

Confusion matrix (rows gold, columns predicted):

| Gold \ Predicted | single | multi | sector | out_of_scope |
|---|---|---|---|---|
| single | 108 | 0 | 4 | 6 |
| multi | 8 | 37 | 1 | 1 |
| sector | 2 | 0 | 6 | 1 |
| out_of_scope | 8 | 1 | 2 | 15 |

**Quarantined sample** (225 texts the owner read; the draw was by predicted class, so this is precision only):

| Predicted class | Read | Right (final key) |
|---|---|---|
| `out_of_scope` | 100 | 93.0% [86.3, 96.6] |
| `single` | 50 | 90.0% [78.6, 95.7] |
| `multi` | 50 | 82.0% [69.2, 90.2] |
| `sector` | 25 | 88.0% [70.0, 95.8] |
| All, weighted | 225 | **90.8% [86.9, 94.7]** |

**What is lost.** On the held-out, real company or sector news read as `out_of_scope`: 8 of 174 (4.6% weighted), and 7 of 100 `out_of_scope` calls of the quarantined sample were wrong (4 of the 7 were sector stories).
Seven of the 8 held-out losses name the company in the title or the first 800 characters, which is all the model reads, and the errors are not concentrated in texts naming several companies (10 of the 23 wrongly scoped news texts, against 47% of all news texts).

**Limits.** The held-out was seen many times and its key was corrected after the errors were visible; its `sector` and `out_of_scope` intervals are wide (9 and 26 texts). Seed noise was barely measured (three fits, all seed 1).
The two figures above (83.6% and 90.8%) are different populations and keys.

### 1.2 Sector (step 4, `docs/sector_step.md`)

The step gives each `sector` text a GICS sector (11 classes) from sentence embeddings (bge-small, pinned), the sector votes of the companies the text
names, and four "not a sector" classes. The gate keeps an answer when the embedding margin is large or the company votes agree; otherwise the text
gets no sector (44% of the sector texts get none, by design).

**The clean test** (2026-11-13): 57 rows drawn by quota with the final configuration, from texts in neither the gold nor the first review; the owner answered yes or no
(`gold/sector_validation.csv`, set 12). 40 sectors assigned, 5 controls, 12 sent out.

| Measure | n | Right | 95% interval |
|---|---|---|---|
| Assigned sector is right (vote rule of the draw) | 40 | 35 (87.5%) | 74 to 95% |
| The same under the rule as it is now (a tie is not agreement) | 39 | 35 (89.7%) | 76 to 96% |
| Controls (least similar sector shown) | 5 | 0 yes | |
| Sent out of scope and the owner agrees | 12 | 7 (58%) | 32 to 81% |

**Precision per sector** (assigned in the clean test, recomputed from the tracked file; 40 rows): the per-sector counts are tiny, and the doc says the
per-sector figures are **not usable**. They are listed so they can be seen, not quoted.

| GICS code | Sector | Assigned | Right | Precision |
|---|---|---|---|---|
| 10 | Energy | 3 | 3 | 1.00 |
| 15 | Materials | 3 | 2 | 0.67 |
| 20 | Industrials | 4 | 4 | 1.00 |
| 25 | Consumer Discretionary | 5 | 4 | 0.80 |
| 30 | Consumer Staples | 2 | 1 | 0.50 |
| 35 | Health Care | 4 | 3 | 0.75 |
| 40 | Financials | 5 | 4 | 0.80 |
| 45 | Information Technology | 5 | 5 | 1.00 |
| 50 | Communication Services | 4 | 4 | 1.00 |
| 55 | Utilities | 2 | 2 | 1.00 |
| 60 | Real Estate | 3 | 3 | 1.00 |
| | **All** | **40** | **35** | **0.875** |

The 5 wrong ones: two were not sector texts (investing, market), three were another sector (a European internet note read as consumer discretionary,
a dollar-store story read as staples, a food story read as health care).

**Recall: not measured per sector.** What is known:
- 44% of the `sector` texts get no sector. Of the 12 sent out in the test, the owner would have given a sector to 5, and the step's best guess was that sector in 4.
  Estimated with the corpus rates (on 12 texts, very wide): per 100 `sector` texts, 56 are assigned, about 50 of them right; the gate gives up roughly a fifth of the
  right sectors available (about 15 of 65).
- Materials, Consumer Staples and Utilities are the sectors the step gives least in the random sample; Materials has no clean tagged text in the gold.
- Tune and test halves (the choices were made on them, so they are optimistic): the gate keeps 65% of the texts at 83.6% precision.

**Limits.** One reviewer, one draw, 39 to 40 rows; the intervals are 20 points wide. About half of the wrong answers of the first review were not sector texts at all,
so the scope step feeds this one. The D24 bar (80% against the held-out gold) is **not measured** for the sub-industry (T-167).

## 2. Cut (step 1.2, `docs/scope_training.md` section 8.3)

The `units` extractor (reference cut `units5-d3e2cbfa830e`) splits a `multi` text into pieces, each meant to hold one company's story. The cut has **no classes**,
so there is no one-vs-rest table. The owner marked each piece with three yes/no questions, which read like a precision per property:

| Question | Yes when |
|---|---|
| `faithful` | the piece says what the article says, with nothing added or changed (the bar of the extractor is 80%, D32) |
| `one_story` | it tells the news of one company or one story |
| `complete` | it reads as a whole (not a clause cut in half or a heading alone) |

### Dataset description

Three kinds of data stand behind the cut, which is a model that decides which company each paragraph of a body belongs to.

| Set | Size | Columns and labels | Characteristics |
|---|---|---|---|
| Attribution training labels (`company_segments_train.csv`) | 33,624 paragraphs of 2,834 bodies; 21,682 `lead`, 11,942 `continuation` | `article_id`, `body_hash`, `year`, `split`, `segment_index`, `ticker`, `kind` (ids and positions only, no text) | labels are automatic, from bodies written as one section per company; not hand-tagged |
| Label check (`segment_check_gold.csv`) | 200 paragraphs (100 `lead`, 100 `continuation`) | `key`, `article_id`, `segment_index`, `ticker`, `kind`, `year`, `correct`, `notes` | `lead` labels were right 88.0%, `continuation` 38.0%, so the model trains on `lead` only |
| Faithfulness sets (`faithfulness_gold.csv`, confirmation set) | first set 30 narrative bodies, 437 texts (336 faithful, 101 not); confirmation set 30 other bodies, 395 texts (345 faithful, 50 not) | `key`, `article_id`, `ticker`, `text_hash`, `extractors`, `revisions`, `pub_date`, `faithful`, `notes` | narrative bodies with several companies; texts of five extractors, with the extractor hidden and rows shuffled; the confirmation pool was exhausted, so no third set can be drawn |
| **Cut sample** (`cut_sample_marks.csv` and texts, the figures of section 2) | 28 bodies, 143 pieces (5.1 per body, at most 13), all from one news site, published 2007 to 2026 | `review_order`, `article_id`, `segment_index`, `ticker`, `pub_date`, `source_domain`, `title`, `text` (the piece), `body` (the whole body, first 6,000 characters), `company`, `faithful`, `one_story`, `complete`, `notes`; stored `kind` | drawn at random from the cut `units5-d3e2cbfa830e`; pieces are short (mean 434 characters, median 266, maximum 3,380); `ticker` and `company` are the crawl's tag of the article, not a mark of the piece; `kind`: `kept` 110, `wrong_scope` 15, `ambiguous` 14, `wrong_company` 4 |

### Model selection

Five extractors were compared on the faithfulness sets; all cut the body into pieces and decide which company each piece belongs to. The bar is 80% of texts faithful.

| Extractor | How it cuts and decides |
|---|---|
| `baseline` | sentences; a sentence belongs to the companies it names, or to those of the sentence before; no model |
| `model` | sentences; a sentence naming nobody goes to the company the trained model scores highest (at least 0.5) |
| **`units`** | lines, bullets and clauses, footer dropped; the trained model decides |
| `units2` | as `units`, with a guard on short forms of names and on lines that open with names and a dash |
| `units3` | as `units2`, and a piece of such a line also belongs to the companies it names |

The trained model is a pair classifier ("is this sentence about this company?") fine-tuned on the encoder of the sentiment model, the company name masked and the previous sentence as context, on `lead` labels only.
`units` was selected on 2026-11-08: on the confirmation set it passes the bar (86.1% [80.9, 90.7] on every company named, 95.3% [92.0, 98.2] on real companies), with shorter texts than the baseline (570 characters against 952) at the same faithful share (86.5% and 86.6%) and
less text of other companies. It is not shown better than the baseline on new bodies; it was kept because it is not worse and leaner. The margin is thin: flipping one to three marks to `no` moves the lower end under 80%. The cut was then re-read as `units4`, and **`units5` (`units5-d3e2cbfa830e`) is the final cut**;
its checkpoint is pinned to the private Hub copy `gamug/FinBERT-news-attribution@436ad700`.

**Sample**: 28 texts drawn at random from the cut, 143 pieces, marked by the owner on 2026-11-12. Share of yes with a Wilson 95% interval over pieces:

| Level | Pieces (texts) | faithful | one_story | complete | clean (3 yes) | Texts all clean |
|---|---|---|---|---|---|---|
| **All (the figure to quote)** | 143 (28) | 84.6% [77.8, 89.6] | 78.3% [70.9, 84.3] | 83.2% [76.2, 88.5] | 75.5% [67.9, 81.8] | 16 of 28 |
| Without the pieces of mis-scoped texts | 128 (27) | 89.1% [82.5, 93.4] | 82.0% [74.5, 87.7] | 87.5% [80.7, 92.2] | 78.9% [71.0, 85.1] | 16 of 27 |
| Also without the ambiguous texts | 114 (26) | 93.9% [87.9, 97.0] | by construction | 91.2% [84.6, 95.2] | by construction | by construction |

The second and third levels were chosen after the owner saw the misses; they are the cut with its inputs' errors set aside, not figures to quote alone.
The intervals that resample whole texts are wider (for `faithful` on all pieces: [74.8, 92.9]).

**Against the 80% bar for `faithful`**: the point estimate passes (84.6%), the lower end of the interval (77.8%) does not.

**Main company per piece** is a measure of step 3 (the relevance model), not of the cut: its training-validation and held-out figures are in sections 3.2 and 3.3.

**What is not measured.**
- **Recall** of the cut (did the cut find every story of the text): not measured.

## 3. NER and the company pick (step 3, `docs/companies_step.md`)

Step 3 has two parts that are measured separately: NER (finds organisation spans) and the relevance model (decides which companies in a text are the main ones).

### Dataset description

There is no NER gold on the texts step 3 reads. What exists:

| Set | Size | Columns and labels | Characteristics |
|---|---|---|---|
| NER training data (FiNER-ORD, used by the model's publisher) | not kept in this repository | entity spans of type `PER`, `LOC`, `ORG` | public financial-news NER data; the repo records the model, not the data |
| Judged evaluation (v1, `docs/evaluation.md`) | 1,000 sampled whole articles, 953 parsed | spans judged `wrong` or `missed` by an LLM judge | not a hand-made gold; whole articles, not the summaries or pieces step 3 reads |
| Relevance training and evaluation (the 809 marked summaries) | 809 (text, company) pairs from 855 tagged summaries (two draws, 450 and 405) | `key`, `article_id`, `ticker`, `company`, `body_hash`, `rule_scope`, `revision`, `predicted`, `pool_class_size`, `block`, `batch`, `slice`, `text_hash`, `pub_date`, plus the mark | each row is a text (a summary of an article, a few sentences) and a company; relevant = the mark has a direction (positive or negative), not relevant = neutral (202 of the 809 are neutral); draws balance the classes the old sentiment model predicted, so figures use weights |
| Main-company marks on sub-news (`data/main_company/to_tag.csv`, local) | 290 rows over 153 sub-news of the `units5` cut (the first version had 353 over 180) | `item_id`, `texto`, `texto_es`, `ticker`, `empresa`, `es_principal` (is it the main company, `yes`/`no`), `falta_la_principal` (is the main company missing), `notas` | six relevance bands; 119 rows of 59 items are marked (100 `yes`, 19 `no`) |

### Model selection

**NER.** `gamug/sec-bert-finer-ord-ner` (SEC-BERT-BASE fine-tuned on FiNER-ORD, pinned at `ba7b9e43`) is the model of the pipeline; it only finds organisation spans, which a matcher turns into universe tickers. The documents read for this report record **no comparison with another NER model**, only the judged run of section 3.1. The option `--candidates ner+text`
(NER plus the matcher run over the text, without lower-case matches and without matches inside disclosure lines) was chosen over `ner` alone by the owner on 2026-11-13, on the share of items it labels (a dry-run output share, not a metric of correctness: its figures are not kept here; see `docs/companies_step.md`, "Dry run").

**Relevance (which company is main).** The base is the zero-shot reading of the category model (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, "this text is mainly about <company>"), the best of nine zero-shot variants by AUC (0.743; the sentiment model's own ranking was 0.795). Fine-tuned on the 809 marks, its out-of-fold AUC is 0.90, and the published
model (`gamug/deberta-v3-company-relevance@8812c8bc`, private) is the one the step runs. The threshold 0.5 was chosen on those out-of-fold scores (main recall 94.7%, precision 90.0%); the other values tried (0.3, 0.7, 0.9) change little on summaries, but on sub-news the share of items with a main company moves widely across the range (a dry-run output share, not kept here), which the 809 summaries cannot settle.

### 3.1 NER (`gamug/sec-bert-finer-ord-ner`)

The only measure is the v1 evaluation of 2026-09: an LLM judge on 1,000 sampled **whole articles** (953 parsed), not on step 3's summaries or pieces.

| Measure | Value |
|---|---|
| Span micro-F1 | 0.7418 |
| Recall / precision | 0.84 / 0.66 (the model over-predicts) |
| Share of predicted spans the judge calls wrong | 33.8% |
| F1 per type | ORG 0.75, LOC 0.82, PER 0.66 |

Only `ORG` spans matter to step 3. There is **no per-class OVR gold on the texts step 3 reads**, and the judge is not the owner. The spans are turned into tickers by a matcher, which has
known gaps (short names such as `Disney` or `JPMorgan`, T-181). The raw spans are not stored.

### 3.2 Relevance: is a company "main" in a text (`gamug/deberta-v3-company-relevance`)

Two classes: main (relevant) and not main. Out-of-fold scores of the **809 marked summaries**, unweighted, threshold 0.5 (the step's value):

| | Value |
|---|---|
| Out-of-fold AUC | 0.90 |
| **Main** class: recall ("relevant rows kept") | 94.7% |
| **Main** class: precision | 90.0% |
| **Not main** class: recall (1 minus "not-relevant rows called main", 31.7%) | 68.3% |
| Not main class: precision | not given in the record |

At the other thresholds (0.30, 0.50, 0.70, 0.90): main recall 96.2 / 94.7 / 92.3 / 87.3%, precision 89.7 / 90.0 / 91.1 / 91.7%, share of not-main rows called main 33.2 / 31.7 / 27.2 / 23.8%.
Between a quarter and a third of the rows the owner marked "not relevant" are called main, whatever the threshold.

**Limits.**
- The model was trained on summaries, not on pieces; its held-out validation on pieces is the owner's partial marks of section 3.3 (119 rows of 59 pieces: main precision 90.9%, recall 30.0% at 0.5) (T-165, T-170).

### 3.3 Main company on pieces: the owner's partial marks (T-170)

The set of the main-company check (`data/main_company/to_tag.csv`, local and git-ignored; strata in `to_tag.strata.json`) has 290 rows over 153 sub-news of the `units5` cut
(`units5-d3e2cbfa830e`, companies `companies-f349536cf701`, seed 20261113), six relevance bands of 30 items (the first version had 353 rows over 180 items; the strata file still lists 180). The owner marked `es_principal` (`yes` or `no`: is this company the main one of the piece)
on **119 rows of 59 items** (100 `yes`, 19 `no`; 171 rows are still unmarked). The model's call is its relevance score at the step's threshold (0.5), taken from `to_tag.strata.json`. Scored on 2026-11-15 from the file as it stood then, unweighted
(the bands were sampled evenly, not in proportion to the population, so these are not corpus rates).

| Class | n marked | Precision | Recall |
|---|---|---|---|
| Main (`yes`) at 0.5 | 100 | 90.9% [76.4, 96.9] (30/33) | 30.0% [21.9, 39.6] (30/100) |
| Not main (`no`) at 0.5 | 19 | 18.6% (16/86) | 84.2% (16/19) |
| Main at 0.3 | 100 | 90.0% [78.6, 95.7] (45/50) | 45.0% [35.6, 54.8] (45/100) |
| Main at 0.7 | 100 | 85.0% [64.0, 94.8] (17/20) | 17.0% [10.9, 25.5] (17/100) |

The score separates the owner's `yes` from `no` with an AUC of **0.663** (unweighted), against 0.896 out of fold on summaries: more than 5 points under, the trigger of the T-170 rule (sub-news get their own threshold, chosen on the marked sub-news). The marks are partial and the owner's rule is wider than the model's (below), so this is a reason to decide, not the decision.
The marks, their texts and the scorer are tracked: `gold/main_company_marks.csv`, `gold/texts/main_company_marks_texts.csv`, `scripts/score_main_company_marks_2026_11_16.py`; the full account is `docs/main_company_training.md`.

Read per item (59 pieces): 53 have a company the owner marked main; the model's threshold finds at least one of them in 30 (56.6%). The model calls a main company on 32 pieces, and the owner agrees on at least one in 30 (93.8%).

Share of `yes` by relevance band of the item (band 0 is the lowest scores): band 0, 19 of 21 rows; band 1, 19 of 25; band 2, 20 of 22; band 3, 20 of 20; band 4, 10 of 11; band 5, 12 of 20.

**What the figures say.**
- **When the model calls a company main, it is nearly always right** (precision 0.91 at 0.5, 0.85 at 0.7). **It misses most of what the owner calls main** (recall 0.30): 19 of the 21 rows in the lowest band are `yes`.
- **Three pieces carry 21 of the 119 rows**, and every row of them is `yes` (items 335155, 343149 and 260642: 10, 7 and 4 companies). The first two are lists of tickers, and the model gives their companies scores of 0.01 to 0.16. Without these three pieces, 98 rows remain (79 `yes`, 19 `no`):
  main precision 30 of 33 = 0.91, recall 30 of 79 = 0.38, not-main precision 16 of 65 = 0.25, recall 16 of 19 = 0.84. The recall stays low, so the lists are not the whole cause.
- The owner's rule may be wider than the model's: pieces with several companies get several `yes`, while the model is trained to pick the one the text is mainly about (the 809 summaries). The marks do not say which reading is wanted; that is a decision, not an error of the marking.
- 59 pieces and one reader; 41% of the rows are marked, and the sample is stratified, so the intervals are wide and the figures are not corpus rates.

What it supports: when the step names a main company on a piece it is usually one the owner accepts; it does not support the claim that the step finds the owner's main companies (recall 0.30 to 0.45). A rate over the corpus is not supported.

## 4. Summarization (step 2, `docs/summary_step.md`, `docs/summarization_training.md`)

Since T-206 (owner, 2026-11-15) the step summarises a text over about 2,000 characters with **DeepSeek** through an API (one call per text, prompt `config/summary_prompt.md`, temperature 0); `--summariser local` keeps `sshleifer/distilbart-cnn-12-6` (chunk-and-reduce, the arm B below) and the v1 chain stays on distilbart. A text at or under about 2,000 characters is stored as it is. Summarization has **no classes**, so there is no precision and recall: the only measure is an LLM judge (DeepSeek, temperature 0) that scores faithfulness, coverage and conciseness from 1 to 5
and lists hallucinations. **The corpus run is not measured**: the figures are of 300 articles, and the judge of the run must not be DeepSeek (`docs/summary_step.md`).

### Dataset description

No labelled data: the evaluation uses the stored output of the pipeline and a judge. The experiment sample is **300 articles, 100 per tier of `num_chunks`** (1, 2 and 3 or more chunks), drawn with seed 1 from the articles of one earlier evaluation run (288 English, 12 Spanish; the summariser reads English only).
The judge's output per summary is `faithfulness`, `coverage` and `conciseness` (1 to 5), a list of `hallucinations` and a rationale; identical summaries are judged once (cached by article and summary hash).
The corpus the step runs on: 93,034 single bodies (59,432 over 2,000 characters) and 161,574 sub-news of the reference cut (5,048 over 2,000 characters); the rest is stored as it is.

### Model selection

**Model.** DeepSeek (`deepseek-chat`, which the API answers as `deepseek-flash`), chosen by the owner on 2026-11-15 on arm D of the experiment below (paired against the earlier step: faithfulness +0.39, coverage +1.29, hallucination -14.3 points); each row of the run keeps the model name the API gave. The earlier model, `sshleifer/distilbart-cnn-12-6` (pinned at `a4f8f3ea`, inherited from the v1 stage and not trained here), stays behind `--summariser local`; it was never compared with another summariser before DeepSeek, only with ways of using it.
**Size limit.** A text under 2,000 characters is stored as it is. The limit was chosen from the counts of texts that would go to the model: 1,500 characters 75,368 single bodies (81.0%) and 9,249 sub-news (5.7%), 17.6 h; **2,000 characters 59,432 (63.9%) and 5,048 (3.1%), 13.9 h**; 3,000 characters 31,992 (34.4%) and 1,731 (1.1%), 7.5 h; 4,000 characters 17,621 (18.9%) and 708 (0.4%), 4.1 h. The hours are the local distilbart figures (GPU). With DeepSeek the cost of the limit is calls and money: `docs/summary_step.md` and `docs/summarization_training.md` (T-206) count 79,283 items sent under the same size rule (about 310.8 million characters) and expect about $16 to $32 at the rates of 2026-11-15, to be confirmed.
**Method of the distilbart arms (2026-11-13).** Six arms were compared (table of this section); the current chunk-and-reduce (arm B) was kept because the alternatives did not beat it on every measure: joining chunk summaries with no gate (C1) is worse (+10.7 points of hallucination), the gate recovers it (C2), the first chunk alone (C0) and verbatim tails (C3) are more faithful but shorter on coverage or longer. Nothing was changed in production by that experiment; the DeepSeek arm (D) came after it.

**The experiment of 2026-11-13** (300 articles, 100 per tier of `num_chunks`, 288 English and 12 Spanish; the same sample as run 44). Higher is better, except hallucination. Arm **B is the earlier step** (distilbart); **D is DeepSeek, the step since T-206** (run 2026-11-15 on the same 300 articles and judge).

| Arm | What it is | Faithfulness | Coverage | Conciseness | Hallucination | Tokens |
|---|---|---|---|---|---|---|
| A | stored v1 summary, re-judged | 4.54 | 3.23 | 4.07 | 19.7% | 71 |
| B | the earlier step (distilbart, re-summarising; `--summariser local`) | 4.60 | 3.24 | 4.08 | 15.0% | 69 |
| C0 | chunk 1 only | 4.70 | 3.31 | 4.33 | 13.0% | 69 |
| C1 | chunk summaries joined, no NLI gate | 4.39 | 3.47 | 3.23 | 25.7% | 122 |
| C2 | C1 with the NLI gate | 4.64 | 3.31 | 3.44 | 14.7% | 100 |
| C3 | gated chunk 1 plus verbatim first sentences of later chunks | 4.81 | 3.13 | 3.32 | 6.7% | 137 |
| **D** | **DeepSeek, the step since T-206** | **4.99** | **4.54** | **4.98** | **0.7%** | **86 (output tokens per call)** |

By tier for the earlier step (B), faithfulness / coverage / hallucination: 1 chunk 4.67 / 3.81 / 14%; 2 chunks 4.64 / 3.11 / 11%; 3+ chunks 4.49 / 2.81 / 20%. Faithfulness falls and hallucination rises with the number of chunks.
Paired against B: C0 +0.10 faithfulness [+0.01, +0.18]; C3 +0.21 [+0.13, +0.30] faithfulness and -8.3 points of hallucination [-12.0, -4.7], at -0.11 coverage; C1 is worse (-0.21 faithfulness, +10.7 points).

**DeepSeek (arm D)** against B, paired by article (95% bootstrap intervals): faithfulness 4.99 against 4.60 (+0.39 [+0.29, +0.49]), coverage 4.54 against 3.24 (+1.29 [+1.21, +1.39]), conciseness 4.98 against 4.08 (+0.90 [+0.80, +1.00]), hallucination 0.7% against 15.0% (-14.3 points [-18.7, -10.3]). By tier (1 / 2 / 3+ chunks), D faithfulness 4.97 / 4.99 / 5.00, coverage 4.87 / 4.50 / 4.24, hallucination 1% / 1% / 0%. At temperature 0 two runs give different words (30 articles twice: 0 identical, mean word overlap 0.66) but the scores held (faithfulness 5.00 and 5.00, coverage 4.70 and 4.63). A blind re-scoring of 24 pairs by the assistant (Claude, not a person; the arm hidden) agreed: faithfulness 4.92 against 4.42, coverage 4.75 against 3.08, hallucination 4% against 21%. The figures are committed in `docs/summarization_results_2026_11_15/`.

**Earlier baseline** (v1 stage, 1,000 sampled articles, `docs/evaluation.md`): mean faithfulness 4.87, hallucination in 5.4% of summaries, coverage 3.02: accurate but not very complete. It is a different judge run and sample, so it is not comparable with the table above.

**Limits.** One sample, one judge, one run per arm. The judge's coverage is one overall score: **nothing checks whether the companies the text is about, or its main event, are in the summary** (no per-company check).
**DeepSeek was summariser and judge** in the evidence for arm D, so its scores are its own reading of its own text (the blind re-scoring of 24 pairs is small, and the arms can be guessed by style). The DeepSeek text is not repeatable at temperature 0 (a repeat of 30 articles kept the scores). The figures are of 300 articles; the corpus run is not yet measured.

## 5. Sentiment (step 6, `docs/sentiment_step.md`, `docs/sentiment_training.md`)

The pair model (text and company, `sentiment_pair`) gives three class probabilities; they are stacked with the relevance score by a logistic regression
(`config/sentiment_stack.json`). Figures are of the **nested cross-validation** over the **809 marked rows** (weighted, D37; these are the training-validation metrics), on the whole marked
population (pairs of other companies included), whereas the step scores main pairs only. The 180-text gold has not been scored with the stack (T-148c4, T-193). **The one held-out check is section 5.1:** the step as it ran on the corpus, scored on **190 pieces tagged by the owner**: accuracy 73.7% [67.0, 79.4] (under the 80% bar of D32), precision 83.0% / 69.4% / 56.7% and recall in the sample 76.0% / 80.6% / 53.1% for positive / negative / neutral; neutral is almost never predicted (weighted recall 4.4%).

### Dataset description

The label is **what a text says about one company**, not the sentiment of its sentences. Each row is a text (usually a summary of an article, a few sentences) and a company; the mark is the price impact an investor would read on that company. If the text gives no fact about the company, the mark is neutral, so neutral mixes "facts with no direction" and "no fact about the company".

| Set | Size | Class counts | Columns | Characteristics |
|---|---|---|---|---|
| Sentence pool (silver) | 5,800 sentences merged, 3,865 after balancing | 1,321 negative, 1,321 neutral, 1,223 positive | not in the gold files | sentences from the project's articles, labelled by an LLM one at a time, not tied to a company; trained the starting model |
| 180-text gold (`sentiment_gold.csv`) | 180: summaries and company texts (120) and extractor texts (60) | 82 positive, 65 negative, 33 neutral | `key`, `article_id`, `ticker`, `kind`, `part`, `text_hash`, `revision`, `stratum`, `extractor`, `predicted`, `population`, `pub_date`, `sentiment`, `notes` | the exam of the pin rule; never tuned on; not yet scored with the stack |
| Training set 1 (`sentiment_train.csv`) | 450 summaries (424 read as a pair) | 232 positive, 113 negative, 105 neutral | `key`, `article_id`, `ticker`, `company`, `body_hash`, `rule_scope`, `revision`, `predicted`, `pool_class_size`, `block`, `batch`, `slice`, `text_hash`, `pub_date`, plus the mark | three batches of 150, 150 per class the old model predicted |
| Training set 2 (`sentiment_train_2.csv`) | 405 summaries (385 read as a pair) | 178 positive, 116 negative, 111 neutral | as above | same design, 135 per predicted class |
| Replay | 341 sentences drawn of 925 eligible (those that name their article's company) | | | read as a pair with that company, mixed into every training set |
| **Held-out check** (`gold/sentiment_val190.csv`, section 5.1) | 190 pieces of the corpus run, two draws (90 + 100); never training data | 96 positive, 62 negative, 32 neutral | `key`, `article_id`, `item_id`, `ticker`, `batch`, `text_source`, `text_hash`, `predicted`, `predicted_score`, `relevance_score`, `pub_date`, the three revisions, `sentiment`, `notes`; the texts are in `gold/texts/sentiment_val190_texts.csv` | pieces (sub-news) of the stored summary step, not summaries of articles; 38 are distilbart summaries, the rest original short texts; the model's own call is kept apart from the owner's tag |

The draws balance the classes the old model predicted, not the true classes, so every figure is weighted by the pool size of the predicted class over the texts drawn from it. Together the two training sets give 809 pair rows. No text of the gold or of a held-out set is in any training set or in the replay.
Sector, market and economy texts (46 rows) are tagged for their scope and left out of the pair model.

### Model selection

The pair model reads `[company] [SEP] [text chunk]` and averages the chunk probabilities; a logistic regression (the stack) combines its log-probabilities with the relevance score. Cross-validated, nested and weighted (training-validation metrics), on the 809 rows:

| Candidate | Score (D37) | What it taught |
|---|---|---|
| Pinned v1 model, text only | 0.648 (60.4% agreement on the 180-text gold) | the starting point; does not hear the company |
| Pair model, the recipe of the grid (27 runs: epochs 2, 3, 4; learning rate 1e-5, 2e-5, 3e-5; 3 seeds) | 0.678 on 83 validation texts; **0.737** by cross-validation | gain +0.089 over the pinned model; neutral recall still 0.442 |
| Larger replay (809 sentences) | 0.723 | not better; the 341 stays |
| Mention flags, neutral prior, cues, a threshold | 0.737 to 0.740 | nothing: three quarters of the neutral texts do name the company |
| Hard gate "company not named, so neutral" | 0.664 to 0.678 | neutral recall up, precision down |
| Plus relevance, gate | 0.789 | neutral recall 0.754, precision 0.680 |
| **Plus relevance, stack (selected)** | **0.814** | neutral precision 0.787, recall 0.686; negative 0.747 / 0.748 |
| Plain `ProsusAI/finbert` as the start, stacked | 0.811 | the same within noise; the project's own endpoint stays |

Selected on 2026-11-10 (the held-out check of section 5.1 came after, and took no part in the selection): the pair model started from the project's endpoint `gamug/FinBERT-financial-news`, trained on all 809 rows plus the 341 replay sentences, stacked with the fine-tuned relevance model. The pin is not decided: it waits on the 180-text gold by the rule of D35.

| Class | Pair model alone P / R | Gate P / R | **Stack (the step) P / R** |
|---|---|---|---|
| Positive | 0.761 / 0.834 | 0.835 / 0.804 | **0.807 / 0.855** |
| Negative | 0.680 / 0.745 | 0.756 / 0.734 | **0.747 / 0.748** |
| Neutral | 0.634 / 0.442 | 0.680 / 0.754 | **0.787 / 0.686** |
| Macro P / R | 0.692 / 0.674 | 0.757 / 0.764 | **0.780 / 0.763** |
| Macro ROC OVR | 0.845 | 0.845 (by construction) | **0.898** |
| Score (D37) | 0.737 | 0.789 | **0.814** |

Against the pinned v1 model on the same slice (score 0.648), the stack gains +0.077 [+0.057, +0.095] over the pair model alone. Negative precision and recall are above the D32 bar (70%) in
the cross-validation. A shuffled-relevance control adds at most +0.005, so the gain of the stack is real information, not noise.

**Limits.**
- The cross-validation scores also served to choose among about a dozen variants on the same 809 rows, and their intervals are per comparison, not corrected for that. **The held-out check of section 5.1 is the measure to quote** (accuracy 73.7% on 190 pieces of the real run; neutral recall in the sample 53.1% against 0.686 in the cross-validation; weighted 4.4%); the two do not measure the same thing (marked summaries, other-company pairs included, against main pairs of pieces).
- **Neutral is the weakest class on its own** (recall 0.44 without the relevance model); the miss is not "the company is absent": the model hears a company and a polarity but does not tell whether the text is about the company.

### 5.1 Held-out check on 190 pieces of the real run (2026-11-16)

The first held-out check of the sentiment step on texts that no model or setting saw: the step as it ran on the corpus, the pair model and stack of
revision `sentiment-5999fe131d8f` (`gamug/FinBERT-company-sentiment@9a7b1eb3`), scored on **190 pieces tagged by the owner** (one reader) in two draws.
The marks, their texts and the scorer are tracked: `gold/sentiment_val190.csv`, `gold/texts/sentiment_val190_texts.csv`, `scripts/score_sentiment_val190_2026_11_16.py`
(`scripts/build_val_gold_2026_11_16.py` builds the gold files from the local working files).

| Item | First draw (90) | Second draw (100) |
|---|---|---|
| Source | sub-news of the stored summary step (`news_summary`, revision `summary-7a9bd2062bc4`) with a main company of the company step (`companies-f349536cf701`, relevance 0.7 or more) and a sentiment score; the text is what the sentiment step reads | the same, with a stricter pool |
| Draw | 30 per class the model predicted (10 stored summaries and 20 original short texts each), one text per article | simple random, one per article |
| Pool | 57,826 candidates (positive 38,559, negative 18,847, neutral 420) | about 21,000 candidates |
| Filters | 250 characters or more and two sentences; mostly letters; no ticker lists or disclosure lines; the company named in the text; English; no article of `gold/sentiment_gold.csv` or the training sets | the same, plus relevance 0.9 or more, 300 to 1,500 characters, the company named twice and within the first 100 characters, fewer than two analyst calls, no article of the first draw (and none of an earlier 200-text sample that was discarded and not committed) |
| Tags | positive 39, negative 27, neutral 24 | positive 57, negative 35, neutral 8 |
| Columns | `review_order`, `key`, `article_id`, `ticker`, `pub_date`, `title`, `text`, `spanish_title`, `spanish_text`, `company`, `sentiment`, `notes` (the prediction is in a separate `key.csv`, with a `batch` column) | the same |

Texts are 250 to 1,626 characters (median about 400). The Spanish title and text are a reading aid (`news_nlp.translate`) and are not copied to `gold/`. One row is noted "company is not the main" and is kept.
The scripts are `scripts/export_sentiment_val_2026_11_15.py` and `scripts/export_sentiment_val100_2026_11_15.py`; the working files stay local (`data/sentiment_val90/`, git-ignored).

**Results by draw** (precision exact within each predicted class, Wilson 95% interval; recall in the sample is the share of the tagged class found; recall weighted gives each predicted class its pool size over the pieces drawn from it):

First draw, 90 (stratified by the prediction): accuracy 66.7% [56.4, 75.5] (60 of 90), weighted 74.4%.

| Class | Precision | Recall (sample) | Recall (weighted) |
|---|---|---|---|
| Positive | 80.0% [62.7, 90.5] (24/30) | 61.5% (24/39) | 88.8% |
| Negative | 63.3% [45.5, 78.1] (19/30) | 70.4% (19/27) | 69.7% |
| Neutral | 56.7% [39.2, 72.6] (17/30) | 70.8% (17/24) | 4.0% |

Second draw, 100 (random, no weighting needed): accuracy 80.0% [71.1, 86.7] (80 of 100).

| Class | Precision | Recall |
|---|---|---|
| Positive | 84.5% [73.1, 91.6] (49/58) | 86.0% [74.7, 92.7] (49/57) |
| Negative | 73.8% [58.9, 84.7] (31/42) | 88.6% [74.0, 95.5] (31/35) |
| Neutral | no call (0/0) | 0% [0.0, 32.4] (0/8) |

**The 190 pooled** (an approximation: the two draws have different pools, and the weighted figures treat them as samples of the first draw's pool of 57,826):

| Class | Precision | Recall (sample) | Recall (weighted) | F1 (sample) | F1 (weighted) |
|---|---|---|---|---|---|
| Positive | 83.0% [73.8, 89.4] (73/88) | 76.0% [66.6, 83.5] (73/96) | 89.4% | 79.3% | 86.1% |
| Negative | 69.4% [58.0, 78.9] (50/72) | 80.6% [69.1, 88.6] (50/62) | 78.6% | 74.6% | 73.7% |
| Neutral | 56.7% [39.2, 72.6] (17/30) | 53.1% [36.4, 69.1] (17/32) | 4.4% | 54.8% | 8.2% |

Accuracy: 73.7% [67.0, 79.4] (140 of 190); weighted to the pool 78.4%.
Confusion (rows the model, columns the owner's tag; positive / negative / neutral): positive 73 / 8 / 7; negative 14 / 50 / 8; neutral 9 / 4 / 17.

**Against the bars of D32** (80% agreement; negative precision and recall of at least 70%): the pooled accuracy, 73.7% [67.0, 79.4], is **under 80%** (78.4% weighted to the pool, also under); the second draw alone, 80.0% [71.1, 86.7], is at the bar by point estimate, with an interval that reaches well below it. Negative precision, 69.4% [58.0, 78.9], is **under 70%** by point estimate; negative recall, 80.6% [69.1, 88.6], is over it.

**Relation to the pin decision.** This is not the D35 gate. The rule of D35 scores the 180-text gold on main pairs (T-148c4, T-193); that scoring has not been made and `config/pipeline_models.json` still names the v1 model. These 190 pieces were not drawn by D35's design; they measure the step as it ran on the corpus. They are evidence for the owner's decision; whether they replace the 180-gold scoring is the owner's to say.

**What it says.**
- **Positive** is the strongest class (precision 83.0%, weighted recall 89.4%).
- **Negative** precision is under the bar in the pooled figure and over it in the cleaner second draw (73.8%); negative recall is over it in both.
- **Neutral is almost never predicted**: the model calls 420 of the 57,826 pool texts neutral (0.7%), while the owner tagged neutral 16.8% of the 190 (32); so the weighted recall is 4.4%. Its precision (56.7%) is moderate. The cross-validation of the pair model gave neutral recall 0.686; on real pieces it is far lower (the two do not measure the same thing: marked summaries against main pairs of pieces). Tracked as T-212 (proposed; the owner decides the fix).
- The second draw is easier (accuracy 80.0% against 66.7%): its filter makes the company the subject of the text, and the first draw also includes texts where it is mentioned in passing, which the model reads as a direction.

**Limits.** The pooled set has 190 tags (positive 96, negative 62, neutral 32) and 190 calls (88, 72, 30). **Positive and negative rest on more than 60 tags and more than 70 calls each; neutral is the thin class**: 32 tags and 30 calls, so its precision [39.2, 72.6] and recall [36.4, 69.1] are wide.
The weighted recalls take their conditional shares from the 190 pieces and the class sizes from the first draw's pool of 57,826 (the two draws have different pools: an approximation). The neutral weighted recall of 4.4% is the neutral tags found among the positive-called pieces (7 of 88) and the negative-called ones (8 of 72), scaled by 38,559 and 18,847, plus the neutral-called ones (17 of 30, scaled by 420): it is a rate built from a few tags and is a pointer, not a measurement.
The second draw is random and needs no weighting, but it holds only 8 neutral tags. The pool is the candidates after the filters, not the whole corpus; the pieces are distilbart-summary revision `summary-7a9bd2062bc4` (38 of the 190 are summaries; the DeepSeek summaries of the corpus run are not covered). One reader, no second reading; the line between neutral and a direction was not fixed beforehand and depends on the rule used.

## 6. Category (step 5, `docs/category_step.md`, `docs/category_train.md`)

Two models are measured here. **The step runs the untouched zero-shot model** (6.1). A **fine-tuned candidate** (the entailment recipe of `config/category_model.json`, 12 labels, one tag per row, refit on 1,216 rows) is **not
wired** into the step; its evaluation is 6.2 to 6.5. Neither reaches the 80% bar (D24). Every run is in MLflow, experiment `news_nlp_train/category` (`uv run mlflow ui --backend-store-uri ./mlruns`), and the candidate's
weights are the registered model `category-entailment-single-12` (version 1, tagged "candidate, not wired").

### Dataset description

Each row is a (text, company) pair with one label of the taxonomy of `docs/category-taxonomy.md`: **14 labels in 6 groups** for the step's model; the candidate reads 12 (`competitive_actions` and `peer_comparison` read as `other`).

| Set | Size | Characteristics | Columns |
|---|---|---|---|
| Train rows (older sheet) | 225 rows with a company label | single bodies and structured multi bodies split by company, from the scope training set; narrative multi bodies left out; baseline hypotheses were written against it | `article_id`, `company`, `label`, `pred` (results files) |
| **Single-tag training rows (candidate)** | 1,216: a 1,000-row sheet reduced to one tag per row (999) plus 217 of the 225 older rows | 12 labels; `earnings_performance` 547, `market_analyst_sentiment` 214, `product_innovation` 90, `mergers_acquisitions` 81, `legal_regulatory` 45, `leadership_governance` 40, `capital_shareholder_returns` 40, `reputational_harm` 39, `labor_human_capital` 36, `partnerships_business_dev` 32, `other` 32, `investor_activity` 20; 82% are short company segments of round-ups (median 272 characters); tags were reviewed after first results were seen | per row: text, company, label, `fold` (5 folds grouped by article) |
| **Held-out** | 200 texts, of which 135 single rows are scorable (116 single bodies); 44 narrative `multi` texts not scored | full articles (median 2,085 characters, none under 600), earlier tag convention (analyst ratings could be `investor_activity`); scored once | `article_id`, `company`, `kind`, `label`, `pred`, plus the 14 label scores in the mixed run |
| Real sub-news sample | 300 rows | a seeded draw in two batches of 150 from the sub-news pairs of the first corpus run whose summarised text is under 1,000 characters (142,588 of 159,401 pairs); one text per body; none from a tagged body; 107 of the 300 are flagged unclear | text, company, label, flags |

### Model selection

The step runs the untouched zero-shot `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` at its pin: premise `<company>: <first 2,000 characters>`, one hypothesis per label ("The news about <company> is about <definition>"), the best entailment probability gives the label, and its group gives the group (6 groups, option B). **Neither it nor any candidate reaches the 80% bar (D24).**

| Candidate | Result | Verdict |
|---|---|---|
| **Untouched zero-shot (the step)** | held-out 43.0% (label), 55.6% (group); 56.4% on the train rows | provisional, runs in production |
| Flat fine-tune, 5-fold cross-validation on 225 rows | 64.0% out of fold | better than zero-shot out of fold |
| Mixed fine-tune (labels plus groups), 225 rows | 62.2% out of fold; held-out 52.6% (flat), 48.9% (two-step), group 62.2% | not published, not used |
| **Entailment candidate**, refit on 1,216 rows, 12 labels | 87.8% out of fold; **held-out 54.1%** [45.9, 62.3]; 71.3% on 300 real sub-news | registered as `category-entailment-single-12` v1, "candidate, not wired" |
| Plain DeBERTa-v3-base retrained on the same rows and folds | 85.7% out of fold | 2 points under the entailment recipe |

The held-out is the only measure that is held out and it fails; the candidate falls far from its 87.8% because its training rows are short round-up segments and the held-out rows are full articles. It is not a replacement for the step's model before the held-out bar is met.

### 6.1 The model the step runs (untouched zero-shot)

**Provisional.** The model is the untouched `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (zero-shot), with the premise `<company>: <text, 2,000 characters>` and one hypothesis per label.
It does **not** meet the 80% bar (D24); the owner's labels are the reference.

| | Train rows (225) | Held-out single rows (135) |
|---|---|---|
| Label agreement | 56.4% | **43.0%** |
| Group agreement (6 groups) | 62.7% | 55.6% |

The held-out figure is the honest one: the definitions were written against the train rows. The held-out holds 135 scorable rows of 200 texts; the 44 narrative `multi` texts are not scored.
Per-label precision and recall, recomputed from `baseline_heldout.csv` and `baseline_train.csv`:

| Label | Held-out n | Held-out P | Held-out R | Train n | Train P | Train R |
|---|---|---|---|---|---|---|
| earnings_performance | 27 | 0.71 | 0.74 | 64 | 0.71 | 0.95 |
| product_innovation | 22 | 0.62 | 0.45 | 26 | 0.62 | 0.58 |
| market_analyst_sentiment | 17 | 0.27 | 0.53 | 21 | 0.35 | 0.43 |
| investor_activity | 16 | 0.50 | 0.19 | 8 | 0.43 | 0.38 |
| legal_regulatory | 11 | 0.40 | 0.55 | 6 | 0.40 | 0.33 |
| leadership_governance | 11 | 1.00 | 0.18 | 9 | 0.60 | 0.33 |
| capital_shareholder_returns | 7 | n/a (never predicted) | 0.00 | 6 | 0.00 | 0.00 |
| reputational_harm | 7 | 1.00 | 0.14 | 23 | 0.86 | 0.26 |
| mergers_acquisitions | 5 | 0.31 | 0.80 | 15 | 0.44 | 0.73 |
| partnerships_business_dev | 5 | 0.25 | 0.20 | 17 | 0.67 | 0.35 |
| labor_human_capital | 3 | 0.50 | 0.67 | 19 | 0.69 | 0.58 |
| other | 3 | 0.00 | 0.00 | 3 | 0.00 | 0.00 |
| competitive_actions | 1 | 0.00 | 0.00 | 4 | 0.00 | 0.00 |
| peer_comparison | 0 | n/a | n/a | 4 | 0.00 | 0.00 |

Read the labels with fewer than about 10 held-out rows as anecdotes: one right or wrong row moves the figure by 10 points or more. Only `earnings_performance` is reliable (recall about 95% on train, 74% held out).
`market_analyst_sentiment` is over-predicted (33 predicted for 17 true), `mergers_acquisitions` too (13 for 5).

**Fine-tuned variant (not published, not used by the step).** The mixed-recipe fine-tune reached 62.2% out-of-fold on train and 48.9% on the held-out (label, two-step), and 62.2% on the held-out
group; its ROC OVR on the held-out is **0.798** macro over the 13 labels with held-out rows (0.783 over the groups). It also does not reach the bar. The per-class precision and recall of that variant
are in `docs/category_results_2026_11_13/` (`heldout_mixed_scores.csv`) and were not tabulated for this report.

**The group is the better field to rely on** (the doc's own advice): 55.6% held-out, against 43.0% for the label.

### 6.2 Candidate: cross-validation (out of fold, 1,216 single-tag rows, 5 folds grouped by article)

Rows: the owner's 1,000-row sheet reduced to one tag per row (999) plus the 225 older rows read again (217 kept); 12 labels (`competitive_actions` and `peer_comparison` read as `other`). **The tags were reviewed after the
first results were seen, and by the assistant**, so this measures how well the recipe learns those tags; it is not a clean accuracy. The rows are short company segments of round-ups (median 272 characters), not full articles.

Accuracy **87.8%**, macro F1 0.748, D37 selection score **0.821** (macro mean of precision, recall and ROC OVR). The same rows scored by the untouched model: 75.5% / 0.525 / D37 0.680 (1,214 of the rows).
Plain DeBERTa-v3-base retrained on the same rows and folds: 85.7% / 0.705 / D37 0.792, i.e. 2 points under the entailment recipe (not tested for significance).

| Label | Rows | Precision | Recall | ROC AUC |
|---|---|---|---|---|
| earnings_performance | 547 | 0.97 | 0.97 | 1.00 |
| mergers_acquisitions | 81 | 0.80 | 0.89 | 0.99 |
| leadership_governance | 40 | 0.88 | 0.90 | 1.00 |
| legal_regulatory | 45 | 0.72 | 0.73 | 0.98 |
| product_innovation | 90 | 0.87 | 0.86 | 0.99 |
| capital_shareholder_returns | 40 | 0.78 | 0.70 | 0.96 |
| labor_human_capital | 36 | 0.84 | 0.86 | 1.00 |
| market_analyst_sentiment | 214 | 0.89 | 0.88 | 0.98 |
| partnerships_business_dev | 32 | 0.77 | 0.75 | 0.97 |
| reputational_harm | 39 | 0.71 | 0.74 | 0.98 |
| investor_activity | 20 | 0.50 | 0.50 | 0.93 |
| other | 32 | 0.29 | 0.19 | 0.80 |

The 95% intervals (by article) are about ±0.12 to ±0.15 for the classes with 20 to 45 rows (`docs/category_results_2026_11_15/ovr_single.md`).

### 6.3 Candidate: the held-out 200, scored once (135 single rows)

The candidate was refit on all 1,216 rows and scored **once** on the held-out single rows (the 44 narrative texts are not scored). The held-out tags are the owner's of the earlier convention (analyst ratings may be
`investor_activity`); they were not reviewed. **Label agreement 54.1% [45.9%, 62.3%]**, group agreement 65.2% [57.4%, 72.7%], macro F1 0.448, D37 score 0.615. The untouched model on the same rows: 43.7% (43.0% with 14 labels).
**The bar of 80% is not met.**

| Label | Rows | Predicted | Precision | Recall | ROC AUC |
|---|---|---|---|---|---|
| earnings_performance | 27 | 34 | 0.74 | 0.93 | 0.96 |
| product_innovation | 22 | 25 | 0.60 | 0.68 | 0.89 |
| market_analyst_sentiment | 17 | 19 | 0.42 | 0.47 | 0.79 |
| investor_activity | 16 | 3 | 0.67 | 0.12 | 0.77 |
| leadership_governance | 11 | 5 | 1.00 | 0.45 | 0.64 |
| legal_regulatory | 11 | 14 | 0.43 | 0.55 | 0.92 |
| capital_shareholder_returns | 7 | 4 | 0.50 | 0.29 | 0.73 |
| reputational_harm | 7 | 4 | 0.50 | 0.29 | 0.93 |
| mergers_acquisitions | 5 | 11 | 0.36 | 0.80 | 0.90 |
| partnerships_business_dev | 5 | 1 | 1.00 | 0.20 | 0.73 |
| other | 4 | 9 | 0.11 | 0.25 | 0.71 |
| labor_human_capital | 3 | 6 | 0.33 | 0.67 | 0.98 |

Eight labels have fewer than 12 rows: read them as anecdotes (intervals run to 0 and 1). Only `earnings_performance` is tight. The largest confusion is `investor_activity` predicted as `market_analyst_sentiment` (8 rows),
the convention that changed. **Why it falls so far from 87.8%:** 82% of the training rows are short round-up segments (median 272 characters) and the held-out rows are full articles (median 2,085, none under 600 characters).

### 6.4 Candidate: a sample of real sub-news (300 rows from the real RESULTS store)

What the step reads: sub-news pairs of the first corpus run (`category-65f33b55b26b`, cut `units5-d3e2cbfa830e`) whose **summarized text is under 1,000 characters** (`news_summary.summary_text`; 142,588 of 159,401 pairs), a seeded
draw of 300 in two batches of 150, one text per body, none from an article or body of any tagged set. **The tags are the assistant's** (blind to both models, rules of `docs/category-taxonomy.md`), not the owner's;
107 of the 300 were flagged unclear. The assistant also tagged the training rows, so agreement is probably higher than an independent tagger's would give.

| | Label agreement | Macro F1 | Tagger sure (193) | Tagger unsure (107) |
|---|---|---|---|---|
| **Candidate** | **71.3%** [66.0%, 76.3%] | 0.637 | 83.9% | 48.6% |
| Untouched (the step) | 65.7% [60.3%, 71.0%] | 0.483 | 79.3% | 41.1% |

Paired: the candidate is right where the untouched model is wrong on 38 rows, the reverse on 21 (sign test p = 0.036). Batches: 69.3% and 73.3% (candidate), 62.7% and 68.7% (untouched).

**Per class, all 300 rows (candidate / selected model)**, precision and recall with 95% intervals (resampled by article):

| class | rows | precision [95%] | recall [95%] | ROC AUC |
|---|---|---|---|---|
| market_analyst_sentiment | 99 | 0.81 [0.73, 0.88] | 0.80 [0.72, 0.87] | 0.95 |
| earnings_performance | 56 | 0.82 [0.72, 0.91] | 0.89 [0.80, 0.96] | 0.98 |
| mergers_acquisitions | 30 | 0.80 [0.65, 0.94] | 0.80 [0.64, 0.94] | 0.98 |
| partnerships_business_dev | 26 | 0.65 [0.48, 0.81] | 0.77 [0.58, 0.93] | 0.95 |
| product_innovation | 18 | 0.46 [0.29, 0.65] | 0.72 [0.50, 0.92] | 0.93 |
| legal_regulatory | 13 | 0.77 [0.50, 1.00] | 0.77 [0.54, 1.00] | 0.99 |
| investor_activity | 7 | 0.75 [0.40, 1.00] | 0.86 [0.50, 1.00] | 0.92 |
| leadership_governance | 5 | 0.29 [0.00, 0.67] | 0.40 [0.00, 1.00] | 0.98 |
| other | 39 | 0.30 [0.00, 0.64] | 0.08 [0.00, 0.18] | 0.68 |

Not in that table (fewer than 5 rows): `labor_human_capital` (4 rows) precision 1.00 [1.00, 1.00], recall 1.00 [1.00, 1.00], ROC 1.00; `reputational_harm` (2) precision 0.33 [0.00, 0.83], recall 1.00 [1.00, 1.00], ROC 0.99;
`capital_shareholder_returns` (1) precision 0.25 [0.00, 1.00], recall 1.00 [1.00, 1.00], ROC 1.00.

**The same 300 rows, the model the step runs (untouched zero-shot, 14 labels read as 12):**

| class | rows | precision [95%] | recall [95%] | ROC AUC |
|---|---|---|---|---|
| market_analyst_sentiment | 99 | 0.67 [0.59, 0.74] | 0.90 [0.84, 0.95] | 0.87 |
| earnings_performance | 56 | 0.79 [0.68, 0.89] | 0.88 [0.78, 0.95] | 0.97 |
| mergers_acquisitions | 30 | 0.59 [0.44, 0.75] | 0.73 [0.58, 0.88] | 0.96 |
| partnerships_business_dev | 26 | 0.78 [0.56, 0.95] | 0.54 [0.32, 0.75] | 0.91 |
| product_innovation | 18 | 0.45 [0.17, 0.78] | 0.28 [0.08, 0.50] | 0.76 |
| legal_regulatory | 13 | 1.00 [1.00, 1.00] | 0.38 [0.12, 0.67] | 0.97 |
| investor_activity | 7 | 0.75 [0.00, 1.00] | 0.43 [0.00, 0.83] | 0.84 |
| leadership_governance | 5 | 0.33 [0.00, 1.00] | 0.20 [0.00, 0.67] | 0.95 |
| other | 39 | 0.28 [0.08, 0.50] | 0.13 [0.03, 0.24] | 0.46 |

(Also: `labor_human_capital` 4 rows 0.75 / 0.75 / 0.99; `reputational_harm` 2 rows 0.20 / 0.50 / 0.92; `capital_shareholder_returns` 1 row, never predicted, recall 0.00, ROC 0.94.)

Labels with fewer than 10 rows (`investor_activity`, `leadership_governance` and the three below the table) are anecdotes. **`other` is the failure in both models**: the candidate gets 3 of 39 right and the untouched model 5 (ROC 0.68 and 0.46); the untouched model over-assigns
`market_analyst_sentiment` (133 predicted for 99 true), which the candidate largely corrects.

### 6.5 The candidate against the model the step runs

| Measure | Untouched (the step) | Candidate | Bar (D24) |
|---|---|---|---|
| Held-out, 135 single rows, owner's earlier tags | 43.7% | **54.1%** | 80%: not met |
| Real sub-news, 300 rows, assistant's tags | 65.7% | **71.3%** | not defined for this set |
| Out of fold, 1,216 rows, reviewed tags | 75.5% | 87.8% | not a held-out measure |

The three measures disagree because they are different texts and different taggers; the held-out is the only one tagged by the owner and held out, and it is the one that fails. The candidate is better than the step's model on all three, and
the gain is statistically clear on the 300 real rows. It should not replace the pinned model before the held-out bar is met (T-201 to T-203).

## What these figures do not say

- **The steps are not compared on one yardstick.** Each is measured on its own sample, with its own reviewer and its own definition of "right"; the summary table shows which are held out.
- **Error chains are not in these figures.** A wrong scope reaches the cut, the company step, the sector step and the category step as bad input; for example, about half of the wrong sector answers of the first review were not sector texts,
  and 15 of the 143 sampled pieces came from texts the scope step should have stopped.
- **The category candidate's three measures are not one yardstick** (6.5): out of fold on tags the assistant reviewed, held out on the owner's earlier tags (full articles), and a real-data sample tagged by the assistant (short segments). Only the held-out is independent, and it fails the bar (54.1% against 80%).
- **Not measured at all:** sector recall per sector, the cut's recall, NER on step 3's own texts, the main company on pieces beyond the partial marks of section 3.3, and the sentiment of the DeepSeek summaries (the 190 pieces of section 5.1 are of the distilbart revision).

## Where each figure comes from

| Step | Source |
|---|---|
| Scope | `docs/scope_training.md` (sections 5, 6, 9); `data/scope_end_to_end_with81b.csv` and `gold/category_gold_v2_test.csv` (per-class table recomputed) |
| Sector | `docs/sector_step.md` ("The clean test", "What this does not show"); `gold/sector_validation.csv` (per-sector precision recomputed) |
| Cut | `docs/scope_training.md` section 8.3; `gold/cut_sample_marks.csv`; `scripts/cut_sample_metrics_2026_11_12.py` |
| Company: NER | `docs/evaluation.md` (baseline of 2026-09, `ner`) |
| Company: relevance | `docs/companies_step.md` (threshold table); `docs/sentiment_training.md` item 6 (AUC 0.90); `docs/main_company_training.md` and `gold/main_company_marks.csv` (the marks on pieces, tracked) |
| Summarization | `docs/summarization_training.md` (arms, tiers and the DeepSeek arm D); `docs/summarization_results_2026_11_15/` (committed figures of arm D); `docs/evaluation.md` (v1 baseline); `docs/summary_step.md` (limits) |
| Sentiment | `docs/sentiment_step.md` ("What the figures are"); `docs/sentiment_training.md` (stack table, nested cross-validation; the held-out check on 190 pieces, section 5.1: `gold/sentiment_val190.csv`, `gold/texts/sentiment_val190_texts.csv`, `scripts/score_sentiment_val190_2026_11_16.py`) |
| Category (the step's model, 6.1) | `docs/category_step.md`; `docs/category_results_2026_11_13/` (per-label tables recomputed from `baseline_heldout.csv` and `baseline_train.csv`) |
| Category (candidate, 6.2 to 6.5) | PR #196: `docs/category_train.md`; `docs/category_results_2026_11_15/` (`ovr_single.md` out of fold; `heldout_selected.md` held-out, scored once by `scripts/category_heldout_2026_11_15.py`; `real_sample.md` and `scripts/category_real_sample_2026_11_15.py` for the 300 real rows, tags in `gold/category_real_sample.csv`); MLflow experiment `news_nlp_train/category` (`scripts/category_mlflow_log_2026_11_15.py`) and the registered model `category-entailment-single-12`; D37 scores from the MLflow runs |
