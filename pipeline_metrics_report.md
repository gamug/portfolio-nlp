# Precision and recall (one-vs-rest) of the pipeline steps

Date: 2026-11-16 (the held-out figures of NER, relevance and sentiment are of 2026-11-15 and 2026-11-16; the rest, of the dates each section gives). The number in a heading is the pipeline's own step number; the sections follow the order the steps run. "The owner" is the project's term for the person who marks and decides. Steps covered, in pipeline order: scope (with the sector step), cut, NER and the company pick, summarization, sentiment, category. At the end, one section puts the metrics of all the models together.

Every figure below is taken from the repository's documents and tracked result files, or recomputed from tracked files (the category
and sector tables were recomputed for this report from `docs/category_results_2026_11_13/*.csv` and `gold/sector_validation.csv`; the category figures are the files of PR #196, `docs/category_results_2026_11_15/`, and its MLflow runs, section 6; the NER ones, `docs/ner_results_2026_11_16/`). **The figures measured on texts of the real corpus run are
the NER held-out (100 texts of step 2), the main-company marks on pieces (section 3.2), the sentiment check on 190 pieces (section 5) and the category real-data sample (section 6)**; the category sample was read from the RESULTS store (read-only: `news_category`, `news_summary`, `sub_news`); all others are from tracked or
result files.

**One-vs-rest (OVR)** means: for each class, precision = of what the step called that class, the share that is right; recall = of the true members of
that class, the share the step found. Where a step has no classes of that kind, or the owner never measured them, the table says so instead of
inventing a figure.

**Which metrics are kept.** For every model trained in the project (the scope classifiers, the cut's pair classifier, the relevance model, the sentiment pair model and stack, the category fine-tune) this report keeps two kinds of metrics only: **training-validation** (cross-validation, out-of-fold and validation-split scores) and **held-out** (scores on texts the model was neither trained nor tuned on).
What was removed in this version is every other measure: the share of a run's items that a step labelled (a dry-run output share) and the counts of a corpus run, which say what a model produces, not whether it is right. The dataset descriptions, the model-selection narratives and the limitations are kept. Models not trained here (summarization, the category step's earlier zero-shot model, the rules) keep the figures they had.
The removed shares and counts are still in the step documents (`docs/companies_step.md`, `docs/main_company_training.md`, `docs/summary_step.md`), which are not pipeline-wide reports and keep them. Dataset descriptions keep the label-quality figures of their sets (for example the label check of the cut's training labels): they describe the data, not a model.

## Summary

| Step | What is measured | Has per-class OVR precision and recall? | Headline | Held out? |
|---|---|---|---|---|
| Scope | one of 4 classes per text (`single`, `multi`, `sector`, `out_of_scope`) | **Yes**, four classes (section 1.1) | held-out agreement 83.6% [78.1, 88.9]; `out_of_scope` recall 53%; quarantined precision 90.8% | Yes (200 texts, seen many times), plus 225 read by the owner |
| Sector | the 2-digit GICS sector assigned to a `sector` text | **Precision only** (per sector, 2 to 5 texts each); recall not measured | 87.5% of the assigned sectors right (35 of 40) | Yes (a draw made after the step was built), one reviewer |
| Cut | the pieces ("sub-news") the cut makes from a multi-company text | **No classes.** Three yes/no questions per piece (precision-like) | 84.6% of pieces faithful; 75.5% clean | Yes (28 texts, 143 pieces) |
| Company: NER | the organisation spans NER finds (and the universe companies the matcher takes from them) | **No classes**: precision, recall and F1 on 100 texts tagged by the owner (321 organisations) | universe companies: precision 90.1%, recall 95.4% on the owner's labels as they are (92.6% / 95.5% with the developer's overlay; 98.7% recall with `ner+text`); all organisations: 72.6% / 90.4% with the overlay | **Yes** (100 texts, one reviewer; 14 labels not in the text, overlay: 9 replaced, 1 kept, 4 dropped); plus an older run judged by an LLM (v1) |
| Company: relevance (which company is main) | is a (text, company) pair "main" | Yes, for the two classes | cross-validation (809 summaries): main-company precision 90.0%, recall 94.7% at threshold 0.5; **held-out (119 marked rows of 59 pieces, the texts the step reads): precision 90.9%, recall 30.0%** | **Yes** (partial marks on pieces, one reviewer); the cross-validation is not |
| Summarization | the summary of a text over about 2,000 characters | **No classes.** LLM-judge scores (1 to 5) | **DeepSeek, the step's summariser since T-206:** faithfulness 4.99, coverage 4.54, hallucination 0.7% (300 articles); the earlier distilbart step: 4.60, 3.24, 15.0% | The judge is DeepSeek too (it graded its own summaries); a blind re-scoring of 24 pairs by the assistant (Claude) agreed in general terms (faithfulness 4.92, coverage 4.75, hallucination 4%); not a gold, and small; the corpus run is not measured |
| Sentiment | positive, negative, neutral for a (text, company) pair | **Yes**, three classes | nested cross-validation (809 rows): score 0.814, negative P/R 0.747 / 0.748; **held-out (190 real pieces): accuracy 73.7%, negative 69.4% / 80.6%** | **Yes** (190 pieces, one reader); the cross-validation is not |
| Category | 14 labels (6 groups) for a (text, company) pair; the model pinned in the step reads 12; **planned: 8 (7 measured plus `other`, not measured)** | **Yes**, per label (precision and recall) | **official (owner, 2026-11-17): a blind LLM judge on 300 real sub-news agrees on 88.0% [83.8, 91.2]** (the model's label is the judge's first or second); with the four weak labels moved to `other`: 92.0% (94.4% over the 7 measured labels). Also: out of fold 91.5%, held-out 57.1%, 82.1% on 300 real sub-news (joint tags), over the 7 measured labels (12-label model remapped, not retrained) | The official judge figure is **above the 80% bar**; the D24 held-out (135 single rows, scored once, not representative) is **not met** (57.1%) |

**Read the last column first.** Only the scope, the cut, the NER, the relevance on pieces, the sentiment on 190 pieces, the category and the sector clean test are measures on texts the model or the rules did not choose
their settings on; the cross-validation figures of relevance and sentiment are out-of-fold and not held out. None of the steps has been scored by a second
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

Five extractors were compared on the faithfulness sets; all cut the body into pieces and decide which company each piece belongs to. `units` was then compared with two later readings, `units4` and `units5`. The bar is 80% of texts faithful.

| Extractor | How it cuts and decides |
|---|---|
| `baseline` | sentences; a sentence belongs to the companies it names, or to those of the sentence before; no model |
| `model` | sentences; a sentence naming nobody goes to the company the trained model scores highest (at least 0.5) |
| **`units`** | lines, bullets and clauses, footer dropped; the trained model decides (selected 2026-11-08) |
| `units2` | as `units`, with a guard on short forms of names and on lines that open with names and a dash |
| `units3` | as `units2`, and a piece of such a line also belongs to the companies it names |
| `units4` | as `units`, with the plain resolver (no guard); the header of a "Names – text" sentence is read per sentence (also in the middle of a paragraph) and a bracketed ticker is part of the header; the sentence goes to the universe companies of the header and to those its text names, and to nobody if the header names none; a headed sentence is not cut into clauses (T-176; the cut from 2026-11-11 to 2026-11-12) |
| **`units5`** | as `units4`, and it also reads the headers `units4` missed: a short name in capitals ("SAP —", "UBS –"), a group label ("Memory stocks —"), a period inside a name ("Amazon.com (AMZN) –") and a lowercase text after the dash only when the header carries its ticker in brackets; a sentence with no header of its own after a header of nobody goes to the universe companies it names (T-177; **the final cut since 2026-11-12**) |

The trained model is a pair classifier ("is this sentence about this company?") fine-tuned on the encoder of the sentiment model, the company name masked and the previous sentence as context, on `lead` labels only.
`units` was selected on 2026-11-08: on the confirmation set it passes the bar (86.1% [80.9, 90.7] on every company named, 95.3% [92.0, 98.2] on real companies), with shorter texts than the baseline (570 characters against 952) at the same faithful share (86.5% and 86.6%) and
less text of other companies. It is not shown better than the baseline on new bodies; it was kept because it is not worse and leaner. The margin is thin: flipping one to three marks to `no` moves the lower end under 80%.

**`units4` and `units5` (readings of `units`, the same trained model).** The first run of the cut on 232 `multi` bodies showed two faults of `units`: 162 company texts (in 79 bodies) with two or more "Name —" headings and 6 texts that were only a heading. On those 232 bodies (`scripts/compare_cut_readings_2026_11_11.py`, GPU):

| | `units` | `units4` | `units5` |
|---|---|---|---|
| company texts | 1,671 | 1,589 | 1,638 |
| texts with several headings | 162 (79 bodies) | 23 (18 bodies) | 10 (the ten legitimate overlaps, read with `several_headings_owner`) |
| texts that are only a heading | 6 | 0 | not recorded |
| faithful on the texts with an owner's mark (the earlier faithfulness marks) | not compared here | 304 of 313, 97.1% [94.5, 99.1] | 351 of 359, 97.8% [95.5, 99.4] |

The gain of `units5` over `units4` comes from added texts that are faithful, not from existing texts that improve (`docs/extractor_training.md`, update of 2026-11-11; `docs/scope_training.md` section 8.1). **`units5` (`units5-d3e2cbfa830e`) is the final cut** (owner, 2026-11-12, after the yes/no sample below); its checkpoint is pinned to the private Hub copy `gamug/FinBERT-news-attribution@436ad700`.

**Sample**: 28 texts drawn at random from the cut, 143 pieces, marked by the owner on 2026-11-12. Share of yes with a Wilson 95% interval over pieces:

| Level | Pieces (texts) | faithful | one_story | complete | clean (3 yes) | Texts all clean |
|---|---|---|---|---|---|---|
| **All (the figure to quote)** | 143 (28) | 84.6% [77.8, 89.6] | 78.3% [70.9, 84.3] | 83.2% [76.2, 88.5] | 75.5% [67.9, 81.8] | 16 of 28 |
| Without the pieces of mis-scoped texts | 128 (27) | 89.1% [82.5, 93.4] | 82.0% [74.5, 87.7] | 87.5% [80.7, 92.2] | 78.9% [71.0, 85.1] | 16 of 27 |
| Also without the ambiguous texts | 114 (26) | 93.9% [87.9, 97.0] | by construction | 91.2% [84.6, 95.2] | by construction | by construction |

The second and third levels were chosen after the owner saw the misses; they are the cut with its inputs' errors set aside, not figures to quote alone.
The intervals that resample whole texts are wider (for `faithful` on all pieces: [74.8, 92.9]).

**Against the 80% bar for `faithful`**: the point estimate passes (84.6%), the lower end of the interval (77.8%) does not.

**Main company per piece** is a measure of step 3 (the relevance model), not of the cut: its training-validation and held-out figures are in section 3.2.

**What is not measured.**
- **Recall** of the cut (did the cut find every story of the text): not measured.

## 3. NER and the company pick (step 3, `docs/companies_step.md`)

Step 3 has two parts that are measured separately: NER (finds organisation spans) and the relevance model (decides which companies in a text are the main ones). Each has its own section and a single table of metrics, with the validation and the held-out together.

### Dataset description

NER has had an owner's held-out set on texts of step 2 since 2026-11-16; relevance has its 809 marked summaries and, besides, marks on pieces. What exists:

| Set | Size | Columns and labels | Characteristics |
|---|---|---|---|
| NER training data (FiNER-ORD, used by the model's publisher) | not kept in this repository | entity spans of type `PER`, `LOC`, `ORG` | public financial-news NER data; the repo records the model, not the data |
| Judged evaluation (v1, `docs/evaluation.md`) | 1,000 sampled whole articles, 953 parsed | spans judged `wrong` or `missed` by an LLM judge | not a hand-made gold; whole articles, not the summaries or pieces step 3 reads |
| **NER held-out** (`data/ner_val100/to_tag.csv`, local and outside git; `docs/ner_training.md`) | 100 texts, 321 organisations | `review_order`, `article_id`, `pub_date`, `title`, `text`, `spanish_title`, `spanish_text`, `key`, `organizations`, `notes` | the owner writes the organisations of each text separated by commas, as they appear in the text; the texts are summaries or originals of step 2 (`summary-7a9bd2062bc4`); **14 labels in 11 rows are not written in the text** (Spanish names, typos, names run together, one defective text, four from another article): the developer proposes an overlay (9 replaced, 1 kept, 4 dropped) and the file was not edited (T-213) |
| Relevance training and evaluation (the 809 marked summaries) | 809 (text, company) pairs from 855 tagged summaries (two draws, 450 and 405) | `key`, `article_id`, `ticker`, `company`, `body_hash`, `rule_scope`, `revision`, `predicted`, `pool_class_size`, `block`, `batch`, `slice`, `text_hash`, `pub_date`, plus the mark | each row is a text (a summary of an article, a few sentences) and a company; relevant = the mark has a direction (positive or negative), not relevant = neutral (202 of the 809 are neutral); draws balance the classes the old sentiment model predicted, so figures use weights |
| **Relevance held-out**: main-company marks on sub-news (`data/main_company/to_tag.csv`, local; `gold/main_company_marks.csv`) | 290 rows over 153 sub-news of the `units5` cut (the first version had 353 over 180); 119 rows of 59 items are marked (100 `yes`, 19 `no`) | `item_id`, `texto`, `texto_es`, `ticker`, `empresa`, `es_principal` (is it the main company, `yes`/`no`), `falta_la_principal`, `notas` | six relevance bands of 30 items (companies `companies-f349536cf701`, seed 20261113); the model never saw sub-news; one reader |

### Model selection

**NER.** `gamug/sec-bert-finer-ord-ner` (SEC-BERT-BASE fine-tuned on FiNER-ORD, pinned at `ba7b9e43`) is the model of the pipeline; it only finds organisation spans, which a matcher turns into universe tickers. The documents read for this report record **no comparison with another NER model**: it is measured on the v1 judged run and on the 100-text held-out of section 3.1; nothing was changed or pinned by that measure. The option `--candidates ner+text`
(NER plus the matcher run over the text, without lower-case matches and without matches inside disclosure lines) was chosen over `ner` alone by the owner on 2026-11-13, on the share of items it labels (a dry-run output share, not a metric of correctness: its figures are not kept here; see `docs/companies_step.md`, "Dry run"); the 100-text held-out now measures it.

**Relevance (which company is main).** The base is the zero-shot reading of the category model (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, "this text is mainly about <company>"), the best of nine zero-shot variants by AUC (0.743; the sentiment model's own ranking was 0.795). Fine-tuned on the 809 marks, its out-of-fold AUC is 0.90, and the published
model (`gamug/deberta-v3-company-relevance@8812c8bc`, private) is the one the step runs. The threshold 0.5 was chosen on those out-of-fold scores (main recall 94.7%, precision 90.0%); the other values tried (0.3, 0.7, 0.9) change little on summaries, but on sub-news the share of items with a main company moves widely across the range (a dry-run output share, not kept here), which the 809 summaries cannot settle; the marks on pieces (below) measure it.

### 3.1 NER (`gamug/sec-bert-finer-ord-ner`)

NER has no per-type classes that matter here: only `ORG` spans count, and what step 3 keeps is the universe companies the matcher takes from them. One table: the v1 judged evaluation (the validation) and the owner's held-out (100 texts, 321 organisations; precision, recall and F1 per text as sets of canonical companies; the 14 labels that are not in the text are treated by the developer's **overlay** in the first rows, a proposal the owner must confirm: 9 replaced, 1 kept, 4 dropped; the labels as they are follow the table).

| Measure | v1 judged evaluation (1,000 whole articles, LLM judge) | Owner's held-out (100 texts) |
|---|---|---|
| All organisations: precision | 0.66 (the model over-predicts; the judge calls 33.8% of the predicted spans wrong) | 72.6% |
| All organisations: recall | 0.84 | 90.4% |
| All organisations: F1 | span micro-F1 0.7418 (ORG 0.75, LOC 0.82, PER 0.66) | 0.805 |
| **Universe companies** (what step 3 keeps): precision | not measured | **92.6%** (150/162) |
| **Universe companies**: recall | not measured | **95.5%** (150/157) |
| **Universe companies**: F1 | not measured | **0.940** |
| Universe companies with `ner+text` (the matcher over the text as well): precision / recall / F1 | not measured | 90.1% / **98.7%** / 0.942 |

With the 14 labels as they are (no overlay): universe companies 90.1% / 95.4% / 0.927 and all organisations 70.3% / 87.0% / 0.777; removing them makes precision worse, because a mistyped label of a real company leaves the model's correct find as an "extra".

**What it says and does not say.**
- **Of the 7 universe companies that are missing, 5 are not tagged by the NER** (Meta Platforms, Hormel, MSFT, Oracle, Las Vegas Sands) and 2 are tagged but the matcher does not map them (`ON Semiconductor`, `Amazon`); `ner+text` recovers the 5. The 12 extra companies are real company names the owner did not write (`Royal Caribbean`, `PayPal`, `Intel`, `Trane`...); each text was not read to see whether it is about them.
- The extra spans on all organisations are media, institutions and products (CNBC, the Federal Reserve, Congress, iPhone) with no ticker, which step 3 drops; they are not a defect of the step.
- The labels are **every organisation named**, not the main ones (section 3.2 measures that). The alias list the names are compared with (Google and Alphabet, Facebook and Meta...) was chosen after the errors were seen, so it is fitted to this set (without it, on all organisations, 68.0% / 86.4%).
- 100 texts, one reviewer, no interval; summaries of the old step 2 (distilbart), not the DeepSeek ones of the corpus run. The overlay on the 14 labels (9 replaced, 1 kept, 4 dropped) is the developer's; the defective texts (words stuck together such as `NHTSAsaid`) are the crawl's, not the model's.
- The v1 judge is not a gold nor the owner, and scores whole articles. Spans are turned into tickers with a matcher with known gaps (short names such as `Disney` or `JPMorgan`, T-181); the raw spans are not stored. Detail: `docs/ner_training.md`; figures: `docs/ner_results_2026_11_16/`; script: `scripts/ner_eval_2026_11_16.py`.

### 3.2 Relevance: is a company "main" in a text (`gamug/deberta-v3-company-relevance`)

Two classes: main and not main. One table: the cross-validation (out-of-fold scores of the 809 marked summaries, unweighted) and the held-out (the 119 rows the owner marked on pieces of the `units5` cut, unweighted: the bands were sampled equally, they are not corpus rates). 95% intervals where computed.

| Measure | Cross-validation (809 summaries) | Held-out (119 rows, 59 pieces) |
|---|---|---|
| AUC | 0.90 | **0.663** |
| Main at 0.5 (the step): precision | 90.0% | 90.9% [76.4, 96.9] (30/33) |
| Main at 0.5: recall | 94.7% | 30.0% [21.9, 39.6] (30/100) |
| Not main at 0.5: precision | not in the record | 18.6% (16/86) |
| Not main at 0.5: recall | 68.3% (1 minus 31.7% of not-relevant rows called main) | 84.2% (16/19) |
| Main at 0.3: precision / recall | 89.7% / 96.2% | 90.0% [78.6, 95.7] (45/50) / 45.0% [35.6, 54.8] (45/100) |
| Main at 0.7: precision / recall | 91.1% / 92.3% | 85.0% [64.0, 94.8] (17/20) / 17.0% [10.9, 25.5] (17/100) |
| Main at 0.9: precision / recall | 91.7% / 87.3% | not computed |

**What it says and does not say.**
- **When the model calls a company main it is almost always right** (precision 0.91 at 0.5), but **it misses most of what the owner calls main** (recall 0.30; 19 of the 21 rows of the lowest band are `yes`). The held-out has an AUC of **0.663** against 0.90 out of fold, more than 5 points under: the trigger of T-170's rule (sub-news get their own threshold, chosen on the marked sub-news); the decision is the owner's and is still open (`docs/main_company_training.md`).
- Three pieces give 21 of the 119 rows (all `yes`; two are ticker lists with scores of 0.01 to 0.16); without them, at 0.5: precision 0.91, recall 0.38: the lists are not the whole cause.
- Per piece (59): 53 have a company the owner marked as main and the threshold finds at least one of them in 30 (56.6%); the model declares a main company in 32 pieces and the owner agrees on at least one in 30 (93.8%).
- The owner's rule may be wider than the model's: pieces with several companies get several `yes`, and the model was trained to choose the one the text is mainly about. That is a decision, not a marking error.
- 41% of the rows are marked (119 of 290), one reader and a stratified sample: the intervals are wide and the figures are not corpus rates. The cross-validation is optimistic for sub-news: it was trained on summaries.
- The split between "no universe company is named" and "one is named in a form the matcher does not know" was not measured. The text candidates of `ner+text` were reviewed by the developer on a few pairs; neither figure is kept here.
- Files: `gold/main_company_marks.csv`, `gold/texts/main_company_marks_texts.csv`, `scripts/score_main_company_marks_2026_11_16.py`; the full count is `docs/main_company_training.md`.

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
(`config/sentiment_stack.json`). It is measured two ways, in the single table below: the **nested cross-validation** over the **809 marked rows** (weighted, D37; not held out; the whole marked population, pairs of other companies included, whereas the step scores main pairs only) and a **held-out of 190 pieces tagged by the owner** (one reader) from the real corpus run, revision `sentiment-5999fe131d8f` (`gamug/FinBERT-company-sentiment@9a7b1eb3`). The 180-text gold has not been scored with the stack (T-148c4, T-193).

### Dataset description

The label is **what a text says about one company**, not the sentiment of its sentences. Each row is a text (usually a summary of an article, a few sentences) and a company; the mark is the price impact an investor would read on that company. If the text gives no fact about the company, the mark is neutral, so neutral mixes "facts with no direction" and "no fact about the company".

| Set | Size | Class counts | Columns | Characteristics |
|---|---|---|---|---|
| Sentence pool (silver) | 5,800 sentences merged, 3,865 after balancing | 1,321 negative, 1,321 neutral, 1,223 positive | not in the gold files | sentences from the project's articles, labelled by an LLM one at a time, not tied to a company; trained the starting model |
| 180-text gold (`sentiment_gold.csv`) | 180: summaries and company texts (120) and extractor texts (60) | 82 positive, 65 negative, 33 neutral | `key`, `article_id`, `ticker`, `kind`, `part`, `text_hash`, `revision`, `stratum`, `extractor`, `predicted`, `population`, `pub_date`, `sentiment`, `notes` | the exam of the pin rule; never tuned on; not yet scored with the stack |
| Training set 1 (`sentiment_train.csv`) | 450 summaries (424 read as a pair) | 232 positive, 113 negative, 105 neutral | `key`, `article_id`, `ticker`, `company`, `body_hash`, `rule_scope`, `revision`, `predicted`, `pool_class_size`, `block`, `batch`, `slice`, `text_hash`, `pub_date`, plus the mark | three batches of 150, 150 per class the old model predicted |
| Training set 2 (`sentiment_train_2.csv`) | 405 summaries (385 read as a pair) | 178 positive, 116 negative, 111 neutral | as above | same design, 135 per predicted class |
| Replay | 341 sentences drawn of 925 eligible (those that name their article's company) | | | read as a pair with that company, mixed into every training set |
| **Held-out** (`gold/sentiment_val190.csv`) | 190 pieces of the corpus run, two draws (90 + 100); never training data | 96 positive, 62 negative, 32 neutral | `key`, `article_id`, `item_id`, `ticker`, `batch`, `text_source`, `text_hash`, `predicted`, `predicted_score`, `relevance_score`, `pub_date`, the three revisions, `sentiment`, `notes`; the texts are in `gold/texts/sentiment_val190_texts.csv` | pieces (sub-news) of the stored summary step (`summary-7a9bd2062bc4`), with a main company of the company step and a sentiment score; not summaries of articles: 38 are distilbart summaries, the rest original short texts; draw 1 (90): 30 per predicted class, no text of the training sets or the gold; draw 2 (100): simple random, a stricter filter (relevance 0.9 or more, the company named twice and early); the model's own call is kept apart from the owner's tag |

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

Selected on 2026-11-10 (the 190-piece held-out came after, and took no part in the selection): the pair model started from the project's endpoint `gamug/FinBERT-financial-news`, trained on all 809 rows plus the 341 replay sentences, stacked with the fine-tuned relevance model. The pin is not decided: it waits on the 180-text gold by the rule of D35.

**Metrics of the stack (the step), validation and held-out in a single table.** Precision (P) and recall (R) one-vs-rest; the held-out carries 95% Wilson intervals and the weighted recall (each predicted class weighs the size of its universe of 57,826 candidates of the first draw, an approximation because the two draws have different universes).

| Measure | Nested cross-validation (809 rows, weighted) | Held-out (190 pieces of the real run) |
|---|---|---|
| Positive: P / R | 0.807 / 0.855 | P 83.0% [73.8, 89.4] (73/88); R 76.0% [66.6, 83.5] (73/96); weighted R 89.4% |
| Negative: P / R | 0.747 / 0.748 | P 69.4% [58.0, 78.9] (50/72); R 80.6% [69.1, 88.6] (50/62); weighted R 78.6% |
| Neutral: P / R | 0.787 / 0.686 | P 56.7% [39.2, 72.6] (17/30); R 53.1% [36.4, 69.1] (17/32); weighted R 4.4% |
| Macro P / R | 0.780 / 0.763 | not computed |
| Macro ROC OVR | 0.898 | not computed |
| Score (D37) / accuracy | score **0.814** | accuracy **73.7%** [67.0, 79.4] (140 of 190); 78.4% weighted to the universe |

The intermediate candidates (pair model alone 0.737, with the gate 0.789) and the pinned v1 model (0.648) are in the selection table above; the stack gains +0.077 [+0.057, +0.095] over the pair model alone.

**Against the bars of D32** (80% agreement; negative precision and recall of at least 70%): in the cross-validation the negative is above them (0.747 / 0.748). In the held-out the accuracy, 73.7%, is **under 80%**; the negative precision, 69.4%, is **under 70%** by point estimate and the recall, 80.6%, clears it.

**Limits.**
- **The cross-validation is not held out**, so it is optimistic; the final models are more confident than the fold models the stack was fitted on (not measured). The held-out is lower, but **the two do not measure the same thing** (marked summaries, other-company pairs included, against main pairs of pieces of the real run), so the gap is not a measure of the optimism.
- **Neutral is almost never predicted** in the run: the model calls 420 of the 57,826 texts of the universe neutral (0.7%), while the owner tagged 16.8% of the 190 pieces neutral; hence the weighted recall of 4.4%, a rate built from few tags (a hint, not a measurement). Recorded as T-212 (proposed; the owner decides the fix). Neutral is also the weakest class on its own (recall 0.44 without the relevance model): the model hears a company and a polarity but does not tell whether the text is about the company.
- **The two draws of the held-out differ**: the first (90, stratified by the prediction) gives accuracy 66.7% [56.4, 75.5] (74.4% weighted) and the second (100, random with a filter that makes the company the topic) 80.0% [71.1, 86.7]; neutral has 32 tags and 30 calls, with wide intervals. The 190 are of the distilbart summary revision; the DeepSeek summaries of the corpus run are not covered. One reader, no second reading; the line between neutral and a direction was not fixed in advance.
- **It is not the gate of D35** (the 180-text gold on main pairs, T-148c4, T-193): `config/pipeline_models.json` still names the v1 model; the held-out is evidence for the owner's decision. Marks, texts and scorer are tracked: `gold/sentiment_val190.csv`, `gold/texts/sentiment_val190_texts.csv`, `scripts/score_sentiment_val190_2026_11_16.py`.
- A shuffled-relevance control adds at most +0.005, so the gain of the stack is real information, not noise.
- The step reads summaries of at most about 2,000 characters, as chunks of at most the room the company leaves in 512 tokens, and averages the chunk probabilities. A sector text gets no sentiment (D42).

## 6. Category (step 5, `docs/category_step.md`, `docs/category_training.md`)

> **Planned change (not built; the pin, the step and every figure below are still the 12-label model).** The owner plans to take `product_innovation`, `leadership_governance`, `capital_shareholder_returns` and `reputational_harm` out of the model to improve its metrics, and to put them together in `other`. The model would then have **8 labels: 7 measured ones** (`earnings_performance`, `market_analyst_sentiment`, `mergers_acquisitions`, `legal_regulatory`, `labor_human_capital`, `partnerships_business_dev`, `investor_activity`) **and `other`, which stays as a tag that is not measured**: it means "the model has nothing to say about this text", so no precision or recall is reported for it. The figures below are not recomputed for this definition and must be re-measured once it is built (new tag files, a new fit, a new pin).

**Since 2026-11-15 (T-203, D43) the step runs the entailment fine-tune** (the recipe of `config/category_model.json`, 12 labels, one tag per row, refit on 1,216 rows; the pin is `config/category_step.json`, `category-entailment-single-12`), **pinned although the 80% bar (D24) is not met** (owner's decision). The step's earlier model, the untouched zero-shot, is kept as the comparison. Every run is in MLflow, experiment `news_nlp_train/category` (`uv run mlflow ui --backend-store-uri ./mlruns`); the weights are the registered model `category-entailment-single-12` (version 1) and the private Hub copy `gamug/deberta-v3-news-category-12`.

### Dataset description

Each row is a (text, company) pair with one label of the taxonomy of `docs/category-taxonomy.md`: **14 labels in 6 groups** for the step's earlier model (zero-shot); the fine-tune the step runs reads 12 (`competitive_actions` and `peer_comparison` read as `other`).

| Set | Size | Characteristics | Columns |
|---|---|---|---|
| Train rows (older sheet) | 225 rows with a company label | single bodies and structured multi bodies split by company, from the scope training set; narrative multi bodies left out; baseline hypotheses were written against it | `article_id`, `company`, `label`, `pred` (results files) |
| **Single-tag training rows (the step's fine-tune)** | 1,216: a 1,000-row sheet reduced to one tag per row (999) plus 217 of the 225 older rows | 12 labels; `earnings_performance` 547, `market_analyst_sentiment` 214, `product_innovation` 90, `mergers_acquisitions` 81, `legal_regulatory` 45, `leadership_governance` 40, `capital_shareholder_returns` 40, `reputational_harm` 39, `labor_human_capital` 36, `partnerships_business_dev` 32, `other` 32, `investor_activity` 20; 82% are short company segments of round-ups (median 272 characters); tags were reviewed after first results were seen | per row: text, company, label, `fold` (5 folds grouped by article) |
| **Held-out** | 200 texts, of which 135 single rows are scorable (116 single bodies); 44 narrative `multi` texts not scored | full articles (median 2,085 characters, none under 600), earlier tag convention (analyst ratings could be `investor_activity`); scored once | `article_id`, `company`, `kind`, `label`, `pred`, plus the 14 label scores in the mixed run |
| Real sub-news sample | 300 rows | a seeded draw in two batches of 150 from the sub-news pairs of the first corpus run whose summarised text is under 1,000 characters (142,588 of 159,401 pairs); one text per body; none from a tagged body; 107 of the 300 are flagged unclear | text, company, label, flags |

### Model selection

The step runs the pinned entailment fine-tune; the earlier model, the untouched zero-shot `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` at its pin, reads the premise `<company>: <first 2,000 characters>` with one hypothesis per label ("The news about <company> is about <definition>"), the best entailment probability gives the label, and its group gives the group (6 groups, option B). **Neither reaches the 80% bar (D24) on the held-out.**

| Candidate | Result | Verdict |
|---|---|---|
| Untouched zero-shot (the step's earlier model) | held-out 43.0% (label, 14 labels), 55.6% (group); 56.4% on the train rows | replaced by the fine-tune |
| Flat fine-tune, 5-fold cross-validation on 225 rows | 64.0% out of fold | better than zero-shot out of fold |
| Mixed fine-tune (labels plus groups), 225 rows | 62.2% out of fold; held-out 52.6% (flat), 48.9% (two-step), group 62.2% | not published, not used |
| **Entailment fine-tune**, refit on 1,216 rows, 12 labels | 87.8% out of fold; **held-out 54.1%** [45.9, 62.3]; 71.3% on 300 real sub-news | **the step's model (T-203)**, registered as `category-entailment-single-12` v1 |
| Plain DeBERTa-v3-base retrained on the same rows and folds | 85.7% out of fold | 2 points under the entailment recipe |

The held-out is the only measure that is held out and it fails; the fine-tune falls far from its 87.8% because its training rows are short round-up segments and the held-out rows are full articles. It was pinned by the owner's decision knowing that the bar is not met.

**Metrics of the fine-tune (the step's model), the three evaluations in a single table.** One-vs-rest precision and recall per label, with the number of rows in brackets (the 12-label model's rows: 1,216, 135 and 300; the accuracy and macro F1 rows use the planned distribution, 975, 84 and 235 rows, see the note under the table). **Out of fold:** 1,216 single-tag rows, 5 folds grouped by article; the tags were reviewed after the first results were seen and by the assistant, so it measures how well the recipe learns those tags (rows: short round-up segments, median 272 characters). **Held-out:** 135 single rows of 200 texts, the owner's tags of the earlier convention (full articles, median 2,085 characters), scored **once**. **Real sub-news:** 300 rows of the first corpus run (summarised text under 1,000 characters, one text per body), tagged jointly by the owner and the assistant, blind to both models (107 of the 300 marked unclear).

| Label | Out of fold P / R (rows) | Held-out P / R (rows) | Real sub-news P / R (rows) |
|---|---|---|---|
| earnings_performance | 0.97 / 0.97 (547) | 0.74 / 0.93 (27) | 0.82 / 0.89 (56) |
| market_analyst_sentiment | 0.89 / 0.88 (214) | 0.42 / 0.47 (17) | 0.81 / 0.80 (99) |
| mergers_acquisitions | 0.80 / 0.89 (81) | 0.36 / 0.80 (5) | 0.80 / 0.80 (30) |
| legal_regulatory | 0.72 / 0.73 (45) | 0.43 / 0.55 (11) | 0.77 / 0.77 (13) |
| labor_human_capital | 0.84 / 0.86 (36) | 0.33 / 0.67 (3) | 1.00 / 1.00 (4) |
| partnerships_business_dev | 0.77 / 0.75 (32) | 1.00 / 0.20 (5) | 0.65 / 0.77 (26) |
| investor_activity | 0.50 / 0.50 (20) | 0.67 / 0.12 (16) | 0.75 / 0.86 (7) |
| other (catch-all, not measured) | — | — | — |
| **Label agreement (accuracy), planned distribution** | **91.5%** [89.6, 93.1] (892 of 975) | **57.1%** [46.5, 67.2] (48 of 84) | **82.1%** [76.7, 86.5] (193 of 235) |
| Macro F1 (7 measured labels) | 0.791 | 0.462 | 0.818 |
| Selection score (D37) | not recomputed | not recomputed | not computed |
| Untouched zero-shot (the earlier model): accuracy, planned distribution | 85.6% [83.3, 87.7] (833 of 973) | 53.6% [43.0, 63.8] (45 of 84) | 78.7% [73.1, 83.5] (185 of 235) |

**Removed from the table for the planned definition:** `product_innovation`, `leadership_governance`, `capital_shareholder_returns` and `reputational_harm` (their rows would count as `other`), and `other` is shown as not measured. **How the rows below were obtained:** the 12-label model's predictions and the tags are remapped (the four labels read as `other`); this is **not a retrained 8-label model**, which has to be measured once it is built. **Accuracy and macro F1 are computed over the rows whose tag is one of the 7 measured labels** (975, 84 and 235 rows; a row of those predicted as `other` counts as wrong; the rows tagged `other` are not scored); the macro F1 is over all rows, one versus rest on the 7 labels, like the per-label table. The per-label precision and recall above do not change with the remap (the 7 labels' predictions and rows are the same). Intervals are Wilson (the 12-label ones were bootstrap by article); the selection score (D37) is not recomputed. The untouched model is read the same way, on the rows both models share (973 of its 1,214 rows out of fold). **Reproduce:** `scripts/category_planned_8_2026_11_16.py` (committed files only, no GPU) writes `docs/category_results_2026_11_15/planned_8_labels.json`; a test pins the figures quoted here. The 12-label figures (87.8%, 54.1%, 71.3%; macro F1 0.748, 0.448, 0.637) are in `docs/category_training.md` and `docs/category_results_2026_11_15/`.

Against the 80% bar (D24): **not met as measured** (54.1%; 57.1% on the planned distribution), **but this held-out cannot judge the bar for the texts the step will read** (note below); no measure on representative text exists yet; group agreement on the held-out 65.2% [57.4, 72.7].

**The official figure, and the four weak labels moved to `other` (owner, 2026-11-17).** The owner takes as the official measure of the model the **blind LLM judge on the 300 real sub-news** (`docs/category_results_2026_11_15/judge_step7_n300.md`; `deepseek-chat`, it never saw a model's label, it applies the project's tagging rules): the model agrees with the judge when its label is the judge's **first or second** label: **88.0% [83.8, 91.2] (264 of 300)** for the pinned model, against 81.0% [76.2, 85.0] (243 of 300) for the untouched zero-shot. Strict agreement (the first label only) is 77.0% against 71.0%. Reading the same rows through the planned definition (`scripts/category_planned_8_2026_11_16.py`, nothing retrained: the four weak labels count as `other` on both sides) raises every measure; "8 labels" scores `other` as a label, "7 measured" leaves out the rows whose reference is `other` (the planned definition does not measure it):

| Measure (pinned model) | 12 labels (as the step runs) | 8 labels (`other` scored) | 7 measured labels (`other` not scored) |
|---|---|---|---|
| **Judge on 300 real sub-news, official (first or second label)** | **88.0%** | **92.0% [88.4, 94.6]** | **94.4% [90.9, 96.7]** (252 rows) |
| Judge, strict (first label only) | 77.0% | 80.0% [75.1, 84.1] | 81.7% [76.5, 86.0] |
| Same judge, untouched zero-shot (first or second) | 81.0% | 85.7% | 90.9% |
| Out of fold (1,216 rows) | 87.8% | 89.4% [87.5, 91.0] | 91.5% |
| Held-out of the owner (135 rows, once) | 54.1% | 60.0% [51.6, 67.9] | 57.1% (84 rows) |
| 300 real sub-news, joint tags | 71.3% | 75.7% [70.5, 80.2] | 82.1% (235 rows) |

Read it with these limits: the judge is an LLM, not a person (the same kind of reader as the assistant's tags), it is **not repeatable** (2 of its 80 labels changed between two runs), and counting its second label is a lenient reading chosen by the owner; the 8-label and 7-label rows are the 12-label model read through the fold, **not a retrained 8-label model**, which must be measured when it is built. The D24 bar is defined on the held-out gold: the held-out stays below it (57.1%) and `bar_met` in `config/category_step.json` stays false; whether the judge figure replaces it for D24 is the owner's decision on record in `PLAN.md` (D43).

**Note on the held-out (owner's reading):** this sample was very dirty and is **not representative of the texts that will be categorized**, which are the summaries of the pipeline's summarization step. Read the held-out figures (54.1% as it is, 57.1% on the planned distribution) as a check on a hard, noisy set, not as the expected accuracy in production. Under the planned distribution only **84 of its 135 rows are scored** (interval 46.5 to 67.2), so it is also small.

**What it says and does not say.** (The bullets are about the 12-label model, with its figures 87.8%, 54.1% and 71.3%, unless they say otherwise.)
- **The three measures disagree because they are different texts and different taggers**; the held-out is the only one held out, and it is the one that fails (its sample was dirty, note above). It falls far from 87.8% because of the difference in texts (short segments against full articles) and because the tagging convention changed (the largest confusion is `investor_activity` predicted as `market_analyst_sentiment`, 8 rows).
- **The fine-tune is better than the earlier model on all three**, and the gain is clear on the 300 real rows (it is right where the earlier one is wrong in 38 rows and the reverse in 21; sign test p = 0.036). The assistant took part in tagging both the training rows and the 300 rows, so the agreement on the 300 rows is probably higher than an independent tagger would give (83.9% where the tagger was sure, 48.6% where not).
- **Only `earnings_performance` is tight.** Labels with fewer than about 10 rows in a set are anecdotes (one row moves the figure 10 points or more; the 95% intervals of the cross-validation are about ±0.12 to ±0.15 for classes of 20 to 45 rows). **`other` is the failure:** the fine-tune gets 3 of 39 on the real sub-news (ROC 0.68; the earlier model, 5 of 39, ROC 0.46); under the planned definition `other` is not measured and this failure leaves the figures.
- The per-label tables of the earlier model (14 labels), of the mixed fine-tune and the intervals are in `docs/category_results_2026_11_13/` and `docs/category_results_2026_11_15/` (`ovr_single.md`, `heldout_selected.md`, `real_sample.md`); the runs, in MLflow.

## What these figures do not say

- **The steps are not compared on one yardstick.** Each is measured on its own sample, with its own reviewer and its own definition of "right"; the summary table shows which are held out.
- **Error chains are not in these figures.** A wrong scope reaches the cut, the company step, the sector step and the category step as bad input; for example about half of the wrong sector answers of the first review were not sector texts,
  and 15 of the 143 sampled pieces came from texts the scope step should have stopped.
- **Four measures are on texts of the real corpus run**: the NER held-out (100 texts of step 2, 3.1), the main-company marks on pieces (3.2, partial), the sentiment check on 190 pieces (5) and the category real-data sample (6, joint tags). The funnel (texts per step, and how many reach a snapshot) comes from the real RESULTS store and is not in this report.
- **The three category measures are not one yardstick** (6): out of fold on tags the assistant reviewed, held-out on the owner's earlier tags (full articles), and a real-data sample tagged jointly by the owner and the assistant (short segments). Only the held-out is independent, and it fails the bar as measured (54.1% against 80%), on a sample that is not representative (6). The official measure is a fourth one, the blind LLM judge on the real sub-news (88.0%, 6): an LLM reading, lenient on the second label, not a person.
- **Not measured at all:** sector recall per sector, the cut's recall, main company per piece beyond the 119 marked rows of section 3.2, NER on the DeepSeek summaries, relevance on pieces beyond those marks, and the held-out score of the sentiment stack beyond the 190 pieces of section 5.

## Where each figure comes from

| Step | Source |
|---|---|
| Scope | `docs/scope_training.md` (sections 5, 6, 9); `data/scope_end_to_end_with81b.csv` and `gold/category_gold_v2_test.csv` (per-class table recomputed) |
| Sector | `docs/sector_step.md` ("The clean test", "What this does not show"); `gold/sector_validation.csv` (precision per sector recomputed) |
| Cut | `docs/scope_training.md` sections 8.1 and 8.3; `docs/extractor_training.md` (update of 2026-11-11, `units4` and `units5`); `gold/cut_sample_marks.csv`; `scripts/cut_sample_metrics_2026_11_12.py` |
| Company: NER | `docs/ner_training.md` (the 100-text held-out); `docs/ner_results_2026_11_16/` (`ner_metrics.json`, `ner_errors.csv`); `scripts/ner_eval_2026_11_16.py`; `docs/evaluation.md` (v1 baseline of 2026-09, `ner`) |
| Company: relevance | `docs/companies_step.md` (threshold table); `docs/sentiment_training.md` item 6 (AUC 0.90); `docs/main_company_training.md` and `gold/main_company_marks.csv` (the marks on pieces, the held-out, tracked) |
| Summarization | `docs/summarization_training.md` (arms, levels and DeepSeek's arm D); `docs/summarization_results_2026_11_15/` (tracked figures of arm D); `docs/evaluation.md` (v1 baseline); `docs/summary_step.md` (limits) |
| Sentiment | `docs/sentiment_step.md` ("What the figures are"); `docs/sentiment_training.md` (stack table, nested cross-validation); the 190-piece held-out: `gold/sentiment_val190.csv`, `gold/texts/sentiment_val190_texts.csv`, `scripts/score_sentiment_val190_2026_11_16.py` |
| Category | `docs/category_step.md`; `docs/category_training.md`; `docs/category_results_2026_11_13/` (earlier zero-shot, tables recomputed from `baseline_heldout.csv` and `baseline_train.csv`); `docs/category_results_2026_11_15/` (`ovr_single.md` out of fold; `heldout_selected.md` held-out, scored once by `scripts/category_heldout_2026_11_15.py`; `real_sample.md` and `scripts/category_real_sample_2026_11_15.py` for the 300 real rows, `judge_step7_n300.md` (the blind judge, official figure) and `planned_8_labels.json` (the fold), tags in `gold/category_real_sample.csv`); MLflow experiment `news_nlp_train/category` (`scripts/category_mlflow_log_2026_11_15.py`) and the registered model `category-entailment-single-12`; D37 scores from the MLflow runs |

## All the models together

One table with the general metrics of every model, to show the results at a glance. Every figure is measured above, in the section of its step, and copied here as it is; "validation" is cross-validation or out-of-fold, and "held-out" is texts the model was neither trained nor tuned on. A dash means that measure does not exist.

| Step | Model | Validation (training) | Held-out | Bar and status |
|---|---|---|---|---|
| Scope (step 1) | two NLI classifiers fine-tuned on `deberta-v3-base-zeroshot-v2.0` | — | weighted agreement 83.6% [78.1, 88.9] (200 texts); quarantined precision 90.8% (225); `out_of_scope` recall 53% | 80%: met by point estimate (lower end 78.1) |
| Sector (step 4) | `bge-small-en-v1.5` with a gate | — | 87.5% of the assigned sectors right (35 of 40); recall not measured | no bar measured (D24 for the sub-industry is not measured, T-167) |
| Cut (step 1.2) | `units5` with the pair classifier (`gamug/FinBERT-news-attribution`) | — | 84.6% [77.8, 89.6] of pieces faithful; 75.5% clean (143 pieces) | 80% faithful: the point estimate passes, the lower end does not |
| NER (step 3) | `gamug/sec-bert-finer-ord-ner` | v1 LLM-judged run (1,000 articles): span F1 0.7418, ORG 0.75 | universe companies: P 92.6%, R 95.5%, F1 0.940 (R 98.7% with `ner+text`); all organisations: P 72.6%, R 90.4% (100 texts; the overlay on 14 labels: 9 replaced, 1 kept, 4 dropped); as they are, universe P 90.1%, R 95.4% | no bar |
| Relevance (step 3) | `gamug/deberta-v3-company-relevance` | AUC 0.90; main P 90.0%, R 94.7% (809 summaries) | AUC 0.663; main P 90.9%, R 30.0% (119 rows of 59 pieces) | no bar; the sub-news threshold is the owner's decision (T-170) |
| Summarization (step 2) | DeepSeek (`deepseek-chat`) | — | no held-out; LLM judge on 300 articles: faithfulness 4.99, coverage 4.54, hallucination 0.7% (the judge is DeepSeek too) | no bar; the corpus run is not measured |
| Sentiment (step 6) | pair model + stack (`gamug/FinBERT-company-sentiment`) | score 0.814; negative P/R 0.747 / 0.748 (809 rows) | accuracy 73.7% [67.0, 79.4]; negative P 69.4%, R 80.6% (190 pieces) | 80% agreement and 70% on the negative (D32): **not met** (accuracy, negative precision) |
| Category (step 5) | entailment fine-tune (`category-entailment-single-12`) | accuracy 91.5%, macro F1 0.791 (975 rows of the 7 measured labels) | **judge on 300 real sub-news (official): 88.0%**; accuracy 57.1% [46.5, 67.2] (84 rows, held-out); 82.1% on 235 real sub-news (joint tags) | 80% (D24): the held-out **is not met**; the official judge figure (88.0%, 300 real sub-news) is above it; pinned by the owner's decision; figures are the 12-label model remapped to the planned distribution, not retrained (section 6; 12-label: 87.8%, 54.1%, 71.3%) |
