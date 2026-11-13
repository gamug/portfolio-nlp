# Step 3: the companies (T-164, T-165; PLAN D39 (2), (3.1), (3.2), D40)

`uv run cli/news_nlp_steps.py --step companies --summary-revision REV --threshold T --candidates ner|ner+text [--cut-revision REV | --items single] [--items sub_news] [--scope-revision REV] [--limit N]` (and `--step companies --status`)

## What it does

For every `single` body of step 1 and every sub-news of the cut it reads **the text step 2 left** (`news_summary`: the summary, or the item itself when it was under the size), and:

1. **NER** (`config/pipeline_models.json` `ner`, `gamug/sec-bert-finer-ord-ner@ba7b9e43`; `ner_stage`'s own feature and inference, reused as they are) finds the organisation spans (3.1).
2. **The matcher** (`news_nlp.triage.CompanyResolver.ticker_of_name`) turns each span into a universe ticker. A span that matches no ticker is **ignored**: it is not stored and never main (D39 (iii)). A span matches by the company's normalized name, the first word of its name when that word is unique and not an everyday word, a short form, or the ticker itself written in capitals and at least three letters long (the NER model tags `MMM` as an organisation; `ON` and `ALL` are words and do not count). The number of spans that name a ticker is its `mention_count`.
3. **The relevance model** (`gamug/deberta-v3-company-relevance@8812c8bc`, private: `HF_TOKEN`; `config/published_models.json`) scores each (text, ticker) pair (3.2). It reads the premise `<text>` and the hypothesis `This text is mainly about <company>.` at 320 tokens, as it was trained (`scripts/final_models_2026_11_10.py`); the score is the probability of entailment, in float32.

Every ticker whose score is **at the threshold or over it is main** (D39 (i)); the others stay as mentions. Rows go to `news_company` (one per item and ticker: `relevance_score`, `is_main`, `mention_count`); `news_company.is_main` is the only source of the main ticker (D40). An item with **no company found, or none over the threshold**, is stored `review` in `pipeline_step_state` with the reason in `error` (D39 (ii); no sentiment until the owner reads it; the list and its export are T-172). A `review` item is not retried at the same revision (a rerun would give the same answer). A text that the models cannot read is `failed` and is retried; a batch that fails is redone item by item, and three batches in a row that fail completely stop the run (as in step 2).

The step is idempotent like the others: a rerun at the same revision adds nothing. It reads, by name and never "the latest": the **summary revision** (`--summary-revision`, `--step summary --status` lists them), the **cut revision** for sub-news (`--cut-revision`) and the **scope revision** for single bodies (`--scope-revision`, default the latest). Options the step does not read are refused.

## Two choices the owner makes, so the command asks for both

Neither is guessed: `--threshold` and `--candidates` have no default, and both are part of the revision (`companies-` plus 12 hex digits of the SHA-256 of the two models and their revisions, the threshold, the candidates, the universe, the hypothesis, the length, the `transformers` version and the code, as parsed: `news_company_stage.hashed_sources()`). **A new value starts new rows and leaves the old ones** (D39 (vi)); the cost of trying a value is one run of the step (about 15 to 40 items a second on the GPU in the dry run below, so a few hours for the whole corpus).

**`--candidates`: where the companies of a text come from.** `ner` is D39 (3.1) as written: only what NER tags. `ner+text` also runs the matcher over the text (`CompanyResolver.mention_spans`) and adds the tickers it finds that NER did not tag; a ticker both find keeps the larger count. Why it is asked: NER does not tag every name (it does not tag `3M`, for one), and the sub-news were cut by the same matcher, so each is about a universe company by construction.

**`--threshold`: from which relevance score a company is main.** The out-of-fold scores of the 809 marked summaries (`scripts/relevance_threshold_candidates_2026_11_13.py`, unweighted):

| threshold | relevant rows kept | not-relevant rows called main | precision |
|---|---|---|---|
| 0.30 | 96.2% | 33.2% | 89.7% |
| 0.50 | 94.7% | 31.7% | 90.0% |
| 0.70 | 92.3% | 27.2% | 91.1% |
| 0.90 | 87.3% | 23.8% | 91.7% |

On summaries the threshold hardly matters (the scores are close to 0 or 1), and no value separates the mentions well: between a quarter and a third of the rows the owner marked not relevant (neutral) are called main, whatever the threshold.

## Dry run (2026-11-13, on a copy of the RESULTS database; nothing in the real one)

Step 2's output at `summary-7a9bd2062bc4` was there for 72,114 single bodies and 156,526 sub-news of `units5-d3e2cbfa830e` (the owner's corpus run, in progress). A **random** sample of 200 of each (seed 20261113), threshold 0.5:

| | single bodies | sub-news |
|---|---|---|
| a candidate company, `ner` | 137 (68.5%) | 178 (89.0%) |
| a candidate company, `ner+text` | 146 (73.0%) | 199 (99.5%) |
| **a main company, `ner`** | 119 (59.5%) | 141 (70.5%) |
| **a main company, `ner+text`** | 124 (62.0%) | 150 (75.0%) |
| candidates but none over 0.5, `ner+text` | 22 | 49 |
| the main company is a ticker the crawl gave the body (`ner+text`) | 103 of 124 (83%) | no crawl ticker per sub-news |
| two or more main companies, `ner+text` | 10 | 33 |

The share of items with a main company at other thresholds (`ner+text`):

| threshold | 0.1 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 |
|---|---|---|---|---|---|---|
| single bodies | 66.5% | 64.0% | 63.0% | 62.0% | 61.5% | 61.0% |
| sub-news | 89.5% | 84.0% | 80.0% | 75.0% | 72.0% | 66.5% |

What this says, and what it does not:

- **On sub-news the threshold matters, and the 809 summaries do not tell which value is right.** The relevance model never saw a sub-news (T-165's risk, T-170): a quarter of them have a candidate but a best score under 0.5, and the share with a main company moves from 66.5% to 89.5% across the range. Nothing here says whether the lower scores are truly not-main or the model is unsure on short text. T-170 measures it with a set the owner marks.
- On single bodies 38% have no main company even with the matcher over the text: some name no universe company, some name one only in a form the matcher does not know (the short names of T-181); the split was not measured.
- Agreement with the crawl's ticker (83% of the bodies with a main company) is a floor on correctness of the pick, not a measure: the crawl's ticker is a tag of the search, D36 keeps it apart from the model's pick, and a body about two companies has two.
- A dry run of 200 and 200 gives an interval of about ±6 points on each share; these are order-of-magnitude figures for choosing, not results.
- The first 200 items by row order are all 3M (the first company in the table): they were the ones that showed NER and the ticker spans, and are not in the figures above.

## What is not stored

The raw NER spans do not go to `article_entities` (D39 (iii) put them there with `sub_news_id` and `ticker`). `article_entities` is keyed by article and read by the v1 NER stage as "this article is done"; a single body has no `sub_news_id`, and the offsets of a summary are not those of the article. The step stores the result that later steps read (`news_company`: ticker, score, `is_main`, `mention_count`); the spans can be recomputed, the NER model being deterministic. If the owner wants them stored, it is a table of its own (a proposal, not built).

## Limits

- The relevance model was trained on summaries (T-170): its scores on sub-news are not validated.
- The matcher's gaps (short names such as `Disney` or `JPMorgan`, T-181) lose those spans; `ner+text` recovers what the matcher knows, not what it does not.
- A text over 320 tokens is read up to 320 (the model's length); step 2 keeps a text under about 2,000 characters, roughly 450 tokens, so the end of a longer original may not be read.
- `--status` prints the counts by state (`review` apart) and, for the revision, the rows, the main rows, the items and the items with a main company.
