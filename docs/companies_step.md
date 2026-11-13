# Step 3: the companies (T-164, T-165; PLAN D39 (2), (3.1), (3.2), D40)

`uv run cli/news_nlp_steps.py --step companies --summary-revision REV --threshold T --candidates ner|ner+text [--cut-revision REV | --items single] [--items sub_news] [--scope-revision REV] [--limit N]` (and `--step companies --status`)

## What it does

For every `single` body of step 1 and every sub-news of the cut it reads **the text step 2 left** (`news_summary`: the summary, or the item itself when it was under the size), and:

1. **NER** (`config/pipeline_models.json` `ner`, `gamug/sec-bert-finer-ord-ner@ba7b9e43`; `ner_stage`'s own feature and inference, reused as they are) finds the organisation spans (3.1).
2. **The matcher** (`news_nlp.triage.CompanyResolver.ticker_of_name`) turns each span into a universe ticker. A span that matches no ticker is **ignored**: it is not stored and never main (D39 (iii)). A span matches by the company's normalized name, the first word of its name when that word is unique and not an everyday word, a short form, or the ticker itself written in capitals and at least three letters long (the NER model tags `MMM` as an organisation; `ON` and `ALL` are words and do not count). The number of spans that name a ticker is its `mention_count`.
3. **The relevance model** (`gamug/deberta-v3-company-relevance@8812c8bc`, private: `HF_TOKEN`; `config/published_models.json`) scores each (text, ticker) pair (3.2). It reads the premise `<text>` and the hypothesis `This text is mainly about <company>.` at 320 tokens, as it was trained (`scripts/final_models_2026_11_10.py`); the score is the probability of entailment, in float32.

Every ticker whose score is **at the threshold or over it is main** (D39 (i)); the others stay as mentions. Rows go to `news_company` (one per item and ticker: `relevance_score`, `is_main`, `mention_count`); `news_company.is_main` is the only source of the main ticker (D40). An item with **no company found, or none over the threshold**, is stored `review` in `pipeline_step_state` with the reason in `error` (D39 (ii); no sentiment until the owner reads it; the list and its export are T-172). A `review` item is not retried at the same revision (a rerun would give the same answer). A text that the models cannot read is `failed` and is retried; a batch that fails is redone item by item, and three batches in a row that fail completely stop the run (as in step 2).

The step is idempotent like the others: a rerun at the same revision adds nothing. It reads, by name and never "the latest": the **summary revision** (`--summary-revision`, `--step summary --status` lists them), the **cut revision** for sub-news (`--cut-revision`) and the **scope revision** for single bodies (`--scope-revision`, default the latest). Options the step does not read are refused.

## Two choices the owner made, so the command asks for both

**Decided by the owner on 2026-11-13 (interactive session): `--threshold 0.5` (revisited with T-170 for sub-news; another value is a new revision), `--candidates ner+text` and the raw spans not stored.** The owner chose `ner+text` on the figures of its **first definition** (the matcher's mentions with no filter: a candidate company in 99.5% of the sampled sub-news, a main one in 75.0%). The definition was corrected in the review of #178 (lower-case matches and disclosures left out) and the figures were measured again, below; **the owner confirmed the corrected definition on 2026-11-13 (interactive session), accepting its limits** (a pattern match: a disclosure in other words passes, a real mention within 160 characters of a disclosure phrase is dropped; T-170's hand check will show if it matters).

The decided values are not defaults. Neither is guessed: `--threshold` and `--candidates` have no default and the command needs them stated, because both are part of the revision (`companies-` plus 12 hex digits of the SHA-256 of the two models and their revisions, the threshold, the candidates, the universe, the hypothesis, the length, the `transformers` version and the code. **The code is the closure of what the step reaches**, as parsed (comments, formatting and docstrings do not count): this module whole, and, in `triage`, `ner_stage` and `chunking`, every function, class and constant that the names this module takes from them reach, followed through the code of each (`news_company_stage.hashed_sources()`). Editing `normalize_name` or the list of word tickers starts new rows; editing the scope features of `triage`, which the matcher never reaches, does not). **A new value starts new rows and leaves the old ones** (D39 (vi)); the cost of trying a value is one run of the step (about 15 to 40 items a second on the GPU in the dry run below, so a few hours for the whole corpus).

**`--candidates`: where the companies of a text come from.** `ner` is D39 (3.1) as written: only what NER tags. `ner+text` also runs the matcher over the text (`news_company_stage.direct_mentions`: the mentions `CompanyResolver.counted_matches` counts, without the matches that do not start with a capital letter or a digit and without those inside a disclosure: `triage.in_disclosure` (phrases such as "is long" or "owns", and a run of five or more ticker symbols whatever number of them are in the universe) or one of the boilerplate words of `_DISCLOSURE_WORDS` within 160 characters ("or its affiliates", "an associate of", "is a trustee of", "investment banking client", "within the past 12 months", "employees of … own")) and adds the tickers it finds that NER did not tag. The rules are pattern matches: **a disclosure written in other words passes, and a real mention within 160 characters of a disclosure phrase is dropped with it.** A ticker both find keeps the **larger of the two counts**: `mention_count` counts mentions in the text, not the readers' sum, and later steps read it as that; the two counts are not stored apart. Why it is asked: NER does not tag every name (it does not tag `3M`, for one), and the sub-news were cut by the same matcher, so each is about a universe company by construction.

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
| a candidate company, `ner+text` | 142 (71.0%) | 187 (93.5%) |
| **a main company, `ner`** | 119 (59.5%) | 141 (70.5%) |
| **a main company, `ner+text`** | 123 (61.5%) | 146 (73.0%) |
| candidates but none over 0.5, `ner+text` | 19 | 41 |
| the main company is a ticker the crawl gave the body, of the bodies with a main company | `ner`: 99 of 119 (83%); `ner+text`: 102 of 123 (83%) | no crawl ticker per sub-news |
| (text, ticker) pairs found by the text and not by NER, `ner+text` (how many of them are main) | 10 (4) | 23 (7) |

**Precision of the text candidates (developer's reading, not the owner's).** Three definitions were measured on this sample, and the counts are of different things, so they are not one series:

1. *The matcher unfiltered* (`CompanyResolver.mention_spans`): **96 (text, ticker) pairs** that NER did not find. A look at the same texts through `counted_matches` with the capital-letter filter gave **109 matches** not found by NER; I read 40 of those 109, and about 30 of the 40 sat in disclosure lines ("Tim Seymour is long AMZN, AAPL, …", "Citigroup Global Markets or its affiliates") or were false friends ("Principal Financial Officer", "Flex LNG"). No lower-case match occurred in the sample; the toy example "Our target is to hit the gap by apple season", which gave Gap and Apple, is removed by the **lower-case** rule, which a test pins.
2. *With only `triage`'s own disclosure rules* (the head of the first review, `914bf04`): **35 pairs**, 13 of them main. I read the 13: 10 right, 3 wrong, and two of the three were disclosures the rules did not catch (a list of seven ticker symbols with two in the universe, and "Citigroup Global Markets … is a trustee of (TGT)").
3. *With the symbol-run and boilerplate-word rules added* (this head): **33 pairs**, 11 of them main. I read the 11: 10 right (3M, CrowdStrike, Target, Best Buy, Lilly, Robinhood, LyondellBasell, Halliburton, Lowe's, Linde), 1 wrong (a list entry, "-Chevron -Elan Corp"). 11 is a reading, not a bar.

The share of items with a main company at other thresholds (`ner+text`):

| threshold | 0.1 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 |
|---|---|---|---|---|---|---|
| single bodies | 65.5% | 63.5% | 62.5% | 61.5% | 61.0% | 60.5% |
| sub-news | 84.0% | 80.0% | 76.5% | 73.0% | 70.5% | 65.5% |

What this says, and what it does not:

- **On sub-news the threshold matters, and the 809 summaries do not tell which value is right.** The relevance model never saw a sub-news (T-165's risk, T-170): a fifth of them have a candidate but a best score under 0.5, and the share with a main company moves from 65.5% to 84.0% across the range. Nothing here says whether the lower scores are truly not-main or the model is unsure on short text. T-170 measures it with a set the owner marks.
- On single bodies 38.5% have no main company even with the matcher over the text: some name no universe company, some name one only in a form the matcher does not know (the short names of T-181); the split was not measured.
- Agreement with the crawl's ticker (83% of the bodies with a main company) is a floor on correctness of the pick, not a measure: the crawl's ticker is a tag of the search, D36 keeps it apart from the model's pick, and a body about two companies has two.
- A dry run of 200 and 200 gives an interval of about ±6 points on each share; these are order-of-magnitude figures for choosing, not results.
- The first 200 items by row order are all 3M (the first company in the table): they showed that NER does not tag `3M` and that tickers come as spans, and are not in the figures above.

## Run it in slices, and again

`--limit` slices follow the order of `news_summary`: the single bodies by their summary row, the sub-news by id, so the first slices are one company at a time (3M comes first) and the first `--status` shows it. `--step companies --help` prints the recipes to run the step again at one revision (`RECOMPANIES_SQL`; an item on the review list is not retried by itself) and, in `RECUT_SQL`, the deletes of the companies of a cut revision that go before a re-cut, because `news_company` is keyed by `sub_news.id`.

## What is not stored

The raw NER spans do not go to `article_entities` (D39 (iii) put them there with `sub_news_id` and `ticker`). `article_entities` is keyed by article and read by the v1 NER stage as "this article is done"; a single body has no `sub_news_id`, and the offsets of a summary are not those of the article. The step stores the result that later steps read (`news_company`: ticker, score, `is_main`, `mention_count`); the spans can be recomputed, the NER model being deterministic. If the owner wants them stored, it is a table of its own (a proposal, not built).

## Limits

- The relevance model was trained on summaries (T-170): its scores on sub-news are not validated.
- The matcher's gaps (short names such as `Disney` or `JPMorgan`, T-181) lose those spans; `ner+text` recovers what the matcher knows, not what it does not.
- **The step needs `HF_TOKEN`.** The relevance repository is private and is read from `config/published_models.json`, not from `config/pipeline_models.json` where the other models are pinned (Pipeline models 1); it is pinned by T-148c4. `--status` imports the stage module (a few seconds) and loads no model; it prints the two models and their revisions.
- **CPU and GPU.** The scorer uses bf16 autocast on CUDA only, as the model was trained and scored; the CPU runs float32. On 220 real pairs the two devices differ by at most 0.021 (mean 0.0014) and no pair changes side of 0.5. The revision does not carry the device.
- A text over 320 tokens is read up to 320 (the model's length); step 2 keeps a text under about 2,000 characters, roughly 450 tokens, so the end of a longer original may not be read.
- `--status` prints the counts by state (`review` apart) and, for the revision, the rows, the main rows, the items and the items with a main company.

## The owner's marks (T-170, 2026-11-13)

Three files, in `data/` (git-ignored), made from a copy of the RESULTS database while the step was running:

- `scripts/main_company_marks_2026_11_13.py draw`: the sheet of main companies, one row per (sub-news, candidate), **no score on it**. The sub-news come from six bands of the best relevance score (0.1, 0.3, 0.5, 0.7, 0.9), the same number from each (default 30), so the bands near the threshold are marked too. The sub-news with no candidate are not on it. The bands and their populations are in the sidecar `.strata.json`, which `score` reads to weigh the marks back to the corpus. The owner writes `si` or `no` in `es_principal`, and the ticker in `falta_la_principal` when the main company is not among the rows.
- `... spans`: 50 organisation spans of 150 random sub-news that the matcher turned into no ticker, for the owner to say which are universe companies.
- `... score`: the weighted AUC of the score against the owner's marks, the precision and recall at each threshold, the share with a main company, and the rule of T-170 (AUC more than 5 points under 0.90: sub-news get a threshold of their own).

`scripts/translate_marks_sheets_2026_11_13.py` adds the Spanish aid to these sheets and to the summary judge's slice (paid LLM calls, cached, `--yes` to buy). Single bodies and the sector texts are not in the sheets yet.
