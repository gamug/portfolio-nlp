# The category step (`--step category`, T-168, T-203; fine-tuned model pinned by the owner although the bar is not met)

One of the **12 labels** (the 14 of D15, `docs/category-taxonomy.md`, with `competitive_actions` and `peer_comparison` read as `other`, owner 2026-11-14) and its group, for each **main company** of each text.
Since 2026-11-15 (T-203, D43) the step runs **the entailment fine-tune `category-entailment-single-12`** (`config/category_step.json`, a git-ignored folder; training, datasets and every measure:
`docs/category_training.md`), **pinned as the default by the owner although it does not meet the bar of D24** (80% on the held-out single rows): it agrees with the owner on 54.1% of them
(the untouched zero-shot model: 43.7%) and with the assistant's tags on 71.3% of 300 real sub-news under 1,000 characters (65.7%). The labels are a first version, not the accepted ones. The rows the
untouched model wrote (revisions `category-65f33b55b26b` and `category-981dacf6f68d`, 14 labels) stay beside this revision, unchanged.

```
uv run cli/news_nlp_steps.py --step category --summary-revision summary-... --companies-revision companies-... --limit 500
uv run cli/news_nlp_steps.py --step category --status
```

`--step summary --status` and `--step companies --status` list the revisions that ran; both are named, never the latest.
`--source-db` and `--results-db` (or `SOURCE_DATABASE_URL` and `DATABASE_URL`) name the two stores, as for the other steps. A rerun
adds nothing: an item already done or skipped at the revision is not read again, a failed one is retried. The step is resumable,
so a long run can be cut with `--limit` and continued; before it starts it prints a warning if some main-company items have no text at the summary revision named (the
companies step read another one), then the model and the revision, then how many items and pairs it will do; the model is loaded before the progress bar.

To run the step again at one revision, or after a re-cut of the sub-news, the SQL is in `--help` (`RECATEGORY_SQL`, `RECUT_SQL`).

## What it reads and writes

* **Reads:** the texts of step 2 (`news_summary` at `--summary-revision`) of every item that has a **main** company at
  `--companies-revision` (`news_company.is_main = 1`). An item on the review list of step 3 has no main company and gets no category.
* **Writes:** `news_category`, one row per (item, main ticker, revision) (no column was added): `label`, `score` (the entailment
  probability of the label), `group_label`, `group_score` (the share of the probability mass the group's labels hold, 0 to 1),
  `slug_scores` (JSON, the probability of each of the 14 labels), `model_name`. The state of the item is in `pipeline_step_state`
  (`step = 'category'`).

## Dating a row and joining it to its article

A row is a (text, main company) pair; to aggregate by day, join it to the article the text comes from. For `item_kind = 'sub_news'` (every row of the first corpus run: step 3 has not read the single
bodies yet, so there is no `body` row; its join will be written when there is one):

```sql
SELECT c.ticker, c.label, c.group_label, c.score,
       substr(a.pub_date, 1, 10) AS news_date, a.source_domain
FROM news_category c
JOIN sub_news n ON c.item_kind = 'sub_news' AND CAST(n.id AS TEXT) = c.item_id
JOIN articles a ON a.id = n.article_id
WHERE c.model_revision = '<category revision>';
```

Three things to know:
1. **The ticker of the row is the company the text is about, not the ticker of the article.** The sub-news of a round-up belongs to the first article of its body, whatever the company: a row for `DHR`
   can join to an article tagged `MMM`. Group by `c.ticker`; use the article only for the date and the source.
2. **`pub_date` has more than one format** in the same column (`2023-04-10T15:40:03Z` next to `2026-07-21T11:21:21+0000`); both start with the day, so `substr(pub_date, 1, 10)` is the day as written
   (no time zone is applied). Do not sort or cut it as text further than that.
3. **The same body is the text of several articles** (about a third of the `ok` bodies are shared by several tickers): join to `sub_news.article_id` as above, never from `articles` to `news_category`
   by body, or a row is counted once for each article.

## Which revision to read

Revisions coexist in `news_category`: the revision follows what decides a score and the two input revisions (below), so a change of any of them starts
a new set of rows beside the old one. **Read one revision, never a mix, and only a complete one.** A revision is complete when its finished items (`done` or `skipped`) are as many as
the items that have a main company at your companies revision, less the `items_without_text` the run printed (items whose text is not at the summary revision named cannot be done):

```sql
-- the revisions of the step and how many items each finished
SELECT model_revision, COUNT(*) AS finished, MAX(finished_at) AS last_item FROM pipeline_step_state
WHERE step = 'category' AND status IN ('done', 'skipped') GROUP BY model_revision ORDER BY last_item DESC;
-- the items the step has to do
SELECT COUNT(DISTINCT item_kind || ':' || item_id) FROM news_company
WHERE model_revision = '<companies revision>' AND is_main = 1;
```

**The revision follows what decides a score and nothing else:** the pinned model, the two input revisions (the summary and the companies revision), the definitions and groups, the premise and the hypothesis, the lengths and the code that scores (`score_of`, `_read_pair`, `company_name`), how the model is loaded (`CategoryScorer._load`: the precision changes outputs) and the order of the definitions (a tie goes to the first label); the bookkeeping of `CategoryScorer.score` is not part of it. An edit of the command, the storage, the logging or a document is **not** a new revision, so a run you stop and start again with the same command resumes: it does only the items that have no state at that revision, and no row is moved. A step 2 or step 3 run again at a new revision, or a change of a definition, is a new revision (new work), and then `--step category --status --summary-revision S --companies-revision C` prints `code_revision`, the revision the code you run would write for those inputs (no model is loaded for it): compare it with the revisions of the first query before you start.

Read the newest revision that is complete (the first row of the first query that reaches the number of the second, less `items_without_text`); delete an old or partial one with `RECATEGORY_SQL` (`--help`). A run stopped by a GPU fault or by you leaves a partial revision: rerun the same command and it completes it.

**The first run of the development** (2026-11-13, commit `82d6b36`) wrote `category-65f33b55b26b` with the untouched zero-shot model and 14 labels, at a time when the revision still hashed the whole module; the second run
(`category-981dacf6f68d`, 10,355 pairs) is entirely inside it. **`LEGACY_REVISIONS`, which let the current code continue that run, is retired** (T-203): the fine-tuned model changes every score, so the revision this code writes
for the same inputs is another one, and a run starts that revision from zero. Delete the old revisions with `RECATEGORY_SQL` when the owner no longer wants them (T-189). The verification and the adoption scripts of 2026-11-13 are **removed** with the mapping (they re-scored with the model of their time and would always report a difference now; recover them from git history, commit `0d5ede7` or earlier);
the evidence of that run stays in `docs/category_results_2026_11_13/legacy_revision_check.json`.

**A partial revision of older code** can be later by time than the one you continue (the second run of the development, `category-981dacf6f68d`, 7,644 items, is entirely inside the first, with the same label in all 10,355 pairs). `--status` with `--summary-revision` and `--companies-revision` reports the revision this code writes, not the latest by time; a consumer that takes "the latest revision" (PLAN D39 (vi)) should read the complete one, and the redundant partial one can be deleted with `RECATEGORY_SQL` once the owner agrees.

**Company names are an input of step 3, not of this revision.** The hypothesis names the company, and the names come from the source store's universe. The companies step hashes that universe in its own revision, so a renamed company is a new companies revision and so a new revision here. A rename in the source store with no run of the companies step again leaves the category rows as they are: run step 3 again to take it in.

**Why this revision is not the companies step's.** The companies step hashes the code it reaches and the version of `transformers`; this step hashes the code that decides a score (`score_of`, `_read_pair`, `company_name`, `_load`) and not the library version, because an edit of the command or an upgrade of the library must not make a stopped run start again. The library version is not in the rows: it is pinned in `uv.lock` (`transformers` 5.16.1 on 2026-11-14); a score that changes with a library upgrade shows as a failing verification (the script above), not as a silent new revision.

## The first run of this model: cost, and what the consumers do

**Cost.** The revision starts from zero over the 159,401 (text, main company) pairs of the first corpus run. The zero-shot run wrote its 159,401 pairs between 2026-11-13 18:48 and 2026-11-14 04:19 (the `processed_at` of `news_category`), about **9.5 hours** on the RTX 4050 with 14 hypotheses per pair; this model reads 12, so **expect about 8 to 10 hours** (an estimate from that run, not a measure of this model). The run is resumable: cut it with `--limit` and continue with the same command.

**Consumers.** `--step snapshot` and `--step period` take the category revision **by name** (`--category-revision`), and a snapshot row keeps the label of the revision that wrote it (`docs/snapshot_steps.md`). So the snapshots already built stay on the 14-label revision, and the per-category counts of a period summary count 14 labels in those and 12 in new ones until the snapshots are rebuilt with the new revision (`--step snapshot ... --category-revision <new> --reset`, which resets the period summary built on them). Read one category revision, never a mix, for a given consumer.

## Restore the weights

The pinned folder is **not in git** (712 MB; `models/` is git-ignored). It is published as a **private** Hub repository, `gamug/deberta-v3-news-category-12` at revision `4485970a` (recorded in `published` of `config/category_step.json` and in `config/published_models.json`; the card says the bar of D24 is not met). To run the step on another machine:

```
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('gamug/deberta-v3-news-category-12', revision='4485970ae653be8844f51dcd67b51326ea48b104', local_dir='models/category_single_all')"   # needs HF_TOKEN (read)
rm models/category_single_all/README.md      # the card is not part of the pinned folder (identity = every top-level file)
uv run cli/news_nlp_steps.py --step category --status
```

The step checks the identity and the manifest of the pin at the first text and names the files that differ; `uv run python scripts/publish_category_model_2026_11_15.py --check` (read-only) checks the repository itself. Training the model again does **not** give the same folder (another identity): the pin would have to change with the new weights and their evidence. Keep a backup of the folder as well.

## The definitions the model reads

The model reads one short definition per label, the ones of the baseline of 2026-11-13 (measured: `docs/category_results_2026_11_13/`). They are shorter than the rules of the
tagger in `docs/category-taxonomy.md` (the "Not this label when" column, owner calls 6 and 15 and so on are not read); the taxonomy is the authority for the human tagger, this table for the
step, and a test keeps this table equal to `DEFINITIONS` and `GROUPS` of the code. Reading the full rules is another revision and has to be measured first (T-187). The fine-tuned candidate selected on 2026-11-15 (12 labels, one tag per row; now the default) and its measures are in `docs/category_training.md`.

| Label | Group | Definition read by the model |
|---|---|---|
| `earnings_performance` | `results_capital` | results, guidance, revenue or profit figures |
| `mergers_acquisitions` | `deals_alliances` | deals, takeovers, divestitures |
| `leadership_governance` | `governance_legal_reputation` | executive appointments and departures, board actions, an executive's insider trade |
| `legal_regulatory` | `governance_legal_reputation` | litigation, regulatory action, sanctions, bankruptcy |
| `product_innovation` | `product_people` | product launches, R&D, technology |
| `capital_shareholder_returns` | `results_capital` | dividends, buybacks, debt and credit actions of the company |
| `labor_human_capital` | `product_people` | layoffs, hiring, workforce and labour relations |
| `market_analyst_sentiment` | `market_view` | analyst ratings and price targets, stock price moves, an investor's bullish or bearish view |
| `partnerships_business_dev` | `deals_alliances` | strategic alliances, joint ventures, distribution deals |
| `reputational_harm` | `governance_legal_reputation` | an event that damages or repairs the company's public standing: apologies, service failures, controversies, boycotts |
| `investor_activity` | `market_view` | a fund or famous investor buys or sells the stock, an activist position, 13F filings, block trades |
| `other` | `other` | an event that fits none of the other topics |

## The model

`category-entailment-single-12` (`config/category_step.json`: the folder `models/category_single_all`, identity `4ac093410757bc50`; **the step refuses a folder whose identity is not this one** and never downloads a model):
the zero-shot model `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (revision `8e7e5af5`) fine-tuned on 1,216 rows, one tag per row, 12 labels (3 epochs, lr 2e-5, seed 0; MLflow experiment `news_nlp_train/category`, registered model
`category-entailment-single-12` version 1; it is not published on the Hub). For each pair the premise is `<company>: <text>` (the first 2,000 characters) and there is one hypothesis per label,
`The news about <company> is about <definition>.`; the best entailment probability gives the label, and the group is the group of that label (option B, owner 2026-11-13: `results_capital`, `market_view`, `deals_alliances`,
`product_people`, `governance_legal_reputation`, `other`, the two folded labels no longer in `market_view` and `deals_alliances`). `config/pipeline_models.json` (`category`) is the zero-shot start model that the v1 stage still reads; the step does not.

## How good it is (read before using the labels)

The bar of D24 is **80%** agreement with the owner's labels on the held-out single rows, and **it is not met**. The figures (label agreement; every one of them in `docs/category_training.md` with the per-class precision, recall and ROC):

| | the model of this step (fine-tune) | the model before (untouched zero-shot) |
|---|---|---|
| held-out, 135 single rows, the owner's earlier tags, scored once | **54.1%** [45.9, 62.3] | 43.7% (43.0% with 14 labels) |
| 300 real sub-news under 1,000 characters, the assistant's tags | **71.3%** [66.0, 76.3] | 65.7% [60.3, 71.0] |
| out of fold, 1,216 rows, tags reviewed after the first results | 87.8% | 75.5% |

| the same 300, a blind LLM judge (`deepseek-chat`; T-209) | 77.0% [72.0, 81.7] | 71.0% [66.0, 76.0] |

The judge agrees on the direction (paired 41 against 23, **nominal** p = 0.033; on the 220 texts outside its first run of 80, 78.2% against 70.9%, p = 0.029). It is LLM output like the assistant's tags, not a second independent reader, and it is not repeatable (2 of 80 labels changed between runs). The pinned model is worse than the untouched one on `market_analyst_sentiment` (89 against 104 of the judge's 115; T-210).

Only the held-out is tagged by the owner and held out, and it is the one that fails. It reads best on short company segments (the texts of the cut) and worst on full articles. `earnings_performance`, `market_analyst_sentiment` and
`mergers_acquisitions` are reliable; **`other` is not** (it gets 3 of 39 right on the real sample), and `investor_activity` and `leadership_governance` have too few rows to measure. The group agrees more than the label does (65.2% held-out).

## The fields, and how they differ from v1

`score` is the entailment probability of `label` (0 to 1; the best of 12 independent probabilities, so it can be low: a text of two companies may have a best score of 0.01). `group_label` is the group of
`label`. **`group_score` is not the `group_score` of the v1 `article_category`** (a softmax over the groups): here it is the share of the entailment mass that the labels of that group hold
(0 to 1), and the group is not decided by a group hypothesis. Do not compare the two columns. `slug_scores` is the JSON of the 12 probabilities; a pair the model read cut at its window of 512
tokens also has `_truncated_chunks` = 1 (read the column with `steps.class_scores`; the `--status` counts them). In the 123,146 items of the first corpus run: 11,476 have 1,000 or more characters, and
0 of 4,000 of them with the longest definition reached the window (measured with the zero-shot model and 14 definitions; the window does not depend on the model).

## Edge cases

A blank text is not classified: the item is `skipped` in `pipeline_step_state` with the reason `blank text`, and is not retried at this
revision. A ticker that is not in the universe is read as its own name (counted as `name_missing` in the run output). The `batch_size`
of `config/pipeline_models.json` (`category`) belongs to the v1 stage and is not read here (nor is its model): the model scores one pair at a time,
12 hypotheses each, and the step stores four items per transaction.

## Metrics

D37 records Precision, Recall and ROC OVR per class. `scripts/category_roc_ovr_2026_11_13.py` gives the ROC OVR from saved scores
(the held-out scoring of the mixed fine-tune, or the out-of-fold scores of a CV run, which `category_train_cv_2026_11_13.py` now saves).
The baseline scripts of the untouched model report accuracy and per-class recall; they do not save scores.

## The record of the runs, and the scripts

The figures above come from `docs/category_results_2026_11_13/` (predictions and scores, no text; provenance in its `README.md`). The scripts of `scripts/category_*_2026_11_13.py`,
`split_headers_2026_11_13.py`, `draw_category_multi_2026_11_13.py` and `translate_category_multi_2026_11_13.py` are research scripts: they end with a file-level `# ruff: noqa` of the nine rules
they break (imports after `sys.path`, one-letter names, lambdas, constants), deliberately; the code of the step (`src/`, `cli/`) has none. The translation cache of `news_nlp.translate` is keyed by
`article_id`, so a tagging set with several rows for one article needs one key per row (the translate script builds `article_id_ticker`); the key itself is T-186.

## Not done

No threshold or review state, by decision (owner, 2026-11-14): every pair gets a label and the `score` is the confidence, so filter on it; an item with no main company stays on the review list of step 3. A score floor would be another revision (and a change of FR-022); an abstention would be another model and revision (T-205 settles `other`). No sector summary (it has no company).
Sentiment (step 6) is a separate step.
