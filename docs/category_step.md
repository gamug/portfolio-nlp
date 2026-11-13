# The category step (`--step category`, T-168; provisional)

One of the 14 labels of D15 (`docs/category-taxonomy.md`) and its group, for each **main company** of each text. It is built
**provisionally with the pinned zero-shot model** so the next team can read a category; the trained model of T-174 replaces it
by changing the pin (it is then another revision of the step; the rows of this one stay).

## Run it

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

## Which revision to read

Revisions coexist in `news_category`: the revision hashes the code of the step, the pinned model, the definitions and the version of `transformers`, so any of them changing starts
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

Read the newest revision that is complete (the first row of the first query that reaches the number of the second, less `items_without_text`); delete an old or partial one with `RECATEGORY_SQL` (`--help`). A run stopped by a GPU fault leaves a partial revision: rerun the same command with
the same code (same revision) and it completes it. A change of the code between two runs is another revision and starts again.

## The definitions the model reads

The model reads one short definition per label, the ones of the baseline of T-174 (measured: `docs/category_results_2026_11_13/`). They are shorter than the rules of the
tagger in `docs/category-taxonomy.md` (the "Not this label when" column, owner calls 6 and 15 and so on are not read); the taxonomy is the authority for the human tagger, this table for the
step, and a test keeps this table equal to `DEFINITIONS` and `GROUPS` of the code. Reading the full rules is another revision and has to be measured first (T-187).

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
| `competitive_actions` | `deals_alliances` | the company's own move or statement against or about a competitor: share fights, pricing pressure |
| `investor_activity` | `market_view` | a fund or famous investor buys or sells the stock, an activist position, 13F filings, block trades |
| `peer_comparison` | `market_view` | a comparison between companies: which is the better stock, who is ahead |
| `other` | `other` | an event that fits none of the other topics |

## The model (provisional)

`MoritzLaurer/deberta-v3-base-zeroshot-v2.0` at the pin of `config/pipeline_models.json` (`category`), untouched. For each pair the
premise is `<company>: <text>` (the first 2,000 characters) and there is one hypothesis per label,
`The news about <company> is about <definition>.`; the best entailment probability gives the label, and the group is the group of that
label (option B, owner 2026-11-13: `results_capital`, `market_view`, `deals_alliances`, `product_people`,
`governance_legal_reputation`, `other`).

## How good it is (read before using the labels)

Agreement with the owner's labels (`scripts/category_baseline_2026_11_13.py`; held-out scored once):

| | Train rows (225) | Held-out single rows (135) |
|---|---|---|
| label | 56.4% | 43.0% |
| group | 62.7% | 55.6% |

The bar of D24 is **80%**: the labels are a first version, not the accepted ones. They are reliable for `earnings_performance`
(recall about 95%) and weak for the rare labels (`capital_shareholder_returns`, `competitive_actions`, `peer_comparison`, `other`
are almost never found). The held-out 200 also has 44 narrative multi bodies that are not scored. The group is the better field to rely on.

A fine-tune of the same model (`scripts/category_train_cv_2026_11_13.py`, the mixed recipe) reached 62.2% out-of-fold on the train rows
and 48.9% on the held-out single rows: better than the untouched model, still under the bar; it is not published and the step does not use it.

## The fields, and how they differ from v1

`score` is the entailment probability of `label` (0 to 1; the best of 14 independent probabilities, so it can be low: a text of two companies may have a best score of 0.01). `group_label` is the group of
`label`. **`group_score` is not the `group_score` of the v1 `article_category`** (a softmax over the groups): here it is the share of the entailment mass that the labels of that group hold
(0 to 1), and the group is not decided by a group hypothesis. Do not compare the two columns. `slug_scores` is the JSON of the 14 probabilities; a pair the model read cut at its window of 512
tokens also has `_truncated_chunks` = 1 (read the column with `steps.class_scores`; the `--status` counts them). In the 123,146 items of the first corpus run: 11,476 have 1,000 or more characters, and
0 of 4,000 of them with the longest definition reached the window.

## Edge cases

A blank text is not classified: the item is `skipped` in `pipeline_step_state` with the reason `blank text`, and is not retried at this
revision. A ticker that is not in the universe is read as its own name (counted as `name_missing` in the run output). The `batch_size`
of `config/pipeline_models.json` (`category`) belongs to the v1 stage and is not read here: the model scores one pair at a time,
14 hypotheses each, and the step stores four items per transaction.

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

No threshold or review state: every pair gets a label (the owner decides what a low score means). No sector summary (it has no company).
Sentiment (step 6) is a separate step.
