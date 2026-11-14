# Condensed for the snapshot steps (D42, FR-024), 2026-11-14

Verbatim text moved out of SPEC.md, PLAN.md and TASKS.md; cut from commit 8424a31 (master). Task IDs unchanged.

## SPEC.md FR-020 (before the pointer to FR-024)

| **FR-020** | **The pipeline (owner, 2026-11-10, PLAN D39 and D40), built in separate steps, Build 1 first.** **Build 1:** (1) scope of each unique body from a title classifier (`single`, `multi`, `sector`, `out_of_scope`; the last stops) and, for `multi`, the sub-news cut by the `units` extractor; (2) summary of each single body and sub-news over the size (2,000 characters, owner 2026-11-12; shorter ones are used as they are, `text_source = original`; `docs/summary_step.md`); the sector body is summarised in step 2 once its sector is known (D39 (4), T-167); (3) NER matched to universe tickers (spans with no ticker are ignored) and the relevance model: tickers over a threshold are main, one row per main ticker, the rest are mentions (`news_company` rows with `is_main = 0`; the raw spans are not stored, owner 2026-11-13), none over it: the review list. (4) the sector of a `sector` body: a 2-digit GICS sector or `out_of_scope` (`body_sector`; `docs/sector_step.md`). **Build 2:** sentiment (pair model + relevance stack; sector summaries by `gamug/FinBERT-financial-news`), then category. Step 5, snapshots, is undefined. Each step has its own table. | Each step reads the stored result of the previous one and writes its own table; for a fixed input and fixed model revisions each step gives one row per item; a single body or sub-news over the size has a `news_summary` row with `text_source = summary` and one at or under it a row with `original` and its own text, at one `summary-…` revision; the main ticker is stored with its relevance score in `news_company` (`is_main`); a `sector` body has one `body_sector` row per revision: a 2-digit sector or `out_of_scope` (a tie is not agreement); the D24 bar is open (T-167); the scope classifier meets the T-147 bar on the held-out 200 before step 1 is accepted. |

## PLAN.md D28 (replaced by D42)

- **D28 Snapshots (undefined, not started: owner 2026-11-10, D40).** New tables in RESULTS, the old ones kept as v1 (D19). A company snapshot
  is one row per (text, main company), where the text is a summary or a sub-news (D39 (i)): text, ticker, D15 label, sentiment, `pub_date`, `body_hash`; a sector
  snapshot is one row per (body, sector): text and sentiment. This repository does not aggregate
  per day: the per-(asset, day) signal of D14 is built downstream from the per-body rows, which
  also keeps one count per unique body for the backtest (a company counts a body once, whatever its number of sub-news). Sketch only: the schema is proposed to
  the owner in the P5 PR (constitution AI #10).

## TASKS.md T-152 (replaced by T-195, T-196, T-197)

- [ ] **T-152** *(undefined, not started: owner 2026-11-10, D40)* *(P5)* The company snapshot table (**a row only for a (text, main company) pair: an item with no main company, the review list of step 3, gets no company snapshot; owner, 2026-11-14, in place of T-161 and T-147b**; one row per (text, main company), text = summary or sub-news, D39 (i); counted once per body for the per-day signal: text, ticker, D15
      label, sentiment, `pub_date`) and the sector snapshot table (one row per (body, sector): text,
      sentiment), RESULTS, the old tables kept as v1 (D28, D19). **Columns approved by the owner on
      2026-11-06** (for the cut's files and as the base of the table): `body_hash`, `article_id` (first),
      `pub_date` (as stored), `scope` (rules), `ticker`, `role` (analysed | mention, D23), `text_kind`
      (summary | company_text | unextracted), `text_source` (summary | parser | baseline | model), `text`,
      `sentiment`, `p_positive`, `p_negative`, `p_neutral`, `sentiment_checked` (yes | no), the
      summary, extractor and sentiment model with their revisions, `provisional` (1), `run_id`. The types,
      the keys and the migration are proposed to the owner in the P5 PR (constitution AI #10); its own FR. → P5.
- **T-153** *(out of the MVP, owner 2026-11-10; text in `archive/PIPELINE-superseded-2026-11-10.md`)*
- **T-156** *(out of the MVP, owner 2026-11-10; text in `archive/PIPELINE-superseded-2026-11-10.md`)*
- **T-157** *(out of the MVP, owner 2026-11-10; text in `archive/PIPELINE-superseded-2026-11-10.md`)*

## PLAN.md D39 (iv), superseded by D42 (owner, 2026-11-14)

(iv) **Sector summary:** it gets a sentiment, from
  `gamug/FinBERT-financial-news` (the pinned chunk-level stage, no company).

Old FR-020 clause: "Build 2: sentiment (pair model + relevance stack; sector summaries by `gamug/FinBERT-financial-news`), then category. Step 5, snapshots, is undefined."

## SPEC.md FR-018, last sentences (pins of other steps)

 The scope model of step 1 is pinned apart, in `config/scope_model.json`, and the attribution model of the cut in `config/attribution_model.json` (private Hub copy, identity checked, `--fetch`/`--allow-unpinned`). The sector embedder is pinned in `config/sector_step.json` and the sentiment pair model (a local folder) in `config/sentiment_step.json`; the category step reads the `category` pin of this file (the same checkpoint and revision as `pipeline.MODEL_REVISIONS`, regression-tested). |

## PLAN.md D40 amendment of 2026-11-11 (full text)

- **D40 amendment (owner, 2026-11-11): scope classifier selected, pinned, accepted under the bar.** The model is `scope-nli-8ff21e50796b`: two fine-tuned NLI models (365 tags), public on the Hub at `gamug/deberta-v3-news-scope-nli@58e4af4b` (card-only commit of 2026-11-11 on `@d14a4d20`, same weights) and pinned in `config/scope_model.json`; step 1 runs on it. Held-out 200, weighted, scored from the saved weights: **83.6% [78.1, 88.9]** (rules 67.2%; paired gain +16.5 [9.6, 23.4]); `single` recall 92%, **`out_of_scope` recall 53% [32, 74]**. **Bar of D20/D33: 80% since 2026-11-14 (owner); it passes.** (Earlier fit of the same recipe: 83.0% [77.5, 88.3], 48%; 81.5% on the key as first tagged.) **The limit is the `out_of_scope` class:** about half of those bodies continue as `single` or `multi`; accepted on purpose (a body wrongly kept costs a review, one wrongly stopped is lost). **Validated on `quarantined` bodies (owner, 2026-11-11, `docs/scope_training.md` §9; tags in `gold/scope_quarantined_validation.csv`):** 200 bodies drawn per predicted scope, final key 90.8% [86.9, 94.7] on 225 bodies (first reading 90.6%; the 200 of the first draw alone: 90.1% and 89.6%); `out_of_scope` precision 93.0% [86.3, 96.6], `single` 90%, `sector` 88%, `multi` 82% on 50. **The gate on the cut's `multi` rows (docs/scope_training.md §8.1) was released by the owner on 2026-11-11, no flag.** Precision only. **The scope validation is closed (owner, 2026-11-11, `docs/scope_training.md` §9.1): the model stays as it is;** the recall of `out_of_scope` among quarantined bodies and the audit of the gold under calls 17, 18 and 21 are not done (closed by decision); the 90.8% and the 83.6% are not compared like for like. **Reopens if** the bar is demanded, or the `out_of_scope` share that continues costs more than the review list takes (6.7% of the bodies was estimated from the 1,000-body sample; over the corpus the model calls 49% `out_of_scope`, 11% of the `ok` and 59% of the `quarantined` bodies, so the share that continues is not known: `docs/scope_training.md` §8). The three-seed check stays open (T-147b), **deferred by the owner on 2026-11-11 (to be fitted later; no time to be picky)**. Story and limits: `docs/scope_training.md`.
