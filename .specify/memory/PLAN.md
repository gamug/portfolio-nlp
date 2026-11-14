# PLAN.md — `portfolio-nlp`

The implementation plan for the live backlog identified in `.specify/memory/SPEC.md`: where the constitution is principles and `SPEC.md` the requirements/architecture contract, this document is the "how, and in what order". Work items are numbered in the order they were filed; closed ones keep their heading here with a one-line status and their plan in `archive/` (see `INDEX.md`).

## Goal

Keep the pipeline's four stages measured and pinned, and ship the triage-first pipeline of Work item 18: a scope per unique article body, a text, sentiment and category per (body, company), and a provisional backtest cut (D34). Work items 1–14 and 16 delivered the pinned checkpoints, the evaluation redesign, the experiment runner and the model-selection config; their goals and the original scope notes are in the archive ("Preamble and Goal").

## Non-goals

Everything else in `SPEC.md` §13 stays exactly as §14 disposed of it —
**not** part of this plan:

- Item 3 — no SOURCE schema contract beyond `body_text`: accepted;
  `SPEC.md` §5's schema-contract table documents the (unenforced)
  dependency, which is sufficient at this scope.
- Item 4's other half — reprocessing legacy unpinned-checkpoint results:
  not planned; pinning going forward is enough (see Work item 1).
- Item 5 — throughput/latency SLA: retired, not applicable.
- Item 6 — `article_category` migration not auto-reprocessing: accepted.
- Item 7 — no scheduled `--summarize` cadence: accepted.
- Item 9 — no per-article failure isolation: accepted.

## Work item 1 — Pin HF model checkpoints to a commit SHA (resolved 2026-09-14)

Closed. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 1"; the tasks are in `CHANGELOG.md`.

## Work item 2 — Make the `--check-regression` gate runnable (out of the MVP, owner 2026-11-10)

Text: `archive/PIPELINE-superseded-2026-11-10.md`.

## Work item 3 — NER: validate the subword-fragmentation fix (resolved 2026-09-12)

Resolved. Its only open task, T-022, is in `TASKS.md`. The acceptance criteria of this item are now only in the archive. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 3".

## Work item 4 — Sentiment: close the entity/net-signal reasoning gap (resolved 2026-09-13: fine-tuned + chunk-level selected and merged)

Resolved. Its only open task, T-032, is in `TASKS.md`. The acceptance criteria of this item are now only in the archive. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 4".

## Work item 5 — Category: hold the line on the hierarchical fix (closed 2026-10-14)

Closed. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 5"; the tasks are in `CHANGELOG.md`.

## Work item 6 — Summarization (`c_summary` + `sector_summary`): validate eval scope, close the coverage gap, and add a lightweight sector-intro check (resolved 2026-09-14 -- new sector_summary gap found and fixed same day: up to 50.2% intro_text hallucination rate)

Closed. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 6"; the tasks are in `CHANGELOG.md`.

## Work item 7 — NER: develop batch processing (code done 2026-09-12; T-062 needs a real GPU)

Resolved. Its only open task, T-062, is in `TASKS.md`. The acceptance criteria of this item are now only in the archive. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 7".

## Work item 8 — Justify each model's selection in the Models evaluation artifact section (done 2026-09-19)

Closed. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 8"; the tasks are in `CHANGELOG.md`.

## Work item 9 — Rebalance the sentiment fine-tuning data (done 2026-09-19)

Closed. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 9"; the tasks are in `CHANGELOG.md`.

## Work item 10 — Formalize the pipeline/evaluation architecture: FTI restructure + evaluation redesign (done 2026-09-18)

Closed. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 10"; the tasks are in `CHANGELOG.md`.

## Work item 11 — JSON-driven, single-command experiment runs (done 2026-09-18)

Closed. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 11"; the tasks are in `CHANGELOG.md`.

## Work item 12 — Bring `tests/` under the mypy gate (done 2026-09-18)

Closed. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 12"; the tasks are in `CHANGELOG.md`.

## Work item 13 — Tighten judge-verdict reuse to also require a matching prediction (done 2026-09-19)

Closed. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 13"; the tasks are in `CHANGELOG.md`.

## Work item 14 — Move production model selection into a git-tracked JSON config (done 2026-09-21)

Closed. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 14"; the tasks are in `CHANGELOG.md`.

## Work item 15 — Category: fix the `other` bucket (calibrate it, and clean up what the tagging fix removes)

**Re-scoped 2026-10-28 into Work item 18; its own plan is superseded.** What stays useful (body-hash grouping, name/ticker resolution, sentence attribution, the cleanup of the results of quarantined pairs: T-131, T-132) continues there. Decisions D1–D10 (relevance gate, `review` pairs, verdict vocabulary, read-only attach, defaults of 2026-10-22) and the original design: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 15".

## Work item 16 — Check the data-mining schema version before reading (closed 2026-10-22, #109; D8 default: build it here)

Closed. Plan, steps and acceptance criteria: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, section "Work item 16"; the tasks are in `CHANGELOG.md`.

## Work item 17 — Articles that name more than one company (decided 2026-10-22: its own work item, sizing first, D9 default)

**Re-scoped 2026-10-28 into Work item 18.** What stays useful here (body-hash grouping,
name/ticker resolution, sentence attribution, the cleanup of removed rows) continues there.

**Why** (hand-off note 2026-10-21, reported, not reproduced): the data is per pair, one
`articles` row per `(url, ticker)` with the same body stored once per pair. Measured
upstream on `accept`/`review` pairs: 22,048 of 132,078 URLs (16.7%) have pairs for more
than one ticker, holding 51,991 of 162,021 pairs (32.1%) and **35,387 of the 98,462
`ok` articles (35.9%)**; 16.7% is a floor (the matcher reads the URL and title for 503
tickers, never the body), and some are share-class twins, not multi-company articles
(deduped by body hash). In this repo every stage keys on `article_id`, i.e. on the pair,
and only sentiment scopes to the subject, so a roundup's one category, entity list and
summary are read as each tagged company's. 10,493 of the 52,039 `review` pairs (20.2%) sit on a URL
that also has an `accept` pair; releasing one because its sibling is `accept` is
explicitly not allowed (a pair is released on its own evidence).

**Approach** (upstream's suggested pipeline, accepted or changed by the owner): sizing
first, building second. Stages numbered as in upstream's hand-off (2026-10-21, 9.2 and
9.4): (0) group by body hash; (1) mention detection by NER resolved
to the universe with data-mining's alias table; (2) type the article (single-subject,
multi-subject, comparison, transaction, incidental), rules first. Run these three on a
stratified sample of about 2,000 groups (all-`accept` multi-pair, `accept` + `review`,
all-`review`, single-pair control) to replace the 16.7% floor with a measured number of
companies per article. Only then decide how much of roles, sentence attribution and
per-company facts (sentiment on attributed spans, numbers bound to the company) to
build, and which fact types the knowledge graph needs first. Rule: **an article-level
score on a multi-subject group carries a `multi` flag and is never read as one company's
score.** It takes effect when T-138 ships; until then it is false today (category, entity
and summary rows are read as one company's), and T-137's report, `docs/evaluation.md` and
the architecture artifacts are the only statement of the risk. The flag is a schema
change (upstream sketches `article_group`, `article_company`, `article_fact` tables): D9
approved a work item and a sizing run, not a column, so the schema is asked of the owner
in T-138's PR (constitution AI behavior #10). Shares the body-hash grouping and the ORG-to-ticker
resolution with T-132, so it is sequenced after it; it is separate work (constitution
Code & Git #9), and absorbs T-127's sentiment finding when that is measured.

**Acceptance**: the sampled groups report the real companies-per-article distribution
per stratum, per group and article-weighted (T-137 fixes the frame); a hand sample (about 20 groups per type, seeded with upstream's 42 News
Corp multi-company URLs) measures attribution precision, fact binding and role
agreement per type, never one average, each with a Wilson interval; thresholds to trust
a fact are a joint decision. The sizing is a committed read-only script plus a dated
follow-up in `docs/evaluation.md` (T-137); the 20-per-type hand sample is a smoke test,
not a measurement, unless the owner sets a larger N. Close-out: T-138b (SPEC item 21, both
artifacts, `CHANGELOG.md`); it closes after T-137's report if the owner stops there.

## Work item 18 — Triage-first pipeline: unique bodies, subject scope, per-company and sector news (re-scoped 2026-10-28)

**Why.** Work item 15 tried to shrink the `other` bucket with a better threshold. T-121(b) stage 1
(eval_run 45, 720 calls) showed that no threshold-and-margin rule can: the articles each candidate
would move out of `other` are 29–33% truly `other`, and the scores do not separate them. The owner
then hand-tagged the same 720 articles (`data/category_gold_review.csv`, git-ignored):

- the LLM labels agree with the owner on 46% of them (58% with `other`, multi-company round-ups and
  non-financial articles merged), so they are silver, not gold;
- 94 are multi-company round-ups ("market weekly update"), the content the owner values most; the
  taxonomy can only call them `other`;
- the notes ask for reputational harm, competitive actions, sector news, whole-market and economy
  news, comparisons and investor-group moves; 31 are not about the economy at all.

The unit is the cause: an `articles` row is a (URL, ticker) pair, the same body is stored once per
tagged ticker (35.9% of `ok` articles share a body), and every stage scores the pair as if the
article were about that ticker. Round-ups, sector and macro news have no correct label in that model.

**Decisions** D11–D24 (first recorded in T-143, 2026-10-28, amended 2026-10-29 with the owner; T-143's original text is in `CHANGELOG.md`; this paragraph is the current wording and the one source; D25–D30 follow in the next section, D31–D35 after them):
D11 the unit is the unique article body (body hash), its subjects found in the content; D12 LLMs
only label and evaluate, the pipeline keeps local pinned models (constitution AI #1/#2); D13 GICS
sector + sub-industry; D14 the knowledge graph keeps per-(asset, day) signals, sector news feeds the
weekly `sector_summary`, market and economy news are out of the MVP (D39); D15 the
per-(body, company) taxonomy is the 9 labels + `reputational_harm`, `competitive_actions`,
`investor_activity`, `peer_comparison`, with `other` (company content that fits none of the 13
labels; settled 2026-10-29: the investing news about no company that D15 first named is the scope
`investing_general`, D22, and carries no category), `uncertain` (no confident category: a
category-level label only, the scope never abstains, D20) and `unrelated` (not about the economy); D16 `unrelated` bodies are
flagged and skipped by every later stage; D17 a multi-company body gets sentiment + category per
company; D18 (silver labelling with LLM calls: out of the MVP, owner 2026-11-10); D19 full backfill of the **81,961 unique `ok` bodies** (not the
459,112 stored rows): the stored rows are moved aside as v1 until the new tables are validated, and
the stored results of the quarantined pairs are moved aside, reversibly, not destroyed (T-131; owner 2026-11-09, D36 (h1), D36 (b): the first wording said "purged"; 377,010 are SOURCE rows, the RESULTS has 365,782 `reject` and `review` rows); D20 the scope triage is accepted
only at >= 90% (80% now) corpus-weighted agreement with the owner's held-out gold on all bodies (no abstention) and
recall >= 80% for each of the six scopes (including a random sample of ~60 bodies the prefilter misses, Q9), both judged on the point estimate with the 95% interval reported
(at 40 bodies the per-scope interval is about +/-13 points; D33 amends: a scope with fewer than 40 gold bodies is reported with its interval, not gated); D21 each
training round sends a stratified sample of about 300 low-confidence bodies to the owner for tagging
(active learning); D22 six scopes (the MVP handles single, multi and sector; market / economy, `investing_general` and `unrelated` stop after the scope, D39); D23 a
body where one company holds more than half of the subject mentions is *single + mentions*: scope
single on the dominant company, the others stored as mentions only; (the gold v2 defines analysed vs mention by content, a sentence of its own, `docs/labelling-spec-v2.md`; the >50% count is the automatic rules' feature) D24 P4 (per-company category) and
P5 (sector sub-industry) are each accepted at >= 80% agreement with the owner's held-out gold (the
owner confirmed on 2026-10-29).
Laya (`convaiinnovations/laya`, a local non-generative typed-decision model) was tried as a
yes/no checker of the triage questions and dropped on 2026-10-29: zero-shot it said "yes" to almost
everything (5.4% scope agreement with the owner, against 70% for the rules on six scopes; T-145).

**The 7-step pipeline and the owner's calls of 2026-11-05 (D25–D30).** After tagging the
gate the owner fixed the order the stages run in, and the backtest team (`portfolio-financial-analysis`,
`to_nlp_backtest.md`) asked what the final tables will hold. Where a call below changes a decision
above, it says so; nothing else above is touched.

- **D25 Order of the stages: replaced by D39** (the old text: `archive/PIPELINE-superseded-2026-11-10.md`).
- **D39 The pipeline (owner, 2026-11-10; replaces D25; amends D22, D27; D28 became D42).** The steps, in the owner's numbering:
  (1) **Scope** of the body. The MVP works on `single`, `multi` and `sector` only; `market_economy` and
  `investing_general` are **out of scope** (the scope is stored, no later step runs) and `unrelated` is out by
  definition (D16). (2) **single**: a summary (`c_summary`); NER on the summary; the relevance model scores (summary,
  company) for each company NER found, and the **main company** is the one the article mainly speaks about; the other
  companies of the summary go to a separate place (mentions). (3) **multi**: the extractor (`units`, D27) cuts the body
  into **sub-news**, company-agnostic; (3.1) NER on each sub-news, and the matcher over its text (`--candidates ner+text`, owner 2026-11-13); (3.2) the relevance model picks the main company or
  companies of each; (3.3) a sub-news over a character size is summarised (2,000 characters, owner 2026-11-12: T-166). (4) **sector**:
  a 2-digit GICS sector or `out_of_scope` is given (embedder, company votes and a gate: `docs/sector_step.md`); its bodies are summarised in step 2 (`--items sector`). (5) market: out of scope. (6) **Sentiment**: the pair
  model + the relevance stack (`docs/sentiment_training.md`) reads each summary or sub-news with its main company.
  (7) **Category** (D29) of each summary or sub-news. After the seven, a separate step writes the snapshots (D42, T-195).
  **Conditions on the development:** (a) the advance of a run is **tracked** (per step and per item: pending, done,
  failed; counts, times, a status command); (b) every step is **idempotent** (it processes only the items without its own
  result row for the same model revisions; a rerun adds and changes nothing); (c) the process is **broken into small
  steps**: one command per step, with a limit, each reading the stored result of the previous one; steps 1 to 7 are never
  run in one pass, so there is no huge queue. **Effects on earlier decisions:** D22 stays as the triage's labels, the MVP
  handles three of them; D27's attribution model (which company a sentence is about) is replaced in the MVP by NER +
  relevance model (3.1, 3.2), the `units` cut and its faithfulness check stay as the cut into sub-news; D42's snapshots
  are the step after 7; D26 and D29 stand, with the relevance model in the sentiment path. **Decided by the owner, 2026-11-10:**
  (i) **Main company:** every company over a relevance threshold is main (the threshold is set on the out-of-fold
  scores of the 809 rows, T-169); the item is **stored once per main company** (a news with two main companies has two
  rows, same text, one per company), so each row has a single main company and step 6 reads one (text, company) pair per
  row. (ii) **No company found, or no score over the threshold:** the item goes to a **review list** for the owner
  (no sentiment until it is read; T-172 builds the list, status `review`). (iii) **A company is a universe ticker.** A NER span is matched to the universe by the matcher of `news_nlp.triage` (D38); a span that matches no ticker is **ignored** (not stored as a mention, never main), and every comparison in this feature (the relevance score, `is_main`, the crawl's ticker against the model's pick) is by ticker only. **Mentions** (the companies that are not main) are `news_company` rows with `is_main = 0`; the raw NER spans are not stored (owner, 2026-11-13: `article_entities` is keyed by article and the v1 NER stage reads it as "done"); the **result per (text, ticker)** (relevance score, `is_main`, mention count) is its own table `news_company` (owner, 2026-11-10, interactive session: a ticker named three times is three span rows, so the pair needs its own row). (iv) **Sector summary:** no sentiment and no category (superseded by D42, owner 2026-11-14; the old clause is in the archive file of D42). (v) **Tracking:** one table
  `pipeline_step_state` (item, kind, step, status `pending/done/failed/skipped/review`, model revision, attempts, error,
  times), key (item, step, revision); `--status` is a query on it. (vi) **A new model revision** adds a new result row and
  the latest revision is the one read downstream; old rows stay. (vii) **CLI:** a new `cli/news_nlp_steps.py` with
  `--step`, `--limit`, `--status`; the v1 `cli/news_nlp_cli.py` is not touched until the backfill (T-150). (viii)
  **Sub-news size (3.3):** measured first (character lengths of the units output on the 438 confirmation texts), then
  the owner sets it (T-166: 2,000 characters, 2026-11-12, `docs/summary_step.md`). **The crawl's tickers** (D36) are kept as a column of the body's result and compared with the model's main tickers by ticker; they do not decide the main company (proposed, to the owner). **Open:** the relevance threshold value (T-169); how the review list is stored; whether
  the relevance model, trained on summaries, holds on sub-news (not measured: a risk of T-165, T-169). Tasks: T-162
  to T-172.
- **D40 Build order, steps and tables (owner, 2026-11-10, after D39; amends D39's order of work, not its steps).**
  **Build 1** (first, with a status command and a dry run): (1) the extractor step gives the scope of every body (single,
  multi, sector, or out of scope) and splits the multi ones into sub-news; (2) the summary of each single body and sub-news
  that fits the size rule (the sector body is summarised in step 2 as `--items sector` once its sector is identified, D39 (4), T-167: owner, 2026-11-12, changed 2026-11-13; the earlier wording is in `archive/SPEC-PLAN-sector-summary-own-step-2026-11-13.md`); (3) NER and the relevance model give the mentions and the main tickers (D39
  (iii)). **Build 2** (after Build 1 runs): sentiment (between 3 and 4, D39 (6)), then category (D39 (7)); the owner trains
  the category model while Build 1 runs. **The snapshots (step 8) and the period summary (step 9): D42.** **Size rule (step 2):** an item over the size (2,000 characters, owner 2026-11-12, T-166) is summarised; a shorter one is used as it is and its row says `text_source = original`; steps 3 to 4
  read the row's text, whichever the source. **The scope comes from a classifier that reads the title** (owner,
  2026-11-10; T-147; trained on `gold/category_gold_v2.csv`, bar on the held-out 200; the rules triage, 68.0% on the same
  200, is the baseline); the `units` extractor cuts the `multi` bodies in the same step 1. **Tables (the owner: each step its own table, sentiment included; names proposed, schema to the owner
  first, AI #10):** step 1 `body_scope` (unique body, scope, rule revision) and `sub_news`; step 2 `news_summary`
  (item kind body or sub_news, text, `text_source`, model revision; `article_summary` is keyed by article and cannot
  hold these); step 3 `news_company` (one row per item and ticker: `relevance_score`, `is_main`, `mention_count`; **`news_company.is_main` is the only source of truth for the main ticker**; the raw spans are not stored, `mention_count` keeps the count);
  sentiment `news_sentiment` and category `news_category`, both per (text, main ticker, read from `news_company.is_main`). **Reads and writes:** step 1
  reads unique bodies of SOURCE; step 2 reads `body_scope` and `sub_news`; step 3 reads `news_summary`; each stops at
  `--limit`. **Decided by the owner, 2026-11-10 (interactive session):** the shared `pipeline_step_state` of D39 (v) **stays**
  beside the per-step tables, and **the schema of this paragraph and of D39 is approved as written** (AI #10); the size of
  the rule is measured on the output of step 1 and then set by the owner (T-166); the scope classifier reads the title
  **and the first lines of the body** (T-147).
- **D40 amendment (owner, 2026-11-11): scope classifier selected, pinned, accepted under the bar.** The model is `scope-nli-8ff21e50796b`: two fine-tuned NLI models (365 tags), public on the Hub at `gamug/deberta-v3-news-scope-nli@58e4af4b` (card-only commit of 2026-11-11 on `@d14a4d20`, same weights) and pinned in `config/scope_model.json`; step 1 runs on it. Held-out 200, weighted, scored from the saved weights: **83.6% [78.1, 88.9]** (rules 67.2%; paired gain +16.5 [9.6, 23.4]); `single` recall 92%, **`out_of_scope` recall 53% [32, 74]**. **Bar of D20/D33: 80% since 2026-11-14 (owner); it passes.** (Earlier fit of the same recipe: 83.0% [77.5, 88.3], 48%; 81.5% on the key as first tagged.) **The limit is the `out_of_scope` class:** about half of those bodies continue as `single` or `multi`; accepted on purpose (a body wrongly kept costs a review, one wrongly stopped is lost). Validation on quarantined bodies, the release of the gate on the cut's `multi` rows and the closing of the scope validation (owner, 2026-11-11): the text of this entry is in `archive/SPEC-PLAN-TASKS-condensed-for-the-snapshot-steps-2026-11-14.md`; story and limits: `docs/scope_training.md` §8 to §9.1. The three-seed check stays open (T-147b), deferred by the owner on 2026-11-11.
- **D39 (iii), step 3 built (2026-11-13; T-164, T-165; `docs/companies_step.md`).** `--step companies` reads the text of step 2 for single bodies and sub-news together: NER, `CompanyResolver.ticker_of_name`, the relevance model per (text, ticker), `news_company` rows, main at the threshold or over it, `review` otherwise (not retried at the same revision). **Owner, 2026-11-13:** `--candidates ner+text` chosen on the figures of its first definition (the matcher unfiltered: candidate 99.5%, main 75.0% of sub-news), `--threshold 0.5` (revisited with T-170: the share of sub-news with a main company runs from 65.5% at 0.8 to 84.0% at 0.1, single bodies from 60.5% to 65.5%), raw spans not stored. **The definition was corrected in the review of #178** (lower-case matches, disclosures and symbol lists left out; sub-news candidate 93.5%, main 73.0%) **and the owner confirmed it on 2026-11-13, accepting the filter's limits (T-185).** Both choices have no default and are part of the revision (D39 (vi)). The relevance model never saw a sub-news (T-170).
- **D39 (4), the sector step built (2026-11-13; T-167, #179; `docs/sector_step.md`).** Decided by the owner in the session of 2026-11-12/13: the sector is the **2-digit GICS sector** (11 classes; the tags of the gold were rolled up to it, eight corrected); it is checked by yes/no review, not by choosing among classes; what does not pass the gate goes to `out_of_scope`. Built: a small embedder (`BAAI/bge-small-en-v1.5`, MIT, `config/sector_step.json`) scores chunks against sector descriptions and four *not a sector* classes, the companies the body names vote, and a sector is assigned if the margin is >= 0.0293 or the votes agree (a tie at the top of the vote, or no vote, is not agreement; two votes were measured and not asked for: owner 2026-11-13); table `body_sector` (columns approved by the owner, 2026-11-13; the count of chunks cut at the embedder's window is kept in its `scores`); the summary of the assigned bodies is `--step summary --items sector --sector-revision REV`, not a step of its own. **What was chosen how:** descriptions on the tune half; the reject classes from the first review's notes; the threshold on the tune half (tagged and reviewed rows); the gate after both halves were printed; larger embedders and hand-tuned descriptions did not generalise. **Measure:** the clean review, 35 of 40 assigned right (87.5%, 74-95%) as drawn and 35 of 39 (89.7%) under the corrected tie rule, 0 of 5 controls, 7 of 12 drops agreed; 56.0% of the sector bodies get a sector; not the D24 measure (>= 80% against the held-out gold, with the sub-industry), which stays open in T-167.
- **D41 (owner, 2026-11-13; amends D40's order).** The category step (D39 (7)) is built ahead of D40's order, provisionally: zero-shot model, 14 labels, 6 groups, FR-022; T-174's model replaces it by changing the pin. Revision: only a new input revision or a changed score restarts it, the first run is continued; D39 (vi) reads the newest *complete* one. Open: T-187, T-189.
- **D42 The snapshot (step 8) and the period summary (step 9), Build 3 (owner, 2026-11-14; replaces D28).** Design, tables and the owner's answers: `docs/snapshot_steps.md`. **Snapshot:** one row per (text, main company), or per `sector` text (crawl date, title, text, mentions (out of scope for now, T-198), category, sentiment; a sector text has neither, superseding D39 (iv)); complete rows only; daily after the 15:00 batch. **Period summary:** every 21 days from 2022-01-03 at a fixed 15:00 in `BATCH_TIMEZONE` (default `America/New_York`), window `(previous cut, this cut]` by crawl time; company counts per category and sentiment, and the same counts per sector summed from the company snapshots through the new `sector` and `company_sector` tables (no LLM; sector texts not counted); no-news ones dropped; a written period is frozen and `period_run`/`period_item` say which period was written and counted what, so a late item is counted in the next; `--catch-up`; state derived from the tables (exception to D39 (v)); the snapshot is always idempotent: its revision is the code's, the input revisions are in each row, `--reset` rebuilds. D14's per-day signal stays downstream.
- **D26 Sentiment reads the summary or the company text, not the body.** The pinned sentiment
  model was fine-tuned on sentences, so it is checked on the pipeline's own texts before it is
  trusted there: a sentiment gold of **180 texts, 60 per class** (positive, negative, neutral;
  at that size an observed 90% per class has a 95% lower bound of about 80%), tagged on real
  output of steps 3-4 after step 4 exists, the sample over-sampling the rare negative class by the
  current model's prediction (never shown to the owner) and re-weighted afterwards. Of the 180, 120 are
  summaries and the company texts of structured multi bodies and **60 (20 per class) are the texts the
  extractor writes for narrative multi bodies** (owner, 2026-11-06), reported apart, because those are
  the per-company texts of about 80% of multi bodies. The label is the
  **price impact on the company** an investor would read (for a sector or market text, on that
  sector or market). The model reads a text the way the pipeline's sentiment stage does: sentence-packed
  chunks of at most 510 tokens, their probabilities averaged with equal weights (the text is already
  the company's, so the v1 subject weighting is not used). The model is fine-tuned on summaries only if it misses the D32 bar.
- **D27 Step 4 without an LLM.** (a) A body written as one segment per company is split by
  `news_nlp.company_texts` (a segment leads with a company when exactly one universe company is
  named in its first 50 characters): 2,867 of the 81,961 unique bodies (3.5%) and 21,941
  company-led segments. (b) The other multi bodies (about 80% of the owner's tagged multi bodies,
  not "a few": of 77 tagged, about 15 are structured, about 14 are one paragraph and about 48 are
  narrative) go
  to the **`units` extractor** (the cut into sub-news, company-agnostic, D39); the **trained attribution model** (a
  small local encoder trained on (a) with the company name masked and the previous sentence as context) is not in the
  MVP, NER and the relevance model pick the companies (D39). The extractor is compared with a mention-and-recency baseline and with a faithfulness check the
  owner reads (about 30 narrative multi bodies, the extracted text per company marked faithful or
  not). Training data: **all years** (owner, 2026-11-05; D31). **D12 stands**: the
  owner first chose to amend D12 for an LLM exception in step 4, then replaced it by this model
  after the narrative share was measured; the amendment is written only if (b) fails its check.
  The training set is `gold/company_segments_train.csv` (ids and positions only; the repository is
  public): 2,834 bodies and 33,624 segments, 21,682 clean `lead` and 11,942 noisy `continuation`
  labels, the 33 bodies shared with tagged or held-out articles left out; a paragraph that names two
  or more companies at its start is `ambiguous` and has no row, nor do the paragraphs after it until
  the next lead. 77% of the bodies are from before 2022 and `val` (1 body in 10 by hash) has the same
  mix, so the model is also reported on the 2022+ bodies apart (the backtest window). The labels are
  automatic and unchecked: the owner reads 100 `lead` and 100 `continuation` paragraphs
  (`scripts/draw_segment_check_2026_11_05.py`, T-148b2) before the model trains, and the measured
  precision of each kind is recorded here (**read 2026-11-06, 200 paragraphs marked: `lead` 88.0% [80.2, 93.0], `continuation` 38.0% [29.1, 47.8], 2022+ `continuation` 55.2% [37.5, 71.6]; the rule below gives: train on `lead` only**; of the 12 wrong `lead` labels, 5 are paragraphs about two or more companies, 3 a different entity with a similar name (a ticker or a person) and 4 are not text about a company: market-mover intros, a table row, a cross-reference). **Rule fixed before the figure is seen (owner,
  2026-11-06):** if at most 80% of the `continuation` paragraphs are right they are dropped from
  training (train on `lead` only); otherwise both kinds are used. With 100 paragraphs each
  precision has an interval of about +/-8 to 10 points, and it is reported for 2022+ apart. **The rule is on
  the point estimate, as the owner fixed it** (the one bar not judged on the interval): the report prints the
  interval and a note when it straddles 80%, but the decision does not change; with no `continuation` row
  marked the report stops instead of choosing (`train_on_lead_only`). The check file must carry `pub_date`.
  **Comparison unit (D32):** the model and the baseline are compared on the same 30 bodies, but the baseline cuts
  by sentence and gives a sentence to every company it names, the structured parser by paragraph; the gain
  therefore measures the extractor as a whole (cut and attribution), not the attribution alone. The model is a new pinned checkpoint (constitution AI #1, #7, #12), **built 2026-11-07 (T-148b3, merged in #141; private Hub copy, #185)**: a pair classifier, *is this sentence about this company?*, on the encoder of the pinned sentiment model (a new 2-label head; input the company's name, the previous sentence and the sentence, max 160 tokens), trained on the `lead` paragraphs only (19,724 paragraphs, 74,512 pairs, 2 epochs, 15.5 minutes; the run refuses unless `train_on_lead_only` holds), `val` 270 bodies never trained on. At run time a sentence that names universe companies belongs to them (the baseline's rule, so a named company always has its text); a sentence naming none goes to the best-scoring company if the score is at least 0.5, else to nobody. Automatic figures on `val` (parser labels, they decide nothing): the right company is ranked above a wrong one 90.1% of the time against 63.7% for the mention-and-recency baseline. **The faithfulness reading (D32) decides.** First set (190 texts of 30 bodies, owner 2026-11-07): baseline 60.5% [45.2, 73.8]; model 76.3% [61.8, 88.1]; `units` (T-148b4: footer stripped, cut at lines, bullets, semicolons and clauses) 85.4% [76.5, 92.2] on its 185 texts, `better` than the baseline (+24.3 [12.9, 39.4]). The confirmation set and the selection of `units` are in (e) below and in `docs/extractor_training.md`. **Open:** the model has not learned to answer "none of these" (a sentence about a company outside the universe goes to the nearest one at a score of 0.5 or more; the wrong cases score 0.9 or more, so a higher threshold does not help): TASKS T-148b3. The build record, the `val` figures, the abstention counts and the second reading of the bar (companies named only in the footer): `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, "D27 build record".
- **D27 (e), amended: the cut reads the body as `units4` (owner, 2026-11-11; T-176, #169), then as `units5` (owner, 2026-11-12; T-177, #176).** The selection of 2026-11-08 (`units`, measured 97.5% on the first set and 97.1% on the confirmation set after #155) stands for the figures it was measured on; it is not rewritten. The first run of the cut on 232 bodies found two faults of `units` (162 texts with several "Name —" headings, 6 that were a heading alone); `units4` (`docs/extractor_training.md`, "Update of 2026-11-11") reads each "Names – text" sentence on its own and gives it to the universe companies of the head and those its text names, or to nobody. Measured: texts with several headings 162 → 23, heading-only 6 → 0, and on the owner's marks 97.4% faithful (298 of 306) against 97.6% (323 of 331), the paired difference −0.2 points [−0.5, 0.0] (a bootstrap over the 60 bodies): not better, possibly a fraction of a point lower; one reader. **The trade-off the owner accepted: 18 of 331 companies (5.4%) lose their text** (17 marked faithful), named in passing in another company's line; the NER step (D39 (iii)) is expected to find them in a sub-news, **an assumption that T-165's closing check measures**. `units4` was `DEFAULT_READING` of `src/cut_stage.py` until #176 made it `units5` (2026-11-12); `--reading units4` and `--reading units` cut as before; the revision of the cut starts with the reading, so old and new rows sit side by side. Review of #169 found a third fault (13 of the 23 remaining texts): T-177. **Interactive session, owner 2026-11-11, after #169:** (1) the owner read the 7 unread changed texts: 6 faithful, 1 not (a roundup tail the cut did not split); two of the six are an analyst named as the investor behind an action, first marked not faithful and then, by the owner's rule for every mark, faithful (tagged `investor_activity` on the other company), so `units4` is **304 of 313, 97.1% [94.6, 98.5]** on the marked texts against 97.6% for `units`; (2) of the 18 lost companies, 9 are the real subject of a roundup line lost after a header of nobody, 8 are named in passing and 1 is a matcher gap, so the premise "the 18 are named in passing" holds for 8 of them; (3) T-177 ships as a new reading `units5` with its own revision (**built as a candidate in #172, `units4` stays the default**; on a blind sample of 30 changed texts 29 are faithful, and on the marked texts it is 97.8% against 97.1% for `units4`, +0.6 [0.1, 1.5], which comes from the added texts being faithful, not from existing texts improving), and also gives a sentence with no header of its own after a header of nobody to the universe companies it names; (4) the short-name matcher gap ("Disney", "JPMorgan"…) is its own task, T-181; (5) the full cut run waits for T-177. **Owner, 2026-11-12:** `units5` becomes the default of the cut (`DEFAULT_READING`, a small PR after #172; the evidence is fewer faults and 12 companies back with no measured loss of faithfulness, not better faithfulness), and the model's output is then sampled, 150 texts, for the owner to test faithfulness before the full run is trusted. **Done in #176:** `units5` is the default, and the cut prints its reading and revision before it starts. The full cut of 2026-11-11 (`units-5f831c298bcb`) was written with the `units` reading, before the switch; the owner cuts again with `units5`, and the 150-text sample is drawn from that cut (T-183).
- **D27 (e), final: `units5` is the final reading of the cut (step 1.2) (owner, 2026-11-12, after T-183).** The reference cut is `units5-d3e2cbfa830e` (161,574 sub-news of 30,305 `multi` bodies, 2,250 `skipped` (6.9%, no company text), 0 `failed`; checkpoint `data/models/attribution_2026_11_08`, identity `8399ccf003c60143`, base `gamug/FinBERT-financial-news@93863fcb`), the one every later step names (T-165; the summary of sub-news: `--cut-revision units5-d3e2cbfa830e`); `units-5f831c298bcb` (the `units` reading) and a partial `units4-d3e2cbfa830e` (same 12 hex digits, other prefix) stay as history. **Evaluation:** (1) *the extractor on the owner's marks*: `units5` 351 of 359 faithful, 97.8% [95.5, 99.4]; (2) *the output of the cut, 143 pieces of 28 bodies marked yes / no (T-183)*: faithful 84.6% [77.8, 89.6] on all 143 marked pieces (the lower end is below the 80% bar, D32) and 89.1% [82.5, 93.4] without the 15 pieces scoped wrongly by step 1.1 (a whole body of 13 and two single pieces); the other levels, the other two questions and the marks are in `docs/scope_training.md` §8.3 and `gold/cut_sample_marks.csv`. **The bar of the extractor (>= 80% faithful) is not met on the pre-set measure, all pieces** (lower end 77.8%); it is met on the levels chosen after the misses, which are the owner's reading, not a pre-set test. The owner takes the cut as good enough and the misses as the tagging of step 1.1 (a body that should be `investment_general`), the company read from the title (step 3, T-181) and ambiguous round-ups. **Not pinned to the Hub:** the checkpoint is local (T-148b12: publishing it is the owner's word); until then the identity above names it.
- **D29 Category** is computed on the same texts as sentiment (step 5): the summary of a
  non-multi body, the company text of a multi body. The D24 target (>= 80% with the held-out gold)
  is measured there.
- **D30 Held-out gold for the scope.** 200 bodies drawn once on 2026-11-05
  (`gold/category_gold_v2_test.csv`; 200 is the minimum at which an observed 90% keeps a 95% lower
  bound near 85%): all 39 bodies the rules call no-company, sector, market or economy, the rest a
  seeded random draw from the 1,000-body corpus sample, none sharing a body with the 720 or the 44;
  checked for leakage on 2026-11-05 (same body, title, ticker and day, near-duplicate text,
  earlier evaluation runs: one article, 247586, was in an old judged run with the previous
  model and judge, and is kept). The 50-article filter check, the ~60 random misses and the ~50
  double-tagged bodies of Q4/Q6/Q9 are unchanged. The sample was also used to size the rules, so a
  rules-versus-gold figure on it is slightly optimistic. **Weights:** the bodies that share an article
  with the 720 or the 44 were left out of the draw and were not a random part of the sample (the 720
  were stratified), so a rate is re-weighted to the corpus with the size of each rule scope in the whole
  1,000-body sample (as drawn, by the matcher at 786766f: it stays so, D38) (`frame` in `data/heldout_strata.csv`, also copied to the `frame` column of `gold/category_gold_v2_test.csv` when the tags are pulled, so a rate can be re-weighted from `gold/` alone; not the `population` column of `heldout_ids.csv`, which counts the bodies left to draw from), not with the bodies left to draw from. The frame (`data/heldout_strata.csv`, written by the draw
  script on every run from the 720 and the 44 only) is: single 514, multi_dominated 245, roundup 108,
  multi 93, sector 18, no_company 12, economy 5, market 5 (1,000 bodies; 7 left out for sharing an
  article with the 720 or the 44, so the weights barely move).

- **T-132 shrunk (owner, 2026-11-05).** It was designed around T-117's call budget and T-130's
  relevance judge, both superseded. Only the upstream alias table stays; the quarantined pairs
  follow D19 (their results are moved aside reversibly, T-131, D36 (h1); the first wording said "purged").

**Owner calls on the four open questions (interactive session, 2026-11-05; D31–D34).**

- **D31 Training years: all years, disclosed.** The sentiment model and the category model are trained on bodies of
  all years. About 30% of the sentiment training sentences, of the gold bodies and of the held-out bodies are from 2022
  on, and the labels have no date cut-off, so nothing limits hindsight. Whoever uses the data for a backtest is told
  these facts and chooses: our models, base `ProsusAI/finbert`, or "not evaluated". A pre-2022 retrain stays possible
  later without a redesign.
- **D32 Acceptance bars (point estimate, 95% interval reported).** Sub-news cut (the `units` extractor; the attribution model is not in the MVP, D39): at
  least **80% of the companies named in the owner's 30 narrative multi bodies (190 company texts today) with a faithful text** (a named company the extractor writes no text for counts as not faithful; the share of the texts it did write is printed beside it, owner 2026-11-06),
  and better than the mention-and-recency baseline on the same 30 (**if the owner has not finished the 190 by 2026-11-09, proposed and to be confirmed by the owner before any figure is read: the first 120 or more rows marked in `review_order` order decide, `report_wi18 faithfulness --partial 120`; with fewer, the baseline is kept and the model is reported as not evaluated**), reported apart for bodies from
  2022 on. Sentiment on the 180-text gold (60 per class **of the model's predicted class**):
  **overall agreement >= 80%, and precision and recall each >= 70% for the negative class** (D37, 2026-11-09: model selection and the pin read precision, recall and ROC OVR per class instead of the agreement; the 80% bar is proposed to stop gating) (the one
  the backtest counts). Because the draw is balanced by prediction, every figure is re-weighted by the
  stratum sizes (`population` in the ids file) and its interval is a bootstrap over the strata.
  Negative **precision** is read inside the predicted-negative stratum (60 texts, about +/-13
  points); negative **recall** needs the true negatives hidden in the other two strata, so it is a
  re-weighted ratio with a wider interval than +/-13, reported as it comes. A bar is **passed** when the
  whole 95% interval is at or above it, **failed** when the whole interval is below it, and "not decided"
  otherwise (the interval straddles the bar, or only touches it from below); **`undefined` when the
  figure has no interval** (nothing was predicted, or no text is in the group: a negative precision of a
  model that says negative for no text): it is not a pass, and it is not an interval that straddles the
  bar, so it is not "not decided" (`Figure.judge`, 2026-11-07); the point estimate is
  always printed and never decides alone (`Figure.judge`, edges tested). **Sentiment result (2026-11-06, the 180 texts marked, `report_wi18 sentiment --gold --out gold` reproduces it from the tracked files): overall agreement 60.4% [52.8, 67.6] (raw 57.2%), so it FAILS the 80% bar; negative precision 78.7% [67.3, 89.3] and recall 77.7% [69.4, 86.8], both not decided against 70%. The largest error is a text the model calls neutral: re-weighted, 54.8% of them are positive for the owner (34 of the 60 drawn). This meets D26's condition (the model is fine-tuned on summaries only if it misses the D32 bar); the decision between fine-tuning (T-148c) and sending the cut with the sentiment layer marked failed is the owner's and is open.** Faithfulness is judged the same way, for all
  texts and for the bodies from 2022 on apart; **its point is the share of faithful texts, and its interval is a
  bootstrap over the bodies** (about 30), not over the texts, because the company texts of one body stand or fall
  together (`faithfulness_report`, `paired_difference` with clusters) (`faithfulness_ids.csv` carries `pub_date`). **The gain over the
  baseline** (model minus baseline on the same company texts, paired bootstrap) is `better` when the whole
  interval is above 0, `not better` when it is below 0, `undefined` when the gain has no interval (NaN), `not decided` otherwise (`gain_verdict`; the edges are those of
  `Figure.judge`: an interval that only touches the bar, or 0, decides nothing); it is
  also printed for 2022+ apart. **A company an extractor names but writes no text for counts as an unfaithful
  text**, in the share and in the gain, so an extractor cannot gain by leaving the hard companies out
  (`faithfulness_coverage.csv`, written by the draw, printed by the report). A file or id the report cannot place
  stops the run naming the rows, and so does a tagged file that lacks rows of the draw or repeats a key. The result "not decided" is reported as such,
  the backtest team is told so and the owner decides whether to fine-tune (default written by the
  developer, for the owner to confirm or change). **Coverage:** the 180 hold the summaries and the
  company texts of structured multi bodies (120) and the texts the extractor writes for narrative
  multi bodies (60: the baseline, or the model if it is ready on 2026-11-09, the date of D34 rule (b)); the two parts are reported
  apart, and the figures of the second part are the ones that say whether the cut's per-company
  sentiment of narrative multi bodies can be trusted. The sentiment intervals resample texts inside each
  stratum, so for the company texts of one body they are a little narrow. **The stratum sizes are those of the pool of the owner's tagged bodies**
  (`gold/category_gold_v2.csv`: the 240-body gate and the 44 sector bodies, built on the tuning sets), **not of the corpus
  or of the 2022+ window**: the sentiment figures are weighted to that pool, they are not corpus figures (owner,
  2026-11-06: relabelled, the draw is kept). The report prints the texts drawn from 2022 on in each stratum. A stratum with fewer
  texts than the draw takes is **drawn whole** (owner, 2026-11-06): the others are unchanged and the report shows the real counts and
  the wider interval; `--pool-only` shows the shortfall before the draw day, and the scored pool is kept so the draw does not score twice (with its conditions: a different extractor, tagged bodies or model revision stops the draw). A stratum drawn whole is held fixed in the bootstrap (owner, 2026-11-06) and marked `(census)` in the report. A company both extractors skipped is in the shares but not in the gain. The model is fine-tuned on summaries only if it misses the bar.
- **D33 D20 amended: no per-scope recall floor under 40 gold bodies.** The overall target
  (>= 90% corpus-weighted agreement on all bodies; 80% now) stays. The recall >= 80% floor applies to a
  scope only if it has at least 40 gold bodies; a scope with fewer is reported with its recall and
  95% interval and not gated. No top-up draw is made. The 200 held-out bodies (D30) are therefore
  enough as drawn.
- **D34 The backtest cut for the 2026-11-20 thesis delivery: out of the MVP (owner, 2026-11-10; T-154, T-153 cropped).** The text: `archive/PIPELINE-superseded-2026-11-10.md`.

- **D35 Sentiment fine-tuned on summaries the owner tags (T-148c, owner 2026-11-07).** The model missed the D32 bar on the 180-text gold (overall agreement 60.4% [52.8, 67.6] against 80%), so D26's condition for fine-tuning on summaries is met.  **Marks in and migrated (2026-11-08):** 450 of 450 (232 positive, 113 negative, 105 neutral; train 180 / 96 / 84, validation 52 / 17 / 21) in `gold/sentiment_train.csv`; **the label is anchored to the tagged company (owner's rule): the mark is what the text says about the company of the row, and it is neutral when the company is not in the facts the text presents, even if other sentences are positive or negative; the title is context to know whether a text speaks about the company, but a sentence is not the company's only because the title names it; the company anchor is a design feature: a model that scores the text without the company is expected to disagree with these marks on the rows where the company is absent.** The file carries `ticker` and `company` on every row (the ticker is blank on 26 sector, market and economy rows). The rule was given on 2026-11-08 after the marks were in; whether it was followed on every row is not recorded (set 5's notes show the same practice, not proven for its 149 rows without a note). The anchor's measured weight is the 19 rows whose note says something about the company by keyword (approximate, no bound: absent 9, nothing relevant 4, other scope 3, namesake 3; all marked neutral, 16 of them called non-neutral by the model); four rows probably carry a namesake company (listed in `docs/gold_datasets.md`, proposed T-148c1b); by `rule_scope` the draw's model agrees with the owner on 55.4% of `single` and 62.8% of `multi_dominated` rows, a difference whose interval [-2.7, 17.1] straddles zero. **The rest of the disagreement is not explained** (`docs/gold_datasets.md`, set 6). Validation shares companies with train (64 of 83 named-company validation rows): proposed for T-148c3, the owner's to accept, a split of the agreement by companies seen and unseen. The retraining (T-148c3) waits for the owner.
  - **Training set:** the owner tags 450 summaries of the cut (`scripts/draw_sentiment_train_2026_11_07.py`), 150 of each class the current model predicts (that prediction is used only to balance the draw and is never shown), never a body of the 180 gold or the held-out set, by body hash or by headline. They are only summaries: the texts of multi bodies (T-148b) do not exist yet, so the gold's 60 extractor texts are not represented. 
  - **Order:** 30 blocks of 15 rows (5 of each predicted class), a batch of 150 is 10 blocks. 
  - **Validation:** the last 2 blocks of each batch (30 rows, 10 per class; 90 in all) are a validation slice, never trained on, that picks the epochs and the settings; the other 360 train. 
  - **The 180-text gold is not used to tune** (a figure chosen by looking at it would be optimistic, and it goes to the backtest team): it is scored for the baseline (the pinned model: run 1 of `gold/sentiment_gold_scoring_log.csv`, which reproduces D32) and for the final run with the settings chosen on the validation slice; every other scoring is logged too (`scripts/score_sentiment_gold_2026_11_07.py` appends each run), and the report that quotes a figure lists all of them, none dropped; the figure that counts is the final run's. It is reported for 2022 on and before apart (D31): the training texts are 2022 and later, the gold is not. 
  - **Length too (2026-11-07, measured):** the longest drawn summary is 108 tokens, the 180-text gold reaches 268 and 19 of its texts are over 128 (the extractor texts), so a drop on the long texts is a distribution gap, not necessarily a defect; the report shows the agreement for texts over 128 tokens apart from the rest (T-148c2b). 
  - **Stopping (owner, 2026-11-07):** no stopping rule: the owner tags all 450 (the three batches today), and the validation figure is information (90 texts: about +/-10 points at 70%). 
  - **Recipe (owner, 2026-11-07, accepted as proposed and fixed before any mark is read):** start from the pinned model `gamug/FinBERT-financial-news`; train on the train slice plus the same number of sentences drawn (seeded) from the earlier balanced pool (`data/sentiment_finetune/`, 3,865 sentences), so that the summaries do not make it forget the sentence level (the replay leaves out the sentences of an article whose body, by hash, is a gold, held-out or drawn-for-tagging body: 2 of the 3,865 came from articles of the 180 gold, by id, 2026-11-07; the trainer prints how many it removed); maximum length 512 (summaries); **the training settings the earlier fine-tune used (`src/train_sentiment.py`) apply unchanged unless said here: per-device batch 16, AdamW with weight decay 0.01, the library's linear decay with no warmup and gradient clipping at 1.0, fp16, no class weights (the 150-per-class draw and the balanced pool are balanced), and the model after the last epoch of the setting, not the best epoch (the epoch count is itself the grid, `load_best_model_at_end` is off); a batch of 8 was used in the timing below, the trainer times its own run at 16; a batch never mixes the two kinds of text (summaries padded to the longest summary, sentences to the longest sentence), and each epoch shuffles the batches of both kinds together, which is what the timing did apart from the shuffle order**; epochs in {2, 3, 4} and learning rate in {1e-5, 2e-5, 3e-5}, 3 seeds each, chosen by the mean agreement on the validation slice (the earlier fine-tune used 2e-5 and 4 epochs); then one final run with the chosen settings, **trained on the train slice only (360 rows plus the replay), the same data the settings were chosen on, not on all 450** (owner, 2026-11-07, accepted as proposed) (adding the 90 validation rows would give a model whose settings were chosen on rows it has seen; the 90 are used only to choose), with a fourth seed that is fixed in the trainer before any run and is not chosen by looking at a result; the gold figure is reported with the mean and the spread of the validation agreement over the three seeds of the chosen settings, so that one seed on 180 texts is not read as more than it is. **Built 2026-11-09 (T-148c3, the amendment the data fact asked for):** the loader leaves out the 26 marked rows that have no company (they stay with the text-only model); the replay keeps only the sentences that name their article's company (925 of 3,865 eligible, 341 drawn, balanced by class); the fourth seed is 4. **Measured on the validation slice (T-148c3, D37 metrics, weighted by stratum, indicative, not the gold):** the pinned model's score is 60.9% read as text and 61.3% as a pair (the pair moves it between classes: positive recall 23.7% against 55.6%); the fine-tuned model's paired gain over the pinned text reading is `not decided` for the selection score (6.8% [-2.1, 15.3]), `better` for negative recall and `not decided` for negative precision (46.0% [26.9, 65.0]) and ROC OVR: **the pin rule would not pin on this slice**; T-148c3 has the table.
  - **Order of the end (D35):** train; publish the candidate to the private Hub repo `gamug/FinBERT-financial-news-summaries` (name accepted by the owner, 2026-11-07; **the published repo is `gamug/FinBERT-company-sentiment` (T-169, 2026-11-10, #164): it holds the pair model and the stack file; the owner confirmed the new name on 2026-11-10**; the `HF_TOKEN` in `.env` has the `write` role, checked 2026-11-07); score that published revision on the 180-text gold once as the `final` run; the repo is made public only in the pin PR (T-148c4), so that a model that failed its bar is not left looking shipped. **Pin rule (D35 (3), (4), (7)):** pinned only if the paired gain on the gold is `better`; not pinned if the paired gain of negative precision or negative recall is `not better` (an `undefined` one blocks too). The full text is in `archive/PLAN-d35-order-of-the-end-2026-11-10.md`.
  - **Budget and tie rule:** one run is timed first (with the GPU otherwise idle), and the 27 runs plus the final one must fit in the hours reserved, else the smaller grid (epochs in {3, 4}, learning rate 2e-5, 3 seeds) is used and the report says so; a setting replaces the earlier one (2e-5, 4 epochs) only if it beats it by more than one standard error of the mean over the seeds (9 settings compared on 90 texts would otherwise pick a winner that is partly noise). 
  - **Path to the cut:** the marks are read by a builder (T-148c1), the gold is scored by a script that runs a candidate model (T-148c2, `report_wi18 sentiment --gold` reads the old model's stored predictions and runs nothing), the scorer is hardened first, with the paired gain over the baseline that the pin rule reads (T-148c2b), the trainer follows the recipe (T-148c3), and the model is published and pinned in `config/pipeline_models.json` in its own PR (T-148c4) before 2026-11-12. 
  - **GPU order (owner, 2026-11-07):** the retraining ran before the attribution model; the attribution model is not in the MVP (D39), so the order no longer matters. The scheduling text and timings: `archive/PIPELINE-superseded-2026-11-10.md`.
  - **Fallback (owner):** if the model still fails the bar on 2026-11-11, the cut goes on 2026-11-12 with the sentiment layer marked as measured, with its figures, and the fine-tuning continues after the delivery.
  - **Questions of D35 (six answered by the owner, 2026-11-07, each as proposed; three more on 2026-11-09, interactive session, see (7) to (9)):** (1) an `undefined` bar counts as not passed (D34 rule (a)); (2) the retraining goes before T-148b3 (GPU); (3) the candidate is pinned only if the paired gain is `better` (Order of the end); (4) no pin if the paired gain of negative precision or recall is `not better` (Order of the end); (5) the candidate repo is `gamug/FinBERT-financial-news-summaries`, private until a pin PR (Order of the end); (6) the final run trains on the 360 train rows plus the replay, a fourth fixed seed (Recipe, T-148c3); (7) (2026-11-09) an `undefined` gain of the negative precision or recall also blocks the pin (read as not passing), as D35 (1) reads an undefined bar; (8) (2026-11-09) the replay stays as built (only sentences that name their article's company); (9) (2026-11-09) T-148c4 has no date of its own: the fallback of 2026-11-11 and the go/no-go of 2026-11-12 stand, and it starts after the review of #154 is closed, with the owner's word.
  - **D35 amendment accepted, final model selected (owner, 2026-11-10):** endpoint pair model on all rows + relevance stack; not yet pinned: `docs/sentiment_training.md`.
- **D36 The properly fetched corpus is preserved (owner, 2026-11-08, interactive session).** `quarantined` (`reject`, `review`) is a judgement about the **pair** (is this company a subject of this article?), made upstream from the URL and the title; it is not a judgement that the article is not financial news. Real examples of `reject` under Agilent's ticker `A`: "Abel puts a big chunk of Berkshire's cash to work", "A huge day and week for Corning as the S&P 500 aims for record close". The SOURCE `articles` table holds, read-only (the row counts of `scripts/count_fetched_corpus_2026_11_08.py`, run 2026-11-09; the unique-body figures of its first run, 2026-11-08, are the same): `ok` 98,462 rows (all with a title and a `body_text`), `quarantined` 377,010 (376,873 with a title, 374,551 with a non-empty `body_text`, 374,419 with both; the first reading said all 377,010 had a body, but 2,459 have a null or empty one), `thin_content` 1,785 (1,652 with a title and a body); the old results are the 459,112 RESULTS rows of the 2026-10-22 count (`accept` 93,330, `reject` 315,822, `review` 49,960), a different table from the SOURCE (which holds 477,257 rows; `accept` 93,330 is not `ok` 98,462, and the 5,132 difference is not explained). **Rule:** an article that was properly fetched (a title and a body) stays in the corpus and eligible for the pipeline whatever its pair class; the pair class says which company a stored result may be attributed to, not whether the body exists. So (a) no step deletes, overwrites or leaves out an article because its pair is quarantined; (b) the derived results of a quarantined pair (T-131) are moved aside in a reversible archive, never destroyed; (c) the unit is still the unique body and its subjects are found in the content (D11), so a body fetched under a wrong ticker is read for what it says, and its tagged ticker is not evidence for or against any company; (d) every size quoted from now on says which corpus it is of: the `ok` pairs (98,462 articles, 81,961 unique bodies, 29,113 in the cut's window) or all properly fetched bodies (309,882; 73,771 in the window).

- **D36, decisions of the session and what is proposed.** **Decided in the same session:** (e) the 2026-11-16 cut stays on the `ok` window (29,113 bodies, summaries done); the preserved corpus joins after the delivery; (f) the `thin_content` rows (1,785; 1,652 with a title and a body) are **not** part of the properly fetched corpus (they stay in SOURCE, preserved, not processed); (g) constitution #16 is kept as written. **Counted 2026-11-08, rerun 2026-11-09 with the rows left out counted too (T-159, `scripts/count_fetched_corpus_2026_11_08.py`, read-only):** unique bodies with a title and a body: `ok` **81,961** (the D19 figure, as it should be), quarantined 260,845, of which **227,921 are not also `ok` under another ticker**, so the properly fetched corpus is **309,882 unique bodies (3.8 times the `ok` one)**; in the cut's window (from 2021-12-31) `ok` 29,113 (the cut's figure), quarantined-only 44,658, properly fetched **73,771 (2.5 times)**, a lower bound: 204 quarantined-only bodies have no readable date and are not in the window. By pair class, the SOURCE rows with a title and a body: `quarantined/reject` 322,902, `quarantined/review` 51,517 (a different table from the RESULTS `reject` 315,822), `failed` 578 (not counted). The result file `data/fetched_corpus_count.json` is local scratch (git-ignored); this paragraph is the record (owner, 2026-11-09: the file is not committed). The quarantined-only bodies are mostly older (about 44,600 are from 2022 on, the rest earlier), median 463 words, 7,503 under 100 words and 54,428 under 300. 505 companies have an `ok` body by tag and 3 have only quarantined ones. Not known yet: how many of the quarantined-only bodies are about a company at all (T-160). **Decided by the owner, 2026-11-09 (interactive session): (h1) to (h3) as proposed in #149, (i) and (j) as recommended in the session, (k) the owner chose the text-only reading instead of the recommended "decide after seeing the shares":** (h1) D19's and T-132's "purged" is reworded "moved aside reversibly" (T-131), as (b) of the rule says; (h2) the lean `articles` row in RESULTS is derived and restorable (`articles_removed`), the SOURCE row is the article; (h3) (e) is the owner's answer to the "stops and asks" of constitution #16 for the cut of 2026-11-16 only, and the constitution's text is not changed; (i) the T-159 result file stays local, this paragraph is the record; (j) T-153 is sent after the company CSV run of T-160 (PR #152); (k) that CSV (T-160, PR #152) reads the names from the text only, with no stored NER spans, so that the `ok` and the quarantined-only bodies are read alike. **Proposed, not decided:** D19's backfill scope after the delivery (today the 81,961 `ok` unique bodies) and T-150, once T-160 says how much of the 227,921 is usable.

- **D37 Metrics of evaluation and of model selection (owner, 2026-11-09).** **The metrics to record, and to use in model selection, are Precision, Recall and ROC OVR** (the ROC AUC of each class against the rest, one-vs-rest, from the model's probability of the class). Read as the owner answered in the session of 2026-11-09: (a) they **replace agreement (accuracy)** as the selection, bar and pin metric; agreement stays a description only (the confusion matrix); (b) they are recorded **per class** (positive, negative, neutral) for every run, the pinned model's readings included; the **selection score** is their macro mean: the mean over the three classes of each metric, then the mean of the three metrics, equal weights, an undefined figure counting 0 (D34 (a)); D35's tie rule (4 epochs at 2e-5 stays unless the best setting beats it by more than 1 SE of this score) reads it; (c) the thresholds stay: D32's bar of 70% for the negative class's precision and recall; ROC OVR is recorded and used, with no bar until the owner gives one; the pin rule reads the paired gain of these metrics. **Proposed, for the owner to confirm (the answers did not settle them):** the D32 agreement bar of 80% no longer gates (it follows from (a)); the pin rule reads (3) the paired gain of the selection score `better` and (4) no pin if the negative class's precision, recall or ROC OVR gain is `not better` (an `undefined` one blocks, D35 (7)); and the rule stands for the other stages' model selection (NER, category, summaries), where `news_nlp.eval` already logs per-class precision, recall and `roc_auc_<class>` (`docs/evaluation.md`), only when each is next evaluated. Built for the sentiment validation slice in T-148c3 (`predict_probs` returns the class probabilities; the report is `src/sentiment_pair_report.py`); the gold scorer logs the same in T-148c2c (proposed).
- **D38 Common words and surnames are not companies in the counts (T-161, PR #155; follows D37 of #154).** 15 tickers had counts far too high (FRT, STT, UNP, BE, XYZ, ...). **Cause:** the text matcher `CompanyResolver` (not the NER model, decision (k)): first-word aliases ("Federal", "Bloom") and one-word names ("Block"); retraining the NER would not change them. **Owner, 2026-11-09:** fold the other aliases into #155; CSV re-run after #155; redraw the gold pool and marks after the last matcher PR.
  **The rest (changes (1) to (8), the measurements, the disclosure and the open items):** `archive/PIPELINE-superseded-2026-11-10.md`, `docs/matcher_measurements_2026-11-09.md`, T-161.

## Sequencing

**Current focus: Work item 18** (P1 sizing and the rules' check is done; P2 labels is open, then the steps of D39, 2026-11-10). Work items 1, 5, 6, 8–14 and 16 are closed (their tasks are in `CHANGELOG.md`); Work item 15 is superseded by 18 and Work item 17 is folded into it; Work items 3, 4 and 7 are resolved except the non-blocking T-022, T-032 and T-062; Work item 2 (the regression gate) is blocked on the maintainer's infrastructure decision. The order inside Work item 18 is its own "Approach". The per-item sequencing notes written before 2026-11-08: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, "Sequencing".

See `TASKS.md` for the discrete, checkable task breakdown.
