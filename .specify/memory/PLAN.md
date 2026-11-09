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

## Work item 2 — Make the `--check-regression` gate runnable

**Why**: `.github/workflows/eval.yml` already invokes
`cli/news_nlp_eval.py … --check-regression` — the code-level gate
`SPEC.md` §13 item 8 was originally written as missing already exists.
What's actually missing is the *infrastructure* the workflow's own header
comment says it assumes and doesn't have: a self-hosted runner with
`$SOURCE_DATABASE_URL`/`$DATABASE_URL` mounted, a persistent
`$MLFLOW_TRACKING_URI` for `--check-regression` to have a prior run to
compare against, and the `LLM_API_KEY`/`LLM_MODEL`/`LLM_URL` repo secrets.
On the current `runs-on: ubuntu-latest`, the workflow has no data to run
against — it's a template, not a working gate, and its `schedule:` trigger
is presently running (or failing) against nothing.

**This is an operational task, not a code task** — constitution AI
behavior #10 ("ask before expanding scope") applies: standing up a
self-hosted runner and provisioning secrets is an infrastructure decision
for the repo owner to make and execute, not something to do
unilaterally inside a PR. This plan documents what's needed; a human
completes it.

**Steps** (for the repo owner/maintainer):

1. Stand up (or designate) a runner with persistent access to the working
   SOURCE/RESULTS SQLite files and a durable `mlruns` directory (or a real
   MLflow tracking server) — the same machine the manual
   `uv run cli/news_nlp_eval.py` invocations already run on is the
   simplest choice.
2. Register it as a GitHub self-hosted runner with (at least) the
   `thesis-data` label `eval.yml` already references in its commented-out
   `runs-on:` line.
3. Add the six repo secrets `eval.yml` reads:
   `LLM_API_KEY`, `LLM_MODEL`, `LLM_URL`, `MLFLOW_TRACKING_URI`,
   `SOURCE_DATABASE_URL`, `DATABASE_URL`.
4. Uncomment `runs-on: [self-hosted, thesis-data]` and remove/replace the
   `runs-on: ubuntu-latest` line beneath it.
5. Trigger once via `workflow_dispatch` to confirm the run completes,
   produces an `eval_run` row, and (on a second manual run) that
   `--check-regression` actually compares against the first.

**Acceptance criteria**:

- A `workflow_dispatch` run of `eval.yml` completes successfully on the
  designated runner and produces a real MLflow run + `eval_run` row.
- A deliberately-induced regression (e.g. temporarily lowering
  `--regression-tolerance` below a known noise level, or comparing against
  a run with a known worse metric) causes the workflow to fail — proving
  the gate actually gates, not just runs.
- `SPEC.md` §13 item 8 updated to reflect the new state (resolved, or
  re-scoped if the maintainer decides differently — e.g. keeping it
  scheduled-only rather than PR-blocking, which is a legitimate choice
  given the LLM cost of running this on every PR).

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
weekly `sector_summary`, market and economy news become `MARKET` / `ECONOMY` rows there; D15 the
per-(body, company) taxonomy is the 9 labels + `reputational_harm`, `competitive_actions`,
`investor_activity`, `peer_comparison`, with `other` (company content that fits none of the 13
labels; settled 2026-10-29: the investing news about no company that D15 first named is the scope
`investing_general`, D22, and carries no category), `uncertain` (no confident category: a
category-level label only, the scope never abstains, D20) and `unrelated` (not about the economy); D16 `unrelated` bodies are
flagged and skipped by every later stage; D17 a multi-company body gets sentiment + category per
company; D18 about 6,000 LLM labelling calls (a hard cap, logged in `docs/evaluation.md`, counted per
task, the run stops at it), the owner re-tags the 720 under the new taxonomy (v1 was tagged blind to the LLM label: the review file had no LLM column; the v2 file keeps it hidden and shows only the owner's v1 tag) (gold v2; the first tagging is "tags v1", a development
set), the revised prompt iterates on a fixed stratified 240 of them (at most 5 iterations) and must agree
with the owner's scope on >= 90% of the full 720 before silver is bought (at most 4,320 bodies: 6,000 minus
1,200 iteration calls minus 480 for the full pass; a 6th iteration is never spent without asking), and the
owner tags about 300 held-out gold bodies drawn across the corpus, at least 40 per scope (D30 drew 200 instead; D33: no top-up, and no recall floor for a scope under 40 gold bodies); D19 full backfill of the **81,961 unique `ok` bodies** (not the
459,112 stored rows): the stored rows are moved aside as v1 until the new tables are validated, and
the stored results of the quarantined pairs are moved aside, reversibly, not destroyed (T-131; owner 2026-11-09, D36 (h1), D36 (b): the first wording said "purged"; 377,010 are SOURCE rows, the RESULTS has 365,782 `reject` and `review` rows); D20 the scope triage is accepted
only at >= 90% corpus-weighted agreement with the owner's held-out gold on all bodies (no abstention) and
recall >= 80% for each of the six scopes (including a random sample of ~60 bodies the prefilter misses, Q9), both judged on the point estimate with the 95% interval reported
(at 40 bodies the per-scope interval is about +/-13 points; D33 amends: a scope with fewer than 40 gold bodies is reported with its interval, not gated); D21 each
training round sends a stratified sample of about 300 low-confidence bodies to the owner for tagging
(active learning); D22 six scopes: single company, several companies, sector, market / economy,
`investing_general` (investing or market-advice content about no company and no sector, the owner's
`other` tag: no per-company analysis, summarised in the `MARKET` row of `sector_summary`) and `unrelated`; D23 a
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

- **D25 Order of the stages.** (1) the scope of the body; (2) NER, only for `single` and `multi`
  bodies; (3) a summary of every `single`, `sector`, `market_economy` and `investing_general` body (not `multi`, not `unrelated`); (4) for a `multi` body, the text each
  mentioned company gets; (5) sentiment, and the D15 category (D29), over the texts of steps 3 and 4;
  (6) company snapshots; (7) sector snapshots. `unrelated` stops at step 1 (D16). NER no longer
  runs on every body: sector, market / economy and `investing_general` bodies skip it.
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
  to a **trained attribution model**: a small local encoder trained on (a) with the company name
  masked and the previous sentence as context, so it learns who a sentence is about from its
  content. It is compared with a mention-and-recency baseline and with a faithfulness check the
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
  therefore measures the extractor as a whole (cut and attribution), not the attribution alone. The model is a new pinned checkpoint (constitution AI #1, #7, #12), **built 2026-11-07 (T-148b3, merged in #141; not published or pinned)**: a pair classifier, *is this sentence about this company?*, on the encoder of the pinned sentiment model (a new 2-label head; input the company's name, the previous sentence and the sentence, max 160 tokens), trained on the `lead` paragraphs only (19,724 paragraphs, 74,512 pairs, 2 epochs, 15.5 minutes; the run refuses unless `train_on_lead_only` holds), `val` 270 bodies never trained on. At run time a sentence that names universe companies belongs to them (the baseline's rule, so a named company always has its text); a sentence naming none goes to the best-scoring company if the score is at least 0.5, else to nobody. Automatic figures on `val` (parser labels, they decide nothing): the right company is ranked above a wrong one 90.1% of the time against 63.7% for the mention-and-recency baseline. **The faithfulness reading (D32) decides.** First set (190 texts of 30 bodies, owner 2026-11-07): baseline 60.5% [45.2, 73.8]; model 76.3% [61.8, 88.1]; `units` (T-148b4: footer stripped, cut at lines, bullets, semicolons and clauses) 85.4% [76.5, 92.2] on its 185 texts, `better` than the baseline (+24.3 [12.9, 39.4]). The confirmation set and the selection of `units` are in (e) below and in `docs/extractor_training.md`. **Open:** the model has not learned to answer "none of these" (a sentence about a company outside the universe goes to the nearest one at a score of 0.5 or more; the wrong cases score 0.9 or more, so a higher threshold does not help): TASKS T-148b3. The build record, the `val` figures, the abstention counts and the second reading of the bar (companies named only in the footer): `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, "D27 build record".
- **D28 Snapshots.** New tables in RESULTS, the old ones kept as v1 (D19). A company snapshot
  is one row per (body, company): text, ticker, D15 label, sentiment, `pub_date`; a sector
  snapshot is one row per (body, sector): text and sentiment. This repository does not aggregate
  per day: the per-(asset, day) signal of D14 is built downstream from the per-body rows, which
  also keeps one count per unique body for the backtest. Sketch only: the schema is proposed to
  the owner in the P5 PR (constitution AI #10).
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
  1,000-body sample (`frame` in `data/heldout_strata.csv`, also copied to the `frame` column of `gold/category_gold_v2_test.csv` when the tags are pulled, so a rate can be re-weighted from `gold/` alone; not the `population` column of `heldout_ids.csv`, which counts the bodies left to draw from), not with the bodies left to draw from. The frame (`data/heldout_strata.csv`, written by the draw
  script on every run from the 720 and the 44 only) is: single 514, multi_dominated 245, roundup 108,
  multi 93, sector 18, no_company 12, economy 5, market 5 (1,000 bodies; 7 left out for sharing an
  article with the 720 or the 44, so the weights barely move).

- **T-132 shrunk (owner, 2026-11-05).** It was designed around T-117's call budget and T-130's
  relevance judge, both superseded. Only the upstream alias table stays; the quarantined pairs
  follow D19 (their results are moved aside reversibly, T-131, D36 (h1); the first wording said "purged").

**Owner calls on the four open questions (interactive session, 2026-11-05; D31–D34).**

- **D31 Training years: all years, disclosed.** The sentiment model, the category model and the
  step-4 attribution model are trained on bodies of all years. About 30% of the sentiment training
  sentences, of the gold bodies and of the held-out bodies are from 2022 on, and the labels have
  no date cut-off, so nothing limits hindsight. The backtest team (decides on dates from
  2021-12-31 to 2026-10-15) is told these facts and chooses: our models, base `ProsusAI/finbert`
  for their runs, or "not evaluated" (T-153). A pre-2022 retrain stays possible later without a
  redesign (about 2,200 of the 2,867 round-up bodies are from before 2022).
- **D32 Acceptance bars (point estimate, 95% interval reported).** Step-4 attribution model: at
  least **80% of the companies named in the owner's 30 narrative multi bodies (190 company texts today) with a faithful text** (a named company the extractor writes no text for counts as not faithful; the share of the texts it did write is printed beside it, owner 2026-11-06),
  and better than the mention-and-recency baseline on the same 30 (**if the owner has not finished the 190 by 2026-11-09, proposed and to be confirmed by the owner before any figure is read: the first 120 or more rows marked in `review_order` order decide, `report_wi18 faithfulness --partial 120`; with fewer, the baseline is kept and the model is reported as not evaluated**), reported apart for bodies from
  2022 on. Sentiment on the 180-text gold (60 per class **of the model's predicted class**):
  **overall agreement >= 80%, and precision and recall each >= 70% for the negative class** (the one
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
  (>= 90% corpus-weighted agreement on all bodies) stays. The recall >= 80% floor applies to a
  scope only if it has at least 40 gold bodies; a scope with fewer is reported with its recall and
  95% interval and not gated. No top-up draw is made. The 200 held-out bodies (D30) are therefore
  enough as drawn.
- **D34 The backtest cut for the 2026-11-20 thesis delivery (T-154).** A short path, built while the
  full plan continues: for the bodies of the backtest window (published 2021-12-31 or later, about
  29,000 unique bodies), scope from the rules (not the classifier), NER for single and multi
  bodies, a summary of each non-multi body (measured 0.84 s per body on the local GPU, up to 7
  hours for the window), the step-4 company texts (D27), and sentiment over the texts checked on
  the 180-text gold (D32; the 60 extractor texts cover narrative multi bodies, and a row's
  `sentiment_checked` says whether its kind of text was in the gold, whose figures are weighted to the tagged-body pool, D32). **Output (owner, 2026-11-06):**
  files, not tables: chunks under `data/cut/` (git-ignored, they hold news text), one row per
  (body, company) with the approved columns of T-152, written chunk by chunk so a stopped run resumes
  from the last chunk; no schema change. **Coverage:** the stored corpus, which ends 2026-08-12; the
  backtest window runs to 2026-10-15, and the bodies the backtest team adds upstream are not in the
  cut. **Provisional**: the output says so and carries the measured figures; it
  is not the accepted pipeline. Left to after the delivery: the trained scope classifier and its
  90% bar, the prompt gate and the silver labels, category (D24) and sub-industry, sector
  snapshots, active learning, the full backfill, the knowledge-graph contract. **Go/no-go on
  2026-11-12**: if the owner's tagging (the held-out 200, 200 label-check paragraphs, the 30
  190 faithfulness texts (30 bodies; 2026-11-06 the draw gave 190, not the 30 assumed) and the 180 sentiment texts: 770 items, 200 of them marked) is done and the extractor passes D32 (**owner call 2026-11-06: tag all 190 faithfulness texts**),
  the data goes to the backtest team about 2026-11-16 (four days before delivery, which is tight
  for them: T-153 asks for the latest date they can use); otherwise they are told, with a week's
  notice, to report the layer as not evaluated. **Measured 2026-11-07 on the held-out 200 (T-146e, PR #142): the rules' scope agreement 68.0% [61.3, 74.5] weighted to the corpus (raw 59.5%); single recall 91.0% [87.1, 94.4] passes the 80% bar, multi recall 52.5% [42.8, 63.5] fails it; sector, market_economy, unrelated and investing_general have 9, 11, 8 and 7 gold bodies, so no recall floor applies (D33) and their recalls (2.6%, 12.1%, 0.0%, 0.0%) are disclosure, not rates. The cut uses the rules' scope (this decision); *proposed, not decided:* report these figures with the cut and leave the trained scope classifier (left to after the delivery) as the answer to the multi gap; the go/no-go is the owner's.** **Rules fixed beforehand (owner, 2026-11-06):**
  (a) a sentiment figure whose interval straddles its bar (D32 "not decided") does not stop the
  cut: it goes with the point estimate and the interval printed; **(owner, 2026-11-07: a figure that FAILS its bar, as overall sentiment agreement does since 2026-11-06, is answered by fine-tuning on summaries: D35; a figure whose interval then straddles the bar is "not decided" and this rule applies); **(owner, 2026-11-07, answered: an `undefined` bar, D32, is not "not decided" and this rule does not cover it: it counts as not passed, and the cut goes only with the sentiment layer marked as measured, as in D35's fallback, never as "not decided")** (b) if the attribution model is not
  ready on 2026-11-09, the faithfulness texts are exported from the baseline alone and the decision
  is on the baseline (**result 2026-11-06, 190 texts of the 30 bodies marked: baseline 60.5% faithful, interval 45.2 to 73.8, so it fails the 80% bar; 2022+ 60.6% [27.6, 86.1], not decided; of the texts it wrote 60.5% [53.4, 67.2]; no company was left without a text (`gold/faithfulness_coverage.csv` is empty), and `report_wi18 faithfulness --gold --out gold` reproduces these figures from the tracked files; one body of 21 texts has 4 faithful. Unless the model (T-148b3) passes, narrative multi bodies are left out of the cut and counted**): the cut uses it for narrative bodies if it is at least 80% faithful (D32 reading: of the companies named),
  otherwise narrative multi bodies are left out of the cut and counted; (c) the label check of
  D27 decides, before the model trains, whether the `continuation` labels are used; (d) a sentiment stratum with fewer texts than the draw takes is drawn whole (a census: held fixed in the bootstrap and marked in the report), the others do not change, and the cut still goes (D32); the scored pool is kept with the extractor, tagged bodies and model revisions it was made under, and a draw under other conditions stops; **(e) Selection of step 4's extractor (owner, 2026-11-08): `units` ("units1") is the base model**, because the accuracy does not need to be above 80% and it clears it: on the confirmation set 86.1% [80.9, 90.7] on every company named (pass) and 95.3% [92.0, 98.2] on real companies; on the first set (development) 94.0% [87.0, 99.2] on real companies (pass) and 83.2% [72.7, 91.0] on every company named (not decided). It is level with the baseline (gain -0.5 [-1.5, 0.0], not decided), so the proposed rule T-148b10 (6b) ("better") is not met and the selection is the owner's call over it. A selection, not a publication or a pin (both need the owner's word), and no code uses it yet (T-148b12, proposed). Rules that stand (owner, 2026-11-07): an extractor whose faithful share is "not decided" against D32's 80% (the point at or above it, the interval straddling it) goes to the cut marked as measured, with the figure and the interval printed, as rule (a) does for sentiment; a FAIL (the whole interval below 80%) does not go; an `undefined` figure counts as not passed; the pull requests of step 4 stay stacked, none merged or published without the owner's word. `units2` and `units3` (the guard and the header rule, #144) are measured fallbacks, not selected (78.4% and 79.9% on every company named in the confirmation set; 7 to 12 real companies left without a text); they are development figures. Record, figures and options: `docs/extractor_training.md`; the history of this rule, with the readings of the 37 and of the 30 names, the manager's review of #144 and the candidate chosen on 2026-11-07: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, "D34 (e)". **Runs:**
  the summaries of the window (29,000 bodies x 0.84 s is up to 6.8 h; only non-multi bodies need
  one) start on 2026-11-07 evening, before the go, because they do not depend on the tagging; the
  model trains on 2026-11-08 on the same GPU after them (**order set by the owner in D35, GPU: the retraining first, about 20 minutes**); NER, step 4 and sentiment run after the
  go, 2026-11-12 to 2026-11-14. **Spreadsheet safety:** the files the owner opens in a spreadsheet get a leading apostrophe on any value that starts with `=`, `+`, `-`, `@`, a tab or a return, outside the columns the owner marks (`retag.spreadsheet_safe`, applied by the four draw scripts; a headline like "-5% on the day" shows with the apostrophe). Concurrent draws stay accepted (single-owner tooling).

- **D35 Sentiment fine-tuned on summaries the owner tags (T-148c, owner 2026-11-07).** The model missed the D32 bar on the 180-text gold (overall agreement 60.4% [52.8, 67.6] against 80%), so D26's condition for fine-tuning on summaries is met.  **Marks in and migrated (2026-11-08):** 450 of 450 (232 positive, 113 negative, 105 neutral; train 180 / 96 / 84, validation 52 / 17 / 21) in `gold/sentiment_train.csv`; **the label is anchored to the tagged company (owner's rule): the mark is what the text says about the company of the row, and it is neutral when the company is not in the facts the text presents, even if other sentences are positive or negative; the title is context to know whether a text speaks about the company, but a sentence is not the company's only because the title names it; the company anchor is a design feature: a model that scores the text without the company is expected to disagree with these marks on the rows where the company is absent.** The file carries `ticker` and `company` on every row (the ticker is blank on 26 sector, market and economy rows). The rule was given on 2026-11-08 after the marks were in; whether it was followed on every row is not recorded (set 5's notes show the same practice, not proven for its 149 rows without a note). The anchor's measured weight is the 19 rows whose note says something about the company by keyword (approximate, no bound: absent 9, nothing relevant 4, other scope 3, namesake 3; all marked neutral, 16 of them called non-neutral by the model); four rows probably carry a namesake company (listed in `docs/gold_datasets.md`, proposed T-148c1b); by `rule_scope` the draw's model agrees with the owner on 55.4% of `single` and 62.8% of `multi_dominated` rows, a difference whose interval [-2.7, 17.1] straddles zero. **The rest of the disagreement is not explained** (`docs/gold_datasets.md`, set 6). Validation shares companies with train (64 of 83 named-company validation rows): proposed for T-148c3, the owner's to accept, a split of the agreement by companies seen and unseen. The retraining (T-148c3) waits for the owner.
  - **Training set:** the owner tags 450 summaries of the cut (`scripts/draw_sentiment_train_2026_11_07.py`), 150 of each class the current model predicts (that prediction is used only to balance the draw and is never shown), never a body of the 180 gold or the held-out set, by body hash or by headline. They are only summaries: the texts of multi bodies (T-148b) do not exist yet, so the gold's 60 extractor texts are not represented. 
  - **Order:** 30 blocks of 15 rows (5 of each predicted class), a batch of 150 is 10 blocks. 
  - **Validation:** the last 2 blocks of each batch (30 rows, 10 per class; 90 in all) are a validation slice, never trained on, that picks the epochs and the settings; the other 360 train. 
  - **The 180-text gold is not used to tune** (a figure chosen by looking at it would be optimistic, and it goes to the backtest team): it is scored for the baseline (the pinned model: run 1 of `gold/sentiment_gold_scoring_log.csv`, which reproduces D32) and for the final run with the settings chosen on the validation slice; every other scoring is logged too (`scripts/score_sentiment_gold_2026_11_07.py` appends each run), and the report that quotes a figure lists all of them, none dropped; the figure that counts is the final run's. It is reported for 2022 on and before apart (D31): the training texts are 2022 and later, the gold is not. 
  - **Length too (2026-11-07, measured):** the longest drawn summary is 108 tokens, the 180-text gold reaches 268 and 19 of its texts are over 128 (the extractor texts), so a drop on the long texts is a distribution gap, not necessarily a defect; the report shows the agreement for texts over 128 tokens apart from the rest (T-148c2b). 
  - **Stopping (owner, 2026-11-07):** no stopping rule: the owner tags all 450 (the three batches today), and the validation figure is information (90 texts: about +/-10 points at 70%). 
  - **Recipe (owner, 2026-11-07, accepted as proposed and fixed before any mark is read):** start from the pinned model `gamug/FinBERT-financial-news`; train on the train slice plus the same number of sentences drawn (seeded) from the earlier balanced pool (`data/sentiment_finetune/`, 3,865 sentences), so that the summaries do not make it forget the sentence level (the replay leaves out the sentences of an article whose body, by hash, is a gold, held-out or drawn-for-tagging body: 2 of the 3,865 came from articles of the 180 gold, by id, 2026-11-07; the trainer prints how many it removed); maximum length 512 (summaries); **the training settings the earlier fine-tune used (`src/train_sentiment.py`) apply unchanged unless said here: per-device batch 16, AdamW with weight decay 0.01, the library's linear decay with no warmup and gradient clipping at 1.0, fp16, no class weights (the 150-per-class draw and the balanced pool are balanced), and the model after the last epoch of the setting, not the best epoch (the epoch count is itself the grid, `load_best_model_at_end` is off); a batch of 8 was used in the timing below, the trainer times its own run at 16; a batch never mixes the two kinds of text (summaries padded to the longest summary, sentences to the longest sentence), and each epoch shuffles the batches of both kinds together, which is what the timing did apart from the shuffle order**; epochs in {2, 3, 4} and learning rate in {1e-5, 2e-5, 3e-5}, 3 seeds each, chosen by the mean agreement on the validation slice (the earlier fine-tune used 2e-5 and 4 epochs); then one final run with the chosen settings, **trained on the train slice only (360 rows plus the replay), the same data the settings were chosen on, not on all 450** (owner, 2026-11-07, accepted as proposed) (adding the 90 validation rows would give a model whose settings were chosen on rows it has seen; the 90 are used only to choose), with a fourth seed that is fixed in the trainer before any run and is not chosen by looking at a result; the gold figure is reported with the mean and the spread of the validation agreement over the three seeds of the chosen settings, so that one seed on 180 texts is not read as more than it is. **Built 2026-11-09 (T-148c3, the amendment the data fact asked for):** the loader leaves out the 26 marked rows that have no company (they stay with the text-only model); the replay keeps only the sentences that name their article's company (925 of 3,865 eligible, 341 drawn, balanced by class); the fourth seed is 4. **Measured on the validation slice (T-148c3, weighted by stratum, indicative, not the gold):** the pinned weights read the pair worse than the text (agreement 41.5% against 55.0%), and the fine-tuned model's paired gain over the pinned text reading is `better` for agreement and negative recall and `not decided` for negative precision (46.0% [27.8, 65.2]); T-148c3 has the table.
  - **Order of the end (so that the gold is looked at once for the candidate):** train; publish the candidate (a Hub revision); score **that published revision** on the 180-text gold as the `final` run (`score_sentiment_gold --role final`); the candidate is published to a **private** Hub repo of its own, `gamug/FinBERT-financial-news-summaries` (name accepted by the owner, 2026-11-07: the pinned `gamug/FinBERT-financial-news` is public and a private revision cannot be added to it; `HF_TOKEN` in `.env` has the `write` role, checked 2026-11-07), scored with `HF_TOKEN` and the repo is made public only in the pin PR, so that a model that failed its bar is not left looking shipped (owner, 2026-11-07, accepted as proposed); the `final` row and the pin name the **commit SHA** of the published revision, never a branch or tag name (those move); pin it in `config/pipeline_models.json` only if the result allows it (**the test keeps the pin to one of the two scored models: the pin is the last `final` row's model and revision and that row's `gain_verdict` is `better`, or the pin is the `baseline` row's, the candidate not having been adopted; the `final` row is the candidate's whenever it is scored, before the verdict is known, so the pin cannot be required to equal it**). **When the result allows it (owner, 2026-11-07, accepted as proposed):** the gain of the final run over the baseline row, on the same 180 texts, with its interval from the paired bootstrap of `gold_metrics.paired_difference` (clustered by body), has the verdict of `gain_verdict` (D32): **`better`** pins it, and the cut uses it, marked as measured if it still misses the D32 bar (the fallback below); **`not better`** keeps the baseline pinned; **`not decided` or `undefined`** keeps the baseline pinned too (as D34 rule (a) does for `undefined`: a change of the pinned model needs evidence, a point estimate above the baseline's by a text or two is not); in every case the cut uses the pinned model, marked as measured while it misses the bar, and the report gives both runs' figures. ****"The baseline row" and "the `final` row" mean the last row of that role (a `--redo-role` adds one; the earlier rows stay in the log as a record, as `last_final_run` does).** **This rule compares with the baseline of this round, the original pinned model; a later round (the fine-tuning continues after the delivery) names its own baseline, the model pinned then, in its own decision.** The baseline's predictions for the gain are the stored `predicted` column of the 180 (the scorer already checks it against a rerun, 0 of 180 differ); the gain is refused if the agreement computed from that column differs from the `baseline` row's logged agreement.** **(owner, 2026-11-07, accepted as proposed: the negative class, which is the signal the backtest team uses, is not guarded by the overall gain, since a balanced retraining could gain overall and lose negative recall; the report gives the paired gain of negative precision and of negative recall next to the overall one, and the model is not pinned if either is `not better`.)** **If the baseline stays pinned** the candidate repo stays private and its `final` row stays in the log as the record; making it public, or deleting it, is the owner's decision later, never a default. The strata (2022 on, before 2022, texts over 128 tokens) are reported, not a condition of the pin: with 49 and 131 texts their intervals (about 14 and 8 points either side) are too wide for a condition to bind. The trainer does not score the gold: it scores the validation slice, through the scorer's reading. **Before anything is published** the final run's own validation agreement (through the scorer's reading) is printed next to the mean and the spread of the three seeds of the chosen settings, and the run stops, publishing nothing, if it is below the mean of the three by more than 2 standard deviations or by more than 3 texts of 90, whichever is larger (a fourth seed that none of the 27 runs measured must not be first checked on the gold; "below the lowest of the three" was rejected: a healthy fourth draw is the lowest of four 1 time in 4 and would be blocked a quarter of the time). **A second `final` scoring (`--redo-role`) is allowed only for a defect in the run (a wrong revision, a scoring error), never because of the figure; the first row stays in the log and in the report.** 
  - **Budget and tie rule:** one run is timed first (with the GPU otherwise idle), and the 27 runs plus the final one must fit in the hours reserved, else the smaller grid (epochs in {3, 4}, learning rate 2e-5, 3 seeds) is used and the report says so; a setting replaces the earlier one (2e-5, 4 epochs) only if it beats it by more than one standard error of the mean over the seeds (9 settings compared on 90 texts would otherwise pick a winner that is partly noise). 
  - **Path to the cut:** the marks are read by a builder (T-148c1), the gold is scored by a script that runs a candidate model (T-148c2, `report_wi18 sentiment --gold` reads the old model's stored predictions and runs nothing), the scorer is hardened first, with the paired gain over the baseline that the pin rule reads (T-148c2b), the trainer follows the recipe (T-148c3), and the model is published and pinned in `config/pipeline_models.json` in its own PR (T-148c4) before 2026-11-12. 
  - **GPU (owner, 2026-11-07: the retraining goes before the attribution model T-148b3, accepted as proposed; how long T-148b3 occupies the GPU is not estimated yet):** the order of the three jobs is: (1) the summaries cut, which started on 2026-11-07 evening and occupies the GPU up to 6.8 h (D34, Runs); (2) the retraining, as soon as the 450 are marked (the owner expects them on 2026-11-07) and the cut has released the GPU, because the budget rule bounds it and the fallback of 2026-11-11 hangs on it; (3) the attribution model (T-148b3), which **replaces D34's "the model trains on 2026-11-08 on the same GPU after them"** (that sentence puts the attribution model right after the cut; if the owner chooses that order, D35 gives way and the retraining runs after it, with the grid reduced by the budget rule), its readiness rule (b) of 2026-11-09 unchanged (if it is not ready then, the baseline's texts are used). How long T-148b3 occupies the GPU is not estimated yet. **Timed 2026-11-07 (GPU idle, after the cut, RTX 4050, batch 8, fp16, 360 summaries plus 360 sentences, the predicted class as a placeholder label because the cost is the length): one epoch about 9 s (12.7 s the first), 2.1 GiB at the peak. The drawn summaries are short (median 73 tokens, longest 108, none over 512), so the 512 limit never cuts. The full grid is 81 epochs plus the final run's 4, about 13 minutes of training, about 20 minutes with loading and the validation scoring of each run; the budget rule's smaller grid will not be needed and the GPU order costs T-148b3 about 20 minutes, not a day: readiness rule (b) of 2026-11-09 is not endangered by the order.** The cost of the order for the rule (b), if T-148b3 were not ready, is the rule's own: the faithfulness texts are exported from the baseline and narrative multi bodies are left out and counted. Scoring the 180 takes about 9 seconds (measured 2026-11-07, model load included). 
  - **Fallback (owner):** if the model still fails the bar on 2026-11-11, the cut goes on 2026-11-12 with the sentiment layer marked as measured, with its figures, and the fine-tuning continues after the delivery.
  - **Questions of D35 (all six answered by the owner, 2026-11-07, each as proposed):** (1) an `undefined` bar counts as not passed (D34 rule (a)); (2) the retraining goes before T-148b3 (GPU); (3) the candidate is pinned only if the paired gain is `better` (Order of the end); (4) no pin if the paired gain of negative precision or recall is `not better` (Order of the end); (5) the candidate repo is `gamug/FinBERT-financial-news-summaries`, private until a pin PR (Order of the end); (6) the final run trains on the 360 train rows plus the replay, a fourth fixed seed (Recipe, T-148c3).

- **D36 The properly fetched corpus is preserved (owner, 2026-11-08, interactive session).** `quarantined` (`reject`, `review`) is a judgement about the **pair** (is this company a subject of this article?), made upstream from the URL and the title; it is not a judgement that the article is not financial news. Real examples of `reject` under Agilent's ticker `A`: "Abel puts a big chunk of Berkshire's cash to work", "A huge day and week for Corning as the S&P 500 aims for record close". The SOURCE `articles` table holds, read-only (the row counts of `scripts/count_fetched_corpus_2026_11_08.py`, run 2026-11-09; the unique-body figures of its first run, 2026-11-08, are the same): `ok` 98,462 rows (all with a title and a `body_text`), `quarantined` 377,010 (376,873 with a title, 374,551 with a non-empty `body_text`, 374,419 with both; the first reading said all 377,010 had a body, but 2,459 have a null or empty one), `thin_content` 1,785 (1,652 with a title and a body); the old results are the 459,112 RESULTS rows of the 2026-10-22 count (`accept` 93,330, `reject` 315,822, `review` 49,960), a different table from the SOURCE (which holds 477,257 rows; `accept` 93,330 is not `ok` 98,462, and the 5,132 difference is not explained). **Rule:** an article that was properly fetched (a title and a body) stays in the corpus and eligible for the pipeline whatever its pair class; the pair class says which company a stored result may be attributed to, not whether the body exists. So (a) no step deletes, overwrites or leaves out an article because its pair is quarantined; (b) the derived results of a quarantined pair (T-131) are moved aside in a reversible archive, never destroyed; (c) the unit is still the unique body and its subjects are found in the content (D11), so a body fetched under a wrong ticker is read for what it says, and its tagged ticker is not evidence for or against any company; (d) every size quoted from now on says which corpus it is of: the `ok` pairs (98,462 articles, 81,961 unique bodies, 29,113 in the cut's window) or all properly fetched bodies (309,882; 73,771 in the window).

- **D36, decisions of the session and what is proposed.** **Decided in the same session:** (e) the 2026-11-16 cut stays on the `ok` window (29,113 bodies, summaries done); the preserved corpus joins after the delivery; (f) the `thin_content` rows (1,785; 1,652 with a title and a body) are **not** part of the properly fetched corpus (they stay in SOURCE, preserved, not processed); (g) constitution #16 is kept as written. **Counted 2026-11-08, rerun 2026-11-09 with the rows left out counted too (T-159, `scripts/count_fetched_corpus_2026_11_08.py`, read-only):** unique bodies with a title and a body: `ok` **81,961** (the D19 figure, as it should be), quarantined 260,845, of which **227,921 are not also `ok` under another ticker**, so the properly fetched corpus is **309,882 unique bodies (3.8 times the `ok` one)**; in the cut's window (from 2021-12-31) `ok` 29,113 (the cut's figure), quarantined-only 44,658, properly fetched **73,771 (2.5 times)**, a lower bound: 204 quarantined-only bodies have no readable date and are not in the window. By pair class, the SOURCE rows with a title and a body: `quarantined/reject` 322,902, `quarantined/review` 51,517 (a different table from the RESULTS `reject` 315,822), `failed` 578 (not counted). The result file `data/fetched_corpus_count.json` is local scratch (git-ignored); this paragraph is the record (owner, 2026-11-09: the file is not committed). The quarantined-only bodies are mostly older (about 44,600 are from 2022 on, the rest earlier), median 463 words, 7,503 under 100 words and 54,428 under 300. 505 companies have an `ok` body by tag and 3 have only quarantined ones. Not known yet: how many of the quarantined-only bodies are about a company at all (T-160). **Decided by the owner, 2026-11-09 (interactive session): (h1) to (h3) as proposed in #149, (i) and (j) as recommended in the session, (k) the owner chose the text-only reading instead of the recommended "decide after seeing the shares":** (h1) D19's and T-132's "purged" is reworded "moved aside reversibly" (T-131), as (b) of the rule says; (h2) the lean `articles` row in RESULTS is derived and restorable (`articles_removed`), the SOURCE row is the article; (h3) (e) is the owner's answer to the "stops and asks" of constitution #16 for the cut of 2026-11-16 only, and the constitution's text is not changed; (i) the T-159 result file stays local, this paragraph is the record; (j) T-153 is sent after the company CSV run of T-160 (PR #152); (k) that CSV (T-160, PR #152) reads the names from the text only, with no stored NER spans, so that the `ok` and the quarantined-only bodies are read alike. **Proposed, not decided:** D19's backfill scope after the delivery (today the 81,961 `ok` unique bodies) and T-150, once T-160 says how much of the 227,921 is usable.

**Owner answers for P2/P3 (2026-10-29), folded into D18/D20 above. No question is open.**
- *Q1* gate on a fixed 240 of the 720, at most 5 iterations; *Q2* point estimate with interval; *Q3*
  the held-out gold is drawn once before any training and excluded from every active-learning round
  (its tags are training data never); the one exception is Q4's second pass on about 50 of its
  bodies, which measures the label ceiling and is not used for training; *Q4* the owner re-tags all
  720 under the new taxonomy with the tags v1 shown as guidance: gold v2, not an independent
  consistency measure, so the ceiling is measured on the ~50 held-out bodies tagged twice a week
  apart (the owner agrees when they are drawn); order: labelling spec v2, re-tag file, then the
  gate on gold v2; the rules' 0.70 is re-measured on gold v2; *Q5* if the classifier stalls below
  its target the work stops and the owner decides; no LLM is added to the pipeline (D12).
- **Answered 2026-10-29.** *Q6 sampling frame:* custom rules prefilter the candidates for each rare
  scope (sector, market / economy, unrelated, `investing_general`), and the owner verifies a sample of
  **50 prefiltered articles** by hand before the held-out gold is drawn from them. Caveat recorded:
  the 50 check the filter's precision; its recall (rare-scope bodies the filter misses) stays
  unmeasured unless a random check sample is also read, and the inverse-sampling-fraction weights are
  valid only for strata defined before the draw, so rare-scope recall figures from this gold are
  conditional on the filter. *Q7 rounds:* **2 active-learning rounds** (about 300 bodies each) before
  the owner is asked again; owner effort about 1,070 bodies before round 1 plus about 600. *Q8 silver
  beyond scope:* per-company category and sub-industry silver labels are bought only at **>= 80%
  agreement with gold v2 on the 240** (the D24 bar); below it silver stays scope-only. *Q9 filter
  misses:* the held-out gold adds **~60 bodies drawn at random from outside the prefilter**, weighted
  by inverse sampling fraction, so per-scope recall (D20) includes what the filter misses; these
  are a third set of bodies, separate from the 50-article filter check and the 50 double-tagged.
  *Q10 calls per body:* **one call per body**; the reply lists every company with its D15 label, so
  the 240-call iteration and the 4,320 ceiling hold. Owner tagging tally: 720 re-tag + ~300
  held-out + ~60 misses + 50 filter check + 50 double-tag = about 1,180 bodies before round 1
  (the heavier tagging: scope, companies, D15 label per company, sub-industry), then 2 x ~300.

**Approach** (one PR per phase; each new table is proposed to the owner in the PR that needs it):

1. **P1 — foundations and sizing (T-144, T-145).** A read-only script counts unique bodies and
   resolves the companies each one names (name and explicit ticker forms, NER `ORG` spans matched
   to universe names; the upstream alias table of T-132 when it arrives), then gives rule-based scope
   candidates and a GICS guess, and scores the rules against the owner's 720 tagged articles. No
   spend, no schema change.
2. **P2 — labels (T-146).** Labelling spec v2, the revised prompt, the D18 silver labels and the
   owner's ~300 gold bodies.
3. **P3 — triage stage (T-147).** On unique bodies: the scope first (single company,
   several companies, sector, market / economy, `investing_general`, `unrelated`) from a local classifier (a fine-tuned
   small encoder over the text, the P1 rule features as inputs) trained on silver and gold, with
   active-learning rounds on its low-confidence bodies (D21) until it reaches >= 90% on the
   held-out gold (D20). Sketch of the tables: a body table keyed by body hash with its member pairs, and a
   per-body triage row (scope, companies, confidence). NER then runs on `single` and `multi`
   bodies only (D25).
4. **P4 — texts, sentiment and category (T-148).** The summary of every non-multi body (the pinned
   local summariser, step 3); for a multi body, the text of each company (step 4, D27: the
   one-segment-per-company parser, then the trained attribution model); then sentiment and the D15
   category over those texts (D26, D29), with the 180-text sentiment gold.
5. **P5 — sector, market and economy news, and the snapshots (T-149, T-152).** GICS sector /
   sub-industry detection, summaries of those bodies into `sector_summary` with `MARKET` /
   `ECONOMY` rows, and the company and sector snapshot tables (D28).
6. **P6 — backfill and close-out (T-150).** Re-process the stored corpus, update the
   knowledge-graph export contract, close Work items 15, 17 and 18 into `CHANGELOG.md`.

**Acceptance.** Every body has a scope, which agrees with the owner's held-out gold on >= 90% of
bodies (corpus-weighted point estimate, interval reported) and reaches recall >= 80% in each of the six
scopes (point estimate, interval reported; the floor applies to a scope with at least 40 gold bodies, D20 as amended by D33), measured with the
complete metric set (AI #12); the per-company category (P4) and the sector sub-industry (P5) each agree
with the held-out gold on >= 80% (D24, confirmed); no multi-company, sector, market or economy body is labelled `other`;
per-company sentiment and category exist for every company a body is about, computed on its
summary or company text (D26); company and sector snapshots exist, one row per body (D28); `sector_summary`
carries sector articles and `MARKET` / `ECONOMY` rows; the knowledge-graph export is updated and
documented; SPEC gains one FR per behaviour shipped.

## Sequencing

**Current focus: Work item 18** (P1 sizing and the rules' check is done; P2 labels is open, then P3–P6 in the stage order of D25, 2026-11-05). Work items 1, 5, 6, 8–14 and 16 are closed (their tasks are in `CHANGELOG.md`); Work item 15 is superseded by 18 and Work item 17 is folded into it; Work items 3, 4 and 7 are resolved except the non-blocking T-022, T-032 and T-062; Work item 2 (the regression gate) is blocked on the maintainer's infrastructure decision. The order inside Work item 18 is its own "Approach". The per-item sequencing notes written before 2026-11-08: `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`, "Sequencing".

See `TASKS.md` for the discrete, checkable task breakdown.
