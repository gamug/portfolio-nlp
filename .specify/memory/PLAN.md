# PLAN.md — `portfolio-nlp`

The implementation plan for the live backlog identified in
`.specify/memory/SPEC.md`. Where the constitution is principles and
`SPEC.md` is the requirements/architecture contract, this document is the
"how, and in what order" for the work that contract still leaves open.

**Scope of this plan was originally narrow, now expanded to cover active
model-performance work.** `SPEC.md` §13 (Open Questions & Risks) now lists
sixteen items; §14 (Scope Boundaries) marks most of the original nine as
**accepted** (permanent characteristics of this project at its current,
non-production scope) and one (§13 item 5, throughput/latency SLA)
**retired** outright. Item 8 was flagged "should fix regardless of scope"
and items 1/2 (sentiment/category accuracy) were originally treated as
accepted research limitations — see Work items 1-2 below for the former.
Items 1/2 have since been **promoted out of "accepted, not a queued
task"**: category's fix already shipped (Work item 5), and sentiment is
now resolved, priority work (Work item 4, and its data-quality follow-up,
Work item 9). Item 10 (`c_summary`) is resolved (Work item 6). Item 15
(the pipeline/eval architecture itself, never formally spec'd — an
organically-grown pragmatic solution, not a from-scratch design) is
**done (Work item 10, closed 2026-09-18)**. Item 16 (running a real model
experiment requires chaining many independent, hand-invoked one-off
commands, with no single declarative, reproducible experiment definition)
is now this plan's **top priority (Work item 11)**, ahead of item 13's
still-pending justification work (Work item 8). All per
`docs/evaluation.md`'s dated follow-ups and the current focus of this
project. This plan still does not resurrect anything §14 leaves closed for
the other items — see Non-goals below.

## Goal

Close the backlog items that are genuinely actionable without expanding
this project's scope beyond what's already in motion:

1. Pin the four HF model checkpoints to a commit SHA (reproducibility;
   part of SPEC.md §13 item 4). — Work item 1.
2. Make the `--check-regression` accuracy gate (`.github/workflows/eval.yml`)
   actually runnable, not just a template (SPEC.md §13 item 8). — Work item 2.
3. Get a post-fix accuracy reading for NER's 2026-09-10
   subword-fragmentation fix and resolve its suspected eval-sampling
   mismatch (SPEC.md §9 NER baseline). — Work item 3.
4. Close the diagnosed-but-unimplemented reasoning gap behind sentiment's
   weak precision, the stage `SPEC.md` §13 item 1 still calls out as
   weakest (SPEC.md §13 item 1). — Work item 4.
5. Hold the line on category's already-shipped hierarchical fix and
   close the one open calibration thread it left (`other`'s own
   precision) (SPEC.md §13 item 2). — Work item 5.
6. Validate the summarization stages: confirm or fix `c_summary`'s own
   suspected eval-sampling scope mismatch, decide whether to address its
   weak `mean_coverage` (SPEC.md §13 item 10), and add a lightweight
   faithfulness check for `sector_summary`'s model-generated intro
   sentence, which shares the same model and currently has no evaluation
   at all. — Work item 6.
7. Develop batch processing for the NER stage (SPEC.md §13 item 11) —
   `run_ner_stage` is the one stage with no batching, unlike category and
   the summarization stages, a real throughput cost measured directly
   during T-025's 2026-09-12 resample. — Work item 7.
8. Add a per-model "why this model" justification sub-section to the
   Models evaluation artifact section, for every model in the pipeline —
   the accuracy numbers now live in one place (2026-09-14 reorg), but
   *why each specific architecture was chosen over the alternatives* does
   not (SPEC.md §13 item 13). — Work item 8.
9. Fix the sentiment fine-tuning data's class imbalance (56.1% neutral /
   22.8% negative / 21.1% positive, never a deliberate target — SPEC.md
   §13 item 14): rebalance the published dataset, retrain the model on
   the rebalanced data, and measure the result against the current
   version. — Work item 9.
10. Formalize the pipeline/evaluation architecture — never a from-scratch
    spec, a pragmatic solution to a real necessity that grew incrementally
    instead (SPEC.md §13 item 15): restructure the four ML stages around a
    shared Feature/Train/Inference (FTI) class hierarchy, move
    `sector_summary` fully into its own non-FTI module, and redesign
    `news_nlp.eval` around the concrete friction this project's own
    sentiment-candidate work exposed — no way to run multiple experiments'
    judged data in one store, no persisted confusion matrix, no ROC, no
    reuse mechanism to avoid re-spending judge-LLM tokens on already-tagged
    data. **Done 2026-09-18.** — Work item 10.
11. **Top priority.** Replace the hand-chained, one-off-script way of
    running a model experiment (dataset prep → train → offline eval →
    downstream eval → publish, each its own command, several mutating
    shared DB tables and needing a manual restore step afterward — SPEC.md
    §13 item 16) with one JSON file per experiment (pretrain or not, which
    dataset, train/test split + stratify strategy, how many articles to
    sample for the LLM judge, the experiment's name, everything needed to
    reproduce it — generic across sentiment/NER/category/`c_summary`) and
    one command that runs it end to end. — Work item 11.

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

**Why**: `setup.py` and every `from_pretrained` call site in `src/pipeline.py`
resolve each model by repo name only (`SENTIMENT_MODEL`, `NER_MODEL`,
`CATEGORY_MODEL`, `SUMMARY_MODEL`, `src/pipeline.py:55-63`) — an upstream
Hugging Face update to any of the four repos changes results silently,
with no signal, undermining the `SPEC.md` §9 accuracy baseline.

**Current call sites** (9 total, all name-only, no `revision=`):

| Model | `src/pipeline.py` |
|---|---|
| `SENTIMENT_MODEL` | lines 150–151 (tokenizer + model) |
| `NER_MODEL` | lines 283–284 |
| `CATEGORY_MODEL` | lines 506–507 |
| `SUMMARY_MODEL` | lines 747–748 and 785–786 (loaded twice — company-summary and sector-summary stages each load it independently) |

Plus `src/setup.py`'s `download_models()`, which calls
`snapshot_download(repo_id)` / `AutoConfig.from_pretrained(repo_id)` for
all four, also name-only.

**Approach**:

1. Add a commit SHA alongside each model name — e.g. a
   `(name, revision)` pair per constant in `src/pipeline.py`, or a small
   `MODEL_REVISIONS: dict[str, str]` keyed by the existing name constants
   — record today's actual HEAD commit SHA for each of the four HF repos
   as the pin (fetch each via the HF API/UI at pin time, don't guess).
2. Pass `revision=` to **every** `from_pretrained` call above and to
   `snapshot_download`/`AutoConfig.from_pretrained` in `setup.py` — pinning
   only `setup.py`'s pre-download and leaving `pipeline.py`'s
   `from_pretrained(name)` unpinned would not actually fix anything (HF's
   local cache resolution isn't guaranteed to serve the pinned snapshot
   for an unpinned call).
3. Document the pin (a short comment next to each revision, or a table in
   `docs/modules/news-nlp.md`) so bumping a pin later is a deliberate,
   reviewed, one-line diff — not a silent drift.

**Acceptance criteria**:

- `grep -rn "from_pretrained\|snapshot_download" src/` shows a `revision=`
  argument at all 9+ call sites, no exceptions.
- `uv run python -m setup` still succeeds and caches the pinned revisions.
- `uv run pytest` stays green (models are monkeypatched in tests, so this
  is a no-op for the hermetic suite — confirms the pin didn't change any
  import-time behavior the tests exercise).
- A follow-up note in `SPEC.md` §13 item 4 marking the reproducibility
  half resolved (leave the item number in place per this repo's
  no-renumbering convention; strike through or annotate, don't delete).

**Out of scope for this work item**: re-running the pipeline against
historical data to check whether the *current* unpinned HEAD differs from
the newly-pinned SHA. If they differ, that's a separate, larger
conversation (which output is "correct"?) — not silently resolved here.

**Executed (2026-09-14)**: `MODEL_REVISIONS: dict[str, str]`
(`src/pipeline.py`, keyed by the existing `SENTIMENT_MODEL`/`NER_MODEL`/
`CATEGORY_MODEL`/`SUMMARY_MODEL` name constants) added, with each SHA
fetched from the HF Hub API (`GET /api/models/<repo_id>`) at pin time —
`gamug/FinBERT-financial-news` (the current, fine-tuned sentiment
model — this table's own model name had gone stale in the process,
still saying `ProsusAI/finbert`), `gamug/sec-bert-finer-ord-ner`,
`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, `sshleifer/
distilbart-cnn-12-6`. `revision=` passed at all 8 `from_pretrained`
call sites in `src/pipeline.py` (down from the originally-scoped 9 —
`sector_summary`'s own model load was removed entirely by the same-day
`intro_text` fix, not missed) and both calls in `src/setup.py`'s
`download_models()`. `test_setup.py` rewritten to assert the revision
is passed through to both `snapshot_download`/`AutoConfig.from_pretrained`
calls. Full suite (230 tests), ruff, mypy all pass. `SPEC.md` §13 item 4
/ §14 updated; exact SHAs documented in `docs/modules/news-nlp.md`'s new
"Model pins" table.

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

**Why**: `merge_bio_predictions` (`src/pipeline.py`) got a word-boundary-
aware fix on 2026-09-10 — only a word's first WordPiece subword can now
open/close/redirect a span, closing off the bogus single-token entities
(e.g. `"3"`/`ORG` split off `"3M"`) that the training/inference asymmetry
in `train_ner.py`'s subword masking was producing. A narrow
`_MIN_ENTITY_TEXT_LEN` length filter was added at write time as a
last-resort net. Neither change has a post-fix accuracy number: the
`micro_f1` 0.7418 / `hallucination_rate` 0.338 baseline in
`docs/evaluation.md` (2026-09-08) predates both, and the fix is
future-runs-only (the existing 17.6M-row `article_entities` table was left
untouched by design).

**Status as of 2026-09-12** (see `docs/evaluation.md`'s dated follow-ups):
**all four steps are done.** The full-article-vs-lead-cap mismatch was
confirmed against real data, then fixed the same day (`ner` added to
`sampling._UNCAPPED_STAGES`, regression-tested). The bulk-reprocessing
question (step 3) was resolved via its lighter alternative (T-025, not
T-022): `article_entities` was versioned (renamed to `article_entities_v1`,
nothing deleted) and a random, seeded 20,000-article sample reprocessed
under the fixed code (714,334 entity rows across 19,988 articles). Step 1
then ran the same day against that pool: `micro_f1` 0.7418→0.8578,
`hallucination_rate` 33.8%→16.0% (n=8000, T-020) — recorded in
`docs/evaluation.md` and `SPEC.md` §9 (step 4, T-023). The only remaining
thread is T-022 (full-corpus backfill of the ~439K articles the T-025
resample didn't cover), which is an open, non-blocking maintainer scope
call, not a defect in this work item.

**Approach**:

1. Run a fresh `--stage ner` eval against the 19,988 post-fix articles, at
   a sample size comparable to the 2026-09-08 baseline (n=1000), with
   `--seed` for reproducibility. **Done 2026-09-12** — ran at n=8000 (8x
   the baseline, a 40% draw of the resample pool rather than the
   baseline's ~0.2% draw of the full corpus, since the resample pool is
   all the post-fix data there is): `micro_f1` 0.7418→0.8578,
   `hallucination_rate` 33.8%→16.0%. Full table: `docs/evaluation.md`'s
   2026-09-12 "T-020 executed" follow-up.
2. While that data exists, also check the "suspected but unverified" note
   in `docs/evaluation.md`: `run_ner_stage` scores the *whole* article, but
   whether the eval judge's sampling matches that scope (vs. the
   lead-chunk-only mismatch already confirmed and fixed for sentiment) has
   never been empirically checked for NER specifically. **Done and fixed
   2026-09-12** — confirmed (21.4% of a real sample exceeds the judge's
   cap; 16.8% of predicted entities start past it), then `ner` added to
   `sampling._UNCAPPED_STAGES` the same day, before step 1's eval run, so
   that run won't be immediately stale.
3. Bring the question of a bulk `article_entities` re-extraction to the
   maintainer as a scope decision — not something to do unilaterally,
   consistent with how the category hierarchical-classifier migration
   handled the same "future-runs-only" trade-off (SPEC.md §13 item 6).
   **Resolved 2026-09-12** via T-025, the lighter alternative: version the
   table (rename, not delete) and reprocess a random 20,000-article sample
   rather than the full ~459K-article corpus. The remaining ~439,000
   pre-fix articles (now in `article_entities_v1`) are an open, no-longer-
   blocking question — a full backfill (T-022) can still happen later if
   the maintainer wants full-corpus coverage.
4. Record the result as a dated follow-up in `docs/evaluation.md` (append,
   don't overwrite the baseline) and update `SPEC.md` §9's NER row. **Done
   2026-09-12** — see `docs/evaluation.md`'s "T-020 executed" follow-up and
   `SPEC.md` §9's ner row.

**Acceptance criteria** (all done, 2026-09-12):

- A fresh eval run's `micro_f1` / `hallucination_rate` / per-type F1 are
  logged to MLflow and `docs/evaluation.md`, comparable to the 2026-09-08
  baseline. **Done**: `micro_f1` 0.8578, `hallucination_rate` 0.1597,
  `f1_ORG`/`f1_LOC`/`f1_PER` 0.8117/0.8807/0.9262 (mlflow `329f9222`).
- The full-article-vs-lead-cap sampling question is either confirmed (and
  the sampling cap fixed, mirroring the sentiment fix) or explicitly ruled
  out with evidence — not left as an open "suspected" note indefinitely.
  **Done 2026-09-12: confirmed, then fixed.**
- `SPEC.md` §9 updated with the new baseline row/date. **Done.**

## Work item 4 — Sentiment: close the entity/net-signal reasoning gap (resolved 2026-09-13: fine-tuned + chunk-level selected and merged)

**Why**: Two rounds of measurement-side improvement already shipped
(the text-scope fix, then stratified sampling + the `recall_negative`
headline switch — both in `docs/evaluation.md`'s 2026-09-08/09
follow-ups), and they worked as designed: `recall_negative` sits around
0.62-0.78 depending on run. But `negative` precision is still only 0.359
in the latest pilot (eval_run 18, n=800) — the diagnosed root cause is
that FinBERT's whole-article softmax average has no per-company or
net-signal reasoning mechanism, while the judge applies both. That gap was
explicitly flagged as "a pipeline-level follow-up (out of scope here) —
not implemented" and still isn't. This is the "pending to improve, even
after changes" stage: the changes made so far improved *measurement*, not
the *model*.

**Approach** (superseded 2026-09-13 — see "Executed" below; kept for
provenance):

1. Make an explicit design decision among (at least) three candidates,
   rather than defaulting to the first one tried:
   - Entity-scoped re-scoring: use `article_entities` to narrow FinBERT's
     input to company-relevant spans/sentences instead of the whole body.
   - A different or fine-tuned sentiment model with company-aware framing
     closer to the judge's own reasoning.
   - An explicit net-signal heuristic layered on top of the existing
     chunk-averaged score (e.g. down-weighting chunks with no company
     mention) — cheapest to try, least likely to fully close the gap.
2. Before evaluating any change, get a stable floor-sized baseline: only
   one stratified pilot (n=800) exists so far, and `docs/evaluation.md`'s
   own "Sample-size floor" section recommends ~1,800-2,200 for a
   regression-tracked number. Run that first, `--seed`-pinned.
3. Re-solve the sample-size-floor purity estimates using this run's actual
   measured per-stratum agreement (today's numbers are planning
   estimates, explicitly flagged as such) before locking in a permanent
   `--sample-size` default for future sentiment regression runs.
4. Implement the chosen design, re-run the eval, and record the result as
   a dated follow-up in `docs/evaluation.md` plus an update to `SPEC.md`
   §13 item 1 and §9's sentiment row.

**Executed (2026-09-13)**: rather than picking one candidate up front, all
three were actually built and measured head-to-head on the *same* 2,000
-article pool (seed=1), after first fixing a sampling bug and diagnosing
a separate ticker-collision data-quality bug (see `docs/evaluation.md`'s
2026-09-13 follow-up for both):

- **(a) Chunk-level entity-scoped re-scoring** (PR #42,
  `feat/sentiment-entity-scoped`): `chunk_text` + `article_entities`
  gate each chunk's contribution to the company's score
  (`_SENTIMENT_SUBJECT_WEIGHT`/`_SENTIMENT_BASELINE_WEIGHT`). Result:
  `recall_negative` 0.856, `precision_negative` 0.376, `macro_f1_vs_judge`
  0.625. Base FinBERT, unchanged weights.
- **(b) Title-only scoring** (PR #43,
  `feat/sentiment-title-only`, branched fresh off master per explicit
  instruction not to merge (a) if this was untried): score only the
  headline. Result: `recall_negative` dropped to 0.533 (the headline
  alone loses too much signal), `precision_negative` 0.484,
  `macro_f1_vs_judge` 0.620 — not a win.
- **(c) Fine-tuned FinBERT + chunk-level weighting** (this branch,
  `feat/finbert-financial-news-finetune`): continued fine-tuning of
  `ProsusAI/finbert` on 5,000 sentences LLM-labeled (DeepSeek,
  investor/price-impact framing matching Financial PhraseBank
  convention) from the same article pool touched by prior sentiment
  evals, stratified 80/10/10 split, `Trainer` continued-training for 4
  epochs. Held-out test set: accuracy 0.813, macro F1 0.774. Wired into
  the same chunk-level entity-scoped aggregation as (a) and re-run on the
  same 2,000-article pool: `recall_negative` 0.812, `precision_negative`
  **0.505**, `f1_negative` 0.623, `macro_f1_vs_judge` **0.737**,
  `agreement_rate` **0.697**, `recall_positive` **0.790**. Published to
  Hugging Face Hub as `gamug/FinBERT-financial-news` (model card includes
  training data/procedure and both the held-out and downstream-pipeline
  metrics). Full table: `docs/evaluation.md`'s 2026-09-13 follow-up.

None of the three has been merged into `master`/wired into
`SENTIMENT_MODEL` yet. (c) is the strongest result on every headline
metric except `recall_negative` (0.812 vs (a)'s 0.856), and was
documented as such rather than silently picked as "the" answer.

**Decision (2026-09-13): (c) selected as the production candidate**,
after one more real round of investigation into why `precision_negative`/
`precision_positive` still sat around 0.50-0.65 post-idiom-fix. Diagnosis
first: a confusion-matrix read of eval_run 30 found 40.6% of directional
predictions were false alarms on judge-neutral articles (vs. only 7.6%
genuine positive↔negative flips), and reading the judge's own rationale
text for all 435 such cases found 89% were multi-company/mixed-signal
roundup articles — a document-structure/aggregation limitation, not a
sentence-classification error. Three candidate fixes were tested on real
data and rejected (confidence threshold, subject-chunk-coverage gate,
zero-shot "realized vs. speculative" materiality gate reusing the
category stage's model) — none discriminated false alarms from correct
predictions above chance, confirming this is the same net-signal
reasoning gap flagged at the start of the whole investigation. (b),
title-only scoring, was then re-tested with the fine-tuned model as a
genuine alternative (it sidesteps aggregation entirely): it wins on
precision (0.619/0.670 negative/positive) and overall agreement (0.746),
but chunk-level wins recall on **both** directional classes (0.808/0.801
vs. title-only's 0.574/0.528) — a real frontier, not a dominated design.
(c) was chosen because this pipeline is deliberately **pessimistic** — a
missed real story (false negative) is an unrecoverable blind spot for a
downstream SEMANTIC score/knowledge graph, while a false alarm is
something a downstream consumer can still discount. This generalizes the
project's existing "recall over precision" reasoning (previously argued
for the negative class alone) to both directional classes, since (c) is
the only design with strong recall on both. Full comparison and rejected-
fix evidence: `docs/evaluation.md`'s 2026-09-13 follow-up. **Merged
(2026-09-13)**: (c) is wired into `src/pipeline.py` — `SENTIMENT_MODEL`
now `gamug/FinBERT-financial-news`, `run_sentiment_stage` doing
chunk-level entity-scoped weighting (`_text_mentions_subject`/
`_sentiment_chunk_weights`, cherry-picked from `feat/sentiment-entity-
scoped`/PR #42's final chunk-level revision), `db.fetch_pending_
sentiment_articles` added for the `(company, ticker)` fetch. Verified
end-to-end against real production data (not just the hermetic test
suite): a live smoke test on 3 previously-unscored articles loaded the
model on CUDA and wrote correct results, including the exact "A"/"ON"
ticker-collision cases this investigation found earlier. Full test suite
(225 tests, up from 214 — PR #42's `test_sentiment_pipeline.py` and
`test_schema.py` additions came along), ruff, and mypy all green.

**Known limitation, disclosed not hidden — then fixed the same day**: a
manual spot-check of the published fine-tuned model found it still
mislabeled an idiomatic sentence ("...crushed earnings.") as negative —
the same idiom-recognition gap that motivated fine-tuning in the first
place. Rather than leave this as an accepted gap, it was diagnosed
(mining showed the original 5,000-sentence draw contained almost none of
this idiom family by chance — a coverage gap, not a labeling error) and
fixed: 900 more sentences mined from the full corpus, correctly split by
the idiom's two-directional polarity ("stock got crushed" = negative vs.
"crushed estimates" = positive — ruling out a lexicon-override
shortcut), 100 held out as a never-trained probe. Result: probe accuracy
0.750→0.870, with the exact original bug case now correct
("crushed earnings" → positive, 0.935 confidence), and the downstream
2,000-article pipeline comparison holding steady (every metric within
±0.01 of the pre-fix version). Model updated in place at the same Hub
repo. Full account: `docs/evaluation.md`'s 2026-09-13 "crushed earnings
idiom gap" follow-up.

**Acceptance criteria**:

- A design decision is made and documented (which candidate, and why —
  same style as the "Why recall, not F1" / "Why precision, not recall"
  write-ups already in `docs/evaluation.md`). **Done 2026-09-13**: (c),
  chunk-level + fine-tuned FinBERT, selected as the production candidate,
  with the "pessimistic, strong recall on both directional classes"
  rationale documented above and in `docs/evaluation.md`. **Merged into
  `src/pipeline.py` the same day** — see above.
- A floor-sized (~1,800-2,200), seeded baseline run exists before any
  before/after comparison is drawn. **Done** — all three candidates were
  compared against the same 2,000-article pool (seed=1).
- The chosen change measurably improves `negative` precision (or another
  explicitly-justified metric) without collapsing `recall_negative` below
  its current range, confirmed via a post-change eval run. **Done** for
  candidate (c): `precision_negative` 0.376→0.505, `recall_negative`
  stays in-range at 0.812.
- `SPEC.md` §13 item 1 and §9 updated with the dated result. **Done.**

## Work item 5 — Category: hold the line on the hierarchical fix

**Why**: Already resolved in substance. The hierarchical two-level
classifier + `CATEGORY_CONFIDENCE_THRESHOLD` 0.6 calibration
(`docs/evaluation.md` 2026-09-09 follow-ups, eval_run 19-21;
`docs/category-taxonomy.md`'s "Hierarchical classification"/"Threshold
calibration") took the three weakest leaf slugs from 0.14/0.16/0.33
accuracy to 0.61/0.51/0.53, and `accuracy_vs_judge` from 0.442 to 0.487
post-calibration. `SPEC.md` §13 item 2 and its §14 disposition-table row
still describe the pre-calibration numbers and treat this as an accepted,
unaddressed limitation — that's now stale, not an open question. One real
calibration thread remains open: `other`'s own precision (0.474) means a
downstream consumer can't yet treat `article_category.label == "other"` as
"verified no category."

**Approach**:

1. Documentation-only pass: update `SPEC.md` §13 item 2 and the §14
   disposition table to reflect the resolved state, annotated in place
   (per this repo's no-renumbering convention).
2. Treat `other`'s precision as a low-priority, not-yet-scheduled
   follow-up — likely a `CATEGORY_GROUP_FLOOR` or slug-specific threshold
   move next time enough post-calibration eval data has accumulated to
   retune against, per `docs/evaluation.md`'s own note. No code change is
   part of this work item.

**Acceptance criteria**:

- `SPEC.md` §13 item 2 and §14's disposition table row for item 2 reflect
  the shipped fix and cite the corrected numbers (not the pre-calibration
  ones).
- `other`'s precision gap is captured as a named, low-priority follow-up
  somewhere durable (`SPEC.md` §13 or a new item) rather than dropped.

## Work item 6 — Summarization (`c_summary` + `sector_summary`): validate eval scope, close the coverage gap, and add a lightweight sector-intro check (resolved 2026-09-14 -- new sector_summary gap found and fixed same day: up to 50.2% intro_text hallucination rate)

**Status as of 2026-09-14**: all four steps are **done**. Step 2 (the
`mean_coverage` decision) is resolved as **accept, no fix** — one
candidate (a bigger output-length budget) was tested and rejected on
real data, and the residual gap is accepted as a deliberate
completeness-vs-correctness trade (see the "Decision" below). Step 4
(the `sector_summary` intro-text eval) shipped and immediately found a
real, sizable problem — up to 50.2% of `intro_text` rows contained a
hallucination — and it was **fixed the same day** with a deterministic
template (see the "Executed" notes below and `docs/evaluation.md`'s
2026-09-14 follow-ups).

**Executed (2026-09-14, step 1)**: measured `article_summary` (458,641
rows) joined to real `source.articles.body_text` directly — no LLM calls,
pure sampling-layer arithmetic. **10.0% (45,867 rows)** have `body_text`
past the judge's 6000-char cap, and every one of those is also a
multi-chunk summary (`num_chunks > 1`) — i.e. exactly the population where
`hierarchical_summarize_batch`'s reduce pass synthesizes content the judge
could never fully see. (A broader 28.8% have `num_chunks > 1`, but most of
that gap is the summarizer's own tighter token budget triggering
multi-chunk on shorter bodies, not the char-cap mismatch — the judge-
relevant figure is the 10.0%.) Same structural bug as sentiment
(2026-09-08) and NER (2026-09-12), smaller in magnitude than NER's 21.4%
but still real. Fixed same-day: `c_summary` added to `_UNCAPPED_STAGES`
(`src/news_nlp/eval/sampling.py`), full detail and exact numbers in
`docs/evaluation.md`'s 2026-09-14 follow-up.

**Executed (2026-09-14, step 3 — post-fix re-run)**: ran
`--stage c_summary --sample-size 1000 --seed 1` against the real stores
(`eval_run` 34, mlflow `72cf167d`). `mean_coverage` improved 3.02→3.64/5
(HT-weighted), but `pct_with_hallucination` rose 5.4%→8.5% — likely a
previously-invisible hallucination gap the old cap was masking (real
fabricated content past char 6000 had no visible ground truth for the
judge to confirm against), not yet independently verified by a rationale
audit. The comparison carries a real confound (the num_chunks-tiered
stratification was implemented the same day as the 2026-09-08 baseline
but after it was recorded, so the baseline used the old un-stratified
design) — disclosed explicitly rather than presented as a clean
before/after. Per-stratum breakdown: `num_chunks >= 3` articles score
worst on both metrics (coverage 2.86/5, faithfulness 4.23/5) — sharper,
more targeted evidence than the original baseline's generic "terse/
extractive tendency" framing. `eval_run` 34 is now the post-fix baseline
for future regression tracking; full numbers in `docs/evaluation.md`.

**Executed (2026-09-14, step 2 attempt — rejected)**: tested raising
`SUMMARY_MIN_OUTPUT_TOKENS`/`SUMMARY_MAX_OUTPUT_TOKENS` (56/142 →
100/220, `distilbart-cnn-12-6`'s own untuned stock defaults) via a
matched-pair experiment: 200 of `eval_run` 34's own judged articles,
summaries regenerated with the new settings through the unmodified
production code path, re-judged with the same judge. Coverage moved
+0.20 (3.315→3.515) but concentrated on the already-fine single-chunk
tier (3.81→4.14); the actual weak `chunks >= 3` tier barely moved
(2.93→2.97). Cost: `pct_with_hallucination` more than doubled
(15.0%→35.5%), faithfulness −0.53, conciseness −0.66. **Rejected** — a
bad trade, not a fix; the `chunks >= 3` problem is reduce-pass
information loss, not output-length starvation. Full numbers and the
`article_sentiment_v1` data wrinkle this experiment surfaced in
`docs/evaluation.md`'s 2026-09-14 follow-up.

**Decision (2026-09-14, step 2 resolved)**: accept `mean_coverage` as a
deliberate completeness-vs-correctness trade, no further fix attempted.
Every measured lever for raising coverage trades it against
faithfulness (more allowed/forced output = more room to pad with
invented detail); `mean_faithfulness` (4.78/5) is the stage's strongest
metric and the one that matters most for a *summary* specifically -- an
incomplete-but-accurate summary is a bounded, honest gap, while a
complete-but-fabricated one gives a reader no way to tell which parts
are real. This is the mirror image of sentiment's 2026-09-13 "pessimist
model" recall-over-precision decision: sentiment risks false alarms to
avoid missing a real signal (a missed signal is the unrecoverable
failure there); c_summary risks incompleteness to avoid fabrication (a
fabricated detail is the unrecoverable failure here). Does not close
off a narrower reduce-pass-specific fix (never tested) as a future
candidate if `chunks >= 3` coverage is later judged unacceptable on its
own. Full reasoning in `docs/evaluation.md`'s 2026-09-14 "Decision"
follow-up.

**Executed (2026-09-14, step 4)**: built a dedicated `sector_summary`
eval stage (T-054-T-057) — full population every run (T-054: only 3,628
rows, no sampling machinery needed), faithfulness-only against
`facts_json` grounding (`SectorIntroVerdict`/`judge_sector_summary`/
`aggregate_sector_summary`/`prompts/sector_summary.md`). First run
(`eval_run` 34, n=3628): `mean_faithfulness` 4.05/5, but
**`pct_with_hallucination` 42.2%** — not noise, a confirmed systematic
pattern from reading actual flagged rows: 64% of hallucinations are a
fabricated source attribution (`"...according to CNN.com's weekly
Newsquiz"`, `"...according to analysts"`) the model tacks onto the
sentence, the rest mostly a self-contradicting repeated-percentage
artifact. Root cause: `distilbart-cnn-12-6` is a *news-article*
summarizer, repurposed here on a short synthetic stats seed
(`build_sector_intro_seed`) it was never trained to summarize. This is a
newly-discovered, real, **undecided** gap — recorded as a finding here,
not fixed in the same pass the eval path was built. Full numbers and
example rows in `docs/evaluation.md`'s 2026-09-14 follow-up.

**Executed (2026-09-14, later the same day): the `intro_text`
hallucination gap fixed.** Before designing a fix, ruled out data
staleness as the cause: regenerated the full `sector_summary` table
against fully-current sentiment data and re-ran the eval — got *worse*
(`mean_faithfulness` 3.83/5, `pct_with_hallucination` 50.2%, 70%
attribution-fabrication), confirming the pattern is a property of the
generation step itself, not the underlying numbers. Fix: `build_sector
_intro_seed`'s own output (`src/news_nlp/sector_summary/composition.py`)
is already a complete, fully-grounded sentence — `run_sector_summary
_stage` (`src/pipeline.py`) no longer runs it through `SUMMARY_MODEL` at
all; `intro_text` is now that seed verbatim (through
`clean_generated_text` for whitespace normalization only). Zero
hallucination risk by construction, not mitigation — same principle as
the rest of this stage's cross-company-blending design. This stage no
longer loads a model or touches the GPU at all. `SECTOR_SUMMARY_FORMAT
_VERSION` bumped 2→3 (`src/news_nlp/schema.py`) so the existing 3,444
pre-fix rows self-heal to the new template the next time the stage runs
— this project's existing designed mechanism (FR-006), no separate
backfill script. Verified against real production data (5 rows deleted
and regenerated, output clean) and the full hermetic suite
(`test_summary_pipeline.py`'s `run_sector_summary_stage` tests rewritten
to assert the model is never loaded). Full numbers and the
persistence-check methodology in `docs/evaluation.md`'s 2026-09-14
follow-up.

**Why**: Both summarization tasks run the exact same model
(`SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"`, `src/pipeline.py`,
loaded independently by `run_company_summary_stage` and
`run_sector_summary_stage`, same `hierarchical_summarize_batch` call, same
generation settings), but today only one of them has any evaluation at
all.

`c_summary` is the strongest stage on its headline metric
(`mean_faithfulness` 4.87/5, `pct_with_hallucination` only 5.4%), but its
weakest metric, `mean_coverage` (3.02/5), reflects a terse/extractive
tendency of `distilbart-cnn-12-6` — a real completeness gap, even if not
a correctness one (`docs/evaluation.md`'s 2026-09-08 baseline notes).
Separately, `docs/evaluation.md` flags `c_summary` (alongside NER) as
"suspected of the same full-article-vs-lead-cap mismatch... but this has
not been empirically investigated" — `run_company_summary_stage`'s
hierarchical reduce processes the *whole* article, but whether the eval
judge sees that same scope has never been checked the way it was for
sentiment (where the mismatch was confirmed and fixed).

`sector_summary` itself is correctly out of scope for eval — it's
deterministic composition (FR-005), and `docs/evaluation.md`'s "What it
evaluates" section says so explicitly. But its one `intro_text` sentence
*is* model-generated (the same model, run a second time), and has **zero**
evaluation today — not even the "simple" kind. Because the model only
ever sees a small, stats-only `facts_json`-derived seed for this call
(never raw company/article text — see `run_sector_summary_stage`'s own
comment on why: "so it has nothing to blend across companies or
categories"), a full LLM-as-judge stage like `c_summary`'s (coverage,
conciseness, article-length grounding) would be overkill for one
sentence. What's missing is something much narrower: a faithfulness-only
check that `intro_text` doesn't state anything unsupported by its own
`facts_json` grounding.

**Approach**:

1. Empirically check the suspected full-article-vs-lead-cap sampling
   mismatch for `c_summary` specifically — same investigation as Work
   item 3's step 2 for NER, applied to the summarization judge/sampling
   path (`src/news_nlp/eval/sampling.py`, `src/news_nlp/eval/prompts/
   c_summary.md`).
2. Make an explicit decision on `mean_coverage`: raise
   `SUMMARY_MIN_OUTPUT_TOKENS`/`SUMMARY_MAX_OUTPUT_TOKENS`
   (`src/pipeline.py`, currently 56/142), change the hierarchical-reduce
   strategy, or explicitly accept the terse tendency as a deliberate
   trade for the already-strong faithfulness score. Not yet decided —
   don't default to the first lever tried.
3. Re-run the `c_summary` eval after any change (or after
   confirming/ruling out the sampling mismatch) and record a dated
   follow-up in `docs/evaluation.md`, same append-only pattern as the
   other stages.
4. Add a **lightweight, faithfulness-only** eval path for
   `sector_summary`'s `intro_text` — narrower than a full new
   `news_nlp.eval` stage:
   - Judge input is `intro_text` + its own `facts_json` (the seed the
     model actually saw), never the underlying article/company text —
     this is a hallucination check against the model's real grounding,
     not an independent accuracy claim.
   - A single metric (e.g. `pct_with_hallucination` or a 1-5
     faithfulness score, reusing the `c_summary` judge's rubric shape
     rather than inventing a new one) — no coverage/conciseness scoring;
     those don't meaningfully apply to one sentence.
   - Population size is naturally small (one row per
     `(gics_sector, gics_sub_industry, week)`, not per-article), so this
     can likely run against the **full population** each time rather than
     needing `sampling.py`'s stratified-sampling machinery — confirm the
     actual row count before assuming this, don't guess it.
   - This is new eval surface, not a variant of an existing one:
     `sampling.STAGES`, `metrics.HEADLINE`, a new prompt file
     (`src/news_nlp/eval/prompts/sector_summary.md`), and
     `cli/news_nlp_eval.py`'s `--stage` choices all need a
     `sector_summary` case added.

**Acceptance criteria**:

- The full-article-vs-lead-cap sampling question is either confirmed
  (and the sampling cap fixed) or explicitly ruled out with evidence —
  same bar as Work item 3's NER acceptance criterion.
- A documented decision on whether/how to raise `mean_coverage`, with
  before/after numbers if a change is made.
- A working `--stage sector_summary` (or equivalent) eval path exists,
  producing at least one faithfulness/hallucination metric for
  `intro_text` against `facts_json`, logged the same way the other
  stages log to MLflow/`eval_run`.
- `SPEC.md` §13 item 10 (new) and §9's `c_summary` baseline row updated
  with the result; a new §9 row (or a documented decision not to add
  one) for the `sector_summary` intro check.

**All four criteria met, 2026-09-14** — sampling mismatch confirmed and
fixed; `mean_coverage` decision made (accept); `--stage sector_summary`
working and run against the full population; `SPEC.md` updated (§9 new
row, §13 new item for the 42.2% hallucination-rate finding).

## Work item 7 — NER: develop batch processing (code done 2026-09-12; T-062 needs a real GPU)

**Status as of 2026-09-12**: steps 1-4 and 6 are **done** — `pipeline._ner_batch`
implements the flatten/tokenize/forward/regroup approach below exactly as
scoped, `NER_BATCH_SIZE = 8` (starting value, see step 5), and the parity
test (step 6) passes against the full 206-test suite (ruff/mypy clean).
Steps 5 and 7 (empirical tuning + real throughput measurement) are **not
done** — they need the project's actual GPU, which the environment this
shipped from doesn't have (CPU-only sandbox, no access to the production
DB either). `NER_BATCH_SIZE=8` is an untested starting guess carried over
from `CATEGORY_BATCH_SIZE`, not a measured value — running T-062 on the
real hardware is the only remaining step in this work item.

**Why**: `run_ner_stage` (`src/pipeline.py`) sends one chunk through the
model per forward pass — the only stage with no batching. `run_category_
stage` pools `CATEGORY_BATCH_SIZE=8` articles' premise/hypothesis pairs
into one call; the summarization stages batch `SUMMARY_BATCH_SIZE=4`
articles via `hierarchical_summarize_batch`. This isn't a documented
trade-off anywhere in the codebase — genuinely unaddressed, not a
deliberate design choice (SPEC.md §13 item 11). It has a real, now-measured
cost: T-025's 2026-09-12 resample processed 20,000 articles unbatched in
~15 minutes (~22 articles/sec) on the project's GPU (RTX 4050 Laptop, 6GB
VRAM) — a full-corpus backfill of the ~439,000 remaining pre-fix articles
(§13 item 6's open T-022 question) would take roughly 5.5x that, over 5
hours, on hardware that had headroom to go faster the whole run.

**Approach**:

1. Batch by article (a new `NER_BATCH_SIZE` constant, same naming
   convention as `CATEGORY_BATCH_SIZE`/`SUMMARY_BATCH_SIZE`): for each
   batch of `NER_BATCH_SIZE` pending articles, run `chunk_text` per
   article as today, but flatten every article's chunks into one list
   tagged with the chunk's owning article index, instead of looping
   articles one at a time. **Done** — `pipeline._ner_batch`.
2. Tokenize that flattened chunk list in **one** call with `padding=True`
   (`return_tensors="pt"`, `return_offsets_mapping=True`) instead of one
   `tokenizer(...)` call per chunk — the batch dimension becomes chunk
   count within the article batch, not article count. **Done.**
3. One forward pass over the padded batch. `merge_bio_predictions` itself
   needs **no change**: HF's fast tokenizers already return `word_ids() ==
   None` for padding positions, the same sentinel the function already
   uses to skip special tokens — so calling it once per chunk (sliced out
   of the batched output via `batch_index=i`) after the batched forward
   pass, exactly as today's per-chunk call already does, should just work.
   **Done, confirmed by the parity test in step 6** — `merge_bio_predictions`
   is untouched.
4. Regroup chunk-level entity spans back to their owning article via the
   tagging from step 1, same offset math (`ch.start_char + e["start_char"]`)
   already used today. **Done.**
5. Tune `NER_BATCH_SIZE` empirically against the 6GB VRAM budget (SPEC.md
   NR-001) — don't copy `CATEGORY_BATCH_SIZE`/`SUMMARY_BATCH_SIZE`'s values
   blind; NER's per-chunk sequence length (up to 512 tokens) and variable
   chunks-per-article shape a different memory profile than either. **Not
   done** — shipped with `NER_BATCH_SIZE = 8` (`CATEGORY_BATCH_SIZE`'s
   value) as an explicitly-untested starting guess; needs the project's
   real GPU to tune for real (T-062).
6. Add a **parity test** before anything else ships: the same fixture
   articles processed through the batched path must produce byte-identical
   `article_entities` rows (same entities, offsets, scores) as today's
   unbatched path. A batching refactor that silently changes results would
   be worse than not batching at all. **Done** —
   `test_batched_and_per_article_ner_processing_produce_identical_entities`:
   two different-length articles run once at `NER_BATCH_SIZE=1` and once
   at `NER_BATCH_SIZE=2` (forcing real cross-article padding) produce
   byte-identical entities; full 206-test suite green, ruff/mypy clean.
7. Measure the real throughput improvement (articles/sec) on a real
   sample — the same kind of resample T-025 already ran is a natural
   before/after comparison — rather than assuming batching helps without
   measuring it. **Not done** — same GPU/production-DB blocker as step 5
   (T-062).

**Acceptance criteria**:

- A parity test passes: batched and unbatched processing of the same
  input produce identical `article_entities` output. **Done.**
- A measured (not assumed) throughput improvement on a real sample, with
  the before/after numbers documented. **Not done — needs T-062 on the
  real GPU.**
- No VRAM regression at the chosen `NER_BATCH_SIZE` against SPEC.md
  NR-001's 6GB budget. **Not done — same T-062 blocker; `NER_BATCH_SIZE=8`
  is unverified against real hardware.**
- The new constant and batching design documented (`docs/modules/
  news-nlp.md` and/or a comment in `pipeline.py`, matching how
  `CATEGORY_BATCH_SIZE`/`SUMMARY_BATCH_SIZE` are documented today). **Done.**
- `SPEC.md` §13 item 11 updated to reflect the shipped state. **Done.**

**Out of scope for this work item**: `run_sentiment_stage` has the
identical unbatched shape and is likely worth the same treatment later,
but doing it isn't part of this item — raised only as a one-line note in
SPEC.md §13 item 11, not its own numbered question, to avoid scope creep
beyond what was asked.

## Work item 8 — Justify each model's selection in the Models evaluation artifact section (priority, pending)

**Why**: The 2026-09-14 [Portfolio NLP artifact](https://claude.ai/code/artifact/65e62819-28dd-495b-b6ec-64f9c1751235)
reorganization (`docs/evaluation.md`'s numbers, centralized) closed the
"where are this model's metrics" question — every stage's accuracy now
lives in one "Models evaluation" section instead of being scattered
across the page. It did **not** close a different, related question:
*why this specific model/architecture, as opposed to a plausible
alternative*. Today the page states what each stage uses and how well it
performs, but the actual selection reasoning is either absent (category,
NER, `c_summary`'s base model choice) or scattered across
`docs/evaluation.md`'s dated follow-ups and only covers a narrower
question — *which variant of the same model family* (chunk-level vs.
title-only FinBERT; base vs. fine-tuned) — not *why this model family at
all* (why FinBERT-shaped, why zero-shot NLI, why SEC-BERT, why
distilbart, why no model for `sector_summary`).

**Approach**: add one "Why this model" sub-block to each of the five
existing per-model blocks already in the artifact's `#eval` ("Every
model's real accuracy, in one place") section — not a new top-level
section, a sub-section within each existing one, so the justification
sits right next to the numbers it explains. Each sub-block should cover,
at minimum:

1. **Sentiment** (`gamug/FinBERT-financial-news`) — why a FinBERT-family
   model (domain-pretrained on financial text) over a general-purpose
   sentiment model or a from-scratch train; why *continuing* fine-tuning
   from `ProsusAI/finbert` specifically rather than starting from a
   generic checkpoint (`bert-base`) or reaching for a larger general LLM;
   why chunk-level + entity-scoped weighting over the sentence-level,
   title-only, and whole-document-average alternatives actually measured
   — this last part is already well-documented in `docs/evaluation.md`'s
   2026-09-13 follow-ups and mostly needs pulling into the new sub-block,
   not fresh research.
2. **Category** (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`) — why
   zero-shot NLI at all (no labeled category training set exists, and
   one is expensive to build for a 10-slug taxonomy that itself may
   change) rather than a trained classifier; why this specific
   zero-shot-NLI checkpoint over other zero-shot options (DeBERTa-v3's
   disentangled-attention architecture and its own zero-shot-NLI
   benchmark provenance — needs sourcing, not yet written anywhere in
   this repo); why the hierarchical two-level taxonomy design over a
   flat 9-way classification (already documented, 2026-09-09 follow-up
   — pull in, don't re-derive).
3. **NER** (`gamug/sec-bert-finer-ord-ner`) — why SEC-BERT (pretrained on
   SEC filings) as the base checkpoint over a generic NER model
   (spaCy, `bert-base-NER`) — financial entity mentions (tickers, filing
   terminology) benefit from domain vocabulary a generic model never
   saw; why FiNER-ORD specifically as the fine-tuning dataset (what it
   is, why its label set fits `PER`/`LOC`/`ORG` here — needs sourcing).
4. **`article_summary` / `c_summary`** (`sshleifer/distilbart-cnn-12-6`)
   — why a *distilled* BART over full `bart-large-cnn` or a modern
   LLM-based summarizer: the 6GB-VRAM / one-model-at-a-time budget
   (`SPEC.md` NR-001) this whole pipeline is built around, and no
   per-call API cost for a batch job over hundreds of thousands of
   articles. This sub-block should **explicitly connect the model choice
   to the `mean_coverage` weakness already documented** — a smaller
   model pretrained on short CNN/DailyMail-style news, never retuned for
   longer/denser financial text, is a plausible root cause behind the
   coverage gap the 2026-09-14 "accept as a trade" decision lives with,
   not an unrelated fact sitting next to it.
5. **`sector_summary`** — this one's justification is already the
   strongest and most complete of the five (this session's own
   2026-09-14 investigation): explicitly explain *why removing the model
   entirely* was the right call, not a downgrade — a deterministic
   template can't fabricate a source or contradict its own numbers *by
   construction*, the same "structural guarantee over probabilistic
   mitigation" principle already used for this stage's cross-company-
   blending design. Mostly a matter of framing the existing "no longer a
   machine learning model" content as an explicit selection decision
   rather than an incidental fact.

**Acceptance criteria**:

- Every one of the five per-model blocks in the artifact's `#eval`
  section has a clearly labeled "Why this model" sub-section, distinct
  from its metrics.
- Each justification names at least one concrete alternative that was
  *not* chosen and says why — a justification that doesn't name a
  rejected alternative isn't a justification, it's a description.
- Sourcing for claims about a model's own training/architecture
  provenance (DeBERTa-v3's zero-shot-NLI lineage, FiNER-ORD's dataset
  scope, `distilbart-cnn-12-6`'s own training corpus) is a real citation
  (the model card, the dataset paper/repo), not an assertion invented to
  fill the section.
- `docs/modules/news-nlp.md` gets the same justification content in
  prose form (the artifact is a presentation layer over the repo's own
  docs, per this project's constitution — it should never say something
  the docs don't already say).

**Out of scope for this work item**: re-litigating any already-made
selection decision (e.g. reopening whether chunk-level + fine-tuned
sentiment was the right call) — this item explains decisions already
made, it does not remake them.

## Work item 9 — Rebalance the sentiment fine-tuning data (priority)

**Why**: the 5,800-sentence training pool behind `gamug/FinBERT-financial-news`
(5,000 base draw + 800 merged idiom-augment sentences) is 3,256 neutral
(56.1%) / 1,321 negative (22.8%) / 1,223 positive (21.1%). That ratio was
never chosen — it's a byproduct of drawing sentences from the eval
harness's confidence-stratified sampling pool (stratified on prediction
confidence, not on label ratio), on top of real financial news skewing
neutral/factual. The only thing currently touching this is class-agnostic:
`train_sentiment.py` picks the best checkpoint by macro F1 (equal
per-class weight at *evaluation* time) and stratifies the
train/validation/test split per label (keeps the splits proportionate to
the source, doesn't rebalance it) — no oversampling of negative/positive,
no undersampling of neutral, no class-weighted loss anywhere in the
training loop.

**Approach**:

1. Downsample the neutral class to representative examples only, sized to
   match the larger of the two minority classes (1,321, negative) —
   keeps every negative/positive example (no minority-class data
   discarded) while bringing neutral back in line, landing close to a
   genuine three-way balance (1,321 / 1,321 / 1,223) without touching the
   other two classes at all. "Representative" means ranked by cosine
   similarity to the neutral class's own TF-IDF centroid, keeping the
   most prototypical examples and dropping the most atypical/outlier
   ones — not a random cut.
2. Publish the rebalanced pool as the training data, replacing the
   published `gamug/FinBERT-financial-news-data` dataset's
   `train`/`validation`/`test` splits (the `idiom_probe` split stays
   untouched — its whole purpose is measuring against real, unfiltered
   idiom-family traffic, not a class-balance concern). Document what
   changed and why directly in the dataset card, not silently.
3. Retrain the sentiment model on the rebalanced data via
   `train_sentiment.py` (same procedure/hyperparameters as the existing
   `ProsusAI/finbert`-based continued fine-tune — this is a data-quality
   fix, not an architecture change), and publish the result to
   `gamug/FinBERT-financial-news` as a new version.
4. Measure and report: per-class precision/recall/F1 on the held-out test
   set, compared directly against the current published model's own
   numbers (already in that model's card) — not just an aggregate
   accuracy/macro-F1 number, since the whole point of this fix is
   per-class behavior.

**Acceptance criteria**:

- The published dataset's `train`/`validation`/`test` splits are
  genuinely closer to balanced across the three classes; the selection
  method (TF-IDF centroid proximity, not random) is documented in the
  dataset card, along with the exact before/after counts.
- The retrained model's per-class precision/recall/F1 (positive/negative/
  neutral) is reported and compared directly against the currently
  published model's own numbers — an honest result, including if some
  metric gets *worse* (e.g. neutral precision, given less neutral
  training data) rather than only reporting improvements.
- `docs/evaluation.md` gets a dated follow-up with the full before/after
  table and methodology, same as every other model-performance change in
  this project.

**Out of scope for this work item**: touching `idiom_probe.jsonl` (stays
exactly as-is, per its own disclosed role); building any general-purpose
class-balancing utility beyond what this one dataset needs; NER's
training data (`gtfintechlab/finer-ord`) — a separate dataset this
project doesn't own or control the composition of.

**Executed (2026-09-14), mostly — steps 1/2/3/4 done, model publish
pending a decision**: the merged training pool (base draw + idiom-augment,
5,800 sentences) confirmed at 3,256 neutral (56.1%) / 1,321 negative
(22.8%) / 1,223 positive (21.1%). `scripts/rebalance_sentiment_data_2026_09_14.py`
downsampled neutral to 1,321 via TF-IDF-centroid cosine similarity
(scikit-learn, already a transitive dependency — no new one added),
verified 0 duplicate sentences and the exact expected counts before
anything left the machine. Republished as v2 of
`gamug/FinBERT-financial-news-data`
(`scripts/publish_finbert_financial_news_dataset_rebalanced_2026_09_14.py`);
`idiom_probe` untouched. `train_sentiment.py` extended (not replaced) with
a `BALANCED_DATA_PATH` branch, preferred when present; retrained on CUDA,
same hyperparameters as the existing fine-tune.

Result is a real trade, not a strict win — full numbers in
`docs/evaluation.md`'s 2026-09-14 follow-up. Test-set negative F1 improves
substantially (0.726→0.829); neutral F1 drops (0.838→0.714), and the
idiom-probe's neutral F1 (a small-n, 10/100-row reading, but a real one)
collapses to 0.0. **The retrained model was deliberately not published to
the Hub yet** — `scripts/publish_finbert_financial_news_v3_2026_09_14.py`
is written and ready, but publishing a model with a disclosed regression
warranted surfacing the numbers first rather than publishing and
explaining after. `src/pipeline.py`'s `MODEL_REVISIONS` was not touched
either way — production still runs the currently-published (v2) model
regardless of what happens with this decision. The downstream,
production-pipeline LLM-judge evaluation (the number that actually
validated v2) has not been run against v3 — a separate, larger step.

**Second experiment, 2026-09-15 (user-requested)**: class-weighted loss
instead of downsampling — train on the full original unbalanced pool
(no sentence discarded) with an inverse-class-frequency-weighted
`CrossEntropyLoss` (`compute_class_weights` + `WeightedLossTrainer` in
`train_sentiment.py`, `--weighted` flag, separate output paths so it
doesn't overwrite v3's artifacts). Because it trains on the full pool, it
evaluates on the exact same test set (n=579) and idiom probe (n=100) as
the published model, unlike v3's smaller rebalanced-pool test set.

Result (referred to as v4; full tables in `docs/evaluation.md`'s
2026-09-15 follow-up): every test-set metric sits within ~0.01 of v2 —
no dramatic negative-F1 win like v3's, but no neutral cost either. The
number that actually matters: idiom-probe neutral F1 lands at 0.471,
essentially identical to v2's 0.47 — **v3's collapse to 0.0 does not
reproduce here**. Class weighting corrects the training signal without
removing the 1,935 neutral sentences v3 discarded, so the model doesn't
lose whatever those harder/less-typical examples taught it. v4 reads as
close to a free lunch on these two eval sets where v3 was a real trade.

Neither v3 nor v4 has been measured against the downstream,
production-pipeline LLM-judge evaluation — still the open step before
adopting either. Neither has been published to the Hub, and
`src/pipeline.py`'s `MODEL_REVISIONS` is untouched by both — the same
"surface the numbers before publishing a candidate with any disclosed
trade-off" reasoning as v3, applied consistently.

**Downstream eval, 2026-09-15 (user-requested)**: v4's production-pipeline
LLM-judge evaluation, the previously-missing step — the same real article
traffic / entity-scoped chunk-level aggregation / LLM-judge methodology
that validated v2 in the first place. Run against a scratch copy of the
results DB (`nlp_.db` copied to `nlp_use.db`; the real, shared file was
never opened for writing) via `scripts/resample_sentiment_v4_2026_09_15.py`
(in-process `pipeline.SENTIMENT_MODEL` monkeypatch to v4's local
checkpoint — `src/pipeline.py` on disk untouched throughout) followed by
`cli/news_nlp_eval.py --stage sentiment --sample-size 2000 --seed 1`, the
documented sample-size floor.

Result (full table in `docs/evaluation.md`'s 2026-09-15 follow-up): v4
delivers on this pipeline's stated priority metric —
`recall_negative` 0.808→0.832 — but `agreement_rate` (0.701→0.674) and
`mean_severity` (0.341→0.369, lower is better) both get worse. A real
trade at the level that actually matters, not a clean win at either level
measured so far. `src/pipeline.py`'s `MODEL_REVISIONS` remains untouched,
still pinning v2; adopting v4 now would mean deliberately trading overall
agreement/severity for negative recall, a decision this evaluation
surfaces rather than makes.

## Work item 10 — Formalize the pipeline/evaluation architecture: FTI restructure + evaluation redesign (done 2026-09-18)

**Why**: `pipeline.py` (four ML stages) and `news_nlp/eval/` (the
LLM-as-judge harness) were never designed against a stated architectural
pattern — each grew as a pragmatic response to a real, immediate necessity
(a missing accuracy baseline, a sampling bias, a class-imbalance fix),
session by session, Work item by Work item. The result works and has been
validated end-to-end, but it's four independent sets of module-level
functions rather than a shared, extensible structure, and this project's
own recent sentiment-candidate work (Work item 9) surfaced concrete,
recurring friction in the eval harness specifically:

- Comparing v2/v3/v4/v5 required a **separate scratch database copy per
  candidate** (`scripts/resample_sentiment_v{3,4,5}_2026_09_15.py`) because
  nothing in the schema lets two experiments' judged rows for the same
  article coexist.
- No confusion matrix is ever persisted — only aggregate metrics
  (`metrics_json`), so a specific misclassification pattern can't be
  queried after the fact without re-deriving it from raw judge verdicts.
- No ROC/AUC is computed anywhere in this codebase, for any stage.
- Re-evaluating the same articles under a repeat invocation always spends
  fresh judge-LLM calls, even when nothing about that `(article, task,
  experiment)` combination has changed since it was last judged.

**Approach**, six parts (SPEC.md FR-011–FR-016):

1. **FTI class hierarchy across all four ML stages** (sentiment, NER,
   category, `c_summary`). One shared abstract base per component:
   - **Feature** — the chunking/premise-construction/subject-weighting
     logic each stage already has (`chunk_text` calls, category's
     `_category_premises`, sentiment's `_sentiment_chunk_weights`) becomes
     a per-stage subclass of one common `FeatureExtractor`-shaped
     interface (`extract(article) -> FeatureBatch`, naming TBD at
     implementation time).
   - **Train** — wraps `train_sentiment.py`/`train_ner.py`'s existing
     logic behind a common `Trainer` interface. Category and `c_summary`
     ship pretrained/zero-shot as-is today — their `Trainer` subclass is
     an explicit, documented no-op, not an omission or a fake training
     step invented to satisfy the interface.
   - **Inference** — wraps the existing `run_sentiment_stage`/
     `run_ner_stage`/`run_category_stage`/`run_company_summary_stage`
     logic (load model at its pinned revision, predict, write results)
     behind a common `predict(articles) -> Predictions` interface, using
     that stage's own `FeatureExtractor` for pre-processing.

   `run_pipeline` keeps orchestrating stage-by-stage in the same fixed
   order (SPEC.md §3's "fixed pipeline, not a DAG" decision is
   unaffected) — this is an internal restructuring of *how* each stage is
   implemented, not a change to *what* the pipeline does externally.
   **No behavioral regression**: every existing hermetic test in
   `tests/news_nlp/` must pass with its assertions unchanged — only
   import paths/construction calls may need updating where a test
   currently reaches into a stage's internals directly (e.g.
   `pipeline._sentiment_chunk_weights`).

2. **`sector_summary` fully separated from the FTI hierarchy.** It has no
   Feature/Train/Inference shape — no model, no GPU, fully deterministic
   (SPEC.md FR-005, resolved 2026-09-14). `news_nlp/sector_summary/`
   already holds its composition/query helpers; this step finishes the
   separation already mostly in place by moving the orchestration
   entrypoint itself (`run_sector_summary_stage`) out of `pipeline.py` and
   into that module, so `pipeline.py` stops doing double duty as "the FTI
   stages" and "the heuristic stage's orchestrator." `run_pipeline` still
   calls it the same way (SPEC.md FR-012).

3. **New `news_nlp.eval` design reuses the FTI `Inference` classes** from
   step 1 for its own model-scoring step, instead of duplicating
   model-loading/forward-pass code independently the way `news_nlp/eval/`
   relates to `pipeline.py` today. **Based on the current implementation,
   not rebuilt from scratch** — `sampling.py`'s stratified-sampling
   design, `judges.py`'s judge-invocation/repair logic, `verdicts.py`'s
   pydantic schemas, `tracking.py`'s MLflow logging, and `regression.py`'s
   headline-metric comparison all carry over unchanged in spirit; only the
   parts touched by steps 4-6 below (schema/storage, and the inference
   step's own model-loading path) actually change.

4. **Split `eval_judgement` into two tables** — one for the sampled
   **model inference** being evaluated, one for the **LLM judge verdict**
   on it (SPEC.md FR-014). Both carry `article_id` (already present on
   today's `eval_judgement`, but reinforced here as a first-class,
   documented traceback key to the SOURCE `urls.db` article — not
   FK-constrained, same point-in-time-snapshot reasoning as today), a
   `task` column (`sentiment` \| `category` \| `ner` \| `c_summary` —
   redundant with `eval_run.stage` today, but a first-class column here so
   a judge-table query never needs a join back to `eval_run` just to know
   what it's looking at), and an `experiment` column (`base`, `v2`, `v3`,
   `v4`, … — freely chosen per invocation, e.g. via `cli/news_nlp_eval.py
   --run-name`, already threaded through MLflow as of the 2026-09-15
   `--run-name` addition — extended here to also tag the DB rows, not just
   the MLflow run). This is what actually fixes the "separate scratch DB
   per candidate" friction: two experiments' rows for the same article
   coexist in the same table, filtered by `experiment`.
5. **Judge-table reuse mechanism** (SPEC.md FR-015): before invoking the
   judge LLM for a sampled `(article_id, task, experiment)`, check whether
   a verdict already exists for that exact key in the redesigned judge
   table and reuse it instead of re-judging. A unique constraint on
   `(article_id, task, experiment)` (or `(article_id, task, experiment,
   run_id)` if a re-judge under the same key is ever deliberately wanted)
   both enforces this and makes the lookup a single indexed read, not a
   scan.
6. **Confusion matrix + one-vs-rest ROC, sentiment/category only**
   (SPEC.md FR-016). A new table, one row per `(experiment, task,
   true_label, predicted_label)` with a count, populated from the same
   judge verdicts already being recorded — sentiment and category only,
   since NER's error-only verdict contract (`NerVerdict.wrong`/`missed`)
   and `c_summary`'s 1-5 rating scales don't have a discrete-label
   confusion-matrix shape to begin with. `aggregate_sentiment`/
   `aggregate_category` gain `roc_auc_<class>` (one-vs-rest, computed from
   each sampled row's own stored prediction probabilities — already
   present in `article_sentiment`'s `positive`/`negative`/`neutral`
   columns and `article_category`'s 9-way NLI distribution, so no new data
   collection is needed, only new aggregation math) alongside the existing
   `precision_<class>`/`recall_<class>`/`f1_<class>`/`accuracy_ovr_<class>`.
   **NER and `c_summary`'s existing metric sets are explicitly preserved,
   byte-identical** — this step does not touch `aggregate_ner`/
   `aggregate_c_summary`'s returned keys at all.

**Acceptance criteria**:

- One importable FTI base class per component (feature extraction,
  training, inference); sentiment/NER/category/`c_summary` each have a
  concrete subclass of all three; category/`c_summary`'s `Trainer`
  subclass is a documented no-op.
- `run_sector_summary_stage` (or its renamed equivalent) lives under
  `news_nlp/sector_summary/`, imports nothing from the FTI base classes,
  and `pipeline.run_pipeline`'s call into it is otherwise unchanged.
- `news_nlp/eval/`'s inference step for a stage calls that stage's own FTI
  `Inference` subclass — no independent `from_pretrained`/forward-pass
  code duplicated inside `news_nlp/eval/`.
- Two tables exist where `eval_judgement` used to hold both inference and
  verdict in one row; both carry non-null `article_id`/`task`/`experiment`
  on every row; two different `experiment` values against the same
  `article_id`/`task` never collide.
- Re-running the same `--stage <s> --run-name <experiment>` invocation
  against an unchanged sample makes zero new judge-LLM calls the second
  time (hermetic-test-countable); a new `article_id` or a different
  `experiment` always judges fresh.
- A confusion-matrix table exists, keyed by `experiment`, for sentiment
  and category; both stages' stored metrics gain `roc_auc_<class>`;
  `aggregate_ner`/`aggregate_c_summary`'s own metric key sets are
  unchanged (a snapshot/regression test on the exact key set).
- Full hermetic suite stays green throughout — this is an architecture
  restructuring with an explicit no-behavioral-regression bar, not a
  rewrite that's allowed to also change what the pipeline produces.

**Out of scope for this work item**: re-litigating any already-made model
selection or aggregation-design decision (Work items 1-9's own choices);
extending confusion-matrix/ROC treatment to NER or `c_summary`; building a
DAG/task-queue orchestrator (SPEC.md §3's fixed-pipeline decision is
explicitly not being reopened); a UI or dashboard over the new confusion
matrix/ROC data (that data becomes queryable, presenting it is a separate,
later concern if ever wanted).

## Work item 11 — JSON-driven, single-command experiment runs (priority — #1)

**Why**: even with Work item 10's FTI/eval redesign landed, running one
real model experiment still means hand-chaining several independent,
one-off scripts — dataset prep, then `train_sentiment.py` (or nothing, for
a stage with no trainable checkpoint), then an offline test-set/idiom-probe
comparison, then a downstream LLM-judge eval, then (if adopted) a Hub
publish — several of which mutate shared DB tables directly and require a
manual restore step afterward (`scripts/resample_sentiment_v4_2026_09_15.py`
+ `scripts/restore_sentiment_after_v4_eval_2026_09_15.py`). Nothing
declares an experiment's full configuration in one place before it runs;
reproducing one means reverse-engineering the exact command sequence from
`docs/evaluation.md`'s prose or from git history. This friction was
surfaced directly (2026-09-18) walking through the full historical
sentiment-candidate sequence (Work item 9) command by command.

**Approach**, five parts (SPEC.md FR-017):

1. **A JSON `ExperimentSpec` schema**, generic across all four ML stages
   (sentiment/NER/category/`c_summary`) — one file fully specifies an
   experiment: whether it trains a new checkpoint (`pretrain.enabled`,
   which base model, which dataset, train/test split + stratify strategy),
   how many articles the LLM judge samples (`eval.sample_size` +
   stratification, direct passthrough of the existing `EvalSettings`
   shape), the experiment's name, and whether to publish the result to
   the Hub. Strict validation (pydantic, this codebase's own existing
   tool for exactly this) rejects an ambiguous or unpinned config before
   any GPU work starts — reproducibility guarantee, not a security
   mechanism.
2. **Making train/test setup genuinely config-driven.** Today
   `stratified_split()`/`SentimentTrainConfig` hardcode the split seed
   (42, a module constant never threaded from any config), fractions
   (80/10/10), and stratify key (`"label"`) — a real gap, not just
   missing CLI plumbing. Parameterize both, defaults preserving today's
   exact behavior for every existing call site.
3. **One orchestration function, `run_experiment(spec)`**, reused by one
   new CLI entrypoint (`cli/run_experiment.py --config <path>.json` — the
   single command): train (if requested, via a stage→`Trainer` registry
   mirroring `news_nlp.eval.candidate`'s own `_STAGE_CLASSES` pattern) →
   auto-resolve the freshly-trained local checkpoint as the eval
   `candidate_model`/`candidate_revision` (the established `"local"`
   placeholder convention `resample_sentiment_v4_2026_09_15.py` already
   set) → evaluate via `news_nlp.eval.runner.run_eval` **unchanged,
   reused verbatim** → optionally publish → write a git-tracked result
   record (resolved config + metrics + `eval_run_id`/`mlflow_run_id`),
   the new structured complement to `docs/evaluation.md`'s hand-written
   narrative follow-ups.
4. **Close the `NoOpTrainer` wiring gap Work item 10 left half-done**:
   `category_stage.py`/`summary_stage.py` currently only *mention*
   `NoOpTrainer` in their docstrings — neither actually instantiates it
   anywhere outside `fti.py`'s own unit test. Wire a real (trivial)
   `Trainer` into both, so `pretrain.enabled` for these two stages is
   rejected by real code, not just documented as always-inapplicable.
5. **Backfill a JSON spec for every experiment that was actually run**
   (sentiment v2/v3/v4/v5 + the un-fine-tuned base-FinBERT comparison
   arm; one production-config spec each for NER/category/`c_summary`).
   Two gaps disclosed rather than forced to fit: the original
   "title-only" sentiment aggregation arms have no surviving code path
   to run (removed after being rejected — not backfillable); category's
   confidence-threshold calibration and `c_summary`'s generation
   output-length-budget test were both inference-time hyperparameter
   experiments, a different axis than this schema's
   pretrain/dataset/split/sample-count shape — out of scope for v1, named
   as a known gap rather than silently unaddressed.

**Acceptance criteria**:

- A single `ExperimentSpec` JSON, validated by a pydantic schema, can
  fully describe an experiment for any of the four ML stages; an invalid
  or ambiguous spec (unpinned base model, `publish` without `pretrain`, an
  unknown `hyperparameters` key for that stage, `pretrain` requested for
  category/`c_summary`) fails validation before any training/eval work
  starts, with a message naming exactly what's wrong.
- `uv run cli/run_experiment.py --config <path>.json` is the **only**
  command needed to reproduce any backfilled historical experiment's
  train+evaluate sequence (publish excluded — see below).
- `stratified_split()`'s split seed/fractions/stratify key are
  spec-overridable; every existing test and script call site keeps its
  exact current (default) behavior unchanged.
- `category_stage.py`/`summary_stage.py` each have a real `Trainer`
  (however trivial) instantiated somewhere reachable from test coverage,
  not just referenced in a docstring.
- A JSON spec exists for every real historical sentiment/NER/category/
  `c_summary` experiment that has a runnable equivalent today; the two
  disclosed gaps (title-only arms, inference-time-hyperparameter
  experiments) are named in `experiments/README.md`, not silently
  omitted.
- Full hermetic suite stays green; a hermetic (stub-judge) end-to-end test
  proves `run_experiment` actually works for at least one training spec
  and one eval-only spec, not just that its pieces exist independently.

**Out of scope for this work item**: actually executing any `publish:
true` spec without an explicit, separate confirmation at run time — the
code path exists, but a Hub push stays the same real, external, "Create
Public Surface" action it already is via the existing `publish_*.py`
scripts (TASKS.md T-073's own still-pending status is the live proof this
gate isn't bypassed); a runtime-hyperparameter experiment axis (category's
confidence threshold, `c_summary`'s generation length) — disclosed above,
a possible future v2 extension, not this one; re-litigating any Work
item 1-10 model/data/architecture decision.

**Executed so far (2026-09-18)**: steps 2 (T-096) and 4 (T-097) landed
first, as small independent prerequisites, then step 1 (T-098) --
`src/experiment.py`'s `ExperimentSpec`/`PretrainSpec`/`TrainTestSplitSpec`/
`EvalSpec`/`PublishSpec`, every rejection case this section names enforced
via one `model_validator`, plus a stricter `extra="forbid"` on every field
-- then the orchestration half of step 3, `run_experiment` (train if
requested, evaluate via `run_eval` reused verbatim, write the git-tracked
result JSON; caught and fixed a real gap along the way -- `NerTrainConfig`
had no `base_model` field at all, so this section's own `base_model`
validation would have been enforced but silently unhonored for NER; fixed
at the source in `train_ner.py`, not worked around here). The CLI half of
step 3 and step 5 (the historical backfill) are not started. See
TASKS.md T-096-T-103 for the discrete, checkable breakdown.

## Work item 12 — Bring `tests/` under the mypy gate (done 2026-09-18)

**Why**: `.code_quality/mypy.ini`'s `files = src, apps, cli` has always
excluded `tests/` from the project's own documented mypy command. That let
a real, systemic annotation-drift bug go undetected: `tests/news_nlp/
conftest.py`'s `conn`/`two_tier_conn` fixtures (and its
`write_stage_predictions`/`seed_article` helpers) are typed
`sqlite3.Connection`, but at runtime they actually construct and return
`NewsNlpDatabase` (`db_module.connect()`/`connect_pipeline()`).
`NewsNlpDatabase` (`src/news_nlp/db.py`) subclasses `portfolio_common.db.
TwoTierDatabase` → `Database`, which wraps a `sqlite3.Connection` **by
composition, not subclassing** (`portfolio_common/db/engine.py`) — so the
two types are genuinely unrelated to mypy, even though `Database` proxies
`execute`/`executemany`/`executescript`/`commit` with matching signatures
(exactly why passing the "wrong" type has always worked fine at runtime).
~18 test files then copied the fixtures' wrong annotation into their own
test function signatures, producing 208 `arg-type` (and a handful of
unrelated) errors the moment `tests/` enters mypy's scope — surfaced
directly (2026-09-18) while landing T-097, running mypy with an explicit
path argument to double-check a docstring claim rather than the project's
own no-argument command. No behavioral divergence anywhere — every
affected value genuinely is a `NewsNlpDatabase` at runtime; this is a pure
type-hint precision gap, confirmed by re-running the exact same suite
unmodified (`git stash`) and getting the identical 208-error count on
`master`.

**Approach**, four parts:

1. **Fix the root cause in `conftest.py`.** Retype the `conn`/
   `two_tier_conn` fixtures and the `write_stage_predictions`/
   `seed_article` helpers from `sqlite3.Connection` to
   `news_nlp.NewsNlpDatabase`, matching what `db_module.connect()`/
   `connect_pipeline()` genuinely return. Zero behavior change — corrects
   the declared type to match the real runtime type.
2. **Cascade the fix.** Every test function across the ~18 affected files
   (`test_corrections.py`, `test_correction_endpoints.py`,
   `test_category_pipeline.py`, `test_summary_pipeline.py`,
   `test_ner_pipeline.py`, `test_sentiment_pipeline.py`,
   `test_sector_summary.py`, `test_queries.py`, `test_schema.py`,
   `test_db.py`, `test_eval_store.py`, `test_eval_sampling.py`,
   `test_eval_candidate.py`, `test_query_endpoints.py`,
   `test_pipeline_progress.py`, `test_fti_base.py`) that copied
   `conn: sqlite3.Connection` from the fixture retypes it to
   `NewsNlpDatabase` — mechanical, the same edit repeated per file.
3. **Fix the handful of unrelated errors `tests/` entering scope also
   surfaces**, so the whole directory is genuinely clean, not just the
   Connection/NewsNlpDatabase class: `test_eval_candidate.py`'s deliberate
   `revision=None` call (the test only exercises
   `candidate_scored_connection`'s earlier `stage`-validation rejection,
   before `revision` — a required `str` — is ever used; swap the `None`
   for a placeholder string instead of loosening the function's own real
   contract); `test_fti_base.py`'s remaining non-Connection issues (raw,
   unparameterized `Feature()`/`Trainer()` calls inferring `Never` for the
   two "raises NotImplementedError" tests, a `no-any-return`, a private
   `_rows` attribute read off a test double — each needs its own small,
   judgment-call fix, not a mechanical retype); `test_eval_tracking.py`'s 3
   mlflow `list[Run] | Any`/`Experiment | None` union-narrowing spots.
4. **Widen `.code_quality/mypy.ini`'s `files`** from `src, apps, cli` to
   `src, apps, cli, tests` — the change that actually closes the gap for
   good, since steps 1-3 alone would leave this free to silently regrow
   exactly as it did here. Deliberately does **not** include `scripts/`:
   those are one-shot, historical, already-completed scripts (`CLAUDE.md`'s
   own `scripts/` convention — "recover from git history if another bulk
   backfill is ever needed") that this project doesn't hold to an ongoing
   quality bar; `scripts/mine_idiom_sentences_2026_09_13.py`'s own 2
   unrelated pre-existing errors are named here as a deliberate, disclosed
   exception, not silently left out.

**Acceptance criteria**:

- `uv run mypy --config-file=.code_quality/mypy.ini` — the project's own
  documented command, no path argument — reports zero errors with `tests`
  now in `mypy.ini`'s `files` list.
- `uv run pytest` stays green throughout with zero assertion changes —
  every fix in this work item is a type annotation or a one-line
  test-argument correction, never a behavioral change.
- `scripts/` stays outside mypy's scope, `scripts/
  mine_idiom_sentences_2026_09_13.py`'s 2 pre-existing errors named in
  `TASKS.md`/`SPEC.md` as a disclosed, deliberate exception rather than
  silently left unaddressed.
- `git stash`-verified before/after: the exact same 208-error count
  reproduces on an unmodified checkout, confirming nothing here was newly
  introduced by T-097 or any other recent change.

**Out of scope**: any change to `NewsNlpDatabase`'s or `Database`'s own
class hierarchy (e.g. making one a real `Connection` subclass, or
introducing a `Protocol`) — the annotations were simply wrong, not the
design; `scripts/`'s own mypy errors (disclosed, not fixed here — see
Approach step 4).

## Sequencing

Work items 1 and 2 are independent of each other — no ordering
dependency. Work item 1 (pin checkpoints) is a small, self-contained code
change with no external setup required and can land immediately. Work
item 2 (runnable regression gate) is blocked on the maintainer's
infrastructure decision and can happen whenever that's ready.

Work items 3-7 (current focus) are also independent of 1-2 and of each
other, and independent of one another except where noted:

- Work item 5 (category) is documentation-only and can land immediately —
  no blockers.
- Work item 3 (NER) is fully resolved (2026-09-12) — sampling mismatch
  fixed, reprocessing done via T-025's table-versioning + 20,000-article
  resample, and the fresh eval run confirmed the fix (`micro_f1`
  0.74→0.858). Only T-022 (optional full-corpus backfill) remains open,
  non-blocking.
- Work item 4 (sentiment) is the largest remaining item: a design
  decision (step 1) is unblocked today, but the floor-sized baseline run
  (step 2) and any before/after comparison depend on that decision being
  made first.
- Work item 6 (`c_summary` + `sector_summary`) is unblocked today — its
  `c_summary` sampling-scope check (step 1) needs no prior decision,
  though whether to act on `mean_coverage` (step 2) is itself an open
  design call, same shape as Work item 4's step 1. The `sector_summary`
  intro-check addition (step 4) is independent of steps 1-3 and can be
  built in parallel.
- Work item 7 (NER batch processing) is code-complete (2026-09-12) — only
  T-062 (empirical GPU tuning) remains, blocked on access to the project's
  real GPU rather than on anything else in this backlog. Worth running
  *before* a future T-022 full-corpus backfill decision, though not a hard
  prerequisite for it.
- Work item 8 (per-model selection justification) is unblocked today and
  independent of every other work item — it's an artifact/docs change,
  not a code change, and doesn't depend on any pending decision. Explicitly
  not to be implemented until specifically requested (2026-09-14) — stays
  in the backlog as a scoped, pending item.
- Work item 9 (rebalance sentiment training data) is unblocked today and
  independent of every other work item — it needs no prior decision, and
  touches the same model Work item 4 already finished tuning, but as a
  data-quality fix, not a re-litigation of that work. Priority because
  it's the next explicitly requested task.
- **Work item 10 (FTI restructure + evaluation redesign) is done
  (2026-09-18)** — all six steps landed in order (FTI hierarchy →
  `sector_summary` module move → eval reusing the FTI `Inference` classes
  → schema split → reuse mechanism → confusion matrix/ROC), followed by a
  direct follow-up (MLflow `experiment`-awareness) closing the two gaps
  that work item's own docs disclosed.
- **Work item 11 (JSON-driven, single-command experiment runs)** —
  unblocked and independent of every other work item's own outcome — same
  shape as Work item 10 before it: restructures *how* an experiment is run
  and recorded, not *what* any stage's already-decided model/data choices
  are (Work items 1-10 stay untouched). Internally sequential: the
  `stratified_split()` parameterization (step 2, T-096), the `NoOpTrainer`
  wiring (step 4, T-097), the `ExperimentSpec` schema (step 1, T-098), and
  the orchestration function (step 3's own function half, T-099) are **all
  done** (2026-09-18); step 3's CLI half (T-100) is next — it must land
  before the historical-experiment JSON backfill (step 5, T-101) can be
  verified by actually running them. Supersedes Work item 8 as "next up"
  in priority ordering, again; Work item 8 stays a valid, scoped, pending
  item, just no longer first in line.
- **Work item 12 (bring `tests/` under the mypy gate) is done
  (2026-09-18)** — surfaced directly while landing Work item 11's T-097,
  independent of every other work item's own outcome (a test-suite
  type-hygiene fix, not a pipeline/eval behavioral change). All four steps
  landed in order: the `conftest.py` root cause (step 1) before cascading
  it through the ~18 affected test files (step 2); the disclosed unrelated
  fixes plus one genuinely new find caught mid-implementation
  (`test_eval_store.py`'s own unrelated `**dict` kwargs-unpacking errors,
  step 3); widening `mypy.ini`'s scope last (step 4), once everything it
  would flag was already clean. `uv run mypy --config-file=
  .code_quality/mypy.ini` reports zero errors across 67 files;
  `uv run pytest` unchanged at 279 passed.

See `TASKS.md` for the discrete, checkable task breakdown.
