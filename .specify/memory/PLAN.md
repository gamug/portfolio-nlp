# PLAN.md — `portfolio-nlp`

The implementation plan for the live backlog identified in
`.specify/memory/SPEC.md`. Where the constitution is principles and
`SPEC.md` is the requirements/architecture contract, this document is the
"how, and in what order" for the work that contract still leaves open.

**Scope of this plan was originally narrow, now expanded to cover active
model-performance work.** `SPEC.md` §13 (Open Questions & Risks) now lists
ten items (a tenth — `c_summary`'s coverage/sampling-scope question — was
added alongside Work item 6, below); §14 (Scope Boundaries) marks most of
the original nine as **accepted** (permanent characteristics of this
project at its current, non-production scope) and one (§13 item 5,
throughput/latency SLA) **retired** outright. Item 8 was flagged "should
fix regardless of scope" and items 1/2 (sentiment/category accuracy) were
originally treated as accepted research limitations — see Work items 1-2
below for the former. Items 1/2 have since been **promoted out of
"accepted, not a queued task"**: category's fix already shipped (Work item
5), and sentiment is now active, priority work (Work item 4). Item 10
(`c_summary`) is new, priority work from the same push (Work item 6). All
per `docs/evaluation.md`'s dated follow-ups and the current focus of this
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

## Work item 1 — Pin HF model checkpoints to a commit SHA

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

## Work item 4 — Sentiment: close the entity/net-signal reasoning gap

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

**Approach**:

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

**Acceptance criteria**:

- A design decision is made and documented (which candidate, and why —
  same style as the "Why recall, not F1" / "Why precision, not recall"
  write-ups already in `docs/evaluation.md`).
- A floor-sized (~1,800-2,200), seeded baseline run exists before any
  before/after comparison is drawn.
- The chosen change measurably improves `negative` precision (or another
  explicitly-justified metric) without collapsing `recall_negative` below
  its current range, confirmed via a post-change eval run.
- `SPEC.md` §13 item 1 and §9 updated with the dated result.

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

## Work item 6 — Summarization (`c_summary` + `sector_summary`): validate eval scope, close the coverage gap, and add a lightweight sector-intro check

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

See `TASKS.md` for the discrete, checkable task breakdown.
