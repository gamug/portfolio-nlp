# PLAN.md — `portfolio-nlp`

The implementation plan for the live backlog identified in
`.specify/memory/SPEC.md`. Where the constitution is principles and
`SPEC.md` is the requirements/architecture contract, this document is the
"how, and in what order" for the work that contract still leaves open.

**Scope of this plan was originally narrow, now expanded to cover active
model-performance work.** `SPEC.md` §13 (Open Questions & Risks) lists nine
items; §14 (Scope Boundaries) marks most of them **accepted** (permanent
characteristics of this project at its current, non-production scope) and
one (§13 item 5, throughput/latency SLA) **retired** outright. Item 8 was
flagged "should fix regardless of scope" and items 1/2 (sentiment/category
accuracy) were originally treated as accepted research limitations — see
Work items 1-2 below for the former. Items 1/2 have since been **promoted
out of "accepted, not a queued task"**: category's fix already shipped
(Work item 5), and sentiment is now active, priority work (Work item 4),
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

## Work item 3 — NER: validate the subword-fragmentation fix

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

**Approach**:

1. Run a fresh `--stage ner` eval against articles processed after the fix
   lands, at a sample size comparable to the 2026-09-08 baseline (n=1000),
   with `--seed` for reproducibility.
2. While that data exists, also check the "suspected but unverified" note
   in `docs/evaluation.md`: `run_ner_stage` scores the *whole* article, but
   whether the eval judge's sampling matches that scope (vs. the
   lead-chunk-only mismatch already confirmed and fixed for sentiment) has
   never been empirically checked for NER specifically.
3. Bring the question of a bulk `article_entities` re-extraction to the
   maintainer as a scope decision — not something to do unilaterally,
   consistent with how the category hierarchical-classifier migration
   handled the same "future-runs-only" trade-off (SPEC.md §13 item 6).
4. Record the result as a dated follow-up in `docs/evaluation.md` (append,
   don't overwrite the baseline) and update `SPEC.md` §9's NER row.

**Acceptance criteria**:

- A fresh eval run's `micro_f1` / `hallucination_rate` / per-type F1 are
  logged to MLflow and `docs/evaluation.md`, comparable to the 2026-09-08
  baseline.
- The full-article-vs-lead-cap sampling question is either confirmed (and
  the sampling cap fixed, mirroring the sentiment fix) or explicitly ruled
  out with evidence — not left as an open "suspected" note indefinitely.
- `SPEC.md` §9 updated with the new baseline row/date.

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

## Sequencing

Work items 1 and 2 are independent of each other — no ordering
dependency. Work item 1 (pin checkpoints) is a small, self-contained code
change with no external setup required and can land immediately. Work
item 2 (runnable regression gate) is blocked on the maintainer's
infrastructure decision and can happen whenever that's ready.

Work items 3-5 (current focus) are also independent of 1-2 and of each
other, and independent of one another except where noted:

- Work item 5 (category) is documentation-only and can land immediately —
  no blockers.
- Work item 3 (NER) needs a fresh eval run against articles processed
  after the 2026-09-10 fix; otherwise unblocked.
- Work item 4 (sentiment) is the largest remaining item: a design
  decision (step 1) is unblocked today, but the floor-sized baseline run
  (step 2) and any before/after comparison depend on that decision being
  made first.

See `TASKS.md` for the discrete, checkable task breakdown.
