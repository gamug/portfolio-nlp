# TASKS.md — `portfolio-nlp`

Discrete, checkable task breakdown for `.specify/memory/PLAN.md`. Each
task references the plan work item and the `SPEC.md` section it closes.
Check a box only when its acceptance criterion (in `PLAN.md`) is actually
met — not when the code is merely written.

**Closed work items live in `.specify/memory/CHANGELOG.md`**, moved there verbatim
(task IDs unchanged) once every task in them is done, superseded, or moved elsewhere —
see constitution AI behavior #15. This file carries only open work items.

Task IDs are stable, same rule as `SPEC.md`'s `FR-0xx`/`NR-0xx`: don't
renumber; mark a cancelled/superseded task in place instead.

## Work item 2 — Make the regression gate runnable (ops, blocked on maintainer)

- [ ] **T-010** *(maintainer)* Designate/stand up a machine with durable
      access to `$SOURCE_DATABASE_URL`/`$DATABASE_URL` and a persistent
      `mlruns` dir or MLflow server. → `PLAN.md` Work item 2, step 1.
- [ ] **T-011** *(maintainer)* Register it as a GitHub self-hosted runner
      labeled `thesis-data`. → step 2.
- [ ] **T-012** *(maintainer)* Add repo secrets: `LLM_API_KEY`,
      `LLM_MODEL`, `LLM_URL`, `MLFLOW_TRACKING_URI`,
      `SOURCE_DATABASE_URL`, `DATABASE_URL`. → step 3.
- [ ] **T-013** Edit `.github/workflows/eval.yml`: uncomment
      `runs-on: [self-hosted, thesis-data]`, remove the
      `runs-on: ubuntu-latest` placeholder. → step 4. (Code change, but
      meaningless without T-010–T-012 done first — don't land this alone
      expecting it to work.)
- [ ] **T-014** *(maintainer)* Trigger `workflow_dispatch` once; confirm a
      real MLflow run + `eval_run` row is produced. → step 5 / first
      acceptance criterion.
- [ ] **T-015** *(maintainer)* Confirm the gate actually gates: induce a
      regression (temporarily lowered `--regression-tolerance`, or a
      second run compared against a known-worse one) and confirm the
      workflow fails. → second acceptance criterion.
- [ ] **T-016** Update `SPEC.md` §13 item 8 to reflect the resolved (or
      deliberately re-scoped) state. → third acceptance criterion. Also
      update the two architecture artifacts per constitution AI behavior
      #11 — reconcile, never rename.

## Work item 3 — NER: validate the subword-fragmentation fix (resolved 2026-09-12)

`src/pipeline.py`'s `merge_bio_predictions` got a word-boundary-aware fix on
2026-09-10 (bogus single-token spans like `"3"`/`ORG` off `"3M"`), plus a
`_MIN_ENTITY_TEXT_LEN` last-resort filter in `run_ner_stage`. Neither has a
post-fix accuracy number yet — the `micro_f1` 0.7418 / `hallucination_rate`
0.338 baseline in `docs/evaluation.md` predates both. → `PLAN.md` Work item 3,
`SPEC.md` §9 (NER baseline row).

**Started and resolved 2026-09-12.** T-021's investigation confirmed the
sampling mismatch; T-024 (the fix) landed the same day. T-020 turned out
to be more tightly blocked than scoped at first — no post-fix data
existed, and no plain pipeline re-run would produce any — but T-025
(chosen over T-022, executed the same day: versioned `article_entities` →
`article_entities_v1`, resampled 20,000 articles under the fixed code)
cleared that blocker, and T-020 then ran the same day: `micro_f1`
0.74→0.858, `hallucination_rate` 33.8%→16.0% (n=8000 against the T-025
pool). T-023 (this doc's/SPEC.md's write-up) is done. T-022 (full-corpus
backfill of the remaining ~439K pre-fix articles) is the only open item
left in this work item, and it's a non-blocking maintainer scope call, not
a defect. See `docs/evaluation.md`'s 2026-09-12 follow-ups for full
findings.

- [x] **T-021** Empirically check (don't just flag) whether `ner` eval
      sampling suffers the same full-article-vs-lead-cap mismatch already
      confirmed for `sentiment`. **Done 2026-09-12** — confirmed, not just
      suspected: ran `sample_for_stage("ner", size=1000, seed=1)` against
      real data (no LLM calls); 21.4% of sampled articles exceed the
      judge's 6000-char cap, and 16.8% of predicted entities in the sample
      start past it (structurally unverifiable by the judge). See
      `docs/evaluation.md`'s 2026-09-12 follow-up for the full numbers.
      → `PLAN.md` Work item 3, step 2.
- [x] **T-024** *(new, follows from T-021)* Fix the confirmed mismatch —
      add `ner` to `sampling._UNCAPPED_STAGES` (`src/news_nlp/eval/
      sampling.py`), same fix shape as sentiment's 2026-09-08 follow-up.
      **Done 2026-09-12**, before T-020, so that run won't be immediately
      stale the way the pre-text-scope-fix sentiment baseline was — new
      regression test `test_ner_uses_full_body_text_since_the_2026_09_12_fix`
      (`tests/news_nlp/test_eval_sampling.py`); full suite + ruff + mypy
      green. → step 2 (completes the "sampling cap fixed" half of
      `PLAN.md`'s acceptance criterion).
- [x] **T-020** Run a fresh `--stage ner` eval against articles processed
      after the fix, to get a post-fix `micro_f1` / `hallucination_rate` /
      per-type F1 reading comparable to the 2026-09-08 baseline. **Done
      2026-09-12**: `--stage ner --seed 1 --sample-size 8000` against the
      T-025 pool (mlflow `329f9222`) — `micro_f1` 0.7418→0.8578,
      `hallucination_rate` 0.3377→0.1597, `f1_PER` 0.6597→0.9262 (the
      largest per-type gain, matching the subword-fragmentation root
      cause). Full table: `docs/evaluation.md`'s 2026-09-12 "T-020
      executed" follow-up. → `PLAN.md` Work item 3, step 1.
- [ ] **T-022** *(maintainer)* Decide whether a bulk re-extraction of the
      remaining ~439,000 pre-fix `article_entities_v1` articles (not
      resampled by T-025) is worth doing — the fix is future-runs-only by
      design (`docs/evaluation.md` 2026-09-10 follow-up, "Scope"). No
      longer blocking T-020 (T-025 unblocked it with a smaller sample);
      this is now purely about full-corpus coverage. → step 3.
- [x] **T-025** *(new, maintainer, lighter alternative to T-022)* **Done
      2026-09-12.** Executed as: version the table (`ALTER TABLE
      article_entities RENAME TO article_entities_v1`, preserving the
      pre-fix snapshot rather than deleting/reprocessing rows in place),
      then a `random.Random(seed=1)`-seeded resample of 20,000 articles
      via the new `sample_seed` param. Result: 714,334 entity rows across
      19,988/20,000 articles (12 predicted zero entities). Dry-run
      validated first against a scratch DB copy. Full detail:
      `docs/evaluation.md`'s 2026-09-12 "T-025 executed" follow-up.
      → step 1 (alternate path) — done via the table-rename refinement,
      not the originally-sketched `delete_entities_for_article` path.
- [x] **T-023** Once T-020 lands, add a dated follow-up entry to
      `docs/evaluation.md` (same append-only pattern as the sentiment/
      category entries) and update `SPEC.md` §9's NER baseline row —
      don't overwrite the 2026-09-08 baseline row itself. **Done
      2026-09-12** — see `docs/evaluation.md`'s "T-020 executed" follow-up
      and `SPEC.md` §9's ner row. → step 4 / `PLAN.md` Work item 3
      acceptance criteria.

## Work item 4 — Sentiment: close the entity/net-signal reasoning gap (resolved 2026-09-13: fine-tuned + chunk-level selected and merged)

Stratified sampling + the `recall_negative` headline switch (both already
shipped, `docs/evaluation.md` 2026-09-08/09) improved what gets measured and
how it's weighted, but did not touch the model itself. The pilot run
(eval_run 18, n=800) still showed `negative` precision only 0.359 — the root
cause (FinBERT's whole-article softmax has no per-company or net-signal
reasoning the judge applies) was diagnosed and, as of 2026-09-13, three
candidate fixes have been implemented and measured against the same
2,000-article pool (seed=1): chunk-level entity-scoped re-scoring (PR #42),
title-only scoring (PR #43, worse, not shipped), and a fine-tuned FinBERT
+ chunk-level weighting (this branch, `feat/finbert-financial-news-finetune`,
the strongest result — published as `gamug/FinBERT-financial-news` on
Hugging Face Hub). **None of the three is merged to `master` yet** — which
(if any) ships is left to the repo owner. → `PLAN.md` Work item 4, `SPEC.md`
§13 item 1.

- [x] **T-030** Design decision: rather than picking one candidate
      up front, all three were implemented and measured head-to-head —
      entity-scoped re-scoring using `article_entities` (PR #42),
      title-only scoring (PR #43), and a fine-tuned sentiment model
      (this branch). → `PLAN.md` Work item 4, step 1. **Done 2026-09-13**
      — see `PLAN.md` Work item 4 "Executed" and `docs/evaluation.md`'s
      2026-09-13 follow-up for the full comparison table.
- [x] **T-031** Ran regression-tracked `--stage sentiment` evals at
      n=2,000 (above the ~1,800-2,200 floor, `docs/evaluation.md`
      "Sample-size floor") — the same seeded 2,000-article pool was reused
      across all three candidates plus the pre-fix pilot, so
      `recall_negative` / `precision_negative` are directly comparable.
      → step 2. **Done 2026-09-13.**
- [ ] **T-032** Re-solve `docs/evaluation.md`'s "Sample-size floor" purity
      estimates using T-031's actual measured per-stratum agreement
      (`strata_json` / `agreement_rate_target_negative` etc.) instead of
      today's planning-only estimates, before locking in a permanent
      `--sample-size` default. → step 3. **Still open** — not done as
      part of the 2026-09-13 fine-tuning work; the four real evals give
      more empirical grounding than before but the purity-estimate
      re-solve itself hasn't been done.
- [x] **T-033** Re-ran the eval for each candidate and added dated
      follow-up entries to `docs/evaluation.md`; updated `SPEC.md` §13
      item 1 and §9's sentiment baseline row with the results. → step 4 /
      `PLAN.md` Work item 4 acceptance criteria. **Done 2026-09-13** for
      all three candidates; final "which one ships" decision explicitly
      left open for the repo owner rather than resolved by these tasks.

## Work item 5 — Category: hold the line on the hierarchical fix (validation only, low priority)

Already fixed — hierarchical two-level classification +
`CATEGORY_CONFIDENCE_THRESHOLD` 0.6 calibration (`docs/evaluation.md`
2026-09-09 follow-ups, eval_run 19–21; `docs/category-taxonomy.md`
"Hierarchical classification" / "Threshold calibration").
`accuracy_vs_judge` 0.442→0.487; the three worst leaf slugs up from
0.14/0.16/0.33 to 0.61/0.51/0.53. Kept here only for the one remaining
open thread and the doc-consistency cleanup. → `PLAN.md` Work item 5,
`SPEC.md` §13 item 2.

- [x] **T-040** Confirm the fix + calibration landed and is measured (no
      action — done, see numbers above).
- [ ] **T-041** `other`'s own precision (0.474) is still an open
      calibration target (`docs/evaluation.md` "The `other` bucket: a
      precision problem of its own") — likely `CATEGORY_GROUP_FLOOR` or a
      slug-specific threshold next, not another global
      `CATEGORY_CONFIDENCE_THRESHOLD` move. Low priority, not blocking:
      revisit once more post-calibration eval data accumulates.
      → `PLAN.md` Work item 5, step 1.
- [x] **T-042** Update `SPEC.md` §13 item 2 and the §14 disposition table
      row to reflect the resolved state — it currently cites the
      pre-calibration 0.14/0.16/0.20/0.33 numbers, superseded by the
      2026-09-09 follow-up. Annotate in place, keep the item number.
      → step 2 / `PLAN.md` Work item 5 acceptance criteria. (Done
      2026-09-12, this pass — see `SPEC.md` §13 item 2 and §14's
      disposition table.)

## Work item 7 — NER: develop batch processing (code done 2026-09-12; T-062 needs a real GPU)

`run_ner_stage` (`src/pipeline.py`) was the only stage with no batching —
one chunk through the model per forward pass, unlike `run_category_stage`
(`CATEGORY_BATCH_SIZE=8`) and the summarization stages
(`SUMMARY_BATCH_SIZE=4`). Not a documented trade-off, genuinely
unaddressed. T-025's 2026-09-12 resample measured the real cost: ~22
articles/sec unbatched — a full-corpus backfill of the remaining ~439,000
pre-fix articles (the open T-022 question) would take ~5+ hours at that
rate on hardware with headroom to go faster. → `PLAN.md` Work item 7,
`SPEC.md` §13 item 11.

- [x] **T-060** Implement batched tokenization + forward pass in
      `run_ner_stage`: a new `NER_BATCH_SIZE` constant, flatten each batch
      of articles' chunks (via `chunk_text`) into one chunk list tagged
      with its owning article index, tokenize with `padding=True` in one
      call, one forward pass, regroup per-article via the tagging (same
      offset math already used today). **Done 2026-09-12** — new
      `_ner_batch` helper in `src/pipeline.py`; `NER_BATCH_SIZE = 8`
      (`CATEGORY_BATCH_SIZE`'s value as a first guess, not tuned yet, see
      T-062). → `PLAN.md` Work item 7, steps 1-4.
- [x] **T-061** Add a parity regression test: the same fixture articles
      processed through the batched path must produce byte-identical
      `article_entities` output (same entities/offsets/scores) as the
      pre-existing unbatched path — land this *before* anything else in
      this work item ships. **Done 2026-09-12** —
      `test_batched_and_per_article_ner_processing_produce_identical_entities`
      (`tests/news_nlp/test_ner_pipeline.py`): two different-length
      articles processed once at `NER_BATCH_SIZE=1` and once at
      `NER_BATCH_SIZE=2` (forcing real cross-article padding) assert
      identical entities; full suite green (206 passed), ruff + mypy
      clean. → step 6.
- [ ] **T-062** Tune `NER_BATCH_SIZE` empirically against the 6GB VRAM
      budget (SPEC.md NR-001) and measure the real throughput improvement
      (articles/sec) on a real sample (mirrors T-025's resample as a
      before/after comparison) — don't assume batching helps without
      measuring it, and don't copy `CATEGORY_BATCH_SIZE`/
      `SUMMARY_BATCH_SIZE`'s values blind (NER's per-chunk sequence
      length and variable chunks-per-article shape a different memory
      profile). **Not done — needs the project's actual GPU**, which the
      environment T-060/T-061 shipped from doesn't have (CPU-only
      sandbox); `NER_BATCH_SIZE=8` is an untested starting guess, not a
      measured value. → steps 5, 7.
- [x] **T-063** Document the new constant and batching design
      (`docs/modules/news-nlp.md` and/or a `pipeline.py` comment, matching
      how `CATEGORY_BATCH_SIZE`/`SUMMARY_BATCH_SIZE` are documented) and
      update `SPEC.md` §13 item 11 to reflect the shipped state. **Done
      2026-09-12** — see `NER_BATCH_SIZE`'s comment in `src/pipeline.py`,
      the NER bullet in `docs/modules/news-nlp.md`, and `SPEC.md` §13
      item 11. → `PLAN.md` Work item 7 acceptance criteria (all but the
      "no VRAM regression" / "measured throughput" criteria, which need
      T-062).

## Status

T-001–T-007 (Work item 1, pin checkpoints) are **done** (2026-09-14) —
see the header of Work item 1 above. T-010–T-016 (Work item 2,
regression gate) remain blocked on the maintainer's infrastructure
decision — nothing in that item has started.

**Model performance checking (Work items 3-7) is fully resolved except
two non-blocking items:** T-040, T-042, T-021, T-024, T-025, T-020,
T-023, T-060, T-061, T-063, T-030, T-031/T-032, T-050–T-057, and T-058
are all done — Work items 3 (NER validation), 4 (sentiment), and 6
(summarization eval) are fully resolved. Work item 3's only open item is
T-022 (full-corpus backfill), non-blocking. Work item 7 (NER batching)
is code-complete, with only T-062 (empirical GPU tuning) left — this
sandbox gained CUDA access 2026-09-14, so T-062 is actionable, just not
yet run.

**Work item 8** (per-model selection justification in the artifact,
T-064–T-069) is **done (2026-09-19)** — requested the same day, one model
at a time: sentiment (T-064), category (T-065), NER (T-066), `c_summary`
(T-067), and `sector_summary` (T-068), each landing a "Why this model" (or,
for `sector_summary`, "why no model") sub-block in its existing per-model
block of the Claude Artifact's `#eval` section, mirrored in
`docs/modules/news-nlp.md` prose in the same commit — closing T-069 as it
went rather than as a separate final pass. Sourced from `justification.md`
(a separately link-verified sourcing doc) and `justification_.md` (the
per-model draft prose). Sentiment's first pass sprawled into several
tables/essays and, after direct user feedback, was rebuilt down to one
metrics table (precision/recall/ROC AUC per class per candidate, later
swapped to accuracy_ovr/recall/ROC AUC per further feedback) plus a short
justification; every subsequent sub-block used that pared-down format
from the start.

**Work item 9** (rebalance sentiment training data) is **done (2026-09-19)**:
T-070–T-072 and T-074–T-080 done 2026-09-14/15 — dataset rebalanced and
republished, three retraining/architecture approaches tried and measured
downstream against real traffic (v3 downsampled, v4 class-weighted, v5
base-checkpoint swap), and the user chose **v4** for production (T-080) —
its `recall_negative` gain (0.808→0.832, this pipeline's priority metric)
outweighed the `agreement_rate`/`mean_severity` cost, and beat v5's own
real but narrower `precision_negative` gain (0.513→0.526) on the metric
that actually decided the choice. **T-073 (publish v4 to the Hub + move
`MODEL_REVISIONS`) is now done too** — see its own entry above; the
previously-blocking "Create Public Surface" permission was given
explicitly (2026-09-19), along with a request to simplify the model card's
comparison tables to `accuracy_ovr_<class>` first. The
`sector_summary` pre-fix rows
(3,444, from Work item 6's T-058 fix) are still queued to self-heal on
the next real `--summarize` run, not yet triggered — a deliberate
production action left to the repo owner, not a task with an ID.

**Work item 10** (formalize the pipeline/evaluation architecture,
T-082–T-095) is **done (2026-09-18)** — all six sequential steps landed in
order: the FTI class hierarchy (T-082–T-086), the `sector_summary` module
move (T-087), the eval module's FTI reuse + live candidate-model scoring
(T-089), the schema split (T-090), the judge-verdict reuse mechanism
(T-091), the confusion-matrix/ROC-AUC additions + their regression test
(T-092–T-094), and the docs/architecture-artifact reconciliation pass
(T-095) that closes it out. Was priority #1 as of 2026-09-16, superseding
Work item 8. A direct follow-up the same day (not its own Work item —
already-filed FR-014's own disclosed gap) closed the two remaining
"not `experiment`-aware" spots: `eval_run` gained a real `experiment`
column, MLflow now tags/filters by it, and `queries.latest_eval_runs`/
`GET /eval/latest` group by `(stage, experiment)`.

**Work item 11** (JSON-driven, single-command experiment runs,
T-096–T-103) is **done (2026-09-18)** — all five parts landed in order.
Surfaced directly while walking through the full historical
sentiment-candidate command sequence (Work item 9) one command at a
time: no single place declared an experiment's full configuration
before it ran, and several of the existing one-off scripts mutated
shared DB tables in place, needing a manual restore step afterward.
T-096/T-097 (small, independent prerequisites — the training-split
parameterization, `NoOpTrainer` wiring) landed first as two separate
PRs; T-098 (`ExperimentSpec`, `src/experiment.py` — every named
rejection case enforced plus a stricter `extra="forbid"`) came next;
then T-099 (`run_experiment` — train, auto-resolve the candidate model,
evaluate via `run_eval` reused verbatim, write a git-tracked result
JSON; caught and fixed a real gap along the way, `NerTrainConfig` had
no `base_model` field at all) and T-100 (`cli/run_experiment.py` — "exit
1 if regressed" free, inherited from `run_eval`'s own `SystemExit(1)`);
T-101 backfilled all 8 real historical `experiments/*.json` specs
(values sourced from `docs/evaluation.md`, not guessed — a third
disclosed gap found along the way, `sentiment_v2_chunklevel_finetuned.json`/
`_v3_downsampled.json` are deliberately content-identical, see
`experiments/README.md`); T-102's own end-to-end test shipped with T-099
instead of being deferred. T-103 closed it out: a new "Experiments"
section in `docs/evaluation.md`, a pointer + a stale-bullet fix in
`docs/modules/news-nlp.md`, and both architecture artifacts reconciled
(same closing pass T-095 did for Work item 10). Superseded Work item 8
as "next up" in priority while it ran — Work item 8 (per-model selection
justification in the artifact, T-064–T-069) is now next up again,
still a valid, scoped, pending item.

**Work item 12** (bring `tests/` under the mypy gate, T-104–T-108) is
**done (2026-09-18)** — surfaced directly while closing out T-097 (PR
#77): running `uv run mypy --config-file=.code_quality/mypy.ini .` with an
explicit path argument (overriding `mypy.ini`'s own `files = src, apps,
cli` scope, purely to sanity-check a docstring claim rather than using the
project's documented no-argument command) turned up 208 pre-existing
errors in `tests/`, confirmed unrelated to T-097 via `git stash`
(identical count reproduces on an unmodified checkout). Root-caused
(read-only investigation) to one systemic type-annotation bug, not 19
independent ones: `tests/news_nlp/conftest.py`'s `conn`/`two_tier_conn`
fixtures (and `write_stage_predictions`) were typed `sqlite3.Connection`
but actually construct and return `NewsNlpDatabase` (a composition wrapper
around a `sqlite3.Connection`, not a subclass of it) — ~18 test files
copied that wrong annotation into their own test function signatures. All
four parts landed in order: T-104 (the `conftest.py` root-cause fix,
including a genuine dual-caller case in `seed_article` needing a
`sqlite3.Connection | NewsNlpDatabase` union rather than a straight
retype) → T-105 (cascading the fix through the ~18 files — 9 by mechanical
`sed`, 2 needing scoped hand-edits around genuine raw-`sqlite3.connect()`
blocks) → T-106 (the disclosed unrelated fixes, plus one
previously-undisclosed find caught while re-verifying from scratch:
`test_eval_store.py`'s own 15 `**dict` kwargs-unpacking errors, fixed with
two local `TypedDict`s) → T-107 (`.code_quality/mypy.ini`'s `files` widened
to `src, apps, cli, tests`, `scripts/` deliberately excluded). T-108's full
verification: `uv run mypy --config-file=.code_quality/mypy.ini` reports
zero errors across 67 files; `uv run pytest` 279 passed (same count as
before this work item, zero assertion changes anywhere — every fix here
was type-only); ruff clean; `git stash` confirmed the pre-fix 208-error
count is genuinely pre-existing, not introduced by this session. Per the
user's 2026-09-18 decision this ran ahead of Work item 11's own T-098,
which is now unblocked.

**Work item 13** (tighten judge-verdict reuse to require a matching
prediction, T-109–T-110) is **done (2026-09-19)** — filed and closed in one
pass after the user asked whether T-091's reuse mechanism was actually
working. Confirmed via direct reproduction before any code changed: two
`run_eval` calls sharing `candidate_model`/`candidate_revision` (the
`run_experiment`/fixed-local-checkpoint-path shape Work item 11 enables)
but a genuinely different underlying prediction silently reused the first
run's stale verdict for the second — zero fresh judge calls, wrong
per-run metrics. `find_verdict_json` now also requires the paired
`eval_inference.prediction_json` to match. `docs/evaluation.md`'s
unrelated stale "`--run-name` is cosmetic only" comment (2026-09-15) fixed
alongside it as a disclosed doc correction, not a code bug.
