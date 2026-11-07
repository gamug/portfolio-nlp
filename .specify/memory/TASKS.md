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

## Work item 15 — Category: fix the `other` bucket (filed 2026-10-14; D1/D2/D3 decided same day; gate conditional on T-117; D4, D5–D9, D10, D2b and the T-117 thresholds decided, see T-139)

**Re-scoped 2026-10-28 into Work item 18** (T-143): see the per-task markers below.

`other` is 195,993 of 459,112 articles (42.7%); its precision of 0.474 is a quoted
figure from a different corpus (see T-123). A read-only investigation (2026-10-14)
suspected it mixes "not about the ticker's company" (ticker symbols that are English
words hold 39.0% of the corpus, plus companies cut to a generic first word) with "about
the company, no dimension clears the 0.6 bar". A population check on 2026-10-22,
stratified by source and period (`docs/evaluation.md`), found wrong tagging explains at
most about 11% of `other` (rejected articles are 68.8% of the corpus and 70.8% of
`other`; `other` is 37.9% of confirmed articles and 43.9% of rejected ones). The main
lever is therefore calibration, which does not wait for the relevance work. Plan:
calibrate per-slug thresholds and a margin rule offline from stored scores (T-121(a) after
T-133's partition, then T-121(b)'s judged sample once the owner approves the spend); clean up the results the tagging fix removed (T-131, after a
pre-purge check); decide the doubtful `review` pairs (T-132); build a relevance gate only
if what remains justifies one (T-117's rule); then re-evaluate and re-classify (T-123,
T-124, by a re-label unless T-140 says otherwise).
→ `PLAN.md` Work item 15, `SPEC.md` §13 item 19.

Tasks are listed in execution order within Work item 15, not ID order (IDs are
stable; T-130 to T-134, T-139 to T-142 were added after the rest; T-135 to T-138
belong to Work items 16 (closed, `CHANGELOG.md`) and 17, which is why the numbers skip;
a letter suffix such as `T-138b` is the close-out of its work item, unlike
T-121(a)/(b), which are parts of one task). The table below and `PLAN.md`'s Sequencing
section are the authority for ordering.

Dependencies at a glance:

| Task | Depends on | Blocks | Can start now? |
|---|---|---|---|
| T-116 audit | none | none | yes |
| T-130 relevance judge | none | T-132 (remainder), T-117 | yes |
| T-128 upstream hand-off | `portfolio-data-mining` T-055–T-058 | T-132, T-117 onward | **done 2026-10-22** |
| T-133 clean-subset re-read | none (read-only SOURCE attach, existing) | T-123's baseline, T-117 | **done 2026-10-22** |
| T-139 decisions D5–D9 | the owner | T-132 (D5, D6), T-117 (D7), WI16 (D8), WI17 (D9) | **decided 2026-10-22, defaults** |
| T-134 date window (D7) | T-139 | T-117 sampling, T-124 size | **decided 2026-10-22, default** |
| T-132 quarantined pairs | T-128, T-139 (D5, D6), T-135 (schema guard, done in #109, `CHANGELOG.md`) | T-117 | **yes** |
| T-137 multi-company sizing (WI17) | T-132 (grouping, ORG-to-ticker), T-139 (D9) | T-138 | no |
| T-138 multi-company build | T-137 result, the owner | T-138b | no (conditional) |
| T-138b WI17 close-out | T-137 (T-138 if built) | none | no |
| T-117 measure R and N | T-128, T-130, T-132 | T-119, T-120 (or their cancellation) | no |
| T-119, T-120 gate, label | T-117 sign-off | T-123 | no |
| T-121(a) offline re-simulation on stored verdicts | T-133 (partition of the stored verdicts; done 2026-10-22) | T-121(b) design check | **done 2026-10-23** |
| T-141 calibration judge + sampler | none (code, hermetic) | T-121(b) | **done 2026-10-23** |
| T-142 D10 spend for T-121(b) | the owner | T-121(b) | **decided 2026-10-22, cap 3,800** |
| T-121(b) judged calibration sample | T-121(a), T-141, T-142, LLM credentials | T-122 | no; population = confirmed articles (the gated-in one only if a gate is built) |
| T-122 ship the rule | T-121 | T-123 | no |
| T-123 re-evaluate | T-121, T-122 (T-119, T-120 if built) | T-124 | no |
| T-131 cleanup (D4) | T-128, T-132 (for `review`), D4 (decided 2026-10-22, defaults) | T-124 | no |
| T-140 D2b mechanism (re-label vs GPU re-inference) | the owner | T-124 | **decided 2026-10-22, re-label from stored scores** |
| T-124 re-classification | T-123 approved, T-131, T-140 | T-129 | no |
| T-125, T-126 contingent | T-123 result | none | no |
| T-127 sentiment impact | none (separate PR) | none | yes |
| T-129 close-out | all of the above | none | no |

- [x] **T-116** **Superseded 2026-10-28 → Work item 18** (re-scope D11–D24: triage per unique body replaces the relevance gate, calibration and re-label; not done). Commit a rerunnable audit script under `scripts/` that
      reproduces the 2026-10-14 figures from the RESULTS store (label
      distribution; level-1-floor vs level-2-threshold split; best-leaf-score
      bands; `other` rate by source and by ticker; share held by
      English-word tickers, including `ALL`). Read-only; no pipeline or schema change.
      → `PLAN.md` Work item 15, step 1.
- [x] **T-130** **Superseded 2026-10-28 → Work item 18** (re-scope D11–D24: triage per unique body replaces the relevance gate, calibration and re-label; not done). *(new; no crawler dependency)* Build the relevance judge
      T-117 depends on: a new `relevance` task in `news_nlp/eval/` ("is this
      article about `{company}`?" — prompt under `eval/prompts/`, verdict
      model, entry in the judge registry beside `judge_sentiment`/
      `judge_category`, storage in the eval tables, hermetic tests with the
      LLM mocked). Needs the `eval` dependency group and LLM credentials
      (`LLM_API_KEY`/`LLM_MODEL`/`LLM_URL`). Validate it against a small
      hand-labeled subset before trusting it. **Constraints:** it must be
      opt-in. `sampling.STAGES` feeds `run_eval`'s default stage list (and the
      CLI's `--stage all`), so adding `relevance` to it would silently change
      what `all` runs; keep it out of `STAGES`' default set, check whether
      `--check-regression` iterates stages and exclude it there too, and leave
      `ExperimentSpec`'s own four-stage `Literal` untouched. Lock this with a
      snapshot test on the exact default stage list and on the other stages'
      metric key sets (same shape as FR-016's). As a binary relevance
      classifier its reported metrics are the complete constitution AI
      behavior #12 set (precision/recall/F1/`accuracy_ovr` per class; overall
      accuracy and macro F1), not a subset. Needs its own FR in `SPEC.md`,
      added in the PR that ships it. → step 3, T-117.
- [x] **T-133** *(new 2026-10-22; no dependency; **done 2026-10-22**, see the closing note)* Measure what the
      wrong tags did to category's measured quality, from stored verdicts, with **no
      new judge calls**.
      - **Goal:** replace the hypothesis in the plan ("corrected corpus", "what wrong
        tags do") with a number: category's complete constitution #12 metric set on the
        articles that survive the re-tag, next to the full stored sample.
      - **Inputs:** the stored category judgements: **only eval_run 43** has any
        (`eval_inference`/`eval_verdict`, 2,800 judged; eval_run 21's rows are not in the
        store and it sampled a corpus of about 100k, so it is not a baseline), the RESULTS
        article ids, and a read-only attach of SOURCE for each article's `fetch_status`
        and `discovered_urls.match_class` (reads the two existing tables).
      - **Progress 2026-10-22 (read-only, on the real store; rerunnable with
        `scripts/measure_corrected_corpus_2026_10_22.py`, numbers in `docs/evaluation.md`'s
        2026-10-22 follow-up):** step 1 is done:
        93,330 `accept` / 315,822 `reject` / 49,960 `review`, the 459,112 (plan, "corrected
        corpus"). A first look at step 3 on eval_run 43's `representative` bucket is in the
        plan ("what wrong tags do"): `other` judged correct 36.8% of accepted vs 67.4% of
        rejected articles. That was unweighted agreement only; steps 2–4 below, with the
        complete #12 set and the sample-size check (run 43 has 538 accepted, 1,953
        rejected, 309 review articles among the 2,800), remain.
      - **Steps:** (1) done. (2) Partition the stored judged items into survivors
        (`accept`/`ok`), rejected and review. (3) Recompute through
        `aggregate_category`, the same code path, per partition and overall: per-class
        precision/recall/F1/`accuracy_ovr`, overall accuracy and macro F1, plus each
        partition's judged count against `docs/evaluation.md`'s sample-size floor.
        (4) Report; change no behavior.
      - **Done when:** a dated follow-up in `docs/evaluation.md` carries the
        side-by-side table and counts; it says whether the surviving partition clears
        the floor (if not, T-123 decides a fresh baseline); the surviving-partition
        baseline is the one T-123 reuses. Hermetic tests only (LLM not called).
      - **Depends on:** nothing new. **Blocks:** T-123's baseline, T-117's sizing.
        → plan step 5.
      - **Done 2026-10-22:** steps 2-4 ran on the real stores (`news_nlp.eval.partition`
        plus section 12 of `scripts/measure_corrected_corpus_2026_10_22.py`; hermetic tests
        in `tests/news_nlp/test_eval_partition.py`; the `all` column reproduces run 43's
        stored `metrics_json`); the side-by-side table is in `docs/evaluation.md`'s
        2026-10-22 follow-up. Result: on `accept` (538 judged) `other` precision is 0.32
        (95% interval 0.24-0.40) against 0.58 for the whole run, and the partition does **not** clear the floor
        (538 against about 2,500-3,000; 11 true `capital_shareholder_returns`), so run 43's
        `accept` partition is not T-123's baseline: T-123 takes the report half of
        T-121(b)'s sample, as planned.
- [x] **T-139** *(user decision record; **decided 2026-10-22: every decision kept at
      its default**; the owner's words that day: "Keep the decisions in default
      values", after being shown the list below with its defaults)* The decisions this work item needed, with the answer recorded:
      - **D5 verdict vocabulary** (blocks T-132): adopt data-mining's `about` /
        `mention` / `other` / `unclear`. Sub-question: a confident `mention` does not
        purge existing results until its precision is measured.
      - **D6 interface** (blocks T-132): read-only attach of SOURCE. It adds
        `discovered_urls`, a table this repo has not read, to the SOURCE contract
        (constitution AI behavior #10: approved by this answer).
      - **D7 date window** (blocks T-117's sampling and T-124's size, see T-134): no
        filter inside the stages.
      - **D8 schema guard** (Work item 16, closed, `CHANGELOG.md`): build it here, minimum version 3; a
        generic helper in `portfolio-common` only if a second consumer asks.
      - **D9 multi-company** (blocks Work item 17): its own work item, sizing first;
        unit of work the body group for compute and the pair for decisions. Still open,
        and not needed before T-137's sizing: which fact types the knowledge graph needs
        first (a question for that repo's owner).
      - **Second round, 2026-10-22, every answer the recommended default (the single
        record of these values; other text points here):**
        **D10 (T-142)** spend approved, cap 3,800 judge calls, credentials and budget in
        place. **D2b (T-140)** re-label from stored scores after the 300-article
        verification. **Gate rule (T-117):** build `not_relevant` only if R >= 5% and
        N >= 20%. **T-123 targets:** `other` precision at least +0.05 (paired, lower bound
        above 0); `other` recall, accuracy and macro F1 each no worse than -0.02; false-reject
        at most 5% if a gate exists; no per-slug regression beyond `--check-regression`.
        **T-131 pre-purge stop/go:** 400-article sample, go if the upper bound is at most 5%.
      - **D4 (T-131), decided 2026-10-22, both recommended defaults preserved:** (1) the
        removed ids' rows are **kept** in `<table>_removed` archives (copy, then delete from
        the live table, one transaction per table, dry run first), not dropped; (2)
        `sector_summary` regenerates by **deleting only the affected `(sector, sub_industry,
        week)` rows**, not by bumping `SECTOR_SUMMARY_FORMAT_VERSION`.
      - **Still provisional:** **the 72-hour GPU stop/go** (T-124), not part of the confirmed set; it matters only
        if the GPU path is used, which D2b makes the exception. The release threshold for
        `about` is set jointly with data-mining after T-132's measurement.
- [x] **T-140** *(new user decision D2b, opened 2026-10-22; **decided 2026-10-22: the recommended default, re-label from stored scores after the 300-article verification**)* D2
      (re-run the entire classification once the rule is tested) stands. What is asked
      is its mechanism. The plan first specified full GPU re-inference; the stored
      scores make a re-label equivalent for rules confined to level 2 (PLAN step 6, with
      the measured counts). **Recommended default:** re-label from stored scores after
      the 300-article verification in T-124, with GPU only for what the stored scores
      cannot reproduce. Alternative: full re-inference regardless (about 98k articles on
      GPU, with the 72-hour stop/go). Blocks T-124.
- [x] **T-134** *(user decision D7, **decided 2026-10-22 at the default**; the plan's
      "date window" bullet)* The date window. 62,406 of 98,462 `ok` articles (63.4%) were
      published before 2022 (the collector's range starts 2022-01-01; its sitemap
      reader never applied it; fixed going forward; nothing deleted; `pub_date` is
      always present). Decision: (a) no filter inside the stages; the window is applied
      explicitly where a consumer needs it and reported separately (classification is
      cheap to keep and a filter cannot be undone cheaply). Rejected for now: (b) filter
      the stages to a stated window; (c) full history everywhere without a separate
      report. Effects recorded: T-117 stratifies by the window, T-124's article count and
      GPU projection use the full corrected corpus, and a sentiment run reads about two
      thirds of its input from outside 2022+. **Blocks (now unblocked):** T-117's
      sampling, T-124's size.
- [ ] **T-132** **→ Work item 18 P1 and P3 (2026-10-28). Body-hash grouping and name/ticker resolution shipped in P1 (`news_nlp.triage`, T-144); **shrunk 2026-11-05 by the owner: only the upstream alias table stays open. The per-pair verdicts, T-130's judge and the block on T-117 (both superseded) are dropped; the quarantined pairs follow D19 (T-131). The text below is the original design, kept as the record.** *(from `portfolio-data-mining` T-055 design v1, approved 2026-10-15;
      reshaped 2026-10-22 to the 2026-10-21 hand-off)* Decide the quarantined
      `review` pairs. Never act on doubt, in either direction.
      - **Goal:** answer "is the company a subject of this article?" for the pairs
        discovery could only mark `review` (weak URL/title evidence). **Input:** all the
        quarantined `review` pairs selected at run time (52,039 on 2026-10-21; the count
        moves). **Subset that already has results:** 49,960 of the articles in this
        repo's RESULTS store are `review` (measured 2026-10-22); only those can have
        results purged (T-131's rule), while a verdict on a pair with no RESULTS rows can
        only feed upstream's release decision. The verdicts are the four data-mining
        applies: `about`, `mention`, `other`, `unclear` (D5).
        The correct rate of this class is **not measured** (only 18 of 20 uncontested
        News Corp URLs); do not assume it.
      - **Inputs:** a read-only attach (D6) of SOURCE, selection
        `articles` ⋈ `discovered_urls` on `id`, `match_class = 'review'`,
        `fetch_status = 'quarantined'`, **re-run on every run** because the extractor
        promotes pairs out of the class and the run records the class it saw. The exact
        query and the quarantine shape are defined only upstream (T-057/T-058); this
        repo never writes `urls.db`. Also the body, the title, the **stored
        `article_entities`** of those articles (verified 2026-10-22: 49,949 of 49,960 have
        them, 49,766 with an `ORG` span; zero GPU) and upstream's alias table (names, aliases, ticker forms, offered as a CSV).
      - **Steps:** (1) dedupe by body hash (share classes carry one body under two
        pairs; decide per pair, compute per body). (2) CPU rule first, no model call:
        `about` on the full normalized name with aliases, an explicit ticker form
        (`NYSE: X`, `$X`, uppercase) or an ORG span resolving to the company, weighted
        by title/lead presence and share of ORG spans; `other` when only a bare
        ticker-as-word or a different entity matches; `mention` for a name in a roundup;
        else `unclear`. (3) Store verdicts in a new additive RESULTS table, key
        `(article_id, ticker, run_id)`, with `verdict`, `confidence`, `model`,
        `evidence`, the class and `match_reason` seen; created idempotently, no existing
        table touched. (4) Export a CSV with exactly `article_id, ticker, verdict,
        confidence, model, run_id, evidence` for upstream's audited apply pass. (5)
        Measure against upstream's hand labels (its shared sample, its fresh validation
        set, the `loose_name` hand sample of 10 News Corp pairs and 20 pairs of other
        companies, which is the **first pilot**: 413 of the 448 `loose_name` pairs, held on purpose upstream, and 42 URLs
        that also name another tagged company): precision/recall per verdict and the
        `unclear` share. (6) The string rule alone is expected to be weak (it rescued 2
        of 15 correct `review` pairs), so NER evidence, then T-130's opt-in judge on the
        `unclear` remainder only (calls budgeted under T-117), decide whether this ships
        a rule or only a report.
      - **What a verdict does:** here, existing RESULTS rows are purged (T-131's
        per-table rule, reversible) only on a confident `other`; `mention` and `unclear`
        keep their rows until a measured rate says otherwise. Upstream applies `about`
        above a threshold **set jointly after the measurement** as `accept`; nothing is
        released on an unmeasured verdict.
      - **Done when:** every selected pair has a verdict and the class seen; the report
        states precision/recall per verdict against the hand labels and the `unclear`
        share; the CSV round-trips (upstream can apply it); purges, if any, are recorded
        and reversible; its columns are added to the T-135 guard list, with a test where
        one is missing (the run fails with the message); an FR is added in the PR that
        ships it; the verdict/`other`
        naming collision is documented in `docs/category-taxonomy.md`.
      - **Depends on:** T-128 (met), T-139 (D5, D6); **T-135** (met, #109: the
        schema guard exists; this task adds its columns to the guard's list,
        `articles.prev_fetch_status` and `discovered_urls`, in `SOURCE_REQUIRED_COLUMNS`). **Blocks:** T-117 (its R and
        N are measured after these pairs are decided). → step 2b.
- [x] **T-117** **Superseded 2026-10-28 → Work item 18** (re-scope D11–D24: triage per unique body replaces the relevance gate, calibration and re-label; not done). *(run on the corrected corpus, after T-128 and T-130; **thresholds confirmed by the owner 2026-10-22, recorded in T-139**; what stays for T-117's sign-off is the measured R and N, not the rule)*
      Prototype the relevance test as string matching over SOURCE `body_text`
      + title (company name with aliases; uppercase/explicit-form ticker only
      — see the plan's design constraints) and **measure it**: share of the
      corpus failing it, cross-tab vs. current labels, and its false-reject
      rate, using T-130's judge. Report **R** (share of the cleaned corpus
      judged not about its tagged company) and **N** (share of the remaining
      `other` rows they make up), then get the owner's sign-off on the
      build/drop rule in `PLAN.md` Work item 15 step 3. **The 5% / 20%
      numbers were confirmed by the owner on 2026-10-22** (kept at the placeholders). *Sample size:* to tell R near
      5% apart from a neighboring value, use a judged sample large enough for
      a tight interval — by the normal approximation n = 1.96²·p(1−p)/h²,
      a ±1.5-point half-width at p = 0.05 needs about 810 judged articles
      (and N is estimated from `other` rows only); re-derive with
      `docs/evaluation.md`'s floor logic before spending judge calls. Two
      different samples, on purpose. (1) The **shared labeled sample** from
      `portfolio-data-mining`'s T-054 — `(url, ticker)` pairs from the *old*
      corpus, stratified by the old match mode, **labeled by hand by the
      maintainer in that repo (done, `data/ticker_tagging_sample.csv`), not with
      T-130's judge**; T-130's judge can instead be validated against those
      labels — is reused here **only for the gate's false-reject rate and the crawler
      rule's retention rate on known-correct pairs** (not built twice). (2) R
      and N describe the *cleaned* corpus, a different population, so they
      need a **fresh draw from the cleaned corpus** (unstratified, or
      stratified and weighted back to population shares); a stratified sample
      is never read as a corpus share unweighted. Budget judge calls for both. That sign-off is what makes
      T-119 and the `not_relevant` part of T-120 go ahead or be cancelled in
      place. Report; do not ship. → step 1, step 3.
      *Window and population (2026-10-22):* R and N describe the corrected corpus
      (98,462 `ok` articles on 2026-10-21, not 459,112), 63.4% of it published before
      2022; the fresh draw is stratified by the D7 window and weighted back to
      population shares, and R and N are reported both with and without the pre-2022
      articles.
      *Owner and date for the 5% / 20% numbers:* the repo owner (gamug) sets
      them at T-117 sign-off, before T-119 starts; until then they stay
      placeholders and T-119/T-120 do not begin.
      *Judge-call budget (planning estimate, re-derive before spending):*
      ~810 for R and N, ~300 to validate T-130 against the hand labels,
      ~400 for T-132's `undecided` remainder, ~400 for T-131's pre-purge check: about 1,900 under T-117's sign-off. T-121(b)'s
      stratified sample (3,300 base, which also provides T-123's baseline and replaces
      the earlier ~900) is funded separately by D10 (T-142), cap 3,800 with the top-up allowance; about 5,700 in total at most. Record the actual spend
      in `docs/evaluation.md`. LLM credentials (`LLM_API_KEY`/`LLM_MODEL`/
      `LLM_URL`) belong to the maintainer; CI tests stay hermetic (mocked
      LLM, no live calls).
- [x] **T-118** *(user decision)* **Decided 2026-10-14.** D1 = new
      `not_relevant` label in the existing `label` column (no table
      migration; added only if a gate is built, see T-117). D2 = yes, re-run
      the entire classification once the new rule is tested and the result is
      satisfactory (T-124). D3 = yes, fix the upstream ticker tagging first
      (T-128). Rejected for D1: a separate yes/no column (readers of `label`
      would still see `other`) and dropping such articles before
      classification (invisible, unauditable).
      *Plain-language record:* `other` currently means two things, "not about
      this company" and "about it, but no category fits". D1 gives the first
      its own label, but only if the post-fix measurement (T-117) says the gate
      is still worth building; D2 re-runs everything once so old and new
      results are never mixed; D3 fixes the cause (wrong tagging) before
      treating the symptom. → step 2.
- [x] **T-119** **Superseded 2026-10-28 → Work item 18** (re-scope D11–D24: triage per unique body replaces the relevance gate, calibration and re-label; not done). *(conditional on T-117's sign-off; cancel in place if the gate is dropped)* Implement the relevance gate in the category stage's feature
      step (CPU-only, before any model call), with hermetic tests covering the
      ambiguous-ticker cases (`ON`, `A`, `IT`, `ARE`) and alias cases (Meta,
      Berkshire, Bank of America). Depends on T-117, T-128. → step 3.
- [x] **T-120** **Superseded 2026-10-28 → Work item 18** (re-scope D11–D24: triage per unique body replaces the relevance gate, calibration and re-label; not done). *(conditional on T-117's sign-off, same as T-119)* Add the `not_relevant` label (D1 = a): `news_nlp/taxonomy.py`
      and the `docs/category-taxonomy.md` table (10 → 11 labels), the API
      response models, and the category judge's prompt/taxonomy in
      `news_nlp/eval/` so it can answer "not about the company". No table
      migration (`label` is unrestricted `TEXT`); confirm nothing else
      enumerates the label list (e.g. `sector_summary`, queries filtering on
      `label != 'other'`). Tell the knowledge-graph owner the vocabulary
      grew. → steps 2, 5.
- [x] **T-121** **Closed 2026-10-28, negative; the rest superseded → Work item 18.** (a) done; (b) stage 1 (720 calls) showed no threshold-and-margin rule passes; stage 2 not run (D10's other 3,080 calls unspent). The owner's tagging of the 720 judged articles then re-scoped the work (Work item 18). Offline calibration: simulate per-slug thresholds and a
      margin-over-runner-up rule against the stored leaf scores and existing
      `eval_verdict` rows (no model re-inference), on the confirmed (`accept`/`ok`)
      articles, or on the gated-in population if T-119 is built; pick parameters by
      per-slug precision; report the trade-off table. **Depends on T-133, not on
      T-117:** the population check of 2026-10-22 (`other` follows the corpus, 37.9% on
      confirmed articles) makes calibration the main lever, and it needs no relevance
      measurement. Two parts (design, sizes and the held-out split: PLAN
      Work item 15 step 4, the single place they are defined):
      **(a) free:** re-simulate on eval_run 43's stored verdicts where the stored label
      equals the argmax slug (score >= 0.6) for a first, coarse trade-off table, **and
      on the stored scores alone (no judge) report, per candidate rule, how many confirmed
      articles it re-labels** (population and per cell). **Power check before any paid
      call:** simulate, per candidate rule, the T-123 estimator itself: the paired
      difference of the weighted `other` precision (a ratio of weighted sums, because the
      candidate re-labels items away from `other` and so changes the denominator). The truth
      model: in the >= 0.6 cells, eval_run 43's measured per-cell agreement rates (they
      graded the stored label, which there is the argmax slug); in the 0.4-0.6 cells and the
      short-circuit stratum there is no measured truth distribution, so assume the measured
      `other` agreement plus an explicit share of articles whose truth is the argmax slug,
      swept over a stated range (for example 10% to 50%), and report the power across the
      range, not one number. Draw
      report-half samples of the planned size, take the within-cell bootstrap 95% interval,
      and report the share of draws whose lower bound is above 0 when the true gain is
      +0.05 (the power). Require power of at least 80%; a candidate below it is reported as
      "not detectable at this sample", not as a failure, and gets the discordant top-up
      (**verdict rule, fixed after T-121(a), 2026-10-23:** a power within 0.10 of 80% is
      re-run at 1,000 draws or more and reported with its Monte Carlo standard error; it
      is "detectable" when power - 2 SE >= 80%, "not detectable" when power + 2 SE < 80%,
      and otherwise "borderline", which is treated like "not detectable": it takes the
      top-up. The verdict uses the null-calibrated power, because the percentile interval
      is mildly liberal at 30-45 articles per cell: the lower bound is compared with the
      value that only 2.5% of zero-gain draws exceed, not with 0, and never with less than 0.)
      (identical labels have identical verdicts, so the paired difference needs only the
      discordant articles judged; a stratified extra draw of them, within D10's top-up
      allowance). **The top-up is drawn only after the rule is frozen and only from
      articles outside the fit half** (the fit-half items were used to tune the rule, so
      judging them again would be optimistic; they are about 2% of the confirmed population
      and are removed from the sampling frame), stratified by cell, and weighted back to
      the discordant population of its own strata, never pooled unweighted with the base
      report items. The rough count rule `g >= 1.96 * sqrt(d)` is an approximation for a net
      accuracy count and is not used for this ratio. **Read-only:** the query joins
      `article_category` to the confirmed class (`accept`/`ok`) through the read-only SOURCE
      attach and reads stored scores only, opened `mode=ro` with `query_only`, as
      `scripts/measure_corrected_corpus_2026_10_22.py` does; it writes nothing to RESULTS.
      **(b) paid, funded by D10 (T-142), not by T-117's sign-off:** the stratified,
      split judged sample, drawn with T-141's code. **Depends on:** T-141 (code), T-142
      (spend), T-121(a). Fit on the fit half only; every reported number states its half.
      **Deliverable added after T-121(a) (2026-10-23):** for every candidate rule, the
      population-weighted share of the articles it would re-label (away from `other`) that are
      truly `other` in the judge's view (`ideal_slug`), with a 95% interval, on the report half,
      and the pass/fail check against the bound of T-121(a) recomputed on the real counts (at
      most about 341 truly-`other` articles among at least 5,718 re-labelled, i.e. 2.1-5.5% for
      the rules studied); also the separation (AUC) of truly-`other` from truly-slug articles by
      the maximum score, the margin and the group score. **The AUC is the main deliverable:** the stored
      verdicts already put the 0.5-0.6 band at 0.19 truly `other` (0.09-0.31) and the 0.2-margin
      subsets at 8-9% (25-36 judged), so a uniform rule passes only if a margin or score isolates a
      2-5.5% subset; oversample the `m0.20` subsets. The power check (which in T-121(a) gives every rule the 0.5-0.6 band average) and the
      discordant top-up are rerun on each rule's own measured removed-set rate before the rule is
      frozen.
      **Short-circuit stratum:** 30 fit / 30 report; it informs only the
      `CATEGORY_GROUP_FLOOR` decision. **Target level, fixed now:** T-123's `other`-precision
      target is the pooled, population-weighted report-half figure over all confirmed
      articles; per-slug and per-cell figures are reported but are not the target. The seed,
      band edges, per-cell populations and the half assignment are stored in the
      run's recorded config (the article ids are already in `eval_inference`), together with the judge model id, its temperature and
      seed settings, the prompt hash and the date, so a later re-judge can tell a model
      change from a wording change; the draw can be reproduced. → step 4.
      **(a) done 2026-10-23** (`news_nlp.eval.calibration`,
      `scripts/simulate_category_rules_2026_10_23.py`, `docs/evaluation.md`'s 2026-10-23
      follow-up): 32 uniform rules (threshold x margin) re-label 0 to 45.5% of the 93,330
      confirmed articles (per-cell grids with `--cells`). On eval_run 43's `accept`
      verdicts a stricter threshold lowers overall accuracy and raises `other` recall at every
      threshold from 0.65, and lowers `other` precision with a paired 95% interval below 0 from
      0.75 up; only loosening rules (`t0.50`, `t0.55`, with or without a margin) can raise it.
      **T-123 also bounds `other` recall (and accuracy, macro F1) at -0.02, and the measured
      aggregates fix what that costs, with no truth model:** with `other` precision 0.319 and
      recall 0.661 on 35,368 `other`-labelled articles, a loosening rule passes iff its
      re-labelled set holds at most 341 truly-`other` articles and at least 5,718 articles:
      **at most 2.1% (`t0.50`), 2.3%, 3.5% (`t0.50+m0.20`), 4.3% (`t0.55`) or 5.5%
      (`t0.55+m0.20`) of it truly `other`**, against 7.5% (>= 0.8) and 15.4% (0.6-0.8) measured among the
      articles the shipped rule labels with a slug. **The stored verdicts do measure the 0.5-0.6
      band** (the judge gave an `ideal_slug` to `other`-labelled items too): 12 of 78 judged
      (weighted 0.19, 95% 0.09-0.31) are truly `other`. Per rule, among the judged articles it
      would take out of `other`: **`t0.50` 0.19 and `t0.50+m0.10` 0.19 (interval excludes the 2.1% /
      2.3% cap: they fail); `t0.50+m0.20` 0.079 (4/36), `t0.55` 0.166 (5/38) and `t0.55+m0.20`
      0.087 (2/25) have intervals that include their 3.5% / 4.3% / 5.5% cap: not excluded**
      (point recall change -0.045 / -0.078 / -0.032 against -0.02). The power tables use the band
      average for every rule. Accuracy does not bind (within -0.02 at `slug_fraction`
      0.25-0.75; measured 0.534); macro F1 is not modelled. The power check (truth model calibrated to the measured baseline;
      the swept quantity is the truly-`other` rate of the 0.5-0.6 band) says the precision test
      alone is passed by `t0.50+m0.20`, `t0.55+m0.00` and `t0.55+m0.20` (power 0.94-1.00) and not
      by `t0.50+m0.00`/`m0.10` (0.65, 0.75; calibrated 0.71-0.74), and that it is the recall
      tolerance that binds. **Deviation from the wording above, recorded here:** the < 0.6 cells
      and the short-circuit stratum are not given one flat share of argmax-slug truths (a flat
      truth gives every rule a precision gain of exactly 0 and, with a share, contradicted the
      stored baseline); they are parametrised by their truly-`other` rate, constrained to the
      measured 0.319. **Staged sample for the margin rules (sizing from
      `news_nlp.eval.calibration.items_to_decide` and `design_effect`, T-121(a); the weights of the stored
      run give a design effect of 1.5-1.6, T-141 reports the sampler's own):** judged articles needed
      inside a rule's removed set, one-sided 80% power: to fail it at the stored rate 233
      (`t0.50+m0.20`) and 565 (`t0.55+m0.20`) at 95%, 339 and 844 at 98.75%; to pass it if its true
      rate were half its cap 893 and 548, at three quarters of the cap 4,001 and 2,452. So: stage 1
      judges 350 articles inside each `m0.20` removed set (inside the 3,800 cap of T-142); a rule whose
      one-sided **98.75%** lower bound of the truly-`other` rate is above its cap stops there (the
      stricter level pays for two looks on two rules); a rule whose upper bound is below its cap
      continues to the joint T-123 check (that is a screen, **not a pass**: the cap is the necessary
      point-estimate condition, T-123 needs the *lower bound* of the recall change within -0.02); one
      that is still undecided gets a second stage of up to 500 more only while the running total stays
      inside the cap; the owner is asked before any stage that would exceed it. The final pass/fail of
      a rule uses the pooled sample at the full 95%. **T-121(b) therefore reports, per candidate rule, the truly-`other` rate of
      the set the rule would re-label with a 95% interval, compared with the bound above (the
      pass/fail check), and how well the maximum score, the margin and the group score separate
      truly-`other` from truly-slug articles (an AUC); per-slug thresholds are tested on that
      data.** Its top-up design waits for the frozen rule.
- [x] **T-141** *(new; done 2026-10-23, `news_nlp/eval/calibration_judge.py`, `calibration_sample.py`, `prompts/calibration.md`; `docs/evaluation.md` "Calibration judge and sampler")* Build the calibration judge and sampler T-121(b) needs, in
      `news_nlp/eval/`, opt-in like T-130 (not in `sampling.STAGES`' default set, not in
      `--stage all`, `ExperimentSpec`'s `Literal` untouched; snapshot test on the default
      stage list). The judge returns the **best-fitting label from the full taxonomy,
      `other` included** (the truth label), stored under its own `eval_verdict.task`
      value; "argmax slug fits" is truth == argmax. Sampler: stratify by predicted slug x
      score band plus the level-1 short-circuit stratum, seeded fit/report halves,
      per-item population weight. Hermetic tests (LLM mocked), including: the verdict-reuse
      key `(article_id, task, experiment, prediction_json)` never lets a stored-label
      verdict satisfy a truth-label lookup, and the same seed reproduces the same draw.
      **The judge wording is frozen in this task, before any sample is judged;** a later
      change to the wording or to D1's label set invalidates the sample and costs a
      re-judge, to be budgeted separately. Needs the `eval` group; no LLM calls in CI.
      **Blocks:** T-121(b). → step 4.
- [x] **T-142** *(new user decision D10, opened 2026-10-22; **decided 2026-10-22: approved, cap 3,800 judge calls, credentials and budget in place**)* The spend for
      T-121(b)'s sample: **default, approve up to 3,800 judge calls** (the
      3,300-call base sample plus a 500-call allowance for the discordant top-up) with the maintainer's
      LLM credentials, decided on its own and not at T-117's sign-off, so calibration
      does not wait for the relevance work. (Before the answer, the default was conditional on credentials and a budget already in
      place in writing, so there was no spend by silence.) Hard
      cap at 3,800 calls (base 3,300 plus the top-up allowance of 500); the actual
      spend is logged against the cap in `docs/evaluation.md`; the run stops at the cap. Blocks T-121(b). The rest of the T-117
      budget (R, N, T-130 validation, T-132, T-131's check, about 1,900) stays with T-117.
      **(b) stage 1 executed (2026-10-23, eval_run 45, 720 of the 3,800 calls of D10;
      `docs/evaluation.md`'s "the judged calibration sample, stage 1"):** the blind truth-label
      judge on the sub-cells inside the candidate removed sets. **Deviation from the staged
      design above, recorded:** the 0.5-0.6 band cells were split into sub-cells A, U, V, W by the
      candidate rules (40 / 24 / 16 / 10 draws per slug, the cell's 90 unchanged) and every drawn A,
      U and V article was judged (720), instead of 350 inside each `m0.20` set, so that a removed set
      is a union of whole sub-cells and each rate is a plain stratified estimate; the third rule
      `t0.55+m0.00` was not in the staged design and sits in the same sub-cells, so the stop level
      is one-sided **99.17%** (three rules, two looks) rather than 98.75%. **All three candidates
      fail T-123's cap**: truly-`other` rate 0.287 (`t0.50+m0.20`, cap 0.035), 0.285
      (`t0.55+m0.00`, cap 0.043), 0.335 (`t0.55+m0.20`, cap 0.055), 99.17% lower bounds 0.218,
      0.208, 0.240 (5,000-repetition bootstrap; 1,000 would put the stop bound on the 8th draw). **Which verdicts need the anchored baseline (precision 0.319, recall 0.661):**
      `t0.50+m0.20` removes about 2,121 truly-`other` articles at its lower bound, more than the 1,866
      the recall tolerance allows even if T were all 93,330 confirmed articles, so it fails whatever
      the baseline; `t0.55+m0.00` (about 1,655) and `t0.55+m0.20` (about 1,488) fail unless T is at
      least 82,750 and 74,400 (89% and 80% of the population); T is at most about 42,000 (the 35,368
      stored-`other` articles plus the slug-labelled ones at the anchored judge's 15.4% / 7.5%), and
      about 55,000 at three times those rates, so they would flip only if the slug-labelled
      articles were about two thirds truly `other`; the baseline is not measured under the blind
      judge. **Measured design effect 2.6-2.8, about 1.7 times the 1.5-1.6 the sizing above
      assumed:** scale its "judged articles needed" by about 1.7. **The AUCs (maximum score 0.42,
      margin 0.40, group score 0.63) are measured only on sub-cells A, U and V**, a restricted range
      (every article scores 0.5 or more with a margin of 0.2 or more, or scores 0.55 or more): they
      say no score isolates a clean subset inside the candidate sets, not that no score separates
      the classes across 0.4-0.6 (the 0.4-0.5 band and W were not judged). No uniform
      threshold-and-margin loosening rule can pass; T-122 has no rule to ship from this family.
      **Stage 2 (2,580 calls, 3,080 of D10 unspent) was not run.**
- [x] **T-122** **Superseded 2026-10-28 → Work item 18** (re-scope D11–D24: triage per unique body replaces the relevance gate, calibration and re-label; not done). Ship T-121's chosen rule as constants beside
      `CATEGORY_CONFIDENCE_THRESHOLD` in `news_nlp/taxonomy.py` and apply it
      in the category stage, with tests. **Also extract the decision into one pure
      function** (`group_score`, nine leaf scores -> label, winning score) in
      `category_stage.py`, used by `CategoryInference.predict_batch` and by T-124's
      re-label. Today the floor is inline in `predict_batch` and the threshold sits in
      `classify_category_scores`, which takes logits, so neither can be applied to stored
      probabilities. Test: under the *current* constants the function reproduces every
      stored label of a fixture of rows before the new constants are switched on. **The
      fixture is synthetic** (CI has no real store): boundary rows below, at and above the
      floor, at and either side of the threshold, a zero-filled third group, and exact ties.
      The real-store check stays the measurement script's non-CI run. **Tie-break, defined
      here:** an exact tie between two leaf scores goes to the earlier slug in
      `CATEGORY_SLUGS` order. Production's `max` over the candidate order (top-2 groups,
      ranked by group score) breaks a tie differently; routing `predict_batch` through the
      pure function makes the stage follow the taxonomy order. Measured on the real store
      (2026-10-22): 141 of 459,112 rows have an exact tie for the highest leaf score, all
      below 0.5 (measured maximum 0.490; two equal scores cannot reach 0.6), so under the current threshold
      no stored label depends on the tie-break; it matters only for a candidate rule whose
      threshold is at or below 0.5 and for which slug a tied article is assigned to in
      T-121's argmax cells. A test covers it. → step 4.
- [x] **T-123** **Superseded 2026-10-28 → Work item 18** (re-scope D11–D24: triage per unique body replaces the relevance gate, calibration and re-label; not done). Re-run `--stage category` eval; baseline = eval_run 43 (mlflow
      `549ddbdf`; the only category run whose judgements are stored, eval_run 21's are
      not) recomputed from stored judgements (restricted to articles still in
      the corrected corpus). **The baseline comes from T-121(b)'s
      stratified sample, report half only** (the stored run has only 68
      `other`-predicted confirmed articles in its unskewed bucket, precision about 0.37,
      interval 26-49%, too few; a +-5-point estimate would need about 359 such
      articles). Funded by D10 (T-142). **Metrics:** with the judge's truth label per sampled
      article, every rule (baseline and candidate) is scored by applying it to the
      stored scores, so per-class precision/recall/F1/`accuracy_ovr`, overall accuracy and
      macro F1 are computed against the truth label with the cell weights, below 0.6
      included. Candidate computed the
      same way, one side-by-side table with the complete constitution #12
      metric set, on the full and the gated-in population (**"full" means all confirmed
      articles**: rejected and `review` articles are outside the sample by design, being
      removed or decided by T-131/T-132, so the result says nothing about the 459,112); run
      `--check-regression`. → step 5 / acceptance criteria.
      *Success targets (confirmed by the owner 2026-10-22, T-139):* `other` precision at least 0.05 above the recomputed
      baseline, as a paired difference on the same report-half items (the same articles are
      scored under both rules, which removes most of the per-cell noise), with the lower
      bound of its 95% interval (bootstrap within cells) above 0, **and no trade bought
      with recall:** the paired differences in `other` recall, overall accuracy and macro F1
      each have a 95% lower bound no worse than -0.02 (tolerance confirmed by the owner,
      T-139); a rule that raises precision but breaks this is reported
      as a trade-off, not a pass; if a gate is built, false-reject rate at most 5% of articles
      the judge calls relevant; no per-slug regression beyond
      `--check-regression`'s tolerance. T-124 does not start unless T-123
      meets them and the owner approves. **If no candidate is detectable** even with the
      top-up, T-124 stays blocked: T-123 reports the point estimates and intervals, and the
      owner decides between accepting a rule on its estimate or opening a larger D10; the
      default is to stay blocked (no full re-run on an unproven rule).
- [x] **T-124** **Superseded 2026-10-28 → Work item 18** (re-scope D11–D24: triage per unique body replaces the relevance gate, calibration and re-label; not done). *(D2 = yes, decided 2026-10-14; mechanism per T-140)* Re-classify the
      **entire** category classification over the corrected corpus (every row remaining
      after the crawler re-tag; record the count at launch, it will no longer
      be 459,112), by a re-label of the stored scores unless the verification below
      fails or the rule needs inputs they lack (PLAN step 6), but only
      after T-123 shows a satisfactory, user-approved result — not before.
      **Always begins by moving the whole `article_category` aside**
      (snapshot, then `ALTER TABLE article_category RENAME TO
      article_category_v1`, T-025's pattern), whatever D4 decides: the
      category stage only selects articles that have no row yet
      (`fetch_pending_category_articles`: `WHERE r.article_id IS NULL`), so
      the "entire" re-run only happens if the current table starts empty.
      This is the **only whole-table rename in this work item**; T-131 never
      renames a table (it moves removed ids only), so the two cannot
      collide, and T-124 runs after T-131 so the snapshot is already free of
      removed articles. Articles the gate rejects skip inference. Needs a bulk
      re-classification driver (none exists in `scripts/`), a dry run on a
      scratch DB copy first (as T-025 did), and a measured throughput so the
      real duration is known before launching. Record the run in
      `docs/evaluation.md`. Runs after T-131. → step 6.
      *Articles with no results yet (2026-10-22):* 5,132 `ok` articles have no
      `article_category` row. Normal runs may classify them before this task under the
      old rule; that is acceptable, not a mix of old and new results, because this task
      moves the whole `article_category` aside first (those rows go into
      `article_category_v1`) and re-classifies every row of the corrected corpus,
      including them. Until it runs, readers see old-rule labels for them, as for every
      other row. The count at launch includes them.
      *Scale (2026-10-22):* the corrected corpus is 98,462 articles (21.4% of the
      459,112 originally classified), at most 32.8% if all 52,039 `review` pairs were
      released, so the cost projection (GPU path only, see T-140) is redone on that count (and the D7 window), not
      reused from the old corpus.
      *Mechanism and verification (D2b, T-140):* first re-infer a random sample of 300
      confirmed articles on a scratch copy, record the device and dtype of both runs (the
      stored scores date from 2026-09-12 or earlier, a CPU-era run; the fresh run will be
      on GPU), and compare the fresh nine leaf scores and `group_score` with the stored
      ones: report the maximum absolute difference and how many articles lie within it of
      a decision threshold. **Pass = zero label flips under the candidate rule on the 300**
      (zero flips in 300 bounds the true flip rate at about 1%, 95%, rule of three). If it
      passes, re-label every
      row from its stored scores with the new rule, **by calling T-122's pure function, not a
      separate re-implementation** (a text-only gate needs no inference);
      GPU inference then covers only the short-circuited articles if the level-1 floor
      is lowered, or every article if the top-2 selection, hypothesis wording (option D),
      the input length (option H) or the model changes. If any compared score differs,
      use full re-inference. The row counts that bound the re-label are in PLAN step 6
      (92,318 of 93,330 confirmed articles are re-labelable from level-2 scores; 1,012
      short-circuited).
      *Cost stop/go (GPU path only):* the scratch run reports articles per hour and a
      projected GPU duration for the articles that need inference; if the projection
      exceeds 72 hours on the available GPU, stop and bring it to the owner before
      launching (provisional threshold, not among the thresholds the owner confirmed on 2026-10-22; it applies only to the GPU path, which D2b makes the exception).
- [x] **T-125** **Superseded 2026-10-28 → Work item 18** (re-scope D11–D24: triage per unique body replaces the relevance gate, calibration and re-label; not done). *(contingent)* Option E: cluster the residual relevant-but-
      `other` rows and propose taxonomy additions. Only if `other` is still
      large after T-123 and the clusters are coherent; amends the taxonomy
      contract with the knowledge-graph module. → step 7.
- [x] **T-126** **Superseded 2026-10-28 → Work item 18** (re-scope D11–D24: triage per unique body replaces the relevance gate, calibration and re-label; not done). *(contingent)* Option H: classify on more than title + lead
      chunk. Only if T-123 shows misses concentrated in weak-lead articles.
      → step 7.
- [x] **T-127** **Superseded 2026-11-05 → Work item 18 T-148c** (D26: sentiment reads the summary or the company text, so the subject weighting of `_text_mentions_subject` is not used by the new pipeline; the v1 `article_sentiment` is replaced at the backfill, T-150; not done). *(separate PR)* Measure the sentiment-side impact of the same
      weakness: `_text_mentions_subject` matches a lowercased ticker as a bare
      word (`\bon\b`, `\ba\b`, …), so subject weighting is likely
      non-discriminating for the eight English-word tickers, and off-topic
      articles get sentiment attributed to the wrong company. Investigate and
      report before any fix; fold the multi-company case into Work item 17 (an
      article-level score on a multi-subject group carries a `multi` flag). → step 8.
- [x] **T-128** *(D3 = fix at the source, decided 2026-10-14; a prerequisite, not
      a follow-up)* **Done 2026-10-22.** Wait for the corrected corpus from
      `portfolio-data-mining` Work item 10 (its `T-053`–`T-060`).
      - **Why:** its matcher accepted a ticker **or the first word of the company
        name** anywhere in a URL, and a hyphen counted as a word boundary, so `A`
        matched `in-a-month` and `Bank of America` was reduced to `Bank`.
      - **Contract:** the quarantine schema and the hand-off format are specified
        **only in that repo** (T-057/T-058); this item links there rather than
        copying it. How this repo consumes it: the three pending queries
        (`fetch_pending_articles`, `fetch_pending_sentiment_articles`,
        `fetch_pending_category_articles`, `queries.py`) filter
        `fetch_status = 'ok'`, so quarantined rows are never selected.
      - **Shared sample:** the labeled `(url, ticker)` sample (that repo's T-054,
        hand-labeled there by the maintainer) is reused, not duplicated, by T-117,
        T-132 and that repo's retention measurement.
      - **Done when (met):** that repo's `T-058` is recorded done in its
        `CHANGELOG.md` (Work item 10, T-057 and T-058 `[x]`, 2026-10-15; the
        hand-off comment with the counts and the selection is on this work item's
        PR), and its Work items 12, 14 and 15 are closed. Evidence for the next
        steps: the corrected corpus is 104,614 `accept`, 51,295 `review`, 325,423
        `reject` pairs, and the 459,112 articles classified here split 93,330 /
        49,960 / 315,822 (measured on the real store 2026-10-22; plan, "corrected
        corpus"). Not claimed: that results already written are clean. The pending
        queries above stop new wrong-company rows; the rows already in RESULTS, and
        the summary and `sector_summary` queries that read them, are T-131's.
        Upstream's one open criterion ("this repo confirms that wrong-company rows
        no longer reach its pending queries") is answered by the first point; tell
        that repo, with the caveat. → step 2.
- [ ] **T-131** **→ Work item 18 P6: backfill (2026-10-28).** *(new; owner decision D4, decided 2026-10-22 at the recommended defaults, see T-139)* Plan and execute
      the cleanup of existing results for articles the crawler fix removes:
      (a) **derive the removed ids at run time with a read-only join** of
      RESULTS article ids to SOURCE (`match_class = 'reject'`, quarantined; about
      315,822 articles on the 2026-10-22 count), not from the 2026-10-15 CSV, which
      is stale: the `reject` class has moved since (448 `loose_name` pairs went to
      `review`, never-fetched sitemap rejects were deleted). Upstream's list of known
      false rejects (at least 433 pairs, `data/retag_known_false_rejects.csv` there,
      mostly News Corp `NWS`/`NWSA`) must be kept out of the purge; most of them now
      sit in `review`, which this task does not purge (T-132 decides them);
      (a2) **pre-purge check of the `reject` class's own error rate.** Upstream's hand
      check puts about 2.5% of `reject` as wrong (95% interval 0.4–12.9%), i.e. about
      7,900 (1,300 to 40,700) of 315,822 articles, and it dates from 2026-10-15, before
      448 pairs moved to `review`. Before any purge, draw 400 `reject` articles with
      RESULTS rows uniformly at random, label "is this article about its tagged
      company?" (T-130's judge, validated on upstream's hand labels, with the owner
      hand-checking the disagreements), and compute the wrong share with its 95%
      interval. **Stop/go (confirmed by the owner 2026-10-22, T-139):** go
      if the upper bound is at most 5% (10 wrong of 400 gives 4.5%; 15 gives 6.1% and
      fails); if it fails, purge only the `match_reason` sub-classes whose own
      sample clears the bound, or stop and bring it to the owner. Reversibility
      (the `<table>_removed` archives) is the second safeguard, not a substitute: a
      purge also forces a `sector_summary` regeneration. The 400 judge calls are in
      T-117's budget; (b) **per-table rule — move
      removed ids only, never rename a table:** for each of `article_sentiment`,
      `article_entities`, `article_category`, `article_summary` and the lean
      `articles` copy, in one transaction per table, copy the rows whose
      article id is in the removed list into an archive table
      (`<table>_removed`) and then delete them from the live table (stage
      tables first, lean `articles` last), dry run on a scratch DB copy
      first. A whole-table rename (T-025's pattern) would also remove the
      *retained* articles' results, make every article pending and force a
      full GPU re-run of sentiment/NER/summary that nobody planned, so it is
      not used here; the one whole-table rename in this item is
      `article_category` in T-124. (c) make `sector_summary` regenerate (bump
      `SECTOR_SUMMARY_FORMAT_VERSION` so every closed week is pending again,
      or delete only the affected `(sector, sub_industry, week)` rows) — it
      regenerates only at the next `--summarize` run; (d) the owner decided D4 on
      2026-10-22 at the recommended defaults: keep the `<table>_removed` archives, and
      delete only the affected `sector_summary` rows (no version bump), which (b) and (c)
      apply. Runs **before** T-124 (never together), so no reader sees a half-cleaned store. *(2026-11-05: T-117, T-124 and T-130 are superseded and not done; the judge calls and the `article_category` rename they carried are dropped, and the v1 move of the old tables is T-150, D19.)*
      → step 6b.
      *D4 (decided 2026-10-22):* the recommended defaults apply (keep the
      `<table>_removed` archives; delete only the affected `sector_summary` rows).
      T-131's execution may
      ship as its own PR; it stays in this work item because T-124 depends on
      its removed-id list and ordering.
- [ ] **T-129** **Re-pointed 2026-10-28: closes with Work item 18's close-out (T-150).** Close-out: annotate `SPEC.md` §13 item 19 / §14, add
      one FR per behavior that ships (results cleanup T-131; the gate, the
      calibrated rule and the relevance judge T-130 are superseded) in the PR that
      ships it (SPEC's own rule), update
      `docs/category-taxonomy.md`, `docs/evaluation.md` (dated follow-up),
      `docs/modules/news-nlp.md`; reconcile both architecture artifacts —
      never rename (constitution AI behavior #11); confirm T-131 is done (T-130 is superseded); then move this work item to
      `CHANGELOG.md` (AI behavior #15). → acceptance criteria.

## Work item 17 — Articles that name more than one company (decided 2026-10-22: its own work item, sizing first, D9 default)

**Re-scoped 2026-10-28 into Work item 18** (T-143): see the per-task markers below.

35.9% of the `ok` articles belong to a URL that has pairs for more than one ticker
(a floor); every stage keys on the pair. Size it before building anything. →
`PLAN.md` Work item 17. Task order: T-137, then T-138 only if D9 and the sizing say so, then T-138b (close-out).

- [x] **T-137** **Superseded 2026-10-29 → Work item 18** (re-scope D11–D24): T-144 sized the corpus by rules (unique bodies, companies named, scope candidates; no alias table, no group typing); the typing of groups (single-subject, multi-subject, comparison, transaction, incidental) and the hand-sampled validation are replaced by the owner's gold labels (P2) and the trained triage (P3); the alias table stays with T-132. Original text follows.
      2026-10-21, 9.2 and 9.4: 0 group, 1 mention detection, 2 type) on a
      stratified sample of about 2,000 groups.
      - **Inputs:** the groups (body hash, tickers, classes, reasons; computed here from
        the T-132 join or taken as upstream's read-only export), the NER spans, upstream's
        alias table.
      - **Steps:** group by body hash; resolve ORG spans, cashtags and `NYSE: X` forms to
        the universe, keeping unresolved organizations as `other` entities; type each
        group (single-subject, multi-subject, comparison, transaction, incidental), rules
        on the number of distinct companies, share of sentences and the lead first;
        strata: all-`accept` multi-pair, `accept` + `review`, all-`review`, single-pair
        control.
      - **Where:** a committed, rerunnable, read-only script under `scripts/` (both stores
        opened `mode=ro`, the T-135 guard called on SOURCE); the exact command and the
        result in a dated follow-up in `docs/evaluation.md`, as Work item 15 does.
      - **Sample:** allocation 500 groups per stratum (equal, so the small strata
        are measured; the all-`review` stratum is the large one, so pooled figures are
        re-weighted by stratum population and both are reported), the seed stored in the
        report, Wilson 95% intervals on every per-stratum and per-type proportion; the
        hand sample of about 20 groups per type is a **smoke test, not a measurement**
        (a proportion on 20 has an interval of about ±20 points) unless the owner sets a
        larger N, and no trust threshold is set from it; labels by the owner, or by a
        written rule recorded with the report.
      - **Unit and frame:** the sampling unit is the group (body hash); the report gives
        the distribution **per group and article-weighted** (each group weighted by its
        number of `ok` articles, weights stated), because the claim it replaces is
        "35.9% of `ok` articles". The frame is the groups whose pairs have NER spans in
        RESULTS; a sampled group with none is excluded, and the report states how many
        were excluded per stratum and whether the exclusion biases the frame (running NER
        on them is a cost and scope change, asked of the owner, not assumed). A stratum
        with fewer than 500 groups is taken whole and its count reported.
      - **Done when:** the real companies-per-article distribution per stratum replaces
        the 16.7% floor, reproducible from the command above; a hand sample (seeded with
        upstream's 42 News Corp multi-company URLs) reports role agreement per type with
        its interval, never one average. Report; do not ship.
      - **Depends on:** T-132's grouping and ORG-to-ticker resolution; D9 (decided, T-139).
- [x] **T-138** **Superseded 2026-11-05 → Work item 18 T-148b and T-152** (D27: the text of each company comes from the one-segment-per-company parser and a trained attribution model, not roles and fact tables; D28: the company snapshot is one row per (body, company), so no `multi` flag on the old result tables is added; not done). *(conditional on T-137's result; D9 decided)* Build the stages the sizing justifies:
      roles, sentence attribution with offsets (a sentence naming two companies is
      flagged shared and keeps both), facts per `(article, company)` from attributed
      spans only, and the `multi` flag on every article-level score of a multi-subject
      group. A `review` pair is released on its own evidence, never because a sibling
      pair on the URL is `accept` (10,493 of the 52,039 pairs are in that situation). Scope and fact
      types set after T-137 and the knowledge-graph owner's answer; one FR per behavior
      in the PR that ships it. The `multi` flag is a schema change (upstream sketches
      `article_group`, `article_company` and `article_fact` tables, or a flag per result
      table): its name, type and place (a column per result table, or a side table keyed
      by `article_id`) are asked of the owner in that PR, not assumed from D9. Until it
      ships the `multi` rule is not in force; T-137's report is the statement of the
      risk.
- [ ] **T-138b** **Re-pointed 2026-10-28: closes with Work item 18's close-out (T-150).** Close-out of Work item 17: annotate `SPEC.md` §13 item 21 / §14, reconcile
      both architecture artifacts (never rename; AI behavior #11), then move the work item
      to `CHANGELOG.md` (AI behavior #15). T-137's sizing was absorbed by T-144 and T-138 is superseded in place (2026-11-05), so it closes with T-150 and waits on nothing else.

## Work item 18 — Triage-first pipeline: unique bodies, subject scope, per-company and sector news (re-scoped from Work items 15 and 17, 2026-10-28)

Plan: `PLAN.md` Work item 18. Decisions D11–D24 (T-143), D25–D30 (T-151). Each phase is its own PR; a schema
change is proposed to the owner in the PR that needs it (constitution AI behavior #10).

- [x] **T-143** *(decision record, 2026-10-28, amended 2026-10-29 with the owner,
      after hand-tagging the 720 judged articles; current list D11–D24)*
      **D11** the unit is the unique article body (body hash), its subjects found in the content.
      **D12** LLMs only label and evaluate; the pipeline stays on local pinned models.
      **D13** GICS sector + sub-industry. **D14** the knowledge graph keeps per-(asset, day)
      signals; sector news feeds `sector_summary`; market and economy news become `MARKET` /
      `ECONOMY` rows there. **D15** the per-(body, company) category taxonomy is the 9 labels +
      `reputational_harm`, `competitive_actions`, `investor_activity`, `peer_comparison`, with
      `other` (company content that fits none of the 13 labels; settled 2026-10-29: the
      investing news about no company that D15 first named is the scope `investing_general`, D22,
      and carries no category), `uncertain` (no confident category; a category-level label only,
      the scope never abstains, D20) and `unrelated` (not about the economy). **D16** `unrelated` bodies are flagged and
      skipped by every later stage. **D17** a multi-company body gets sentiment + category per
      company. **D18** labelling budget about **6,000** LLM calls, a **hard cap**: spend is
      logged in `docs/evaluation.md`, counted per task (`calibration_judge.calls_spent(conn,
      task)`, so it is not mixed with D10's 3,800) and the run stops at the cap; the revised
      owner re-tags the 720 under the new taxonomy (gold v2; the first tagging is tags v1); the prompt
      iterates on a fixed stratified 240 (at most 5 iterations, a 6th never without asking) and must
      agree with the owner's scope on **>= 90%** of the full 720 before silver (at most 4,320 bodies)
      is bought; the owner tags about 300 held-out gold bodies, at least 40 per scope *(D30 drew 200; D33: no top-up, no recall floor under 40 gold bodies)*. **D19** full backfill of the
      **81,961 unique `ok` bodies** (not the 459,112 stored rows); the stored rows are moved aside
      as v1 until the new tables are validated, and the results of the 377,010 quarantined pairs
      are purged (T-131). **D20** the scope triage is accepted only at **>= 90% agreement with the
      owner's held-out gold on all bodies** (every body gets a scope, no abstention), as a
      **corpus-weighted** accuracy (the held-out sample over-samples the rare scopes, so each
      item carries its sampling weight), **and recall >= 80% for each of the six scopes**, both on the point
      estimate with the 95% interval reported, so a rare scope cannot be ignored. **D21** each
      training round exports a stratified sample of about 300 low-confidence bodies for the
      owner's tagging (active learning). **D22** the scope list has six scopes: single company,
      several companies, sector, market / economy, **`investing_general`** (investing or
      market-advice content about no company and no sector: the owner's `other` tag; no
      per-company analysis, summarised in the `MARKET` row of `sector_summary`) and `unrelated`. **D23** a body
      where one company holds more than half of the subject mentions is **single + mentions**:
      scope single on the dominant company (summary and sentiment for it), the other companies
      stored as mentions only, with no per-company analysis. (gold v2 defines analysed vs mention by content, `docs/labelling-spec-v2.md`; the >50% count is the rules' feature only). **D24** P4 (per-company category) and
      P5 (sector sub-industry detection) are each accepted at **>= 80% agreement with the owner's
      held-out gold** (the owner confirmed on 2026-10-29: at least 80% agreement with the
      held-out gold, each). Laya (`convaiinnovations/laya`) was to join the main dependencies as
      a local yes/no checker; tried and dropped on 2026-10-29 (T-145). → PLAN WI18 "Decisions".
- [x] **T-144** *(P1, this re-scope's PR; **done 2026-10-29**, `docs/evaluation.md` 2026-10-29 follow-up: 81,961 unique bodies; rules agree with the owner's six-scope labels on 0.70 of the 720 (0.663-0.730), 0.78 outside `investing_general`: a development figure; mentions counted once)* Read-only sizing script
      `scripts/size_triage_2026_10_28.py`: unique bodies vs stored pairs, companies per body
      (name and explicit ticker forms, NER `ORG` spans matched to universe names), rule-based
      scope candidates (single, multi, no company: sector / market / economy / unknown) and a
      GICS sub-industry guess; `docs/evaluation.md` follow-up. Absorbs T-137's sizing. → P1.
- [x] **T-145** **Cancelled 2026-10-29 by the owner after the gold check** *(P1, same PR)* Laya
      yes/no check of steps (a)–(g). On the 720 articles tagged v1 (English checkpoint, revision
      `55cf4c4e`, zero-shot) Laya answered "yes" to almost every question: at the 0.5 cut its
      combined scope matched the owner's on 39 of 720 (5.4%), against 70% for the rules (six scopes); as a
      ranking it carried some signal for "several companies" / "one company" (AUC 0.82) and none
      for "unrelated" (0.24). The owner aborted the approach; the dependency and the check
      script were removed in the same PR, the sample run was not made. → P1.
- [ ] **T-146** *(P2; spend D18: hard cap 6,000 calls, logged in `docs/evaluation.md`, counted per task, the run stops at the cap; owner answers 2026-10-29 in PLAN WI18; Q1-Q10 answered)* Split:
      - [x] **T-146a** *(done 2026-10-29: `docs/labelling-spec-v2.md`, the owner's calls answered; PR #118, merge #117 first because the spec cites its script and CSVs; #118 then has nothing left to merge and is closed as merged through #117)* Labelling spec v2: scope (six), companies, sector, D15 labels, with examples
        for the owner (a guide like `category_labels_guide.csv`). Gold v2 and the held-out gold
        carry, per body: scope; the companies the body is about; a D15 label per company; the GICS
        sub-industry for sector bodies (D24 tests the last two). Tagging effort per body is
        therefore larger than scope only (PLAN WI18 Q7).
      - [ ] **T-146b** *(export/check built 2026-10-29 in PR #117, aligned with `docs/labelling-spec-v2.md` (T-146a, PR #118) on 2026-10-29: round-up mentions, no sector-level `sub_industry`, `unrelated` only as a scope; a test checks that the spec names every scope, label and column; `news_nlp.retag`, `scripts/retag_gold_2026_10_29.py export|check`, file `data/category_gold_v2_review.csv`, 720 rows with v1 label/note/`v1_scope_derived` and empty `scope`/`companies`/`company_labels`/`mentions`/`sub_industry`/`notes`; single = exactly one analysed company (others are mentions, D23); multi = every company labelled (owner, 2026-10-29); tickers and sub-industries are checked against SOURCE; `check` verifies all 720 ids present once; the LLM label stays in the key. Added 2026-10-30: `normalise` turns names and nicknames the owner typed in `companies` / `mentions` / `company_labels` into tickers (`news_nlp.company_names`, `data/category_gold_v2_review_tickers.csv`; unknown or ambiguous texts are listed and left as typed, never guessed), and `check --file` validates that copy (it refuses to overwrite a copy edited after the review file, `--force` overrides; `normalise --self-check` verifies every universe name resolves to its own ticker; `guide` rewrites only the guide CSVs, also after tagging has started; a returned file saved with another delimiter or code page is refused with a message, by `export`, `normalise` and `check`). Changed 2026-10-30 (owner's request, tagging started): no `companies` column (the analysed companies are the labelled ones in `company_labels`) and `,` instead of `;` in `company_labels` / `mentions` (a comma inside a name like `Booking Holdings, Inc.` is kept); `migrate --file NAME` converts a file of the first format into a new file (dated name, never over an earlier copy); `sub_industry` takes the 8-digit GICS code (table of the Wikipedia GICS page, `news_nlp/gics_codes.py`) or the name, required for sector, optional on every other scope (owner call 8); a missing review file names the similar files; a company outside the 509 of the universe (Macy's, Twitter) is kept as typed, counts as analysed, and `check` lists those names so a typo shows (owner call 9). Added 2026-10-30 (owner's request): every file to tag carries `spanish_title` / `spanish_text` after `text` (`news_nlp.translate`, `translate` and `export` commands, cache `data/gold_translations_es.csv`, paid DeepSeek translation approved for this, about 1 USD per 720 bodies, logged in `docs/evaluation.md`; `export` / `translate` print the count and a token estimate and buy nothing without `--yes` (at most `--max-new`, default 800); a failed article or a stopped run (bad key or balance, 5 failures in a row) writes nothing and a re-run pays only for the missing ones; the cache key holds the model and prompt tag (the 720 legacy rows were stamped once with `translate --adopt-legacy --yes`, no cost; the cache file is replaced atomically) and every paid call is counted, retries included; the same applies to the held-out gold T-146e, about 300 bodies, and to any later tagging export). Open: owner: close `data/category_gold_v2_guide.csv` in Excel and run `guide` (the copy on disk predates the column rows), then tag (T-146a is merged); the re-measure of the rules on gold v2 runs once the file is tagged.)* Re-tag export of the 720 with the tags v1 shown as guidance, and the importer
        for the owner's new columns (protected like `gold_review`: refuses to overwrite a tagged
        file); gold v2; re-measure the rules' 0.70 on it (`gold_scope` maps v1 tags, v2 uses new
        columns).
      - [ ] **T-146c** Revised LLM labelling prompt and gate: iterate on a fixed stratified 240 of
        gold v2 (at most 5, about 240 calls each), then the full 720 (480 new calls) at >= 90% scope agreement. The 240 is stratified on the
        scope the v1 notes imply (`v1_scope_derived`), because gold v2 scopes do not exist before the owner tags; it is not a random
        sample, since the owner's already tagged rows come first. With a few items per rare scope it gates the overall figure only
        (per-scope recall is judged on the held-out gold). Silver D15 labels and sub-industries are bought in P2 only if
        a second gate on the 240 (per-company category and sub-industry agreement with gold v2,
        threshold >= 80%, owner answer Q8) passes; otherwise silver is scope-only and P4/P5 labels
        come from the owner's gold in later rounds. Budget: silver <= 6,000 - 240k - 480 for k
        iterations (k = 5 gives 4,320); judge repair re-calls count against the 6,000.
        - *The fixed 240 (owner answer 2026-11-04).* `scripts/draw_gate240_2026_11_04.py` draws it once, quotas sector 9,
          market_economy 9, unrelated 31, investing_general 40, multi 60, single 91 (`retag.GATE_QUOTAS`); it stops, writing
          nothing, if the draw is not 240 or a stratum holds more tagged rows than its quota. The set is
          `data/gate240_ids.csv` (local, git-ignored); the tracked gold files are the train set `gold/category_gold_v2.csv` (240 + 44 sector) and the held-out test set `gold/category_gold_v2_test.csv` (200, tagged by the owner and migrated 2026-11-07, PR #142), both added by #123, and every later scope feature reads those two (owner call 2026-11-05); a third, `gold/company_segments_train.csv` (T-148b2), holds automatic attribution labels, not tags; the gold is built from the gate file, `merge` is only a copy into the 720-row review file. The owner tags it in `data/gate240_to_tag.csv` (the review file's columns,
          Spanish aid after `text`, existing tags included; never overwritten); `merge` copies its tags into the review
          file as a new `_merged_` file, and the newest one is the current review file. The other 480 of the 720 can wait until
          the 240 passes; the full 720 is needed only for the confirmation run and the rules re-measure. Workflow, conflicts and `merge --gate-wins`: `docs/labelling-spec-v2.md`, "The fixed 240 of the prompt gate".
        - *2026-11-05 owner calls after tagging the 240: company labels allowed on sector/market_economy/investing_general bodies, several labels per company, `sub_industry` as GICS codes of any level and several per article (the gate compares by shared prefix); see `docs/labelling-spec-v2.md` calls 11-13. Because the 240 hold only 16 sector bodies, 44 more sector-like bodies are drawn by free rules into `data/sector_extra_to_tag.csv` (`scripts/draw_sector_extra_2026_11_05.py`; the owner tags scope, sub_industry and notes) for the sub-industry gate, which compares codes by shared prefix. 2026-11-05: both sessions (240 + 44, all valid) are in the tracked `gold/category_gold_v2.csv`, tags only (the article text stays out of git: news text, public repo), built by `scripts/build_gold_2026_11_05.py`.*
      - [ ] **T-146d** Silver: at most 4,320 bodies, bought only after T-146c passes, oversampling
        the candidates for the rare scopes.
      - [ ] **T-146e1** Prefilter rules for sector / market-economy / unrelated / `investing_general`
        (tests; measured on gold v2); the same candidates feed T-146d's oversampling.
      - [ ] **T-146e2** Export/import of the 50 prefiltered articles the owner checks (protected like
        `gold_review`); a separate set from the ~50 double-tagged and the ~60 random misses.
      - [ ] **T-146e** *(drawn 2026-11-05 as 200 bodies, D30: `gold/category_gold_v2_test.csv`; **tagged by the owner and migrated 2026-11-07, PR #142: 200 of 200 valid (118 single, 47 multi, 11 market_economy, 9 sector, 8 unrelated, 7 investing_general); measured on it (`report_wi18 scope --gold --out gold`): the rules' scope agreement 68.0% [61.3, 74.5] weighted to the corpus, single recall 91.0% [87.1, 94.4] pass, multi recall 52.5% [42.8, 63.5] fails the 80% bar, the other four scopes not gated (9, 11, 8 and 7 gold bodies, D33)**; a scope under 40 gold bodies is not topped up and has no recall floor, D33; the "about 300 bodies, at least 40 per scope" below is the original plan)* Held-out gold, drawn once before
        any training. *As drawn (D30): all 39 bodies the rules call no-company, sector, market or economy,
        the rest a seeded random draw from the 1,000-body sample; the original plan below (candidates
        prefiltered by custom rules, 50 of them checked by the owner first (Q6), ~60 random bodies from
        outside the filter (Q9), about 50 held-out bodies tagged twice (Q4)) is not what was drawn.*
- [ ] **T-147** *(P3)* Triage stage on unique bodies: body-hash grouping (T-132), the scope first (NER then runs on
      `single` and `multi` bodies only, D25), from a local classifier (a fine-tuned small encoder reading the text, with the
      `news_nlp.triage` rule features as inputs) trained on silver and gold, with a calibrated
      confidence; **accepted at >= 90% corpus-weighted agreement with the held-out gold on all bodies and
      recall >= 80% for each of the six scopes that has >= 40 gold bodies, the others reported with their interval (point estimates with intervals; D20 as amended by D33)**, the
      complete metric set per scope (AI #12). Each round exports a stratified sample of about 300
      low-confidence bodies for the owner's tagging, fed back into training (D21), until the
      target holds; `unrelated` flag-and-skip; tables proposed to the owner in the PR; its own
      FR. **Model governance (constitution AI #1, #12, Work item 8):** the encoder is a new pinned
      checkpoint in `config/pipeline_models.json` with its revision, its selection is justified
      against alternatives like every other model, the `ExperimentSpec` stage list is extended
      deliberately (a snapshot test), and the complete metric set is reported per scope. → P3.
- [ ] **T-148** *(P4)* Texts, sentiment and category per (body, company), steps 3-5 of D25; each
      sub-task its own PR and its own FR when it ships. Single-company bodies use the whole summary;
      a dominated body (D23) is single + mentions; **the per-company category is accepted at >= 80%
      agreement with the owner's held-out gold (D24, confirmed)**. → P4.
      - [ ] **T-148a** Summaries: the pinned local summariser (`c_summary`) on every non-multi body
        (single, sector, market / economy, `investing_general`), with the input the pipeline
        already builds minus the NLP features (title and body only: sentiment now follows the summary).
        For the cut (T-154) it writes chunks under `data/cut/summaries/` and resumes from the last chunk
        (**code written 2026-11-06**: `scripts/cut_summaries_2026_11_07.py`, chunks of 250 bodies, stop with Ctrl-C and run the same command to resume, `--status`; **launch: `uv run scripts/cut_summaries_2026_11_07.py --require-cuda` (PR #132: the first line names the device, a tqdm bar counts bodies and moves per saved chunk; the device is a condition of the manifest, so a first run on the CPU would fix the cut to it, hence the flag)**; smoke-run on 58 bodies; full run about 29,113 unique bodies, about 70% summarised, 5 to 6 GPU hours; it starts the evening of 2026-11-07). **Two facts from the real window, 2026-11-06:** the rules read the stored NER `ORG` spans, and 30,906 of the window's 36,063 articles have them (none for 2026-09 and 2026-10, about half for 2026-08), so for the newest bodies the scope is read from the text alone, unlike the held-out sizing. **Measured (`scripts/cut_spans_effect_2026_11_07.py`, 500 bodies published since 2026-03 that have spans, scoped with and without them): the rule scope is the same for 500 of 500 (upper 95% bound of a change about 0.6%) although the company mentions change in 21 of them, and so is the summarise-or-not decision; the missing spans do not change the cut's scope.** Each chunk row records `spans`, and `--status` counts the bodies without. Bodies the rules call `multi`, `roundup` or `no_company` get no summary, and the rules never predict `investing_general` (D25 summarises it): `--status` after the first chunks shows the share, to decide before the long run. **Order against NER on the window (open, owner to decide before the long run):** (a) run NER on the window first, then this job, so every row has its spans; or (b) run now and let T-154 recompute scope and mentions from the final spans, using the chunks only for the summaries (`--accept-new-spans` keeps the chunks but their `rule_scope` stays the one read without the new spans; the mentions differ in about 4% of bodies). Until decided, T-154 must not read `rule_scope` from the chunks. Checks that
        a body longer than the model's window is chunked and reduced, as `hierarchical_summarize_batch` does. → D25, D26.
      - [ ] **T-148b** Step 4, the text of each company in a multi body (D27). Split:
        - [x] **T-148b1** `news_nlp.company_texts` (`label_segments`, `split_company_texts`, `segment_text`), the one-segment-per-company
          parser, with tests *(2026-11-05; a segment naming two companies at its start is `ambiguous`: no label, nothing inherited past it; a body is structured with at least 4 segments, 50% of them led by a company and 3 companies, the rule that finds 2,867 of the 81,961 unique bodies)*.
        - [ ] **T-148b2** Training set for the attribution model from those bodies (company name
          masked, previous sentence as context, all years). **Set built 2026-11-05** (`scripts/build_company_segments_2026_11_05.py`, `gold/company_segments_train.csv`, ids and positions only): 2,834 bodies, 33,624 labelled segments (21,682 `lead`, 11,942 `continuation`; ambiguous paragraphs have no row), 1 body in 10 as `val` (205 bodies before 2022, 65 from 2022 on: report the check on the 2022+ ones apart); the 33 structured bodies that share text with a tagged or held-out article are left out; the script stops and names the local git-ignored files it needs, and `tests/news_nlp/test_company_texts.py` pins the parser constants and the counts. Built 2026-11-06: the mention-and-recency baseline (`news_nlp.attribution_baseline`), the owner's label check (100 `lead` + 100 `continuation` paragraphs, `scripts/draw_segment_check_2026_11_05.py`, Spanish columns after `text` and a Spanish version of the previous paragraph), the faithfulness check (`scripts/draw_faithfulness_2026_11_06.py`: 30 narrative multi bodies drawn once, one blind file with the baseline's and the model's texts, shuffled), and the readers and figures of every owner file (`news_nlp.gold_metrics`, `scripts/report_wi18_2026_11_06.py scope|label-check|sentiment|faithfulness`, D32 bootstrap, refusing a half-marked file). **Rule fixed before the label check is read (owner, 2026-11-06):** if at most 80% of the `continuation` paragraphs are right they are dropped from training (train on `lead` only); the `lead` labels are kept in any case; the precision is recorded in PLAN D27. **Label check read 2026-11-06: `lead` 88.0% (80.2 to 93.0), `continuation` 38.0% (29.1 to 47.8), so the model trains on `lead` only; the 12 wrong `lead` labels are 5 multi-company paragraphs, 3 look-alike entities and 4 non-company text (a junk-paragraph filter is cheaper than fixing labels by hand, not done).** Still to do: the faithfulness file (190 texts, not the 30 assumed), then the model (T-148b3), `lead` only. Acceptance (D32): at least 80% of the companies named in the 30 with a faithful text (a company left without a text counts as not faithful), and better than the baseline on the same 30.
        - [ ] **T-148b3** The trained attribution model: a new pinned checkpoint in
          `config/pipeline_models.json` with its revision, selection justified against the baseline
          (constitution AI #1, #7, #12), the `ExperimentSpec` stage list extended deliberately.
          **Trains on `lead` rows only (D27, label check 2026-11-06)**: the training set is built with `kind == lead`, the rows used are counted and printed, and the run refuses to start unless `gold/segment_check_gold.csv` gives `gold_metrics.train_on_lead_only` true (so the rule is applied, not remembered).
          Bodies it cannot attribute are flagged `unextracted` and counted, never given a default text.
          A one-paragraph body is split into sentences and each sentence is attributed by the same model.
      - [ ] **T-148c** Sentiment over the texts of T-148a and T-148b (D26): after T-148b exists,
        draw the 180-text sentiment gold (60 per class, `scripts/draw_sentiment_gold_2026_11_05.py`: 120
        from summaries and structured multi texts, 60 from the texts the extractor writes for narrative
        multi bodies, `--extractor baseline` today and `model` once T-148b3 exists; the pool sizes and what
        is left out are in `data/sentiment_gold_pool.json`), the owner tags the price impact,
        the current model is measured on it, and fine-tuned on summaries only if it misses the D32 bar (overall agreement >= 80%, negative precision and recall >= 70%). **Measured 2026-11-06 (PR #131): overall agreement 60.4% [52.8, 67.6] fails the 80% bar, so the fine-tuning condition is met; the owner chose fine-tuning on summaries (2026-11-07, PLAN D35): 450 summaries to tag (`scripts/draw_sentiment_train_2026_11_07.py`, 150 per predicted class, balanced blocks, a validation slice of 90, never a gold or held-out body); the 180-text gold is scored once per run that could ship, every run listed; fallback if it still fails on 2026-11-11: the 2026-11-12 cut goes with sentiment marked as measured.** Covers T-127's question on real output.
        - [x] **T-148c1** (written 2026-11-07: `scripts/build_sentiment_train_2026_11_07.py`, `gold/sentiment_train.csv`; tried on the real files with trial marks; **`gold/sentiment_train.csv` does not exist until the owner's 450 marks are in and the builder is run on them: done here means the code, not the data**) Builder of the marks: reads `data/sentiment_train_to_tag.csv` and `data/sentiment_train_ids.csv`, stops unless every row is marked, the text hash is the one drawn and the ids match; writes the marks, the slice and the ids (no news text) to `gold/sentiment_train.csv`. **Today, 2026-11-07** (the owner expects to have all 450 marked today); written and tested on the untagged file.
        - [x] **T-148c2** (written 2026-11-07: `scripts/score_sentiment_gold_2026_11_07.py`, every run in `gold/sentiment_gold_scoring_log.csv`; the pinned model reproduces D32) Scorer of a candidate model on the 180-text gold: takes a model path or revision, reads the texts of `data/sentiment_gold_ids.csv`, scores them as the pipeline's stage reads a text, writes the predictions in the format `report_wi18 sentiment --gold` reads, and gives the figures for 2022 on and before apart (D35). **Today, 2026-11-07.**
        - [ ] **T-148c2b** Scorer hardening, its own small PR, **today or 2026-11-08, it does not wait for the owner's tags** and must be merged before the final scoring (a rule is applied, not remembered; as T-148b3's `train_on_lead_only`). Done when each of these holds, with its test:
          - a Hub model needs a 40-hexadecimal `--revision` (today any non-empty text passes, `main` included); a local directory is unchanged;
          - `--redo-role` requires a `--redo-reason`, written to the log row (new column `redo_reason` in `retag.SENTIMENT_SCORING_LOG_COLUMNS`);
          - the committed `gold/sentiment_gold_scoring_log.csv` gets the new columns with empty values in its baseline row (every other value of that row unchanged), and the committed-gold-files test moves in the same PR;
          - the report shows the agreement for texts over 128 tokens apart from the rest, as it does for 2022 on and before (D35);
          - the `final` run's report gives the paired gain over the **last** `baseline` row (same keys; `gold_metrics.paired_difference` clustered by body; the verdict of `gain_verdict`); the baseline's predictions are the stored `predicted` column, and the gain is refused if its agreement differs from that row's logged one; tests: a hand-made pair, a gain whose interval spans 0 (`not decided`), keys that do not match (refused);
          - the row logs `gain_verdict`, `gain_negative_precision_verdict` and `gain_negative_recall_verdict` (new columns; empty in the baseline row);
          - `test_the_pinned_sentiment_model_is_the_one_scored_as_final` is changed: the pin passes if it is the last `final` row's model and revision and that row's `gain_verdict` is `better`, or if it is the last `baseline` row's (a rejected candidate); cases: candidate pinned and better (passes), candidate pinned and not better (fails), candidate rejected and baseline pinned (passes), a pin that is neither (fails).
        - [ ] **T-148c3** Trainer of D35's recipe (base, replay of the earlier sentences, search on the validation slice, 3 seeds, a final run). Acceptance: the replay leaves out the sentences of gold, held-out and drawn-for-tagging bodies (by body hash) and prints how many; one run is timed first and the grid is reduced if 28 runs do not fit (D35); the tie rule of D35 picks the settings. **The trainer does not score the 180-text gold (the gold is scored once, by the scorer, as the `final` run of the published revision, T-148c4).** A batch never mixes summaries with sentences, and each epoch shuffles the batches of both kinds together (D35). The training settings are D35's (the earlier fine-tune's batch 16, weight decay 0.01, fp16, linear decay, no class weights, the last epoch's model), written in the trainer's config and printed at the start of each run. The final run trains on the 360 train rows plus the replay, with the fourth seed fixed in the code, and the report gives the spread over the three seeds (D35); its own validation agreement is printed next to the three-seed mean and spread, and the run stops before publishing if it is below the three-seed mean by more than 2 standard deviations or by more than 3 texts of 90, whichever is larger (not "below the lowest": that blocks a healthy run 1 time in 4). The trainer's validation agreement is reproduced by scoring the same validation texts through the scorer's reading (T-148c2: chunks of 510 tokens, averaged), within **1 text of the 90 (the two figures differ by at most 1.1 points)**, both printed, and the run stops if it is not (two inference paths, and a label order that is permuted but made of valid class names, would otherwise go unnoticed). **By 2026-11-08**; first run when the 450 are marked; timed then. **Timed 2026-11-07, GPU idle** (two epochs of one run at batch 8: 12.7 s and 8.6 s; summaries median 73 tokens, longest 108; the full 28 runs **about 15 to 30 minutes**, an order of magnitude, not a figure to schedule against): the full grid fits whenever the tags arrive, so the smaller grid is not expected; the date that limits is T-148c4's 2026-11-09, so the tags are needed by the morning of 2026-11-09 at the latest (developer's estimate: trainer plus publish plus final scoring take hours, not days). The trainer times its own first run again with the GPU idle and keeps the budget rule.
        - [ ] **T-148c4** Publish the fine-tuned model (Hub write token) and pin it in `config/pipeline_models.json`, its own PR. The candidate is published to the private Hub repo `gamug/FinBERT-financial-news-summaries` (D35: the owner accepted the name, 2026-11-07) and made public only in this PR; the `final` row and the pin name the commit SHA, not a branch or tag (D35). A second `final` scoring only for a defect in the run, never for the figure (D35). **The scorer's checks (SHA, redo reason, the over-128 split, the paired gain the pin rule reads) are T-148c2b, merged before this task's final scoring; the pin PR applies D35's rule to that gain (`better` pins; `not better`, `not decided` and `undefined` keep the baseline, owner, 2026-11-07; and no pin if the paired gain of negative precision or recall is `not better`).** The final run's own validation agreement is checked before publishing (D35). **By 2026-11-09**; the final scoring of the gold decides the fallback, with room before **2026-11-11**. The pin PR carries the `final` row: the model is scored as the `final` run **as the published Hub id and revision** (not a local checkpoint) and `config/pipeline_models.json` names that same model and revision when the candidate is adopted, or still the baseline's when it is not (`test_the_pinned_sentiment_model_is_the_one_scored_as_final`, changed in T-148c2b, enforces both). If the baseline stays, there is no pin PR for the candidate: its repo stays private and its `final` row stays in the log (D35); the owner decides later. The owner tags all 450 on 2026-11-07, no stopping rule.
      - [ ] **T-148d** Category over the same texts (D29), measured against the owner's gold. → D24.
- [ ] **T-149** *(P5)* Sector detection (GICS sector + sub-industry) for sector bodies;
      summaries of sector, market, economy and `investing_general` bodies feeding `sector_summary`,
      with `MARKET` / `ECONOMY` rows; **sector sub-industry detection accepted at >= 80% agreement
      with the owner's held-out gold (D24)**; its own FR. → P5.
- [ ] **T-150** *(P6, close-out)* Full backfill (old tables to v1, T-131), knowledge-graph export
      contract update (`portfolio_common.news_export`), `SPEC.md` §13 items 19 and 21, both
      architecture artifacts (AI #11), then Work items 15, 17 and 18 to `CHANGELOG.md` (AI #15,
      closing T-129 and T-138b). → acceptance.

- [x] **T-151** *(decision record, 2026-11-05, the owner)* **D25–D30**: the order of the stages,
      sentiment over the summary or company text, step 4 without an LLM, the snapshot tables, category
      on the same texts, the held-out 200. Full text: `PLAN.md` Work item 18. The four open questions were answered the same day as D31–D34 (all years disclosed; the bars of the extractor and of sentiment; no recall floor under 40 gold bodies; the backtest cut).
- [ ] **T-152** *(P5)* The company snapshot table (one row per (body, company): text, ticker, D15
      label, sentiment, `pub_date`) and the sector snapshot table (one row per (body, sector): text,
      sentiment), RESULTS, the old tables kept as v1 (D28, D19). **Columns approved by the owner on
      2026-11-06** (for the cut's files and as the base of the table): `body_hash`, `article_id` (first),
      `pub_date` (as stored), `scope` (rules), `ticker`, `role` (analysed | mention, D23), `text_kind`
      (summary | company_text | unextracted), `text_source` (summary | parser | baseline | model), `text`,
      `sentiment`, `p_positive`, `p_negative`, `p_neutral`, `sentiment_checked` (yes | no), the
      summary, extractor and sentiment model with their revisions, `provisional` (1), `run_id`. The types,
      the keys and the migration are proposed to the owner in the P5 PR (constitution AI #10); its own FR. → P5.
- [ ] **T-153** *(coordination)* Reply to `portfolio-financial-analysis` (`to_nlp_backtest.md`, their 8
      questions): the training-data facts (about 30% of the sentences and bodies are from 2022 on, the
      labels have no date cut-off, nothing limits hindsight), what is decided (D31, D34) and what is not, their question 4 (which bodies count: the `role` column
      separates analysed from mention-only, D23), question 6 (running the pipeline on a SOURCE they provide: not yet; the cut covers the stored corpus, which ends 2026-08-12, and the bodies they add are not in it), question 8 (the agreement figures travel with the output and are in `PLAN.md` D32), `pub_date` formats (24 and 20 characters and empty, UTC not yet checked),
      the backtest cut (T-154: provisional, delivered about 2026-11-16 if the 2026-11-12 go/no-go holds) and the latest date their runs can use, and the final tables once T-152 is approved. Written for the owner's approval; nothing is sent.

- [ ] **T-156** *(test hygiene, 2026-11-06)* Flaky test `tests/news_nlp/test_eval_calibration_judge.py::test_an_interrupt_cancels_the_queue_instead_of_paying_for_it`: failed once in CI (PR #129, `assert 4 <= 3`), passed on the next run. It counts calls made by worker threads against a 0.05 s sleep, so the bound depends on timing. Fix: bound by the number of workers plus one, or stop the queue with an event instead of a sleep.
- [ ] **T-157** *(Windows, T-148a, 2026-11-07)* The summaries cut's run lock probed a pid with `os.kill(pid, 0)`, which
  terminates on Windows (a resume crashed with `WinError 11`). PR #134. See `docs/db-topology.md`.
  - Done: the probe asks `OpenProcess` + `WaitForSingleObject` and answers dead only when the OS says so; the lock holds the
    pid and the process start time (a reused pid is taken over); Ctrl-Break stops like Ctrl-C; a failed `os.link` names its
    cause (NTFS only when the error says no hard links); the `win32`-only test in `tests/news_nlp/test_cut_summaries.py`.
  - Decision: no Windows CI job (it would need the torch wheel, which that test file imports); the Windows branch is
    verified when the cut is resumed on the Windows machine.
  - Closes when: the owner has resumed the cut once on Windows with a leftover lock and got no refusal.
  - Open: the takeover of a stale lock is not atomic (two starts at the same instant can both run: duplicated work, chunk
    files are atomic); a lock written by this version is read as stale by an older script (do not share an output folder
    across versions); closing the console window cannot be caught (the next start takes the lock over).
- [ ] **T-155** *(gold sets, `docs/gold_datasets.md`, 2026-11-06)* The builders that move the label check (**done 2026-11-06**: `scripts/build_segment_check_gold_2026_11_06.py`, 200 rows in `gold/segment_check_gold.csv`; **the extractor sample done 2026-11-06**: `scripts/build_faithfulness_gold_2026_11_06.py`, 190 rows in `gold/faithfulness_gold.csv` and `gold/faithfulness_coverage.csv`; the builder protects the tracked marks and coverage from silent change (`--overwrite-marks` to accept), ties the coverage to the draw (`faithfulness_bodies.csv` is required), writes the two files together with a put-back on failure, and `report_wi18 faithfulness --gold` reproduces the D32 figures from them; **the sentiment set done 2026-11-06**: `scripts/build_sentiment_gold_2026_11_06.py`, 180 rows in `gold/sentiment_gold.csv`, `report_wi18 sentiment --gold` reproduces the figures: agreement 60.4% [52.8, 67.6], fails the D32 bar; the gold keeps `pub_date`, protects the weights, and the report stops on a sample that is not the one drawn; **the held-out scope set done 2026-11-07** (PR #142: `build_gold_2026_11_05.py --set test`, 200 rows in `gold/category_gold_v2_test.csv` with the draw's `stratum` and `frame`; `report_wi18 scope --gold --out gold` reproduces the figures from the tracked file and tests pin the counts, the shape, the GICS codes offline and a digest of the tags; the builder refuses a different set of articles and, since the review of PR #142, a change to a tracked tag without `--overwrite-marks`); what is left of T-155 is the typed reader per gold file in `news_nlp.retag`, and a decision on where the only copy of the 180 texts, `data/sentiment_gold_ids.csv`, is kept (nothing enforces the backup), the extractor sample and the sentiment set from
  `data/` to `gold/`, as `build_gold_2026_11_05.py` does for the scope sets: label check
  (`gold/segment_check_gold.csv`), extractor sample (`gold/faithfulness_gold.csv`, keyed by article, ticker and text hash, not by
  `review_order`) and sentiment (`gold/sentiment_gold.csv`, with the draw's `stratum`, `extractor`, `predicted` and `population`).
  `text_hash` (`retag.text_hash`), for sentiment `revision` and for the extractor sample `revisions` (`retag.code_revision`) are written by the draws into their ids files and copied by the builders, never recomputed. Each builder refuses unless every row is marked and valid, writes marks and ids only (no news text), and ships a typed reader per gold file in `news_nlp.retag` (today only the column tuples exist); then `report_wi18` reads the
  `gold/` file, so the figures of D27 and D32 no longer need `data/`. `scripts/scaffold_gold_sets_2026_11_06.py` writes the empty
  files (tracked in `gold/`, local in `data/`); the draw scripts fill a header-only working file. Do it when the owner has tagged
  each set; the order to tag is on the page.
- [ ] **T-154** *(backtest cut, D34; the full plan continues around it)* A provisional output for the
      bodies of the backtest window (published 2021-12-31 or later) by about 2026-11-16: scope from the
      rules, NER for single and multi bodies, summaries (T-148a), company texts (T-148b), sentiment
      over the texts (T-148c), written (owner, 2026-11-06) as chunked files under `data/cut/` in the columns
      approved in T-152, resumable, no schema change; the stored corpus only (it ends 2026-08-12: the bodies the
      backtest team adds upstream are not in the cut); marked provisional, with the
      measured figures (rules' scope agreement on the held-out 200, extractor faithfulness, sentiment
      agreement on the 180) next to it. **Schedule (owner, 2026-11-06):**
      - *11-07:* the baseline (done) and the exports: the label check (200 paragraphs, translation with `--yes`)
        and, from the baseline, the faithfulness file, and `draw_sentiment_gold --pool-only` (scores the pool and shows a short stratum four days before the draw; D34 (d)); T-148a code (the summary run over the window's non-multi
        bodies, scope from the rules) and its start in the evening (about 7 GPU hours overnight; it does not
        depend on the tagging; wasted only on a no-go); the T-153 reply is drafted for the owner.
      - *11-07 to 11-08:* the owner tags the label check first, then the held-out 200.
      - *11-08:* `report_wi18 label-check` applies the rule of T-148b2; the attribution model trains (GPU, after the summaries; **order set by the owner in D35, GPU: the retraining first, about 20 minutes**).
      - *11-09:* the faithfulness texts (190 for the baseline, plus the model's) are exported (one blind file); T-153 is sent
        after the owner's approval, so the backtest team plans before the decision.
      - *11-09 to 11-10:* the owner tags the 30. *(the 180 sentiment texts were drawn, tagged and scored on 2026-11-06, 60.4%, PR #131; the 450 summaries for the retraining are tagged on 2026-11-07, D35.)*
      - *11-12:* decision, from `report_wi18 scope|sentiment|faithfulness`. After a go: NER, step 4 and sentiment over
        the window 11-12 to 11-14, hand-over about 11-16.
      **Checkpoint 11-09 (end of day):** the tagging of the label check and the held-out 200 (400 items) must be
      back, and the faithfulness file exported. If the held-out 200 is not back, the scope figure is reported as
      "not evaluated" and the cut still goes if the other three pass; if the faithfulness marks (the sentiment marks of the 180 are in; the retraining's dates are T-148c3/c4, fallback 2026-11-11) are not
      back by 11-11 evening, the decision on 11-12 is no-go and the backtest team is told (the week's notice). The
      developer reports the count of marked rows per file on 11-09, 11-10 and 11-11 (`report_wi18` refuses a half-marked
      file and names the missing rows).
      The owner's tagging is 770 items in six days (200 + 200 + 190 + 180). **Rules fixed beforehand (owner,
      2026-11-06, also in PLAN D34):** a sentiment figure whose interval straddles its bar ("not decided") still goes,
      with the interval printed; if the model is not ready on 11-09, the 190 texts are exported from the baseline
      alone (`--extractors baseline`) and the decision is on the baseline (the cut uses it for narrative bodies if it
      is at least 80% faithful, otherwise narrative multi bodies are left out of the cut and counted).
      **Go/no-go on 2026-11-12:** the owner's tagging (the held-out 200 for the scope, 200
      items of the label check, the 190 faithfulness texts and the 180 sentiment texts) is done and the extractor
      (or, if the model is late, the baseline) passes D32; otherwise the backtest team is told with a week's
      notice. Not in the cut: the trained scope classifier, the prompt gate and silver labels,
      category, sub-industry, sector snapshots, active learning, the full backfill. → D34.

## Status

**2026-11-05:** the owner fixed the 7-step pipeline order (D25–D30, T-151). P4 is split into T-148a–d, the snapshot tables are T-152, the reply to the backtest team is T-153, and the backtest cut for 2026-11-20 is T-154 (go/no-go 2026-11-12). T-127 and T-138 are superseded in place (their designs are replaced by D26–D28); T-132, T-131, T-129 and T-138b continue under Work item 18 unchanged.

**Work item 18 (2026-10-28): Work items 15 and 17 are re-scoped into it** (T-143 records D11–D24).
T-121 is closed (negative), T-116/T-117/T-119/T-120/T-122–T-126/T-130 are superseded in place, and
T-131, T-132 (shrunk to the alias table, 2026-11-05) and the sizing of T-137 (absorbed by T-144) continue under Work item 18; T-127 and T-138 were superseded on 2026-11-05. T-144 (sizing and the rules' check on the
720 articles tagged v1) is the first PR; T-145 (Laya) was tried and cancelled.

Closed Work items 1, 5, 6 and 8–14 are in `CHANGELOG.md` (Work item 5 closed 2026-10-14; its
only open task, T-041, was superseded by Work item 15).

T-010–T-016 (Work item 2, regression gate) remain blocked on the maintainer's
infrastructure decision — nothing in that item has started.

**Work item 15's own plan (filed 2026-10-14) is superseded by Work item 18**: T-116, T-117, T-119, T-120, T-122–T-126 and T-130 each say so in place and none is on the roadmap (T-118, T-139 and T-140 are closed decision records, T-121 closed negative). What carries over is the corrected corpus (T-128, done 2026-10-22), the results cleanup (T-131, now the backfill of P6), the alias table (T-132, shrunk) and the close-out (T-129). Work item 16 (schema guard, T-135–T-136) is closed in `CHANGELOG.md`; Work item 17 (multi-company articles) is re-scoped into Work item 18, its sizing absorbed by T-144 and its design superseded by D27–D28.

**Model performance checking (Work items 3, 4, 7) is resolved except
non-blocking items:** Work item 3's only open task is T-022 (full-corpus
backfill); Work item 4's is T-032 (re-solve the purity estimates from measured
agreement); Work item 7 (NER batching) is code-complete with only T-062 (empirical GPU tuning)
left — CUDA has been available since 2026-09-14, so it is actionable, just not
yet run.

Not a task with an ID: the 3,444 pre-fix `sector_summary` rows (from Work item
6's T-058 fix) are still queued to self-heal on the next real `--summarize`
run — a deliberate production action left to the repo owner.
