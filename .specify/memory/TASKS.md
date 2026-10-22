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

## Work item 15 — Category: fix the `other` bucket (filed 2026-10-14; D1/D2/D3 decided same day; gate conditional on T-117; D4 open; D5–D7 decided, see T-139)

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

Tasks are listed in execution order, not ID order (IDs are stable; T-130 to T-134,
T-139 to T-142 were added after the rest).

Dependencies at a glance:

| Task | Depends on | Blocks | Can start now? |
|---|---|---|---|
| T-116 audit | none | none | yes |
| T-130 relevance judge | none | T-132 (remainder), T-117 | yes |
| T-128 upstream hand-off | `portfolio-data-mining` T-055–T-058 | T-132, T-117 onward | **done 2026-10-22** |
| T-133 clean-subset re-read | none (read-only SOURCE attach, existing) | T-123's baseline, T-117 | **yes** |
| T-139 decisions D5–D7 | the owner | T-132 (D5, D6), T-117 (D7) | **decided 2026-10-22, defaults** |
| T-134 date window (D7) | T-139 | T-117 sampling, T-124 size | **decided 2026-10-22, default** |
| T-132 quarantined pairs | T-128, T-139 (D5, D6) | T-117 | **yes** |
| T-117 measure R and N | T-128, T-130, T-132 | T-119, T-120 (or their cancellation) | no |
| T-119, T-120 gate, label | T-117 sign-off | T-123 | no |
| T-121(a) offline re-simulation on stored verdicts | T-133 (partition of the stored verdicts) | T-121(b) design check | **yes, after T-133** (no cost) |
| T-141 calibration judge + sampler | none (code, hermetic) | T-121(b) | **yes** |
| T-142 D10 spend for T-121(b) | the owner | T-121(b) | **open; conditional default (needs credentials/budget already in place), else waits** |
| T-121(b) judged calibration sample | T-121(a), T-141, T-142, LLM credentials | T-122 | no; population = confirmed articles (the gated-in one only if a gate is built) |
| T-122 ship the rule | T-121 | T-123 | no |
| T-123 re-evaluate | T-121, T-122 (T-119, T-120 if built) | T-124 | no |
| T-131 cleanup (D4) | T-128, T-132 (for `review`), D4 | T-124 | no |
| T-140 D2b mechanism (re-label vs GPU re-inference) | the owner | T-124 | **open; default applies at T-123's approval** |
| T-124 re-classification | T-123 approved, T-131, T-140 | T-129 | no |
| T-125, T-126 contingent | T-123 result | none | no |
| T-127 sentiment impact | none (separate PR) | none | yes |
| T-129 close-out | all of the above | none | no |

- [ ] **T-116** Commit a rerunnable audit script under `scripts/` that
      reproduces the 2026-10-14 figures from the RESULTS store (label
      distribution; level-1-floor vs level-2-threshold split; best-leaf-score
      bands; `other` rate by source and by ticker; share held by
      English-word tickers, including `ALL`). Read-only; no pipeline or schema change.
      → `PLAN.md` Work item 15, step 1.
- [ ] **T-130** *(new; no crawler dependency)* Build the relevance judge
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
- [ ] **T-133** *(new 2026-10-22; no dependency, can start now)* Measure what the
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
- [x] **T-139** *(user decision record for D5–D7; **decided 2026-10-22: every decision kept at
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
      - **Unchanged, as already recorded:** the release threshold for `about` is set
        jointly with data-mining after T-132's measurement; D4 keeps its recommended
        default; T-123's targets stay provisional until T-117's sign-off.
- [ ] **T-140** *(new user decision D2b, opened 2026-10-22; **open**; if unanswered
      by the time T-123 is approved, the recommended default applies, as for D4)* D2
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
- [ ] **T-132** *(from `portfolio-data-mining` T-055 design v1, approved 2026-10-15;
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
        and reversible; an FR is added in the PR that ships it; the verdict/`other`
        naming collision is documented in `docs/category-taxonomy.md`.
      - **Depends on:** T-128 (met), T-139 (D5, D6). A check of the schema version and
        columns before reading `discovered_urls` is recommended and proposed separately;
        it is not a dependency. **Blocks:** T-117 (its R and
        N are measured after these pairs are decided). → step 2b.
- [ ] **T-117** *(run on the corrected corpus, after T-128 and T-130)*
      Prototype the relevance test as string matching over SOURCE `body_text`
      + title (company name with aliases; uppercase/explicit-form ticker only
      — see the plan's design constraints) and **measure it**: share of the
      corpus failing it, cross-tab vs. current labels, and its false-reject
      rate, using T-130's judge. Report **R** (share of the cleaned corpus
      judged not about its tagged company) and **N** (share of the remaining
      `other` rows they make up), then get the owner's sign-off on the
      build/drop rule in `PLAN.md` Work item 15 step 3. **The 5% / 20%
      numbers are arbitrary placeholders** chosen only to make the rule
      concrete; the owner sets the real ones. *Sample size:* to tell R near
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
      stratified sample (up to 3,300, which also provides T-123's baseline and replaces
      the earlier ~900) is funded separately by D10 (T-142); about 5,200 in total at most. Record the actual spend
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
- [ ] **T-119** *(conditional on T-117's sign-off; cancel in place if the gate is dropped)* Implement the relevance gate in the category stage's feature
      step (CPU-only, before any model call), with hermetic tests covering the
      ambiguous-ticker cases (`ON`, `A`, `IT`, `ARE`) and alias cases (Meta,
      Berkshire, Bank of America). Depends on T-117, T-128. → step 3.
- [ ] **T-120** *(conditional on T-117's sign-off, same as T-119)* Add the `not_relevant` label (D1 = a): `news_nlp/taxonomy.py`
      and the `docs/category-taxonomy.md` table (10 → 11 labels), the API
      response models, and the category judge's prompt/taxonomy in
      `news_nlp/eval/` so it can answer "not about the company". No table
      migration (`label` is unrestricted `TEXT`); confirm nothing else
      enumerates the label list (e.g. `sector_summary`, queries filtering on
      `label != 'other'`). Tell the knowledge-graph owner the vocabulary
      grew. → steps 2, 5.
- [ ] **T-121** Offline calibration: simulate per-slug thresholds and a
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
      call:** a paired difference depends only on the articles the two rules label
      differently (the discordant set); with `d` discordant report-half items and a net
      gain `g`, the 95% lower bound is above 0 only if `g >= 1.96 * sqrt(d)`. Record `d`
      and the `g` that +0.05 on `other` precision implies for each candidate. A candidate
      whose `d` is too small is reported as "not detectable at this sample", not as a
      failure. Because identical labels have identical verdicts, the paired difference can
      be estimated from the discordant set alone: T-123 may judge a stratified extra draw of
      the discordant articles (within D10's cap) to reach the needed `d`.
      **(b) paid, funded by D10 (T-142), not by T-117's sign-off:** the stratified,
      split judged sample, drawn with T-141's code. **Depends on:** T-141 (code), T-142
      (spend), T-121(a). Fit on the fit half only; every reported number states its half.
      **Short-circuit stratum:** 30 fit / 30 report; it informs only the
      `CATEGORY_GROUP_FLOOR` decision. **Target level, fixed now:** T-123's `other`-precision
      target is the pooled, population-weighted report-half figure over all confirmed
      articles; per-slug and per-cell figures are reported but are not the target. The seed,
      band edges, per-cell populations and the half assignment are stored in the
      run's recorded config (the article ids are already in `eval_inference`), so the
      draw can be reproduced. → step 4.
- [ ] **T-141** *(new)* Build the calibration judge and sampler T-121(b) needs, in
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
- [ ] **T-142** *(new user decision D10, opened 2026-10-22; **open**; if unanswered by
      the time T-141 is done, the default applies, as for D4 and T-140)* The spend for
      T-121(b)'s sample: **default, approve up to 3,300 judge calls** with the maintainer's
      LLM credentials, decided on its own and not at T-117's sign-off, so calibration
      does not wait for the relevance work. **The default is conditional:** it applies only
      if the maintainer has already put the LLM credentials and a budget in place in
      writing; otherwise T-121(b) waits for an explicit answer (no spend by silence). Hard
      cap at 3,300 calls (the discordant top-up in T-121 counts against it); the actual
      spend is logged against the cap in `docs/evaluation.md`; the run stops at the cap. Blocks T-121(b). The rest of the T-117
      budget (R, N, T-130 validation, T-132, T-131's check, about 1,900) stays with T-117.
- [ ] **T-122** Ship T-121's chosen rule as constants beside
      `CATEGORY_CONFIDENCE_THRESHOLD` in `news_nlp/taxonomy.py` and apply it
      in the category stage, with tests. → step 4.
- [ ] **T-123** Re-run `--stage category` eval; baseline = eval_run 43 (mlflow
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
      metric set, on the full and the gated-in population; run
      `--check-regression`. → step 5 / acceptance criteria.
      *Provisional success targets (the owner confirms or changes them at
      T-117 sign-off):* `other` precision at least 0.05 above the recomputed
      baseline, as a paired difference on the same report-half items (the same articles are
      scored under both rules, which removes most of the per-cell noise), with the lower
      bound of its 95% interval (bootstrap within cells) above 0; if a gate is built, false-reject rate at most 5% of articles
      the judge calls relevant; no per-slug regression beyond
      `--check-regression`'s tolerance. T-124 does not start unless T-123
      meets them and the owner approves.
- [ ] **T-124** *(D2 = yes, decided 2026-10-14; mechanism per T-140)* Re-classify the
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
      row from its stored scores with the new rule (a text-only gate needs no inference);
      GPU inference then covers only the short-circuited articles if the level-1 floor
      is lowered, or every article if the top-2 selection, hypothesis wording (option D),
      the input length (option H) or the model changes. If any compared score differs,
      use full re-inference. The row counts that bound the re-label are in PLAN step 6
      (92,318 of 93,330 confirmed articles are re-labelable from level-2 scores; 1,012
      short-circuited).
      *Cost stop/go (GPU path only):* the scratch run reports articles per hour and a
      projected GPU duration for the articles that need inference; if the projection
      exceeds 72 hours on the available GPU, stop and bring it to the owner before
      launching (provisional threshold, the owner may change it).
- [ ] **T-125** *(contingent)* Option E: cluster the residual relevant-but-
      `other` rows and propose taxonomy additions. Only if `other` is still
      large after T-123 and the clusters are coherent; amends the taxonomy
      contract with the knowledge-graph module. → step 7.
- [ ] **T-126** *(contingent)* Option H: classify on more than title + lead
      chunk. Only if T-123 shows misses concentrated in weak-lead articles.
      → step 7.
- [ ] **T-127** *(separate PR)* Measure the sentiment-side impact of the same
      weakness: `_text_mentions_subject` matches a lowercased ticker as a bare
      word (`\bon\b`, `\ba\b`, …), so subject weighting is likely
      non-discriminating for the eight English-word tickers, and off-topic
      articles get sentiment attributed to the wrong company. Investigate and
      report before any fix; include the multi-company case (an article-level score on an
      article that names several companies is not one company's score). → step 8.
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
- [ ] **T-131** *(new; owner decision D4)* Plan and, once decided, execute
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
      interval. **Provisional stop/go (the owner confirms at the T-117 sign-off):** go
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
      regenerates only at the next `--summarize` run; (d) the owner decides D4
      (keep the `<table>_removed` archives — recommended default — or drop
      them after verification; whole-table bump vs. targeted deletion of
      `sector_summary` rows), which (b) and (c) then apply. Runs **before**
      T-124 (never together), so no reader sees a half-cleaned store.
      → step 6b.
      *D4 default and deadline:* T-131 and T-124 need a D4 answer. If the
      owner has not decided by the time T-123 is approved, the recommended
      default applies (keep the `<table>_removed` archives;
      delete only the affected `sector_summary` rows). T-131's execution may
      ship as its own PR; it stays in this work item because T-124 depends on
      its removed-id list and ordering.
- [ ] **T-129** Close-out: annotate `SPEC.md` §13 item 19 / §14, add
      one FR per behavior that ships (gate if built, calibrated rule, the
      opt-in relevance judge T-130, results cleanup T-131) in the PR that
      ships it (SPEC's own rule), update
      `docs/category-taxonomy.md`, `docs/evaluation.md` (dated follow-up),
      `docs/modules/news-nlp.md`; reconcile both architecture artifacts —
      never rename (constitution AI behavior #11); confirm T-130 and T-131 are
      done or cancelled in place; then move this work item to
      `CHANGELOG.md` (AI behavior #15). → acceptance criteria.

## Status

Closed Work items 1, 5, 6 and 8–14 are in `CHANGELOG.md` (Work item 5 closed 2026-10-14; its
only open task, T-041, was superseded by Work item 15).

T-010–T-016 (Work item 2, regression gate) remain blocked on the maintainer's
infrastructure decision — nothing in that item has started.

**Work item 15 update (2026-10-22):** T-128 is done (data-mining's Work item 10 and
T-058 hand-off are recorded done upstream), so development can start. The corrected
corpus is about 80% smaller than the RESULTS store, which resizes T-131, T-132 and
T-124 (plan, "corrected corpus"). T-116, T-130 and T-133 can run now; D5–D7 were
settled at their defaults on 2026-10-22 (T-139).

**Work item 15 (category `other` bucket), filed 2026-10-14:** T-118 records D1–D3
(the `not_relevant` label representation, used only if T-117 justifies building a
gate; full re-run after a satisfactory test; fix the crawler first, T-128, now done).
`portfolio-data-mining` has handed off the corrected corpus and its quarantined
`review` pairs (T-132 decides them, before T-117). D4 (what to do with existing
results for removed articles, T-131) is open (D5–D7 were settled at their defaults on
2026-10-22, T-139). T-124 waits on
T-123's result and on T-131.

**Model performance checking (Work items 3, 4, 7) is resolved except
non-blocking items:** Work item 3's only open task is T-022 (full-corpus
backfill); Work item 4's is T-032 (re-solve the purity estimates from measured
agreement); Work item 7 (NER batching) is code-complete with only T-062 (empirical GPU tuning)
left — CUDA has been available since 2026-09-14, so it is actionable, just not
yet run.

Not a task with an ID: the 3,444 pre-fix `sector_summary` rows (from Work item
6's T-058 fix) are still queued to self-heal on the next real `--summarize`
run — a deliberate production action left to the repo owner.
