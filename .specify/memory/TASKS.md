# TASKS.md — `portfolio-nlp`

Discrete, checkable task breakdown for `.specify/memory/PLAN.md`. Each
task references the plan work item and the `SPEC.md` section it closes.
Check a box only when its acceptance criterion (in `PLAN.md`) is actually
met — not when the code is merely written.

Task IDs are stable, same rule as `SPEC.md`'s `FR-0xx`/`NR-0xx`: don't
renumber; mark a cancelled/superseded task in place instead.

## Work item 1 — Pin HF model checkpoints (resolved 2026-09-14)

- [x] **T-001** Look up and record the current commit SHA for each of the
      four HF model repos (`ProsusAI/finbert`, `gamug/sec-bert-finer-ord-ner`,
      `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`,
      `sshleifer/distilbart-cnn-12-6`) — from the HF Hub API/UI, not
      guessed. → `PLAN.md` Work item 1, step 1. **Done 2026-09-14** —
      fetched via `GET /api/models/<repo_id>`'s `"sha"` field; the
      sentiment model pinned is actually `gamug/FinBERT-financial-news`
      (this list itself had gone stale after the 2026-09-13 merge, still
      naming `ProsusAI/finbert`).
- [x] **T-002** Add a revision constant per model in `src/pipeline.py`
      (paired with the existing `SENTIMENT_MODEL`/`NER_MODEL`/
      `CATEGORY_MODEL`/`SUMMARY_MODEL` name constants). → step 1. **Done
      2026-09-14** — one `MODEL_REVISIONS: dict[str, str]`, not four
      separate constants.
- [x] **T-003** Pass `revision=` at all 9 `from_pretrained` call sites in
      `src/pipeline.py` (lines 150–151, 283–284, 506–507, 747–748,
      785–786). → step 2. **Done 2026-09-14** — 8 call sites, not 9:
      `sector_summary`'s own model load was removed entirely by the
      same-day `intro_text` determinism fix (PR #47), not missed here.
- [x] **T-004** Pass `revision=` in `src/setup.py`'s `download_models()`
      (`snapshot_download` + `AutoConfig.from_pretrained`, both calls, for
      all four models). → step 2. **Done 2026-09-14.**
- [x] **T-005** Document the four pins (comment or a table in
      `docs/modules/news-nlp.md`) so a future bump is a reviewed, visible
      diff. → step 3. **Done 2026-09-14** — "Model pins" table added.
- [x] **T-006** Verify: `grep -rn "from_pretrained\|snapshot_download" src/`
      shows `revision=` at every call site; `uv run python -m setup`
      succeeds; `uv run pytest` stays green. → `PLAN.md` acceptance
      criteria. **Done 2026-09-14** — `python -m setup` actually run
      (`PYTHONPATH=src`), all four models fetched at their pinned SHA
      from the real HF Hub; `test_setup.py` rewritten to assert the
      revision is threaded through; full suite (230 tests), ruff, mypy
      all pass.
- [x] **T-007** Update `SPEC.md` §13 item 4 to note the reproducibility
      half resolved (annotate in place, keep the item number). → `PLAN.md`
      Work item 1, last acceptance criterion. Also update the two
      architecture artifacts per constitution AI behavior #11 (Portfolio
      Thesis + Portfolio NLP) — reconcile, never rename. **Done
      2026-09-14** for `SPEC.md`/`docs/modules/news-nlp.md`; the
      Portfolio Thesis / Portfolio NLP architecture artifacts not
      updated this pass — see the note in `TASKS.md`'s Status section.

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

## Work item 6 — Summarization (`c_summary` + `sector_summary`): validate eval scope, close the coverage gap, and add a lightweight sector-intro check (resolved 2026-09-14 -- new sector_summary gap found and fixed same day: up to 50.2% intro_text hallucination rate)

Both summarization tasks run the same model (`SUMMARY_MODEL =
"sshleifer/distilbart-cnn-12-6"`, loaded independently by
`run_company_summary_stage` and `run_sector_summary_stage`, same
`hierarchical_summarize_batch` call), but only `c_summary` has any
evaluation today. `c_summary` is the strongest stage on
`mean_faithfulness` (4.87/5, `pct_with_hallucination` 5.4%) but weakest on
`mean_coverage` (3.02/5) — a terse/extractive tendency of
`distilbart-cnn-12-6`, not a correctness problem (`docs/evaluation.md`
2026-09-08 baseline notes). It's also, like NER, flagged as "suspected of
the same full-article-vs-lead-cap mismatch... not empirically
investigated." `sector_summary` itself stays out of scope (deterministic
composition), but its one model-generated `intro_text` sentence has
**zero** eval coverage, even a simple one — worth a narrow,
faithfulness-only check given it shares the same model. →
`PLAN.md` Work item 6, `SPEC.md` §13 item 10 (new), §9 (`c_summary`
baseline row).

- [x] **T-050** Empirically check whether `c_summary` eval sampling
      suffers the same full-article-vs-lead-cap mismatch already
      confirmed for `sentiment` — `run_company_summary_stage`'s
      hierarchical reduce (`src/pipeline.py`) processes the whole
      article; confirm the judge's scope (`src/news_nlp/eval/sampling.py`,
      `.../prompts/c_summary.md`) matches. → `PLAN.md` Work item 6, step 1.
      **Done 2026-09-14** — confirmed (10.0% of `article_summary` rows
      exceed the judge's cap, all multi-chunk) and fixed (`c_summary`
      joined `_UNCAPPED_STAGES`). PR #45.
- [x] **T-051** Decide whether/how to address `mean_coverage`'s weakness
      (3.02/5, weakest `c_summary` metric) — candidates: raise
      `SUMMARY_MIN_OUTPUT_TOKENS`/`SUMMARY_MAX_OUTPUT_TOKENS`
      (`src/pipeline.py`, currently 56/142), change the hierarchical-reduce
      strategy, or explicitly accept the terse tendency as a deliberate
      trade for the already-strong faithfulness score. Not yet decided.
      → step 2. **Done 2026-09-14** — the output-length raise was tested
      (matched-pair experiment) and rejected (coverage +0.20 but
      `pct_with_hallucination` more than doubled, 15.0%→35.5%); decided
      to accept the gap as a deliberate completeness-vs-correctness
      trade, mirroring sentiment's recall-over-precision call. PR #45.
- [x] **T-052** Run a fresh `--stage c_summary` eval after any change
      lands (or once T-050 rules out a code change) and add a dated
      follow-up entry to `docs/evaluation.md`. → step 3. **Done
      2026-09-14** — `eval_run` 34, n=1000, now the post-fix baseline.
- [x] **T-053** Add `SPEC.md` §13 item 10 (new — `c_summary` coverage +
      eval-scope question) and its §14 disposition-table row, and update
      §9's `c_summary` baseline row with the result. → `PLAN.md` Work
      item 6 acceptance criteria. **Done 2026-09-14.**
- [x] **T-054** Confirm the `sector_summary` population size (rows =
      distinct `(gics_sector, gics_sub_industry, week)`) to decide whether
      a full-population judge pass is affordable each run, instead of
      assuming it and building sampling machinery that isn't needed.
      → `PLAN.md` Work item 6, step 4. **Done 2026-09-14** -- 3,628
      rows, small enough to judge in full every run; sample_for_stage's
      sector_summary branch skips the stratification machinery entirely.
- [x] **T-055** Add a `sector_summary` case to the eval framework: a new
      prompt (`src/news_nlp/eval/prompts/sector_summary.md`) judging
      `intro_text` for faithfulness against its own `facts_json` grounding
      only (never raw article/company text — that's not what the model
      saw); wire it into `sampling.STAGES`, `metrics.HEADLINE`, and
      `cli/news_nlp_eval.py`'s `--stage` choices. → step 4. **Done
      2026-09-14** -- SectorIntroVerdict/judge_sector_summary/
      aggregate_sector_summary, 4 new hermetic tests, full suite green.
- [x] **T-056** Run the new `--stage sector_summary` eval, log it to
      MLflow/`eval_run` the same way as the other stages, and record the
      result in `docs/evaluation.md`. → step 4. **Done 2026-09-14** --
      eval_run 34, n=3628 (full population): mean_faithfulness 4.05/5,
      pct_with_hallucination **42.2%** (a real, characterized pattern --
      fabricated source attributions + self-contradicting percentages,
      not noise).
- [x] **T-057** Add a §9 baseline row (or a documented decision not to,
      if T-054/T-055 conclude a full new baseline isn't warranted) for
      the `sector_summary` intro check. → `PLAN.md` Work item 6
      acceptance criteria. **Done 2026-09-14.**
- [x] **T-058** *(new, follows from T-056's finding)* Confirm the
      `intro_text` hallucination pattern (T-056) is independent of the
      then-just-fixed sentiment data (not a staleness artifact), then fix
      it: replace the `SUMMARY_MODEL` paraphrase step with
      `build_sector_intro_seed`'s own deterministic output. **Done
      2026-09-14** — regenerated `sector_summary` against fresh sentiment
      and re-ran the eval first (got worse, 42.2%→50.2%, ruling out
      staleness); `run_sector_summary_stage` no longer loads a model;
      `SECTOR_SUMMARY_FORMAT_VERSION` bumped 2→3 so existing rows
      self-heal; `SPEC.md` FR-005/§9/§13 item 12 updated. → `PLAN.md`
      Work item 6.

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

## Work item 8 — Justify each model's selection in the Models evaluation artifact section (done 2026-09-19)

The 2026-09-14 artifact reorg centralized every model's *accuracy*
numbers into one "Models evaluation" section, but not *why this model
over a plausible alternative* — that reasoning is either missing
entirely (category, NER, `c_summary`'s base architecture) or only
covers *which variant of the same model family* (chunk-level vs.
title-only FinBERT), not *why that family at all*. → `PLAN.md` Work
item 8.

- [x] **T-064** Write the sentiment "Why this model" sub-block: FinBERT
      family over a general-purpose/from-scratch alternative; continuing
      from `ProsusAI/finbert` specifically over a generic checkpoint;
      chunk-level + entity-scoped weighting over the measured
      alternatives (pull from `docs/evaluation.md`'s 2026-09-13
      follow-ups, don't re-derive). → step 1. **Done 2026-09-19** — sourced
      from `justification.md`'s separately link-verified sourcing doc (every
      external link fetched and confirmed live before use) and
      `justification_.md`'s draft prose (typos/informal phrasing cleaned up,
      content preserved): why FinBERT-family at all (Araci 2019,
      arXiv:1908.10063 — the one freely-available checkpoint already
      fine-tuned for financial-domain sentiment, not just pretrained on
      financial text); why continuing from `ProsusAI/finbert` specifically
      (itself a continued fine-tune on Financial PhraseBank, Malo et al.
      2014, arXiv:1307.5336 — small enough to further fine-tune cheaply on
      this project's own 5,900-sentence set); chunk-level + entity-scoped
      weighting pointed at the existing 2026-09-13 four-candidate comparison
      already in the same artifact block, not re-derived. Added to both the
      Claude Artifact's `#eval-sentiment` block and `docs/modules/
      news-nlp.md`'s item 1 (identical content, per T-069's own
      never-say-something-the-docs-don't-say rule).
- [x] **T-065** Write the category "Why this model" sub-block: zero-shot
      NLI over a trained classifier (no labeled taxonomy training set);
      `deberta-v3-base-zeroshot-v2.0` specifically over other zero-shot
      checkpoints, sourced from the model's own card/benchmark
      provenance (new research — not yet written anywhere in this repo);
      the hierarchical two-level taxonomy over flat 9-way (pull from the
      2026-09-09 follow-up). → step 2. **Done 2026-09-19** — kept to two
      short paragraphs (why zero-shot NLI + this checkpoint; why
      hierarchical over flat 9-way), matching sentiment's T-064 pared-down
      format after user feedback that the first sentiment pass was too
      long. Sourced from `justification.md`'s verified links
      (`deberta-v3-base-zeroshot-v2.0`'s own card: 0.619 avg F1-macro
      across 28 zero-shot datasets vs. `facebook/bart-large-mnli`'s 0.497)
      and `docs/evaluation.md`'s 2026-09-09 follow-up (flat 9-way softmax
      diluted signal — correct-slug raw score on misses averaged
      0.14-0.16, barely above the 9-way uniform baseline of 0.111 — not
      re-derived). Added to the Claude Artifact's `#eval-category` block
      and mirrored in `docs/modules/news-nlp.md`.
- [x] **T-066** Write the NER "Why this model" sub-block: SEC-BERT
      (domain-pretrained on SEC filings) over a generic NER checkpoint;
      FiNER-ORD as the fine-tuning dataset, sourced from the dataset's
      own paper/repo (new research). → step 3. **Done 2026-09-19** — SEC-BERT
      over spaCy/`bert-base-NER` (domain-pretrained on 260,773 SEC 10-K
      filings, own ~30k-subword financial vocabulary — Loukas et al. 2022,
      arXiv:2203.06482, tied to this project's own 2026-09-12
      subword-fragmentation fix); FiNER-ORD as the fine-tuning dataset (201
      manually-annotated articles, `PER`/`LOC`/`ORG` BIO tagging — Shah,
      Gullapalli et al. 2024, arXiv:2302.11157), whose general-NER label
      set matches `article_entities`' actual need rather than a
      filings-specific tag set. Sourced from `justification.md`'s
      verified links. Added to the Claude Artifact's `#eval-ner` block and
      mirrored in `docs/modules/news-nlp.md`.
- [x] **T-067** Write the `article_summary`/`c_summary` "Why this model"
      sub-block: `distilbart-cnn-12-6` over full `bart-large-cnn` or a
      modern LLM summarizer, tied explicitly to the 6GB-VRAM /
      one-model-at-a-time budget (`SPEC.md` NR-001) and no per-call API
      cost at this corpus size — and connect the model choice to the
      already-documented `mean_coverage` weakness (a small model
      pretrained on short CNN/DailyMail news, never retuned for
      financial-news density) rather than leaving them as two unrelated
      facts. → step 4. **Done 2026-09-19** — distilled BART over full
      `bart-large-cnn`/a modern LLM summarizer, tied to NR-001's 6GB-VRAM
      budget and no per-call API cost; near-identical quality per the
      model's own card (ROUGE-2 21.26/ROUGE-L 30.59 vs. `bart-large-cnn`'s
      21.06/30.63); explicitly connected the CNN/DailyMail distillation
      data + generation defaults (`max_length=142`/`min_length=56`,
      matched by `SUMMARY_MAX_OUTPUT_TOKENS`/`SUMMARY_MIN_OUTPUT_TOKENS`)
      to the already-documented `mean_coverage` gap, not left as two
      unrelated facts. Added to the Claude Artifact's `#eval-csummary`
      block and mirrored in `docs/modules/news-nlp.md`.
- [x] **T-068** Write the `sector_summary` "Why this model" sub-block:
      frame the 2026-09-14 model-removal decision explicitly as a
      selection choice (deterministic template over any model,
      structural guarantee over probabilistic mitigation), not an
      incidental fact — this one is mostly reframing content this
      session already wrote, not new research. → step 5. **Done
      2026-09-19** — a short explicit "no model, deliberately" framing
      paragraph added ahead of the existing 2026-09-14 hallucination-rate
      content (42.2%→50.2%, both numbers already measured/documented, not
      re-derived); the structural-guarantee-over-probabilistic-mitigation
      framing ties to the same principle already used for this stage's
      cross-company-blending design. Added to the Claude Artifact's
      `#eval-sector` block and mirrored in `docs/modules/news-nlp.md`.
- [x] **T-069** Publish the updated artifact (all five sub-blocks live
      under their existing per-model blocks in `#eval`, not a new
      top-level section) and mirror the same justification content in
      `docs/modules/news-nlp.md` prose — the artifact must never say
      something the repo's own docs don't already say. → `PLAN.md` Work
      item 8 acceptance criteria. **Done 2026-09-19** — satisfied
      incrementally as each of T-064–T-068 landed, not as a separate
      final pass: every "Why this model" sub-block was added directly to
      its existing per-model block in the artifact's `#eval` section (no
      new top-level section) and mirrored in `docs/modules/news-nlp.md`
      in the same commit. Each justification names at least one concrete
      rejected alternative (sentiment: general-purpose/from-scratch,
      generic `bert-base`; category: a trained classifier, other
      zero-shot checkpoints, flat 9-way; NER: spaCy/`bert-base-NER`;
      `c_summary`: full `bart-large-cnn`, a modern LLM summarizer;
      `sector_summary`: keeping the paraphrase model). All architecture/
      benchmark provenance claims (Araci 2019, Malo et al. 2014, Loukas
      et al. 2022, Shah et al. 2024, DeBERTa-v3's own benchmark card) are
      real citations from `justification.md`'s separately link-verified
      sourcing, not invented. Sentiment's first pass (T-064) sprawled
      into several tables/essays and was rebuilt down to one metrics
      table + a short justification after direct user feedback; T-065–
      T-068 used that pared-down format from the start.

## Work item 9 — Rebalance sentiment training data (done 2026-09-19)

`gamug/FinBERT-financial-news`'s training pool is 56.1% neutral / 22.8%
negative / 21.1% positive — never a deliberate target, a byproduct of
the eval harness's confidence-stratified sampling source. → `PLAN.md`
Work item 9.

- [x] **T-070** Downsample the neutral class to the larger minority
      class's size (1,321), selecting the most representative examples
      via TF-IDF cosine similarity to the neutral class's own centroid
      (not random) — keep every negative/positive example untouched.
      → step 1. **Done 2026-09-14** —
      `scripts/rebalance_sentiment_data_2026_09_14.py`; verified exactly
      0 duplicate sentences and the expected 1,321/1,321/1,223 counts
      before anything was published.
- [x] **T-071** Publish the rebalanced `train`/`validation`/`test` splits
      to `gamug/FinBERT-financial-news-data`, `idiom_probe` untouched;
      document the before/after class counts and the selection method in
      the dataset card. → step 2. **Done 2026-09-14** — v2 of the
      dataset live; splits verified against the actual
      `stratified_split()` output before publishing, not guessed.
- [x] **T-072** Retrain the sentiment model on the rebalanced data via
      `train_sentiment.py` (same procedure/hyperparameters as the
      existing fine-tune). → step 3. **Done 2026-09-14** —
      `BALANCED_DATA_PATH` branch added to `train_sentiment.py`
      (preferred when present, doesn't double-merge idiom_augment);
      trained on CUDA, 4 epochs, same hyperparameters as v2.
- [x] **T-073** Publish the retrained model to
      `gamug/FinBERT-financial-news` as a new version, model card updated
      with the rebalance rationale. → step 3. **User decision made
      2026-09-15**: adopt **v4** (class-weighted), not v3 — see T-080.
      **Done 2026-09-19** — before running, the model card's three
      comparison tables (offline test set, idiom probe, downstream
      production eval) were simplified from a full precision/recall/F1
      breakdown per class down to one `accuracy_ovr_<class>` row per class
      (numbers re-verified against `data/sentiment_finetune/
      test_metrics_weighted.json` and `candidates_comparison_2026_09_15.json`
      before writing, not re-typed from memory; the downstream table's
      `docs/evaluation.md`-sourced numbers unchanged). Then
      `uv run python scripts/publish_finbert_financial_news_v4_2026_09_15.py`
      run for real — pushed the real trained weights
      (`models/finbert-financial-news-weighted/`) and the edited card to
      the public Hub repo, commit `93863fcb7252874e7c0339081b34f691f9e17ff6`
      (cross-checked against the Hub's own `GET /api/models/<repo_id>`
      `"sha"` field before using it anywhere, per this repo's own pin
      convention). `src/pipeline.py`'s `MODEL_REVISIONS[SENTIMENT_MODEL]`
      and `docs/modules/news-nlp.md`'s pin table both updated to that SHA
      in the same change. v3's own publish script
      (`scripts/publish_finbert_financial_news_v3_2026_09_14.py`) remains
      written and unrun — v3 was not chosen, no reason to publish it.
- [x] **T-074** Measure per-class precision/recall/F1 (positive/negative/
      neutral) on the held-out test set and report it directly against
      the currently-published model's own numbers — including any metric
      that gets worse, not just improvements. → step 4 / acceptance
      criteria. **Done 2026-09-14** — full before/after table (test set +
      idiom probe) in `docs/evaluation.md`'s 2026-09-14 follow-up;
      negative F1 up (0.726→0.829), neutral F1 down (0.838→0.714,
      idiom-probe neutral F1 0.47→0.0) — reported honestly, not filtered
      to the improvements.
- [x] **T-075** Add a dated follow-up to `docs/evaluation.md` with the
      full before/after table and methodology. → acceptance criteria.
      **Done 2026-09-14.**
- [x] **T-076** *(new, user-requested second approach)* Try
      class-weighted loss as an alternative to downsampling: train on the
      full original unbalanced pool (no data discarded) with an
      inverse-class-frequency-weighted `CrossEntropyLoss`, and measure it
      against both v2 (published) and v3 (downsampled). **Done
      2026-09-15** — `train_sentiment.py --weighted`
      (`compute_class_weights` + `WeightedLossTrainer`); results in
      `docs/evaluation.md`'s 2026-09-15 follow-up. Idiom-probe neutral F1
      lands at 0.471 (v2: 0.47, v3: 0.0) — the v3 regression doesn't
      reproduce here; every other metric sits within ~0.01-0.03 of v2.
      Not yet published to the Hub, same reasoning as v3 (T-073) — a
      publish decision, not a technical one.
- [x] **T-077** *(new, user-requested)* Run the downstream,
      production-pipeline LLM-judge evaluation on v4 — the number that
      actually validated v2, still missing for every retrained candidate
      until now. **Done 2026-09-15** — against a scratch copy of the
      results DB (`nlp_use.db`, copied so the real, shared `nlp_.db` is
      never opened for writing) via
      `scripts/resample_sentiment_v4_2026_09_15.py` (in-process
      `pipeline.SENTIMENT_MODEL` monkeypatch to v4's local checkpoint —
      `src/pipeline.py` on disk untouched) +
      `cli/news_nlp_eval.py --stage sentiment --sample-size 2000 --seed 1`
      (the documented floor). Full table in `docs/evaluation.md`'s
      2026-09-15 follow-up: `recall_negative` (this pipeline's priority
      metric) up 0.808→0.832, but `agreement_rate` (0.701→0.674) and
      `mean_severity` (0.341→0.369, lower is better) both worse — a real
      trade, not a clean win. `MODEL_REVISIONS` still pins v2.
- [x] **T-078** *(new, third rebalancing-adjacent approach)* Try swapping
      the base checkpoint (`nlpaueb/sec-bert-base`, already this project's
      NER base) instead of another data-side intervention, motivated by
      `precision_negative` being stuck at 0.505/0.513/0.507 downstream
      across v1/v2/v4 despite two different rebalancing approaches.
      **Done 2026-09-15** — `train_sentiment.py --base-model` (new flag);
      results in `docs/evaluation.md`'s 2026-09-15 follow-up. Rejected
      before a downstream eval: this candidate (v5) loses to v2 on every
      sentence-level and idiom-probe metric (macro F1 0.779→0.732, neutral
      F1 0.471→0.316 on the idiom probe), with no compensating gain to
      weigh a downstream run against — unlike v3/v4, it never clears the
      cheaper sentence-level gate. `MODEL_REVISIONS` still pins v2.
- [x] **T-079** *(new, user-requested despite T-078's rejection)* Run the
      downstream, production-pipeline LLM-judge evaluation on v5 anyway.
      **Done 2026-09-15** — `scripts/resample_sentiment_v5_2026_09_15.py`
      (modeled on T-077's v4 script, against a **fresh** scratch copy
      `nlp_use_v5.db`, not v4's own `nlp_use.db`) +
      `cli/news_nlp_eval.py --stage sentiment --sample-size 2000 --seed 1`.
      Full table in `docs/evaluation.md`'s 2026-09-15 follow-up: more
      nuanced than the offline gate suggested — `precision_negative` ticks
      up (0.513→0.526, the specific metric this experiment targeted) and
      `accuracy_ovr_negative`/neutral F1/recall are v5's best of the three
      candidates, but `recall_negative` (0.760) loses to both v2 and v4,
      and `agreement_rate`/`macro_f1_vs_judge` both land worse than v2.
      `MODEL_REVISIONS` untouched; real `nlp_.db`/`nlp.db` never opened
      for writing.
- [x] **T-080** *(new, user decision)* Choose a candidate for production
      given all three downstream-measured options (v2 baseline, v4, v5).
      **Done 2026-09-15** — user chose **v4** (class-weighted): its
      `recall_negative` gain (0.808→0.832) is the deciding factor, this
      pipeline's stated priority metric, despite v5's own real
      `precision_negative` gain — v5 didn't beat v4 on the metric that
      actually decided it. v3 (idiom-probe neutral collapse) and v5 (loses
      `recall_negative` to v4) stay documented, unpublished candidates.
      See T-073 for the publish/pin step this decision unblocks.
- [x] **T-081** *(new, user-requested)* Narrow sentiment's downstream eval to
      one-vs-rest metrics only — repeated confusion across this work item's
      reports (chat, artifact, this doc) from mixing per-class numbers with
      aggregate/blended ones in the same table. **Done 2026-09-15** —
      `news_nlp.eval.metrics.aggregate_sentiment` no longer computes
      `agreement_rate`/`macro_f1_vs_judge`/`mean_severity` at all (not just
      hidden from a report); only `precision_<class>`/`recall_<class>`/
      `f1_<class>`/`accuracy_ovr_<class>` (HT + naive-pooled) and
      `n`/`parse_fail_rate` remain. Scoped to sentiment only — category/NER/
      summarization keep their full complete metric set. `HEADLINE["sentiment"]`
      (`recall_negative`) unaffected. Constitution AI behavior #12 amendment
      to match drafted on `docs/constitution-complete-metric-reporting`
      (separate PR, not yet merged as of this task — MAJOR version bump, a
      redefinition not an addition, per this project's own governance
      rule). Full suite (231 tests), ruff, mypy green. See
      `docs/evaluation.md`'s 2026-09-15 follow-up.

## Work item 10 — Formalize the pipeline/evaluation architecture (done 2026-09-18)

`PLAN.md` Work item 10 / `SPEC.md` §13 item 15, FR-011–FR-016. Six
sequential steps (each depends on the ones before it) — not independent
efforts.

- [x] **T-082** Design the FTI base-class interfaces (feature extraction,
      training, inference) — the abstract contracts every stage subclasses,
      written and reviewed before any stage migrates onto them. → step 1 /
      SPEC.md FR-011. **Done 2026-09-16** — new `src/fti.py`: `Feature`,
      `Trainer`/`NoOpTrainer`, `Inference` (plain classes with
      `NotImplementedError`-raising template methods, not `abc.ABC` — no
      precedent in this codebase — and not `typing.Protocol` — reserved
      elsewhere for a swappable external dependency, not shared-code reuse
      across four concrete stages). Three independent top-level classes,
      no umbrella `Stage`, per FR-011's own "importable independently"
      criterion and `NoOpTrainer` needing to be one reusable class shared
      by category/`c_summary`, not re-declared per stage. `Inference.run()`
      preserves every `run_<stage>_stage` function's exact external
      contract today, including real per-article `on_progress` granularity
      for batched stages (not test-locked today, preserved deliberately
      anyway) and a small disclosed hardening
      (`try`/`finally` around load/free — today's code leaks a loaded
      model on a mid-batch exception). New
      `tests/news_nlp/test_fti_base.py` (7 tests, structural only, no real
      stage/model). No existing file touched; full suite green (239 —
      232 baseline + 7 new).
- [x] **T-083** Migrate the sentiment stage onto the FTI hierarchy
      (`_sentiment_chunk_weights`/`_text_mentions_subject` → `Feature`;
      `train_sentiment.py` → `Trainer`; `run_sentiment_stage` →
      `Inference`). No behavioral change — existing sentiment hermetic
      tests pass unmodified except import paths. → step 1 / FR-011.
      **Done 2026-09-16** — new `src/sentiment_stage.py`
      (`SentimentFeature`/`SentimentInference`); `train_sentiment.py`
      gains `SentimentTrainConfig`/`SentimentTrainer` (migrated `main()`'s
      body verbatim, byte-for-byte diffed against the pre-migration
      version — only `args_ns.X` → `config.X`). `run_sentiment_stage`
      stays a thin, settable module-level wrapper in `pipeline.py` reading
      `SENTIMENT_MODEL`/`MODEL_REVISIONS` fresh each call, so
      `test_pipeline_run.py`'s stub monkeypatch and the three
      `resample_sentiment_v{3,4,5}_2026_09_15.py` scripts' model-swap
      monkeypatch both keep working unchanged. `fti.Inference` gained one
      small additive extension (an explicit `revision` override) to
      support this. Corrected an over-cautious T-082-era assumption along
      the way: monkeypatching a class (`pipeline.AutoTokenizer`, `pipeline.
      db`) is visible globally regardless of which module's import
      triggered it, since import binds a name to the same object rather
      than copying it — so only the 5 tests calling the relocated pure
      functions directly needed an import-path change, not the model-
      loading/db-fetch monkeypatches. Caught and fixed one real,
      undisclosed-by-any-test behavioral-regression risk during design:
      `SentimentInference` overrides `batch_size()` to `1` to preserve
      today's real per-article commit granularity (the base class's
      default would have silently reduced it to one commit per run).
      241 tests (239 + 2 new `revision` override tests), ruff, mypy all
      green; `uv run src/train_sentiment.py --help` smoke-tested.
- [x] **T-084** Migrate the NER stage onto the FTI hierarchy
      (`_ner_batch`/`merge_bio_predictions` → `Feature`; `train_ner.py` →
      `Trainer`; `run_ner_stage` → `Inference`). No behavioral change. →
      step 1 / FR-011. Done 2026-09-16: `src/ner_stage.py` (new) holds
      `merge_bio_predictions` (moved verbatim), `NerFeature` (cross-article
      chunk flattening + padded tokenizer call -- `extract_batch` overridden
      directly, not `extract_one`, since NER's real batching can't be
      expressed per-row) and `NerInference` (forward pass + BIO merge +
      regrouping). Required one extension to `fti.Inference` itself: a
      `batch_size` constructor override (mirroring the existing `revision`
      pattern) so `pipeline.run_ner_stage`'s thin wrapper can pass
      `NER_BATCH_SIZE` through fresh on every call -- needed because a test
      monkeypatches `pipeline.NER_BATCH_SIZE` between two calls in the same
      test. `train_ner.py` (no CLI flags, zero test coverage, like
      `train_sentiment.py`) got a `NerTrainConfig`/`NerTrainer` the same
      way. 245 tests (243 + 2 new `batch_size` override tests), ruff, mypy
      all green; `train_ner` import-smoke-tested (no `--help` path since it
      has no argparse).
- [x] **T-085** Migrate the category stage onto the FTI hierarchy
      (`_category_premises`/level-1-level-2 batch logic → `Feature`; a
      documented no-op `Trainer` — zero-shot, no fine-tuning step exists
      today; `run_category_stage` → `Inference`). No behavioral change. →
      step 1 / FR-011. Done 2026-09-16: `src/category_stage.py` (new)
      holds `classify_group_scores`/`top2_groups`/`classify_category_scores`
      (moved verbatim), `CategoryFeature` (one premise per article --
      simpler than NER, reuses `Feature`'s own default `extract_batch`
      loop since category's real cross-article batching happens one level
      down, inside the (premise, hypothesis)-pair tokenizer calls) and
      `CategoryInference` (level-1 routing + level-2 forward pass, folded
      into one `predict_batch` call per outer batch). `Trainer` is a plain
      reuse of `fti.NoOpTrainer` (zero-shot, nothing to fine-tune) --
      documented in the module docstring, no subclass needed.
      `write_predictions` writes in two passes with an explicit
      mid-`conn.commit()` between them to preserve the original code's
      exact two-commit-per-batch behavior ("flush this batch's
      short-circuited rows now"), which the base `run()` template's
      single trailing commit alone wouldn't reproduce. Caught and fixed
      one real, pre-existing coupling this migration would otherwise have
      broken: `tests/news_nlp/test_pipeline_progress.py`'s sentiment
      empty-progress test and `test_sentiment_pipeline.py`'s own
      integration test both still monkeypatched `pipeline.
      AutoModelForSequenceClassification` (a T-083 leftover that only kept
      working because `pipeline.py` still imported that class for
      category's sake) -- both retargeted to `sentiment_stage.
      AutoModelForSequenceClassification`. 243 tests, ruff, mypy all
      green.
- [x] **T-086** Migrate `c_summary` onto the FTI hierarchy
      (`hierarchical_summarize_batch`'s chunk/reduce logic → `Feature`; a
      documented no-op `Trainer`; `run_company_summary_stage` →
      `Inference`). No behavioral change. → step 1 / FR-011. Done
      2026-09-18: `src/summary_stage.py` (new) holds
      `hierarchical_summarize_batch`/`_leaf_summarize_batch`/`_reduce_pass`/
      `_summarize_in_batches`/`_summarize_batch` (moved verbatim) plus
      `SummaryFeature` (just builds each row's raw input text -- unlike
      the other three stages, the actual chunking/reduction can't be
      hoisted into a pure pre-model `Feature` step, since each reduce
      pass re-chunks and re-summarizes the *previous* pass's own model
      output) and `SummaryInference`. `Trainer` is a plain reuse of
      `fti.NoOpTrainer` (pretrained off-the-shelf checkpoint, nothing to
      fine-tune), matching category's T-085 precedent. This was the last
      of the four ML stages to migrate, so `pipeline.py` no longer
      imports any `transformers` class at module scope at all -- caught
      and fixed a real, broader-than-expected coupling this exposed: 7
      monkeypatch call sites across `test_category_pipeline.py`,
      `test_pipeline_progress.py`, `test_ner_pipeline.py`, and
      `test_sentiment_pipeline.py` still targeted `pipeline.AutoTokenizer`
      (a leftover surviving purely because `pipeline.py` kept importing
      it for `c_summary`'s sake after each of those stages' own migration
      moved their real usage elsewhere) -- all retargeted to each test's
      own already-migrated stage module. 243 tests, ruff, mypy all green.
- [x] **T-087** Move `run_sector_summary_stage` out of `pipeline.py` into
      `news_nlp/sector_summary/`, completing the separation already mostly
      in place (`composition.py`/`queries.py` already live there as of
      2026-09-14) — imports nothing from the FTI base classes.
      `pipeline.run_pipeline`'s call into it stays external-behavior
      identical. → step 2 / FR-012. Done 2026-09-18: new
      `src/news_nlp/sector_summary/stage.py` holds `run_sector_summary_stage`
      and `SECTOR_INTRO_METHOD` (moved verbatim, imports nothing from
      `src/fti.py`), re-exported flat from `news_nlp/sector_summary/
      __init__.py` alongside `composition.py`/`queries.py`. `pipeline.py`
      now just imports the function (`from news_nlp.sector_summary import
      run_sector_summary_stage`) rather than wrapping it -- unlike the four
      FTI stages, this one has no per-call model/config to read fresh, so
      a thin-wrapper wasn't needed, only a straight re-export.
      `pipeline.run_pipeline`'s call site and `test_pipeline_run.py`'s
      `_stub_out_stages` monkeypatch of `pipeline.run_sector_summary_stage`
      both keep working unchanged (the imported name still resolves via
      `pipeline.py`'s own module globals at call time, same mechanism
      already relied on for the four FTI stages' thin wrappers). One
      test-only retarget: `test_summary_pipeline.py`'s
      `pipeline.SECTOR_INTRO_METHOD` reference moved to `news_nlp.
      sector_summary.SECTOR_INTRO_METHOD` (the constant no longer lives in
      `pipeline.py` at all). 243 tests, ruff, mypy all green.
- [x] **T-088** Full-suite regression check after T-083–T-087: every
      existing hermetic test in `tests/news_nlp/` green, with only
      import-path/construction changes where a test reached into a stage's
      internals directly — no assertion changes. → acceptance criteria
      ("no behavioral regression"). Done 2026-09-18: audited
      `git diff --stat 4dedac9 origin/master -- tests/news_nlp/` (the
      commit right before T-082 started) — 6 files touched, exactly the
      ones expected (`test_fti_base.py` new; `test_category_pipeline.py`/
      `test_ner_pipeline.py`/`test_pipeline_progress.py`/
      `test_sentiment_pipeline.py`/`test_summary_pipeline.py` modified).
      Inspected every changed `assert` and `monkeypatch.setattr` line
      individually: every one is a bare module-prefix swap (e.g.
      `pipeline.CATEGORY_GROUP_FLOOR` → `CATEGORY_GROUP_FLOOR`,
      `pipeline.AutoTokenizer` → `ner_stage.AutoTokenizer`) — zero
      assertion-logic or expected-value changes found. Also grep-confirmed
      NR-006 compliance (no raw `sqlite3` imports) across all five new
      stage modules. Fresh `uv run pytest`/`ruff`/`mypy` from a clean
      `origin/master` checkout (post-T-087): 243 passed, both clean.
- [x] **T-089** Redesign `news_nlp/eval/`'s inference step to call each
      stage's own FTI `Inference` subclass (T-083–T-086) for model-scoring,
      removing whatever independent `from_pretrained`/forward-pass code the
      eval module duplicates today. Based on the current
      `news_nlp/eval/` implementation — `sampling.py`/`judges.py`/
      `verdicts.py`/`tracking.py`/`regression.py` carry over unchanged in
      spirit. → step 3 / FR-013. Done 2026-09-18: a code-search for
      duplicated model-loading inside `news_nlp/eval/` already returned
      zero hits — the package had no live inference step at all;
      `sampling._prediction` only ever read a *prior* `pipeline.py` run's
      already-stored predictions. The real, still-unbuilt half of FR-013
      was giving eval a way to score a **candidate** model at all, which
      previously only existed as a manual, destructive scratch-DB-copy-
      and-restore workaround (`scripts/resample_sentiment_v{3,4,5}_
      2026_09_15.py`). New `src/news_nlp/eval/candidate.py`'s
      `candidate_scored_connection` reuses `Inference.run()` wholesale
      (the *same class* `pipeline.py` uses per stage, parameterized with a
      different `model_name`/`revision` — its constructor's own
      documented extension point, needing zero `fti.py` changes) against
      a fresh, auto-cleaned scratch RESULTS file, never the real one.
      `EvalSettings` gained `candidate_model`/`candidate_revision`/
      `candidate_prescore_size`; `runner._run_stage` points
      `sample_for_stage` at the scratch connection instead of production
      when set (stratification logic itself untouched — just reads the
      candidate's own freshly-written scores); `cli/news_nlp_eval.py`
      gained matching flags, validated to require exactly one `--stage`
      (a candidate swap is inherently stage-specific) plus a non-empty
      revision (this project's own pin-every-model convention, SPEC.md
      SS13 item 4 — an unrecognized `model_name` with no revision would
      otherwise silently `KeyError` against the production model's own
      `MODEL_REVISIONS` dict). Had to additionally teach the scratch file
      to stand up a structurally exact clone of SOURCE's `articles` table
      (cloned via `sqlite_master.sql`, not `CREATE TABLE AS SELECT`, which
      would drop the `id` PRIMARY KEY every result table's foreign key
      needs) — `init_schema` deliberately never creates `articles` itself,
      since a real RESULTS file already has it from a one-time historical
      migration. `sector_summary` excluded from the new mechanism (no
      Feature/Train/Inference shape, FR-012). 248 tests (243 + 5 new),
      ruff, mypy all green.
- [x] **T-090** Split `eval_judgement` into two tables (model inference /
      LLM judge verdict), each with `article_id` (reinforced as a
      first-class SOURCE-traceback key), `task`, and `experiment` columns;
      additive schema migration (existing `eval_run`/`eval_judgement` rows
      untouched, same self-heal precedent as `sector_summary`/
      `article_category`'s own migrations in `schema.py`). → step 4 /
      FR-014. Done 2026-09-18: new `eval_inference`/`eval_verdict` tables
      in `schema.py` (both carrying `article_id`/`task`/`experiment`, a
      `UNIQUE(article_id, task, experiment, run_id)` defensive
      anti-double-insert guard — not T-091's reuse-lookup mechanism, which
      is a separate later task). Genuinely simpler than every prior
      additive migration in this file: two brand-new tables need only
      `CREATE TABLE IF NOT EXISTS`, no `ensure_columns` self-heal function
      at all. `eval_judgement`'s own DDL and historical rows are untouched
      — new eval runs simply stop writing there.
      `store.record_judgement` → `record_inference`/`record_verdict`;
      `runner._run_stage` resolves `experiment =
      settings.candidate_model or settings.run_name or "base"` (`task` is
      exactly today's `stage` string, verbatim, already ranging over all
      five stages including `sector_summary`). Two disclosed,
      out-of-scope gaps found and left unfixed (both pre-existing from
      T-089, not introduced by this split): `queries.latest_eval_runs`/
      `GET /eval/latest` and `regression.check_regression`'s MLflow
      previous-run lookup are still not `experiment`-aware, so a
      candidate-model run can still be picked up as "latest"/"previous"
      for its stage — noted in `docs/evaluation.md`'s 2026-09-18
      follow-up, not fixed here (FR-014's acceptance criteria is scoped
      to the two new tables, not `eval_run`'s own query semantics).
      `scripts/label_sentiment_sentences_2026_09_13.py` (a one-shot,
      already-completed script) still reads legacy `eval_judgement` —
      commented, not rewired, since it has no remaining active use. 248
      tests, ruff, mypy all green.
- [x] **T-091** Implement the judge-table reuse mechanism: before invoking
      the judge LLM for a sampled `(article_id, task, experiment)`, check
      the redesigned judge table for an existing verdict under that exact
      key and reuse it; unique constraint on `(article_id, task,
      experiment)` enforces this as an indexed lookup, not a scan. →
      step 5 / FR-015. Done 2026-09-18: kept `eval_verdict`'s existing
      4-column `UNIQUE(article_id, task, experiment, run_id)` constraint
      from T-090 rather than migrating it — PLAN.md Work item 5's own
      text explicitly sanctions this variant ("...or `(article_id, task,
      experiment, run_id)` if a re-judge under the same key is ever
      deliberately wanted"), and this design does exactly that: every run
      still records a full `eval_inference`+`eval_verdict` row (history
      preserved), reusing a prior verdict's *content* to skip the judge
      LLM call rather than skipping the row. Avoided being this
      codebase's first constraint-altering migration (confirmed via
      direct research: `portfolio_common.db` only supports additive
      `ensure_columns`; SQLite itself needs a full create-copy-drop-rename
      dance to change a UNIQUE constraint, with zero precedent anywhere in
      this repo). Added a dedicated
      `idx_eval_verdict_article_task_experiment` index (`eval_inference`
      already had the equivalent) so the lookup is genuinely indexed, not
      relying on the 4-column constraint's own index's leftmost-prefix
      property. New `store.find_verdict_json` (the first *read* function
      in that module) + `verdicts.VERDICT_MODELS` (stage → pydantic
      class, for reconstructing a stored `verdict_json` back into the
      right type — confirmed via direct code reading that every
      `aggregate_<stage>` function touches verdicts via plain attribute
      access only, so a reconstructed verdict is 100% interchangeable
      with a freshly-judged one). `runner._resolve_verdicts` partitions
      each stage's sampled items into reused vs. needs-judging before
      building the `ThreadPoolExecutor` pool — only needs-judging items
      go through it. Two new hermetic tests directly exercise FR-015's
      literal acceptance criterion: same experiment across two `run_eval`
      calls → zero new judge-LLM calls on the second; a different
      experiment → judges fresh. 251 tests (248 + 3 new), ruff, mypy all
      green.
- [x] **T-092** Add the confusion-matrix table (sentiment/category only:
      one row per `(experiment, task, true_label, predicted_label)` with a
      count), populated from the same judge verdicts T-090/T-091 already
      record — no new judge calls needed. → step 6 / FR-016. Done
      2026-09-18: new `eval_confusion` table (sentiment/category only,
      sparse — a cell with zero occurrences a run simply has no row),
      scoped per `run_id` like `eval_inference`/`eval_verdict`'s own
      T-090/T-091 precedent (full per-run history; a cumulative view
      across an experiment's every historical run is a plain `SUM(count)
      GROUP BY (experiment, task, true_label, predicted_label)` at query
      time, not something write-time maintains — confirmed
      `portfolio_common.db.Dialect`'s only upsert primitive is SQLite's
      whole-row `INSERT OR REPLACE`, no increment-on-conflict precedent
      anywhere in this codebase, so this design deliberately doesn't need
      one). New `metrics.confusion_pairs` (mirrors `aggregate_sentiment`/
      `aggregate_category`'s own `pairs`/`parse_failed`-filter derivation,
      disclosed duplication rather than refactoring those two tested
      functions for a two-line derivation) + `store.record_confusion_cells`
      (one INSERT per distinct cell, same low-complexity pattern as
      `record_inference`/`record_verdict`) + one call per `_run_stage`
      invocation in `runner.py`, right after the existing per-item write
      loop, using `collections.Counter` (already imported) over the same
      `(item, verdict)` pairs already judged/resolved — genuinely no new
      judge calls. Confirmed via `CategoryVerdict._coerce_unknown_slug`
      that category's `"other"` is a legitimate value on the *true* axis
      too, not just predicted — needed no special handling, since the
      table only stores whatever labels actually appear (no fixed-label
      enumeration). DB-only, no MLflow artifact — FR-016's acceptance
      criterion only names the DB table. 257 tests (251 + 6 new), ruff,
      mypy all green.
- [x] **T-093** Add one-vs-rest ROC/AUC (`roc_auc_<class>`) to
      `aggregate_sentiment`/`aggregate_category`, computed from each sampled
      row's own stored prediction probabilities (`article_sentiment`'s
      `positive`/`negative`/`neutral`; `article_category`'s 9-way NLI
      distribution) — no new data collection required. → step 6 / FR-016.
      Done 2026-09-18: `roc_auc_<class>` (HT-weighted) +
      `roc_auc_<class>_naive_pooled`, matching every other per-class metric
      in this module. AUC via the weighted Mann-Whitney U statistic — a
      recognized generalization of unweighted rank-based AUC to
      Horvitz-Thompson per-item weights (`_ht_weights`, new: exposes the
      per-bucket `population_h/n_h'` weight `_ht_sum` already computes
      internally, but per-item rather than folded into one total, since
      `_weighted_auc`'s statistic needs a weight per row). Computed via an
      O(n log n) sort-and-single-pass algorithm (`_weighted_auc`), not the
      naive O(n²) pairwise reading of the formula — real regression-tracked
      sentiment runs sample ~1,800-2,800 rows (`docs/evaluation.md`), where
      O(n²) would be real added cost for nothing. `aggregate_category`'s
      `roc_auc_<slug>` loop iterates the fixed `CATEGORY_SLUGS` (9), not the
      dynamic `classes` set the other per-slug loops use — `"other"` is a
      threshold fallback with no NLI hypothesis/score column of its own
      (`taxonomy.py`), so `roc_auc_other` is never produced. No
      `runner.py`/`store.py`/`tracking.py` changes — `roc_auc_<class>` lands
      in the same `dict[str, float]` those already thread through
      `eval_run.metrics_json`/MLflow. `aggregate_ner`/`aggregate_c_summary`
      untouched. 264 tests (257 + 11 new: 9 in `test_eval_metrics.py`
      covering `_weighted_auc`/`_ht_weights` directly plus
      perfect-separation/tied-score/HT-vs-naive/`"other"`-exclusion cases
      through the public aggregators, 2 in `test_eval_runner.py`), ruff,
      mypy all green.
- [x] **T-094** Regression test locking `aggregate_ner`/
      `aggregate_c_summary`'s returned metric key sets as byte-identical to
      their pre-this-work-item shape — confusion-matrix/ROC treatment is
      sentiment/category only, and this test is what actually enforces
      that boundary rather than just stating it. → step 6 / FR-016 /
      acceptance criteria. Done 2026-09-18: neither T-092 nor T-093 touched
      `aggregate_ner`/`aggregate_c_summary` at all, so this was purely
      additive — two new tests
      (`test_aggregate_ner_metric_key_set_is_unchanged`,
      `test_aggregate_c_summary_metric_key_set_is_unchanged`) asserting
      `set(out)` against an explicit, hand-enumerated, hardcoded key set
      (not derived from the function's own output, which would be
      tautological and could never catch a regression), reusing this
      file's existing `test_ner_error_only_contract_prf`/
      `test_c_summary_scales_and_hallucination_flag` fixtures so the
      dynamic `f1_<etype>`/`mean_faithfulness_<bucket>`/
      `mean_coverage_<bucket>` keys are deterministic. No production code
      changes. 266 tests (264 + 2 new), ruff, mypy all green.
- [x] **T-095** Update `docs/evaluation.md` (methodology section),
      `docs/modules/news-nlp.md`, `docs/db-topology.md` (new tables), and
      reconcile the two architecture artifacts (constitution AI behavior
      #11 — Portfolio Thesis + Portfolio NLP, reconcile only, never rename)
      once the restructuring lands. Done 2026-09-18: `docs/evaluation.md`
      gained a new "Architecture: the eval module reuses the pipeline's own
      inference classes" section, a new 2026-09-18 dated follow-up covering
      T-091/T-092/T-093/T-094, `roc_auc_<class>` added to the metrics
      table, and `--candidate-model`/`--candidate-revision`/
      `--candidate-prescore-size` added to the Flags list + a usage
      example (all previously undocumented despite existing in the real
      CLI/code). `docs/modules/news-nlp.md` gained an FTI-architecture
      bullet, `sector_summary`'s new module location, and fixed stale
      `eval_run`/`eval_judgement` + `pipeline.py` file-path references.
      `docs/db-topology.md` now lists the eval run-log tables alongside
      the 5 result tables. Both architecture artifacts reconciled (content
      only, titles untouched): Portfolio NLP gained an architecture-note
      callout in its eval section + a new footer changelog entry; Portfolio
      Thesis gained a shorter equivalent (diagram tooltip + status-table
      clause + footer entry), proportionate to its compressed per-repo
      role. Incidental one-line fixes (same stale `eval_judgement` table
      name, found adjacent to what was already being edited):
      `src/news_nlp/eval/__init__.py`, `cli/news_nlp_eval.py`, and
      `CLAUDE.md`'s `news_nlp/eval/` bullet. No production code changes;
      ruff/mypy untouched by anything but the two docstring edits (both
      clean). This closes Work item 10 in full (T-082–T-095 all done).

## Work item 11 — JSON-driven, single-command experiment runs (done 2026-09-18)

`PLAN.md` Work item 11 / `SPEC.md` §13 item 16, FR-017. Five parts;
T-096/T-097 are small, independent prerequisites, T-098 must land before
T-099/T-100, and T-101 (the historical backfill) is the acceptance proof
for everything before it — not independent efforts.

- [x] **T-096** Parameterize `stratified_split()` (`src/train_sentiment.py`)
      and `SentimentTrainConfig` with `split_seed`/`test_frac`/`val_frac`
      — today these are hardcoded module constants (`seed=42`, 80/10/10),
      never threaded from any config or CLI flag. Defaults must reproduce
      today's exact behavior for every existing call site (no test
      assertion changes). → step 2 / SPEC.md FR-017. Done 2026-09-18:
      `stratified_split(rows, seed=_SEED, test_frac=_TEST_FRAC,
      val_frac=_VAL_FRAC)` — new keyword params, defaults are the exact
      pre-existing module constants, so every call site's behavior is
      byte-identical unless it now opts into an override.
      `SentimentTrainConfig` gained matching `split_seed`/`test_frac`/
      `val_frac` fields (same defaults); `SentimentTrainer.train()` now
      threads them into `stratified_split(...)` instead of calling it with
      no arguments. New `tests/news_nlp/test_train_sentiment.py` (6 tests:
      defaults match the module constants and reproduce the exact
      no-args call; stratification-per-label correctness; seed
      reproducibility; overriding `test_frac`/`val_frac` actually changes
      split sizes, not silently ignored; the `max(1, ...)` floor holds at
      tiny fractions; `SentimentTrainConfig`'s new fields are independently
      overridable without disturbing `weighted`/`base_model`) — written
      now rather than deferred to T-102 as originally scoped, matching
      every other task this session's own precedent of shipping code and
      tests together; T-102 below is adjusted accordingly. 277 tests
      passing (271 + 6 new), ruff/mypy clean.
- [x] **T-097** Wire a real `Trainer` (`NoOpTrainer`-based) into
      `category_stage.py`/`summary_stage.py` — both currently only
      *mention* `NoOpTrainer` in a docstring; neither instantiates it
      anywhere reachable outside `fti.py`'s own unit test. → step 4 /
      SPEC.md FR-017. **Done 2026-09-18** — `CategoryTrainer(NoOpTrainer)`/
      `SummaryTrainer(NoOpTrainer)`, trivial body-less subclasses (not a
      bare `NoOpTrainer` reference) so this stage has a real, named
      `Trainer` class matching sentiment's/NER's own
      `SentimentTrainer`/`NerTrainer`, for the future stage→`Trainer`
      registry (T-099, mirrors `news_nlp.eval.candidate`'s own
      `_STAGE_CLASSES` pattern) to look up. Each is exercised by a new
      hermetic test (`test_category_trainer_is_a_real_reachable_no_op`,
      `test_summary_trainer_is_a_real_reachable_no_op`) asserting
      `TrainedArtifact(output_dir=None, metrics=None)`, closing the "never
      instantiated outside `fti.py`'s own unit test" gap for real. 279
      tests (277 + 2 new), ruff, mypy, pre-commit all green.
- [x] **T-098** Design + implement the `ExperimentSpec` pydantic schema
      (`src/experiment.py`): `PretrainSpec`/`TrainTestSplitSpec`/
      `EvalSpec`/`PublishSpec` nested under one top-level spec, generic
      across sentiment/NER/category/`c_summary`. Strict validation —
      unpinned/missing `base_model` when `pretrain.enabled`; `pretrain`
      rejected outright for category/`c_summary`; unknown
      `hyperparameters` keys rejected by name against that stage's real
      `TrainConfig` fields; `publish.enabled` requires `pretrain.enabled`
      + `repo_id`; `eval.candidate_model` must be unset when
      `pretrain.enabled` (auto-filled downstream, never user-supplied
      there). → step 1 / SPEC.md FR-017. **Done 2026-09-18** —
      `EvalSpec` is a direct passthrough subset of `EvalSettings`' own
      non-secret fields (never `llm_api_key`/`llm_model`/`llm_url`/
      `mlflow_tracking_uri`, which stay `.env`/CLI-sourced, never a
      git-tracked JSON file); `PublishSpec` only records *that* a publish
      is wanted and where (`repo_id`), not a model-card template — each
      historical publish script's card stays hand-written, matching
      T-099's own scope note that a Hub push keeps its explicit,
      separate confirmation gate. `PretrainSpec.hyperparameters`' allowed
      keys are computed from each stage's real `TrainConfig` dataclass
      (`SentimentTrainConfig`/`NerTrainConfig`, imported lazily inside
      `_train_config_class` so validating a spec never pays
      `train_sentiment`/`train_ner`'s own heavier import cost unless that
      stage is the one being validated) minus the four fields already
      structured as `PretrainSpec.base_model`/`.split.*` — so e.g.
      `split_seed` can't be set a second, conflicting way via
      `hyperparameters`. Every model additionally rejects unknown
      top-level/nested keys outright (`extra="forbid"`, a `_StrictModel`
      base every spec class shares) — a stricter bar than T-098's own
      five named rejection cases, added for the same "reproducibility
      guarantee" reason the module docstring states. Following T-096's
      precedent, the validation test suite shipped in this same pass
      rather than waiting for T-102 (adjusted below): new
      `tests/news_nlp/test_experiment.py`, 20 tests covering every named
      rejection case plus the `extra="forbid"` additions and a JSON
      round-trip. 299 tests (279 + 20 new), ruff, mypy clean — nested
      specs are constructed as real `PretrainSpec`/`EvalSpec`/
      `PublishSpec` instances in the tests, not dict literals: this repo
      has no pydantic mypy plugin configured, so a plain dict doesn't
      type-check against a nested `BaseModel`-typed field even though
      pydantic itself would coerce it at runtime.
- [x] **T-099** Implement `run_experiment(spec, *, source_db=None,
      results_db=None) -> ExperimentResult` (`src/experiment.py`): a
      stage→`(TrainConfig, Trainer)` registry (mirrors
      `news_nlp.eval.candidate`'s own `_STAGE_CLASSES` pattern) drives an
      optional train step; on success, `candidate_model`/
      `candidate_revision` auto-resolve to the fresh local checkpoint
      (`revision="local"`, the established convention
      `resample_sentiment_v4_2026_09_15.py` already set); evaluation
      reuses `news_nlp.eval.runner.run_eval` verbatim; an optional publish
      step (sentiment only, reusing the existing
      `publish_finbert_financial_news_v4_2026_09_15.py` model-card
      pattern) requires the same explicit, separate confirmation any Hub
      push already needs — this task does not change that gate. Writes a
      git-tracked `experiments/results/<name>.result.json` (resolved
      spec + `code_version` + `TrainedArtifact.metrics` + the `run_eval`
      result dict). → steps 1/3 / SPEC.md FR-017. **Done 2026-09-18** —
      `_train_config_class`/`_trainer_class` (both lazy-imported, mirroring
      T-098's own pattern) resolve `(SentimentTrainConfig,
      SentimentTrainer)`/`(NerTrainConfig, NerTrainer)` for the two
      trainable stages, `(TrainConfig, NoOpTrainer)` for category/
      `c_summary` (structurally unreachable there — `ExperimentSpec`'s own
      validator already rejects `pretrain.enabled` for both — kept for
      symmetry with T-098's own registry, not because it's ever called).
      Publishing is recorded in the echoed `spec` but never executed — no
      new code path added here, matching the task's own scope note.
      `check_regression=True` reuses `run_eval`'s existing `SystemExit(1)`
      on a regression past tolerance verbatim, uncaught (same behavior
      `cli/news_nlp_eval.py` already has for this flag) — deliberately not
      wrapped in try/except, so no result file is written for a run that
      regressed past tolerance; disclosed in the function's own docstring,
      left for T-100's CLI to decide what to do with.
      Caught and fixed one real, undisclosed-by-T-098's-own-design gap
      while implementing: `train_ner.NerTrainConfig` had **zero fields**
      (train_ner.py takes no CLI flags) — meaning `PretrainSpec.base_model`
      would have been silently ignored for `stage="ner"`, since
      `NerTrainer.train()` always fine-tuned the hardcoded `MODEL_NAME`
      regardless of what config it received. Fixed at the source (not
      worked around in `experiment.py`): `NerTrainConfig` gained a real
      `base_model: str = MODEL_NAME` field, threaded into both
      `from_pretrained` calls in `NerTrainer.train()`; default preserves
      every existing invocation's behavior unchanged. New
      `tests/news_nlp/test_train_ner.py` (2 tests, config-defaults-only
      scope, matching `test_train_sentiment.py`'s own precedent — the real
      GPU fine-tuning path stays untested, as everywhere else in this
      suite).
      Hermetic end-to-end tests shipped with the code (T-102's own
      scope, following T-096/T-098's precedent):
      `tests/news_nlp/test_experiment_run.py`, 2 tests — an eval-only spec
      and a `pretrain.enabled=true` spec (stubbed judge per
      `test_eval_runner.py`'s pattern, a fake `SentimentTrainer.train`, and
      a fake local-checkpoint model/tokenizer per
      `test_eval_candidate.py`'s own pattern — `experiment.RESULTS_DIR`
      monkeypatched to a `tmp_path` so the test suite never writes into the
      real repo's `experiments/` directory). 303 tests (299 + 4 new: 2 in
      `test_train_ner.py`, 2 in `test_experiment_run.py`), ruff, mypy
      clean.
      **Follow-up fix (2026-09-18, reported by the user running a real
      backfilled spec off `uv run cli/run_experiment.py --config
      experiments/sentiment_v2_chunklevel_finetuned.json` without
      `uv sync --group eval`)**: `run_experiment` had imported
      `news_nlp.eval.runner.run_eval` at module scope, not lazily —
      violating NR-005 for `experiment.py` itself (the module docstring
      already documents the same lazy-import discipline for
      `_train_config_class`/`_trainer_class`, but this import was missed).
      That meant merely importing `ExperimentSpec` — no `run_experiment`
      call, just schema validation — required `strands`/`mlflow` to be
      installed, breaking `uv sync` (no `--group eval`) +
      `uv run pytest`'s documented invariant for
      `tests/news_nlp/test_experiment.py` (not guarded by
      `pytest.importorskip`, unlike the eval-specific test files, since it
      was never meant to need the eval group). Fixed: `run_eval`'s own
      import moved inside `run_experiment()`'s body, matching this file's
      existing lazy-import pattern and `news_nlp/eval/__init__.py`'s own
      documented rationale (`from news_nlp.eval import run_eval` would
      **not** have fixed this — PEP 562 `__getattr__` still fires at
      import time for a `from`-import, only a genuinely local import
      inside the function body defers it to call time). Verified
      `import experiment` no longer touches `sys.modules['strands']`/
      `['mlflow']`. This does not change what `cli/run_experiment.py`
      itself needs — every real run still evaluates, so the CLI always
      needs `--group eval`, same as `cli/news_nlp_eval.py`. 314 tests
      (unchanged), ruff, mypy clean.
- [x] **T-100** `cli/run_experiment.py` — the one command
      (`uv run cli/run_experiment.py --config <path>.json`, optional
      `--results-db`/`--source-db` overrides matching
      `cli/news_nlp_eval.py`'s own convention): load + validate the JSON,
      call `run_experiment`, print a one-line summary
      (`runner.summary_table`-style), exit 1 if `regressed`. → step 3 /
      SPEC.md FR-017. **Done 2026-09-18** — `--config`/`--source-db`/
      `--results-db` (`type=Path`, same defaults-to-`None`-override
      convention `cli/news_nlp_eval.py` already uses), same `sys.path`
      bootstrap every `cli/*.py`/`apps/*.py` entrypoint uses. "Exit 1 if
      `regressed`" needed no code of its own: `run_experiment` reuses
      `run_eval` verbatim (T-099), which already raises `SystemExit(1)`
      uncaught the moment a stage's headline metric drops past tolerance,
      before `run_experiment` can even return — same behavior
      `cli/news_nlp_eval.py`'s own `--check-regression` already has;
      disclosed in this module's own docstring rather than duplicated as
      dead code. An invalid spec's `ValidationError` likewise propagates
      uncaught — pydantic's own message already names exactly what's
      wrong (T-098's acceptance criterion), nothing here re-wraps it.
      Manually smoke-tested (real `uv run`, no monkeypatching): `--help`;
      an invalid spec (`pretrain.enabled` for `category`) surfaces
      pydantic's exact validation message and a non-zero exit; a
      monkeypatched `run_experiment` confirms `--source-db`/`--results-db`
      plumb through as plain strings and `summary_table`/`result_path`
      print correctly. No new test file — matches `cli/news_nlp_eval.py`'s
      own precedent of zero direct test coverage for the entrypoint script
      itself (only the library functions it calls are hermetically
      tested). 303 tests (unchanged — no test/src-behavior changes beyond
      the new CLI file itself), ruff, mypy clean.
- [x] **T-101** Backfill a JSON spec for every real historical experiment
      into `experiments/`: sentiment (`sentiment_v2_chunklevel_finetuned`,
      `_v3_downsampled`, `_v4_class_weighted`, `_v5_secbert_base`,
      `_chunklevel_base_finbert`), one production-config spec each for
      NER/category/`c_summary`. `experiments/README.md` discloses the two
      genuine gaps plainly: the removed "title-only" sentiment code path
      (no runnable equivalent) and category/`c_summary`'s real historical
      experiments (confidence-threshold calibration; generation
      output-length budget) being inference-time hyperparameters, a
      different axis than this schema — not silently omitted. → step 5 /
      SPEC.md FR-017. **Done 2026-09-18** — all 8 files written and
      validated against `ExperimentSpec` (values sourced from
      `docs/evaluation.md`'s own 2026-09-13/14/15 follow-ups, not
      guessed): the four sentiment `pretrain.enabled=true` specs share
      `split_seed=42`/`test_frac=0.1`/`val_frac=0.1`
      (`train_sentiment.py`'s own `_SEED`/`_TEST_FRAC`/`_VAL_FRAC`) and
      `eval.candidate_prescore_size=2500` (v4's/v5's own real
      `--candidate-prescore-size 2500 --sample-size 2000 --seed 1`
      invocation — `EvalSettings` otherwise silently defaults the
      pre-score pool to `sample_size` itself, under-reproducing the
      historical run without this override); `ner_production`/
      `category_production`/`c_summary_production` mirror each stage's
      own documented SPEC.md §9 baseline sample size/seed
      (8000/2800/1000, all seed=1).
      Found and disclosed a **third** gap beyond the task's own two
      while researching the exact historical values (forked a read-only
      verification pass rather than guess): `sentiment_v2_chunklevel_finetuned.json`
      and `sentiment_v3_downsampled.json` are byte-identical except
      `name` — `docs/evaluation.md` states v3 used "the same
      procedure/hyperparameters as before" as v2, differing only in
      whether a data file happens to exist on disk at train time
      (`train_sentiment.py`'s own silent `BALANCED_DATA_PATH` preference)
      — genuinely unrepresentable in `ExperimentSpec`, and (a real,
      disclosed consequence) running either spec **today** reproduces
      v3's behavior, not v2's original pre-rebalance pool, since the
      rebalanced file is now permanently checked into this repo.
      `experiments/README.md` also discloses that `ProsusAI/finbert`
      (the base-FinBERT comparison arm's `candidate_model`, and every
      sentiment `pretrain` spec's own `base_model`) was never pinned
      anywhere in this repo before — pinned here for the first time
      (`4556d13015211d73dccd3fdd39d39232506f3e43`, fetched live from the
      HF Hub API the same way T-001 pinned the four production models),
      scoped to this one spec file only, not added to `pipeline.py`'s own
      `MODEL_REVISIONS`.
      New `tests/news_nlp/test_experiments_backfill.py` (11 tests):
      every file in `experiments/*.json` validates against
      `ExperimentSpec` and its filename matches `spec.name`; the v2/v3
      content-identity is locked in as a regression test (not just
      prose), so a future "de-duplication" cleanup that silently changes
      one without updating the README gets caught; category/`c_summary`'s
      production specs never enable `pretrain`. 314 tests (303 + 11 new),
      ruff, mypy clean.
- [x] **T-102** Tests: a hermetic end-to-end `run_experiment` test (stub
      judge, no real GPU/LLM — the pattern already established in
      `test_eval_runner.py`) for at least one `pretrain.enabled=true` spec
      and one eval-only spec. (The `stratified_split`/`SentimentTrainConfig`
      regression test originally scoped here shipped with T-096 instead,
      and `ExperimentSpec` validation — every rejection case T-098 names,
      plus its `extra="forbid"` additions — shipped with T-098; see each
      task's own completion note.) Full hermetic suite stays green
      throughout. **Done 2026-09-18** — this task's one remaining named
      item (the end-to-end `run_experiment` test) shipped with T-099
      instead of being deferred here, same precedent as the other two;
      nothing left in this task's own scope. `tests/news_nlp/
      test_experiment_run.py`'s 2 tests are the acceptance proof — 303
      tests, full suite green throughout every task in this work item so
      far.
- [x] **T-103** Docs: new `docs/evaluation.md` section covering the
      schema + one-command workflow + the two disclosed gaps;
      `docs/modules/news-nlp.md` gains a pointer. Reconcile the two
      architecture artifacts (constitution AI behavior #11 — Portfolio
      Thesis + Portfolio NLP, reconcile only, never rename), same closing
      pass T-095 did for Work item 10. **Done 2026-09-18** —
      `docs/evaluation.md` gained a new "Experiments: JSON-driven,
      single-command runs" section (schema shape + example, the one
      command, and all **three** disclosed gaps — the task's own two plus
      the v2/v3 content-identity one found during T-101, summarized here
      and pointing to `experiments/README.md` as the source of truth
      rather than duplicating its full prose). `docs/modules/news-nlp.md`
      gained a new "### Experiments" pointer under its existing
      "Evaluation" section, plus a fix to its own now-stale "FTI
      architecture" bullet (still said category/`c_summary` "reuse
      `fti.NoOpTrainer` verbatim rather than declaring their own no-op
      `Trainer` subclass" — no longer true since T-097's
      `CategoryTrainer`/`SummaryTrainer`; also notes NER's new
      `base_model` field from T-099). `docs/db-topology.md` needed no
      change (Work item 11 added no DB tables). Both architecture
      artifacts reconciled (content only, titles untouched): Portfolio
      NLP gained a new "Experiments, added 2026-09-18" rule box in its
      eval section plus a footer changelog entry; Portfolio Thesis gained
      the proportionate shorter equivalent (diagram tooltip + status-table
      "next" clause + footer entry), matching T-095's own precedent for
      Work item 10's closing pass. This closes Work item 11 in full
      (T-096–T-103 all done).

## Work item 12 — Bring `tests/` under the mypy gate (done 2026-09-18)

`PLAN.md` Work item 12 / `SPEC.md` NR-007, §13 item 17. Surfaced directly
(2026-09-18) landing T-097: running mypy with an explicit path argument
(overriding `.code_quality/mypy.ini`'s own `files = src, apps, cli` scope,
purely to sanity-check a docstring claim) turned up 208 pre-existing
errors in `tests/`, `git stash`-confirmed unrelated to T-097. Root cause is
one systemic type-annotation bug, not 19 independent ones:
`tests/news_nlp/conftest.py`'s `conn`/`two_tier_conn` fixtures (and its
`write_stage_predictions`/`seed_article` helpers) are typed
`sqlite3.Connection` but actually construct and return `NewsNlpDatabase`
(`db_module.connect()`/`connect_pipeline()`) — a composition wrapper
around a `sqlite3.Connection`, not a subclass of it — and ~18 test files
copied that wrong annotation into their own test function signatures. Pure
type-hint precision gap, zero behavioral divergence. Four parts, mostly
sequential: T-104 (the `conftest.py` root-cause fix) must land before
T-105 (cascading it through the affected test files); T-106 (the small
disclosed unrelated fixes) and T-107 (widening `mypy.ini`'s scope) can
each follow once T-104/T-105 are in, with T-107 landing last so the gate
only tightens once everything it would flag is already clean; T-108
verifies the whole thing end to end.

- [x] **T-104** Retype `tests/news_nlp/conftest.py`'s `conn`/
      `two_tier_conn` fixtures and its `write_stage_predictions`/
      `seed_article` helpers from `sqlite3.Connection` to
      `news_nlp.NewsNlpDatabase`, matching what `db_module.connect()`/
      `connect_pipeline()` genuinely return. Zero behavior change — a
      declared-type correction only. → `PLAN.md` Work item 12, step 1.
      **Done 2026-09-18** — `conn`/`two_tier_conn` retyped
      `Iterator[NewsNlpDatabase]`; `write_stage_predictions` retyped
      `NewsNlpDatabase` (always called with one); `seed_article` turned out
      to have genuine dual callers — raw `sqlite3.Connection` from two of
      `conftest.py`'s own bootstrap helpers (`source_db_path`,
      `build_eval_source`) *and* the `conn` fixture's `NewsNlpDatabase`
      from other test files — so it's typed
      `sqlite3.Connection | NewsNlpDatabase` (both sides support the
      `.execute()` call it actually makes), not force-retyped, a deviation
      from the original "mechanical retype" framing caught while
      implementing. `test_db_path`/`results_db_path` also had their own
      local `conn` variable reused across two incompatible types
      (`sqlite3.connect()` then reassigned to `db_module.connect()`'s
      result) — renamed to `raw_conn`/`nlp_conn`. 12/12 of `conftest.py`'s
      own errors resolved, 0 new errors introduced.
- [x] **T-105** Cascade that corrected annotation into every test function
      across the ~18 affected files (`test_corrections.py`,
      `test_correction_endpoints.py`, `test_category_pipeline.py`,
      `test_summary_pipeline.py`, `test_ner_pipeline.py`,
      `test_sentiment_pipeline.py`, `test_sector_summary.py`,
      `test_queries.py`, `test_schema.py`, `test_db.py`,
      `test_eval_store.py`, `test_eval_sampling.py`,
      `test_eval_candidate.py`, `test_query_endpoints.py`,
      `test_pipeline_progress.py`, `test_fti_base.py`) that copied
      `conn: sqlite3.Connection` from the fixture — mechanical, the same
      edit repeated per file. → step 2. **Done 2026-09-18** — 9 files were
      a pure mechanical `sed` retype (no other `sqlite3.` usage in any of
      them, `import sqlite3` dropped as unused); `test_schema.py`/
      `test_db.py` needed the retype scoped to only their fixture-typed
      function parameters, leaving each file's own genuine raw
      `sqlite3.connect()` blocks (schema/WAL/foreign-key introspection)
      untouched; `test_pipeline_progress.py` needed `import news_nlp as db`
      added (didn't have it). One stale `# type: ignore[attr-defined]` on
      `two_tier_conn.articles_rel` (`test_db.py`) removed — no longer
      needed once `two_tier_conn` carries its real type.
      `test_eval_sampling.py` needed no direct edit (fixed transitively by
      T-104's `write_stage_predictions` retype). 191→163 errors after this
      step (208 total minus T-104's 45); 0 new errors.
- [x] **T-106** Fix the small number of genuinely unrelated errors `tests/`
      entering scope also surfaces, each its own small, disclosed fix, not
      a mechanical retype: `test_eval_candidate.py`'s deliberate
      `revision=None` call (swap for a placeholder string —
      `candidate_scored_connection`'s `revision` is a required `str`, and
      this test only exercises the earlier `stage`-validation rejection
      that never reaches it); `test_fti_base.py`'s remaining
      non-Connection issues (`Never`-typed raw `Feature()`/`Trainer()`
      calls in the two "raises NotImplementedError" tests, a
      `no-any-return`, a private `_rows` attribute read off a test
      double); `test_eval_tracking.py`'s 3 mlflow `list[Run] | Any`/
      `Experiment | None` union-narrowing spots. → step 3. **Done
      2026-09-18** — plus one genuinely new, previously-undisclosed find
      while re-verifying the file-by-file error breakdown from scratch
      (not caught by this work item's own filing pass): `test_eval_store.py`
      had 15 of its own errors, unrelated to the Connection pattern — two
      `common: dict[str, ...]` kwargs literals passed as `**common` into
      `create_eval_run`'s heterogeneously-typed keyword-only params, which
      a plain dict's inferred value-union type can't satisfy. Fixed with
      two local `TypedDict`s (`_RunKwargs`/`_RunKwargsWithStage`), not a
      production signature change. `test_fti_base.py`'s fixes:
      `Feature[Any, Any]()`/`Trainer[TrainConfig]()` explicit type params
      (bare `Feature()`/`Trainer()` infer `Never`); `_FakeInference.__init__`
      now sets `self._rows: list[str] = []` (was an undeclared ad-hoc
      instance attribute — a `ClassVar` alternative was tried first and
      rejected, since tests reassign `inf._rows` per-instance, which mypy
      forbids through a `ClassVar`); `_FakeConn`/`None` conn arguments cast
      via `typing.cast(NewsNlpDatabase, ...)`, documenting the fake/absent
      connection is never actually used on those code paths.
      `test_eval_tracking.py`: `assert not isinstance(runs, list)` narrows
      `mlflow.search_runs`'s `list[Run] | Any` return before `.iloc`/
      `.columns`; `assert exp is not None` before `.experiment_id`. 25
      tests across the four touched files re-run individually, all pass.
- [x] **T-107** Widen `.code_quality/mypy.ini`'s `files` from
      `src, apps, cli` to `src, apps, cli, tests` — the change that
      actually closes the gap for good. Deliberately excludes `scripts/`
      (one-shot historical scripts, `CLAUDE.md`'s own `scripts/`
      convention) — document `scripts/mine_idiom_sentences_2026_09_13.py`'s
      2 unrelated pre-existing errors here as a disclosed, deliberate
      exception, not silently left out. → step 4. **Done 2026-09-18** —
      `files = src, apps, cli, tests`, with an inline comment recording why
      and the `scripts/` exception; `uv run mypy --config-file=
      .code_quality/mypy.ini` (no path argument) now checks 67 files (not
      82 — confirms `scripts/` genuinely stays out) and reports zero
      errors.
- [x] **T-108** Verify: `uv run mypy --config-file=.code_quality/mypy.ini`
      (no path argument, the project's own documented command) reports
      zero errors under the widened scope; `uv run pytest` stays green
      throughout with zero assertion changes; `git stash`-verify the
      pre-fix state still reproduces the identical 208-error count (no
      surprise regressions folded in). Update `SPEC.md` §13 item 17 to
      resolved. → `PLAN.md` Work item 12 acceptance criteria. **Done
      2026-09-18** — `uv run mypy --config-file=.code_quality/mypy.ini`:
      `Success: no issues found in 67 source files`; `uv run pytest`: 279
      passed (same count as before this work item, zero assertion
      changes); `uv run ruff check .`/`ruff format --check .`: clean;
      `git stash` reproduced the identical 208-error/19-file count on the
      unmodified checkout, confirming nothing here was a pre-existing
      regression from elsewhere. `SPEC.md` §13 item 17 updated to resolved
      (see below).

## Work item 13 — Tighten judge-verdict reuse to also require a matching prediction (done 2026-09-19)

`PLAN.md` Work item 13 / `SPEC.md` FR-015 (tightened), §13 item 18. Surfaced
directly from a user request to look at whether the judge-verdict reuse
mechanism (T-091) was actually working. `experiment` (T-090:
`candidate_model or run_name or "base"`) doesn't prove "same model":
`run_experiment` (Work item 11) re-resolves `candidate_model` to a fixed
local checkpoint path (`train_sentiment.py`'s `OUTPUT_DIR`/
`OUTPUT_DIR_WEIGHTED`), so re-running the same `ExperimentSpec` after a
retrain reuses that same path/label even though the weights — and
therefore the prediction — genuinely changed. `find_verdict_json` only
keyed on `(article_id, task, experiment)`, so a second run's fresh,
different prediction silently got paired with the first run's now-stale
verdict: zero fresh judge calls, and the second run's own metrics computed
against a verdict that describes a different prediction than what the
model actually output that run. Confirmed by a direct, standalone
reproduction (not a hypothesis) before any code changed: two `run_eval`
calls, same `candidate_model`/`candidate_revision`, a fake model returning
`"positive"` then `"negative"` — run 2's `eval_inference.prediction_json`
correctly showed `"negative"`, but its `eval_verdict.verdict_json` still
read `ideal_label: "positive"`, `rationale: "judged:positive"`, and
`n_judged`'s progress bar showed `0it` (zero judge calls) for run 2.
Separately, and not itself a code bug, `docs/evaluation.md` had one stale
2026-09-15 comment calling `--run-name` "cosmetic only" — contradicted by
the actual T-090 behavior and by two other correct passages in the same
doc; fixed alongside this as a disclosed doc correction, not folded into
the code fix's own scope.

- [x] **T-109** Tighten `store.find_verdict_json` to require the
      candidate's paired `eval_inference.prediction_json` to match this
      run's own prediction, not just `(article_id, task, experiment)` —
      join `eval_verdict` to `eval_inference` on `inference_id` and add
      `AND i.prediction_json = ?`; thread the current item's
      `json.dumps(item.prediction, default=str)` through
      `runner._resolve_verdicts`'s existing call site (the same
      serialization `record_inference` uses, so the comparison is exact).
      A changed prediction under an otherwise-unchanged key now falls
      through to a fresh judge call, same as a genuinely new key. →
      `PLAN.md` Work item 13. **Done 2026-09-19** —
      `src/news_nlp/eval/store.py`'s `find_verdict_json` gained a required
      `prediction_json: str` kwarg and the join; `src/news_nlp/eval/
      runner.py`'s `_resolve_verdicts` passes it. Existing reuse tests
      (`test_run_eval_reuses_verdicts_for_an_unchanged_experiment`,
      `test_run_eval_judges_fresh_for_a_different_experiment`) needed no
      changes — both compare against unchanged production data, so their
      predictions never move between runs.
- [x] **T-110** Add regression coverage that fails without T-109's fix and
      passes with it, at both the unit level (`store.py`) and end-to-end
      (`runner.run_eval` with a real, toggling candidate model) — the
      end-to-end case directly mirrors the standalone reproduction used to
      confirm the bug, not a synthetic shortcut. Verify: full `uv run
      pytest` green, `uv run mypy --config-file=.code_quality/mypy.ini`
      (no path argument) zero errors, `uv run ruff check .`/
      `ruff format --check .` clean. → `PLAN.md` Work item 13 acceptance
      criteria. **Done 2026-09-19** —
      `test_find_verdict_json_ignores_a_stale_verdict_whose_prediction_changed`
      (`test_eval_store.py`) asserts the unit-level lookup directly;
      `test_run_eval_judges_fresh_when_the_candidate_prediction_changes_under_the_same_experiment`
      (`test_eval_runner.py`) runs `run_eval` twice with the same
      `candidate_model`/`candidate_revision` and a model that flips its
      predicted label between runs, asserting both a fresh judge call
      (`call_count == 2`, not `1`) and that each run's stored verdict
      matches that run's own prediction, not the other run's. `uv run
      pytest`: 317 passed. `uv run mypy --config-file=.code_quality/
      mypy.ini`: zero errors. `ruff check`/`ruff format --check`: clean.

## Work item 14 — Move production model selection into a git-tracked JSON config (done 2026-09-21)

`PLAN.md` Work item 14 / `SPEC.md` FR-018. The user asked to make the
pipeline's model selection JSON-driven, the same way Work item 11's
`ExperimentSpec` made experiments JSON-driven. Two decisions locked in
with the user up front, both driven by constitution AI-behavior #1
("model selection is pinned, not dynamic — swapping a model is a
spec-level change"): (1) a fixed, git-tracked config file with no runtime
CLI/env override — changing a model still means a PR, only the format
moved from Python to JSON; (2) per-stage scope of model + revision +
batch size, not just model + revision.

- [x] **T-111** Add `src/pipeline_config.py`: `PipelineModelsConfig`
      (strict pydantic, `extra="forbid"`, mirroring `experiment.py`'s
      `_StrictModel` convention) with one nested config per stage —
      `sentiment` (`model`/`revision` only) and `ner`/`category`/
      `c_summary` (`model`/`revision`/`batch_size`); `sentiment` has no
      `batch_size` field since `sentiment_stage.SentimentInference.
      batch_size()` is hardcoded to `1`, with no constant to move.
      `load_pipeline_models_config(path=CONFIG_PATH)` parses
      `config/pipeline_models.json`, `CONFIG_PATH` anchored via
      `Path(__file__)` since this loads at import time, not from one
      explicit CLI invocation. → `PLAN.md` Work item 14 step 1. **Done
      2026-09-21.**
- [x] **T-112** Add `config/pipeline_models.json` (new top-level `config/`
      directory) reproducing the pre-existing hardcoded model
      names/revisions/batch sizes verbatim — a pure move, zero behavior
      change. → `PLAN.md` Work item 14 step 2. **Done 2026-09-21.**
- [x] **T-113** Wire `src/pipeline.py`: call `load_pipeline_models_config()`
      once at import time (right after `load_dotenv()`) and derive
      `SENTIMENT_MODEL`/`NER_MODEL`/`CATEGORY_MODEL`/`SUMMARY_MODEL`/
      `MODEL_REVISIONS`/`NER_BATCH_SIZE`/`CATEGORY_BATCH_SIZE`/
      `SUMMARY_BATCH_SIZE` from it as the same plain module attributes as
      before (`MODEL_REVISIONS` stays a real, mutable `dict`) — every
      existing call site (`run_<stage>_stage`'s fresh-read-per-call
      pattern, `setup.py`'s direct import, every `pipeline.<NAME>` test
      monkeypatch/assertion, `scripts/resample_sentiment_v{3,4,5}_
      2026_09_15.py`'s runtime `MODEL_REVISIONS[...] = "local"` mutation)
      needed zero changes. Relocate (not drop) the design-rationale
      comments that used to sit next to the literal constants into the
      new loading block's own comment, since JSON has no comment syntax.
      A malformed/missing config file now fails on `import pipeline`
      itself, uncaught and unreworded, deliberately (not softened with a
      try/except fallback). → `PLAN.md` Work item 14 steps 3-4, 6. **Done
      2026-09-21.**
- [x] **T-114** Add `tests/news_nlp/test_pipeline_model_config.py`: the
      real config file parses and matches the pre-existing hardcoded
      values; `pipeline.py`'s exposed globals match a fresh config load;
      `MODEL_REVISIONS` stays a real `dict`; strict-mode rejection tests
      (unknown key, missing stage, non-positive `batch_size`, a
      `batch_size` under `sentiment`) via `tmp_path` fixtures, mirroring
      `test_experiment.py`'s `pytest.raises(ValidationError)` pattern.
      Verify: full `uv run pytest` green (zero changes to any
      pre-existing test), `uv run mypy --config-file=.code_quality/
      mypy.ini` zero errors, `ruff check .`/`ruff format --check .`
      clean. → `PLAN.md` Work item 14 step 5 acceptance criteria. **Done
      2026-09-21.**
- [x] **T-115** Update `docs/modules/news-nlp.md`'s "Model pins" section
      and `SPEC.md` §13 item 4 to name `config/pipeline_models.json` /
      `src/pipeline_config.py` as the source of truth instead of a
      `pipeline.py` dict literal, noting no constitution amendment was
      needed. → `PLAN.md` Work item 14 step 6. **Done 2026-09-21.**

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
