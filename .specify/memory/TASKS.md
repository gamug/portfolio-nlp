# TASKS.md — `portfolio-nlp`

Discrete, checkable task breakdown for `.specify/memory/PLAN.md`. Each
task references the plan work item and the `SPEC.md` section it closes.
Check a box only when its acceptance criterion (in `PLAN.md`) is actually
met — not when the code is merely written.

Task IDs are stable, same rule as `SPEC.md`'s `FR-0xx`/`NR-0xx`: don't
renumber; mark a cancelled/superseded task in place instead.

## Work item 1 — Pin HF model checkpoints (code, no blockers)

- [ ] **T-001** Look up and record the current commit SHA for each of the
      four HF model repos (`ProsusAI/finbert`, `gamug/sec-bert-finer-ord-ner`,
      `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`,
      `sshleifer/distilbart-cnn-12-6`) — from the HF Hub API/UI, not
      guessed. → `PLAN.md` Work item 1, step 1.
- [ ] **T-002** Add a revision constant per model in `src/pipeline.py`
      (paired with the existing `SENTIMENT_MODEL`/`NER_MODEL`/
      `CATEGORY_MODEL`/`SUMMARY_MODEL` name constants). → step 1.
- [ ] **T-003** Pass `revision=` at all 9 `from_pretrained` call sites in
      `src/pipeline.py` (lines 150–151, 283–284, 506–507, 747–748,
      785–786). → step 2.
- [ ] **T-004** Pass `revision=` in `src/setup.py`'s `download_models()`
      (`snapshot_download` + `AutoConfig.from_pretrained`, both calls, for
      all four models). → step 2.
- [ ] **T-005** Document the four pins (comment or a table in
      `docs/modules/news-nlp.md`) so a future bump is a reviewed, visible
      diff. → step 3.
- [ ] **T-006** Verify: `grep -rn "from_pretrained\|snapshot_download" src/`
      shows `revision=` at every call site; `uv run python -m setup`
      succeeds; `uv run pytest` stays green. → `PLAN.md` acceptance
      criteria.
- [ ] **T-007** Update `SPEC.md` §13 item 4 to note the reproducibility
      half resolved (annotate in place, keep the item number). → `PLAN.md`
      Work item 1, last acceptance criterion. Also update the two
      architecture artifacts per constitution AI behavior #11 (Portfolio
      Thesis + Portfolio NLP) — reconcile, never rename.

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

## Work item 3 — NER: validate the subword-fragmentation fix (priority, in progress)

`src/pipeline.py`'s `merge_bio_predictions` got a word-boundary-aware fix on
2026-09-10 (bogus single-token spans like `"3"`/`ORG` off `"3M"`), plus a
`_MIN_ENTITY_TEXT_LEN` last-resort filter in `run_ner_stage`. Neither has a
post-fix accuracy number yet — the `micro_f1` 0.7418 / `hallucination_rate`
0.338 baseline in `docs/evaluation.md` predates both. → `PLAN.md` Work item 3,
`SPEC.md` §9 (NER baseline row).

**Started 2026-09-12.** T-021's investigation is done and confirmed the
sampling mismatch is real; T-024 (the fix) landed the same day. T-020
turned out to be more tightly blocked than scoped, though — there's no
post-fix data to evaluate at all yet, and no plain pipeline re-run
produces any (see `docs/evaluation.md`'s 2026-09-12 follow-up for the full
findings). T-025 (new) is the lighter alternative to T-022 that fell out
of this investigation; both are still open maintainer decisions blocking
T-020.

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
- [ ] **T-020** Run a fresh `--stage ner` eval (`uv run cli/news_nlp_eval.py
      --stage ner --seed 1 --sample-size 1000`, or larger) against articles
      processed after the fix, to get a post-fix `micro_f1` /
      `hallucination_rate` / per-type F1 reading comparable to the
      2026-09-08 baseline. **Blocked, more tightly than originally scoped**
      — confirmed 2026-09-12 that no post-fix NER data exists at all: the
      stage hasn't run since 2026-08-19, and there is no backlog (SOURCE's
      max article id matches RESULTS' exactly) for a plain pipeline re-run
      to pick up. Cannot proceed until T-022/T-025 resolves how to produce
      *some* post-fix rows. → `PLAN.md` Work item 3, step 1.
- [ ] **T-022** *(maintainer)* Decide whether a bulk re-extraction of the
      existing 17.6M-row `article_entities` table (written under the
      pre-fix `merge_bio_predictions`) is worth doing — the fix is
      future-runs-only by design (`docs/evaluation.md` 2026-09-10
      follow-up, "Scope"). → step 3.
- [ ] **T-025** *(new, maintainer, lighter alternative to T-022)* Decide
      whether to reprocess a **targeted, modest sample** (e.g.
      1,000-2,000 articles) instead of the full corpus — technically
      available today via `news_nlp.corrections.delete_entities_for_article`
      (makes an article eligible for reprocessing again, same mechanism
      `delete_category` uses) followed by a normal pipeline run. Would
      unblock T-020 without committing to the full-corpus question T-022
      is scoped around. → step 1 (alternate path).
- [ ] **T-023** Once T-020 lands, add a dated follow-up entry to
      `docs/evaluation.md` (same append-only pattern as the sentiment/
      category entries) and update `SPEC.md` §9's NER baseline row —
      don't overwrite the 2026-09-08 baseline row itself. → step 4 /
      `PLAN.md` Work item 3 acceptance criteria.

## Work item 4 — Sentiment: close the entity/net-signal reasoning gap (priority, pending)

Stratified sampling + the `recall_negative` headline switch (both already
shipped, `docs/evaluation.md` 2026-09-08/09) improved what gets measured and
how it's weighted, but did not touch the model itself. The pilot run
(eval_run 18, n=800) still shows `negative` precision only 0.359 — the root
cause (FinBERT's whole-article softmax has no per-company or net-signal
reasoning the judge applies) is diagnosed but **not implemented**. This is
the stage still "pending to improve" despite the changes already made.
→ `PLAN.md` Work item 4, `SPEC.md` §13 item 1.

- [ ] **T-030** Design decision (not yet chosen) for closing the reasoning
      gap in `run_sentiment_stage` (`src/pipeline.py`): candidates include
      entity-scoped re-scoring using `article_entities`, a different
      sentiment model, or an explicit net-signal heuristic layered on the
      existing chunk-averaged score. → `PLAN.md` Work item 4, step 1.
- [ ] **T-031** Run a regression-tracked `--stage sentiment` eval at the
      recommended sample-size floor (~1,800-2,200, `docs/evaluation.md`
      "Sample-size floor") — only one stratified pilot (n=800) exists so
      far; a floor-sized run is needed before today's `recall_negative` /
      `precision_negative` can be trusted as a stable `--check-regression`
      baseline. → step 2.
- [ ] **T-032** Re-solve `docs/evaluation.md`'s "Sample-size floor" purity
      estimates using T-031's actual measured per-stratum agreement
      (`strata_json` / `agreement_rate_target_negative` etc.) instead of
      today's planning-only estimates, before locking in a permanent
      `--sample-size` default. → step 3.
- [ ] **T-033** After T-030 ships a change, re-run the eval and add a dated
      follow-up entry to `docs/evaluation.md`; update `SPEC.md` §13 item 1
      and §9's sentiment baseline row with the result. → step 4 /
      `PLAN.md` Work item 4 acceptance criteria.

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

## Work item 6 — Summarization (`c_summary` + `sector_summary`): validate eval scope, close the coverage gap, and add a lightweight sector-intro check (priority, pending)

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

- [ ] **T-050** Empirically check whether `c_summary` eval sampling
      suffers the same full-article-vs-lead-cap mismatch already
      confirmed for `sentiment` — `run_company_summary_stage`'s
      hierarchical reduce (`src/pipeline.py`) processes the whole
      article; confirm the judge's scope (`src/news_nlp/eval/sampling.py`,
      `.../prompts/c_summary.md`) matches. → `PLAN.md` Work item 6, step 1.
- [ ] **T-051** Decide whether/how to address `mean_coverage`'s weakness
      (3.02/5, weakest `c_summary` metric) — candidates: raise
      `SUMMARY_MIN_OUTPUT_TOKENS`/`SUMMARY_MAX_OUTPUT_TOKENS`
      (`src/pipeline.py`, currently 56/142), change the hierarchical-reduce
      strategy, or explicitly accept the terse tendency as a deliberate
      trade for the already-strong faithfulness score. Not yet decided.
      → step 2.
- [ ] **T-052** Run a fresh `--stage c_summary` eval after any change
      lands (or once T-050 rules out a code change) and add a dated
      follow-up entry to `docs/evaluation.md`. → step 3.
- [ ] **T-053** Add `SPEC.md` §13 item 10 (new — `c_summary` coverage +
      eval-scope question) and its §14 disposition-table row, and update
      §9's `c_summary` baseline row with the result. → `PLAN.md` Work
      item 6 acceptance criteria.
- [ ] **T-054** Confirm the `sector_summary` population size (rows =
      distinct `(gics_sector, gics_sub_industry, week)`) to decide whether
      a full-population judge pass is affordable each run, instead of
      assuming it and building sampling machinery that isn't needed.
      → `PLAN.md` Work item 6, step 4.
- [ ] **T-055** Add a `sector_summary` case to the eval framework: a new
      prompt (`src/news_nlp/eval/prompts/sector_summary.md`) judging
      `intro_text` for faithfulness against its own `facts_json` grounding
      only (never raw article/company text — that's not what the model
      saw); wire it into `sampling.STAGES`, `metrics.HEADLINE`, and
      `cli/news_nlp_eval.py`'s `--stage` choices. → step 4.
- [ ] **T-056** Run the new `--stage sector_summary` eval, log it to
      MLflow/`eval_run` the same way as the other stages, and record the
      result in `docs/evaluation.md`. → step 4.
- [ ] **T-057** Add a §9 baseline row (or a documented decision not to,
      if T-054/T-055 conclude a full new baseline isn't warranted) for
      the `sector_summary` intro check. → `PLAN.md` Work item 6
      acceptance criteria.

## Status

T-001–T-007 have no blockers and can begin immediately; T-010–T-016 are
blocked on the maintainer's infrastructure decision (see `PLAN.md` Work
item 2). Nothing in Work items 1–2 has started.

**Current focus is model performance checking (Work items 3-6):** T-040,
T-042, T-021, and T-024 are already done. Work item 3 (NER) is now
waiting on a maintainer call between T-022 (full backfill) and T-025
(targeted resample) to unblock T-020 — everything code-side that doesn't
need that decision is done. T-030 (sentiment design decision),
T-050 (`c_summary` sampling check), and T-054/T-055 (`sector_summary`
intro-check build-out) are the other next actionable, unblocked steps; the
rest of Work item 4 and T-041 follow once those land; T-051 (`c_summary`
coverage decision) is unblocked but, like T-030, needs a design call before
further steps.
