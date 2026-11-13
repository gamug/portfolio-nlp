# Model accuracy evaluation (`news_nlp.eval`)

The v1 chain stores `article_sentiment` / `article_entities` / `article_category`
/ `article_summary` and everything downstream (`portfolio-financial-analysis`'s
SEMANTIC score, the knowledge-graph projection) treats them as ground truth.
Nothing measures whether they are right. `news_nlp.eval` fills that gap with an
**LLM-as-judge** evaluation: an LLM scores a sample of the stored predictions
against the source article text, and the aggregate metrics + per-row verdicts go
to MLflow and to a handful of run-log tables in the RESULTS store (see
"Where results go" below).

The gold sets of Work item 18 (scope train and test, label check, extractor sample, sentiment) are described in
`docs/gold_datasets.md`: what each is for and where its files live. The models of the stepped pipeline (Work item 18) are
measured on those sets, not by this judge: `docs/scope_training.md`, `docs/extractor_training.md`, `docs/sentiment_training.md`,
`docs/summary_step.md`.

**The cut (step 1.2), final reading `units5` (owner, 2026-11-12).** Measured on the owner's marks, not by the judge: the extractor 351 of 359 faithful, 97.8% [95.5, 99.4] (`docs/extractor_training.md`); the output of the corpus cut `units5-d3e2cbfa830e`, 143 pieces of 28 bodies marked yes / no: faithful 84.6% [77.8, 89.6] on all pieces (the lower end is below the >= 80% bar, D32) and 89.1% [82.5, 93.4] without the 15 pieces scoped wrongly by step 1.1; all the levels and the other questions in `docs/scope_training.md` §8.3 (`scripts/cut_sample_metrics_2026_11_12.py`, marks in `gold/cut_sample_marks.csv`).

**What moved to legacy (2026-11-12).** The owner deprecated the v1 sentiment, zero-shot category and `sector_summary`
stages on 2026-11-12, and Work item 15 was superseded by Work item 18 on 2026-10-28: their follow-ups were moved verbatim
to `docs/legacy/evaluation-v1-stages.md` (a line in their place names each one). This page keeps the evaluation machinery, the NER
and `c_summary` follow-ups (NER and the summariser serve the stepped pipeline too) and the Work item 18 follow-ups. A
reference elsewhere to "`docs/evaluation.md`'s <date> follow-up" on sentiment, category, `sector_summary` or Work item 15
now resolves in that file.

## Architecture: the eval module reuses the pipeline's own inference classes

`news_nlp/eval/` does not load models independently. Each stage's own FTI
(Feature/Trainer/Inference, `src/fti.py`) `Inference` subclass —
`sentiment_stage.SentimentInference`, `ner_stage.NerInference`,
`category_stage.CategoryInference`, `summary_stage.SummaryInference` — is
the one place `from_pretrained`/forward-pass code for that stage lives;
`news_nlp/eval/candidate.py`'s `candidate_scored_connection` calls the same
classes directly to score a candidate model/revision (`--candidate-model`/
`--candidate-revision`, below) into a throwaway scratch RESULTS file,
rather than duplicating model-loading logic inside the eval module or
requiring the scratch-database-copy workaround the earlier
`resample_sentiment_v{3,4,5}` scripts used. `sampling.py` (stratified
sampling), `judges.py` (the judge LLM call), `verdicts.py` (structured
verdict models), `metrics.py` (pure aggregation), and `tracking.py`
(MLflow logging) are unaffected by this — only the model-scoring step
changed, from independent loading to reusing the pipeline's own classes
(PLAN.md Work item 10 step 3, TASKS.md T-082–T-089).

## The judge is a model, not a gold set

There is no hand-labelled corpus. The numbers this produces are
**agreement-with-a-judge**, not truth. A blind spot shared by FinBERT and the
judge inflates the score. Mitigations: the low-confidence bucket (below) surfaces
the rows most likely to be wrong, and every verdict's `rationale` is stored for
manual spot-checking. Treat a *drop* between runs as the signal, not the
absolute value.

## Baseline (2026-09-08)

First full-corpus run (`--sample-size 1000`, all four stages), taken as the
baseline for future `--check-regression` comparisons. Run IDs are the
`eval_run` primary key (RESULTS store); MLflow run ID is the artifact/param
store (`news_nlp_eval/<stage>`).

| stage | n | headline metric | value | eval_run / mlflow |
|---|---|---|---|---|
| `sentiment` | 1000 | `macro_f1_vs_judge` | 0.4006 | 10 / `823579c3` |
| `category` | 1000 | `accuracy_vs_judge` | 0.6890 | 11 / `e40ed7d6` |
| `ner` | 1000 | `micro_f1` | 0.7418 | 12 / `3b41cbec` |
| `c_summary` | 1000 | `mean_faithfulness` | 4.8670 | 13 / `053079fd` |

Per-stage notes from the full metric set (not just the headline):

- **sentiment** and **category** (v1 stages, deprecated by the owner on 2026-11-12): their baseline notes moved verbatim to `docs/legacy/evaluation-v1-stages.md`, "Baseline notes".
- **ner** — recall (0.84) well above precision (0.66): the model
  over-predicts. `hallucination_rate` 0.338 — about a third of predicted
  spans are wrong per judge. `PER` is the weakest type (F1 0.66 vs. `LOC`
  0.82 / `ORG` 0.75). `parse_fail_rate` 4.7% (n=953 of 1000 sampled) —
  expected per the error-only judge contract on entity-dense articles (see
  below).
- **c_summary (strongest stage)** — faithfulness 4.87/5 and
  `pct_with_hallucination` only 5.4%, but `mean_coverage` is just 3.02/5:
  summaries are accurate but not very complete, a terse/extractive tendency
  of `distilbart-cnn-12-6` more than a correctness problem.

*Moved verbatim to `docs/legacy/evaluation-v1-stages.md` (v1 stage deprecated by the owner on 2026-11-12, or Work item 15, superseded by Work item 18): "Follow-up (2026-09-08): sentiment judge text-scope bug found and fixed"; "Why recall, not F1, for sentiment negative (2026-09-08)"; "How much of the sentiment run-to-run swing is noise? (2026-09-08)".*

### Follow-up (2026-09-08): stratified sampling implemented

The section above flagged, but didn't implement, a sampling redesign to fix
the true-negative scarcity behind sentiment's wide `recall_negative` CI. It's
now implemented — see "Sampling" and "Statistical methodology" below for the
design (a `target_negative` stratum, soft-probability-targeted, on top of the
existing `low_conf`/random split) and "Sample-size floor" for updated,
measured-not-just-estimated `--sample-size` guidance. The historical numbers
in the "Baseline" and "How much of the sentiment run-to-run swing is noise?" (now in `docs/legacy/evaluation-v1-stages.md`)
sections above predate this and were produced by the old two-bucket design —
left as-is (not rewritten) as the historical record.

*Moved verbatim to `docs/legacy/evaluation-v1-stages.md` (v1 stage deprecated by the owner on 2026-11-12, or Work item 15, superseded by Work item 18): "Pilot run and why this error shape is the right one (2026-09-09)".*

*Moved verbatim to `docs/legacy/evaluation-v1-stages.md` (v1 stage deprecated by the owner on 2026-11-12, or Work item 15, superseded by Work item 18): "Follow-up (2026-09-09): hierarchical category classification"; "Follow-up (2026-09-09): first post-hierarchy pilot and threshold calibration"; "Follow-up (2026-09-09): corrected post-calibration numbers, and per-slug precision/recall/F1"; "Why precision, not recall, for category (2026-09-09)"; "The `other` bucket: a precision problem of its own (2026-09-09)".*

### Follow-up (2026-09-10): NER subword-fragmentation root cause and fix

Investigated a pattern the user spotted directly in real data: entities like
the bare digit `"3"` tagged `ORG`, apparently meant to represent 3M. This
turned out to be a real, previously-unidentified contributor to NER's
documented weakness (`hallucination_rate` 0.338, precision 0.66 vs recall
0.84 — Baseline section above) — not the whole gap, but a concrete, fixable
piece of it.

**Root cause**: a training/inference asymmetry in subword handling.
`train_ner.py`'s `make_tokenize_fn` labels only the first WordPiece subword
of each source word and masks every continuation subword to
`IGNORED_LABEL_ID` in the loss — the model gets zero training signal for
continuation-subword predictions. `pipeline.py`'s `merge_bio_predictions`
never accounted for this at inference: it merged spans by every token's own
raw BIO argmax, continuation subwords included. For `"3M"` tokenized as
`["3", "##M"]`, the model often predicted `B-ORG` on `"3"` but something
other than `I-ORG` on the untrained-for `"##M"` position, closing the span
after just `"3"` — with no length/sanity filter anywhere in the write path
to catch the resulting bogus `{entity_type: "ORG", text: "3"}` row.

**Quantified impact** (real data, `article_entities`, 17.6M rows): 2,982
bare single-digit `ORG` entities, of which **`"3"` alone is 1,859 (62%)** —
**628 of those in MMM (3M)-ticker articles, hitting 185 of 316 MMM articles
(59%)**. The same articles correctly tag whole `"3M"`/`"MM"`/`"M"` as `ORG`
459 times too — the model is genuinely inconsistent, not uniformly broken.
Confirmed via SOURCE `body_text` these are literal mid-word splits of real
"3M" mentions. The same bug class (not digit-specific) is independently
corroborated by real LLM-judge `wrong`-verdicts from past NER eval runs:
bogus `"3"` split off `"3Q24"`, bogus `"1"` split off `"1MDB"`, bogus `"20"`
split off `"20 años"`, bogus `"777"` split off `"Boeing's 777 jet"`. Also
affects `L3Harris` (`"L"`/`"L3"` fragments) but not `F5 Inc.`/`Phillips 66`
— specifically triggered by a digit glued directly to letters at a
WordPiece split boundary.

**Fix**: `merge_bio_predictions` is now word-id aware — only a word's first
subword ever opens, closes, or redirects a span; a continuation subword
(same `word_id`) just extends whatever the owning word's first subword
already decided, matching how the model was actually trained. A secondary,
narrow length-≥2 filter was added at write time in `run_ner_stage` as a
last-resort net for whatever the word-id fix doesn't structurally prevent —
not a substitute for it. `excludes_bare_digit`
(`portfolio_common.db.dialect.SqliteDialect`) remains in place on its two
existing call sites (`fetch_pending_company_summaries`,
`fetch_sector_week_entity_stats`) but was never the general fix — it only
ever caught bare single digits on two downstream read-side aggregate
queries, not the write path, not other fragment shapes (e.g. a lone `"L"`),
and not the main article-detail read.

**Scope**: future pipeline runs only, matching this repo's established
precedent (the category hierarchical-classification change was also
future-runs-only). The existing 17.6M-row `article_entities` table is
unchanged; a bulk re-extraction is a separate, not-yet-built follow-up.

This is a *different* issue from the full-article-vs-lead-cap NER
eval-sampling mismatch flagged earlier in this doc as "suspected but
unverified" — that's still open and unrelated to subword fragmentation.

### Follow-up (2026-09-12): NER full-article-vs-lead-cap mismatch confirmed; no post-fix data exists yet

Two findings while starting work on the NER validation task (`PLAN.md` Work
item 3 / `TASKS.md` T-020–T-023).

**The suspected sampling mismatch is real, not just suspected.** Ran
`sample_for_stage(conn, "ner", size=1000, seed=1)` against the real
RESULTS/SOURCE stores (no LLM calls — this only exercises the sampling
code) and compared the judge's `_MAX_BODY_CHARS` (6000) cap against the
*untruncated* `source.articles.body_text` for the same 1000 articles:

- **214/1000 (21.4%)** of sampled articles exceed 6000 chars (median body
  length 3549.5, mean 4800, **max 156,053** — over 4x the longest article
  `_SENTIMENT_MAX_BODY_CHARS`'s 100,000-char safety ceiling was sized
  against as of the 2026-09-08 sentiment fix (`docs/legacy/evaluation-v1-stages.md`); NER's population apparently
  has a much longer tail than sentiment's did).
- Of the **67,049** entities `run_ner_stage` predicted across those 1000
  articles, **11,262 (16.8%)** have a `start_char >= 6000` — i.e. the judge
  is structurally incapable of confirming or denying them, because it's
  never shown the text they came from. **209/1000 (20.9%)** of sampled
  articles have at least one such entity.
- This is a more direct problem than sentiment's version of the same bug:
  sentiment's judge saw a truncated body and produced a *worse-calibrated
  score*; NER's `parse_fail_rate`-adjacent error-only judge contract means
  a predicted entity the judge can't see the source text for likely reads
  as an ungrounded/`wrong` verdict (inflating false "hallucination" calls)
  or is silently skipped, while any *real* entity past char 6000 can never
  be named as `missed` (deflating the false-negative count) — both push
  the stage's already-weak `hallucination_rate` (0.338, 2026-09-08
  baseline) and precision/recall numbers in directions that don't reflect
  the model's actual behavior on the untruncated article.
- **Fixed same-day**: raised `ner` into `_UNCAPPED_STAGES`
  (`src/news_nlp/eval/sampling.py`) the same way `sentiment` was fixed
  (2026-09-08 follow-up, now in `docs/legacy/evaluation-v1-stages.md`) — `run_ner_stage` chunks and predicts over
  the *entire* article, so judging it on a lead-only cap was never a
  faithful comparison, same reasoning as sentiment's fix. Landed **before**
  spending any judge-call budget on a fresh NER eval run (below), so that
  run won't be immediately stale the way the pre-text-scope-fix sentiment
  baseline was. Unlike sentiment, `ner`'s uncapped safety ceiling
  (`_SENTIMENT_MAX_BODY_CHARS`, shared across both stages, not renamed) does
  engage against real data: the 156,053-char article found above exceeds
  it, so that one article's tail gets truncated rather than the request
  failing outright — the intended degradation, not a gap.

**No post-fix NER data exists to evaluate yet, and none can be produced by
just re-running the pipeline.** Checked the real stores directly:
`article_entities`'s newest row (by `id`) is dated **2026-08-19**, and
`article_sentiment`'s newest is **2026-08-17** — both stages haven't run
since, well before the 2026-09-11 `merge_bio_predictions` fix. Checked
whether a normal pipeline re-run would pick up new work: SOURCE
(`urls.db`) articles' max `id` (485,352) is **identical** to RESULTS'
lean-`articles` max `id` — there is no backlog of un-processed articles.
Every article already has a (pre-fix) NER row, and `run_ner_stage` only
processes rows *missing* from `article_entities` (FR-006, idempotent by
design) — so simply invoking `uv run cli/news_nlp_cli.py` again right now
would find zero pending NER work and write nothing new.

This means **T-020 is more tightly blocked on T-022 than originally
scoped** — not "a fresh eval run, then separately decide on backfill
scope," but "no fresh eval run is possible at all until *some* reprocessing
happens," even a small one. `delete_entities_for_article` (`news_nlp.
corrections`) already exists and makes a single article eligible for
reprocessing again (same mechanism `delete_category` uses, FR-009) — a
**targeted reprocessing of a modest sample** (e.g. 1,000-2,000 articles,
not the full 17.6M-row table) is technically available today without
committing to the full-corpus backfill question T-022 was originally
scoped around. Whether to do that, at what size, and whether category's
precedent (existing rows read back unmigrated until reprocessed, SPEC.md
§13 item 6) is an acceptable model for NER too, is still a decision for
the repo owner — not made unilaterally here.

### Follow-up (2026-09-12): T-025 executed — `article_entities` versioned, 20,000-article random resample

The repo owner chose T-025 (targeted resample) over T-022 (full-corpus
backfill), with one refinement: **version the table by renaming it**
rather than reprocessing specific rows in place via
`delete_entities_for_article`. `scripts/resample_ner_2026_09_12.py`
(one-shot, kept in git history per this repo's precedent for this kind of
maintenance script — see `docs/legacy/migration-2026-09-01.md`):

1. Renamed the pre-fix `article_entities` (17,666,722 rows / 458,867
   distinct articles) to **`article_entities_v1`** — nothing deleted, a
   durable, queryable snapshot of the pre-fix model's output. Its index
   was explicitly re-homed (`idx_article_entities_v1_article_id`) rather
   than left in place: a real SQLite gotcha found and verified (with a
   throwaway in-memory repro) before touching production data —
   `ALTER TABLE ... RENAME TO` does **not** rename a table's indexes, so
   the original `idx_article_entities_article_id` name would have stayed
   bound to the renamed table, making `init_schema`'s
   `CREATE INDEX IF NOT EXISTS` of that same name against the fresh table
   silently no-op and leave it unindexed.
2. Recreated a fresh, empty `article_entities` via `news_nlp.init_schema`
   (verified indexed, verified empty, before proceeding).
3. Ran `run_ner_stage` over a `random.Random(seed=1)`-seeded, reproducible
   random sample of 20,000 pending articles (`news_nlp.queries.
   fetch_pending_articles`'s new `sample_seed` parameter — every article
   was "pending" against the just-emptied table) under the current, fixed
   code (both the 2026-09-10 word-boundary fix and the 2026-09-12
   uncapped-eval-sampling fix above).

Dry-run validated end to end against a scratch copy of the real store
(50-article sample) before running for real.

**Result**: 714,334 entity rows across **19,988 of the 20,000** sampled
articles (12 articles genuinely predicted zero entities — `write_entities`
writes no row at all for those, a pre-existing, unrelated behavior). Ran in
~15 minutes on the project's GPU (RTX 4050 Laptop, matching `SPEC.md`
NR-001's 6GB VRAM budget). Per-type split: `ORG` 409,575 (57.3%), `PER`
159,912 (22.4%), `LOC` 144,847 (20.3%) — close to the pre-fix corpus's
documented ORG 55%/PER 25%/LOC 20% split (Sampling section below), so the
fix didn't distort the overall type balance.

**T-020 is now unblocked**: 19,988 post-fix articles exist to draw an eval
sample from, where none existed before this follow-up. Running that eval
(`--stage ner`) is the natural next step but was **not** done as part of
this follow-up — it spends real judge-call budget and is a distinct
action from the reprocessing done here.

### Follow-up (2026-09-12): T-020 executed — post-fix NER eval, both fixes confirmed

Ran `uv run cli/news_nlp_eval.py --stage ner --seed 1 --sample-size 8000`
(mlflow run `329f9222`, `news_nlp_eval/ner`, `code_version=18fc30d` — both
the word-boundary merge fix and the uncapped-sampling fix are in). Sample
size is 8x the 2026-09-08 baseline's 1000 and deliberately large relative
to the population: it's a 40% draw of the 19,988-article T-025 resample
pool, not a ~0.2% draw of the full ~459K-article corpus like the baseline
was — that resample is the only place post-fix rows exist yet (see the
T-025 follow-up above). `n_judged=8000`, usable `n=7965` after parse
failures.

| metric | 2026-09-08 baseline (pre-fix, `3b41cbec`) | 2026-09-12 (post-fix, `329f9222`) | Δ |
|---|---|---|---|
| `micro_f1` | 0.7418 | **0.8578** | +0.116 (+15.6%) |
| `micro_precision` | 0.6623 | 0.8403 | +0.178 (+26.9%) |
| `micro_recall` | 0.8429 | 0.8761 | +0.033 (+3.9%) |
| `hallucination_rate` | 0.3377 | **0.1597** | −0.178 (−52.7%) |
| `miss_rate` | 0.1571 | 0.1239 | −0.033 (−21.1%) |
| `macro_f1` | 0.7429 | 0.8729 | +0.130 |
| `f1_ORG` | 0.7454 | 0.8117 | +0.066 |
| `f1_LOC` | 0.8236 | 0.8807 | +0.057 |
| `f1_PER` | 0.6597 | **0.9262** | +0.267 (+40.4%) |
| `parse_fail_rate` | 4.7% | 0.44% | −91% (relative) |
| `mean_entities_per_article` | 43.25 | 32.45 | −25% |

**Reads as the fix working, not noise**: the gain is concentrated in
precision (+26.9%) and `hallucination_rate` (halved), while recall barely
moves (+3.9%) — exactly the shape expected from a fix that removes bogus
spans rather than one that would help the model find new true entities.
`f1_PER`'s outsized jump (+40.4% relative, the largest of the three types)
matches the root cause: person names carry more multi-subword WordPiece
splits than org/location names, so they were hit hardest by the pre-fix
training/inference asymmetry. `parse_fail_rate` collapsing 4.7%→0.44%
independently corroborates the same-day uncapped-sampling fix (T-024) —
the judge is no longer choking on entities past the old 6000-char cap.
`mean_entities_per_article` dropping 25% is the same precision effect
showing up as a raw count (fewer bogus single-token spans written at all).

**Caveat — not a full-corpus result.** This run only speaks to the 19,988
articles reprocessed under the fixed code (T-025). The remaining ~439,000
pre-fix articles (`article_entities_v1`) are untouched and would still
score at the old ~0.74 baseline until reprocessed — T-022's full-corpus
backfill question is unaffected by this result either way. Also note: the
baseline run had no `--seed`, so strata composition isn't reproducibly
comparable row-for-row against this seeded run — the metric-level
before/after above is still valid, just not a row-identical replay.

This clears the last open item in `PLAN.md` Work item 3 / `TASKS.md`
T-020; `SPEC.md` §9's NER row is updated below.

*Moved verbatim to `docs/legacy/evaluation-v1-stages.md` (v1 stage deprecated by the owner on 2026-11-12, or Work item 15, superseded by Work item 18): "Follow-up (2026-09-13): sentiment — three designs measured; fine-tuning wins"; "Follow-up (2026-09-13): the "crushed earnings" idiom gap — diagnosed, mined, fixed, measured"; "Follow-up (2026-09-13): diagnosing the remaining precision gap, testing title-only against it, and selecting a production candidate"; "Follow-up (2026-09-13, same day): the selected design merged into the pipeline"; "Follow-up (2026-09-14): sentiment training data rebalanced — a real trade, not a strict win"; "Follow-up (2026-09-15): a second rebalancing approach — class-weighted loss, no data discarded"; "Follow-up (2026-09-15, same day): a complete, consistent metric set for all three candidates — accuracy_ovr added, v2's precision/recall backfilled"; "Follow-up (2026-09-15, same day): v4's downstream production-pipeline eval — the number that actually validated v2, now run for v4 too"; "Follow-up (2026-09-15, same day): a third approach — swap the base checkpoint (`nlpaueb/sec-bert-base`), rejected before a downstream eval"; "Follow-up (2026-09-15, same day): v5's downstream production-pipeline eval, run despite the offline rejection above"; "Follow-up (2026-09-15, same day): Work item 9 decided — v4 adopted for production"; "Follow-up (2026-09-15, same day): sentiment's downstream eval narrowed to one-vs-rest metrics only".*

### Follow-up (2026-09-14): c_summary full-article-vs-lead-cap mismatch confirmed and fixed

Started `PLAN.md` Work item 6 (summarization eval validation) by checking
the same suspicion already confirmed for sentiment (2026-09-08) and NER
(2026-09-12): does `c_summary`'s eval judge see the whole article the
pipeline actually summarized, or just the lead capped at `_MAX_BODY_CHARS`
(6000 chars)? Code inspection alone already pointed at yes — `pipeline.
run_company_summary_stage` -> `hierarchical_summarize_batch` chunks and
reduces over the *entire* `body_text` (via `build_company_summary_input`,
which concatenates the full, untruncated body), while `c_summary` was not
in `_UNCAPPED_STAGES` — but this project's standing rule is to measure
before fixing, not infer from code shape alone.

Measured directly against real data (`article_summary` joined to
`source.articles.body_text`, all 458,641 rows — no LLM calls, pure
sampling-layer arithmetic):

- **45,867 / 458,641 (10.0%)** of `article_summary` rows have `body_text`
  longer than the judge's 6000-char cap. `body_text` length: p50=2,763,
  p90=6,001, p95=7,499, p99=13,538, **max=156,053** (the same tail NER's
  2026-09-12 follow-up found, since both stages draw from the same
  article population).
- **Every one of those 45,867** is also a multi-chunk summary
  (`num_chunks > 1`) — i.e. this is exactly the population where
  `hierarchical_summarize_batch`'s reduce pass synthesizes content from
  more than one chunk, content a 6000-char judge view can never fully
  show.
- A broader **131,944 / 458,641 (28.8%)** have `num_chunks > 1`, but the
  other 86,077 of those are multi-chunk despite a body *under* 6000
  chars — that's the summarizer's own token budget (dense text hitting
  BART's ~1000-token leaf-chunk limit before 6000 characters), a
  separate phenomenon from the judge-cap mismatch, not evidence of it.
  The clean, judge-relevant figure is **10.0%**, not 28.8%.
- Smaller in magnitude than NER's 21.4% (2026-09-12), but the same
  structural bug: a real, double-digit-percent slice of the population
  where the eval judge is scoring the model against a truncated view of
  the text it actually processed.

**Fixed same-day**: added `c_summary` to `_UNCAPPED_STAGES`
(`src/news_nlp/eval/sampling.py`), the same fix shape as sentiment
(2026-09-08) and NER (2026-09-12) — it inherits the shared
`_SENTIMENT_MAX_BODY_CHARS` (100,000) safety ceiling, which does engage
for the 156,053-char tail article the same way it already does for NER.
Added `test_c_summary_uses_full_body_text_since_the_2026_09_14_fix`
(`tests/news_nlp/test_eval_sampling.py`), mirroring the existing
sentiment/NER tests of the same shape. Full hermetic suite still passes.

Not yet done (remaining Work item 6 steps): decide whether/how to address
`c_summary`'s weak `mean_coverage` (3.02/5, pre-fix baseline); re-run the
`c_summary` eval post-fix and record the result here; add a
faithfulness-only eval path for `sector_summary`'s `intro_text` (currently
zero coverage).

### Follow-up (2026-09-14, same day): post-fix c_summary re-run — coverage up, a previously-invisible hallucination gap revealed

Re-ran `c_summary` (`--stage c_summary --sample-size 1000 --seed 1`,
`code_version` `6312be9` — includes the uncap fix above) against the real
stores (`eval_run` 34, mlflow `72cf167d`). Strata drawn: `low_conf`
200/458,641, `target_chunks_1` 48/326,590, `target_chunks_2` 144/110,911,
`target_chunks_ge3` 288/20,940, `representative` 320/457,961.

| metric | 2026-09-08 baseline (pre-fix, old sampling design) | 2026-09-14 (post-fix, HT) | 2026-09-14 (naive pooled) |
|---|---|---|---|
| `mean_faithfulness` | 4.867 | 4.781 | 4.583 |
| `mean_coverage` | 3.02 | **3.639** | 3.304 |
| `mean_conciseness` | (not in baseline headline) | 4.568 | 4.362 |
| `pct_with_hallucination` | 0.054 | **0.085** | 0.154 |

Two things changed since the 2026-09-08 baseline at once, not one — an
honest confound, same shape as NER's 2026-09-12 before/after: (a) this
fix (judge now sees the whole article), and (b) the `num_chunks`-tiered
target stratification (`_CSUMMARY_CHUNK_TIERS`) was implemented the same
day as the 2026-09-08 baseline but *after* it was recorded, so the
baseline used the old, un-stratified low_conf/random design. The
headline deltas above can't be cleanly attributed to the uncap fix
alone.

What the per-stratum breakdown shows regardless of that confound —
`num_chunks >= 3` articles (`target_chunks_ge3`, exactly the population
this fix targets) are the weak spot on **both** metrics, more sharply
than the aggregate suggests:

| stratum | mean_faithfulness | mean_coverage |
|---|---|---|
| `target_chunks_1` (single-chunk) | 4.938 | 3.833 |
| `target_chunks_2` | 4.542 | 3.229 |
| `target_chunks_ge3` | **4.229** | **2.861** |

**`mean_coverage` genuinely improved** (3.02 → 3.64 HT) — plausible
mechanism: capped at 6000 chars, the judge could previously only check
the summary against the article's lead, so real coverage of later
material (which the hierarchical reduce does draw on) had no visible
ground truth to confirm — this fix lets the judge actually credit it.

**`pct_with_hallucination` went up, not down** (0.054 → 0.085 HT / 0.154
naive) — the opposite direction a pure "judge sees more, catches more
context" story would predict if the model were unchanged and only
overlooked-hallucinations were surfaced from behind the old cap. Plausible
mechanism: a hallucinated detail drawn from post-cap content previously
had no ground truth for the judge to check it against either, so it likely
read as unverifiable rather than confirmed-wrong under the old cap; the
uncap fix lets the judge actually confirm real fabrications it couldn't
see before. **Not yet independently verified** — no rationale-text audit
of the newly-flagged hallucination cases has been done this pass (unlike
the sentiment diagnosis's confusion-matrix/rationale audit); flagged here
as a plausible but unconfirmed mechanism, per this project's evidence bar.

**Treat `eval_run` 34 (this run) as the new post-fix `c_summary` baseline**
for regression tracking going forward — the 2026-09-08 numbers predate
both the uncap fix and the stratification redesign and are no longer a
fair `--check-regression` comparator, the same disposition NER's
pre-2026-09-12 baseline was given.

**Work item 6 step 2 (the `mean_coverage` decision), informed by this
data**: the fresh HT `mean_coverage` (3.64/5) is meaningfully better than
the stale 3.02/5 figure the item was originally scoped against, and the
per-stratum breakdown shows the residual weakness concentrated in
`num_chunks >= 3` articles (2.86/5, 4.6% of the corpus) rather than
spread evenly.

### Follow-up (2026-09-14, same day): a bigger output-length budget was tried and rejected

Tested the obvious next lever — `SUMMARY_MIN_OUTPUT_TOKENS`/
`SUMMARY_MAX_OUTPUT_TOKENS` (56/142, `src/pipeline.py`) are literally
`distilbart-cnn-12-6`'s own stock CNN/DailyMail generation defaults,
never retuned for longer, denser financial-news articles — before
touching the full corpus.

**Matched-pair experiment, not a fresh independent sample**: took 200
articles already judged in `eval_run` 34 (70/70/60 across the
`num_chunks` 1/2/3+ tiers), regenerated their summaries with
`min=100, max=220` — everything else byte-for-byte the same production
code path (`hierarchical_summarize_batch`, same model, same chunking) —
and re-judged the new summaries with the same judge/prompt, so the
comparison is the same articles, old vs. new settings, not two
different random draws. (One data wrinkle surfaced and fixed along the
way: reconstructing each article's original summarization input needed
the sentiment label/score that was live *when that summary was
generated* — `article_sentiment_v1`, the pre-fine-tune-swap table with
full 459,112-row coverage — not the current `article_sentiment` table,
which only holds 7,520 rows re-scored under the new fine-tuned model so
far.)

| metric | old (56/142) | new (100/220) | Δ |
|---|---|---|---|
| `mean_faithfulness` | 4.595 | 4.070 | −0.53 |
| `mean_coverage` | 3.315 | 3.515 | +0.20 |
| `mean_conciseness` | 4.380 | 3.725 | −0.66 |
| `pct_with_hallucination` | 15.0% | **35.5%** | more than doubled |

Per-tier coverage: `chunks=1` 3.814→**4.143** (crosses the 4/5 target),
`chunks=2` 3.143→3.357, `chunks>=3` (the actual weak spot) barely moves:
2.933→2.967.

**Rejected.** A bigger output budget does nudge coverage up, but mostly
on the single-chunk summaries that were already closest to fine — it
does almost nothing for the `chunks >= 3` population that's actually
driving the weak aggregate, and it more than doubles the hallucination
rate while dragging faithfulness and conciseness down with it. That
population's problem isn't output length; it's information loss
compounding across multiple summarize-of-summaries reduce passes
(`_reduce_pass`) — a bigger per-pass token ceiling doesn't undo hops of
lossy compression that already happened. This is the same discipline
this project applied to the sentiment precision investigation: test the
obvious lever on real data before adopting it, and report a negative
result plainly rather than force a metric up at a hidden cost.

**Decision (2026-09-14): accept `mean_coverage` as a deliberate
completeness-vs-correctness trade — no further fix attempted.** Every
measured lever for raising coverage moves the same knob the same
direction: more output allowed/forced means more room for the model to
pad with plausible-sounding invented detail alongside (or instead of)
genuinely-extracted fact, which is exactly what the rejected experiment
above showed happening (`pct_with_hallucination` more than doubling).
`mean_faithfulness` (4.78/5) is the stage's strongest metric and the one
that matters most for a *summary* specifically — a reader relies on a
summary to say only true things, more than to say every true thing. An
incomplete-but-accurate summary is a bounded, honest gap: a reader gets
less than the full article but nothing they read is wrong. A
complete-but-sometimes-fabricated one is worse: a reader has no way to
tell which parts are real without going back to the source, which
defeats the point of summarizing at all.

This mirrors, in the opposite direction, the same trade-off shape
sentiment's 2026-09-13 "pessimist model" decision made: sentiment
chose to risk more false alarms (lower precision) rather than risk
missing a real signal (lower recall), because a missed signal is an
unrecoverable blind spot downstream while a false alarm is visible and
discountable. Summarization's asymmetry runs the other way — a
fabricated detail is the unrecoverable failure mode here (a reader
can't tell it's fabricated), while an omitted detail is the visible,
recoverable one (the reader can tell the summary is thin and go read
the source). Recall was worth the false-alarm cost for sentiment;
completeness is not worth the fabrication cost for c_summary.

**What this decision does NOT close off**: a narrower fix scoped only
to the reduce pass itself (e.g. a larger `max_input_tokens` per chunk,
to cut the number of lossy reduce hops `chunks >= 3` articles go
through, rather than a bigger *output* budget on every pass) was never
tested and remains a legitimate future candidate if `chunks >= 3`
coverage (2.86/5) is later judged unacceptable on its own. This
decision is about not chasing the output-length lever further, not
about the coverage gap being permanently untouchable.

*Moved verbatim to `docs/legacy/evaluation-v1-stages.md` (v1 stage deprecated by the owner on 2026-11-12, or Work item 15, superseded by Work item 18): "Follow-up (2026-09-14, same day): sector_summary's intro_text eval added — a real, sizable faithfulness gap found"; "Follow-up (2026-09-14, same day): confirmed the hallucination pattern is data-independent, then fixed it with a deterministic template".*

### Follow-up (2026-09-18): FTI restructure, judge-verdict reuse, confusion matrix, and ROC/AUC (Work item 10)

The FTI migration itself (all four ML stages moved onto `src/fti.py`'s
`Feature`/`Trainer`/`Inference` base classes, `sector_summary`'s
orchestration moved to its own `news_nlp/sector_summary/` package, fully
decoupled from FTI since it never touches a model/GPU) is what let
`news_nlp/eval` stop duplicating model-loading code — see "Architecture"
above. Three further pieces landed on top of that, all confined to
`news_nlp/eval/` and its schema:

- **Judge-verdict reuse (TASKS.md T-091, tightened by T-109).** Before
  invoking the judge LLM for a sampled `(article_id, task, experiment)`,
  `runner.py` now checks `eval_verdict` for an existing row at that exact
  key **and** whose paired `eval_inference.prediction_json` matches this
  run's own prediction for that article, and reuses it
  (`store.find_verdict_json` + `pydantic`'s `model_validate_json`) instead
  of re-judging. The concrete guarantee this gives: re-running the same
  `--stage <s> --run-name <experiment>` invocation against an unchanged
  sample makes **zero** new judge-LLM calls; a new `article_id`, a
  different `experiment`, or a changed prediction under the same
  `(article_id, task, experiment)` key always judges fresh. That last case
  matters because `experiment` alone doesn't prove "same model": `--run-
  name`/`--candidate-model` (and `run_experiment`, which resolves
  `candidate_model` to a fixed local checkpoint path) can stay identical
  across a retrain, so without the prediction check a retrained model's
  genuinely new prediction for an already-judged article would silently
  reuse the *old* model's stale verdict (found + reproduced 2026-09-19,
  `TASKS.md` T-109, `SPEC.md` FR-015). `eval_verdict`'s unique
  constraint is `(article_id, task, experiment, run_id)` — a 4-column key,
  not the 3-column `(article_id, task, experiment)` PLAN.md's original text
  named, since this repo's own T-090 PR had already shipped the 4-column
  version; a dedicated index covers the 3-column reuse lookup instead of a
  schema migration.
- **`eval_confusion` (TASKS.md T-092).** One row per distinct
  `(run_id, task, experiment, true_label, predicted_label)` cell actually
  observed in a run's own sample — sentiment/category only (the only two
  stages with a discrete predicted/ideal label shape), built from the same
  judge verdicts T-090/T-091 already record, no new judge calls. Sparse (a
  cell with zero occurrences has no row) and scoped per `run_id`, matching
  `eval_inference`/`eval_verdict`'s own full-history design — a cumulative
  matrix across an experiment's every historical run is a plain
  `SUM(count) GROUP BY (experiment, task, true_label, predicted_label)` at
  query time, not something write time maintains.
- **`roc_auc_<class>` (TASKS.md T-093).** One-vs-rest ROC AUC added to
  `aggregate_sentiment`/`aggregate_category`, computed from each sampled
  row's own stored per-class probability (`article_sentiment`'s
  `positive`/`negative`/`neutral` columns; `article_category`'s 9-way NLI
  distribution) — no new data collection. Via the weighted Mann-Whitney U
  statistic, the natural generalization of rank-based AUC to this doc's
  own Horvitz-Thompson per-item weighting (below): both an HT-weighted
  default and a `_naive_pooled` companion, matching every other per-class
  metric here. `category`'s `roc_auc_<slug>` iterates the fixed 9
  substantive slugs only — `"other"` is a threshold fallback with no raw
  NLI score column of its own, so `roc_auc_other` is never produced (every
  other per-slug metric family here does include `other`, since those
  compare discrete labels, not raw scores). `aggregate_ner`/
  `aggregate_c_summary`'s metric key sets are explicitly unchanged — locked
  by a dedicated regression test (TASKS.md T-094) asserting their exact
  key sets, not just documented as a claim.

(The `GET /eval/latest`/`--check-regression` `experiment`-awareness gap
this follow-up originally left open was closed later the same day — see
"Where results go" below.)

*Moved verbatim to `docs/legacy/evaluation-v1-stages.md` (v1 stage deprecated by the owner on 2026-11-12, or Work item 15, superseded by Work item 18): "Follow-up (2026-10-22): what the upstream ticker re-tag did to the stored results and to category's measured numbers (Work item 15, T-133 step 1)"; "Follow-up (2026-10-23): offline simulation of candidate decision rules, no judge call (Work item 15, T-121(a))".*

## What it evaluates

Four per-article stages, plus `sector_summary`'s own one-sentence
`intro_text` (added 2026-09-14) — the rest of `sector_summary` stays out
of scope, since it's deterministic composition (§9/§13, `SPEC.md`). These are the v1 chain's stages; the code still
evaluates all five, and the follow-ups of the deprecated three (sentiment, category, `sector_summary`) are in
`docs/legacy/evaluation-v1-stages.md`. The `--check-regression` gate is out of the MVP (owner, 2026-11-10).

| stage | headline metric | also logged |
|---|---|---|
| `sentiment` | `recall_negative`¹ ² | per-class precision/recall/F1, `accuracy_ovr_<class>`, `roc_auc_<class>` ³ (all one-vs-rest — no aggregate/multi-class summary, see "sentiment's downstream eval narrowed to one-vs-rest metrics only" in `docs/legacy/evaluation-v1-stages.md`) |
| `category` | `accuracy_vs_judge` ² | macro-F1, per-slug precision/recall/F1/`accuracy_ovr`/`roc_auc` ³, model vs judge `other`-rate, mean severity |
| `ner` | `micro_f1` | span micro/macro P/R/F1, per-type F1, hallucination rate, miss rate. Error-only judge contract: it names just the `wrong` predicted spans + `missed` entities (not a verdict per span, which overflows on entity-dense articles); TP/FP/FN are derived from the predicted count. |
| `c_summary` | `mean_faithfulness` ² | mean coverage / conciseness (1-5), `pct_with_hallucination` |
| `sector_summary` | `mean_faithfulness` ² | `pct_with_hallucination` — faithfulness-only (no coverage/conciseness), full population every run, not a sample (see the 2026-09-14 follow-up in `docs/legacy/evaluation-v1-stages.md`) |

> **Metrics of record and of model selection (owner, 2026-11-09; `PLAN.md` D37, `SPEC.md` FR-019).** The
> metrics to record, and to use in model selection, are **Precision, Recall and ROC OVR**, per class
> (ROC OVR is the AUC of one class against the rest, from the model's probability of that class). They
> are what `news_nlp.eval` already logs per class for `sentiment` and `category` (`precision_<class>`,
> `recall_<class>`, `roc_auc_<class>`, below), and what the sentiment fine-tune on the owner's summaries
> records for every run and selects by (`src/sentiment_pair_report.py`: the selection score is the mean of
> the nine figures of the three classes; the pin rule reads their paired gain). **Accuracy (agreement) is a
> description, not a selection metric.** The `headline metric` column below is the older judge-based
> headline of this module and is not a selection rule; where the two disagree, D37 decides.
>
> The experiments after the pair trainer (cross-validation over all tags, the relevance model and the
> stack) and their figures are in `docs/sentiment_training.md`.

Every run also logs `n` (rows judged) and `parse_fail_rate` (judge replies that
were not valid JSON after one repair attempt — excluded from the accuracy
numbers).

¹ Not F1 or accuracy: a missed real negative-sentiment article costs more for
portfolio construction than an over-flagged neutral one, so `--check-regression`
gates sentiment on recall specifically — see "Why recall, not F1, for
sentiment negative" in `docs/legacy/evaluation-v1-stages.md`.

² Every headline/"also logged" metric marked here has a `_naive_pooled`
companion (e.g. `recall_negative_naive_pooled`, `accuracy_vs_judge_naive_pooled`,
`mean_faithfulness_naive_pooled`) — the old flat pooled computation, kept for
sanity-checking the Horvitz-Thompson reweighting below and for continuity with
pre-redesign runs. `ner`'s metrics have no such companion because they're
mathematically identical to it already (see "Statistical methodology" below).

³ `roc_auc_<class>` (added 2026-09-18, TASKS.md T-093) is computed from each
row's own stored per-class probability, not from the discrete predicted
label the other per-class metrics use — see "FTI restructure, judge-verdict
reuse, confusion matrix, and ROC/AUC" above for the formula.

*Moved verbatim to `docs/legacy/evaluation-v1-stages.md` (v1 stage deprecated by the owner on 2026-11-12, or Work item 15, superseded by Work item 18): "Follow-up (2026-10-23): the judged calibration sample, stage 1 (Work item 15, T-121(b))".*

### Follow-up (2026-10-29): the owner's tags v1, and sizing the triage-first pipeline (Work item 18, T-144/T-145)

**The owner's 720 tags (tags v1, a development set; gold v2 will be the re-tag under the new taxonomy, so every agreement figure below is against tags v1 and is re-measured on gold v2).** The owner hand-tagged the 720 articles of eval_run 45
(`data/category_gold_review.csv`, git-ignored; `scripts/export_gold_review_2026_10_26.py`). The
LLM's labels agree with the owner on 46.3% of them (57.9% with `other`, the owner's new
`miscellany` and `unrelated` merged). The owner's tags: `market_analyst_sentiment` 141,
`miscellany` 94 (multi-company round-ups), `product_innovation` 82, `other` 80,
`labor_human_capital` 77, `mergers_acquisitions` 64, `partnerships_business_dev` 45,
`legal_regulatory` 42, `leadership_governance` 42, `unrelated` 31 (not about the economy),
`earnings_performance` 20, `capital_shareholder_returns` 2. The notes ask for reputational harm (10),
competitive actions (about 11), sector news, whole-market / economy news and investor moves. This
re-scoped Work items 15 and 17 into Work item 18 (decisions D11–D24, `PLAN.md` Work item 18).

**Scope implied by the tags** (`news_nlp.triage.gold_scope`; six scopes, D22): single company 458,
several companies 135, **`investing_general` 78** (the owner's `other` tag: investing or
market-advice content about neither a company nor a sector), unrelated 31, sector 9, market /
economy 9. Sector and market / economy are too rare in this set to be measured; the ~300 gold
bodies of P2 must be drawn to include them.

**Laya, zero-shot (T-145, cancelled).** Yes/no questions to `convaiinnovations/laya` (English
checkpoint, revision `55cf4c4e`; title + text, of which it reads about the first 480 tokens): it
answered "yes" to almost every question (sector 604 of 720, market / economy 496, unrelated 467), so
its combined scope matched the owner's on 39 of 720 (5.4%, 95% 4.0-7.3%). As a ranking it carried
signal for "several companies" and "one company" (AUC 0.82) but not for "unrelated" (0.24, inverted).
The owner dropped the approach.

**Rules against the 720 tags** (`scripts/size_triage_2026_10_28.py gold`; `news_nlp.triage`): the
companies a body names are resolved from names, short aliases (the first word of a name: "Meta"),
explicit ticker forms and the stored NER `ORG` spans; **each mention counts once** (the larger of the
text matches and the NER spans of one pair of the body, not their sum, and not once per pair); a
company only ever named as an opinion's source ("Goldman Sachs analysts") is left out; a body is
multi-company unless one company holds a strict majority of the subject mentions (*single +
mentions*, D23). Each step, cumulative, against the six scopes:

| step | agreement with the owner's scope | 95% | multi recall | multi precision |
|---|---|---|---|---|
| every named company | 0.689 | 0.654-0.722 | 109/135 | 109/206 |
| + subjects (named twice or in the title) | 0.696 | 0.661-0.728 | 95/135 | 95/147 |
| + boilerplate removed, analysts' banks not counted | 0.694 | 0.660-0.727 | 95/135 | 95/147 |
| + round-up structure (headline / URL slug, 3+ companies in the title, 5+ ticker forms) | 0.697 | 0.663-0.730 | 102/135 | 102/162 |

**The figures read the title and the first 6,000 characters of each body, what the owner read (review 7 of #115; before, the whole body: 0.699, 101/135, 101/158); the dominance sweep below was run on the whole body and is not repeated, the 0.5 cut gives 0.697.**

**Two corrections to the first version of this table.** (1) It reported 0.79: that scored the
owner's `other` tag as "single" (five scopes); with `investing_general` as a scope (D22) the rules
miss 77 of those 78 articles (they name a company). (2) The first counts added the NER spans to the
text matches, so one passing mention counted twice and the "subjects" step did nothing
(0.703 -> 0.700); counted once, the step prunes false multi-company calls (multi precision 108/203 ->
94/143) and the final figures are the ones above. Outside `investing_general` the rules agree on 503
of 642 (0.78).

**The dominance cut, re-swept after the correction** (final step; subjects named at least twice /
three times): 0.4 -> 0.714 / 0.703, **0.5 -> 0.699 / 0.697**, 0.6 -> 0.678 / 0.667, 0.7 -> 0.642 /
0.643; multi recall 93, **101**, 110, 114 of 135 at 2 mentions. The cut stays at 0.5, the strict
majority of D23: the differences are inside the interval (about +-3.3 points) and the 720 are the
development set, so it is not re-tuned on them; a lower cut buys precision (fewer false multi) at the
price of recall (the bodies that go to the per-company deep dive). These 720 are a **development
set**: every figure is optimistic; the held-out test is P2's gold. The remaining errors are the
`investing_general` articles, borderline articles (a company and its rivals, "tariffs hit Ford and
GM"), 23 `unrelated` articles the rules call single-company, and the sector / market cases (none of
the 9 + 9 found). Rules that read names and counts cannot find these; that is what the trained
triage (P3) is for. (It was built: the scope classifier selected on 2026-11-11 reads the title and the first 800 characters and reaches 83.6% on the held-out 200 against 67.2% for these rules; `docs/scope_training.md`.)

**Sizing the corpus** (`scripts/size_triage_2026_10_28.py`, read-only, about 47 minutes; mentions
counted once, the corrected rules above; **measured with the matcher at 786766f: PR #155 moves the
scope shares by about +4 points single and -3 single + mentions, see
`docs/matcher_measurements_2026-11-09.md`, not recomputed**): 98,462 `ok` (URL, ticker) pairs hold **81,961 unique
bodies**; 85.5% of bodies sit under one pair, 10.7% under two, 3.7% under three or more; 1,904 pairs
are share-class twins. Subject companies per body (named twice or in the title): none 4.4%, one 52.4%,
two 21.5%, three to five 15.6%, six to ten 4.7%, more than ten 1.5%.

| rule scope | bodies | share | pairs | tagged ticker among the subjects |
|---|---|---|---|---|
| single company | 41,773 | 51.0% | 42,970 | 94.9% |
| several, one dominant (*single + mentions*, D23) | 18,796 | 22.9% | 21,958 | 96.7% |
| round-up | 9,988 | 12.2% | 19,442 | 95.3% |
| several companies | 7,887 | 9.6% | 10,243 | 93.7% |
| names no company, sector keywords | 1,295 | 1.6% | 1,496 | - |
| names no company, no keyword | 1,293 | 1.6% | 1,386 | - |
| names no company, market keywords | 515 | 0.6% | 541 | - |
| names no company, economy keywords | 414 | 0.5% | 426 | - |

Round-ups plus several companies are 17,875 bodies (21.8%) and hold 29,685 pairs (30.1%): the content
one category per pair cannot represent; another 22.9% of bodies name a second company without being
about it. More than half of the multi-company bodies (20,643 of 36,671, 56%) share a GICS sub-industry
(top: hardware 2,929, semiconductors 1,525, broadline retail 1,116, diversified banks 1,055), a start
for sector detection. **The sector / market / economy rows (2,224 bodies, 2.7%) are a lower bound from
a detector with no measured recall**: it finds only bodies that name no company, and on the owner's tags v1
it finds none of the 9 sector and 9 market / economy articles, because they name companies. They size
the detector, not the news; P5 must not be sized from them. The figures carry the alias gap (a company
known by another name is missed unless the NER span or a ticker form catches it), which undercounts
companies and so pushes bodies toward "single" and "names no company".


**Gold v2 file (T-146b, PR #117, 2026-10-29).** `scripts/retag_gold_2026_10_29.py export` writes `data/category_gold_v2_review.csv` (git-ignored): the 720 articles with the owner's v1 label and note shown, EMPTY scope / companies / D15 label per company / mentions / sub-industry columns, and the LLM's label only in the key file; `check` validates the returned file against the SOURCE universe. The owner tags it after labelling spec v2 (T-146a); the rules' agreement is re-measured on it then. All the figures above are against tags v1.

## Sampling: low-confidence + targeted + representative strata

Each run draws `--sample-size` rows per stage as a **disjoint,
priority-ordered stack of strata** (`src/news_nlp/eval/sampling.py`), each
excluding every earlier-priority stratum's already-drawn ids:

1. **`low_conf`** (`--low-conf-frac` of `--sample-size`, default 0.2) — the
   *least-confident* stored rows, deterministically, so every run re-checks
   the true worst case: lowest `article_sentiment.score`; category picks
   within ±0.1 of `CATEGORY_CONFIDENCE_THRESHOLD` (0.6 as of the 2026-09-09
   calibration — see "Follow-up: hierarchical category classification"
   in `docs/legacy/evaluation-v1-stages.md`) or labelled `other`;
   lowest per-article `MIN(article_entities.score)`; for `c_summary` (no
   score), articles whose sentiment/NER inputs were themselves low-confidence.
   **Diagnostic-only**: deliberately biased toward hard cases, so it's
   excluded from every headline/population-estimate metric (see "Statistical
   methodology" below) — reported only as a `*_low_conf` per-stratum split.
2. **`target_<x>`** (`--target-frac` of the post-`low_conf` budget, default
   0.6; sentiment and category only) — rows clearing a threshold on a *raw
   per-class score*, regardless of which class won the argmax:
   - **sentiment**: `target_negative` (weight 0.6, `negative >= 0.35`),
     `target_positive` (0.2, `positive >= 0.35`), `target_neutral` (0.2,
     `neutral >= 0.35`). `negative` is weighted highest — it's the headline
     class.
   - **category**: one stratum per the 6 worst 2026-09-08-baseline per-slug
     accuracies plus `legal_regulatory`, all thresholded at `>= 0.35` (raised
     from an original 0.2 once the hierarchical category classifier changed
     the leaf-score baseline from ~0.111 to ~0.333 — see the "hierarchical
     category classification" follow-up in `docs/legacy/evaluation-v1-stages.md`), weights summing to 1.0:
     `partnerships_business_dev` 0.20, `labor_human_capital` 0.18,
     `leadership_governance` 0.16, `mergers_acquisitions` 0.14,
     `capital_shareholder_returns` 0.13, `product_innovation` 0.12,
     `legal_regulatory` 0.07.
   - **c_summary** has no discrete classes, so its "targets" instead
     partition on `num_chunks` (a TRUE partition — every row has exactly
     one, so c_summary draws no `representative` bucket at all): `1` (weight
     0.10), `2` (0.30), `>= 3` (0.60) — weighted toward the rare multi-chunk
     tail (~4.6% of the corpus) since `mean_coverage` is the weakest
     c_summary metric and multi-chunk articles pass through more
     hierarchical-reduce steps.
   - **ner** has none: its type imbalance (ORG 55% / PER 25% / LOC 20% of
     spans) is far milder than sentiment's/category's article-level
     imbalance, and there's no secondary per-entity score to target.
   - **Why these thresholds are well below each stage's winning bar** (0.5
     for sentiment's 3-way softmax; `CATEGORY_CONFIDENCE_THRESHOLD` 0.6 for
     category): a distribution over mutually exclusive classes can have at
     most one class exceed 0.5, so a threshold at or above that would
     mathematically exclude every false-negative candidate for that class —
     silently defeating the whole point (catching rows the model *almost*
     called this class but didn't, not just rows it did call this class).
3. **`representative`** (the remainder) — a `--seed`-reproducible uniform
   draw from whatever's left. The only bucket that's a plain, unweighted
   random sample of the population (`low_conf` and `target_<x>` are each
   deliberately non-representative by construction).

Metrics are reported both as a population estimate (`headline_metric`,
Horvitz-Thompson-reweighted across every non-`low_conf` stratum — see below)
and per-stratum (`*_low_conf`, `*_target_negative`, `*_representative`, …),
so you see both the population number and where it's weak.

**Pass `--seed` when comparing two runs.** With no seed, `target_<x>` and
`representative` are fresh OS-entropy draws every time, so two unseeded runs
differ in both *which* articles were judged and (per the LLM) how they were
judged — any delta between them mixes real signal with resampling noise, and
you can't tell how much of each. `low_conf` is already deterministic (same
rows every run, seed or not) since it's a fixed `ORDER BY ... LIMIT`; a shared
`--seed` makes the rest deterministic too, so a before/after comparison (e.g.
across a pipeline or eval-harness change) isolates the change's effect instead
of conflating it with which rows happened to get drawn. See "How much of the
sentiment run-to-run swing is noise?" (`docs/legacy/evaluation-v1-stages.md`) for a worked example of how large
that conflation can be.

**Migration note**: a run against pre-redesign code isn't seed-reproducible
against this code even with the same seed — drawing `target_<x>` strata
inserts extra RNG draws before the final `representative` shuffle. `eval_run`
rows from before this redesign have `strata_json = '{}'`.

## Statistical methodology: Horvitz–Thompson reweighting

Naively pooling a deliberately-oversampled `target_<x>` stratum with
`representative` would be just as statistically invalid as the old design's
`low_conf`+`random` pooling (see the "run-to-run swing" section in `docs/legacy/evaluation-v1-stages.md` for why
that mattered). The fix is a standard stratified-sampling ratio estimator.

For any population total (e.g. total true negatives across the whole
corpus), each non-excluded stratum `h` contributes `(N_h / n_h') *
(count within h)`, where `N_h` is that stratum's population size (computed
from the exact SQL result the sample was drawn from, never a second
independently-written count) and `n_h'` is the count of rows *actually
judged* in that stratum (post-parse-failure — a deliberate MCAR assumption:
a judge JSON-parse failure is treated as independent of the row's true
label, so survivors of stratum `h` are still ~a uniform sample of size `n_h'`
from `N_h`; this is an accepted, unverified risk, not solved). A ratio metric
(recall, precision, accuracy, a mean) is the ratio of two such weighted sums.
`low_conf` is always excluded — it's diagnostic by design, never part of a
population estimate.

Implementation: `_ht_sum` / `_ht_ratio` / `_prf_ht` / `_macro_f1_ht` in
`src/news_nlp/eval/metrics.py`. Worked example (also a unit test,
`test_ht_sum_and_ht_ratio_hand_computed_example`): stratum A `N=100,n=4`
(TP=2, FN=1), stratum B `N=900,n=6` (TP=1, FN=0):

```
TP_hat = (100/4)*2 + (900/6)*1 = 200
FN_hat = (100/4)*1 + (900/6)*0 = 25
recall_hat = 200 / 225 = 0.8889          (vs. naive pooled 3/4 = 0.75)
```

`ner` needs no such reweighting: with only `representative` as a non-excluded
stratum, `N_h/n_h` is a constant that cancels identically in any ratio, so
`aggregate_ner` is mathematically unchanged from the old pooled computation —
confirmed by leaving its code untouched rather than adding a no-op HT call.

## Sample-size floor

`n_stratum(class c) = n_true_target(c) / purity_target(c)`, where
`n_true_target` is the Wilson-CI true-example count needed for a target
recall/precision confidence interval (±0.05 needs ~362 true examples at
sentiment's ~55-62% base recall; category's worst slugs need ~200-370), and
`purity_target` is the fraction of a threshold stratum a judge actually
confirms as class `c`. Below are *planning* estimates from population share +
baseline precision, not measured purity — re-solve with a pilot run's actual
per-stratum agreement before locking in a permanent default (a pilot's
`agreement_rate_target_negative` etc., or the `strata_json` blob, gives you
the real `n`/`N` to recompute this from).

- **Sentiment**: `negative >= 0.35` population is 213,604 (46.5% of the
  corpus) vs. argmax-`negative`'s 41.1% baseline precision of 19.3% — purity
  should land a bit below that (looser threshold, more true negatives).
  Planning range 15-19% → `n_stratum ≈ 1,905-2,413` — a real but modest
  ~30-45% reduction vs. the ~3,289 an unstratified design would need (the
  threshold is deliberately loose to keep catching FN candidates, which
  dilutes purity by design).
- **Category**: e.g. `partnerships_business_dev >= 0.2` population is 24,390
  (5.31%, vs. 2.95% true share) → purity roughly 6-12% → `n_stratum ≈
  1,667-6,167`. Because the 7 target strata **share one weighted budget**
  rather than running independently, hitting the single worst slug's own
  ±0.05 target inside one regular run can still need a total `--sample-size`
  in the same range the unstratified design would — an honest result, not
  oversold. **Predates the hierarchical category classifier** (the follow-up
  in `docs/legacy/evaluation-v1-stages.md`) and its `>= 0.2` → `>= 0.35` threshold change — both the population
  count and the threshold in this bullet are stale; re-derive against the new
  classifier's actual score distribution once enough articles have been
  reclassified under it, rather than trusting this number as-is.
- **Recommendation** (two-tier, not one asserted number):
  1. Regular regression-tracked runs: `--sample-size` floor **~1,800-2,200**
     for sentiment, **~2,500-3,000** for category (targets a looser
     ±0.07-0.10 CI — still tight enough to catch a real move against the
     existing 0.05 `--regression-tolerance`).
  2. Occasional deep-dive runs (manual): 5,000+, or a temporary
     single-slug-weighted `_CATEGORY_TARGET_WEIGHTS` override, when chasing a
     specific rare-class regression.
  3. Before locking in either number: run a pilot at `--sample-size 800`
     under this design, read the actual purity per `target_<x>` stratum from
     that run's metrics, and re-solve the formula above with measured (not
     estimated) purity.
  4. `ner` and `c_summary` need no floor increase — `ner`'s stratification is
     unchanged, and c_summary's `target_chunks_ge3` (4.6% of the corpus)
     already gets outsized attention (weight 0.6) at today's sizes.
  5. `sector_summary` has no floor at all — `--sample-size` is ignored for
     it, every run judges the full 3,628-row population (2026-09-14
     follow-up in `docs/legacy/evaluation-v1-stages.md`).

## Running it

Needs an OpenAI-compatible chat endpoint (DeepSeek in practice), same as
`portfolio-financial-analysis`. Put in `.env` (see `.env.example`):

```
LLM_API_KEY=...
LLM_MODEL=deepseek-chat
LLM_URL=https://api.deepseek.com
# MLFLOW_TRACKING_URI=./mlruns
```

```bash
uv sync --group eval

# all four stages, 80 rows each, into ./mlruns
uv run cli/news_nlp_eval.py --stage all --sample-size 80

# one stage, reproducible sample
uv run cli/news_nlp_eval.py --stage sentiment --seed 1 --sample-size 40

# a regression-tracked run at the recommended sentiment sample-size floor
uv run cli/news_nlp_eval.py --stage sentiment --sample-size 2000 --seed 1

# label the run in MLflow's UI (2026-09-15) -- useful when comparing named
# candidates (v4, v5, ...) against a scratch --results-db, since they'd
# otherwise all land in news_nlp_eval/sentiment with random auto-generated
# names. NOT cosmetic (correction, 2026-09-19): whenever --candidate-model
# is unset, --run-name IS the resolved `experiment` (TASKS.md T-090, SPEC.md
# FR-014) -- it drives judge-verdict reuse (FR-015) and --check-regression's
# "which prior run to compare against" (see the invocation a few sections up
# and the note near --check-regression below, both correct already; this
# comment was the stale one)
uv run cli/news_nlp_eval.py --stage sentiment --sample-size 2000 --seed 1 \
    --run-name v5-sec-bert-base --results-db /path/to/scratch.db --source-db /path/to/source.db

# score a candidate model/checkpoint live -- no production run needed first
# (2026-09-18, TASKS.md T-089): scores --candidate-prescore-size SOURCE
# articles with the candidate into a throwaway scratch RESULTS file via
# that stage's own FTI Inference class, then samples/judges from it. Needs
# exactly one --stage; --candidate-revision is required (a Hub commit SHA,
# or any placeholder for a local checkpoint path).
uv run cli/news_nlp_eval.py --stage sentiment --sample-size 2000 --seed 1 \
    --candidate-model gamug/FinBERT-financial-news --candidate-revision abc1234

# fail (exit 1) if a headline metric dropped > 0.05 vs the previous MLflow run
uv run cli/news_nlp_eval.py --stage all --check-regression

uv run mlflow ui            # browse runs at http://127.0.0.1:5000
```

Flags: `--stage` (repeatable; `all` = every stage), `--sample-size`,
`--low-conf-frac` (default 0.2), `--target-frac` (default 0.6, no-op for
`ner`), `--seed`, `--max-workers` (concurrent judge calls, default 4),
`--source-db` / `--results-db` (override `$SOURCE_DATABASE_URL` /
`$DATABASE_URL`), `--mlflow-uri`, `--run-name` (label the MLflow run instead
of its auto-generated name; cosmetic, applied to every stage in the
invocation), `--check-regression`, `--regression-tolerance` (default 0.05),
`--candidate-model` / `--candidate-revision` / `--candidate-prescore-size`
(2026-09-18, TASKS.md T-089 — live candidate-model scoring, see the example
above; `--candidate-model` requires exactly one `--stage` and a non-empty
`--candidate-revision`; `--candidate-prescore-size` defaults to
`--sample-size`).

## Where results go

- **MLflow** — one run per `(stage, invocation)` in experiment
  `news_nlp_eval/<stage>` (an MLflow "experiment," unrelated to
  `news_nlp.eval`'s own `experiment` concept below — the two share a name
  by coincidence, not design). Params (judge model/url, sample size,
  per-stratum `n_<bucket>` counts, seed, git SHA, **`experiment`**),
  metrics, a `judgements.json` artifact (every sampled row: the model
  prediction + the parsed judge verdict) and the `judge_prompt.md` used.
  `experiment` is also set as an MLflow **tag** (`tracking.log_to_mlflow`,
  2026-09-18) so `--check-regression`'s previous-run lookup can filter by
  it — see the follow-up below.
- **RESULTS store** — `eval_run` (one row per invocation: stage, timestamps,
  `low_conf_n`/`random_n` — the latter now "every non-`low_conf` stratum
  combined" — `strata_json` (`{bucket: {"population": N_h, "n": n_h}}`, the
  Horvitz-Thompson bookkeeping; `'{}'` for pre-redesign rows), judge model,
  `code_version`, `mlflow_run_id`, the metrics blob, `status`, **`experiment`**
  — `'base'` for pre-2026-09-18 rows) and, as of the 2026-09-18 follow-up
  below, `eval_inference`/`eval_verdict` (one row per sampled article each)
  plus `eval_confusion` (sentiment/category only — one row per `(run_id,
  task, experiment, true_label, predicted_label)` cell observed). DDL in
  `news_nlp.schema`; `init_schema` creates them. `GET /eval/latest` on the
  FastAPI service returns the newest `eval_run` per `(stage, experiment)` —
  a `--candidate-model` run no longer displaces production's own "latest"
  row for a stage.

**2026-09-18 follow-up (PLAN.md Work item 10 step 4, TASKS.md T-090, SPEC.md
FR-014)**: `eval_judgement` (one row per sampled article, holding both the
model's prediction and the judge's verdict together) is superseded by two
tables — `eval_inference` (the sampled prediction: `article_id`, `task`,
`experiment`, `bucket`, `prediction_json`) and `eval_verdict` (the judge's
call on one `eval_inference` row: same `article_id`/`task`/`experiment`
plus `verdict_json`/`correct`/`severity`/`rationale`). `task` mirrors
`eval_run.stage`; `experiment` is a free-form label for which model
produced the inference (`base` for the pinned production model, or
whatever `--candidate-model`/`--run-name` resolves to — see
`news_nlp.eval.runner._run_stage`'s resolution order) — this is what lets
a candidate model's judged data (scored live via `news_nlp.eval.candidate`,
TASKS.md T-089) coexist with production's own judged data for the same
articles, instead of needing the scratch-database-per-candidate workaround
the `resample_sentiment_v{3,4,5}` scripts used. `eval_judgement`'s DDL and
historical rows are untouched (additive split, not a migration) — new
eval runs simply stop writing there.

**Later the same day**: the limitation this follow-up originally disclosed
here — neither `GET /eval/latest` nor `--check-regression`'s previous-run
lookup was `experiment`-aware — is now fixed. `eval_run` itself gained an
`experiment` column (`'base'` default, additive migration via
`ensure_columns`, same pattern as `strata_json`'s own addition);
`queries.latest_eval_runs` now groups by `(stage, experiment)` instead of
`stage` alone; `tracking.log_to_mlflow` sets `experiment` as both an
MLflow tag and a logged param; `tracking.previous_headline` (and therefore
`regression.check_regression`) now filters its MLflow `search_runs` call
by `tags.experiment`, so a candidate-model run's headline metric is only
ever compared against a *prior run of that same candidate* — never
production's own number, and never another candidate's.

## CI

`ci.yml` installs `--group eval` and runs the hermetic `tests/news_nlp/
test_eval_*.py` (the LLM and MLflow are mocked / a tmp file store). The **real**
eval is `.github/workflows/eval.yml` — `workflow_dispatch` + a weekly schedule,
with `LLM_*` repo secrets and `--check-regression`. It needs the actual crawl /
results databases, which live on the working machine, so it is written as a
self-hosted-runner template; the reliable path is `uv run cli/news_nlp_eval.py`
on a local cron with `MLFLOW_TRACKING_URI` pointed at a persistent store.
The scheduled gate (`--check-regression`, PLAN Work item 2) is out of the MVP (owner, 2026-11-10).

## Prompts

`src/news_nlp/eval/prompts/{sentiment,category,ner,c_summary}.md` — judge SOPs,
each ending with a strict "reply with ONLY a raw JSON object" instruction.
`category.md` embeds the v1 10-label taxonomy; keep it in sync with
`news_nlp.taxonomy` / `docs/legacy/category-taxonomy-v1.md` if the labels change (the D15 taxonomy of the stepped pipeline is
`docs/category-taxonomy.md`).

*Moved verbatim to `docs/legacy/evaluation-v1-stages.md` (v1 stage deprecated by the owner on 2026-11-12, or Work item 15, superseded by Work item 18): "Calibration judge and sampler (T-141)".*

## Experiments: JSON-driven, single-command runs

(2026-09-18, `PLAN.md` Work item 11, `SPEC.md` FR-017, `TASKS.md`
T-096-T-103.) Everything above this section describes `cli/news_nlp_eval.py`
directly evaluating whatever's already in the RESULTS store (or, with
`--candidate-model`, live-scoring one candidate). Reproducing a full
sentiment-candidate comparison (Work item 9's v2→v5 sequence) still meant
hand-chaining several independent commands — dataset prep, `train_sentiment.py`,
an offline test-set/idiom-probe check, a downstream `--candidate-model` eval,
sometimes a Hub publish — several of which mutated shared DB tables in place
and needed a manual restore step afterward
(`scripts/resample_sentiment_v4_2026_09_15.py` +
`scripts/restore_sentiment_after_v4_eval_2026_09_15.py`). Nothing declared an
experiment's full configuration in one reproducible, validated place before
it ran.

**`src/experiment.py`'s `ExperimentSpec`** (a pydantic schema) fixes that: one
JSON file fully describes an experiment for any of the four ML stages
(sentiment/NER/category/`c_summary` — `sector_summary` excluded, no
Feature/Train/Inference shape) —

```json
{
  "name": "sentiment_v4_class_weighted",
  "stage": "sentiment",
  "pretrain": {
    "enabled": true,
    "base_model": "ProsusAI/finbert",
    "split": {"split_seed": 42, "test_frac": 0.1, "val_frac": 0.1},
    "hyperparameters": {"weighted": true}
  },
  "eval": {"sample_size": 2000, "seed": 1, "candidate_prescore_size": 2500}
}
```

— whether it trains a new checkpoint (`pretrain`, nested under
`PretrainSpec`/`TrainTestSplitSpec`; `hyperparameters` is validated by name
against that stage's real `TrainConfig` dataclass fields, e.g.
`SentimentTrainConfig`'s `weighted`), how the result is evaluated
(`eval`, a direct passthrough subset of `EvalSettings`' own non-secret
fields — never `LLM_API_KEY`/`LLM_MODEL`/`LLM_URL`/`MLFLOW_TRACKING_URI`,
which stay `.env`/CLI-sourced, never a git-tracked file), and whether to
publish the result (`publish` — records the intent only; a Hub push still
needs the same explicit, separate confirmation the existing
`scripts/publish_*.py` pattern already requires, unchanged by this schema).
Every field rejects unknowns outright (`extra="forbid"`) — a reproducibility
guarantee, not a security mechanism: an unpinned `base_model`, `pretrain`
requested for category/`c_summary` (no trainable checkpoint, FR-011), an
unknown `hyperparameters` key, `publish.enabled` without `pretrain.enabled`
+ `repo_id`, or `eval.candidate_model` set alongside `pretrain.enabled` (it
auto-resolves downstream instead) all fail validation before any train/eval
work starts, with a message naming exactly what's wrong.

```bash
uv run cli/run_experiment.py --config experiments/sentiment_v4_class_weighted.json
uv run cli/run_experiment.py --config experiments/ner_production.json --source-db urls.db
```

is the one command: trains (if `pretrain.enabled`) via a lazily-imported
stage→`(TrainConfig, Trainer)` resolution, auto-resolves
`candidate_model`/`candidate_revision` to the fresh local checkpoint on
success (`revision="local"`, the established
`resample_sentiment_v4_2026_09_15.py` convention), evaluates via
`run_eval` — **reused verbatim, not reimplemented** — and writes a
git-tracked `experiments/results/<name>.result.json` (the resolved spec +
`code_version` + the training step's own metrics + `run_eval`'s own result
dict). "Exit 1 if regressed" needed no code of its own:
`run_experiment`/the CLI inherit `run_eval`'s existing `SystemExit(1)`
(uncaught, same behavior `--check-regression` already has) the moment a
stage's headline metric drops past tolerance — before a result file can
even be written for that run.

`experiments/` holds a backfilled spec for every real historical experiment
that has a runnable equivalent today (`TASKS.md` T-101): sentiment's five
candidates (v2/v3/v4/v5 + the un-fine-tuned base-FinBERT comparison arm)
and one production-config eval-only spec each for NER/category/`c_summary`.
**`experiments/README.md`** is the source of truth for the full account —
this doc doesn't duplicate it, only summarizes the three disclosed gaps it
names:

1. The removed "title-only" sentiment aggregation arms (PR #43 and its
   fine-tuned variant, rejected in the 2026-09-13 follow-up, now in `docs/legacy/evaluation-v1-stages.md`) have no
   surviving code path — not backfillable.
2. Category's confidence-threshold calibration and `c_summary`'s
   generation output-length-budget test (both above) tuned an
   *inference-time* constant, a different axis than this schema's
   pretrain/dataset/split/sample-count shape — out of scope for v1.
3. (Found while backfilling, not anticipated when this schema was
   designed) `sentiment_v2_chunklevel_finetuned.json` and
   `_v3_downsampled.json` are deliberately content-identical except
   `name` — v2 and v3 used "the same procedure/hyperparameters," differing
   only in whether a data file happens to exist on disk at train time,
   which `ExperimentSpec` has no field for. Running either spec **today**
   reproduces v3's behavior, not v2's original pre-rebalance pool.

### Follow-up (2026-10-30): Spanish reading aid for the gold v2 file (Work item 18, T-146b)

Not a labelling run: `deepseek-chat` (`LLM_*`) translated the title and text of the 720 bodies of
`data/category_gold_v2_review.csv` into Spanish for the owner, who finds the English finance jargon
hard to read (`news_nlp.translate`, `scripts/retag_gold_2026_10_29.py translate`). It asks for no
label and is outside the D18 labelling budget, but it is paid, so it is logged here. **Spend:
720 articles = 3 pilot calls + 717 calls, 0 failures; 558,469 prompt + 641,963 completion tokens
for the 717 plus 2,142 + 2,378 for the pilot; about 1 USD at list prices (estimate).** These count the successful replies only: the run kept no count of retried calls (later runs count every paid call, retries and rejected replies included), so they are a lower bound. Spanish
text was 1.05-1.19 times the English length (nothing cut), every original column identical to the
backup, the key file untouched. A later export of another tagging file (the held-out gold T-146e,
about 300 bodies) is estimated at 0.8k prompt + 0.9k completion tokens per article and needs
`--yes`; one run may buy at most `--max-new` (800) articles.

### Follow-up (2026-11-05): Spanish reading aid for 44 extra sector bodies (Work item 18, T-146c)

Same engine and purpose as above, for `data/sector_extra_to_tag.csv`: 44 bodies drawn by free rules
(`scripts/draw_sector_extra_2026_11_05.py`) so that the sub-industry gate has more than the 16 sector
bodies of the 240. Outside the D18 labelling budget, paid, logged here. **Spend: 44 articles = 44
paid calls, 0 failures; 37,940 prompt + 45,359 completion tokens (retries and rejected replies
included); well under 1 USD at list prices (estimate).**

### Follow-up (2026-11-07): how the attribution model is evaluated (Work item 18, T-148b3)

Three readings, in rising order of weight; only the last decides (PLAN D27, D32).

1. **Pairs on `val`** (`scripts/train_attribution_2026_11_08.py`; 270 bodies, 7,644 pairs, never
   trained on): the right company against one wrong one, ranked right 90.1% of the time against 63.7%
   for the mention-and-recency baseline (the previous sentence mentions the candidate, read by the
   resolver; 59.0% with the legal name only). Also by period, for the first sentence of a paragraph,
   and for the sentences that name no universe company (94.4% against 86.6%), which are the only
   ones scored at run time and carry no `[COMPANY]` mask.
2. **The run-time task on `val`** (the 1,153 sentences that name no company; every company of the body
   a candidate; threshold 0.5; a tie for the best score is nobody): the model gives the sentence to
   the right company 86.4% of the time, a wrong one 11.8%, nobody 1.8%; the baseline (the company
   mentioned last in the previous sentence) 70.8%, 6.4% and 22.8%. The labels are the parser's
   (88% of the `lead` ones right in the owner's check), so part of the "wrong" is label noise.
3. **The owner's faithfulness reading** of the 30 narrative bodies (D32). Abstention is not in `val`
   (every sentence there has a right company); `train_attribution_2026_11_08.py --abstention DIR`
   measures it on the 30 bodies: of the 421 sentences that name no company the model gives 145 (34%)
   to nobody and 276 to one company, the baseline 19 to nobody, 281 to one company and 121 to
   several. A seeded sample read by the developer (not the owner's reading; 5 and 5, so only a
   direction): of the 5 given to nobody, 4 are right (the company is not in the universe: Nextracker,
   Macy's and Gap, Staples, Nikola) and 1 is wrong ("JPMorgan said in the note ...": the resolver
   does not read the one-word "JPMorgan", a name-matcher gap, T-148b7); of the 5 given to a
   company by the 0.5 rule, none is right (a venture investment given to Comcast because the previous
   sentence is Comcast's disclosure, a market index to EQR, Motorola's president's quote to Apple,
   Boeing's next chief to Amazon, a start-up's partners to BSX). The model was trained only on
   sentences that belong to one of the body's companies, so it has not learned to answer "none of
   these"; a sentence about a company outside the universe is given to the nearest one when the score
   reaches 0.5.
 The 276 sentences given by the rule have these best scores: 38 from 0.5 to 0.6, 79 from 0.6 to 0.8,
   159 from 0.8 to 1 (and 102 of the 276 name, in the previous sentence, the company chosen); the five
   wrong examples scored 0.999, 0.988, 0.997, 0.893 and 0.505. They are confident, not borderline: a
   higher threshold would not remove them. The threshold is the owner's to decide after reading the texts; none is proposed.

A `val` body is left out as a copy when 80% of a sample of its 8-word shingles are in one train
body (none was); an earlier key on the start and the end of a body, and then 30% of all the train
shingles together, matched footers, a shared closing paragraph and stories of one template, not
copies. `--eval-only` keeps `report_at_training` in `metrics.json` and adds an `eval` block tied to
the hash of the model files it scored.

### Follow-up (2026-11-08): the `units` extractor and the guards on the owner's marks (Work item 18, T-148b4)

`units` reads the body without its footer and cut at lines, bullets, semicolons and clauses, so a
list with one line per company is not one sentence that every company receives whole, and a
disclosure block does not attach to the companies whose tickers are in it. The owner's reading
(2026-11-07, 402 rows): 85.4% faithful [76.5, 92.2] on 185 texts, "not decided" against the 80% bar;
+24.3 over the baseline (`better`), +8.6 over the model (`not decided`). Figures and the failures
left (first-word names) are in PLAN D27 and TASKS T-148b4.

Two things about reading that figure. First, `units` is not asked for a text of a company the body
names only in its footer, so its bar share is on fewer companies than the baseline's; the report
prints a second reading with those companies counted as unfaithful (`draw_faithfulness --footer-only`
writes them with the revision of the footer rules; the report skips the reading, and says so, when
the rules changed). Second, the marks are on texts: `draw_faithfulness --stale` (free, read-only, with
`--attribution-model`) re-runs the extractors over the 30 bodies and lists the marked rows the code
no longer writes (at the time of writing, 2026-11-08, the Apple and Amazon rows of body 122960), the new texts and the rows
written by another revision of the code (the text is the same: the mark stands); the report quotes
the last check of each extractor in one line (a run of some extractors keeps the last check of the
others), says which extractors of the marks were never checked, and warns when a check was made with
other code than the current one (the code part of a revision needs no model). The exit code is a sum:
1 if a marked row is stale, 2 if an extractor of the marks was neither run nor left out with
`--only NAMES`; 0 when everything asked for was checked and is clean.

### Follow-up (2026-11-08): `units2` and `units3`, and what their bar leaves out (Work item 18, T-148b5, T-148b6)

`units2` reads `units` with the guarded resolver (a short form of a company's name counts only when
the full name or a ticker form is in the body) and the lines headed by names ("NAMES - text") given
to those names; `units3` also gives a piece of such a line to the universe companies it names. The
owner's readings (2026-11-07): `units2` 84.4% [72.5, 97.2] and `units3` 91.6% [84.7, 97.8] on D32's
bar, `units` 85.4%. Two cautions before those figures are used. First, they leave out the companies
the extractor is not asked about: on the 30 bodies 36 of the 190 companies named get no text from
`units3` and are counted nowhere (32 the guard rejects, 5 only in the footer, minus 1); with them
counted as unfaithful `units3` is 74.2% [66.0, 81.5], `units2` 68.4% [57.9, 80.3] and `units` 83.2%
[72.7, 91.0] (`report_wi18 faithfulness` prints both readings; the reading of the 36 by the owner,
`faithfulness_exempt_to_read.csv`, gives the bar on real companies). Second, the guard and `units3`
were designed on the failures of these same 30 bodies, so the figures are development figures; a
confirmation on bodies not used before is proposed in TASKS T-148b8.

**The owner's reading of the 37 (2026-11-08)** settles the denominator: 23 of them are not companies
("Bloom" the person, "Equity markets", "Union Square"), 14 are. With the 23 out of every extractor's
share, the texts the others wrote about them included, all extractors are on the same 167
companies (190 named, minus the 23 read and found not to be companies; the other 167 were not read one
by one): baseline 68.9% [53.6, 82.1], `model` 86.2% [71.7, 97.5], `units` **94.0% [87.0, 99.2], pass**,
`units2` 77.8% [64.1, 92.9], `units3` 84.4% [76.4, 93.3] (not decided); `units3` minus `units` -9.6
[-16.5, -1.6] (not better). **What the 94.0% leaves out:** the 23
non-companies take 22 of `units`' 185 texts out of its share, and 21 of them were unfaithful (the extractor wrote
a text for "Home Depot" about "Home smart speakers", for "Franklin Resources" about a person); they are the
extractor's own errors, not an artefact of the bar, and a cut with `units` ships them (the report prints them
apart: "N of the M texts it wrote are for a company that is not one"). On every company named `units` is
83.2% [72.7, 91.0], not decided. The guard (`units3`) writes none of them (0 of 148) and has 7 unfaithful texts
of 148 against 27 of 185 for `units`, but leaves 19 real companies without a text against 4 for `units`, mostly
lines headed by a short form. The choice between the two is the owner's trade (PLAN D34 (e), T-148b10). The decision on
the candidate for the cut stays the owner's (PLAN D34 (e), TASKS T-148b8, T-148b9).

**Spend (2026-11-08, approved by the owner): Spanish for `faithfulness_exempt_to_read.csv`.** The title and the
line that names each of the 37 exempt companies (no article body), the approved engine, `draw_faithfulness
--exempt --yes`: 37 articles = 37 paid calls, 0 failures; 8,848 prompt and 4,910 completion tokens (retries and
rejected replies included). The file is merged, never replaced: a spreadsheet may reorder its columns, the
owner's marks and values stay, and only empty Spanish cells are bought.

### Follow-up (2026-11-08): the confirmation draw and how to read it (Work item 18, T-148b10)

The figures above are development figures. The confirmation is a second set of narrative multi bodies not
used before: `uv run scripts/draw_faithfulness_2026_11_06.py --out data/faithfulness_confirmation
--exclude-from data --extractors baseline,units --attribution-model data/models/attribution_2026_11_08
--n 30 --seed 20261109 --yes` (`--exclude-from` leaves out the bodies of the first draw, twins by body hash
included, and the held-out set). The pool is small: 32 narrative multi bodies are left, 30 are drawn. **The draw cannot be repeated:** the bodies file is never drawn again by design, so the same command a second time draws nothing and a different `--seed` does not redraw either; the confirmation set is this one, and a different set needs a different `--out` (and fewer bodies are left to draw). The
owner marks `faithful` in `faithfulness_to_tag.csv` as in the first draw. **Before any reading, rebuild
the derived file: `draw_faithfulness --exempt --out data/faithfulness_confirmation --attribution-model <dir>`
(free; the owner's file keeps its rows, marks, column order and line endings).** It is written by the rules of
five modules and goes stale with any edit to them; the report then skips the "every company named" line. The owner has decided to read only these 357 texts, so the marking of the non-companies is
**optional and was declined**: `draw_faithfulness --exempt --out data/faithfulness_confirmation` (free) would write
`faithfulness_exempt.json` and `faithfulness_exempt_to_read.csv` (the companies the plain resolver names that
the guard would reject, or the footer holds; `--yes` pays for the Spanish of its empty cells, and it is not
bought), and `real_company` yes or no on those rows would take the non-companies out of every extractor's
share. Without it, run `uv run scripts/report_wi18_2026_11_06.py faithfulness --out data/faithfulness_confirmation`:
it prints the block `[all]` and, for `units`, the line "the bar on every company named", and skips the
`real` blocks with no error (a file with no mark in it is the same as no file). Without the marks the
plain resolver's non-companies count against `units`: on the first 30 bodies that is 83.2% [72.7, 91.0]
(not decided) against 94.0% [87.0, 99.2] (pass) with the owner's 23 marks, and the gain over the baseline
hardly moves (24.3% [12.9, 39.4] against 25.1% [12.3, 41.4]). The rule proposed for reading the confirmation
without the marks is TASKS T-148b10 (6b). `faithfulness_exempt.json` is derived from the code of five modules and is free to
rebuild: if it is rebuilt after an edit the owner's marks (by article and company) are kept; the report skips
the readings that need it, with a line, while it was written by other code.

**Result and selection (2026-11-08).** The 357 texts were tagged, then `units2` and `units3` were added to the same file (38 new texts, 38 paid translations, 11,899 prompt and 9,240 completion tokens, approved by the owner) and tagged, and the 30 exempt names were read (25 not companies). `units` is 86.1% [80.9, 90.7] on every company named and 95.3% [92.0, 98.2] on real companies; the baseline 86.6% and 95.9%; `units2` 78.4% and 89.9%; `units3` 79.9% and 91.7%. **The owner selected `units` as the base model (80% is enough accuracy; not a pin or a publication).** Datasets, metrics, results and options: `docs/extractor_training.md`. After PR #155 (matcher change) the figures were re-read: see `docs/extractor_training.md`, "Update of 2026-11-10" (`units`: 87.3% counting every mark and 97.1% on the texts the current code writes, confirmation set; first set 97.5%).

**Spend (2026-11-08, approved by the owner): the confirmation set, baseline and `units`.** 30 bodies, 357 texts
and 30 bodies translated: 387 articles = 387 paid calls, 0 failures; 150,259 prompt and 134,950 completion
tokens (retries and rejected replies included). Files: `data/faithfulness_confirmation/`.

