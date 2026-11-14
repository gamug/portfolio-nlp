# TASKS.md — `portfolio-nlp`

Discrete, checkable task breakdown for `.specify/memory/PLAN.md`. Each
task references the plan work item and the `SPEC.md` section it closes.
Check a box only when its acceptance criterion (in `PLAN.md`) is actually
met — not when the code is merely written.

**Closed work items live in `.specify/memory/CHANGELOG.md`**, moved there verbatim
(task IDs unchanged) once every task in them is done, superseded, or moved elsewhere —
see constitution AI behavior #15. This file carries only open work: done tasks of an open work item are in
`CHANGELOG.md` ("Tasks closed while their work item is still open"), and the long history of an open task is in
`.specify/memory/archive/`. An entry is a status, the decision, what is open and a pointer (`.specify/memory/INDEX.md`).

Task IDs are stable, same rule as `SPEC.md`'s `FR-0xx`/`NR-0xx`: never renumbered or
reused. A cancelled or superseded task is checked, its closure noted, and moved to
`CHANGELOG.md` (constitution AI behavior #15).

## Work item 2 — Make the regression gate runnable (out of the MVP, owner 2026-11-10)

Its tasks (T-010 to T-016) and plan are in `archive/PIPELINE-superseded-2026-11-10.md`.

## Work item 3 — NER: validate the subword-fragmentation fix (resolved 2026-09-12)

*5 done tasks of this work item moved to `CHANGELOG.md`, section "Tasks closed while their work item is still open" (task IDs unchanged; moved 2026-11-08).*

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

- **T-022** *(out of the MVP, owner 2026-11-10; text in `archive/PIPELINE-superseded-2026-11-10.md`)*
## Work item 4 — Sentiment: close the entity/net-signal reasoning gap (resolved 2026-09-13: fine-tuned + chunk-level selected and merged)

*3 done tasks of this work item moved to `CHANGELOG.md`, section "Tasks closed while their work item is still open" (task IDs unchanged; moved 2026-11-08).*

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

- **T-032** *(out of the MVP, owner 2026-11-10; text in `archive/PIPELINE-superseded-2026-11-10.md`)*
## Work item 7 — NER: develop batch processing (code done 2026-09-12; T-062 needs a real GPU)

*3 done tasks of this work item moved to `CHANGELOG.md`, section "Tasks closed while their work item is still open" (task IDs unchanged; moved 2026-11-08).*

`run_ner_stage` (`src/pipeline.py`) was the only stage with no batching —
one chunk through the model per forward pass, unlike `run_category_stage`
(`CATEGORY_BATCH_SIZE=8`) and the summarization stages
(`SUMMARY_BATCH_SIZE=4`). Not a documented trade-off, genuinely
unaddressed. T-025's 2026-09-12 resample measured the real cost: ~22
articles/sec unbatched — a full-corpus backfill of the remaining ~439,000
pre-fix articles (the open T-022 question) would take ~5+ hours at that
rate on hardware with headroom to go faster. → `PLAN.md` Work item 7,
`SPEC.md` §13 item 11.

- **T-062** *(out of the MVP, owner 2026-11-10; text in `archive/PIPELINE-superseded-2026-11-10.md`)*
## Work item 15 — Category: fix the `other` bucket (**superseded by Work item 18, 2026-10-28**; filed 2026-10-14)

*20 done tasks of this work item moved to `CHANGELOG.md`, section "Tasks closed while their work item is still open" (task IDs unchanged; moved 2026-11-08).*

**Superseded by Work item 18 (2026-10-28); its plan is a stub in `PLAN.md` and the text is in the archive.** Open here: T-129 (close-out), T-131 (the backfill, P6) and T-132 (the upstream alias table). Everything else is in `CHANGELOG.md` ("Tasks closed while their work item is still open"), all marked done: *superseded by Work item 18* T-116, T-117, T-119, T-120, T-122–T-126, T-130 (2026-10-28), T-137 (2026-10-29), T-127 and T-138 (2026-11-05); *closed negative* T-121; *done or decided* T-118, T-128, T-133, T-134, T-139–T-142. The dependency table and preamble that described them: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md`, "Work item 15 preamble and dependency table".

- [ ] **T-132** **→ Work item 18 P1 and P3.** Body-hash grouping and name/ticker resolution shipped in P1 (`news_nlp.triage`, T-144). Shrunk by the owner 2026-11-05: **only the upstream alias table stays open** (names, aliases and ticker forms, offered by `portfolio-data-mining` as a CSV). The per-pair `review` verdicts, T-130's judge and the block on T-117 are superseded; the quarantined pairs follow D19 (T-131). Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-132.
- [ ] **T-131** **→ Work item 18 P6: backfill. D36: only derived results are moved aside (reversible archive), never an article; the lean `articles` row is derived and restorable, the SOURCE row is the article (owner, 2026-11-09, D36 (h2)).** *(owner decision D4, 2026-10-22, recommended defaults)* Move aside the existing results of quarantined pairs (D36 (b), (h1): no article is removed; the first wording said "clean up the results of the articles the crawler fix removes").
      - (a) Derive the removed ids at run time with a read-only join of RESULTS article ids to SOURCE (`match_class = 'reject'`, quarantined; about 315,822 RESULTS articles on the 2026-10-22 count (the SOURCE has 322,902 `reject` rows with a title and a body, D36); **proposed:** `review` (49,960 RESULTS rows on the 2026-10-22 count) is moved in a second step, after the owner has seen the `reject` result, or in the same one if the owner says so), not from the stale 2026-10-15 CSV; keep out upstream's known false rejects (at least 433 pairs, mostly News Corp `NWS`/`NWSA`) (**proposed:** still kept out until the owner decides whether the exception is needed for a reversible move). (a2) Before any move, a pre-move check of the `reject` class's own error rate on 400 random articles; go only if the upper 95% bound is at most 5% (**proposed:** the gate stays for `reject`, though the move is reversible; the owner decides).
      - (b) Per table (`article_sentiment`, `article_entities`, `article_category`, `article_summary`, lean `articles`), in one transaction: copy the removed ids into `<table>_removed`, then delete them from the live table; dry run on a scratch copy first; **never rename a whole table** (it would make every article pending and force a full GPU re-run). (c) Delete only the affected `sector_summary` rows (no version bump); **proposed:** copy them to `sector_summary_removed` like the other tables, or, as they are recomputable (a deterministic template, no model), say so here and keep D36 (b) as the owner reads it; the owner decides. **Acceptance (proposed):** the dry run on the scratch copy puts every `<table>_removed` row back in its live table and the row counts return to the starting ones.
      - T-117, T-124 and T-130 are superseded (2026-11-05): the judge calls and the `article_category` rename are dropped; the v1 move of the old tables is T-150 (D19). → step 6b. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-131.
- [ ] **T-129** **Re-pointed 2026-10-28: closes with Work item 18's close-out (T-150).** Close-out: annotate `SPEC.md` §13 item 19 / §14, add
      one FR per behavior that ships (results cleanup T-131; the gate, the
      calibrated rule and the relevance judge T-130 are superseded) in the PR that
      ships it (SPEC's own rule), update
      `docs/legacy/category-taxonomy-v1.md`, `docs/evaluation.md` (dated follow-up),
      `docs/modules/news-nlp.md`; reconcile both architecture artifacts —
      never rename (constitution AI behavior #11); confirm T-131 is done (T-130 is superseded); then move this work item to
      `CHANGELOG.md` (AI behavior #15). → acceptance criteria.

## Work item 17 — Articles that name more than one company (decided 2026-10-22: its own work item, sizing first, D9 default)

*2 done tasks of this work item moved to `CHANGELOG.md`, section "Tasks closed while their work item is still open" (task IDs unchanged; moved 2026-11-08).*

**Re-scoped 2026-10-28 into Work item 18** (`PLAN.md`): see the per-task markers below.

**Superseded by Work item 18 (2026-10-28).** The sizing (T-137) was absorbed by T-144 and the multi-company design (T-138) was superseded by D27–D28 (2026-11-05); both are in `CHANGELOG.md`. What is left here is T-138b, the close-out, which closes with Work item 18's close-out (T-150). The original preamble and task order: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md`, "Work item 17 preamble".

- [ ] **T-138b** **Re-pointed 2026-10-28: closes with Work item 18's close-out (T-150).** Close-out of Work item 17: annotate `SPEC.md` §13 item 21 / §14, reconcile
      both architecture artifacts (never rename; AI behavior #11), then move the work item
      to `CHANGELOG.md` (AI behavior #15). T-137's sizing was absorbed by T-144 and T-138 was superseded (2026-11-05; moved to `CHANGELOG.md`), so it closes with T-150 and waits on nothing else.

## Work item 18 — Triage-first pipeline: unique bodies, subject scope, per-company and sector news (re-scoped from Work items 15 and 17, 2026-10-28)

*6 done tasks of this work item moved to `CHANGELOG.md`, section "Tasks closed while their work item is still open" (task IDs unchanged; moved 2026-11-08).*

Plan: `PLAN.md` Work item 18. Decisions D11–D24: its "Decisions" paragraph (first recorded in T-143, now in `CHANGELOG.md`); D25–D39: the sections that follow it (D25 and D34 are pointers; D39 is the pipeline). Each phase is its own PR; a schema
change is proposed to the owner in the PR that needs it (constitution AI behavior #10).

- [ ] **T-146** *(P2; spend D18: hard cap 6,000 calls, logged in `docs/evaluation.md`, counted per task, the run stops at the cap; owner answers 2026-10-29 in PLAN WI18; Q1-Q10 answered)* Split:
- [ ] **T-146b** *(P2; export/check built 2026-10-29, PR #117, aligned with `docs/labelling-spec-v2.md`)* The re-tag export of the 720 gold bodies for the owner and the importer of the owner's new columns (`scope`, `company_labels`, `mentions`, `sub_industry`, `notes`), protected like `gold_review` (refuses to overwrite a tagged file); gold v2; re-measure the rules' 0.70 on it.
      - Built (`news_nlp.retag`, `scripts/retag_gold_2026_10_29.py export|check|normalise|guide|migrate|translate`): `check` verifies all 720 ids once and tickers/sub-industries against SOURCE; `normalise` turns typed names into tickers; `migrate` converts an old-format file into a new one; `sub_industry` takes the 8-digit GICS code or the name.
      - Every file to tag carries `spanish_title` / `spanish_text` after `text` (`news_nlp.translate`, cache `data/gold_translations_es.csv`, paid DeepSeek translation, about 1 USD per 720 bodies; buys nothing without `--yes`, at most `--max-new`; a failed run writes nothing and a re-run pays only for the missing ones).
      - Open (owner): run `guide` after closing `data/category_gold_v2_guide.csv` in Excel, then tag; the re-measure on gold v2 runs once it is tagged. → P2, D18. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-146b.
- **T-146c** *(out of the MVP, owner 2026-11-10; D18's silver labelling)*
      - **T-146d** *(out of the MVP, owner 2026-11-10)*
      - [ ] **T-146e2** Export/import of the 50 prefiltered articles the owner checks (protected like
        `gold_review`); a separate set from the ~50 double-tagged and the ~60 random misses.
      - [ ] **T-146e** *(drawn 2026-11-05 as 200 bodies, D30: `gold/category_gold_v2_test.csv`; **tagged by the owner and migrated 2026-11-07, PR #142: 200 of 200 valid (118 single, 47 multi, 11 market_economy, 9 sector, 8 unrelated, 7 investing_general); measured on it (`report_wi18 scope --gold --out gold`): the rules' scope agreement 68.0% [61.3, 74.5] weighted to the corpus, single recall 91.0% [87.1, 94.4] pass, multi recall 52.5% [42.8, 63.5] fails the 80% bar, the other four scopes not gated (9, 11, 8 and 7 gold bodies, D33)**; a scope under 40 gold bodies is not topped up and has no recall floor, D33; the "about 300 bodies, at least 40 per scope" below is the original plan)* Held-out gold, drawn once before
        any training; the draw and the original plan: `archive/PIPELINE-superseded-2026-11-10.md`.
- [ ] **T-148** *(P4; rewritten 2026-11-10, D39)* Texts, sentiment and category per (body, company): replaced by **T-162 to T-168** below; the per-company category is accepted at >= 80% agreement with the held-out gold (D24). Old text: `archive/PIPELINE-superseded-2026-11-10.md`.
- **T-148b2** *(out of the MVP, owner 2026-11-10; text in `archive/PIPELINE-superseded-2026-11-10.md`)*
- **T-148b3** *(out of the MVP, owner 2026-11-10; text in `archive/PIPELINE-superseded-2026-11-10.md`)*
- [ ] **T-148b11** *(partly done 2026-11-10: the marks of every extractor are in `gold/faithfulness_gold.csv` (453) and `gold/faithfulness_confirmation/` (438), `docs/gold_datasets.md` set 4; **left:** the exempt-company readings and the staleness record, still only in `data/`; the text below is of before)* **Migrate the evidence for the cut to `gold/`** (the marks of the `units` rows, `units2`/`units3`, the exempt-company readings and the confirmation set exist only in the git-ignored `data/`, so D34 (e) cannot be re-derived from the repository; the faithfulness-gold builder must carry the counts per extractor and set, the exempt readings, the code revision of each extractor's rows and the text hashes the owner read, with tests). Full text: `.specify/memory/archive/SPEC-TASKS-condensed-for-the-category-step-2026-11-13.md` § T-148b11.
- **T-158** *(out of the MVP, owner 2026-11-10; text in `archive/PIPELINE-superseded-2026-11-10.md`)*
- **T-160** *(out of the MVP, owner 2026-11-10; text in `archive/PIPELINE-superseded-2026-11-10.md`)*
- **T-148b7** *(owner 2026-11-10: covered by PR #155 (D38, T-161: common words and surnames); text in `archive/PIPELINE-superseded-2026-11-10.md`)*
- [ ] **T-148c** Sentiment over the texts of T-148a and T-148b (D26): the 180-text gold (`scripts/draw_sentiment_gold_2026_11_05.py`: 120 from summaries and structured multi texts, 60 from the extractor's texts) is tagged and the current model is measured on it; fine-tuning on summaries only if it misses the D32 bar (agreement >= 80%, negative precision and recall >= 70%). **Measured 2026-11-06 (PR #131): 60.4% [52.8, 67.6], fails; the owner chose fine-tuning on summaries (2026-11-07, PLAN D35):** 450 summaries tagged (`scripts/draw_sentiment_train_2026_11_07.py`, 150 per predicted class, a validation slice of 90, never a gold or held-out body); the gold is scored once per run that could ship. Fallback if it still fails on 2026-11-11: the cut goes with sentiment marked as measured. Covers T-127's question on real output. Sub-tasks: T-148c2b, T-148c2c (restored, the verdicts part), T-148c3, T-148c4 (T-148c1b and T-148c3e dropped by the owner, 2026-11-14). → D26, D35. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148c.
- [ ] **T-148c2b** *(built, #148, 2026-11-08; the owner's review is left)* Scorer hardening for the sentiment scoring: a Hub model needs a 40-hex `--revision`; `--redo-role` needs `--redo-reason`; the log gains `retag.SENTIMENT_SCORING_LOG_COLUMNS`; the report gives the agreement apart for texts over 128 tokens; a `final` run gives the paired gain of agreement and of negative precision and recall over the last `baseline` row (`gold_metrics.paired_stratified_gain`) with the gold's verdicts (T-148c4 reads them; the scorer decides no pin). `--pair` scores a model that reads the pair company + text (`src/sentiment_pair.py`); the 37 of 180 rows with no ticker go to the text-only model, and the row logs `input_format` and the rows each model read; `--pair` cannot be the baseline. The pin key for a second model is PR C. → D35. Full text before 2026-11-09: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148c2b.
- [ ] **T-148c2c** *(restored 2026-11-14, owner: the verdicts part only; proposed, not built; D37, owner 2026-11-09)* The gold scorer (`score_sentiment_gold`) records the precision, recall and ROC OVR of each class (`draw_sentiment_gold.predict_probs` gives the probabilities) and the paired gain of the score and of the negative class's three, with verdicts, in its log (`retag.SENTIMENT_SCORING_LOG_COLUMNS`); T-148c4 reads them through `pin_decision`. → D37. **Added 2026-11-10 (#164):** it also writes the confidence check of T-148c4 (mean largest class probability on the 180 texts, marks unread).
- [ ] **T-148c3** *(built 2026-11-09, PR B #154; grid and final run made; nothing published, the gold not scored)* Trainer of D35's recipe, **read as the pair company + text** (`src/sentiment_pair_train.py`, `scripts/train_sentiment_pair_2026_11_09.py`; the numbers are in `grid_results.json`, the commands in `docs/modules/news-nlp.md`). Data: marks train 341 + validation 83 (26 company-less rows left out), replay 341 of 925 eligible. Metrics D37. Grid: 2 epochs, 3e-5. Final (seed 4): score 67.8% [58.0, 76.8]; paired gain over the pinned model `not decided`, so the pin rule (D35 (3), (4)) would NOT pin on this slice; D32's bar: negative precision `fail`, recall `pass`. Pinned model: as text 60.9%, as a pair 61.3%. **Replay kept as built** (owner, 2026-11-09). The bar is the gold's (T-148c4). After this: T-148c3b to T-148c3f, `docs/sentiment_training.md`. Earlier text: `archive/TASKS-open-entries-2026-11-08.md` § T-148c3.
        - [ ] **T-148c3f** *(owner)* Confirm or change the 270/135 split of `gold/sentiment_train_2.csv`.
        - [ ] **T-148c3c** *(selected, owner 2026-11-10; the final relevance model and the stack's coefficients are built and published, private, T-169; **left: the strict nesting**)* Strict nesting of the stack. **Owner, 2026-11-10: accepted as a limit of the gold score (stated beside it); optional, after T-148c4.**
        - [ ] **T-148c3d** *(proposed, after c3c)* The stack in `sentiment_stage`: cost, disagreement rule, 180-text gold (T-148c4).
- [ ] **T-148c4** Publish the fine-tuned sentiment model and pin it in `config/pipeline_models.json`, its own PR. The candidate is on the private Hub repo `gamug/FinBERT-company-sentiment@9a7b1eb3` (T-169, 2026-11-10; not the D35 name `...-summaries`, which is replaced: `PLAN.md` D35, `config/published_models.json`) and is made public only in this PR (**first recreate the repo: revisions `1d1942be` and `1631bdac` keep wrong card text in their history**); **before any pin or publication update both cards** (title "(private)", the gold result, the confidence-check result, licence) and run `publish_final_models_2026_11_10.py --check`; the `final` row and the pin name the commit SHA. The scorer's checks are T-148c2b (merged before the final scoring). Pin rule (D35, owner 2026-11-07): only `better` pins; `not better`/`not decided`/`undefined` keep the baseline. The final run's own validation agreement is checked before publishing; a second `final` scoring only for a defect in the run, never for the figure. The model is scored as the published Hub id and revision (not a local checkpoint). **First, a confidence check (owner, 2026-11-10):** mean largest class probability of the model on the 180 texts, marks unread, against 0.862 on the out-of-fold scores; more than 0.05 above is reported beside the score (`docs/sentiment_training.md`, "Published"); T-148c2c's scorer writes it. If the baseline stays, there is no pin PR and the repo stays private (the owner decides later). **No date of its own** (owner, 2026-11-09; fallback 2026-11-11); starts with the owner's word after #154's review; the pin is decided by `sentiment_pair_report.pin_decision` on the verdicts of the scorer's `final` row (T-148c2c adds them), nowhere else. → D35. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148c4. **Candidate selected (owner, 2026-11-14, reading `docs/sentiment_training.md`, "The decision"): the stack, i.e. the pair model `gamug/FinBERT-company-sentiment@9a7b1eb3` (run `final_all__e2_lr3e-05_s4`, from `gamug/FinBERT-financial-news@93863fcb` on all 809 rows + 341 replay) with the relevance model `gamug/deberta-v3-company-relevance@8812c8bc` and `config/sentiment_stack.json`: 0.814 nested weighted D37 against 0.737 for the sentiment model alone and 0.648 for the pinned one (cross-validation, not held-out). **Pin rule (owner, 2026-11-14): the rule of D35 on the verdicts of the scorer's `final` row; T-148c2c is restored for them.** **The stack is scored on the pairs the step reads, the main ones (owner's population; 0.814 is of all marked pairs).**
**Before Build 1 starts (D40; owner, interactive session 2026-11-10):** schema of the per-step tables approved as written, and `pipeline_step_state` stays (decided); the scope classifier reads title plus the first lines of the body (decided; T-147's measurement decides if it can pass, below); the size of the rule is measured on step 1's output and then set by the owner (decided; blocks step 2 only). Can start now: T-162, and T-163's tables and status command; T-147 continues in parallel. Step 1's scope waits for T-147 (or the owner's fallback to the rules).
- [ ] **T-181** *(owner 2026-11-11: its own task, matcher line, D38/T-161, closed, CHANGELOG)* **Short names are not matched:** "Disney", "JPMorgan", "Smucker", "Huntington Ingalls", "Discovery", "Rowe Price", "International Business Machines" name universe companies and the resolver finds none, so a line headed by one goes to nobody (12 of 352 headers in 500 bodies). Needs aliases and a count of false matches; not part of a cut reading.
- [ ] **T-185** *(Build 1, D40; the run of step 3)* **Run step 3 on the single bodies** (`--items single`, the owner's run; step 2 has done them). **Done:** the sub-news of `units5-d3e2cbfa830e` (123,146 items with a main company, 38,428 on the review list, `companies-f349536cf701`; snapshot of 2026-11-13). **Decided by the owner (2026-11-13, #178):** `--threshold 0.5` (revisited with T-170; a new value is a new revision), the raw NER spans are not stored, `--candidates ner+text`, the corrected definition of review #178 and its limits accepted (`docs/companies_step.md`). → D39 (i), (iii), D40.
- [ ] **T-168** *(Build 2, D40; category part built provisionally 2026-11-13, `docs/category_step.md`, its test is T-192; **sentiment part built 2026-11-14**, `docs/sentiment_step.md`: the stack on the local pair model, nothing downloaded, its test is T-193)* *(D39 (6), (7))* Sentiment (pair + relevance stack: T-148c3c, T-148c3d, T-148c4) and category over each summary or sub-news with its main company; a sector summary gets a sentiment from `gamug/FinBERT-financial-news`, the pinned chunk-level stage, with no company (D39 (iv)); items on the review list get none until read. **Left of the sentiment part:** the sector summary's sentiment with no company (D39 (iv)) and the text-only route; the pin of `config/pipeline_models.json` stays T-148c4's; **the two architecture artifacts (constitution AI #11, Portfolio Thesis and Portfolio NLP) are reconciled by the owner's word, when #188 is ready to merge, not before**. → D39, D26, D29. **Closing check:** every item of a dry run has sentiment and category, or a `skipped` / `review` status; counts per status are printed.
- [ ] **T-169** *(Build 2, D40)* *(final training; owner 2026-11-10)* **Final training of the two models of step 6:** (a) the relevance model on the 809 rows (T-148c3c) with its strict nesting check; (b) the sentiment pair model on every row from `gamug/FinBERT-financial-news` (`--final-all`, T-148c3g: the existing run `final_all__e2_lr3e-05_s4` or a repeat, the owner chooses). Then the stack's coefficients on out-of-fold scores, both published to the private Hub and scored once on the 180-text gold (T-148c3d, T-148c4). Both are also what step 3.2 and step 2 use as the relevance model (T-164, T-165, closed, CHANGELOG). → D39, `docs/sentiment_training.md`. **Order:** (a) comes first and gives the threshold (D39 (i)); T-164 and T-165 run with it (a threshold change is a new step revision, D39 (vi)). **If the gold score fails the D35 bar:** the model is not pinned and the endpoint's current revision stays; the owner decides the next step. **Built and published, private (2026-11-10, owner's word):** `gamug/FinBERT-company-sentiment@9a7b1eb3` (the existing run `final_all__e2_lr3e-05_s4` + `config/sentiment_stack.json`) and `gamug/deberta-v3-company-relevance@8812c8bc` (809 rows, seed 1); cards with data, recipe, figures and limits; details in `docs/sentiment_training.md` "Published". **Left:** the threshold (from the out-of-fold scores), the gold score once (T-148c4), the pin only if D35 passes (the strict nesting, T-148c3c, is accepted as a limit, optional). Revisions: `config/published_models.json`. **Order of step 6 (owner, 2026-11-14): T-169, then T-148c4 (publish and pin), then build step 6 (T-168), then T-193 (its test) and T-171 (the docs).**
- [ ] **T-170** *(testing, with T-191; D40)* *(D39; review of #160)* **Evaluate the main-company pick:** how often the relevance model's main company (single: on the summary; multi: on each sub-news) is the one the owner would pick, and how the relevance model, trained on summaries, behaves on sub-news. Needs a small set the owner marks (which companies are main, on summaries and on sub-news); the pair-labelled sets (sentiment 180, 450, 405) say the company's sentiment, not which one is main, so they are not enough. The size of the set and the bar: the owner. → D39, D24. **Rule before it is run:** if the relevance model on sub-news is more than 5 points of AUC under its 0.90 on summaries, sub-news get their own threshold chosen on the marked sub-news. Sector summaries get a few texts in the set so sector sentiment can be checked (SPEC FR-001). **Owner to say:** the size of the set and the bar. **Also reported:** the share of NER spans that resolve to a ticker, and a read by the owner of 50 ignored spans (are real universe companies lost?). **Evidence of 2026-11-13** (`docs/companies_step.md`): on sub-news the share with a main company moves from 65.5% to 84.0% between the thresholds 0.8 and 0.1, on single bodies only from 60.5% to 65.5%: the 809 summaries do not set the sub-news threshold, this set does.
- [ ] **T-171** *(D39; when the pair model is pinned, T-148c4)* Update `CLAUDE.md` (stage table), `docs/modules/news-nlp.md` and `docs/evaluation.md` to the pair model and the relevance stack, and the review list of D39 (ii).
- [ ] **T-173** *(Build 1, D40)* *(D39, FR-021; closes the block)* **End-to-end dry run:** 200 bodies through every step, one command at a time with `--limit`, each step re-run once to show zero rows added or changed; the `--status` table of the run is pasted in the PR. → D39.
- [ ] **T-190** *(testing; step 2, summary)* **Test the summaries:** (1) their faithfulness, with yes / no questions like T-183 on a sample the owner marks (size and bar: the owner); (2) the determinism of the GPU run and the time per item (`scripts/summary_determinism_check_2026_11_12.py`, `docs/summary_step.md`). The corpus run is done. → D39, T-184 (closed, CHANGELOG).
- [ ] **T-191** *(testing; step 3, companies: NER and the main company)* **Test step 3** with T-170 (the owner's marked set and the bar): (1) on a 100-body dry run of the single path, the share of items with a main ticker, on the review list and with ignored spans; (2) the same on the sub-news of `units5`; (3) the owner reads 50 ignored spans (are real companies lost?). The run on the single bodies is T-185. → D39, T-164, T-165 (closed, CHANGELOG).
- [ ] **T-192** *(testing; step 7, category)* **Validate the category step:** (1) the owner's yes / no review of a sample of `news_category` rows (size and bar: the owner; the provisional model measures 43.0% label / 55.6% group on the held-out single rows, `docs/category_step.md`); (2) the counts per label and group of the first complete run; (3) the check of T-168: every item of a dry run has a category or a `skipped` / `review` status. → D41, FR-022.
- [ ] **T-193** *(testing; step 6, sentiment; the model tasks are apart)* **Validate sentiment per (text, main company)** (the step exists since 2026-11-14, `docs/sentiment_step.md`): the 180-text gold scored once (T-148c3d) and the owner's review of a sample of the run (size and bar: the owner). It depends on the model tasks (T-148c, T-169), which this entry does not touch. → D39, D37. Score the gold on **main pairs** (relevance at or over the threshold), the population of the step.
- [ ] **T-194** *(testing; step 1.2, the cut: the attribution model)* **Test that the Hub copy is the model the cut used** (`gamug/FinBERT-news-attribution@436ad700`, private, `HF_TOKEN`; published in #185; the cut is pinned to it by #186; once that merges: `--step cut --fetch`): (1) download **only the five model files** into a clean folder (a `README.md` there changes the identity) and check that `directory_identity` is `8399ccf003c60143` and that `scripts/publish_attribution_model_2026_11_14.py --check` says yes to all five lines; (2) repeat the cut with `--model-dir` on that folder for a sample of `multi` bodies of `units5-d3e2cbfa830e` and compare the sub-news with the stored ones (same count and texts; a difference is reported, not accepted); (3) re-score the validation pairs of `docs/attribution_training_record.json` (ranking 0.901 on 3,822 pairs) from the Hub weights. (4) the card on the Hub says "not pinned in the pipeline" while the cut is pinned in #186: the owner decides whether to republish it, which gives a new revision (the pin of `config/attribution_model.json` and the `published_models.json` record change with it). The size of the sample: the owner. → T-148b12 (closed), #185.
- [ ] **T-174** *(Build 2, owner does it; D40)* **Train the category model** while Build 1 runs: data `gold/category_gold_v2.csv` (284) and `gold/category_gold_v2_test.csv` (200), texts in `gold/texts/`; bar >= 80% agreement on the held-out 200 (D24); the output the pipeline waits for is the pinned revision read by T-168. → D24, D29. **Split:** the same 200 held-out as T-147, never trained on by either model; the 284 carry the 12 label changes of 2026-11-11 (`git log gold/category_gold_v2.csv`), so use the file as it is now. T-147's scope model trains on 365 (284 + sets 7 and 8); this one on the 284 unless the owner adds them. **Status 2026-11-13 (D41, `docs/category_results_2026_11_13/`):** 225 train rows, option B groups; CV 64.0% flat / 62.2% mixed; held-out single rows (135 of 200; 44 narrative unscored), scored once with the mixed fine-tune: 48.9% two-step; bar 80% not met; not pinned. Next: T-187.
- [ ] **T-186** *(#182)* `news_nlp.translate`'s cache is keyed by `article_id`, so several rows of one article share one translation (the scripts work around it with `article_id_ticker`). Key it per row, with a test, without paying again.
- [ ] **T-187** *(owner, T-174)* Tag the 1,000 rows of `data/category_multi_to_tag.csv`, migrate them to `gold/` as a new set (`docs/gold_datasets.md`), cross-validate (225 + 1,000 rows), measure the taxonomy's full definitions. The held-out 200 is not scored again without the owner's word; a refit on all rows only after the bar is shown.
- [ ] **T-189** *(owner, #183)* When the first revision is complete, delete `category-981dacf6f68d` (inside it, same labels; `RECATEGORY_SQL`); drop `LEGACY_REVISIONS` at the first score change (a test pins it).
- [ ] **T-150** *(P6, close-out)* Full backfill (old tables to v1, T-131), knowledge-graph export
      contract update (`portfolio_common.news_export`), `SPEC.md` §13 items 19 and 21, both
      architecture artifacts (AI #11), then Work items 15, 17 and 18 to `CHANGELOG.md` (AI #15,
      closing T-129 and T-138b). → acceptance.

- [ ] **T-195** *(Build 3, D42, FR-024; replaces T-152)* **The snapshot step** (`--step snapshot`): `news_snapshot` per (text, main company) or `sector` text (no category or sentiment for it), from the five step revisions and the cut revision; complete rows only; `companies_mentioned` left empty for now (T-198); `.env` `BATCH_TIMEZONE` (default `America/New_York`) in `.env.example`. Design and tables: `docs/snapshot_steps.md`. The first real run waits for sentiment and category on the store (T-192, T-193).
- [ ] **T-196** *(Build 3, D42, FR-024)* **The period summary step** (`--step period`): window `(previous cut, this cut]` by `fetched_at`, a fixed 15:00 in `BATCH_TIMEZONE`, 21 days from 2022-01-03; counts per company and category, and the same counts per sector summed from the company snapshots (`sector` and `company_sector` tables, no LLM; one count per snapshot; sector texts not counted); `period_item` says which period counted which snapshot (late = its window was closed); frozen periods; `--catch-up`, `--render`.
- [ ] **T-197** *(Build 3, D42; tests of T-195 and T-196)* Hermetic tests: the period ends and the daylight-saving conversion, the first window, a late item through `period_item`, idempotent rerun, a period never rewritten, a sector with no news dropped, a sector line equal to the sum of its companies, a company with no sector in `unclassified`, the `sector` table equal to the constants, no category or sentiment on a sector snapshot, `BATCH_TIMEZONE` invalid refused.
- [ ] **T-198** *(deferred, owner 2026-11-14; D42)* **`companies_mentioned` and its floor:** out of scope for now. When it returns: run the two read-only queries of `docs/snapshot_steps.md` ("The mentions floor") on the real RESULTS store, choose `mentions_floor` (`config/snapshot_step.json`) from the distribution of the non-main scores, and make the snapshot step refuse to run while it is unset.
- [ ] **T-199** *(deferred, owner 2026-11-14; D42)* **Extra sectors** (those GICS does not have): none now, the 507 tickers of `gold/universe.csv` all have a GICS sector; a sector is a row of `sector` (`kind = extra`, code from 90) and its tickers rows of `company_sector`.
- [ ] **T-155** *(gold sets, `docs/gold_datasets.md`; validated against D39 on 2026-11-10)* In place in `gold/`: scope train 284 and test 200, label check 200, extractor faithfulness 453 and its confirmation set 438 (migrated 2026-11-10), sentiment 180, training sets 450 and 405 (marks and ids only). No gold is needed for the relevance model's main company, NER or the sub-news summaries (owner: the relevance model, the trained NER and the existing summariser are the design). **Texts:** tracked in `gold/texts/` since 2026-11-10 (owner: the repository and its data are public; `docs/gold_datasets.md`). **Left:** nothing for D39. Text before 2026-11-10: `archive/PIPELINE-superseded-2026-11-10.md`.
- **T-154** *(out of the MVP, owner 2026-11-10; text in `archive/PIPELINE-superseded-2026-11-10.md`)*
## Status

**2026-11-14 (owner, interactive session):** the development tasks of the steps built (scope, cut, summary, companies, sector, category) are closed and moved to `CHANGELOG.md` ("Work item 18 — development of the steps closed"); what is open for them is testing (T-170, T-173, T-190 to T-194) and the run of step 3 on the single bodies (T-185). The scope bar of D20/D33 is lowered from 90% to 80% (the pinned model passes). The model tasks were walked through with the owner the same day: T-175 done (the licence card, on the Hub), T-148c1b and T-148c3e dropped, T-148c2c restored for the verdicts, T-161, T-147b and T-148b12 closed, the sentiment candidate selected (T-148c4). T-181 (short names) is not built and stays open.

**2026-11-05:** history of the first pipeline plan (the 7 steps of D25 are replaced by D39, 2026-11-10); T-127 and T-138 were superseded (moved to `CHANGELOG.md`); T-132, T-131, T-129 and T-138b continue under Work item 18 unchanged.

**Work item 18 (2026-10-28): Work items 15 and 17 are re-scoped into it** (D11–D24 are recorded in `PLAN.md` Work item 18).
T-121 is closed (negative), T-116/T-117/T-119/T-120/T-122–T-126/T-130 are superseded (their entries were moved out of this file, 2026-11-08), and
T-131, T-132 (shrunk to the alias table, 2026-11-05) and the sizing of T-137 (absorbed by T-144) continue under Work item 18; T-127 and T-138 were superseded on 2026-11-05. T-144 (sizing and the rules' check on the
720 articles tagged v1) is the first PR; T-145 (Laya) was tried and cancelled.

Closed Work items 1, 5, 6 and 8–14 are in `CHANGELOG.md` (Work item 5 closed 2026-10-14; its
only open task, T-041, was superseded by Work item 15).

T-010–T-016 (Work item 2, regression gate) remain blocked on the maintainer's
infrastructure decision — nothing in that item has started.

**Work item 15's own plan is superseded by Work item 18.** What carries over is the corrected corpus (T-128, done 2026-10-22), the results cleanup (T-131, the backfill of P6), the alias table (T-132, shrunk) and the close-out (T-129). Work item 16 (schema guard) is closed in `CHANGELOG.md`; Work item 17 (multi-company articles) is re-scoped into Work item 18, its sizing absorbed by T-144 and its design superseded by D27–D28. The superseded tasks' entries are in `CHANGELOG.md` and the archive.

**Model performance checking (Work items 3, 4, 7) is resolved except
non-blocking items:** Work item 3's only open task is T-022 (full-corpus
backfill); Work item 4's is T-032 (re-solve the purity estimates from measured
agreement); Work item 7 (NER batching) is code-complete with only T-062 (empirical GPU tuning)
left — CUDA has been available since 2026-09-14, so it is actionable, just not
yet run.

Not a task with an ID: the 3,444 pre-fix `sector_summary` rows (from Work item
6's T-058 fix) are still queued to self-heal on the next real `--summarize`
run — a deliberate production action left to the repo owner.
