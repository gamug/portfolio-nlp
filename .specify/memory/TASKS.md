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

- [ ] **T-022** *(maintainer)* Decide whether a bulk re-extraction of the
      remaining ~439,000 pre-fix `article_entities_v1` articles (not
      resampled by T-025) is worth doing — the fix is future-runs-only by
      design (`docs/evaluation.md` 2026-09-10 follow-up, "Scope"). No
      longer blocking T-020 (T-025 unblocked it with a smaller sample);
      this is now purely about full-corpus coverage. → step 3.
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

- [ ] **T-032** Re-solve `docs/evaluation.md`'s "Sample-size floor" purity
      estimates using T-031's actual measured per-stratum agreement
      (`strata_json` / `agreement_rate_target_negative` etc.) instead of
      today's planning-only estimates, before locking in a permanent
      `--sample-size` default. → step 3. **Still open** — not done as
      part of the 2026-09-13 fine-tuning work; the four real evals give
      more empirical grounding than before but the purity-estimate
      re-solve itself hasn't been done.
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
## Work item 15 — Category: fix the `other` bucket (**superseded by Work item 18, 2026-10-28**; filed 2026-10-14)

*20 done tasks of this work item moved to `CHANGELOG.md`, section "Tasks closed while their work item is still open" (task IDs unchanged; moved 2026-11-08).*

**Superseded by Work item 18 (2026-10-28); its plan is a stub in `PLAN.md` and the text is in the archive.** Open here: T-129 (close-out), T-131 (the backfill, P6) and T-132 (the upstream alias table). Everything else is in `CHANGELOG.md` ("Tasks closed while their work item is still open"), all marked done: *superseded by Work item 18* T-116, T-117, T-119, T-120, T-122–T-126, T-130 (2026-10-28), T-137 (2026-10-29), T-127 and T-138 (2026-11-05); *closed negative* T-121; *done or decided* T-118, T-128, T-133, T-134, T-139–T-142. The dependency table and preamble that described them: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md`, "Work item 15 preamble and dependency table".

- [ ] **T-132** **→ Work item 18 P1 and P3.** Body-hash grouping and name/ticker resolution shipped in P1 (`news_nlp.triage`, T-144). Shrunk by the owner 2026-11-05: **only the upstream alias table stays open** (names, aliases and ticker forms, offered by `portfolio-data-mining` as a CSV). The per-pair `review` verdicts, T-130's judge and the block on T-117 are superseded; the quarantined pairs follow D19 (T-131). Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-132.
- [ ] **T-131** **→ Work item 18 P6: backfill. D36: only derived results are moved aside (reversible archive), never an article.** *(owner decision D4, 2026-10-22, recommended defaults)* Clean up the existing results of the articles the crawler fix removes.
      - (a) Derive the removed ids at run time with a read-only join of RESULTS article ids to SOURCE (`match_class = 'reject'`, quarantined; about 315,822 articles on the 2026-10-22 count), not from the stale 2026-10-15 CSV; keep out upstream's known false rejects (at least 433 pairs, mostly News Corp `NWS`/`NWSA`). (a2) Before any purge, a pre-purge check of the `reject` class's own error rate on 400 random articles; go only if the upper 95% bound is at most 5%.
      - (b) Per table (`article_sentiment`, `article_entities`, `article_category`, `article_summary`, lean `articles`), in one transaction: copy the removed ids into `<table>_removed`, then delete them from the live table; dry run on a scratch copy first; **never rename a whole table** (it would make every article pending and force a full GPU re-run). (c) Delete only the affected `sector_summary` rows (no version bump).
      - T-117, T-124 and T-130 are superseded (2026-11-05): the judge calls and the `article_category` rename are dropped; the v1 move of the old tables is T-150 (D19). → step 6b. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-131.
- [ ] **T-129** **Re-pointed 2026-10-28: closes with Work item 18's close-out (T-150).** Close-out: annotate `SPEC.md` §13 item 19 / §14, add
      one FR per behavior that ships (results cleanup T-131; the gate, the
      calibrated rule and the relevance judge T-130 are superseded) in the PR that
      ships it (SPEC's own rule), update
      `docs/category-taxonomy.md`, `docs/evaluation.md` (dated follow-up),
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

Plan: `PLAN.md` Work item 18. Decisions D11–D24: its "Decisions" paragraph (first recorded in T-143, now in `CHANGELOG.md`); D25–D35: the sections that follow it. Each phase is its own PR; a schema
change is proposed to the owner in the PR that needs it (constitution AI behavior #10).

- [ ] **T-146** *(P2; spend D18: hard cap 6,000 calls, logged in `docs/evaluation.md`, counted per task, the run stops at the cap; owner answers 2026-10-29 in PLAN WI18; Q1-Q10 answered)* Split:
- [ ] **T-146b** *(P2; export/check built 2026-10-29, PR #117, aligned with `docs/labelling-spec-v2.md`)* The re-tag export of the 720 gold bodies for the owner and the importer of the owner's new columns (`scope`, `company_labels`, `mentions`, `sub_industry`, `notes`), protected like `gold_review` (refuses to overwrite a tagged file); gold v2; re-measure the rules' 0.70 on it.
      - Built (`news_nlp.retag`, `scripts/retag_gold_2026_10_29.py export|check|normalise|guide|migrate|translate`): `check` verifies all 720 ids once and tickers/sub-industries against SOURCE; `normalise` turns typed names into tickers; `migrate` converts an old-format file into a new one; `sub_industry` takes the 8-digit GICS code or the name.
      - Every file to tag carries `spanish_title` / `spanish_text` after `text` (`news_nlp.translate`, cache `data/gold_translations_es.csv`, paid DeepSeek translation, about 1 USD per 720 bodies; buys nothing without `--yes`, at most `--max-new`; a failed run writes nothing and a re-run pays only for the missing ones).
      - Open (owner): run `guide` after closing `data/category_gold_v2_guide.csv` in Excel, then tag; the re-measure on gold v2 runs once it is tagged. → P2, D18. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-146b.
- [ ] **T-146c** Revised LLM labelling prompt and gate: iterate on a fixed stratified 240 of gold v2 (at most 5 iterations, about 240 calls each), then the full 720 (480 new calls) at >= 90% scope agreement. Silver D15 labels and sub-industries are bought in P2 only if a second gate on the 240 (per-company category and sub-industry agreement with gold v2, >= 80%) passes; otherwise silver is scope-only. Budget: silver <= 6,000 - 240k - 480 for k iterations; judge repairs count against the 6,000.
      - The 240 is drawn once by `scripts/draw_gate240_2026_11_04.py` (quotas in `retag.GATE_QUOTAS`; local `data/gate240_ids.csv`); the owner tags it in `data/gate240_to_tag.csv`. The tracked gold files are `gold/category_gold_v2.csv` (240 + 44 sector) and `gold/category_gold_v2_test.csv` (200 held-out, #142), built by `scripts/build_gold_2026_11_05.py`; the article text stays out of git. Workflow: `docs/labelling-spec-v2.md`, "The fixed 240 of the prompt gate".
      - Owner calls 2026-11-05: company labels on sector/market_economy/investing_general bodies, several labels per company, `sub_industry` as GICS codes of any level; 44 more sector-like bodies were drawn for the sub-industry gate. → P2, D18. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-146c.
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
- [ ] **T-147** *(P3)* Triage stage on unique bodies: body-hash grouping (T-132), the scope first (NER then runs on `single` and `multi` bodies only, D25), from a local classifier (a small fine-tuned encoder with the `news_nlp.triage` rule features as inputs) trained on silver and gold, with a calibrated confidence. **Accepted at >= 90% corpus-weighted agreement with the held-out gold on all bodies and recall >= 80% for each scope with >= 40 gold bodies** (D20 as amended by D33; the rest reported with their interval); complete metric set per scope (AI #12). Each round exports about 300 low-confidence bodies for the owner's tagging (D21) until the target holds. `unrelated` is flagged and skipped; tables proposed to the owner in the PR; its own FR. **Model governance (AI #1, #12, Work item 8):** a new pinned checkpoint in `config/pipeline_models.json` with its revision, selection justified against alternatives, the `ExperimentSpec` stage list extended deliberately (a snapshot test). → P3. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-147.
- [ ] **T-148** *(P4)* Texts, sentiment and category per (body, company), steps 3-5 of D25; each
      sub-task its own PR and its own FR when it ships. Single-company bodies use the whole summary;
      a dominated body (D23) is single + mentions; **the per-company category is accepted at >= 80%
      agreement with the owner's held-out gold (D24, confirmed)**. → P4.
- [ ] **T-148a** Summaries: the pinned local summariser (`c_summary`) on every non-multi body (single, sector, market / economy, `investing_general`), title and body only. For the cut (T-154) it writes chunks of 250 bodies under `data/cut/summaries/` and resumes from the last chunk (`scripts/cut_summaries_2026_11_07.py`, `--status`, Ctrl-C and run again to resume). **Launch: `uv run scripts/cut_summaries_2026_11_07.py --require-cuda`** (the device is part of the manifest; PR #132); about 29,113 unique bodies, about 70% summarised, 5 to 6 GPU hours. Bodies the rules call `multi`, `roundup` or `no_company` get no summary; the rules never predict `investing_general` (D25 summarises it).
      - Measured 2026-11-06 (`scripts/cut_spans_effect_2026_11_07.py`, 500 bodies): the rule scope is the same with and without the stored NER spans (500 of 500), so the missing spans (none for 2026-09 and 2026-10) do not change the cut's scope; each chunk row records `spans`.
      - **Open (owner to decide before the long run), the order against NER on the window:** (a) NER first, then this job; or (b) run now and let T-154 recompute scope and mentions from the final spans (`--accept-new-spans` keeps the chunks; their `rule_scope` stays the old one; mentions differ in about 4% of bodies). Until decided, T-154 must not read `rule_scope` from the chunks. → D25, D26. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148a.
      - [ ] **T-148b** Step 4, the text of each company in a multi body (D27). Split:
- [ ] **T-148b2** Training set for the attribution model from the structured bodies (company name masked, previous sentence as context, all years). **Built** (`scripts/build_company_segments_2026_11_05.py`, `gold/company_segments_train.csv`, ids and positions only): 2,834 bodies, 33,624 segments (21,682 `lead`, 11,942 `continuation`), 1 body in 10 as `val` (205 before 2022, 65 from 2022 on, reported apart); the 33 bodies that share text with a tagged or held-out article are left out; `tests/news_nlp/test_company_texts.py` pins the parser constants and counts. Built 2026-11-06: the mention-and-recency baseline (`news_nlp.attribution_baseline`) and the label check (100 `lead` + 100 `continuation`, `gold/segment_check_gold.csv`).
      - **Rule fixed before the check (owner, 2026-11-06):** `continuation` is dropped from training if at most 80% are right. **Read:** `lead` 88.0% [80.2, 93.0], `continuation` 38.0% [29.1, 47.8], so the model trains on `lead` only (PLAN D27). Acceptance (D32): at least 80% of the companies named in the faithfulness bodies with a faithful text, and better than the baseline on the same bodies; read in T-148b10. → D27, D32. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148b2.
- [ ] **T-148b3** *(trained model, merged, #141; **not published, not pinned**)* The trained attribution model (`src/news_nlp/attribution_model.py`, `scripts/train_attribution_2026_11_08.py`, `CompanyResolver.mention_spans`, `draw_faithfulness --attribution-model`). Trains on `lead` rows only (D27, label check 2026-11-06; the run refuses to start unless `gold_metrics.train_on_lead_only` is true); bodies it cannot attribute are flagged `unextracted`, never given a default text.
      - **Left (owner's word):** publish, the pin in `config/pipeline_models.json` under `attribution` with its revision, the `ExperimentSpec` stage. The model behind every `units` figure is the local directory `data/models/attribution_2026_11_08` (`local:8399ccf003c60143`).
      - **Open question (proposed, not decided):** the model has not learned to answer "none of these": a sentence about a company outside the universe goes to the nearest one (wrong cases scored 0.9 or more, so a higher threshold does not help; (c) is not supported). Options: (a) retrain with negatives that belong to nobody; (b) measure again after the stricter cuts of #143 and #144. One-word names (`JPMorgan`) are not read by the resolver (T-148b7). → D27. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148b3.
- [ ] **T-148b4** *(owner's reading 2026-11-07, PLAN D27; merged, #143)* The `units` extractor: the attribution reads units cut at lines, bullets and clauses, without the site's footer.
      - Built: `src/news_nlp/attribution_units.py` (`strip_footer`, `attribution_units`), `model_company_texts(..., units=True)`, the extractor `units` in the faithfulness draw, `units` minus `baseline` / `model` in the report, the footer audit (`scripts/audit_footer_2026_11_08.py`, 4,000 bodies, seed 20261108), the `--stale` check (`faithfulness_stale.json`; exit code a sum, 1 stale and 2 unchecked; `--only`) and per-extractor code revisions. `chunking.split_sentences` is not changed.
      - Read by the owner (402 rows): 85.4% [76.5, 92.2] on the first set (not decided, D32), gain over the baseline +24.3 [12.9, 39.4] `better`; 21 of its 27 unfaithful texts came from first-word names (→ T-148b5). The figures that matter now are in T-148b10 (9).
      - Proposed, not built: **T-148b4b**, the gold builder carries `faithfulness_footer_only.json` and `faithfulness_stale.json` into `gold/` (precondition: the owner's decision on which reading of the bar counts). Not in this item: the first-word name matcher (T-148b5) and the model's own per-sentence attribution. → D27, D34 (e). Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148b4.
- [ ] **T-148b5** *(owner's decisions 2026-11-07, PLAN D34 (e); merged, #144)* The reading of the 402 rows left two causes before the model: first-word names taken for companies ("Bloom", "Union", "Home", "Equity", "Baker", "Boston": 21 of the 27 unfaithful `units` texts) and lines headed by a list of names and a dash whose text names other companies. **Built:** `attribution_units.GuardedResolver` (a short form counts only when its full name or a ticker form is in the body; `CompanyResolver.mention_spans(text, aliases=False)` is the only change to the resolver), `attribution_pieces(..., headers=True)` and `header_owners`, the reading `units2` (`attribute_sentences(reading=...)`, validated), a guard-aware faithfulness report (each extractor's own share and the paired gain on shared companies). First set: 135 texts of `units2`; the owner read the 33 new ones (T-148b6). Companies reached only by a short name are lost (11 of 32 of them were right). The figures that matter now are in T-148b10 (9). → D27, D34 (e). Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148b5.
- [ ] **T-148b6** *(2026-11-07, after the owner's reading of the 33 `units2` texts)* `units2` on the 30 bodies: 130 of 135 texts faithful, 84.4% [72.5, 97.2] on the bar (not decided: 19 real companies named have no text because a "NAMES - text" line went whole to the header names). **Built:** `reading="units3"` = `units2` plus a piece of such a line also belongs to the universe companies it names itself. First set: `units3` 91.6% [84.7, 97.8] (141 of 148 + 6 left without), `units3` minus `units` -3.9% [-7.3, 0.0], not decided. The owner chose `units3` as the candidate on 2026-11-07; **replaced by `units` on 2026-11-08** (T-148b10). Known limit, same family as T-148b7: a header line can carry a second company's sentence. The 5 omissions not looked at (CSCO, NSC, KR, NKE, SPG) are left to T-148b7. → D34 (e). Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148b6.
- [ ] **T-148b10** *(owner's decisions, 2026-11-08)* Candidate for step 4's cut, its confirmation and its selection.
      - (1) the candidate is `units`; `units2` and `units3` stay as measured fallbacks. (2) T-148b9 is out of scope. (3) The confirmation draw (30 new bodies, `draw_faithfulness --exclude-from data --out data/faithfulness_confirmation`; the pool is used up, 30 of 32 drawn) is done and tagged. (4) The stack #141, #143, #144 is merged, in that order, by the owner. (5) The sentiment tagging of the 450 summaries is done and its marks are in `gold/` (#145); the retraining (T-148c3) waits for the owner's word.
      - (6), (6b), (7), (8) were rules and schedules proposed before the confirmation was read (the bar on real companies, the bar on every company named, the held cut, the go/no-go schedule); superseded by (9).
      - **(9) Owner's selection, 2026-11-08: `units` ("units1") is the base model of step 4; the owner judged 80% accuracy enough.** Confirmation set: 86.1% [80.9, 90.7] on every company named (pass) and 95.3% [92.0, 98.2] on real companies; level with the baseline (gain -0.5 [-1.5, 0.0], not decided), so the rule proposed as (6b) is not met and the selection is the owner's call over it (that rule: `units` passes if the lower end of its interval is at or above 80% **and** its gain over the baseline on the same bodies is `better`, D32's two criteria, read on every company named). First set (development): 94.0% [87.0, 99.2] on real companies, 83.2% [72.7, 91.0] on every company named (not decided). `units2` and `units3` are not selected. Not a pin, not a publication. Record: `docs/extractor_training.md`. Follow-ups: T-148b11, T-148b12. → `PLAN.md` Work item 18, D34 (e). Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148b10.
        - [ ] **T-148b11** *(proposed, not built; after the owner has read the confirmation set)* **Migrate the evidence for the cut to `gold/`**, as T-155 did for the first 190 baseline marks and #145 for the sentiment marks. Today `gold/faithfulness_gold.csv` holds the 190 baseline rows only (a test asserts `{"baseline"}`); the owner's marks on the `units` rows (402), `units2`/`units3` (35), the 37 exempt-company readings (`faithfulness_exempt_to_read.csv`) and the confirmation set (357 texts, 30 companies) exist only in the git-ignored `data/`, so the decision recorded in PLAN D34 (e) cannot be re-derived from the repository. The faithfulness-gold builder already lets a row gain an extractor name; it must also carry, with tests that pin them: the counts per extractor and per set (first, confirmation), the exempt readings (`real_company` yes/no by article and company, no news text), the code revision of each extractor's rows, and the text hashes the owner read. Not built before the confirmation is read; the item is here so that it is not forgotten.
        - [ ] **T-148b12** *(proposed, not built, not started; after the owner's selection of `units`, 2026-11-08)* **Wire the selected extractor into step 4.** Today `attribution_model` defaults to `reading="sentences"` (the baseline), `cut_summaries_2026_11_07.py` calls no extractor and only `draw_faithfulness` runs `units`, so the selection of (9) is a record. Needed: a default or a setting naming the extractor (`units`) for the company texts of the cut, the model directory it loads (today the local `data/models/attribution_2026_11_08`, `local:8399ccf003c60143`, git-ignored; publishing and pinning it is the owner's word), and a test that the cut's reading is the selected one. Not started until the owner asks.
- [ ] **T-158** *(manager's review of #147, 2026-11-08; proposed, not started)* Apply the status-decision-open-pointer shape to the long paragraphs of `PLAN.md` Work item 18 (D34 (a) to (d), D35's recipe, order of the end and GPU: lines of 2 to 4.4 KB), moving the history verbatim to `archive/`. The section is 53 KB of a 58 KB cap (`tests/test_memory_size.py`); one more round of decision notes would fill it. Not before the current decisions of Work item 18 settle.
- [ ] **T-159** *(owner's decision D36, 2026-11-08; proposed, not started)* **Count the properly fetched corpus.** Read-only, no model and no spend: from the SOURCE `articles` (+ `discovered_urls`), the unique bodies (the same hash the triage uses) with a title and a `body_text`, by pair class (`accept`, `reject`, `review`, none), by whether the same body is also `ok` under another ticker, and by year (the cut's window from 2021-12-31). Answer: how many bodies the pipeline would gain over the 81,961 `ok` ones, how many per company by the tags and by the content, and whether the `thin_content` rows (1,785) count as properly fetched (the owner's call). The result is what D19's backfill scope, T-150 and the size of the cut's window are decided on; until then every figure says which corpus it is of. → D36.
- [ ] **T-148b8** *(manager's review of #144, 2026-11-08; the evidence under `units3` as the candidate)* (a) **The denominator:** the 91.6% of `units3` left out 36 companies the extractors are not asked about; on every company named it is 74.2% [66.0, 81.5] (`units2` 68.4%, `units` 83.2%). `draw_faithfulness --exempt` lists them (`faithfulness_exempt.json`, `faithfulness_exempt_to_read.csv`); the owner read the 37 of the first set (23 not companies, 14 are) and the 30 of the confirmation (25 not companies, 5 are); `report_wi18 faithfulness` gives the bar on real companies. (b) **A confirmation:** done (T-148b10). (c) The header audit (`scripts/audit_headers_2026_11_08.py`): datelines and labels in capitals and a dash inside a sentence are not headers (fixed, no text of the 30 bodies changes); 67 lines given to nobody still name a universe company after the dash (7 behind a short form the guard rejects: T-148b7), 15 headers are missed for a full stop in the first part. → D32, D34 (e). Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148b8.
        - [ ] **T-148b9** *(proposed improvement, **out of scope** by the owner's decision of 2026-11-08: the performance of `units` as it is is acceptable; after the owner's reading of the 37 exempt companies, 2026-11-08)* The guard loses 13 real companies on the 30 bodies, mostly lines headed by short forms ("Merck - Merck shares retreated...", "Colgate-Palmolive - Colgate named...", "United Airlines, American Airlines - ... Delta ...", "Molson Coors - ..."): the header rule gives such a line to nobody because the guard does not accept the short form in the head. Option, to be measured as a new reading (not a change of `units3`, whose texts the owner read): match the part before the dash with the plain resolver, since a name before a dash is deliberate; the owner would read the texts that change. Not built; whether it is worth building depends on the decision between `units` and a guarded reading for the cut.
- [ ] **T-148b7** *(owner's decision 2026-11-07)* Person names and common words ("home", "union") that the guard does not stop: an item of its own, **after the cut** (not before it), to be measured against the owner's next tagging round. Nothing is built for it now.
- [ ] **T-148c** Sentiment over the texts of T-148a and T-148b (D26): the 180-text gold (`scripts/draw_sentiment_gold_2026_11_05.py`: 120 from summaries and structured multi texts, 60 from the extractor's texts) is tagged and the current model is measured on it; fine-tuning on summaries only if it misses the D32 bar (agreement >= 80%, negative precision and recall >= 70%). **Measured 2026-11-06 (PR #131): 60.4% [52.8, 67.6], fails; the owner chose fine-tuning on summaries (2026-11-07, PLAN D35):** 450 summaries tagged (`scripts/draw_sentiment_train_2026_11_07.py`, 150 per predicted class, a validation slice of 90, never a gold or held-out body); the gold is scored once per run that could ship. Fallback if it still fails on 2026-11-11: the cut goes with sentiment marked as measured. Covers T-127's question on real output. Sub-tasks: T-148c1b, T-148c2b, T-148c3, T-148c4. → D26, D35. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148c.
        - [ ] **T-148c1b** (proposed 2026-11-08, not started, not decided) Namesake companies in the 450 training summaries: the owner's notes name four rows whose `company` is probably a namesake (listed in `docs/gold_datasets.md`, set 6). Measure how many rows of `gold/sentiment_train.csv` carry a company the text does not speak of, with the resolver the draw used (first-word and ticker matches; the same family the faithfulness reading of PR #144 found: first-word and namesake matches), no tagging asked and no mark changed. What to do with them (keep, drop from the training, relabel) is the owner's.
- [ ] **T-148c2b** Scorer hardening for the sentiment scoring (its own small PR, merged before the final scoring; "a rule is applied, not remembered"). **Built (#148, owner's session 2026-11-08):** a Hub model needs a 40-hexadecimal `--revision`; `--redo-role` needs `--redo-reason` (new log column `redo_reason`); the log gains the columns of `retag.SENTIMENT_SCORING_LOG_COLUMNS` after `texts` (the committed baseline row keeps its values, the new ones empty); the report and the row give the agreement for texts over 128 tokens apart; a `final` run gives the paired gain of the agreement and of the negative precision and recall over the **last** `baseline` row (`gold_metrics.paired_stratified_gain`: the report's weights and strata, the same resample for both; refused if the baseline's agreement recomputed from the stored predictions is not the one its row logged) with the verdicts `gain_verdict`, `gain_negative_precision_verdict`, `gain_negative_recall_verdict`; the pin test passes for the candidate only with a `better` gain, or for the baseline. **Added by the owner's choice of the pair input (D35 amended, PR C will write it):** `--pair` scores a model that reads the pair company + text (`src/sentiment_pair.py`, shared by the trainer, the scorer and the stage); a row with no ticker (37 of the 180) is scored by the text-only model (`--company-less-model`, default the pinned one), so the figure is that of the routed system; the row logs `input_format`, the text-only model and how many rows each read (`rows_pair`, `rows_text_only`); the routing rule is `sentiment_pair.reads_as_pair` (a ticker and a company scope), which the trainer and the stage will use too; `--pair` cannot be the baseline. Left: the owner's review; the pin key for a second model is PR C. → D35.
- [ ] **T-148c3** Trainer of D35's recipe for the sentiment model on summaries (base, replay of the earlier sentences, search on the validation slice, 3 seeds, a final run). Acceptance: the replay leaves out the sentences of gold, held-out and drawn-for-tagging bodies (by body hash) and prints how many; one run is timed first and the grid is reduced if the 28 runs do not fit; the tie rule of D35 picks the settings; a batch never mixes summaries with sentences; settings are D35's (batch 16, weight decay 0.01, fp16, linear decay, no class weights, last epoch's model), printed at the start of each run. **The trainer does not score the 180-text gold** (it is scored once, as the `final` run of the published revision, T-148c4); the run's own validation agreement is cross-checked against the scorer's within 1.1 points.
      - Timed 2026-11-07 (GPU idle): about 15 to 30 minutes for the 28 runs, so the full grid fits. **Data fact (checked 2026-11-08):** 26 of the 450 training rows (19 sector, 4 market, 3 economy) have no ticker and a placeholder `[sector]` / `[market / economy]` in `company`, not a company; proposed for the pair loader: exclude them (they belong to the text-only side, which stays the pinned model), and say so in the D35 amendment. Waits for the owner's word (#145 is merged); the date that limits is T-148c4's. → D35. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148c3.
- [ ] **T-148c4** Publish the fine-tuned sentiment model and pin it in `config/pipeline_models.json`, its own PR. The candidate goes to the private Hub repo `gamug/FinBERT-financial-news-summaries` (D35, name accepted 2026-11-07) and is made public only in this PR; the `final` row and the pin name the commit SHA. The scorer's checks are T-148c2b (merged before the final scoring). The pin rule (D35, owner 2026-11-07): `better` pins; `not better`, `not decided` and `undefined` keep the baseline; no pin if the paired gain of negative precision or recall is `not better`. The final run's own validation agreement is checked before publishing; a second `final` scoring only for a defect in the run, never for the figure. The model is scored as the published Hub id and revision (not a local checkpoint). If the baseline stays, there is no pin PR and the repo stays private (the owner decides later). **By 2026-11-09**, with room before the fallback on **2026-11-11**. → D35. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148c4.
      - [ ] **T-148d** Category over the same texts (D29), measured against the owner's gold. → D24.
- [ ] **T-149** *(P5)* Sector detection (GICS sector + sub-industry) for sector bodies;
      summaries of sector, market, economy and `investing_general` bodies feeding `sector_summary`,
      with `MARKET` / `ECONOMY` rows; **sector sub-industry detection accepted at >= 80% agreement
      with the owner's held-out gold (D24)**; its own FR. → P5.
- [ ] **T-150** *(P6, close-out)* Full backfill (old tables to v1, T-131), knowledge-graph export
      contract update (`portfolio_common.news_export`), `SPEC.md` §13 items 19 and 21, both
      architecture artifacts (AI #11), then Work items 15, 17 and 18 to `CHANGELOG.md` (AI #15,
      closing T-129 and T-138b). → acceptance.

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
- [ ] **T-157** *(Windows, T-148a, 2026-11-07; PR #134, `docs/db-topology.md`)* The summaries cut's run lock probed a pid with `os.kill(pid, 0)`, which terminates on Windows. **Done:** the probe uses `OpenProcess` + `WaitForSingleObject`; the lock holds the pid and the process start time; Ctrl-Break stops like Ctrl-C; a `win32`-only test. Decision: no Windows CI job. **Closes when** the owner has resumed the cut once on Windows with a leftover lock and got no refusal. **Open:** taking over a stale lock is not atomic (chunk files are, so only work is duplicated); a lock of this version reads as stale to an older script; closing the console window cannot be caught. → T-148a. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-157.
- [ ] **T-155** *(gold sets, `docs/gold_datasets.md`)* Move the tagged sets from `data/` to `gold/` with builders that write marks and ids only (no news text), refuse unless every row is marked and valid, protect the tracked marks from silent change (`--overwrite-marks`), and let `report_wi18 ... --gold` reproduce the figures.
      - **Done:** label check (`build_segment_check_gold_2026_11_06.py`, 200 rows), extractor sample (`build_faithfulness_gold_2026_11_06.py`, 190 baseline rows + coverage), sentiment (`build_sentiment_gold_2026_11_06.py`, 180 rows, agreement 60.4% [52.8, 67.6]), held-out scope (#142, 200 rows, row hashes in `tests/news_nlp/heldout_gold_row_hashes.csv`).
      - **Left:** a typed reader per gold file in `news_nlp.retag` (today only the column tuples exist); where the only copy of the 180 texts (`data/sentiment_gold_ids.csv`) is kept (nothing enforces a backup); the extractor evidence of T-148b11 (the `units*` marks, the confirmation set, the exempt readings). → `docs/gold_datasets.md`, D31–D33. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-155.
- [ ] **T-154** *(backtest cut, D34)* A provisional output for the bodies of the backtest window (published 2021-12-31 or later), about 2026-11-16: scope from the rules, NER for single and multi bodies, summaries (T-148a), company texts (T-148b, extractor `units`, T-148b10 (9)), sentiment over the texts (T-148c). Written as chunked files under `data/cut/` in the columns approved in T-152, resumable, no schema change; the stored corpus only (it ends 2026-08-12); marked provisional, with the measured figures (rules' scope agreement on the held-out 200, extractor faithfulness, sentiment agreement on the 180) next to it.
      - **Go/no-go 2026-11-12:** the owner's tagging (held-out 200, label check 200, faithfulness texts, sentiment 180) is done and the extractor passes D32; otherwise the backtest team is told with a week's notice. After a go: NER, step 4 and sentiment over the window 11-12 to 11-14, hand-over about 11-16.
      - **Rules fixed beforehand (owner, 2026-11-06, PLAN D34):** a sentiment figure whose interval straddles its bar ("not decided") still goes, with the interval printed; a faithfulness FAIL does not go. If the held-out 200 is not back, the scope figure is reported as "not evaluated" and the cut goes if the other three pass; the faithfulness and sentiment fallbacks are in T-148b10 and T-148c.
      - Not in the cut: the trained scope classifier, the prompt gate and silver labels, category, sub-industry, sector snapshots, active learning, the full backfill. → D34. Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-154.

## Status

**2026-11-05:** the owner fixed the 7-step pipeline order (D25–D30). P4 is split into T-148a–d, the snapshot tables are T-152, the reply to the backtest team is T-153, and the backtest cut for 2026-11-20 is T-154 (go/no-go 2026-11-12). T-127 and T-138 were superseded (moved to `CHANGELOG.md`) (their designs are replaced by D26–D28); T-132, T-131, T-129 and T-138b continue under Work item 18 unchanged.

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
