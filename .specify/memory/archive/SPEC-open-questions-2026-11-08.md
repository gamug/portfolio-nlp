# SPEC.md §13 items as they stood before 2026-11-08

Verbatim text of the resolved or re-scoped items of `SPEC.md` §13 "Open Questions & Risks", which now carry a short form with the same number. Read only when the history is needed.


### Item 1

1. **Sentiment was the weakest stage** (`macro_f1_vs_judge` 0.40): FinBERT's
   whole-article softmax average has no per-company or net-signal reasoning
   the judge applies (`docs/evaluation.md`'s 2026-09-08 follow-up). **Update
   (2026-09-12/13): three designs prototyped and real-data validated**,
   each on its own branch — `PLAN.md` Work item 4 / `TASKS.md` T-030–T-038,
   `docs/evaluation.md`'s 2026-09-13 follow-ups have the full account.
   (a) Entity-scoped chunk-weighting (PR #42): `recall_negative` 0.856 vs.
   the 0.783 pre-change pilot, but `precision_negative` stayed weak
   (0.376). (b) Title-only scoring (PR #43): `recall_negative` 0.533, did
   not beat (a). (c) **Fine-tuning FinBERT itself** on 5,000 LLM-labeled
   sentences from this project's own real corpus (`ProsusAI/finbert`'s own
   training data is 2014 Nordic-company news — a real, measured
   vocabulary/domain gap, e.g. missing "crushed" as a positive idiom),
   published as [`gamug/FinBERT-financial-news`](https://huggingface.co/gamug/FinBERT-financial-news):
   substituted into (a)'s aggregation, `recall_negative` 0.812 (slightly
   below (a), still above the pilot) with `precision_negative` 0.505
   (+13 points), `macro_f1_vs_judge` 0.737 — the strongest, most
   broad-based result of the three, though a manual spot-check afterward
   still found the specific "crushed earnings" idiom gap unresolved.
   **Update (2026-09-13, same day): idiom gap diagnosed and fixed.**
   Mining found the original 5,000-sentence draw contained almost none of
   this idiom family by chance (only 75 hits across the 11k-article eval
   pool) — a coverage gap, not a labeling error. The idiom is genuinely
   two-directional ("stock got crushed" = negative vs. "crushed
   estimates" = positive), ruling out a lexicon-override shortcut; instead
   900 more sentences were mined from the full ~480k-article corpus and
   LLM-labeled covering both directions, 100 held out as a never-trained
   probe to measure the fix directly. Result on that probe: accuracy
   0.750→0.870, recall_positive 0.710→0.903 (the original bug), recall_negative
   0.797→0.932. Re-run on the same 2,000-article downstream pipeline
   comparison: every metric within ±0.01 of the pre-fix version (noise) —
   the fix cost nothing measurable on real traffic. Model updated in place
   at the same Hub repo (new commit, not a new model name). Full account:
   `docs/evaluation.md`'s 2026-09-13 "crushed earnings idiom gap" follow-up.
   **Decision (2026-09-13): (c), chunk-level + fine-tuned FinBERT, selected
   as the production candidate.** First diagnosed the remaining precision
   gap directly (confusion matrix: 40.6% of directional predictions were
   false alarms on judge-neutral articles, dominated 89% by multi-company/
   mixed-signal roundups — a document-structure/aggregation limitation,
   not a sentence-classification error); tested and rejected three
   candidate fixes on real data (confidence threshold, subject-coverage
   gate, zero-shot materiality gate — none discriminated false alarms from
   correct predictions above chance); then re-tested (b) with the
   fine-tuned model instead of base FinBERT as a genuine alternative.
   Result: title-only+fine-tuned wins on precision (0.619/0.670 negative/
   positive) and overall agreement (0.746), but chunk-level+fine-tuned
   wins on recall for **both** directional classes (0.808 negative / 0.801
   positive vs. title-only's 0.574 / 0.528) — a real precision/recall
   frontier, not one design dominating. Chosen because this pipeline is
   deliberately **pessimistic**: a missed real story (false negative) is
   an unrecoverable blind spot for a downstream SEMANTIC score/knowledge
   graph, while a false alarm is a story a downstream consumer can still
   discount — the same "recall over precision" reasoning this document
   already applied to the negative class alone, now generalized to both
   directional classes because chunk-level is the only design with strong
   recall on both. Full comparison, diagnosis, and rejected-fix evidence:
   `docs/evaluation.md`'s 2026-09-13 follow-up. **Merged (2026-09-13)**:
   `src/pipeline.py`'s `SENTIMENT_MODEL` now points at
   `gamug/FinBERT-financial-news`, with `run_sentiment_stage` doing the
   chunk-level entity-scoped weighting from PR #42 (`_text_mentions_subject`/
   `_sentiment_chunk_weights`) — verified end-to-end against real
   production data (a live smoke test scored 3 previously-unscored
   articles correctly, including the "A"/"ON" ticker-collision cases from
   this same investigation). This section's item is now fully closed, not
   just a recorded recommendation.


### Item 2

2. **Four category leaf labels are near-guessing** (`product_innovation`
   0.14, `partnerships_business_dev` 0.16, `capital_shareholder_returns`
   0.20, `leadership_governance` 0.33 accuracy) even after the hierarchical
   redesign. `CATEGORY_CONFIDENCE_THRESHOLD` (0.6) is a reasoned but not
   fully validated calibration point (`news_nlp/taxonomy.py`).
   **Update (2026-09-12): resolved.** The 0.6 threshold calibration
   (`docs/evaluation.md`'s 2026-09-09 "Threshold calibration" follow-up,
   eval_run 19-21) took three of these four slugs to 0.61
   (`product_innovation`), 0.51 (`partnerships_business_dev`), and 0.53
   (`leadership_governance`) accuracy — the numbers above are stale,
   pre-calibration figures, kept here only as the historical record (see
   `PLAN.md` Work item 5). `capital_shareholder_returns` recall came in at
   0.293 (precision 0.074, a separate, not yet root-caused confusion with
   its own hierarchy group-mates) — still weak, but no longer
   near-guessing across the board. One related thread stays open:
   `other`'s own precision (0.474) — see `docs/evaluation.md`'s "The
   `other` bucket: a precision problem of its own" — tracked as
   `TASKS.md` T-041, low priority. **Update (2026-10-14)**: T-041 is
   superseded by §13 item 19 / `PLAN.md` Work item 15.


### Item 10

10. **`c_summary`'s `mean_coverage` is weak (3.02/5) despite the stage's
    strong headline metric** (`mean_faithfulness` 4.87/5,
    `pct_with_hallucination` 5.4%) — a terse/extractive tendency of
    `distilbart-cnn-12-6`, not a correctness problem
    (`docs/evaluation.md`'s 2026-09-08 baseline notes). `c_summary` was
    also, like NER, suspected of the same full-article-vs-lead-cap
    eval-sampling mismatch — **confirmed and fixed 2026-09-14**: 10.0%
    of `article_summary` rows have `body_text` past the judge's cap,
    every one of them multi-chunk; `c_summary` added to
    `_UNCAPPED_STAGES` (`src/news_nlp/eval/sampling.py`) the same way
    sentiment/NER were. **Re-run post-fix the same day** (`eval_run`
    34): `mean_coverage` improved 3.02→3.64/5 (HT), but
    `pct_with_hallucination` rose 5.4%→8.5% — a previously-invisible
    hallucination gap the cap had been masking, not yet independently
    confirmed by a rationale audit. Both metrics concentrate in
    `num_chunks >= 3` articles (coverage 2.86/5, faithfulness 4.23/5) —
    see `docs/evaluation.md`'s 2026-09-14 follow-ups for full numbers
    and the sampling-design confound this comparison carries. A first
    candidate fix (raising `SUMMARY_MIN`/`MAX_OUTPUT_TOKENS`) was
    tested via a matched-pair experiment and **rejected**: coverage
    moved +0.20 but `pct_with_hallucination` more than doubled
    (15.0%→35.5%) and the actual weak `chunks >= 3` tier barely
    moved — a bad trade, not a fix. Decided 2026-09-14: **accept, no further fix** — a
    deliberate completeness-vs-correctness trade, the mirror image of
    sentiment's recall-over-precision "pessimist model" call (§9's
    sentiment row) — a fabricated detail is c_summary's unrecoverable
    failure mode, an omitted one is the recoverable, visible one.
    Doesn't close off a narrower reduce-pass-specific fix later if
    `chunks >= 3` coverage is judged unacceptable on its own. Full
    reasoning — `PLAN.md` Work item 6 / `docs/evaluation.md`'s
    2026-09-14 "Decision" follow-up.
    `sector_summary`
    itself stays out of scope for this item (and for eval generally):
    it's deterministic composition, only its `intro_text` sentence is
    generative. That sentence, though, runs through the exact same
    `SUMMARY_MODEL` and currently has **no evaluation at all**, not even
    a simple one — a gap in its own right, closed via a narrow
    faithfulness-only check (not a full new eval stage the size of
    `c_summary`'s) — `PLAN.md` Work item 6 step 4 / `TASKS.md`
    T-054–T-057.


### Item 11

11. **`run_ner_stage` has no batching** (`src/pipeline.py`) — one chunk
    through the model per forward pass, unlike `run_category_stage`
    (`CATEGORY_BATCH_SIZE=8` articles' premise/hypothesis pairs pooled
    into one call) and the summarization stages
    (`SUMMARY_BATCH_SIZE=4` via `hierarchical_summarize_batch`). Not a
    documented trade-off anywhere in the codebase — genuinely
    unaddressed, not a deliberate design choice. Measured impact: the
    2026-09-12 T-025 resample (`docs/evaluation.md`) processed 20,000
    articles unbatched in ~15 minutes (~22 articles/sec) on the project's
    GPU (6GB VRAM, well under budget the whole run) — a full-corpus
    backfill of the ~439,000 remaining pre-fix articles (§13 item 6's
    open T-022 question) would take roughly 5.5x that, over 5 hours,
    single-chunk-at-a-time, on hardware with headroom to go faster.
    **Batching implemented 2026-09-12** (`TASKS.md` T-060/T-061) — a new
    `NER_BATCH_SIZE` constant (starts at 8, `CATEGORY_BATCH_SIZE`'s value
    as a first guess) flattens every article's chunks in a batch into one
    padded tokenizer call + one forward pass (`pipeline._ner_batch`),
    regrouped back per article afterward; a parity test
    (`test_batched_and_per_article_ner_processing_produce_identical_entities`)
    confirms batched and one-article-at-a-time processing produce
    identical `article_entities`. **Still open**: T-062, empirically
    tuning `NER_BATCH_SIZE` against the 6GB VRAM budget and measuring the
    real throughput gain on a GPU — not done as of 2026-09-12 (no GPU
    access in the environment that shipped the batching code); the
    starting value was untested against real hardware. **No longer
    GPU-blocked as of 2026-09-14** — this sandbox gained CUDA access
    (used for the Work item 6 summarization experiments the same day),
    so T-062 is now actionable, just not yet done. `PLAN.md` Work item 7
    has the full detail. (`run_sentiment_stage` has the identical
    unbatched shape and is likely worth the same treatment later, but is
    out of scope for this item — not raised here as its own numbered
    question to avoid scope creep beyond what was asked.)


### Item 12

12. **`sector_summary`'s `intro_text` had a real, sizable faithfulness
    gap — up to 50.2% hallucination rate — resolved 2026-09-14.** Unlike
    `c_summary` (which summarizes real article text), `intro_text` was
    asking `distilbart-cnn-12-6` — a news-article summarizer — to turn a
    short, synthetic stats-only seed sentence (`build_sector_intro_seed`)
    into prose, a task shape it was never trained on. First-ever eval for
    this text (`PLAN.md` Work item 6 step 4, `TASKS.md` T-054–T-057,
    `eval_run` 34, full 3,628-row population, 2026-09-14):
    `mean_faithfulness` 4.05/5, `pct_with_hallucination` 42.2% —
    confirmed as a real pattern by reading flagged rows, not an artifact
    of the metric: 64% were a fabricated source attribution
    (`"...according to CNN.com's weekly Newsquiz"`, `"...according to
    analysts"`) the model invented, the rest mostly a self-contradicting
    repeated-percentage generation artifact. Re-checked against fully
    fresh sentiment data the same day to rule out staleness as the cause:
    got *worse* (3.83/5, 50.2% hallucination, 70% attribution-fabrication)
    — confirming the pattern was independent of the underlying data.
    **Fixed same day**: `run_sector_summary_stage` (`src/pipeline.py`) no
    longer runs `build_sector_intro_seed`'s output through
    `SUMMARY_MODEL` — that seed was already a complete, fully-grounded
    sentence, so `intro_text` is now that seed verbatim. Zero
    hallucination risk by construction, not mitigation; this stage no
    longer loads a model or touches the GPU at all.
    `SECTOR_SUMMARY_FORMAT_VERSION` bumped 2→3 so the existing 3,444
    pre-fix rows self-heal to the new template the next time the stage
    runs (this project's existing designed mechanism, FR-006 — no
    separate backfill script). Full numbers, example rows, and the
    persistence-check methodology in `docs/evaluation.md`'s 2026-09-14
    follow-ups.


### Item 13

13. **The Models evaluation artifact section documents *how well* each
    model performs, not *why that model was chosen* over a plausible
    alternative.** The 2026-09-14 artifact reorg centralized every
    stage's accuracy numbers into one section, closing the "scattered
    metrics" problem — it did not add model-selection reasoning, which
    today is either absent (category's, NER's, and `c_summary`'s base
    architecture choice) or only covers *which variant of the same
    model family* (e.g. chunk-level vs. title-only FinBERT weighting),
    not *why that family at all*. **New, priority, pending (2026-09-14)**
    — `PLAN.md` Work item 8 / `TASKS.md` T-064–T-069.


### Item 14

14. **The sentiment fine-tuning data is class-imbalanced and was never
    deliberately balanced.** The 5,800-sentence training pool behind
    `gamug/FinBERT-financial-news` (base draw + idiom-augmentation round)
    is 3,256 neutral (56.1%) / 1,321 negative (22.8%) / 1,223 positive
    (21.1%) — not a target ratio anyone chose, but a byproduct of drawing
    sentences from the eval harness's confidence-stratified sampling pool
    (stratified on prediction confidence, not on label ratio) plus real
    financial news skewing neutral/factual. The only mitigation in place
    is class-agnostic: `train_sentiment.py` selects the best checkpoint by
    macro F1 (equal per-class weight at *evaluation* time), and stratifies
    the train/validation/test split per label so those splits aren't
    *more* skewed than the source — neither touches the underlying
    training-set skew itself. **New, priority, pending (2026-09-14)** —
    `PLAN.md` Work item 9 / `TASKS.md` T-070–T-075.


### Item 15

15. **The pipeline and evaluation architecture was never a formal spec —
    it grew as a pragmatic solution to a real, immediate necessity.** Four
    ML stages (`pipeline.py`) and the LLM-as-judge harness
    (`news_nlp/eval/`) were each built incrementally, session by session,
    as concrete gaps surfaced (a missing accuracy baseline, a sampling
    bias, a class-imbalance fix) — not designed up front against a stated
    architectural pattern. The result works and has been validated
    end-to-end (Work items 1-9), but it's four independent sets of
    module-level functions rather than a shared, extensible structure, and
    the eval harness has real, concrete friction this session's own
    sentiment-candidate work exposed directly: no way to run two
    experiments' judged data in the same store (`scripts/
    resample_sentiment_v{3,4,5}_2026_09_15.py` each needed their own
    scratch database copy — PLAN.md Work item 9), no persisted confusion
    matrix, no ROC, and no mechanism to avoid re-spending judge-LLM tokens
    on a row already tagged under the same experiment. **New, priority
    #1, pending (2026-09-16)** — formalize the existing shape into a
    Feature/Train/Inference (FTI) class hierarchy shared across all four
    ML stages, move `sector_summary` fully into its own non-FTI module,
    and redesign the eval schema/module around the friction points above
    — FR-011–FR-016, `PLAN.md` Work item 10, `TASKS.md` T-082 onward.
    **Done 2026-09-18.**


### Item 16

16. **Running a real model experiment still requires hand-chaining several
    independent, one-off commands, with no single place that declares an
    experiment's full configuration before it runs.** Item 15's FTI/eval
    redesign (Work item 10) fixed the *architecture* underneath each
    stage and the eval harness's own schema/reuse friction — it did not
    change how an experiment is actually *run*. Walking through the full
    historical sentiment-candidate sequence (Work item 9: dataset prep →
    train → offline eval → downstream eval → publish) command by command
    (2026-09-18) surfaced the gap directly: several of those commands
    mutate shared DB tables in place and need a manual restore step
    afterward (`scripts/resample_sentiment_v4_2026_09_15.py` +
    `scripts/restore_sentiment_after_v4_eval_2026_09_15.py`); nothing
    captures an experiment's full configuration (pretrain or not, dataset,
    train/test split + stratify strategy, LLM-judge sample count) in one
    reproducible, validated place before it runs. **Resolved (2026-09-18)**
    — a JSON `ExperimentSpec` schema (`src/experiment.py`), generic across
    all four ML stages, plus one command (`cli/run_experiment.py`) that
    runs an experiment end to end from it, reusing `run_eval` verbatim;
    backfilled for every real historical experiment
    (`experiments/*.json`, three disclosed gaps in `experiments/README.md`
    rather than silently omitted) — FR-017, `PLAN.md` Work item 11,
    `TASKS.md` T-096–T-103.


### Item 17

17. **`tests/` was never covered by this project's own mypy gate.**
    `.code_quality/mypy.ini`'s `files = src, apps, cli` excludes it, so a
    real, systemic type-annotation drift went undetected: `tests/news_nlp/
    conftest.py`'s `conn`/`two_tier_conn` fixtures (and the
    `write_stage_predictions`/`seed_article` helpers) are typed
    `sqlite3.Connection`, but at runtime they construct and return
    `NewsNlpDatabase` (`db_module.connect()`/`connect_pipeline()`) — a
    composition wrapper around a `sqlite3.Connection`
    (`portfolio_common.db.engine.Database`), not a subclass of it, so the
    two types are genuinely unrelated to mypy even though `Database`
    proxies `execute`/`executemany`/`executescript`/`commit` with matching
    signatures (why passing the "wrong" type has always worked fine at
    runtime). ~18 test files copied that wrong annotation into their own
    test function signatures. Surfaced directly (2026-09-18) running mypy
    with an explicit path argument — which overrides `mypy.ini`'s own
    `files` scope — while landing T-097, purely to sanity-check a docstring
    claim rather than using the project's own documented no-argument
    command: 208 errors, `git stash`-confirmed pre-existing (identical
    count on `master`) and behaviorally inert (every affected value
    genuinely is a `NewsNlpDatabase` throughout — a type-hint precision
    gap, not a runtime bug). **Resolved (2026-09-18)** — `conftest.py`'s
    fixtures/helpers retyped at the source (`seed_article` turned out to
    need a genuine `sqlite3.Connection | NewsNlpDatabase` union, not a
    straight retype — it has real callers of both kinds), the fix cascaded
    through all ~18 affected files, a previously-undisclosed 19th file
    (`test_eval_store.py`, an unrelated `**dict` kwargs-unpacking gap) found
    and fixed in the same pass, and `mypy.ini`'s `files` widened to
    `src, apps, cli, tests` (`scripts/` deliberately excluded — one
    disclosed, unaddressed exception, `scripts/
    mine_idiom_sentences_2026_09_13.py`). `uv run mypy --config-file=
    .code_quality/mypy.ini` now reports zero errors across 67 files;
    `uv run pytest` unchanged at 279 passed (zero assertion changes — every
    fix was type-only) — NR-007, `PLAN.md` Work item 12, `TASKS.md`
    T-104–T-108.


### Item 18

18. **The judge-verdict reuse mechanism (item 15/FR-015, T-091) could
    silently pair a stale verdict with a changed prediction under an
    unchanged `experiment` label.** `find_verdict_json` keyed reuse on
    `(article_id, task, experiment)` alone; `experiment` doesn't prove
    "same model" — `run_experiment` (item 16/Work item 11) re-resolves
    `candidate_model` to a fixed local checkpoint path
    (`train_sentiment.py`'s `OUTPUT_DIR`/`OUTPUT_DIR_WEIGHTED`), so
    re-running the same `ExperimentSpec` after a retrain reuses that exact
    same label even though the weights — and the prediction — genuinely
    changed. Surfaced directly (2026-09-19) from a user request to check
    whether the reuse mechanism was actually working; confirmed by a
    direct, standalone reproduction before any code changed: two
    `run_eval` calls sharing `candidate_model`/`candidate_revision`, a fake
    model predicting `"positive"` then `"negative"` for the same article —
    run 2's `eval_inference.prediction_json` correctly recorded
    `"negative"`, but its `eval_verdict.verdict_json` still read
    `ideal_label: "positive"` (run 1's verdict, reused with zero judge-LLM
    calls), so run 2's own aggregated metrics were computed against a
    verdict describing a different prediction than the one actually
    produced that run. **Resolved (2026-09-19)** — `find_verdict_json`
    additionally requires the candidate's paired `eval_inference.
    prediction_json` to match; `runner._resolve_verdicts` threads the
    current item's own serialized prediction through. Regression coverage
    at both the unit level (`test_eval_store.py`) and end-to-end
    (`test_eval_runner.py`, directly mirroring the standalone reproduction)
    fails without the fix and passes with it. A separate, smaller,
    disclosed finding from the same investigation: `docs/evaluation.md` had
    one stale 2026-09-15 comment calling `--run-name` "cosmetic only,"
    contradicted by the real T-090 behavior and by two other correct
    passages in the same doc — corrected alongside this, not itself a code
    bug — FR-015 (tightened), `PLAN.md` Work item 13, `TASKS.md`
    T-109–T-110.


### Item 19

19. **Category's `other` label covers 42.7% of the corpus and means two
    different things.** `article_category.label = 'other'` is set on
    195,993 of 459,112 articles (2026-10-14, RESULTS store); `other`'s own
    precision is 0.474 (`docs/evaluation.md`, eval_run 21). Only 8.5% of
    those rows were rejected by level 1's `CATEGORY_GROUP_FLOOR`; 91.5%
    cleared level 1 and then failed level 2's 0.60 threshold, many of them
    narrowly (~70k have a best leaf score of 0.5–0.6). The label conflates
    (a) articles that are not about the ticker's company at all — a
    relevance problem rooted in the crawler's ticker tagging (confirmed in
    `portfolio-data-mining`'s `BaseConnector._company_pattern`: a ticker **or
    the first word of the company name** matching anywhere in a URL slug,
    with a hyphen counting as a word boundary), where eight symbols that are
    ordinary English words (`ON`, `A`, `ARE`, `IT`, `ALL`, `HAS`, `TECH`,
    `NOW`) hold 39.0% of the corpus and companies cut to a generic first word
    (`State`, `Target`, `Live`, `Dollar`) add more — with (b) articles about the company that no
    dimension clears the bar for, a calibration problem. A hand-read
    30-title sample put roughly two thirds in (a) and one fifth as clear
    misses within (b); that is a planning estimate from a small sample, not a measurement
    (constitution AI behavior #5), and is what Work item 15's first step
    replaces. Until resolved, downstream consumers must not treat
    `label = 'other'` as "verified no category" (already stated for the
    precision gap in `docs/evaluation.md`), nor assume every article is about
    its ticker's company. **Decided (2026-10-14)**: *if* a relevance gate is built — which
    `TASKS.md` T-117 decides by measuring what irrelevance remains after the
    crawler fix — articles it rejects get a new `not_relevant` label (11
    labels; `label` is unrestricted `TEXT`, so no table migration); and the
    entire classification is re-run once the new rule is tested and approved.
    Decided (D4, 2026-10-22, recommended defaults): the existing results for articles the
    crawler fix removes are archived, not dropped, and only the affected `sector_summary`
    rows are deleted (`TASKS.md` T-131).
    The wrong ticker tagging is fixed at its source first
    (`portfolio-data-mining` Work item 10, decided 2026-10-14; its quarantined `review`
    pairs are decided here, `TASKS.md` T-132), then
    re-measured here. Absorbs the open thread of item 2 (`TASKS.md`
    T-041). Requirements (FR-019 onward) are added in the PRs that ship each
    behavior — the relevance gate if built, the calibrated decision rule, the
    opt-in relevance judge, the results cleanup — per this document's own
    rule. The calibrated decision rule is fitted on a stratified, held-out judged sample
    (`TASKS.md` T-121, T-141; spend decision D10, T-142). `PLAN.md` Work item 15,
    `TASKS.md` T-116–T-134 and T-139–T-142 (T-135–T-138 are Work items 16 and 17).
    **Update (2026-10-22): the upstream fix has landed and resized the problem.**
    `portfolio-data-mining`'s re-tag (2026-10-15) classed 481,332 pairs as `accept`
    104,614 / `review` 51,295 / `reject` 325,423; the 459,112 articles classified here
    split 93,224 / 49,645 / 316,243 on the 2026-10-15 figures and 93,330 / 49,960 /
    315,822 measured on the real store on 2026-10-22, so about 80% of the RESULTS store
    is for articles no longer in the corrected corpus. The correct rate of the `review` class is not measured, and
    the counts keep moving (2026-10-21: 109,982 / 52,039; 98,462 `ok` articles, 63.4%
    published before 2022). The category classifier and its judge never see the
    company, so wrong tags bias the *measured* numbers through sample composition
    rather than the model's reading. A population check on 2026-10-22 found `other` is
    37.9% of confirmed and 43.9% of rejected articles (rejected articles are 68.8% of
    the corpus and 70.8% of `other`), so wrong tagging explains at most about 11% of
    the 42.7% (stratified by source and period, in `docs/evaluation.md`); calibration
    is the main remaining lever (first measurement 2026-10-22 in
    `docs/evaluation.md`: `other` was judged correct far more often on rejected than on
    confirmed articles; the complete metric set per upstream class, T-133 done, is in
    `docs/evaluation.md`); sentiment
    does use the company and is affected directly (T-127). The quarantined `review`
    pairs get the verdicts `about` / `mention` / `other` / `unclear` (D5), never acted
    on in doubt, in either direction. Owner decisions D5–D9 were settled at their defaults on 2026-10-22 (T-139).
    **Re-scoped 2026-10-28 → `PLAN.md` Work item 18** (triage per unique body; decisions
    D11–D24 in `TASKS.md` T-143; the stage order, D25–D30, in T-151: scope, NER for single and
    multi bodies, summary of non-multi bodies, company text of multi bodies, sentiment and category
    over those texts, company and sector snapshots). No FR is added until a behaviour ships.
