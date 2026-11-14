# CHANGELOG.md — `portfolio-nlp`

Legacy record of **closed** work items, moved here verbatim from
`.specify/memory/TASKS.md` so that file carries only open work. A work item is
closed once every task in it is checked, or explicitly superseded/moved elsewhere.
Task IDs are stable and never reused. A closed work item's plan and acceptance
criteria are in `.specify/memory/archive/PLAN-closed-work-items-2026-11-08.md`
(`PLAN.md` keeps its heading and one line). Ordered by work item number; the section "Tasks closed while their work item is still open" at the end is grouped by work item and holds done tasks, not whole items.

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

## Work item 5 — Category: hold the line on the hierarchical fix (closed 2026-10-14; T-041 superseded by Work item 15)

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
- [x] **T-041** **Superseded 2026-10-14 → Work item 15** (the `other` bucket
      turned out to mix an upstream relevance problem with the calibration
      problem described here; both are now tracked there; no work done under
      this ID; Work item 15's calibration step, T-121, now plans a stratified
      judged sample of 3,300 calls (cap 3,800) with a fit/report split, a dedicated judge and
      sampler (T-141) and its own spend decision (T-142), 2026-10-22; the owner approved D10 at a 3,800-call cap, chose re-label for D2b and confirmed the T-117 thresholds and D4 at the recommended defaults the same day).
      Original text: `other`'s own precision (0.474) is still an open
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

## Work item 16 — Check the data-mining schema version before reading (closed 2026-10-22, #109; D8 default: build it here)

SOURCE (`urls.db`) is stamped with `PRAGMA user_version` (5 today; additive; a
documented rollback sets 5 back to 3). This repo reads it on trust, and T-132 will
read a second table. A read-only guard at the single choke point, minimum 3, fail
on lower and on 0, warn once on higher, never cached, plus a column guard. →
`PLAN.md` Work item 16. Task order: T-135, T-136, then T-136b (close-out).

- [x] **T-135** *(done 2026-10-22, `feat/source-schema-guard`)* Implement `check_schema(source, minimum, columns)` (name open) in
      `news_nlp.db`, called by `connect_pipeline` / `attach_source` before any query.
      `source` is a path or an open read-only `sqlite3` connection, so it has no
      dependency on `NewsNlpDatabase` and scripts (the 2026-10-22 measurement script)
      can call it too; `attach_source` is one caller, not the only one.
      - **Inputs:** the constant `MIN_SOURCE_SCHEMA_VERSION = 3` copied into this repo
        (never import `data_mining`), and the columns this repo reads from SOURCE
        **today** (`articles.id`, `fetch_status`, `body_text`, `company`, `ticker`,
        `pub_date`, plus what `copy_row_lean` copies). The helper takes the column list
        as a parameter: T-132 adds `articles.prev_fetch_status` and
        `discovered_urls` (`id`, `url`, `ticker`, `match_class`, `match_reason`) to it in
        its own PR, so no run fails on a column nothing reads yet. The minimum 3 is a
        behavior change (a version-2 file works today); the real `urls.db` is at 5, so
        nothing is lost, and T-136 says it.
      - **Scope:** the guard runs only when a SOURCE is attached (SOURCE distinct from
        RESULTS). A single-file run (SOURCE and RESULTS the same path, RESULTS never
        stamped) and the serving/query endpoints that need no SOURCE do not run it. The
        eval scratch run (`eval/candidate.py`) runs it on the real SOURCE before cloning.
      - **Steps:** read `PRAGMA user_version` through a `mode=ro` open; fail on 0 and on
        lower with the path, the version found, the minimum and the action ("start the
        collector or the extractor so it migrates the file, or point to the right
        file"); warn once per process on a version above the latest seen, 5 (a module-level flag; each new process warns again; 3, 4 and 5 do not warn); then `PRAGMA table_info` on the tables in the list and fail on
        a missing column (the parameterized list) even if the number passes; never cache
        between opens; write nothing.
      - **Done when:** tests on small generated files pass: version 0 and 2 fail; 3, 4, 5
        pass; 6 passes with one warning (3, 4, 5 none; two opens in one process give one warning; a second
        process warns again); version 3 without `body_text` fails on the column guard
        (the `prev_fetch_status` case belongs to T-132's Done-when); a file re-stamped 5 → 3 between two opens passes both times; a
        WAL file with a live writer opens read-only; the file hash is identical before
        and after. CLI exits non-zero with the message. `POST /pipeline/run` (the only API
        path that attaches SOURCE; it answers 202 and runs in a background worker) runs the
        guard **synchronously in the handler, before** the worker and before the run slot
        is taken, and answers 503 carrying the message (a configuration error, not a
        client error); tested: 503, no 409 afterwards, status not left "running". The
        no-SOURCE endpoints are unaffected (tested). **The existing suite stays green:**
        a shared fixture helper in `tests/news_nlp/conftest.py` stamps the version and
        creates the guarded columns on the SOURCE fixtures (today they are unstamped,
        i.e. 0), and tests that need an old or unstamped file build one explicitly. The
        shipping PR extends FR-007's acceptance (SOURCE is read-only and checked before
        use), per SPEC §14.
      - **Also:** `scripts/measure_corrected_corpus_2026_10_22.py` drops its own
        version check for this guard (its constant said 5; upstream's table puts
        `match_class` at 2 and `prev_fetch_status` at 3, so the minimum is 3). Any new
        SOURCE table or column read (T-132, T-137, …) adds itself to the guard list in
        its own PR. Test: the helper runs on a bare `mode=ro` connection with no
        `NewsNlpDatabase`.
      - **Depends on:** D8 (decided, T-139). **Blocks:** T-132.
- [x] **T-136** *(done 2026-10-22, in `docs/db-topology.md`, "How a pipeline run uses both" step 3)* Docs: one line in `docs/db-topology.md` with the minimum version, why a
      minimum and not an exact match, and that `sitemap_entries`, `sitemap_crawls` and
      `discovery_progress` are not part of the contract. Depends on T-135.
- [x] **T-136b** Close-out of Work item 16: annotate `SPEC.md` §13 item 20 / §14 as
      built, add the FR-007 acceptance extension if T-135's PR did not, reconcile both
      architecture artifacts (never rename; constitution AI behavior #11), then move the
      work item to `CHANGELOG.md` (AI behavior #15). Depends on T-135, T-136.
      **Done 2026-10-22** — SPEC item 20 and its §14 row annotated as built (#109);
      FR-007's acceptance was extended in #109 itself; Portfolio NLP and Portfolio
      Thesis reconciled and republished, titles unchanged; the work item moved here.

## Tasks closed while their work item is still open

Done tasks moved verbatim from `TASKS.md` on 2026-11-08 (their work item still has open tasks, so the section
above does not hold it). Grouped by work item; task IDs unchanged.

### Work item 3 — NER

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

### Work item 4 — Sentiment

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

- [x] **T-033** Re-ran the eval for each candidate and added dated
      follow-up entries to `docs/evaluation.md`; updated `SPEC.md` §13
      item 1 and §9's sentiment baseline row with the results. → step 4 /
      `PLAN.md` Work item 4 acceptance criteria. **Done 2026-09-13** for
      all three candidates; final "which one ships" decision explicitly
      left open for the repo owner rather than resolved by these tasks.

### Work item 7 — NER

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

- [x] **T-063** Document the new constant and batching design
      (`docs/modules/news-nlp.md` and/or a `pipeline.py` comment, matching
      how `CATEGORY_BATCH_SIZE`/`SUMMARY_BATCH_SIZE` are documented) and
      update `SPEC.md` §13 item 11 to reflect the shipped state. **Done
      2026-09-12** — see `NER_BATCH_SIZE`'s comment in `src/pipeline.py`,
      the NER bullet in `docs/modules/news-nlp.md`, and `SPEC.md` §13
      item 11. → `PLAN.md` Work item 7 acceptance criteria (all but the
      "no VRAM regression" / "measured throughput" criteria, which need
      T-062).

### Work item 15 — Category

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

### Work item 17 — Articles that name more than one company

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

### Work item 18 — Triage-first pipeline

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

      - [x] **T-146a** *(done 2026-10-29: `docs/labelling-spec-v2.md`, the owner's calls answered; PR #118, merge #117 first because the spec cites its script and CSVs; #118 then has nothing left to merge and is closed as merged through #117)* Labelling spec v2: scope (six), companies, sector, D15 labels, with examples
        for the owner (a guide like `category_labels_guide.csv`). Gold v2 and the held-out gold
        carry, per body: scope; the companies the body is about; a D15 label per company; the GICS
        sub-industry for sector bodies (D24 tests the last two). Tagging effort per body is
        therefore larger than scope only (PLAN WI18 Q7).

        - [x] **T-148b1** `news_nlp.company_texts` (`label_segments`, `split_company_texts`, `segment_text`), the one-segment-per-company
          parser, with tests *(2026-11-05; a segment naming two companies at its start is `ambiguous`: no label, nothing inherited past it; a body is structured with at least 4 segments, 50% of them led by a company and 3 companies, the rule that finds 2,867 of the 81,961 unique bodies)*.

        - [x] **T-148c1** (written 2026-11-07: `scripts/build_sentiment_train_2026_11_07.py`, `gold/sentiment_train.csv`; tried on the real files with trial marks; **Data migrated 2026-11-08: `gold/sentiment_train.csv` holds the owner's 450 marks (232 positive, 113 negative, 105 neutral; 360 train, 90 validation), checked by the builder and pinned by a test; the label is anchored to the tagged company (neutral when the company is not in the facts the text presents), see `docs/gold_datasets.md` set 6; round 1 of the review: `ticker` and `company` added to the file and protected by the builder, the date of the rule and the set 5 comparison recorded, agreement by `rule_scope` measured; round 2: counts corrected, the rest of the disagreement recorded as unexplained; round 3: the notes read into four kinds by the builder (no bound claimed), the stale "by design" sentences rewritten, the namesake rows listed, T-148c1b proposed**) Builder of the marks: reads `data/sentiment_train_to_tag.csv` and `data/sentiment_train_ids.csv`, stops unless every row is marked, the text hash is the one drawn and the ids match; writes the marks, the slice and the ids (no news text) to `gold/sentiment_train.csv`. **Today, 2026-11-07** (the owner expects to have all 450 marked today); written and tested on the untagged file.

        - [x] **T-148c2** (written 2026-11-07: `scripts/score_sentiment_gold_2026_11_07.py`, every run in `gold/sentiment_gold_scoring_log.csv`; the pinned model reproduces D32) Scorer of a candidate model on the 180-text gold: takes a model path or revision, reads the texts of `data/sentiment_gold_ids.csv`, scores them as the pipeline's stage reads a text, writes the predictions in the format `report_wi18 sentiment --gold` reads, and gives the figures for 2022 on and before apart (D35). **Today, 2026-11-07.**

### Work item 18 — decision records moved from `TASKS.md` (2026-11-08)

T-143's text (D11–D24) now lives in `PLAN.md` Work item 18, "Decision record"; T-151 is a pointer to D25–D34 there.

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

- [x] **T-151** *(decision record, 2026-11-05, the owner)* **D25–D30**: the order of the stages,
      sentiment over the summary or company text, step 4 without an LLM, the snapshot tables, category
      on the same texts, the held-out 200. Full text: `PLAN.md` Work item 18. The four open questions were answered the same day as D31–D34 (all years disclosed; the bars of the extractor and of sentiment; no recall floor under 40 gold bodies; the backtest cut).

### Work item 18 — T-159, the count of the properly fetched corpus (moved from `TASKS.md`, 2026-11-08)

- [x] **T-159** *(owner's decision D36, 2026-11-08; done, see below)* **Count the properly fetched corpus.** Read-only, no model and no spend: from the SOURCE `articles` (+ `discovered_urls`), the unique bodies (the same hash the triage uses) with a title and a `body_text`, by pair class (`accept`, `reject`, `review`, none), by whether the same body is also `ok` under another ticker, and by year (the cut's window from 2021-12-31). Answer: how many bodies the pipeline would gain over the 81,961 `ok` ones, how many per company by the tags and by the content. The `thin_content` rows (1,785) are out of the properly fetched corpus (owner, 2026-11-08) and are counted apart. The cut stays on the `ok` window (owner, 2026-11-08); the result is what D19's backfill scope after the delivery and T-150 are decided on; until then every figure says which corpus it is of. → D36.
      **Done 2026-11-08** (`scripts/count_fetched_corpus_2026_11_08.py`, test `tests/news_nlp/test_count_fetched_corpus.py`; result in `PLAN.md` D36; `thin_content` was 1,785 rows when first measured, 1,652 with a title and a body): `ok` 81,961 unique bodies (29,113 in the window, the cut's figures, which checks the count); quarantined 260,845, of which 227,921 not also `ok`; properly fetched 309,882 (3.8x) and 73,771 in the window (2.5x); quarantined-only median 463 words, 7,503 under 100 and 54,428 under 300; rows with a title and a body by `fetch_status`: ok 98,462, quarantined 374,419, thin_content 1,652 (apart), failed 578 (not counted); 505 companies with an `ok` body by tag, 3 only quarantined.

### Work item 18 — T-148c3b, the second set of sentiment training marks (moved from `TASKS.md`, 2026-11-10)

- [x] **T-148c3b** *(built 2026-11-10, PR #154; the owner's confirmation of the slice split is T-148c3f)* The second set of the owner's marks, `gold/sentiment_train_2.csv` (405 rows, batch "2": 270 `train`, 135 `validation`; the slice split was chosen by the assistant, not by the owner: say so if it should change), read with the first by `--final-all` and by the cross-validation (`MARK_SETS`); 855 rows, 809 read as a pair (46 sector, market and economy rows stay with the text-only model). Figures and limits: `docs/sentiment_training.md`.

### Work item 18 — T-148c3g, the sentiment model on every row (moved from `TASKS.md`, 2026-11-10)

- [x] **T-148c3g** *(done 2026-11-10; the owner chose the existing run as the candidate)* The sentiment pair model trained on the whole dataset: `--final-all`, 809 rows + 341 replay, seed 4, one run (MLflow `final_all__e2_lr3e-05_s4`), not repeated, so there is no selection among runs. Published private as `gamug/FinBERT-company-sentiment@9a7b1eb3` with the stack file (T-169, #164). The gold score is T-148c4.

### Work item 18 — T-176, the faults of the cut (moved from `TASKS.md`, 2026-11-11)

- **T-147b, done part (moved from `TASKS.md`, 2026-11-11, owner's word):** refit and checkpoints (`scripts/train_scope_final_2026_11_11.py`); held-out 200 scored from the saved weights (`score_scope_heldout_2026_11_11.py`): 83.6% [78.1, 88.9], gain +16.5, `out_of_scope` recall 53%; uploaded **public** to `gamug/deberta-v3-news-scope-nli@d14a4d20` and pinned in `config/scope_model.json` with the step revision `scope-nli-8ff21e50796b`. 

- [x] **T-176** *(from #168; done, owner 2026-11-11)* Cut faults: `units4` is the cut's default (`--reading units` as before; `docs/extractor_training.md`): texts with several companies 162→23, heading-only 6→0, 97.4% vs 97.6% on the owner's marks; 18 of 331 companies lose their text, which the NER step (D39 (iii)) recovers. The full cut run is the owner's to start. T-177 built as a candidate (`--reading units5`, #172; `units4` stays the default; blind sample of 30 changed texts: 29 faithful; no evidence of a faithfulness difference: 97.8% vs 97.1% for `units4` on the marked texts, +0.6 [0.1, 1.5] comes from added texts, not better). Marks of the 7 unread texts (2026-11-11): 6 yes, 1 no (analyst-only counts faithful, owner's rule), `units4` 304 of 313 (97.1%). Review of #169 (2026-11-11): the real command (`--step cut --limit 500`) gives 3,134 sub-news from 500 bodies, 0 failed, determinism 0 of 30; of the 23 texts that still hold several headings, 10 are one company named in other companies' lines and 13 a third fault (T-177).

### Work item 18 — T-180, the Hub card of the scope model (moved from `TASKS.md`, 2026-11-11)

- [x] **T-180** *(Build 1, D40; #170's review)* **Republish the Hub card of `gamug/deberta-v3-news-scope-nli`:** its README line 29 names `docs/scope_training.md` (now `scope_training.md`). Needs the owner's word and a write `HF_TOKEN`; a card commit changes the revision: re-pin `config/scope_model.json`. → T-147b. **Done 2026-11-11 (owner's word):** `publish_scope_model_2026_11_11.py --card-only` committed `README.md` alone as `gamug/deberta-v3-news-scope-nli@58e4af4b`; the card differs from the live one in one line (the doc path); `--check` passes (weights and tokenizers equal to local); the pin and `published_models.json` moved to the new commit.

### Work item 18 — T-178 and T-179, the scope validation closed (moved from `TASKS.md`, 2026-11-11)

- [x] **T-178** *(Build 1, D40; #170's review)* **Audit the gold under calls 17, 18 and 21** (crypto, deals, pick lists; §9) and settle the borderline rows #39, #85, #199, #76, #154 of `gold/scope_quarantined_validation.csv`: count the `gold/` rows (365 training, held-out 200) that break them, list them for the owner, write the decisions as owner calls in the spec. **Until then 83.6% and 90.1% are not compared; no refit on the gold as it is.** **Closed by decision, not done (owner, 2026-11-11, `docs/scope_training.md` §9.1): the scope model is left as it is.**

- [x] **T-179** *(Build 1, D40; #170's review)* **Recall of `out_of_scope` among quarantined bodies:** a sample drawn by the owner's tag (`docs/scope_training.md` §9 measures precision only); the owner sets the size. **Closed by decision, not done (owner, 2026-11-11, `docs/scope_training.md` §9.1): the scope model is left as it is.**

### Work item 18 — T-162, T-163 and T-166, the pipeline's first steps built (moved from `TASKS.md`, 2026-11-12)

- [x] **T-162** *(Build 1, D40)* *(D39 conditions; schema approved 2026-11-10, AI #10)* **Run tracking and step runner:** one table `pipeline_step_state` (item, kind, step, status pending/done/failed/skipped/review, model revision, attempts, error, times; key item + step + revision; D39 (v)) and a new `cli/news_nlp_steps.py` (`--step`, `--limit`, `--status`; D39 (vii)), idempotent on the step's own result row; a new model revision adds a row, the latest is read downstream (D39 (vi)). Acceptance tests first: a rerun adds and changes nothing; `--limit N` touches at most N items; `--status` counts add up; a failed item is retried without redoing the done ones. Never steps 1 to 7 in one pass. The v1 `run_pipeline` chain stays until the backfill (T-150). → D39. Further test: a new extractor revision leaves the old `sub_news` rows alone and re-queues the steps that depend on them; the `body` item is the unique body, so a body under several tickers runs once. **Schema built** (#162; state helpers, `--status`, `--step scope` in #166). Detail of the schema sub-task and its key tests: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-162 (moved verbatim 2026-11-11). **Closed 2026-11-12 (owner: close what is developed):** the table, the runner and its three steps are built and merged: schema and `--status` (#162), `--step scope` (#166, #167), `--step cut` (#168, #169), `--step summary` (#173, #174); each step has its own tests (a rerun adds nothing, `--limit N`, a failed item retried). What no step does yet (NER, relevance, sentiment, category, review list) belongs to T-164, T-165, T-168 and T-172.

- [x] **T-163** *(Build 1, D40; step 1)* **Scope and cut:** `news_nlp_steps.py --step scope --limit N` classifies unique bodies of SOURCE into `body_scope` (T-147 classifier); `--step cut` writes `sub_news` for the `multi` ones with the `units` extractor in its `units5` reading (T-148b10, T-176, T-177). `out_of_scope` bodies are stored and stop there. → D40. **Built 2026-11-11:** scope step (#166, #167; corpus run done), cut step (#168; `docs/scope_training.md` §8.1). **Quarantined scope validated 2026-11-11 (§9): 90.8% [86.9, 94.7] on 225 bodies; `multi` 82% on 50.** **Not built:** the full cut run (~3 h GPU, owner's, after T-177); held-out figures from the command (T-147b, backlog); the checkpoint is not pinned (T-148b12). **Closed 2026-11-12 (owner: close what is developed):** the scope step ran on the corpus (309,972 bodies: single 93,034 / multi 32,555 / sector 32,887 / out_of_scope 151,496) and the cut ran on the `multi` bodies (the owner's run, revision `units-5f831c298bcb`, the `units` reading): 30,298 bodies wrote 165,192 sub-news (of 32,555 `multi`; the state of the other 2,257 was not read, the copy of the database used here has a damaged state table). Left, each in its own task: the held-out figures from the command (T-147b, backlog), the checkpoint pin (T-148b12), the evaluation of the cut's output (T-183) and the re-cut with `units5`, the default since #176 (T-177).

- [x] **T-166** *(Build 1, D40)* *(D39 (2), (3.3))* **Step 2, the summary:** `--step summary` summarises single bodies and sub-news over **2,000 characters** (owner 2026-11-12; the measure and the run rules are in `docs/summary_step.md`) with the pinned `c_summary`; shorter ones are stored as they are (`text_source = original`); tracked and idempotent, `news_summary`. **Built** (#173); 12 bodies run; determinism checked on CPU (4 items, like for like). **Left:** the corpus run (about 15 h of GPU, an order of magnitude; the counts and their cut are in `docs/summary_step.md`; determinism on the GPU), sector bodies (T-167), faithfulness of the summaries. → D39. **Closed 2026-11-12 (owner: close what is developed):** the step is built and merged (#173; the progress bars and the plan line, #174). What is left is not development and is in T-184.

### Work item 18 — T-177 and T-183, the final reading of the cut and its evaluation (moved from `TASKS.md`, 2026-11-12)

- [x] **T-177** *(owner 2026-11-11; **done 2026-11-12**)* **Headers the cut does not read, and the lost companies:** `--reading units5` (#172), the default since #176, and **the final reading of the cut** (owner 2026-11-12, after T-183). Measured on 232 bodies: several-heading texts 23 → 10, 12 of the 18 lost companies back; faithfulness on the owner's marks 97.8% against 97.1% (`units4`) and 97.6% (`units`); blind 30: 29 faithful. The corpus cut `units5-d3e2cbfa830e`: 161,574 sub-news of 30,305 bodies. `docs/scope_training.md` §8.1; PLAN D27 (e).

- [x] **T-183** *(Build 1, D40; owner 2026-11-12; **done**)* **Evaluate the cut with yes / no questions:** 28 bodies, 143 pieces of `units5-d3e2cbfa830e`, marked by the owner (`faithful`, `one_story`, `complete`; `scripts/draw_cut_sample_2026_11_12.py`, `build_cut_sample_gold_2026_11_12.py`, `cut_sample_metrics_2026_11_12.py`, tests first). Marks in `gold/cut_sample_marks.csv`. Faithful 84.6% [77.8, 89.6] on all 143 marked pieces (the lower end is below the 80% bar, D32) and 89.1% [82.5, 93.4] without the 15 pieces scoped wrongly by step 1.1 (a whole body of 13 and two single pieces). The owner took it as good enough: `units5` is final (PLAN D27 (e)). The cut's outcome: 30,305 bodies `done`, 2,250 `skipped`, 0 `failed`. What is not the cut's: a body scoped wrongly (step 1.1) and the company read from the title (step 3, T-181). `docs/scope_training.md` §8.2 and §8.3 (the one place with all the figures); `docs/gold_datasets.md` set 11.

### Work item 18 — T-194, the attribution model of the cut is pinned (#186, 2026-11-14)

- **Changed behaviour:** `--step cut` reads the pin `config/attribution_model.json` (`gamug/FinBERT-news-attribution@436ad700`, private, identity `8399ccf003c60143`). `--fetch` downloads the five model files into an empty `--model-dir` (`HF_TOKEN`; a failure is one line) and keeps them only if the identity matches; a folder with another identity is **refused** unless `--allow-unpinned`, where before any folder ran (a missing pin file refuses every folder, it does not switch the check off). The cut revision `units5-d3e2cbfa830e` does not change. The identity is hashed once per run. A run that passes a development copy needs `--allow-unpinned`. The test of the Hub copy is T-194 (in #184).

### Work item 18 — T-182, the architecture artifacts reconciled (moved from `TASKS.md`, 2026-11-12)

- [x] **T-182** *(constitution AI #11; found 2026-11-12)* **Reconcile the two architecture artifacts** (Portfolio Thesis, Portfolio NLP) for #169 to #176: the cut with `units5` as default, step 2 (the summary), the cut sample (T-183), the steps closed (T-162, T-163, T-166) and the gaps and plan they close; models: scope pinned, sentiment pair and relevance published but not pinned, category to train (T-174), summariser pinned. **Pending (owner, 2026-11-12):** both artifact URLs return "not found" for the account Claude Code is signed in with, so they cannot be read or updated from a session (retried 2026-11-12: still not found, and the account lists no artifacts); done when access is fixed. Content only: their titles never change. **Done 2026-11-12:** the owner shared both artifacts with this account (edit access); both were read in full and updated in place, titles unchanged: Portfolio NLP rebuilt around the stepped pipeline (version 45, after a first patch, version 44, that the owner rejected because it still led with the five-stage chain): why the unit changed, the step graph, each step and model with its state, current gaps and plan; the v1 chain, its diagram, its September evaluations and the engine seam in one section. Portfolio Thesis (version 48): the nlp pill, band boxes (v2 built / pending), diagram description, status row and changelog; the other repos untouched.

### Work item 18 — T-188, the low scores of the category step (decided 2026-11-14)

- [x] **T-188** *(owner's decision, T-168; #182)* An item with no main company has no category row and no state, while T-168 asks every item for a category or a `skipped` / `review` status; a low `score` is still a label. Decide the `skipped` state and a score floor for `review`. **Decided (owner, 2026-11-14): keep every label; the `score` is the confidence and consumers filter on it. Items with no main company stay on the review list of step 3 (no row and no state in the category step). A score floor is not added: it would be another revision and a change of FR-022; an abstention belongs to the model of T-174.**

### Work item 18 — development of the steps closed (owner, 2026-11-14)

The development tasks of the steps that are built (scope, cut, summary, companies, sector, category) closed by the owner on 2026-11-14; what is left of them is tested by T-170, T-173 and T-190 to T-194, or run (T-185). Moved verbatim from `TASKS.md`; the state of the store is the snapshot of 2026-11-13 23:46 UTC. The model tasks are not part of it.

- [x] **T-147** *(P3, Build 1, D40; owner 2026-11-10)* **Scope classifier from the title and the first lines of the body.** **Owner's choice (2026-11-10): (b) the body's first lines are added to the title.** First measurement: **a bag of words does not gain from them** (below), because it cannot count the companies a body names, which is what tells `single` from `multi`. Second measurement, title + **matcher features**: agreement level with the rules, `single` recall worse than theirs, `sector` undecided (below); **selected 2026-11-11** (below); weights not pinned (T-147b). → D40, D20, D33.
      - **Decision (owner):** a local classifier reads the **title** of a unique body and gives `single`, `multi`, `sector` or `out_of_scope` (`market_economy`, `investing_general`, `unrelated` merged). New pinned checkpoint in `config/pipeline_models.json` (AI #1, #12); tables to the owner (AI #10).
      - **Data:** train `gold/category_gold_v2.csv` (284), held-out `gold/category_gold_v2_test.csv` (200, never trained on; the same split as T-174; a refit on all 484 only after the bar is shown). Different samples (sector 17% against 4.5%): figures are weighted to the corpus with the `stratum` and `frame` of the 200 (D33) and the priors are corrected, not learned from the 284.
      - **Feasibility, matcher features and later experiments (2026-11-10/11):** moved verbatim to `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-147 on 2026-11-11 (the figures are in `docs/scope_training.md`).
      - **SELECTED (owner, 2026-11-11; story and limits in `docs/scope_training.md`):** two fine-tuned NLI models (topic, then single/multi) on 365 reviewed bodies (284 + 51 + 30), title + 800 chars: pinned fit (saved weights): **83.6% [78.1, 88.9]** weighted on the reviewed 200 (rules 67.2%; gain +16.5 [9.6, 23.4]); `single` recall 92%, `out_of_scope` 53%: the accepted trade-off. Earlier fit: 83.0%, 48%; old key 81.5%; optimistic (doc §4, §6). Under the 90% bar, recorded in D40's amendment. Pinned: **T-147b**.
      - **Acceptance:** >= 90% corpus-weighted agreement on the 200 (D20/D33); recall per scope reported, gated only where the gold has >= 40 bodies (sector 7 and `out_of_scope` 33 on the reviewed key are reported with their interval, not gated); the three `out_of_scope` classes reported merged and apart. If the bar is not met, the owner decides.
      - **Report:** confusion matrix; recall of `multi` and `single` against each other; a missing or unusable title goes to `body_scope` with status `review`.
  - **Closed (owner, 2026-11-14):** built, pinned and accepted. The scope bar of D20/D33 is lowered from 90% to 80% by the owner: the model passes (83.6% [78.1, 88.9] on the held-out 200). T-147b (the deferred backlog) is closed too: the owner dropped it on 2026-11-14 and replaced it, with T-161, by the rule of T-152 (no company snapshot for an item with no main company).
- [x] **T-148a** Summaries: the pinned local summariser (`c_summary`) on every `single` body and every sub-news over a size (step 2, T-166), and on the assigned `sector` bodies (built: `--items sector`, T-167, #179); title and body only. Done in chunks that resume from the last one (`scripts/cut_summaries_2026_11_07.py`, `--status`, `--require-cuda`, PR #132); the old cut's numbers are in the archive. The run of the single bodies and sub-news is step 2 (T-166).
      - Measured 2026-11-06 (`scripts/cut_spans_effect_2026_11_07.py`, 500 bodies): the rule scope is the same with and without the stored NER spans (500 of 500; matcher at 786766f), so the missing spans (none for 2026-09 and 2026-10) do not change the cut's scope; each chunk row records `spans`.
  - **Closed (owner, 2026-11-14):** built: step 2 ran on the corpus (111,546 body and 321,720 sub-news summaries at `summary-7a9bd2062bc4`, snapshot of 2026-11-13); its checks are T-190.
- [x] **T-148b** (old step 4 of D25; now D39 (3)) The text of each company in a multi body, the cut into sub-news (D27). Split:
  - **Closed (owner, 2026-11-14):** the umbrella of the step-4 extractor, closed with its children: the cut ran on `units5` and was tested (T-183). T-148b11 and T-148b12 stay open.
- [x] **T-148b4** *(owner's reading 2026-11-07, PLAN D27; merged, #143)* The `units` extractor: the attribution reads units cut at lines, bullets and clauses, without the site's footer.
      - Built: `src/news_nlp/attribution_units.py` (`strip_footer`, `attribution_units`), `model_company_texts(..., units=True)`, the extractor `units` in the faithfulness draw, `units` minus `baseline` / `model` in the report, the footer audit (`scripts/audit_footer_2026_11_08.py`, 4,000 bodies, seed 20261108), the `--stale` check (`faithfulness_stale.json`; exit code a sum, 1 stale and 2 unchecked; `--only`) and per-extractor code revisions. `chunking.split_sentences` is not changed.
      - Read by the owner (402 rows): 85.4% [76.5, 92.2] on the first set (not decided, D32), gain over the baseline +24.3 [12.9, 39.4] `better`; 21 of its 27 unfaithful texts came from first-word names (→ T-148b5). The figures that matter now are in T-148b10 (9).
      - Proposed, not built: **T-148b4b**, the gold builder carries `faithfulness_footer_only.json` and `faithfulness_stale.json` into `gold/` (precondition: the owner's decision on which reading of the bar counts). Not in this item: the first-word name matcher (T-148b5) and the model's own per-sentence attribution. → D27, D34 (e). Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148b4.
  - **Closed (owner, 2026-11-14):** built and merged (#143); the extractor is `units5`, tested (T-183).
- [x] **T-148b5** *(owner's decisions 2026-11-07, PLAN D34 (e); merged, #144)* The reading of the 402 rows left two causes before the model: first-word names taken for companies ("Bloom", "Union", "Home", "Equity", "Baker", "Boston": 21 of the 27 unfaithful `units` texts) and lines headed by a list of names and a dash whose text names other companies. **Built:** `GuardedResolver`, `attribution_pieces(..., headers=True)`, the reading `units2`, a guard-aware faithfulness report; first set 135 texts, the 33 new ones read by the owner (T-148b6). The figures that matter now are in T-148b10 (9). → D27, D34 (e). Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148b5.
  - **Closed (owner, 2026-11-14):** built and merged (#144); superseded by the later readings (`units5`), tested (T-183).
- [x] **T-148b6** *(2026-11-07, after the owner's reading of the 33 `units2` texts)* `units2` on the 30 bodies: 130 of 135 texts faithful, 84.4% [72.5, 97.2] on the bar (not decided: 19 real companies named have no text because a "NAMES - text" line went whole to the header names). **Built:** `reading="units3"` = `units2` plus a piece of such a line also belongs to the universe companies it names itself. First set: `units3` 91.6% [84.7, 97.8] (141 of 148 + 6 left without), `units3` minus `units` -3.9% [-7.3, 0.0], not decided. The owner chose `units3` as the candidate on 2026-11-07; **replaced by `units` on 2026-11-08** (T-148b10). Known limit, same family as T-148b7: a header line can carry a second company's sentence. The 5 omissions not looked at (CSCO, NSC, KR, NKE, SPG) are left to T-148b7. → D34 (e). Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148b6.
  - **Closed (owner, 2026-11-14):** built; replaced by `units` (T-148b10) and later `units5`; tested (T-183).
- [x] **T-148b10** *(owner's decisions, 2026-11-08)* Candidate for step 4's cut, its confirmation and its selection.
      - (1) the candidate is `units`; `units2` and `units3` stay as measured fallbacks. (2) T-148b9 is out of scope. (3) The confirmation draw (30 new bodies, `draw_faithfulness --exclude-from data --out data/faithfulness_confirmation`; the pool is used up, 30 of 32 drawn) is done and tagged. (4) The stack #141, #143, #144 is merged, in that order, by the owner. (5) The sentiment tagging of the 450 summaries is done and its marks are in `gold/` (#145); the retraining (T-148c3) waits for the owner's word.
      - (6), (6b), (7), (8) were rules and schedules proposed before the confirmation was read (the bar on real companies, the bar on every company named, the held cut, the go/no-go schedule); superseded by (9).
      - **(9) Owner's selection, 2026-11-08: `units` ("units1") is the base model of step 4; the owner judged 80% accuracy enough** (confirmation 86.1% on every company named, 95.3% on real companies; level with the baseline, so the rule of (6b) is not met and the selection is the owner's call). `units2`, `units3` not selected; not a pin, not a publication. → `PLAN.md` Work item 18, D34 (e); `docs/extractor_training.md`. Follow-ups: T-148b11, T-148b12. Full text: `.specify/memory/archive/SPEC-TASKS-condensed-for-the-category-step-2026-11-13.md` § T-148b10 (9).
  - **Closed (owner, 2026-11-14):** the selection of the extractor is done (`units`, then `units5`, tested: T-183). T-148b11 and T-148b12 (nested here) stay open as their own entries.
- [x] **T-148b8** *(manager's review of #144, 2026-11-08; the evidence under `units3` as the candidate)* (a) **The denominator:** the 91.6% of `units3` left out 36 companies the extractors are not asked about; on every company named it is 74.2% [66.0, 81.5] (`units2` 68.4%, `units` 83.2%). `draw_faithfulness --exempt` lists them (`faithfulness_exempt.json`, `faithfulness_exempt_to_read.csv`); the owner read the 37 of the first set (23 not companies, 14 are) and the 30 of the confirmation (25 not companies, 5 are); `report_wi18 faithfulness` gives the bar on real companies. (b) **A confirmation:** done (T-148b10). (c) The header audit (`scripts/audit_headers_2026_11_08.py`): datelines and labels in capitals and a dash inside a sentence are not headers (fixed, no text of the 30 bodies changes); 67 lines given to nobody still name a universe company after the dash (7 behind a short form the guard rejects: T-148b7), 15 headers are missed for a full stop in the first part. → D32, D34 (e). Full text before 2026-11-08: `.specify/memory/archive/TASKS-open-entries-2026-11-08.md` § T-148b8.
        - [ ] **T-148b9** *(proposed improvement, **out of scope** by the owner's decision of 2026-11-08: the performance of `units` as it is is acceptable; after the owner's reading of the 37 exempt companies, 2026-11-08)* The guard loses 13 real companies on the 30 bodies, mostly lines headed by short forms ("Merck - Merck shares retreated...", "Colgate-Palmolive - Colgate named...", "United Airlines, American Airlines - ... Delta ...", "Molson Coors - ..."): the header rule gives such a line to nobody because the guard does not accept the short form in the head. Option, to be measured as a new reading (not a change of `units3`, whose texts the owner read): match the part before the dash with the plain resolver, since a name before a dash is deliberate; the owner would read the texts that change. Not built; whether it is worth building depends on the decision between `units` and a guarded reading for the cut.
  - **Closed (owner, 2026-11-14):** closed with the extractor (the evidence is in `docs/extractor_training.md`); its out-of-scope child T-148b9 closes with it.
- [x] **T-164** *(Build 1, D40)* *(D39 (2))* Single path: summary (T-148a), NER on the summary, the matcher turns each NER span into a universe ticker (spans with none are ignored; comparison by ticker only, D39 (iii)); the relevance model per (summary, ticker) gives the main company; the other companies stay as mentions, the result per ticker goes to `news_company` (relevance score, `is_main`; D39 (iii), D40); main = every company over the threshold, one stored row per main company (D39 (i)); none over it: the review list (D39 (ii)). Was blocked by T-162 (schema, done) and the threshold (decided 2026-11-13: 0.5, T-185). → D39. **Closing check:** on a 100-body dry run, the share of items with a main ticker, on the review list and with ignored spans only. **Acceptance:** one `news_company` row per (text, ticker) is its key (a test inserts a ticker named three times and finds one row); `is_main` set only above the threshold; none over it gives no `is_main` row and status `review`. **Built 2026-11-13** (`--step companies`, one command for the single bodies and the sub-news, `src/news_company_stage.py`, `docs/companies_step.md`); the acceptance tests pass. **Closing check on a random sample of 200 single bodies of a copy of the RESULTS database (not the 100 bodies written above), threshold 0.5, `ner+text`:** 61.5% have a main company, 9.5% have a candidate and none over the threshold, 29.0% no company found. The threshold and `--candidates` are the owner's: T-185.
  - **Closed (owner, 2026-11-14):** built (step 3, #178; `docs/companies_step.md`); its closing check (the dry run: shares of items with a main ticker, on the review list, with ignored spans) is T-191.
- [x] **T-165** *(Build 1, D40)* *(D39 (3), (3.1), (3.2))* Multi path: the `units` extractor, read as `units5` (T-148b10, T-176, T-177; built in #168/#169/#172, default in #176), cuts sub-news, NER on each, the relevance model picks the main companies (all over the threshold, one stored row per company, D39 (i)); none: the review list. Risk: the relevance model was trained on summaries, sub-news are shorter (see T-170). **Gate of §8.1 released by the owner 2026-11-11, no flag on `multi`** (`docs/scope_training.md` §9: `multi` 82% on 50); reads `sub_news`. Replaces the attribution model (T-148b3, not pinned) in the MVP. → D39. **Reads the `units5-…` revision by name** (`units-…` and `units4-…` rows are older readings; §8.1). **Closing check:** on 50 multi bodies, sub-news per body and the share with a main ticker; **and** the assumption behind `units4` (D39 (iii)): of the companies it leaves without a text (18 of 331 on the owner's marks), how many NER finds in a sub-news of the same body. **Built with T-164 (same step).** Dry run on 200 random sub-news of `units5-d3e2cbfa830e`: a main company in 73.0% (`ner+text`, corrected definition; 70.5% with `ner`), 20.5% a candidate and none over 0.5. **The closing check on 50 multi bodies and on the companies `units4` left without a text is not done** (the sample is of sub-news, not of bodies).
  - **Closed (owner, 2026-11-14):** built (steps 1.2 and 3, #168/#169/#172/#176/#178); its closing check is T-191.
- [x] **T-184** *(Build 1, D40; what is left of T-166)* **Step 2 to run and to check:** (1) the corpus run, the owner's (about 15 h of GPU, `docs/summary_step.md`): 93,034 single bodies + the 161,574 sub-news of `units5-d3e2cbfa830e` = 254,608 items, 64,480 to the model and 190,128 stored as they are; (2) the GPU time per item and the determinism check on the GPU (`scripts/summary_determinism_check_2026_11_12.py`); (3) faithfulness of the summaries, with yes / no questions like T-183, when there are enough; (4) the sector bodies, in T-167's step. → D39.
  - **Closed (owner, 2026-11-14):** (1) the corpus run is done (111,546 body and 321,720 sub-news summaries; the 18,512 assigned sector bodies included, so (4) too); (2) the determinism / GPU time check and (3) the faithfulness are T-190.
- [x] **T-167** *(Build 1, D40)* *(D39 (4); was T-149)* Sector path. **Built 2026-11-13 (#179, `docs/sector_step.md`, PLAN D39 (4); `body_sector` and the vote rule approved by the owner):** `--step sector` gives a `sector` body its 2-digit GICS sector or `out_of_scope`; `--step summary --items sector --sector-revision REV` summarises the assigned ones. **Measured:** the owner's clean review, 35 of 40 assigned right as drawn (87.5%), 35 of 39 under the corrected tie rule (89.7%, 76-96%), 0 of 5 controls; one reviewer, not the D24 measure; 56.0% of the sector bodies get a sector (random sample of 1,500). **Open:** (1) the run on the 32,887 `sector` bodies (about 12 min on the GPU, 2 h on the CPU); (2) the D24 bar (>= 80% on the held-out gold) and the sub-industry: **not met or measured here**; (3) Materials and Consumer Staples have too few rows; (4) the scope step lets non-sector bodies through. **Closing check:** the sector's accuracy on the held-out gold with the sub-industry (D24), and the full run with the summary count per run.
  - **Closed (owner, 2026-11-14):** built (#179), ran on the 32,887 sector bodies (18,512 assigned, 14,375 `out_of_scope`) and tested (89.7% on the owner's review, 35 of 39). Not done and not tracked: the sub-industry and the D24 figure against the held-out gold.
- [x] **T-172** *(closed as not built, owner 2026-11-14; Build 1, D40)* *(D39 (ii); proposed, storage to the owner)* **Review list:** an item with no company over the threshold gets `status = review` in `pipeline_step_state`; `news_nlp_steps.py --export-review` writes the file the owner reads (ids, text, Spanish columns) and `--import-review` reads the owner's answer (the main company, or "none"), which re-enters the pipeline at step 6. `--status` counts `review` apart, so a run is not reported complete while they wait. Before the threshold is chosen, T-169 reports how many of the 809 rows fall under each candidate threshold (a threshold that sends about 30% to a human is rejected).
  - **Closed (owner, 2026-11-14):** not built, closed by the owner's instruction; the `review` state in `pipeline_step_state` stays (step 3 writes it); `--export-review` and `--import-review` are dropped. Reopen it if the owner wants the commands.

- **T-185 (before it was trimmed to the run left, 2026-11-14):**
  - [ ] **T-185** *(Build 1, D40; T-164, T-165)* **Run step 3.** **Decided by the owner, 2026-11-13 (interactive session, #178):** `--threshold 0.5` now, to be revisited with T-170 (a new value is a new revision); the raw NER spans are **not** stored (`mention_count` keeps the count); `--candidates ner+text`, chosen on the first definition of the text candidates (sub-news: candidate 99.5%, main 75.0%). **Confirmed 2026-11-13:** the corrected definition of review #178 (lower-case matches, disclosures and symbol lists left out; candidate 93.5%, main 73.0%; `docs/companies_step.md`), its limits accepted; the sub-news threshold stays 0.5 until T-170. Left: #178 to merge, then the corpus run, the owner's, in slices: `--items single` once step 2 has done them, sub-news with the full `--cut-revision units5-d3e2cbfa830e` (the partial `units4-d3e2cbfa830e` shares its 12 hex digits). A new value is a new revision and the old rows stay. → D39 (i), (iii), D40.
- [x] **T-149** *(P5)* Moved to **T-167** (D39).
  - **Closed (owner, 2026-11-14):** a pointer to T-167, which closed with the development of the steps.

### Work item 18 — sentiment proposals dropped (owner, 2026-11-14)

- [x] **T-148c1b** *(closed as not built, owner 2026-11-14)* (proposed 2026-11-08, not started, not decided) Namesake companies in the 450 training summaries: the owner's notes name four rows whose `company` is probably a namesake (listed in `docs/gold_datasets.md`, set 6). Measure how many rows of `gold/sentiment_train.csv` carry a company the text does not speak of, with the resolver the draw used (first-word and ticker matches; the same family the faithfulness reading of PR #144 found: first-word and namesake matches), no tagging asked and no mark changed. What to do with them (keep, drop from the training, relabel) is the owner's.
  - **Closed (owner, 2026-11-14):** proposed, never built; dropped by the owner.
- [x] **T-148c2c** *(closed 2026-11-14 as a whole; the verdicts part was restored and is open in `TASKS.md`; proposed, not built; D37, owner 2026-11-09)* The gold scorer (`score_sentiment_gold`) records the precision, recall and ROC OVR of each class (`draw_sentiment_gold.predict_probs` gives the probabilities) and the paired gain of the score and of the negative class's three, with verdicts, in its log (`retag.SENTIMENT_SCORING_LOG_COLUMNS`); T-148c4 reads them through `pin_decision`. → D37. **Added 2026-11-10 (#164):** it also writes the confidence check of T-148c4 (mean largest class probability on the 180 texts, marks unread).
  - **Closed (owner, 2026-11-14):** proposed, never built; dropped by the owner.
  - **Restored (owner, 2026-11-14), the verdicts part only:** the pin rule of T-148c4 reads the scorer's verdicts; T-148c2c is open again in `TASKS.md` for them (T-148c1b and T-148c3e stay dropped).
- [x] **T-148c3e** *(closed as not built, owner 2026-11-14; proposed)* Class imbalance of the tags: weights or resampling in the CV.
  - **Closed (owner, 2026-11-14):** proposed, never built; dropped by the owner.

### Work item 18 — T-175, the licence of the sentiment model (done 2026-11-14)

- [x] **T-175** *(owner's decision, small; #164)* `gamug/FinBERT-financial-news` is `cc-by-nc-4.0` but its base `ProsusAI/finbert` is `cc-by-nc-sa-4.0` (share-alike): **Decided (owner, 2026-11-14): correct the card of `gamug/FinBERT-financial-news` to `cc-by-nc-sa-4.0`.** **Done on the Hub, 2026-11-14 (owner's word): commit `7305d765`, model card only** (`docs/sentiment_training.md`). The sentiment card already says `cc-by-nc-sa-4.0`.
  - **Closed (owner, 2026-11-14):** decided and done: the card of `gamug/FinBERT-financial-news` (public) is corrected to `cc-by-nc-sa-4.0`, the licence of the base model `ProsusAI/finbert`: Hub commit `7305d765`, model card only (the pinned revision `93863fcb` still resolves and the weights are identical). The one-shot publish scripts of 2026-09-13/14 still hold the old text and are not run again; the dataset repository `gamug/FinBERT-financial-news-data` is a separate card and was not touched.

### Work item 18 — closed by the owner, 2026-11-14 (model line)

- [x] **T-161** *(closed, owner 2026-11-14: replaced by the rule of T-152; D38; PR #155, proposed fix built)* **Matcher: common words and surnames are not companies.** Merge #155, then re-run T-160's CSV (owner, ~4.4 h CPU) before T-153 (D36 (j)); redraw the gold pool and marks only after the last matcher PR merges (stale on master already; the matcher changes the revisions again). Then retrain the attribution checkpoint (T-148b3, D38 (d)); the cut stays frozen (D38 (e)).
  - **Closed (owner, 2026-11-14):** closed by the owner: the matcher re-run, the gold redraw and the retraining of the attribution checkpoint are dropped (the matcher fix #155 is merged). Replaced by a rule in the snapshot step (T-152): a company snapshot excludes every item with no main company.
- [x] **T-147b** *(closed, owner 2026-11-14: dropped, replaced by the rule of T-152; Build 1, D40; after T-147 SELECTED)* *(backlog; **deferred, owner 2026-11-11: left apart, to be fitted later; not time to be picky**)* **Make the scope classifier usable.** **Done 2026-11-11 (owner's word):** refit and checkpoints (`scripts/train_scope_final_2026_11_11.py`); held-out 200 scored from the saved weights (`score_scope_heldout_2026_11_11.py`): 83.6% [78.1, 88.9], gain +16.5, `out_of_scope` recall 53%; uploaded **public** to `gamug/deberta-v3-news-scope-nli@58e4af4b` (card-only commit on `@d14a4d20`, same weights) and pinned in `config/scope_model.json` with the step revision `scope-nli-8ff21e50796b`. **Left:** the models on `fti.py` with an MLflow run, and three seeds (about 45 min GPU). A refit that adds the 200: owner's word (AI #1, #7, #12). → T-147, T-163.
  - **Closed (owner, 2026-11-14):** closed by the owner: the deferred scope backlog is dropped (the scope classifier is pinned and accepted, with the bar at 80%). Replaced, with T-161, by the rule in the snapshot step (T-152): items with no main company are excluded.
- [x] **T-148b12** *(wired in #168/#169; closed by the owner 2026-11-14; published in #185, pinned in #186)* **Wire the extractor into the cut.** **What it is:** the trained attribution model, which gives the pieces of a round-up that name no company to the company they belong to, or to nobody (the cut uses it). **Left:** the checkpoint (`data/models/attribution_2026_11_08`, git-ignored; identity `8399ccf003c60143`) is not published or pinned to the Hub: the owner's word. The cut it made, `units5-d3e2cbfa830e`, is the reference cut and `units5` the final reading (owner 2026-11-12, D27 (e)). **Decided (owner, 2026-11-14): publish the checkpoint to the Hub and pin it in `config/` (its own PR); the cut does not change.**
  - **Closed (owner, 2026-11-14):** closed by the owner, replacing the decision above to publish and pin the checkpoint: the attribution model (the one the cut uses for the pieces that name no company) stays local, `data/models/attribution_2026_11_08` (git-ignored, identity `8399ccf003c60143`); the reference cut is `units5-d3e2cbfa830e`. A reader without the folder cannot repeat the cut.

### Work item 18 — T-195, the snapshot step (built 2026-11-14)

- [x] **T-195** *(Build 3, D42, FR-024; replaces T-152; **built 2026-11-14**: `src/news_snapshot_stage.py`, `tests/news_nlp/test_snapshot_step.py`)* **The snapshot step** (`--step snapshot`): `news_snapshot` per (text, main company) or `sector` text (no category or sentiment for it), (**revised in #190: the revision is the code's only; the input revisions are in each row, `docs/snapshot_steps.md`**) from the step revisions a company row reads, refused when a revision has no rows or the rows are not main pairs at the companies revision; `fetched_at` of a body = the earliest of its articles; complete rows only; `companies_mentioned` left empty for now (T-198); `.env` `BATCH_TIMEZONE` (default `America/New_York`) in `.env.example`. Design and tables: `docs/snapshot_steps.md`. State is derived from the tables (no `pipeline_step_state`; `--status` is a new function, per pair). The first real run waits for the owner to run sentiment and category on the real store (T-192 and T-193 validate those runs; no task runs them).

### Work item 18 — T-195 follow-up, progress on the snapshot step (#192, 2026-11-14)

- **Changed behaviour:** `--step snapshot` shows a line before the corpus read and one `tqdm` bar with the time left over the rows to write (company pairs and sector texts together), on stderr. `snapshot.run` takes an optional `on_progress(event, n)` (the bar lives in `cli/news_nlp_steps.py`, like the other steps) and commits every `COMMIT_EVERY` (500) rows instead of once at the end, so an interrupted run keeps its rows. Written rows and the JSON output are unchanged. Tests: 3 new in `tests/news_nlp/test_snapshot_step.py`.
