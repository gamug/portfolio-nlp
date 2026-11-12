# news_nlp: the v1 sentiment, zero-shot category and sector_summary stages (legacy)

**Legacy (moved 2026-11-12).** The owner deprecated these three v1 stages on 2026-11-12: the stepped pipeline of Work item 18 replaces them (sentiment per text and main company with the pair model, category with the D15 taxonomy, sector bodies in their own step; `docs/modules/news-nlp.md`). Their code still runs in `pipeline.run_pipeline` and their tables are still read downstream until the backfill (T-150). The passages below are cut verbatim from `docs/modules/news-nlp.md` at commit `09d90ae` (line ranges in the comments); the v1 NER and `c_summary` passages stayed there, because NER and the summariser are used by the stepped pipeline too. The evaluations of these stages are in `docs/legacy/evaluation-v1-stages.md`; the v1 taxonomy in `docs/legacy/category-taxonomy-v1.md`.

## Stage 1, sentiment

<!-- docs/modules/news-nlp.md lines 15-58 at 09d90ae -->
1. **Sentiment** — a continued fine-tune of FinBERT on 5,900 real, LLM-labeled
   in-domain sentences, published at
   [gamug/FinBERT-financial-news](https://huggingface.co/gamug/FinBERT-financial-news)
   (training data published separately at
   [gamug/FinBERT-financial-news-data](https://huggingface.co/datasets/gamug/FinBERT-financial-news-data),
   2026-09-14 — the exact `train`/`validation`/`test`/`idiom_probe` split the model was
   actually trained/evaluated on, reproduced via `train_sentiment.stratified_split()` itself
   rather than re-derived; `scripts/publish_finbert_financial_news_dataset_2026_09_14.py`)
   (selected 2026-09-13 over base `ProsusAI/finbert`, after real-data evaluation
   found a measurable vocabulary/domain gap — see `docs/evaluation.md`'s 2026-09-13
   follow-ups; now pinned to v4, the class-weighted retrain — `PLAN.md` Work item 9,
   `docs/evaluation.md`'s 2026-09-15 follow-ups) → `article_sentiment`.

   **Why this model at all.** A FinBERT-family model — domain-pretrained on financial
   text rather than a general-purpose sentiment model or a from-scratch train — was
   the starting point because it's the one freely-available checkpoint already
   *fine-tuned* for financial-domain sentiment specifically, not just pretrained on
   financial text: Araci (2019), *"FinBERT: Financial Sentiment Analysis with
   Pre-trained Language Models"* ([arXiv:1908.10063](https://arxiv.org/abs/1908.10063)).
   Continuing fine-tuning from `ProsusAI/finbert` specifically, rather than a generic
   `bert-base` checkpoint or reaching for a larger general-purpose LLM, follows
   directly from that: `ProsusAI/finbert` is itself a continued fine-tune of BERT-base
   on [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
   (~4,840 sentences, Malo, Sinha, Korhonen, Wallenius & Takala 2014,
   [arXiv:1307.5336](https://arxiv.org/abs/1307.5336)) — a model already carrying real
   financial-sentiment signal, small enough (BERT-base) to further fine-tune cheaply on
   this project's own 5,900-sentence in-domain set rather than requiring a much larger
   dataset or model from scratch. Chunk-level + entity-scoped weighting (below) over
   sentence-level/title-only/whole-document alternatives was a separate, later, directly
   measured decision — see `docs/evaluation.md`'s 2026-09-13 four-candidate comparison,
   not re-derived here. Scored per ~510-token chunk (`chunk_text`,
   same helper NER/category use — preserves several sentences' worth of real
   discourse per forward pass) and aggregated with **entity-scoped weighting**: a
   chunk naming the article's own `company`/`ticker` counts more than one that
   doesn't (a different company's news, or generic market commentary) — see
   `_sentiment_chunk_weights` (`src/sentiment_stage.py`) and `PLAN.md` Work item 4. (A
   first version of this scored per *sentence* instead of per chunk; reverted
   2026-09-13 after real-data evaluation — see that function's docstring
   "Revision history". A title-only alternative was also real-data validated and
   rejected in favor of this design's stronger, both-classes recall — see
   `docs/evaluation.md`'s 2026-09-13 "pessimist model" follow-up.) Known,
   disclosed limitation: ~40% of directional predictions are false alarms on
   multi-company/mixed-signal articles the aggregation can't structurally net
   out — chosen anyway because this pipeline favors recall over precision.

## Stage 3, category (zero-shot)

<!-- docs/modules/news-nlp.md lines 82-106 at 09d90ae -->
3. **Category** — zero-shot NLI classification (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`)
   against a fixed 10-category taxonomy (9 dimensions of company performance + `other`) →
   `article_category`. See [`../category-taxonomy.md`](../category-taxonomy.md) for the
   taxonomy, its sourcing (RavenPack, SASB, IPTC Media Topics, Refinitiv/TRNA), and the
   confidence-threshold mechanism. Classifies on the article's title + lead body chunk
   (not the full article — a per-label NLI pass makes full-article chunking too expensive
   for a mandatory, every-article stage — and not `article_summary`, since that's opt-in
   and this stage isn't).

   **Why this model.** No labeled training set exists for this project's own 10-slug
   taxonomy — and building one for a taxonomy that may still evolve is expensive — so a
   zero-shot NLI classifier was the only realistic option, not a trained one.
   `deberta-v3-base-zeroshot-v2.0` specifically: DeBERTa-v3's disentangled-attention
   architecture, further trained on MNLI + FEVER-NLI + synthetic data across 500+
   classification tasks, benchmarks at 0.619 average F1-macro across 28 held-out zero-shot
   datasets per the model's own card — against 0.497 for `facebook/bart-large-mnli`, the
   reference zero-shot-NLI baseline it's commonly compared against. The taxonomy itself is
   hierarchical, not flat 9-way: a 2026-09-09 investigation found a flat 9-way softmax
   diluted real signal on several slugs (the model's own raw score for the correct slug on
   missed rows averaged only 0.14-0.16, barely above the 9-way uniform baseline of 0.111) —
   not unbalanced data or a sampling artifact. `run_category_stage` now classifies through a
   two-level hierarchy (`news_nlp.taxonomy.CATEGORY_GROUPS`) instead, restoring a much
   stronger ~0.333 baseline for the child-group softmax that actually decides a hard case —
   see `docs/category-taxonomy.md`'s "Hierarchical classification" section for the full
   design.

## Stage 5, sector_summary

<!-- docs/modules/news-nlp.md lines 127-163 at 09d90ae -->
5. **`sector_summary`** — one row per `gics_sub_industry` per closed calendar week →
   `sector_summary`. **Why this model: no model, deliberately.** Removing the one model
   this stage used to run (below) is a selection choice, not an incidental fact — a
   deterministic template is a structural guarantee against hallucination, not a
   probabilistic mitigation of it, the same principle that already makes cross-company
   blending structurally impossible in this stage's design. Lives in its own
   `news_nlp/sector_summary/` package
   (`queries.py` for DB reads/writes, `composition.py` for the pure
   composition logic, `stage.py` for the `run_sector_summary_stage`
   orchestration entrypoint `pipeline.run_pipeline` calls) — deliberately
   decoupled from the FTI hierarchy below, since it never loads a model or
   touches the GPU and so has no Feature/Train/Inference shape to share
   with the four ML stages. A deterministic, non-generative composition
   (`db.compose_sector_summary`):
   a stats overview (sentiment %, top entities) plus one section per NLP category (stage 3's
   taxonomy) present that week, each listing its contributing companies' `c_summary` text
   verbatim and attributed to its own ticker — company text is never blended with another
   company's, and never crosses a category-section boundary. The intro sentence
   (`db.build_sector_intro_seed`) — the one piece of prose in this stage that names
   aggregate stats rather than quoting a company's own text verbatim — used to be run
   through `sshleifer/distilbart-cnn-12-6` as a "paraphrase"; **as of 2026-09-14 it no
   longer is**, after that step was found to hallucinate on 42-50% of rows (a fabricated
   source attribution, or a self-contradicting repeated percentage — see
   `docs/evaluation.md`'s 2026-09-14 follow-up). `build_sector_intro_seed`'s own output
   was already a complete, fully-grounded sentence, so `intro_text` is now that seed
   verbatim — zero hallucination risk by construction, and this stage no longer loads a
   model or touches the GPU at all. No ticker, company name, or `c_summary` text ever
   reaches it either way, which is what makes cross-company/cross-topic blending
   structurally impossible rather than merely unlikely. Articles whose `c_summary` exists but have no
   `article_category` row (historical data predating stage 3 becoming mandatory, or a
   partial/direct stage invocation) are excluded from sector_summary generation entirely.
   Sub-industries with no qualifying articles in a given week get no row; "closed" means the
   week (Mon–Sun) has fully ended, so a summary is never generated from a partial week and
   later need regenerating. `sector_summary.format_version` is bumped whenever this
   generation logic changes shape — a row below the current version is treated as stale and
   self-heals (regenerated via an upsert on the `(gics_sector, gics_sub_industry, week_start)`
   key) the next time the stage runs, no separate backfill needed.

## The Evaluation section as it was (it describes the v1 stages' eval paths, the sector intro eval and Work item 15's calibration tools)

<!-- docs/modules/news-nlp.md lines 475-518 at 09d90ae -->
## Evaluation

`news_nlp.eval` measures how good the stage outputs actually are, using an
LLM-as-judge over a stratified sample (a deterministic low-confidence
stack, stage-specific soft-probability-targeted strata, and a
representative random remainder — see `docs/evaluation.md`'s "Sampling"
section) of the stored predictions (sentiment / category / NER /
`c_summary`), with metrics tracked in MLflow and in the `eval_run` /
`eval_inference` / `eval_verdict` tables (plus `eval_confusion` for
sentiment/category; the legacy `eval_judgement` table is superseded,
2026-09-18). The model-scoring step calls each stage's own FTI `Inference`
subclass directly (see "FTI architecture" above) rather than loading
models independently, and can score a candidate model/checkpoint live
(`--candidate-model`/`--candidate-revision`) without a production run
first. A stored category run can also be re-read per subset of the
corpus (for example per upstream `match_class`) with no judge call, reweighted as a
domain estimate and with bootstrap intervals (`news_nlp.eval.partition`; see
`docs/evaluation.md`'s 2026-10-22 follow-up). Candidate category decision rules
are simulated against stored scores and stored verdicts, and their power estimated, by
`news_nlp.eval.calibration` (2026-10-23 follow-up). The paid sample that settles them is
drawn by `news_nlp.eval.calibration_sample` (confirmed articles stratified by slug x score band,
seeded fit/report halves, weights) and judged by the opt-in truth-label judge
`news_nlp.eval.calibration_judge` (`prompts/calibration.md`, frozen; crash-safe, resumable, capped
on calls; see `docs/evaluation.md`'s "Calibration judge and sampler"). `sector_summary`'s
`intro_text` had its own dedicated eval path added 2026-09-14 — its
population is small enough (thousands of rows) to judge in full every run
instead of sampling, checked for faithfulness against its own `facts_json`
grounding only. It found a real, sizable hallucination rate (42-50%) in the
old model-paraphrase `intro_text`, and stayed useful past that: with
`intro_text` now a deterministic template (`docs/evaluation.md`'s 2026-09-14
follow-ups), this eval doubles as a regression guard — it should read
~0% hallucination on freshly-generated rows going forward, and a jump away
from that is a real signal something broke. It is a separate `eval`
dependency group and needs an OpenAI-compatible LLM endpoint
(`LLM_API_KEY` / `LLM_MODEL` / `LLM_URL`).

```bash
uv sync --group eval
uv run cli/news_nlp_eval.py --stage all --sample-size 80
uv run mlflow ui
```

Full detail — sampling, per-stage metrics, the "judge is a model, not gold"
caveat, and the CI / scheduled story — in `docs/evaluation.md`.
