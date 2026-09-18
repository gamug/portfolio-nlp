# news_nlp — pipeline stage 3: sentiment + NER + categorization + summarization

**Extracted from** `portfolio-data-mining` (which had consolidated it from the standalone `news-nlp` project).

Runs up to five sequential batch stages over the `articles` rows in
`data/nlp.db` (override with `$DATABASE_URL`, see root `.env.example`), each stage owning
its own results table, and exposes everything through a FastAPI service. There is no scraper
here — `articles` is treated as pre-populated input.

Sentiment, NER, and category (stages 1–3) always run. `c_summary`/`sector_summary`
(stages 4–5) are **opt-in** — pass `--summarize` on the CLI or `"summarize": true` in the
API's `/pipeline/run` body; the default (`summarize=False`) skips them entirely, so the
summarization model never loads and its VRAM/latency cost is never paid unless asked for.

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
   follow-ups) → `article_sentiment`. Scored per ~510-token chunk (`chunk_text`,
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
2. **NER** — a fine-tuned SEC-BERT-BASE model trained on FiNER-ORD, published at
   [gamug/sec-bert-finer-ord-ner](https://huggingface.co/gamug/sec-bert-finer-ord-ner) →
   `article_entities`. Batched `NER_BATCH_SIZE` (`src/pipeline.py`) articles per forward
   pass: every article's chunks in the batch are flattened into one padded tokenizer call,
   so the actual batch width is that batch's *total chunk count*, not `NER_BATCH_SIZE`
   itself — see the constant's comment for why (variable per-article chunk counts, unlike
   category's fixed 9-pairs-per-article width).
3. **Category** — zero-shot NLI classification (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`)
   against a fixed 10-category taxonomy (9 dimensions of company performance + `other`) →
   `article_category`. See [`../category-taxonomy.md`](../category-taxonomy.md) for the
   taxonomy, its sourcing (RavenPack, SASB, IPTC Media Topics, Refinitiv/TRNA), and the
   confidence-threshold mechanism. Classifies on the article's title + lead body chunk
   (not the full article — a per-label NLI pass makes full-article chunking too expensive
   for a mandatory, every-article stage — and not `article_summary`, since that's opt-in
   and this stage isn't).
4. **`c_summary`** — one abstractive summary per article (`sshleifer/distilbart-cnn-12-6`),
   generated from the article's body plus its already-computed sentiment/entities →
   `article_summary`.
5. **`sector_summary`** — one row per `gics_sub_industry` per closed calendar week →
   `sector_summary`. Lives in its own `news_nlp/sector_summary/` package
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

- **FTI architecture** (`src/fti.py`, added 2026-09-16/18 — PLAN.md Work
  item 10) — `Feature`/`Trainer`/`Inference` base classes formalizing the
  fetch → load → batch/predict → write → free pattern every ML stage
  follows; one concrete subclass triple per stage
  (`sentiment_stage.py`/`ner_stage.py`/`category_stage.py`/
  `summary_stage.py`). Category and `c_summary` — zero-shot/pretrained,
  no fine-tuning step — get a trivial, body-less `Trainer` subclass each
  (`CategoryTrainer`/`SummaryTrainer`, both just `NoOpTrainer` under a
  stage-specific name, added 2026-09-18 — TASKS.md T-097) so every stage
  has a real, named `Trainer` a stage→`Trainer` registry can look up
  (`src/experiment.py`, see "Experiments" below); sentiment and NER each
  have a real, trainable `Trainer` subclass (`train_sentiment.py`/
  `train_ner.py`) — NER's own `NerTrainConfig` gained a `base_model` field
  the same day (2026-09-18, T-099), closing a gap where it was the only
  stage whose base checkpoint wasn't configurable at all.
  `pipeline.py`'s `run_<stage>_stage` functions are thin wrappers that
  construct a stage's `Inference` subclass and call `.run(...)` — kept as
  real module-level functions (not inlined) so existing test/resample-script
  monkeypatching keeps working. `news_nlp.eval` reuses these same
  `Inference` subclasses for its own model-scoring step (see "Evaluation"
  below) instead of loading models independently.
- **Sentence-aware chunking** (`src/chunking.py`) — articles run up to ~13K
  words, far past BERT's 512-token limit; chunks are packed on sentence boundaries. The
  category stage reuses it (with a tighter token budget, to leave headroom for the NLI
  hypothesis text) to take just the lead chunk; the summarization stages reuse it for
  BART's 1024-token cap, plus a **hierarchical reduce**
  (`hierarchical_summarize_batch`, `src/summary_stage.py`): summarize each chunk, then if
  more than one chunk resulted, recursively summarize the concatenated chunk-summaries
  until they collapse into a single pass. Sentiment reuses the same chunker
  (`max_tokens=510`) as NER, scoring one forward pass per chunk (see above).
- **Idempotent, resumable batch processing** — each stage only processes rows missing
  from its results table (articles for stages 1–4, `(gics_sector, gics_sub_industry,
  week_start)` groups for stage 5, enforced by a `UNIQUE` constraint on `sector_summary`).
- **6GB-VRAM-friendly** — only one model resident on the GPU at a time; each stage loads
  its model, runs to completion, and frees the GPU before the next stage loads (skipped
  entirely when a stage has nothing pending).
- `c_summary` only covers articles that have both a computed sentiment **and** at least
  one entity scoring above 0.8 confidence (`article_entities.score > 0.8`, excluding
  single-character digit noise) — an article with sentiment but no qualifying entities
  never gets summarized. This mirrors the original `query.sql` design this feature is
  based on.

## Setup

```bash
uv sync                          # torch is a pinned direct dep (cu124 wheel index) — no manual install
uv run python -m setup           # pre-fetch the four HF models into the local cache (one-time, safe to re-run)
```

### Model pins

Every `from_pretrained`/`snapshot_download` call (in `src/pipeline.py` and `src/setup.py`) is pinned to a commit SHA via `pipeline.MODEL_REVISIONS` (added 2026-09-14, `SPEC.md` §13 item 4) — an upstream push to any of these repos no longer changes results silently:

| Model | Repo | Pinned SHA |
|---|---|---|
| Sentiment | `gamug/FinBERT-financial-news` | `072712344f1f82e54391e6721b0b39e7b944e898` |
| NER | `gamug/sec-bert-finer-ord-ner` | `ba7b9e43e4aa023ec5691f955b276dc58158354c` |
| Category | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8` |
| Summarization | `sshleifer/distilbart-cnn-12-6` | `a4f8f3ea906ed274767e9906dbaede7531d660ff` |

Bumping a pin later is a deliberate, reviewed one-line diff against `MODEL_REVISIONS` — fetch the new SHA from the HF Hub API (`GET /api/models/<repo_id>`, the `"sha"` field), don't guess it. `src/train_sentiment.py`/`src/train_ner.py` (standalone, one-time fine-tuning scripts, not imported by the pipeline) are out of scope for this pin — they fine-tune *from* a base checkpoint at training time, not a production inference path the §9 accuracy baseline depends on.

## Running

```bash
# CLI (direct — no server)
uv run cli/news_nlp_cli.py --limit 50
uv run cli/news_nlp_cli.py   # process every pending article, sentiment + NER + category only
uv run cli/news_nlp_cli.py --summarize   # also run c_summary/sector_summary

# API
uv run apps/news_nlp_api.py
# -> http://127.0.0.1:8003/docs
# POST /pipeline/run  {"limit": 50, "summarize": true}
```

`cli/news_nlp_cli.py` wraps `pipeline.run_pipeline()` directly with real
`--limit`/`--summarize` flags, driving sentiment + NER + category (and, with `--summarize`,
`c_summary` + `sector_summary`) in sequence. `src/pipeline.py` also still has its
own bare `if __name__ == "__main__":` (usable via `python -m pipeline [limit]`, a
positional arg instead of a flag, sentiment + NER + category only — no way to opt into
summarization from that entrypoint) — kept for backward compatibility, but
`cli/news_nlp_cli.py` is the documented entrypoint going forward.

Query results via the API: `GET /articles/{id}` now includes `"category"` and `"summary"`
keys (same shape as `"sentiment"`/`"entities"` — `"category"` is `None` until stage 3 has
processed that article, `"summary"` until stage 4 has), `GET /stats/categories` (optional
`company`/`date_from`/`date_to` filters) returns per-label article counts, and
`GET /sectors/summary` (optional `sector`/`sub_industry`/`week_start` filters) lists
`sector_summary` rows, newest week first.

## Database

The DB layer lives in **`src/news_nlp/`**, a local package vendored from
`portfolio-common` (see `docs/portfolio-common-v1-migration-plan.md`); imported
directly as `import news_nlp as db` / `from news_nlp import corrections` /
`from news_nlp.taxonomy import ...` — no re-export facade. As of
`portfolio-common` v1.2.0 it names **no** database engine: connections, the
two-tier attach, schema introspection, the row type, and every dialect SQL
fragment go through `portfolio_common.db` (`connect_url`, `two_store`,
`table_columns`, `Row`, `conn.dialect`). See
[`docs/portfolio-common-v1.2-engine-agnostic.md`](../portfolio-common-v1.2-engine-agnostic.md).
Two-tier — full detail in [`docs/db-topology.md`](../db-topology.md):

- **RESULTS store** (`$DATABASE_URL`, unset → `<repo>/data/nlp.db`; working file
  `D:\thesis\data\nlp.db`) — the NLP result tables plus a `body_text`-free
  `articles` subset. Opened read/write as schema `main`.
- **SOURCE store** (`$SOURCE_DATABASE_URL`, **required** for the text stages) —
  `articles` including `body_text` (the legacy crawl DB `urls.db`). ATTACHed
  read-only as schema `source`; never written.

The stages that read article text — sentiment, NER, category, per-article
summary (`fetch_pending_articles` / `fetch_pending_category_articles` /
`fetch_pending_company_summaries`, i.e. `cli/news_nlp_cli.py` and
`POST /pipeline/run`) — read `source.articles`. Every result write also
copies a lean `main.articles` row (metadata only) from `source` if it isn't
there yet, so the RESULTS store stays foreign-key-consistent without a
separate migration step. `db.require_source_text`
fails the run fast (before any model loads) if `SOURCE_DATABASE_URL` is unset or
points at a DB with no usable `body_text` — the old failure mode there was a
silent *"0 pending"*.

Against the RESULTS store alone (no SOURCE) the read/query surface works:
`GET /articles/{id}`, `GET /stats/categories`, `GET /sectors/summary`, the
correction (`PATCH`/`DELETE`) endpoints, and the `sector_summary` stage.

The pre-existing results were backfilled once out of `urls.db` (see
[`docs/migration-2026-09-01.md`](../migration-2026-09-01.md)); the one-shot
script that did it has since been removed now that the pipeline populates the
lean `articles` rows itself.

## Testing

```bash
uv run pytest tests/news_nlp -q
```

The suite is hermetic — every model load is monkeypatched, so no torch download, GPU, or
network is needed at test time. All tests pass (part of the repo's CI gate; see
`.github/workflows/ci.yml`).

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
first. `sector_summary`'s
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

### Experiments

Reproducing a full experiment (train a candidate checkpoint, evaluate it,
optionally publish) is one command, not a hand-chained script sequence:
`src/experiment.py`'s `ExperimentSpec` (a strictly-validated pydantic
schema) plus `cli/run_experiment.py` (`uv run cli/run_experiment.py
--config experiments/<name>.json`), reusing `run_eval` verbatim for the
evaluation step. `experiments/` holds a backfilled spec for every real
historical sentiment/NER/category/`c_summary` experiment
(`experiments/README.md` discloses the ones that don't have a runnable
equivalent). Full detail — the schema's own validation rules, the one
command, and the disclosed gaps — in `docs/evaluation.md`'s "Experiments"
section.
