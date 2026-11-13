# news_nlp — the NLP layer of portfolio-nlp

**Extracted from** `portfolio-data-mining` (which had consolidated it from the standalone `news-nlp` project).

It reads article text crawled by `portfolio-data-mining` (the SOURCE store, read-only) and writes its results to the RESULTS store (`data/nlp.db`, or `$DATABASE_URL`; root `.env.example`). There is no scraper here — `articles` is treated as pre-populated input.

**Two pipelines live here, one replacing the other.**

- **The stepped pipeline (Work item 18, `PLAN.md` D39 and D40), the current approach.** It works on **unique article bodies** (81,961 unique `ok` bodies, D19; with the `quarantined` ones kept, D36, 309,972), finds the scope of each first, cuts a round-up into sub-news, and gives every text one main company before sentiment and category. One command per step (`cli/news_nlp_steps.py`), tracked in `pipeline_step_state`, idempotent. **State on 2026-11-12. Built:** 1.1 scope, 1.2 the cut, 2 the summary. **Pending:** 3 the companies (NER, matcher, relevance model), 4 sector, 6 sentiment (pair model + relevance stack, published privately, not pinned), 7 category (the D15 taxonomy, `docs/category-taxonomy.md`; model trained by the owner, T-174); the company snapshots are undefined. Section "The stepped pipeline" below.
- **The v1 chain (`pipeline.run_pipeline`, `cli/news_nlp_cli.py`, the FastAPI service), until the backfill (T-150).** Five stages over (URL, ticker) pairs: sentiment, NER, category, and the opt-in `c_summary` and `sector_summary`. Downstream repos still read its `article_*` tables. **The owner deprecated its sentiment, zero-shot category and `sector_summary` stages on 2026-11-12**; their description and evaluations are in `docs/legacy/news-nlp-v1-stages.md` and `docs/legacy/evaluation-v1-stages.md`. NER and the summariser stay documented here because the stepped pipeline uses them (steps 3 and 2).

## The stepped pipeline (D39, D40; the current approach)

`init_schema()` also creates `body_scope`, `sub_news`, `news_summary`, `news_company`, `news_sentiment`, `news_category`
and `pipeline_step_state` (RESULTS only; keys in `SPEC.md`), and adds the nullable `sub_news_id` and `ticker` to
`article_entities`. Its `score` stays the NER confidence. Step 1.1 writes `body_scope`, step 1.2 `sub_news` and step 2 `news_summary`, each with its state in `pipeline_step_state`; `news_company`, `news_sentiment` and `news_category` are created and not written yet (steps 3, 6 and 7 are not built).

### Step 1: scope (T-163)

`uv run cli/news_nlp_steps.py --step scope [--limit N] [--status [--titles]] [--model-dir DIR]` classifies the unique
bodies of the properly fetched corpus (`ok` and `quarantined`) into single / multi / sector / out_of_scope with the two
NLI models of `src/scope_stage.py` (a GPU is required; weights from `scripts/train_scope_final_2026_11_11.py` in
`models/scope_final`, git-ignored; pinned in `config/scope_model.json` to the public Hub repository `gamug/deberta-v3-news-scope-nli`). Run tracking is `src/news_nlp/steps.py`: a rerun at the same revision adds
nothing, `--limit` bounds the work, failures are retried, and `--status` needs no model. The revision hashes the training
record and the weights. Measurements and the dry run: `docs/scope_training.md`, section 8. Validation on a sample of `quarantined` bodies (precision per predicted scope):
`uv run scripts/draw_scope_quarantined_check_2026_11_11.py --results-db <copy of nlp.db with its -wal> [--counts multi=75 --exclude <earlier draw>.csv ... --out <new dir>] [--yes]` draws it (CPU; the Spanish aid is paid and needs `--yes`),
and `uv run scripts/score_scope_validation_2026_11_11.py [--draw main|multi_extra|all] [--key final|first]` scores the tracked tags in `gold/scope_quarantined_validation.csv`; results in section 9.

### Step 1.2: the cut (T-163)

`uv run cli/news_nlp_steps.py --step cut [--limit N] [--scope-revision REV] [--model-dir DIR] [--status]` reads the `multi`
bodies of `body_scope` (at `--scope-revision`, default the latest), finds them in SOURCE by hash (the same proper corpus as
step 1) and cuts each with the `units` extractor (read as `units5` by default, T-177; `units4` was the default of 2026-11-11, T-176) and the attribution checkpoint (`data/models/attribution_2026_11_08`,
git-ignored, a GPU is advised); one `sub_news` row per company text (`src/cut_stage.py`). The revision is the reading (`units5`, or `units4` / `units` with `--reading`) plus `-` and a hash of
the model folder's files, the company names it is asked about, its threshold and the code that cuts: any change starts a new set
and leaves the old rows. **Rows of another reading (`units-…`, `units4-…`) stay beside the `units5-…` set; `--status` lists all
(the latest with state first), and a reader (T-165, scripts) names the revision it wants, `units5-…`, and never "the latest". The run prints the reading and the revision it writes before it starts.** To cut again at the
same revision, `DELETE FROM pipeline_step_state WHERE step = 'cut' AND model_revision = '<revision>'` and `DELETE FROM sub_news WHERE extractor_revision = '<revision>'`
(deleting `sub_news` alone does nothing: the pending set comes from `pipeline_step_state`); the same text is in `--help`. State per body in `pipeline_step_state`: `done`, `skipped` ("no company text": not retried) or
`failed` (retried; the reason is `Type: message`, as in step 1). Ten failures in a row stop the run. `--reading units5|units4|units` picks how the body is read (`units5`, T-177, is the default since 2026-11-12; `units4` is the default of 2026-11-11; `units` cuts as before;
`docs/extractor_training.md`). `--fetch`,
`--allow-unpinned` and `--titles` belong to the scope step and are refused. `--status` needs no model: it shows the latest
revision with state and lists the others. **`units5` is the final reading (owner 2026-11-12) and `units5-d3e2cbfa830e` the reference cut** (161,574 sub-news of 30,305 bodies); on the owner's 143 marked pieces faithful 84.6% [77.8, 89.6] (the lower end is below the 80% bar), 89.1% [82.5, 93.4] without the 15 pieces scoped wrongly by step 1.1 (`docs/scope_training.md` §8.3); 30,305 bodies `done`, 2,250 `skipped`, 0 `failed`. The database also holds a partial `units4-d3e2cbfa830e` with the same 12 hex digits as `units5-d3e2cbfa830e`: name the full revision. **The checkpoint is local and not pinned (T-148b12; identity `8399ccf003c60143`): until it is, the cut cannot be
reproduced from the repository alone.** Measurements and the quality check: `docs/scope_training.md`, section 8.1.


### Step 2: the summary (T-164, T-166)

`uv run cli/news_nlp_steps.py --step summary (--cut-revision REV | --items single) [--items sub_news] [--limit N]` (and `--step summary --status`) gives each `single` body of
step 1 and each sub-news of the cut one text in `news_summary`: over **2,000 characters** (the owner, 2026-11-12) it is summarised by the pinned
`c_summary` (`src/news_summary_stage.py`, reusing `summary_stage.hierarchical_summarize_batch`), at or under it the text is stored as it is
(`text_source = original`). Tracked and idempotent in `pipeline_step_state`, revision `summary-…` (model, size, input, limits, code). **The sub-news come from a
cut named with `--cut-revision`, never "the latest"** (the command stops and lists the cuts that ran); `--items single` needs none. The size, its measure, how to
run it again and the first run: `docs/summary_step.md`.

### Step 3: the companies (T-164, T-165)

`uv run cli/news_nlp_steps.py --step companies --summary-revision REV --threshold T --candidates ner|ner+text [--cut-revision REV | --items single] [--limit N]` (and `--step companies --status`) reads the text step 2 left for each `single` body and sub-news, finds the universe companies it names (NER, then the matcher: a span with no ticker is ignored), scores each (text, ticker) with the relevance model `gamug/deberta-v3-company-relevance` (private, `HF_TOKEN`) and writes one `news_company` row per ticker; every ticker at the threshold or over it is main (`is_main`), an item with none is `review`. Revision `companies-…` (the two models, the threshold, the candidates, the universe, the code). **The threshold (0.5) and `--candidates` (`ner+text`) are the owner's, decided 2026-11-13 (the corrected definition of the text candidates awaits their confirmation, T-185), and have no default**; the measurements for choosing, the dry run and the limits (the relevance model never saw a sub-news): `docs/companies_step.md`.

### Steps 4 to 7 (pending)

Not built (`TASKS.md` T-167, T-168, T-172; Build 2 after Build 1, D40): **4**, the GICS sector of a `sector` body, then its summary; **6**, sentiment per (text, main company) with the pair model `gamug/FinBERT-company-sentiment` stacked with the relevance model (`docs/sentiment_training.md`; published privately, not pinned until the threshold and one 180-text gold score, T-169, T-148c4), a sector summary by `gamug/FinBERT-financial-news` with no company; **7**, category per (text, main company) with the D15 taxonomy (`docs/category-taxonomy.md`), model trained by the owner (T-174). Step 5 of D40, the company snapshots, is undefined (D28, T-152).

## Models both pipelines use

**NER** — a fine-tuned SEC-BERT-BASE model trained on FiNER-ORD, published at
[gamug/sec-bert-finer-ord-ner](https://huggingface.co/gamug/sec-bert-finer-ord-ner) →
`article_entities`. Batched `NER_BATCH_SIZE` (`src/pipeline.py`) articles per forward
pass: every article's chunks in the batch are flattened into one padded tokenizer call,
so the actual batch width is that batch's *total chunk count*, not `NER_BATCH_SIZE`
itself — see the constant's comment for why (variable per-article chunk counts, unlike
category's fixed 9-pairs-per-article width).

**Why this model.** `nlpaueb/sec-bert-base` over a generic NER checkpoint (spaCy,
`bert-base-NER`): it's domain-pretrained on 260,773 SEC 10-K filings (1993-2021) with its
own ~30,000-subword financial vocabulary, not the generic BERT vocabulary (Loukas,
Fergadiotis, Chalkidis et al. 2022, *"FiNER: Financial Numeric Entity Recognition for
XBRL Tagging"*, arXiv:2203.06482) — financial entity mentions (tickers, filing
terminology, numeric-heavy contexts) benefit from vocabulary a generic model never saw,
including subword handling for numbers specifically relevant to this project's own
2026-09-12 subword-fragmentation fix (a word-boundary bug producing bogus spans off
things like `"3M"`). `gtfintechlab/finer-ord` as the fine-tuning dataset: 201
manually-annotated financial news articles (116,721 tokens, `PER`/`LOC`/`ORG` BIO
tagging — Shah, Gullapalli et al. 2024, *"FiNER-ORD: Financial Named Entity Recognition
Open Research Dataset"*, arXiv:2302.11157), whose general-NER label set matches this
project's own `article_entities` need (named entities in financial *news*) rather than a
filings-specific tag set like XBRL, making it directly usable for fine-tuning SEC-BERT
toward this project's actual task.

**The summariser, `sshleifer/distilbart-cnn-12-6`** (step 2 of the stepped pipeline, and the v1 `c_summary` stage) — in the v1 stage, one abstractive summary per article (`sshleifer/distilbart-cnn-12-6`),
generated from the article's body plus its already-computed sentiment/entities →
`article_summary`.

**Why this model.** A distilled BART over the full `bart-large-cnn` or a modern
LLM-based summarizer: the 6GB-VRAM / one-model-at-a-time budget this whole pipeline is
built around (`SPEC.md` NR-001), and no per-call API cost for a batch job over hundreds
of thousands of articles. `distilbart-cnn-12-6` (12 encoder / 6 decoder layers vs.
`bart-large-cnn`'s 12/12, ~1.24x faster inference) reports near-identical quality on its
own model card — ROUGE-2 21.26 / ROUGE-L 30.59, against `bart-large-cnn`'s 21.06 / 30.63
— essentially the same summarization quality at a fraction of the size, the real
trade-off this choice buys. That size is not free, though, and connects directly to a
weakness already measured here: the model's own distillation data is CNN/DailyMail
(short news-article summarization pairs), and its generation defaults
(`max_length=142`/`min_length=56`, matched by this project's own
`SUMMARY_MAX_OUTPUT_TOKENS`/`SUMMARY_MIN_OUTPUT_TOKENS`) were tuned against that short,
CNN/DailyMail-style news — never retuned for longer, denser financial-news text. That's
a plausible root cause behind the `mean_coverage` gap this project measured and accepted
as a deliberate completeness-vs-correctness trade (2026-09-14), not an unrelated fact
sitting next to it.

## The v1 chain (until the backfill)

Stages 1–3 (sentiment, NER, category) always run; 4–5 (`c_summary`, `sector_summary`) only with `--summarize` (CLI) or `"summarize": true` (`/pipeline/run`). Each stage owns its results table and processes only the rows missing from it. Stages 1, 3 and 5 are deprecated (`docs/legacy/news-nlp-v1-stages.md`); NER and the summariser are described above.

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

### Running the v1 chain

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

## Setup

```bash
uv sync                          # torch is a pinned direct dep (cu124 wheel index) — no manual install
uv run python -m setup           # pre-fetch the four HF models into the local cache (one-time, safe to re-run)
```

### Model pins

Production model selection (name, commit SHA, and batch size per ML stage) lives in the git-tracked `config/pipeline_models.json`, schema-validated by `src/pipeline_config.py`'s `PipelineModelsConfig` (added 2026-09-21, `SPEC.md` FR-018, `PLAN.md` Work item 14 — supersedes the `pipeline.py`-literal-constants design from 2026-09-14, `SPEC.md` §13 item 4). `pipeline.py` loads this file at import time and re-exposes it as the same `SENTIMENT_MODEL`/`NER_MODEL`/`CATEGORY_MODEL`/`SUMMARY_MODEL`/`MODEL_REVISIONS`/`NER_BATCH_SIZE`/`CATEGORY_BATCH_SIZE`/`SUMMARY_BATCH_SIZE` module attributes as before, so every existing monkeypatch/test/script call site is unaffected. Every `from_pretrained`/`snapshot_download` call (in `src/pipeline.py` and `src/setup.py`) is pinned to the commit SHA this file names — an upstream push to any of these repos no longer changes results silently. Deliberately a fixed, git-tracked file with no CLI/env runtime override (constitution AI-behavior #1: "model selection is pinned, not dynamic") — changing a model still means editing this file through normal PR review, only the format moved from Python to JSON:

| Model | Repo | Pinned SHA |
|---|---|---|
| Sentiment (v1 stage, deprecated; also the base of the pair model and the sentiment of sector summaries) | `gamug/FinBERT-financial-news` | `93863fcb7252874e7c0339081b34f691f9e17ff6` (v4, class-weighted — `PLAN.md` Work item 9, published 2026-09-19) |
| NER | `gamug/sec-bert-finer-ord-ner` | `ba7b9e43e4aa023ec5691f955b276dc58158354c` (batch size 8) |
| Category (v1 zero-shot stage, deprecated; also the base checkpoint of the scope and relevance models) | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8` (batch size 8) |
| Summarization (step 2 and v1 `c_summary`) | `sshleifer/distilbart-cnn-12-6` | `a4f8f3ea906ed274767e9906dbaede7531d660ff` (batch size 4) |

The scope model of step 1 is pinned in a second file, `config/scope_model.json` (repository `gamug/deberta-v3-news-scope-nli`, public; commit; and the step revision stored in every `body_scope` row), because no stage of the v1 pipeline uses it and the pin carries a step revision the four stages do not have. `uv run cli/news_nlp_steps.py --step scope --fetch` downloads that commit into an empty `models/scope_final` (no token needed) and deletes a copy whose step revision is not the pinned one; the step refuses a model that is not the pinned one unless `--allow-unpinned`.

Sentiment has no `batch_size` entry — `sentiment_stage.SentimentInference.batch_size()` is a hardcoded `1`, not a tunable constant `pipeline.py` reads, so there is nothing for a config field to move.

Bumping a pin later is a deliberate, reviewed one-line diff against `config/pipeline_models.json` — fetch the new SHA from the HF Hub API (`GET /api/models/<repo_id>`, the `"sha"` field), don't guess it. `src/train_sentiment.py`/`src/train_ner.py` (standalone, one-time fine-tuning scripts, not imported by the pipeline) are out of scope for this pin — they fine-tune *from* a base checkpoint at training time, not a production inference path the §9 accuracy baseline depends on.

## Database

The DB layer lives in **`src/news_nlp/`**, a local package vendored from
`portfolio-common` (see `docs/legacy/portfolio-common-v1-migration-plan.md`); imported
directly as `import news_nlp as db` / `from news_nlp import corrections` /
`from news_nlp.taxonomy import ...` — no re-export facade. Since
`portfolio-common` v1.2.0 it names **no** database engine: connections, the
two-tier attach, schema introspection, the row type, and every dialect SQL
fragment go through `portfolio_common.db` (`connect_url`, `two_store`,
`table_columns`, `Row`, `conn.dialect`). See
[`docs/legacy/portfolio-common-v1.2-engine-agnostic.md`](../legacy/portfolio-common-v1.2-engine-agnostic.md).
Two-tier — full detail in [`docs/db-topology.md`](../db-topology.md):

- **RESULTS store** (`$DATABASE_URL`, unset → `<repo>/data/nlp.db`; working file
  `D:\thesis\data\nlp.db`) — the NLP result tables plus a `body_text`-free
  `articles` subset. Opened read/write as schema `main`.
- **SOURCE store** (`$SOURCE_DATABASE_URL`, **required** for the text stages) —
  `articles` including `body_text` (the legacy crawl DB `urls.db`). ATTACHed
  read-only as schema `source`; never written.

The stepped pipeline's steps (`cli/news_nlp_steps.py`) open the same two-tier connection and read the unique bodies of `source.articles`. The v1 stages that read article text — sentiment, NER, category, per-article
summary (`fetch_pending_articles` / `fetch_pending_category_articles` /
`fetch_pending_company_summaries`, i.e. `cli/news_nlp_cli.py` and
`POST /pipeline/run`) — read `source.articles`. Every result write also
copies a lean `main.articles` row (metadata only) from `source` if it isn't
there yet, so the RESULTS store stays foreign-key-consistent without a
separate migration step. When a SOURCE is attached (not on single-file runs or the
no-SOURCE endpoints), `db.check_schema` first checks it read-only before any query:
`PRAGMA user_version` at least 3 (`MIN_SOURCE_SCHEMA_VERSION`; a minimum, not an exact
match, since upstream's migrations are additive) and the columns this repo reads
(`SOURCE_REQUIRED_COLUMNS`), else `SourceSchemaError`; a version above the latest seen
(5) warns once per process. `POST /pipeline/run` runs it before its background worker
and answers 503 with the message; the CLIs exit with it on one line. Detail:
`docs/db-topology.md`. Then `db.require_source_text`
fails the run fast (before any model loads) if `SOURCE_DATABASE_URL` is unset or
points at a DB with no usable `body_text` — the old failure mode there was a
silent *"0 pending"*.

Against the RESULTS store alone (no SOURCE) the read/query surface works:
`GET /articles/{id}`, `GET /stats/categories`, `GET /sectors/summary`, the
correction (`PATCH`/`DELETE`) endpoints, and the `sector_summary` stage.

The pre-existing results were backfilled once out of `urls.db` (see
[`docs/legacy/migration-2026-09-01.md`](../legacy/migration-2026-09-01.md)); the one-shot
script that did it has since been removed now that the pipeline populates the
lean `articles` rows itself.

## Triage rules (Work item 18, development only)

- `news_nlp/triage.py` — rule-based scope triage over unique article bodies: `body_hash`,
  `CompanyResolver` (names, aliases, explicit ticker forms, NER `ORG` spans; each mention counted
  once), boilerplate stripping, `subjects`, `is_roundup`, `scope_candidate`, `gold_scope` (the
  owner's tags mapped to the six scopes) and `wilson`. Not a pipeline stage: it sized the corpus
  and was the baseline of the scope classifier, which replaced it (67.2% against 83.6% on the held-out
  200, `docs/scope_training.md`). Agreement with the owner's 720 tags is 0.70, a
  development figure (`docs/evaluation.md`, 2026-10-29 follow-up). `CompanyResolver` (the matcher) is
  current: the cut and the counts read companies through it.
  **Common words (D38, PR #155):** a hyphen, a dash, an apostrophe (also the backtick and the acute
  accent) in a name reads as the space it replaced ("Coca-Cola", "Casey's"), a line break does not.
  The first word of a company's name is not a mention when it is an everyday word, a place or a
  first name (`_COMMON_FIRST_WORDS`, part of `_ALIAS_STOP`: "Federal", "Boston", "Ralph", ...); the
  short form of the name ("Federal Realty", "Take-Two", "Jack Henry"; a comma only before a legal
  word, never a full stop) and a few known names (`_KNOWN_NAMES`: "Abbott Labs") are. Surnames and
  nouns (`_CORROBORATED`: Bloom, Ross, Wells, Morgan, Block, Kraft, Warner, ...) count only when the
  body names the company another way (full name, short form, ticker form): in `resolve` (counting,
  the stored NER spans too) and in `attribution_units.GuardedResolver` (the attribution), through
  one method, `CompanyResolver.firm_spans`. Names that are also lower-case nouns (`_WORD_NAMES`:
  Block, Travelers, Williams, Workday, Booking, Visa, Flex, Cognizant, Coherent, ...) are read
  capitalised only; bare tickers that are words (`_WORD_TICKERS`: COO, MET, ...) only with an
  explicit form. `mention_spans` itself keeps the corroborated aliases (it places the masks of single
  sentences, where the full name is often absent): the guard decides those with the whole body.
  Names with initials, an exclamation mark or an accent are found as written ("J.B. Hunt", "JB Hunt",
  "U.S. Bancorp", "Yum! Brands", "Estée Lauder", also with the accent decomposed): the initials are
  upper case ("us" is not "U.S."), a full stop follows an initial, spaces between initials are
  optional, `squash_initials` gives the key of the name without them. A disclosure or ticker list
  ("Trader A Owns (INTC)", "is long X;", "have positions in X") is told from a sentence by
  `triage.in_disclosure`; `CompanyResolver.disclosure_only` gives the tickers a text names only that
  way, and `counted_matches` / `is_ticker_form` the mentions `resolve` counts and their kind.
- `scripts/company_news_counts_2026_11_09.py` (T-160, PLAN D36, D38) — the news counts per company for
  the owner's partner: for three corpora (`ok_`, `quarantined_only_`, `all_`) and four measures:
  `*_bodies_naming`, `*_bodies_subject`, `*_bodies_single`, and `*_bodies_disclosure` (of the bodies
  that name the company, those that name it only in a disclosure; `subject` and `single` are not
  adjusted for it, the README gives the run's own shares). `scripts/matcher_before_after_2026_11_09.py`
  compares the matcher with a git revision (`--base`), checks that each company is found by its own
  name (`--self-check`), measures the disclosure share (`--disclosures`) and prints the labels of
  `docs/matcher_labels_2026-11-09.csv` with their context (`--render`) and recomputes the precision of
  its sets (`--labels-summary`).
- `docs/matcher_labels_2026-11-09.csv` (PLAN D38) — what was read to measure the matcher, one reader
  (the developer), no news text. Columns: `set` (the sample: `r<residue>_<what>`, residue of
  `id % 60`), `n` (its number in the set), `article_id` (SOURCE), `ticker`, `word` (the matched
  text), `rule` (which rule matched), `offset` (character offset of `word` in the body as stored),
  `residue`, `seed` (of the random draw; `all` = every mention), `label`, `note`. Labels by kind of
  set: sets of **mentions** (`*_owner15`, `*_alias`, `*_short_corr`, `*_name_ticker`, `*_dropped_*`,
  `*_ten_names_new`): `real` (the company), `not`, `ambiguous` (counted as wrong in the tables),
  `unreadable` (left out); the "precision" of a `*_dropped_*` set is the share of dropped mentions
  that were the company, i.e. what the rule costs. Sets of **flagged** pairs (named only inside a
  disclosure by the detector): `disclosure`, `not-disclosure`, `ambiguous`, `unreadable`. Sets of
  **unflagged** ticker-only pairs: `prose-real`, `not-company`, `disclosure-missed`, `ambiguous`,
  `unreadable` (counts only). `r59_unflagged_by_tightening` was drawn under the round-6 detector (what the next version stopped flagging); `r17_*` were read on a residue none of the edits was chosen from. A test reads the file, checks every label against its set and pins the
  figures D38 quotes.
- `news_nlp/gold_review.py` — writes the owner's review and key CSVs; refuses to overwrite a review
  file that already holds a `human_label`.
- `news_nlp/retag.py` — gold v2 (T-146b): the re-tag file of the 720 (v1 tag shown, LLM label hidden), D15 label set and the check of the returned file (`scripts/retag_gold_2026_10_29.py export|check`).
- `news_nlp/attribution_model.py` — the trained attribution model of step 4 (T-148b3, D27; **not
  published, not pinned**): a pair classifier, *is this sentence about this company?* (the company's
  name, the previous sentence and the sentence in; trained on `lead` paragraphs only, with the
  company's name masked). A sentence that names universe companies belongs to them; one that names
  none goes to the best-scoring company of the body if the score is at least 0.5, else to nobody.
  `CompanyResolver.mention_spans` gives the mentions with positions (what is masked); `CODE_MODULES`
  is the code the model's identity covers. Trained by `scripts/train_attribution_2026_11_08.py`
  (`--eval-only DIR` re-scores a trained model and keeps the training-time report); used as the
  `model` extractor by `scripts/draw_faithfulness_2026_11_06.py --attribution-model DIR_OR_ID`
  (a Hub id needs `--attribution-revision`). Figures: PLAN D27 and `docs/evaluation.md`.
- `news_nlp/attribution_units.py` — what the attribution reads (T-148b4, D27): `strip_footer` (credit
  and "read more" lines anywhere, a "more from" header with its link titles, a disclosure block that
  opens in the last 40% of the body, a run of footer lines at the end; label words are anchored and a
  long paragraph is story, a legal notice is not) and `attribution_units` (a line, a bullet, a
  sentence, a part between semicolons, a sentence of two companies cut between them).
  `model_company_texts(..., units=True)` reads it, and then the candidates are the companies of the
  body without its footer; the faithfulness draw registers it as the `units` extractor.
  `chunking.split_sentences` is not changed (sentiment and NER use it). Tools:
  `scripts/audit_footer_2026_11_08.py` (seeded audit of the footer rules on the corpus),
  `draw_faithfulness --footer-only` (the companies a body names only in its footer, with the revision
  of the rules) and `draw_faithfulness --stale` (which marked rows the code no longer writes; exit
  code 1 if any, +2 if an extractor of the marks was not checked, `--only NAMES` limits it; keeps the
  last check of each extractor in `faithfulness_stale.json`, which `report_wi18 faithfulness` quotes).
- The guard and the headers (T-148b5, T-148b6, `attribution_units.py`; `units2` and `units3` of
  `attribute_sentences(reading=...)`, one of `sentences`, `units`, `units2`, `units3`, anything else
  is a `ValueError`): `GuardedResolver` counts a company named by a first word or by a surname or noun ("Bloom",
  "Williams", "Block") only when the body also has its full name, its short form or a ticker form
  (`CompanyResolver.firm_spans`; PLAN D38: the scope rules count a body with `resolve`, which applies
  the same rule);
  `header_owners` gives a "NAMES - text" line to the universe companies among the names, to nobody
  when the names are of a company outside the universe, and does not take a dateline or a label in
  capitals ("DETROIT - ..."), nor a dash inside a sentence (the text after it does not start with a capital, a
  digit or a quote), for a header; `units3` also gives the piece of such a line to the
  universe companies it names itself. **Known loss:** a company named only by a short form has no
  text, and is not counted as an omission; `draw_faithfulness --exempt` lists those companies (and
  the footer-only ones) in `faithfulness_exempt.json` and, for the owner to read, in
  `faithfulness_exempt_to_read.csv`, and `report_wi18 faithfulness` gives the bar also with them
  counted as unfaithful (every company named) and, once read, on the real companies. Audit of the
  header rule: `scripts/audit_headers_2026_11_08.py`. Text measures of the company texts (characters, mixed and nameless sentences, paired
  intervals): `scripts/text_measures_2026_11_08.py`. What replaced the marked texts the code no longer writes (per
  extractor, read-only, stops on an unfinished file): `scripts/dead_text_replacement_2026_11_10.py`.
- Scripts: `scripts/size_triage_2026_10_28.py` (`size`, `gold`), read-only;
  `scripts/export_gold_review_2026_10_26.py`.

- `scripts/scope_title_baseline_2026_11_10.py` (T-147, development only) — the scope classifier from the title, measured
  on the 200 held-out bodies, weighted to the corpus. Runs: no flag (TF-IDF + logistic regression on the title: 66.4%);
  `--body-chars N` (title + first N characters: 65.7% / 62.5% / 57.1% for 300 / 600 / 1500); `--company-features
  [--first M] [--count-scale S | --choose-scale]` (adds the capped count of distinct universe companies in the title,
  the first M characters and the body, scaled by S; `--choose-scale` picks S by cross-validation on the training rows
  only; 68.1% at S = 0.3, 66.9% at S = 1). It prints the paired differences against the title-only model and the rules
  (exploratory per class), the learned count weights, and the SHA-256 of `gold/universe.csv` and `triage.py`. Needs
  `scikit-learn` and `scipy` (eval group). Figures and the owner's options: `.specify/memory/TASKS.md`, T-147.

## Sentiment on the owner's summaries (Work item 18, T-148c, development only)

A candidate sentiment model, fine-tuned from the pinned one on the owner's tagged summaries and read as the
**pair** `[CLS] company [SEP] text [SEP]` (D35). It is not in the pipeline: the stage still runs the pinned
model, and the gold is scored once, by `scripts/score_sentiment_gold_2026_11_07.py --pair` (T-148c4). Numbers
and decisions are in `TASKS.md` T-148c3 and `PLAN.md` D35, not copied here. **The metrics that are
recorded and that select a model are Precision, Recall and ROC OVR per class (owner, 2026-11-09, `PLAN.md` D37):
the validation score of a run is the mean of those nine figures; accuracy is a description only.**

- `src/sentiment_pair.py` — the pair's encoding and the routing rule (`reads_as_pair`), shared by the trainer,
  the draw's `predict` and the scorer; step 6 will use it (the stack in `sentiment_stage`, T-148c3d; the step, T-168; not built).
- `src/sentiment_pair_train.py` — the recipe: the loader of the 450 marks, the replay of earlier sentences (only
  those that name their article's company, not of a gold, held-out or marked body or headline), batches that
  never mix summaries with sentences, one run, the grid of 27, D35's tie rule, and a fingerprint of the data,
  recipe and code. Its revision is part of the fingerprint, so a change to it invalidates the runs.
- `src/sentiment_pair_report.py` — how a run is *shown*, kept apart so that changing it does not invalidate
  the runs: the precision, recall and ROC OVR of each class and the selection score (weighted, with a bootstrap
  interval, because the slice is drawn per class the pinned model predicted), the paired weighted gain over the
  pinned model, D32's verdicts on the negative class, the pin rule (`pin_decision`), and the raw description
  of the slice (confusion matrix, McNemar).
- `scripts/train_sentiment_pair_2026_11_09.py` — `--baseline` (the pinned model on the validation slice),
  `--time-one`, `--grid [--fresh]` (27 runs, resumable over the same fingerprint), `--final` (one run, seed 4,
  the model kept), `--report` (from the predictions on disk), `--mlflow` (experiment
  `sentiment_pair_summaries`, runs tagged with the fingerprint). `--final-all [--models DIR]` (T-148c3b): one run on
  every row of **both** gold files (`MARK_SETS`: `gold/sentiment_train.csv` and `sentiment_train_2.csv`) plus the
  replay, no validation slice, always logged to experiment `sentiment_pair_final_all` (`--mlflow` does not apply; the
  store is `MLFLOW_TRACKING_URI` or `./mlruns`, printed); the grid and `--final` read only the first file. The
  experiments after the grid are in `docs/sentiment_training.md`.
- `scripts/final_models_2026_11_10.py` (T-169) — `relevance` (the final relevance model on the 809 pair rows; needs a GPU and
  only tracked files, `--order-of` repeats the published key order) and `stack` (the logistic regression on the
  out-of-fold scores -> `config/sentiment_stack.json`; `apply_stack` is the formula the model card shows).
- `scripts/publish_final_models_2026_11_10.py` (T-169) — cards and the private upload of both models (`--dry-run`
  needs no token; `--check` is a read-only check of the live repositories); writes the revisions and card hashes to `config/published_models.json`. Nothing is pinned. Details and limits:
  `docs/sentiment_training.md`, "Published".
- Where things are, all git-ignored: `data/sentiment_pair/` (`grid_results.json`, the record of the numbers,
  `baseline.json`, `replay.json`, `predictions/<tag>__<run>.csv`), `models/sentiment_pair/final`, `models/sentiment_pair_all/final` (`--final-all`), `mlruns/`.

## Testing

```bash
uv run pytest tests/news_nlp -q
```

The suite is hermetic — every model load is monkeypatched, so no torch download, GPU, or
network is needed at test time. All tests pass (part of the repo's CI gate; see
`.github/workflows/ci.yml`).

**Reproducing CI exactly.** CI installs `--group dev --group eval` and runs, in this order, ruff, `ruff format --check`,
mypy and pytest (`.github/workflows/ci.yml`):

```bash
uv sync --group dev --group eval
uv run --no-sync mypy --config-file=.code_quality/mypy.ini
```

The pre-commit mypy hook runs `uv run mypy`, which installs only the default `dev` group: in a checkout or worktree
whose environment has no `eval` group (`mlflow`, `strands-agents`) mypy sees those packages as untyped, so the hook
can pass where CI fails (it did on #154: `mlflow.search_runs` is `list | DataFrame` to the typed package). Before
pushing code that imports `mlflow`, run the two commands above. A stale `.mypy_cache` after installing the group can
make mypy stop with an internal error; delete it and run again.

## Evaluation

`news_nlp.eval` (LLM-as-judge over stored predictions, MLflow, the `eval_*` tables; separate `eval` dependency group, an OpenAI-compatible endpoint in `LLM_API_KEY` / `LLM_MODEL` / `LLM_URL`) evaluates the v1 stages' stored outputs: sentiment, category, NER, `c_summary`, and the `sector_summary` intro. Its model-scoring step reuses each stage's FTI `Inference` subclass and can score a candidate model live (`--candidate-model` / `--candidate-revision`). The models of the stepped pipeline are measured on the owner's gold sets instead (`docs/gold_datasets.md`; the figures in `docs/scope_training.md`, `docs/extractor_training.md`, `docs/sentiment_training.md`, `docs/summary_step.md`): LLMs label and evaluate, never run inside the pipeline (D12). The `--check-regression` gate is out of the MVP (owner, 2026-11-10).

```bash
uv sync --group eval
uv run cli/news_nlp_eval.py --stage all --sample-size 80
uv run mlflow ui
```

Full detail — sampling, per-stage metrics, the "judge is a model, not gold" caveat — in `docs/evaluation.md`; the v1 stages' own follow-ups in `docs/legacy/evaluation-v1-stages.md`.

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
