# portfolio-nlp

Financial-news NLP pipeline, extracted from
[`portfolio-data-mining`](https://github.com/gamug/portfolio-data-mining):

- **Sentiment** — FinBERT (`ProsusAI/finbert`)
- **NER** — SEC-BERT (`gamug/sec-bert-finer-ord-ner`)
- **Category** — zero-shot DeBERTa-v3 against a fixed 10-category taxonomy
  (`docs/category-taxonomy.md`)
- **Summarization** (opt-in) — distilbart per-article + per-sector-week roll-ups

Reads an `articles` table and writes `article_sentiment`, `article_entities`,
`article_category`, `article_summary`, `sector_summary`, keyed by `article_id`.

## Setup

```bash
uv sync
cp .env.example .env          # then set DATABASE_URL (+ SOURCE_DATABASE_URL to run the pipeline)
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
uv run python -m setup            # download the HF models
```

## Run

```bash
uv run apps/news_nlp_api.py            # FastAPI -> http://127.0.0.1:8003/docs
uv run cli/news_nlp_cli.py --help      # batch pipeline, no server
uv run cli/news_nlp_cli.py --limit 50  # process 50 pending articles — needs SOURCE_DATABASE_URL set to a DB with articles.body_text (see docs/db-topology.md)
```

## Test / lint

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy --config-file=.code_quality/mypy.ini
```

## Database

Two-tier (see `docs/db-topology.md`):

- **`DATABASE_URL`** — the RESULTS / serving store (result tables + a
  `body_text`-free `articles` subset). A filesystem path today; unset defaults to
  `data/nlp.db` under the repo root. Working file: `D:\thesis\data\nlp.db`.
- **`SOURCE_DATABASE_URL`** — the read-only SOURCE store, which must have
  `articles.body_text` (the legacy crawl DB `urls.db`). **Required** by the
  text-reading stages (sentiment / NER / category / c_summary); it is ATTACHed
  read-only and never written. Not needed for serving/query endpoints or the
  `sector_summary` stage.

A pipeline run reads text from SOURCE and writes results — plus a lean
`articles` row per processed article — into RESULTS, so the RESULTS store stays
self-consistent with no separate migration step. (The pre-existing results were
backfilled once from `urls.db`; see `docs/migration-2026-09-01.md`.)
