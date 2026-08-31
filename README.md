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
cp .env.example .env          # then set DATABASE_URL
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
uv run python -m news_nlp.setup   # download the HF models
```

## Run

```bash
uv run apps/news_nlp_api.py            # FastAPI -> http://127.0.0.1:8003/docs
uv run cli/news_nlp_cli.py --help      # batch pipeline, no server
uv run cli/news_nlp_cli.py --limit 50  # process 50 pending articles
```

## Test / lint

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy
```

## Database

`DATABASE_URL` (a filesystem path today) selects the SQLite file; unset it
defaults to `data/nlp.db` under the repo root. The working database is
`D:\thesis\data\nlp.db`. `scripts/migrate_from_urls_db.py` populates it from
the legacy shared `urls.db`.
