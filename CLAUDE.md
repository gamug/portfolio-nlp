# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`portfolio-nlp` — the NLP stage extracted from `portfolio-data-mining`
(https://github.com/gamug/portfolio-data-mining). It runs four transformer
stages over financial-news article text and stores the results in SQLite:

| Stage | Model | Table |
|---|---|---|
| Sentiment | `ProsusAI/finbert` | `article_sentiment` |
| NER | `gamug/sec-bert-finer-ord-ner` | `article_entities` |
| Category (zero-shot, 10-label taxonomy) | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | `article_category` |
| Summarization (opt-in) | `sshleifer/distilbart-cnn-12-6` | `article_summary`, `sector_summary` |

Category taxonomy: `docs/category-taxonomy.md`. Per-module detail:
`docs/modules/news-nlp.md`.

## Layout

- `src/news_nlp/` — the package. Relative imports internally
  (`from .categories import ...`).
  - `db.py` — SQLite access layer: `connect()`, `init_schema()`, read/write
    helpers, and the read-only query helpers behind the FastAPI query
    endpoints. `DB_PATH` is computed at import from `$DATABASE_URL`
    (unset -> `<repo>/data/nlp.db`; relative values resolve against the repo
    root, not CWD).
  - `pipeline.py` — the batch pipeline (`run_pipeline`) and per-stage
    functions. `DEVICE` is CUDA if available else CPU.
  - `categories.py` / `chunking.py` / `corrections.py` — taxonomy constants,
    text chunking + char-span merging, and the correction (manual-edit)
    helpers.
  - `setup.py` — `python -m news_nlp.setup` pre-downloads the four HF models.
  - `train_ner.py` — standalone SEC-BERT NER fine-tuning script (not imported
    by the pipeline).
- `apps/news_nlp_api.py` — FastAPI service (title `news-nlp`, port 8003).
  Prepends `src/` to `sys.path` so it runs standalone.
- `cli/news_nlp_cli.py` — argparse batch driver (`--limit`, `--summarize`),
  same `sys.path` bootstrap.
- `tests/news_nlp/` — hermetic: `conftest.py` builds a minimal `articles`
  table and every model load is monkeypatched. No network or GPU.
- `tests/migration/` — covers `scripts/migrate_from_urls_db.py`.
- `scripts/migrate_from_urls_db.py` — one-shot copy of the NLP tables (+ a
  `body_text`-free `articles` subset) out of the legacy shared `urls.db`.

## Commands

```bash
uv sync
uv run python -m news_nlp.setup           # download models
uv run apps/news_nlp_api.py               # -> :8003/docs
uv run cli/news_nlp_cli.py --limit 50
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pre-commit run --all-files
```

## Config

- Lint/format: `.code_quality/ruff.toml` (root `ruff.toml` only `extend`s it).
- Types: `.code_quality/mypy.ini` (`mypy_path = src`).
- `pytest.ini`: `pythonpath = src .` (the `.` lets `tests/news_nlp/conftest.py`
  import `apps.news_nlp_api`).
- `.env` (git-ignored) holds `DATABASE_URL`; `.env.example` is the template.

## Database

Standalone SQLite. Working file: `D:\thesis\data\nlp.db`. Schema created by
`news_nlp.db.init_schema()`. Not shared with any other service.
