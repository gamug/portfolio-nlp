# NLP data migration — `urls.db` → `nlp.db`

> **Historical record.** This was a one-time backfill. `scripts/migrate_from_urls_db.py`
> and `tests/migration/` were removed once the pipeline took over lean-`articles`
> population itself (`db._ensure_article_row`, on every result write) — see
> [`docs/db-topology.md`](db-topology.md). Recover the script from git history
> (`git log -- scripts/migrate_from_urls_db.py`) if another bulk backfill is ever
> needed. The run log below is kept as-is.

**Date:** 2026-09-01
**Script:** `scripts/migrate_from_urls_db.py` (removed 2026-09-03)
**Source:** `D:\thesis\data\urls.db` (4.7 GB, read-only) — the legacy shared pipeline DB in `portfolio-data-mining`
**Destination:** `D:\thesis\data\nlp.db` (created; 2.54 GB after the run)

## Run

`python scripts/migrate_from_urls_db.py` (defaults), wall time **1m12s**, exit 0.

| table | rows copied |
|---|---:|
| `articles` (lean — no `body_text`, only rows referenced by a copied result) | 459,112 |
| `article_sentiment` | 459,112 |
| `article_category` | 459,112 |
| `article_summary` | 458,641 |
| `article_entities` | 17,666,722 |
| `sector_summary` | 3,628 |
| **total** | **19,506,327** |

`discovered_urls` / `discovery_progress` were **not** copied (out of scope — "articles processing, not articles mining").

## Verification (post-run, against `nlp.db`)

- Row counts match the `--dry-run` probe exactly.
- `articles` has 15 columns; **`body_text` absent** (the heavy crawled-text column stays in `urls.db`).
- `PRAGMA foreign_key_check` → **0 rows** (every migrated result row's `article_id` resolves).
- `PRAGMA integrity_check` → `ok`.
- `idx_article_entities_article_id` present.
- `sector_summary`: the real `urls.db` column order differs from the schema `news_nlp.db.init_schema()` builds (`format_version` / `facts_json` / `intro_text` / `processed_at` ordering), so the copy uses **explicit column lists** on both sides of `INSERT … SELECT` — verified the 3,628 rows landed values in the correct columns (dates in date columns, ints in int columns, valid JSON in `facts_json`).
- A 3-table join (`articles ⨝ article_sentiment ⨝ article_category`) returns real rows — the query endpoints that join article metadata work.

## Config

`.env` (git-ignored) now sets `DATABASE_URL=D:/thesis/data/nlp.db`. The code default (unset `DATABASE_URL`) remains `<repo>/data/nlp.db`; `.env.example` documents the absolute working path.

## Re-running

The script is idempotent (`INSERT OR IGNORE` on every copy, keyed by the source PKs). Re-running against the same `--source`/`--dest` copies 0 rows.

## Using nlp.db

`nlp.db` is a **results store** for querying/serving, not a pipeline input. Its `articles` subset has no `body_text`, so the stages that read article text — sentiment, NER, category, per-article summary (`cli/news_nlp_cli.py`, `POST /pipeline/run`) — cannot run against it (`no such column: a.body_text`). Re-running the pipeline needs `DATABASE_URL` pointed at a DB that has `articles.body_text` (the legacy `urls.db`); results land in that DB's result tables.

What works directly against `nlp.db`: the read/query endpoints (`GET /articles/{id}`, `GET /stats/categories`, `GET /sectors/summary`), the correction (`PATCH`/`DELETE`) endpoints, and the `sector_summary` pipeline stage.
