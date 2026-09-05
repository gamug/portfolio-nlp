# Database topology: the two-tier contract

The DB layer lives in **`src/news_nlp/`**, a local package owned by this repo
(vendored from `portfolio-common`'s `business_folders/news_nlp/` staging area —
see `docs/portfolio-common-v1-migration-plan.md`). It's imported directly by
`pipeline.py` / `apps/news_nlp_api.py` / tests (`import news_nlp as db`, `from
news_nlp import corrections`, `from news_nlp.taxonomy import ...`) — no
`portfolio_common.news_nlp` and no local re-export facade.

**No engine-specific code lives here.** As of `portfolio-common` v1.2.0 this
package names no database engine: `NewsNlpDatabase` subclasses
`portfolio_common.db.TwoTierDatabase` (itself a `Database`), connections are
opened via `Database.connect_url` (a path today, a `scheme://` URL if the
engine ever changes), the `main`/`source` schema handle is
`TwoTierDatabase.read_schema` (kept available under the old name
`db.articles_rel`), schema introspection goes through
`Database.table_columns` / `.ensure_columns` / `.create_schema`, the lean
cross-store row copy through `Database.copy_row_lean`, and every dialect SQL
fragment (the upsert verb, `strftime`/`date(...)` week bucketing, `GLOB`, the
`AUTOINCREMENT` DDL token) through `conn.dialect`. Swapping the engine is a
change to `portfolio-common` alone — see `docs/engine-agnostic-rollout.md`
and `docs/portfolio-common-v1.2-engine-agnostic.md`. `portfolio-common` stays
pinned in `pyproject.toml`'s `[tool.uv.sources]`.

`portfolio-nlp` uses **two** SQLite databases with distinct roles, selected by
two environment variables and never conflated in code.

| | **SOURCE** | **RESULTS** |
|---|---|---|
| env var | `SOURCE_DATABASE_URL` | `DATABASE_URL` |
| resolver | `news_nlp.env.source_db_path()` | `news_nlp.env.results_db_path()` |
| default | none — **required** for the text-reading stages | `data/nlp.db` |
| opened | read-only (`file:…?mode=ro`), `ATTACH`ed as schema `source` | read/write, schema `main` |
| holds | `articles` **including `body_text`** (written by the upstream crawler) | the 5 result tables (`news_nlp.schema.SCHEMA`) + a lean `articles` subset (**no `body_text`**) |
| written by this repo | never | result rows, plus one lean `articles` row per processed article |

Path resolution (`news_nlp.env`): an **absolute** value is used as-is; a
**relative** value is left relative — resolved against the process's current
working directory. Unset → `results_db_path()` falls back to `data/nlp.db`,
`source_db_path()` → `None`.

## Upgrading from the single-`DATABASE_URL` setup

Before the two-tier contract there was one knob: you pointed `DATABASE_URL` at
`urls.db` to run the pipeline and at `nlp.db` to serve. That recipe now **fails
fast by design** — a text-reading run with no `SOURCE_DATABASE_URL` raises

> `SOURCE_DATABASE_URL is not set. The text-reading pipeline stages … require a
> read-only source database that has articles.body_text …`

Fixes, by what you were doing:

| Old | New |
|---|---|
| `DATABASE_URL=…/urls.db` to run the pipeline | `SOURCE_DATABASE_URL=…/urls.db` **and** `DATABASE_URL=…/nlp.db` (results land in `nlp.db`, not back in `urls.db`) |
| `DATABASE_URL=…/nlp.db` to serve / query | unchanged — serving never needed a SOURCE |
| a single file that genuinely has `body_text` **and** the result tables | set **both** vars to that one path (see *Single physical file* below) |
| a `.env` with only `DATABASE_URL` | add `SOURCE_DATABASE_URL` (see `.env.example`) |
| `POST /pipeline/run` on a serving box | that box now needs `SOURCE_DATABASE_URL` too, or the run ends `status: "error"` |

No data migration is involved — the switch is purely which env var points where.
`nlp.db` keeps every result row it already had; the pipeline now also writes a
lean `articles` row per processed article, so it stays self-consistent without
`scripts/migrate_from_urls_db.py` (removed).

## How a pipeline run uses both

`pipeline.run_pipeline` → `news_nlp.db.connect_pipeline()`:

1. opens the RESULTS store read/write (`main`) as a `NewsNlpDatabase` (a
   `portfolio_common.db.TwoTierDatabase` subclass) via
   `portfolio_common.db.connect_two_store(results, source, alias="source",
   factory=NewsNlpDatabase, foreign_keys=True)` — URI-mode, so a later
   read-only `ATTACH` on it is honored;
2. unless `SOURCE_DATABASE_URL` resolves to the **same path** as `DATABASE_URL`,
   `connect_two_store` `ATTACH`es the SOURCE store read-only as `source` and
   sets `db.read_schema` (aliased `db.articles_rel`) to `"source"` —
   delegating the ATTACH and its stale-WAL preflight to
   `Database.attach(source, "source", read_only=True)`.
3. `news_nlp.db.require_source_text()` then fails fast (before any model loads) if
   that `articles` table has no `body_text` column or no non-empty `body_text`
   row — the common "pointed a text stage at the results store" mistake, which
   used to surface as a silent *"0 pending"*.

The three `body_text` readers (`fetch_pending_articles`,
`fetch_pending_category_articles`, `fetch_pending_company_summaries`) qualify
`articles` with `db._articles_rel(conn)` (Allowlist-checked against `"main"` /
`"source"` -- see `news_nlp/db.py`), so they read `source.articles` during a
two-tier run and `main.articles` otherwise. All result-table joins stay in
`main`. Every result-table write (`write_sentiment` / `write_category` /
`write_entities` / `write_company_summary`) first `db._ensure_article_row` →
`Database.copy_row_lean` upserts the lean `articles` row (metadata only,
SOURCE ∩ RESULTS columns minus `body_text`) from `source` into `main`, so the
RESULTS store stays foreign-key-consistent on its own — no separate migration
step.

The `sector_summary` stage and every query/correction endpoint read only result
tables + `articles` metadata, never `body_text`, so they need **no** SOURCE — a
plain `news_nlp.db.connect()` is enough.

## Operator recipes

**Two-tier run (the normal case).** SOURCE is the crawl DB, RESULTS is `nlp.db`:

```bash
export SOURCE_DATABASE_URL=D:/thesis/data/urls.db
export DATABASE_URL=D:/thesis/data/nlp.db
uv run cli/news_nlp_cli.py --limit 50        # or --summarize
# one-off override without env vars:
uv run cli/news_nlp_cli.py --source-db urls.db --results-db nlp.db --limit 50
```

**Serving / queries only.** No SOURCE needed:

```bash
export DATABASE_URL=D:/thesis/data/nlp.db
uv run apps/news_nlp_api.py                  # GET /articles, /stats/*, /sectors/summary, PATCH/DELETE
```

`POST /pipeline/run` still needs `SOURCE_DATABASE_URL`; without it the run
finishes with `status: "error"` and the `SOURCE_DATABASE_URL is not set…`
message on `GET /pipeline/status` (the POST itself still returns 202).

**Re-run / catch up.** Same as a two-tier run — every stage only processes rows
missing from its result table, so re-pointing at the same SOURCE/RESULTS pair
resumes where it left off.

**Single physical file.** If one file genuinely has both `body_text` and the
result tables, point *both* vars at it:

```bash
export SOURCE_DATABASE_URL=D:/thesis/data/urls.db
export DATABASE_URL=D:/thesis/data/urls.db   # same path
```

`connect_two_store` (via `connect_pipeline`) sees the paths match, skips the
ATTACH, and reads/writes the one file (`read_schema` / `articles_rel` stays
`"main"`, the lean upsert is a no-op).

## External consumers

`list_articles` / `get_article_detail` / the `*_stats` functions are shaped for
`portfolio-nlp`'s own FastAPI endpoints (paginated, dict-per-call). Within this
repo, `fetch_processed_articles(conn, limit=...)` is the read-only export join
for every fully-processed article as plain rows, still using
`connect_pipeline(results_db=..., source_db=...)` to hide the SOURCE/RESULTS
split from the caller the same way the pipeline's own readers do.

As of `portfolio-common` v1.1.0, that join's SQL is no longer this repo's alone
to own: it delegates to `portfolio_common.news_export.fetch_processed_articles`,
the one piece of this read contract genuinely shared with a consumer outside
`portfolio-nlp` — `portfolio-knowledge-graph`'s ETL, which imports
`portfolio_common.news_export` directly rather than depending on this repo or
carrying its own copy of the query. See
`docs/portfolio-common-v1.1-news-export.md` for why. Don't import
`news_nlp.db._articles_rel` (or any other underscore-prefixed name) directly
from outside this package to build a bespoke query — it's internal; ask for
(or add) a named export in `portfolio_common.news_export` instead if another
repo needs a second shared read.

## Caveats

- **Stale WAL on the SOURCE.** A `mode=ro` open cannot replay a write-ahead log,
  so a SOURCE left with an uncheckpointed `-wal` (crawler killed mid-run, or the
  file copied without checkpointing) is unusable. `attach_source` checks for a
  non-empty `<source>-wal` sidecar up front and raises a `RuntimeError` telling
  you to run `sqlite3 <source> 'PRAGMA wal_checkpoint(TRUNCATE);'` (or let the
  writer close cleanly) before re-running — rather than failing with a bare
  `sqlite3.OperationalError` or, worse, silently reading stale data. Do not work
  around this with `immutable=1`.
- **`SOURCE_DATABASE_URL` is required** for the text-reading stages. The old
  single-file recipe of pointing `DATABASE_URL` at `urls.db` now errors with a
  clear message until `SOURCE_DATABASE_URL` is set (or both vars point at the
  same file).
