# Database topology: the two-tier contract

The DB layer lives in **`portfolio_common.news_nlp`** (git-tag-pinned in
`pyproject.toml`'s `[tool.uv.sources]`; `src/db.py` is a thin re-export facade).
The contract itself — the five result tables, the SOURCE/RESULTS roles, the
`ATTACH` mechanics, the lean-`articles` FK invariant — is in
[`portfolio-common/docs/news-nlp-db-topology.md`](https://github.com/gamug/portfolio-common/blob/master/docs/news-nlp-db-topology.md).
This page keeps the `portfolio-nlp` **operator recipes**.

`portfolio-nlp` uses **two** SQLite databases with distinct roles, selected by
two environment variables and never conflated in code.

| | **SOURCE** | **RESULTS** |
|---|---|---|
| env var | `SOURCE_DATABASE_URL` | `DATABASE_URL` |
| default | none — **required** for the text-reading stages | `data/nlp.db` |
| opened | read-only (`file:…?mode=ro`), ATTACHed as schema `source` | read/write, schema `main` |
| holds | `articles` **including `body_text`** (written by the upstream crawler) | the 5 result tables + a lean `articles` subset (**no `body_text`**) |
| written by this repo | never | result rows, plus one lean `articles` row per processed article |

Path resolution (`portfolio_common.news_nlp.env`): an **absolute** value is used
as-is; a **relative** value is left relative — resolved against the process's
**current working directory** (it used to resolve against the repo root; there is
no repo root once the code lives in an installed package). Unset → `DATABASE_URL`
falls back to `data/nlp.db`, `SOURCE_DATABASE_URL` → `None`.

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

`pipeline.run_pipeline` → `db.connect_pipeline()`:

1. opens the RESULTS store read/write (`main`);
2. unless `SOURCE_DATABASE_URL` resolves to the **same path** as `DATABASE_URL`,
   `ATTACH`es the SOURCE store read-only as `source` and flips
   `conn.articles_rel` to `"source"`;
3. `db.require_source_text()` then fails fast (before any model loads) if that
   `articles` table has no `body_text` column or no non-empty `body_text` rows —
   the common "pointed a text stage at the results store" mistake, which used to
   surface as a silent *"0 pending"*.

The three `body_text` readers (`fetch_pending_articles`,
`fetch_pending_category_articles`, `fetch_pending_company_summaries`) qualify
`articles` with `conn.articles_rel`, so they read `source.articles` during a
two-tier run and `main.articles` otherwise. All result-table joins stay in
`main`. Every result-table write (`db.write_sentiment` / `write_category` /
`write_entities` / `write_company_summary`) first `INSERT OR IGNORE`s the lean
`articles` row (metadata only) from `source` into `main`, so the RESULTS store
stays foreign-key-consistent on its own — no separate migration step.

The `sector_summary` stage and every query/correction endpoint read only result
tables + `articles` metadata, never `body_text`, so they need **no** SOURCE.

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

`connect_pipeline` sees the paths match, skips the ATTACH, and reads/writes the
one file (`articles_rel` stays `"main"`, the lean upsert is a no-op).

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
