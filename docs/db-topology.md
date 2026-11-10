# Database topology: the two-tier contract

The DB layer lives in **`src/news_nlp/`**, a local package owned by this repo
(vendored from `portfolio-common`'s `business_folders/news_nlp/` staging area —
see `docs/portfolio-common-v1-migration-plan.md`). It's imported directly by
`pipeline.py` / `apps/news_nlp_api.py` / tests (`import news_nlp as db`, `from
news_nlp import corrections`, `from news_nlp.taxonomy import ...`) — no
`portfolio_common.news_nlp` and no local re-export facade.

**No engine-specific code lives here.** Since `portfolio-common` v1.2.0 this
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
| holds | `articles` **including `body_text`** (written by the upstream crawler) | the 5 result tables (`news_nlp.schema.SCHEMA`) + a lean `articles` subset (**no `body_text`**), plus `news_nlp.eval`'s run-log tables — `eval_run`, `eval_inference`, `eval_verdict`, `eval_confusion` (sentiment/category only); the legacy `eval_judgement` table is kept, superseded (see `docs/evaluation.md`) |
| written by this repo | never | result rows, plus one lean `articles` row per processed article |

Path resolution (`news_nlp.env`): an **absolute** value is used as-is; a
**relative** value is left relative — resolved against the process's current
working directory. Unset → `results_db_path()` falls back to `data/nlp.db`,
`source_db_path()` → `None`.

**Where the variables come from.** `news_nlp.env` reads only real environment
variables. The entry points that import `pipeline` (the API, the CLI,
`python -m pipeline`) load `<repo>/.env` first; the scripts under `scripts/`
call `news_nlp.env.load_env_file()` first thing in `main` (a test fails for a
script that connects and does not). A variable already set in the shell wins
over the file. Write **absolute** paths in `.env`: a relative one resolves
against the folder the command is run from, not the repo. When
`SOURCE_DATABASE_URL` is missing the scripts that need it say whether the shell
and `<repo>/.env` were looked at.

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
   delegating the ATTACH and its orphaned-WAL preflight to
   `Database.attach(source, "source", read_only=True)`.
3. `news_nlp.db.check_schema()` then checks the attached SOURCE (only when one is
   attached, so not on single-file runs or the no-SOURCE endpoints): `PRAGMA user_version`
   must be **at least `MIN_SOURCE_SCHEMA_VERSION` (3)** and the columns this repo reads
   (`SOURCE_REQUIRED_COLUMNS`) must exist, else `SourceSchemaError`; a version above the
   latest seen (5) warns once per process. It is a minimum, not an exact match, because
   `portfolio-data-mining`'s migrations are additive and forward-only (a documented
   cache rollback re-stamps 5 back to 3, which an exact match would reject). The
   minimum is a copied constant, never an import of `data_mining`; the real `urls.db` is at 5. The cache tables
   (`sitemap_entries`, `sitemap_crawls`) and `discovery_progress` are not part of the
   contract and are never read, joined or attached. `POST /pipeline/run` runs the same
   check before its background worker and answers 503 with the message; the CLIs
   (`news_nlp_cli`, `news_nlp_eval`, `run_experiment`) exit with it on one line
   (`error: ...`, status 1) instead of a traceback.
4. `news_nlp.db.require_source_text()` then fails fast (before any model loads) if
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

### Working with a live SOURCE

Both recipes use Python because the `sqlite3` CLI is not installed on the
Windows machine that runs this (the orphaned-WAL error message from
`portfolio-common` names that CLI: gamug/portfolio-common#17). Substitute
`<source>` / `<copy>` with real paths; checked on Linux with SQLite 3.40 only
(Windows paths: see #106).

- **Checkpoint an orphaned WAL** — the one sanctioned write to SOURCE, an
  operator action this repo never performs. Run it only with the collector
  stopped or on a copy you own; the command does not enforce that, and it will
  checkpoint a file a live writer still has open:
  `python -c "import sqlite3; print(sqlite3.connect(r'<source>').execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone())"`.
  It prints `(busy, log, checkpointed)`. `busy` must be `0`. `1` means another
  connection has an **active read or write transaction**, the checkpoint did
  not complete and the orphaned-WAL error will come back: let that transaction
  finish and retry. (An idle open connection does not set `busy`.)
- **Take a consistent copy** of a SOURCE a writer still holds, through SQLite's
  backup API from a read-only connection (never plain-copy `.db` + `-wal`):
  `sqlite3.connect('file:<source>?mode=ro', uri=True).backup(sqlite3.connect('<copy>'))`,
  or `sqlite3.connect('file:<source>?mode=ro', uri=True).execute("VACUUM INTO '<copy>'")`.
  `VACUUM INTO` fails with `output file already exists` if `<copy>` exists, so
  use a new filename; `backup()` replaces the contents of whatever database it is given.

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

## What "quarantined" means (D36, 2026-11-08)

`articles.fetch_status` has `ok`, `quarantined` and `thin_content`. **`quarantined` is a verdict about the company tag, not about the article:** upstream (`portfolio-data-mining`) classes each (article, ticker) pair as `accept`, `reject` or `review` from the URL and the title, and quarantines the last two. A quarantined row still has what it was fetched with: on 2026-11-08, of the 377,010 quarantined rows, 374,551 had a non-empty `body_text` and 374,419 a title too, against 98,462 `ok`. As unique bodies (the same text under several tickers is one): `ok` 81,961; quarantined 260,845, of which **227,921 are not also `ok` under another ticker**; together 309,882 (`scripts/count_fetched_corpus_2026_11_08.py`). Example (real rows, ticker `A`, Agilent): a CNBC piece on Berkshire's cash and one on Corning and the S&P 500 are `reject` for Agilent, and are financial news.

The rule (`PLAN.md` D36): a properly fetched article is preserved and stays eligible for the pipeline; the pair class decides which company a stored result may be attributed to, not whether the body exists. Nothing here changes the read-only contract of SOURCE. The default articles selection still reads `ok`: the 2026-11-16 cut stays on it and the preserved corpus joins after the delivery, its scope decided from the counts of `TASKS.md` T-160. `thin_content` rows are preserved but are not part of the properly fetched corpus (owner, 2026-11-08).

## Target tables of the stepped pipeline (D39, D40; approved 2026-11-10)

Created by `init_schema()` since #162 (nothing writes them yet); the schema was approved by the owner on 2026-11-10 (constitution AI #10). `sub_news` (one row per piece the `units` extractor cuts), `pipeline_step_state` (status per item and step, with the model revision) and `news_company` (the result per text and ticker) and two columns on `article_entities` (`sub_news_id`, `ticker`). Keys and meaning: `SPEC.md` §data model. They live in RESULTS; SOURCE is still read-only.

## Caveats

- **Orphaned WAL on the SOURCE.** As of `portfolio-common` v1.2.2, `attach_source`
  does *not* reject a SOURCE just because its `-wal` is non-empty — a live
  writer (e.g. `portfolio-data-mining`'s collector, still running) or a hard-killed
  one (crawler killed mid-run) both leave a non-empty `-wal` *and* a matching
  `-shm`, and SQLite's own WAL recovery handles both safely on a `mode=ro` open.
  What's still rejected is a `-wal` copied without its `-shm` (e.g. a backup/rsync
  that skipped the `-shm`, which SQLite's own docs call disposable): nothing then
  proves the two files are one consistent snapshot. `attach_source` checks for
  exactly that shape — non-empty `<source>-wal`, missing `<source>-shm` — and
  raises a `RuntimeError` telling you to checkpoint first, rather than failing with
  a bare `sqlite3.OperationalError` or, worse, silently reading stale data. Do not
  work around this with `immutable=1`. This check is not a backup-safety
  guarantee either way — it cannot catch a checkpoint-torn copy; see
  `portfolio-common`'s `test_attach_does_not_catch_a_checkpoint_torn_copy`.
- **A live-writer SOURCE is a moving target, not a snapshot.** The read-only
  attach does not freeze SOURCE: each statement sees whatever the writer has
  committed so far. This is observed SQLite connection behaviour, not a
  contract this repo promises, so wrapping a run in one read transaction would
  be a legitimate change. Within one `run_pipeline`, a stage reads its pending
  list once, so an article collected mid-run can reach a later stage and miss
  an earlier one (e.g. NER and category but not sentiment). That is consistent,
  not corrupting: every query that combines stage outputs (`sector_summary`,
  the company summary) reads RESULTS tables with INNER JOINs, so such an
  article is simply excluded until the next run completes its missing stages,
  and the next run does (see
  `test_run_pipeline_runs_against_a_live_writers_source`, which also asserts
  SOURCE is not modified by the pipeline, and in `test_sector_summary.py`
  `..._excludes_an_article_without_sentiment` /
  `..._excludes_articles_without_sentiment`). Pre-existing, not specific to a live
  writer: a `sector_summary` week that was already composed is not regenerated
  when an article for that week arrives later.
- **Windows is only partly verified** (remaining work tracked in
  [#106](https://github.com/gamug/portfolio-nlp/issues/106)). Reported, not
  reproduced here: the `portfolio-data-mining` maintainers' hand-off note
  (2026-10-21, not published in either repo) says a read-only open of the real
  WAL-mode SOURCE (`D:/thesis/data/urls.db`, not the RESULTS `nlp.db`) already
  works on the Windows machine. Not known: the v1.2.2 orphaned-WAL rejection
  (it keys on a `-shm` file next to a non-empty `-wal`, and `-shm` lifecycle
  and copying a live file differ on Windows) and a full `run_pipeline` run
  there. The WAL tests here run on Linux/CI only. Until #106 is closed, follow
  [Working with a live SOURCE](#working-with-a-live-source) above.
- **The summaries cut's run lock on Windows (found 2026-11-07, PR #134).**
  - Cause and fix: `os.kill(pid, 0)` is not a "does it exist" probe on Windows (Python calls
    `TerminateProcess`), so a resume after an interrupted run crashed with `WinError 11` and
    could have killed a process that held the pid. `scripts/cut_summaries_2026_11_07.py` now
    asks `OpenProcess` + `WaitForSingleObject` and answers "dead" only when the OS says so (any
    error reads as alive).
  - Lock format: `run.lock` holds `<pid> <process start time>`, so a pid Windows hands out again
    is not taken for the run that left the lock. The start time is read with the query right
    alone (protected processes refuse the wait right). An old pid-only lock still works, and
    `--break-lock` remains. An older script reads the new format as stale and deletes it: do
    not share an output folder between checkouts of different versions.
  - Folder: the lock is a hard link, so the output folder must be on NTFS. A failed link says
    NTFS only when the error means "no hard links here" (exFAT, some shares); anything else says
    something holds the file (antivirus, indexer, editor).
  - Signals: Ctrl-Break stops like Ctrl-C. Closing the console window cannot be caught and
    leaves a lock the next start takes over.
  - Limits: the takeover of a stale lock is not atomic (two starts at the same instant can both
    run: duplicated work, chunk files are atomic).
  - Verified: dispatch and the stale-file scenario on Linux/CI. Not verified until run on the
    Windows machine: the ctypes branch itself (a `win32`-only test covers it there).
- **`SOURCE_DATABASE_URL` is required** for the text-reading stages. The old
  single-file recipe of pointing `DATABASE_URL` at `urls.db` now errors with a
  clear message until `SOURCE_DATABASE_URL` is set (or both vars point at the
  same file).
