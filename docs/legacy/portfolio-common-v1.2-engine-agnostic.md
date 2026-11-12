# Coordination note: `portfolio-common` v1.2.0 — the engine-agnostic seam

## What this resolves

`src/news_nlp/` was hard-wired to SQLite: `import sqlite3`, `sqlite3.Row`
type hints, `PRAGMA table_info` introspection, `ATTACH`/`main.`/`source.`
schema literals, `file:...?mode=ro`, and dialect SQL fragments
(`INSERT OR REPLACE`/`INSERT OR IGNORE`, `strftime`, `date(col,'weekday
0',...)`, `date('now')`, `GROUP_CONCAT`, `NOT GLOB '[0-9]'`, `INTEGER PRIMARY
KEY AUTOINCREMENT`). "Which database engine" was a change spread across this
repo's whole DB layer.

The goal: **`portfolio-common` is the single place that knows the engine**, so
a future switch is a clean, self-contained change there — `portfolio-nlp`
untouched (or a pin bump only).

## The approach — Option B

`portfolio-common` v1.2.0 grows an engine + dialect seam; `portfolio-nlp`
keeps its domain query modules (`schema` / `queries` / `corrections` /
`sector_summary`) but routes every engine-specific fragment through that seam.
This preserves the v1.0.0 "`news_nlp` has one owner — this repo" split (the
rejected alternative was re-merging the query layer back into
`portfolio-common`).

SQLite stays the only real backend; **no SQL semantics change** —
`SqliteDialect` emits the exact strings the code used inline, so this is a
pure refactor.

## `portfolio-common` v1.2.0 additions
([gamug/portfolio-common#9](https://github.com/gamug/portfolio-common/pull/9))

- `portfolio_common.db.dialect` — `Dialect` (Protocol) + `SqliteDialect` +
  `get_dialect()`: `placeholder` / `placeholders(n)`, `insert` /
  `insert_or_ignore` / `upsert` / `insert_or_ignore_select`, `year_expr` /
  `year_month_expr` / `week_start_expr` / `week_end_expr` (Monday-start ISO
  week) / `current_date_expr` (UTC), `group_concat`, `excludes_bare_digit`,
  `autoincrement_pk`.
- `Database.connect_url(url, **kw)` — a path, a `file:` URI, or
  `sqlite:///path` (scheme dispatch; `"D:/..."` is a path, not a `d:` URL).
  Always `uri=True` so a later read-only `ATTACH` is honored.
- `Database.table_columns` / `table_exists` / `ensure_columns` /
  `create_schema` / `copy_row_lean` — runtime introspection + DDL.
- `portfolio_common.db.Row` (alias for `sqlite3.Row`) + `RowLike` (Protocol)
  — the row type consumers annotate with, and the mapping + positional +
  `dict(row)` access contract a future engine's row factory must meet.
- `portfolio_common.db.two_store` — `connect_two_store(primary, secondary, *,
  alias="source", factory=, **kw)` + `TwoTierDatabase` (tracks
  `read_schema`). The generic form of this repo's two-tier SOURCE/RESULTS
  attach. `news_export.connect_readonly` now delegates to it.

## What changed here

- `src/news_nlp/db.py` — `NewsNlpDatabase(TwoTierDatabase)`; `articles_rel` is
  now a read-only property over `read_schema`; `connect()` →
  `connect_url`; `connect_pipeline()` → `connect_two_store(...,
  factory=NewsNlpDatabase, foreign_keys=True)`; `_ensure_article_row` →
  `copy_row_lean`; `require_source_text` → `table_columns`;
  `_compute_lean_article_columns` deleted (the cache moved into
  `copy_row_lean`). `_articles_rel()` + `Allowlist("main","source")` kept.
- `src/news_nlp/schema.py` — `SCHEMA` → `build_schema(dialect)` with the
  `AUTOINCREMENT` PK token from `dialect.autoincrement_pk` (module-level
  `SCHEMA = build_schema()` kept for the public API/tests);
  `_migrate_sector_summary_schema` → one `conn.ensure_columns(...)` call;
  `init_schema` → `conn.create_schema(...)`. No `import sqlite3`.
- `src/news_nlp/queries.py` — return hints `list[sqlite3.Row]` → `list[Row]`;
  `write_*` `INSERT OR REPLACE` → `conn.dialect.upsert(...)`; `write_entities`
  `INSERT` → `conn.dialect.insert(...)`; `fetch_pending_company_summaries`
  `GROUP_CONCAT`/`NOT GLOB` → `conn.dialect.group_concat` /
  `excludes_bare_digit`; `sentiment_stats` `strftime` grouping → a
  `_group_exprs(conn.dialect)` helper. Hand-assembled `WHERE ... = ?` /
  `LIMIT ? OFFSET ?` fragments keep the qmark marker (marked, not rewritten).
- `src/news_nlp/sector_summary/queries.py` — `_WEEK_*_EXPR` templates →
  `conn.dialect.week_start_expr` / `week_end_expr`; `date('now')` →
  `conn.dialect.current_date_expr()`; `NOT GLOB` → `excludes_bare_digit`;
  `INSERT OR REPLACE` → `conn.dialect.upsert(...)`. `list[Row]` hints.
- `src/news_nlp/sector_summary/composition.py`, `src/pipeline.py` —
  `sqlite3.Row` type hints → `portfolio_common.db.Row`; drop `import
  sqlite3`.
- `src/news_nlp/env.py` — docstring only: the env vars resolve to whatever
  `Database.connect_url` accepts (a path today); `DATABASE_URL` /
  `SOURCE_DATABASE_URL` stay `Path`-typed. `test_env.py` unchanged.
- `pyproject.toml` — `[tool.uv.sources]` `portfolio-common` pin bumps to
  `tag = "v1.2.0"`.
- Tests: `list[sqlite3.Row]` annotations in helpers → `Row`; two
  `test_schema.py` migration tests now build the legacy table with raw
  `sqlite3` then call `_migrate_sector_summary_schema` on a `db.connect()`
  connection (it needs the `ensure_columns` seam). `conftest.py` fixtures,
  `test_db.py` `PRAGMA` / WAL-preflight / FK / byte-hash assertions,
  `test_sector_summary.py` behavioural assertions — all unchanged (SQLite
  stays the reference engine for this suite; new-engine coverage would live
  in `portfolio-common`).

## Companion PRs

- `portfolio-common`:
  [#9](https://github.com/gamug/portfolio-common/pull/9) — the seam. Tag
  `v1.2.0` after merge.
- `portfolio-knowledge-graph` / `portfolio-financial-analysis` /
  `portfolio-data-mining` — same treatment, tracked in
  `docs/engine-agnostic-rollout.md`.

---
🤖 Generated with [Claude Code](https://claude.com/claude-code) as part of
making the Portfolio Thesis repos database-engine-agnostic.
Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
