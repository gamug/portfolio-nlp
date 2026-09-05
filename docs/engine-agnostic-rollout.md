# Engine-agnostic rollout — cross-repo implementation plan

## Goal (same scope for every repo)

The database engine is known in **exactly one place — `portfolio-common`**.
Every other repo stops naming a concrete engine (`import sqlite3`,
`sqlite3.Row`, `PRAGMA`, `sqlite_master`, `ATTACH`/`main.`/`source.` literals,
`file:...?mode=ro`, and dialect SQL: `INSERT OR REPLACE`/`INSERT OR IGNORE`,
`INSERT ... ON CONFLICT ... DO`, `strftime`, `date(...,'weekday 0',...)`,
`date('now')`, `GROUP_CONCAT`, `NOT GLOB`, `AUTOINCREMENT`, `json_extract`).
It routes every engine-specific fragment through `portfolio-common`'s
`Dialect` seam + neutral `Row` type + `connect_url` + introspection helpers
+ `two_store`.

- **Keep SQLite** as the only real backend everywhere. **No SQL semantic
  change** — `SqliteDialect` emits today's exact strings.
- A future engine switch = a new `Dialect` impl + a registered `connect_url`
  scheme in `portfolio-common`; consumers get a pin bump.
- Out of scope: any second backend implementation, SQLAlchemy/ORM, SQL
  rewrites. **No new lint rule** is added to enforce "no `import sqlite3`" —
  it's a review checklist item per PR.

## Status

| Phase | Repo | State |
|---|---|---|
| 1 | `portfolio-common` v1.2.0 | **Merged**, tagged `v1.2.0` — [#9](https://github.com/gamug/portfolio-common/pull/9). |
| 1b | `portfolio-common` v1.2.1 | **Merged**, tagged `v1.2.1` — [#10](https://github.com/gamug/portfolio-common/pull/10). `upsert` `ON CONFLICT DO UPDATE`/`DO NOTHING`, `json_extract`/`json_each`, `relation_exists`/`relation_kind`/`relation_ddl`, `schema_version`, `DatabaseError` — the extra seam Phases 4–5 needed. |
| 2 | `portfolio-nlp` | **Merged** — [#24](https://github.com/gamug/portfolio-nlp/pull/24) + re-pin [#25](https://github.com/gamug/portfolio-nlp/pull/25). `src/news_nlp/` names no engine; 143 tests green. |
| 3 | `portfolio-knowledge-graph` | **PR open** — [#9](https://github.com/gamug/portfolio-knowledge-graph/pull/9). Pin → `v1.2.0`; one `import sqlite3` + one `Row` hint. |
| 4 | `portfolio-financial-analysis` | **PR open** — [#32](https://github.com/gamug/portfolio-financial-analysis/pull/32). Pin → `v1.2.1`; no `import sqlite3` in `src/`; `PRAGMA`/`sqlite_master`/`executescript` routed; 203 tests green. `INSERT OR IGNORE`/`ON CONFLICT`/`json_extract` string literals flagged as the remaining seam-adoption work (see that PR). |
| 5 | `portfolio-data-mining` | **PR open** — [#26](https://github.com/gamug/portfolio-data-mining/pull/26). Pin → `v1.2.1`; no `import sqlite3` in `src/`; `PRAGMA user_version`→`set_schema_version`, `executescript`→`create_schema`, the `articles` `INSERT OR REPLACE` + queue `INSERT OR IGNORE`/`ON CONFLICT` routed through `conn.dialect`; 213 tests green. |

Also done: **every consumer normalised onto a `v1.2.x` tag** — the three that
were pinned to loose `portfolio-common` branch commits (`-knowledge-graph`,
`-data-mining`) or a stale tag (`-financial-analysis` → `v1.0.0`) now point at
`v1.2.0` / `v1.2.1`.

---

## Phase 1 — `portfolio-common` v1.2.0

Delivered in PR #9. New: `db/dialect.py` (`Dialect` + `SqliteDialect` +
`get_dialect`), `db/engine.py` additions (`Row`, `RowLike`, `Database.dialect`,
`connect_url`, `table_columns`/`table_exists`/`ensure_columns`/`create_schema`/
`copy_row_lean`), `db/two_store.py` (`connect_two_store` + `TwoTierDatabase`),
`news_export.fetch_processed_articles -> list[Row]`. Backward compatible
(`Row` stays `sqlite3.Row`; every existing public name preserved).

---

## Phase 2 — `portfolio-nlp`

Per `~/.claude/plans/…meerkat.md` and
`docs/portfolio-common-v1.2-engine-agnostic.md`. Rewrote
`src/news_nlp/{db,schema,queries,corrections}.py` +
`sector_summary/queries.py` + `sector_summary/composition.py` +
`pipeline.py` onto the seam; dropped every `import sqlite3` from `src/`;
`env.py` stays `Path`-typed; two `test_schema.py` migration tests adjusted;
docs updated. Pin → `tag = "v1.2.0"` when Phase 1 tags.

---

## Phase 3 — `portfolio-knowledge-graph` (smallest — ~1 hour)

Explored. It is **already almost uncoupled**: no SQL strings anywhere, no
`PRAGMA`/`sqlite_master`/`ATTACH`, no own database (emits a flat `data.ttl`;
the RDF store is external GraphDB/Fuseki). Only touch-points:

1. `pyproject.toml` — bump `portfolio-common` `rev = "2f8f93a…"` → `tag =
   "v1.2.0"`; `uv sync`. `news_export` names/signatures are unchanged, so
   `src/etl/news_to_rdf.py` keeps working.
2. `src/etl/news_to_rdf.py:32` — drop `import sqlite3`.
3. `src/etl/news_to_rdf.py:79` — `def from_row(cls, row: sqlite3.Row)` →
   `row: Row` (`from portfolio_common.db import Row`). Row access is
   `row["name"]` only, already `RowLike`-safe.
4. `src/etl/config.py:60-69` — `KG_URLS_DB` / `KG_RESULTS_DB` stay
   filesystem paths (matches `portfolio-nlp`'s `env.py` decision). No change
   needed; a docstring note that they feed `connect_url` is optional.
5. Docs: a short `docs/` coordination note mirroring
   `portfolio-nlp/docs/portfolio-common-v1.2-engine-agnostic.md`.

No tests exist; no CI (pre-commit only: ruff + mypy-in-venv + commitizen).
`.code_quality/ruff.toml` `target-version` is stale at `py310` — fix while
there. Verify: `uv run mypy` (the pre-commit hook) + a manual
`news_to_rdf.py` smoke run against a scratch SOURCE+RESULTS pair.

---

## Phase 4 — `portfolio-financial-analysis` (largest)

Explored. `src/kg_schema/` owns a big additive schema + a real migration
framework; `quant` / `fundamental_agent` / `pricing_agent` / `cycle` /
`entity_resolution` / `api` all read/write the shared `KG_FINANCIAL_DB`
through `kg_schema.connect`. Connection acquisition is **already a single
chokepoint** (`src/kg_schema/db.py` `connect`/`connect_ro` →
`Database.connect(wal=False, foreign_keys=True)`); no raw `sqlite3.connect`
in `src/`. Env resolution is path-based (`KG_FINANCIAL_DB`, `KG_UNIVERSE_DB`,
`KG_NEWS_DB`) — keep as paths, route through `connect_url`.

Work, by kind:

1. **Pin** — `tag v1.0.0` → `tag = "v1.2.0"`; `uv sync`; fix any v1.0→v1.2
   drift (none expected — additive).
2. **`sqlite3.Row` hints → `Row`** — 6 `src/` modules (`quant/db.py`,
   `cycle/data.py`, `cycle/orchestrator.py`, `pricing_agent/db.py`,
   `fundamental_agent/db.py`, `fundamental_agent/pipeline.py`) + the
   `isinstance(row, sqlite3.Row)` branch in `kg_schema/queries.py:313`
   (becomes an `isinstance(row, RowLike)` check, or restructure).
3. **`sqlite3.OperationalError` "absent table/view" idiom** — 11 `except`
   sites (`kg_schema` ×3, `api` ×5, `quant` ×3, `fundamental_agent` ×1) +
   one `sqlite3.Error` (`quant/actions.py:219`). `portfolio-common` should
   expose a neutral exception (e.g. `portfolio_common.db.DatabaseError` /
   `MissingRelation`) for these, or a `Database.relation_exists(name)`
   helper so the `try/except` becomes an explicit check. **This is a Phase 1
   follow-up** — `portfolio-common` v1.2.0 as shipped has no neutral DB
   exception; add it before Phase 4 (call it v1.2.1 or fold into v1.2.0 if
   #9 hasn't tagged).
4. **`PRAGMA table_info` feature-detection** → `Database.table_columns` —
   `kg_schema/__init__.py:65`, `kg_schema/migrations.py:41`,
   `fundamental_agent/db.py:187/317/373`, `pricing_agent/db.py:124`.
5. **`sqlite_master` introspection** → `Database.table_exists` /
   `Database.relation_exists` — `kg_schema/__init__.py:61`,
   `kg_schema/migrations.py` ×5. Two of those (`migrations.py:185,257`) read
   `SELECT sql FROM sqlite_master` to string-match a CHECK clause — needs a
   dedicated `portfolio-common` helper (`Database.relation_ddl(name)`) or
   stays a documented SQLite-only migration internal.
6. **`.executescript()` multi-statement DDL** → `Database.create_schema` —
   `kg_schema` ×9, `quant/db.py:53`, `fundamental_agent/db.py:183`,
   `pricing_agent/db.py:122`.
7. **`ALTER TABLE ... ADD COLUMN`** → `Database.ensure_columns` —
   `kg_schema/__init__.py:68` (the `REQUIRED_COLUMNS` loop).
8. **DDL dialect** — `ddl.py` + each package's `SCHEMA`: `INTEGER PRIMARY
   KEY` rowid aliases, partial indexes (`... WHERE valid_to IS NULL` ×5),
   inline `CHECK (... IN (...))`, `ON DELETE CASCADE`. Route the
   `AUTOINCREMENT`/rowid-PK token through `dialect.autoincrement_pk`; the
   rest is standard-enough SQL to keep, with a marker comment. Partial
   indexes and `CHECK` are SQLite+Postgres common — leave, note.
9. **`INSERT OR IGNORE` ×18 / `ON CONFLICT ... DO UPDATE/NOTHING` ×20** →
   `conn.dialect.insert_or_ignore(...)` / `conn.dialect.upsert(...)`. The
   `upsert` seam needs an `on_conflict_do_nothing` variant and `excluded.`
   column support — **Phase 1 follow-up**: `portfolio-common` v1.2.0's
   `upsert` only emits `INSERT OR REPLACE`; this repo's `ON CONFLICT ... DO
   UPDATE SET x = excluded.x` needs `upsert(..., update=[...])` to actually
   render the update list (SQLite supports `ON CONFLICT DO UPDATE` too), and
   a `do_nothing=True` option. Extend the dialect before Phase 4.
10. **`json_extract` / `json_each` ×~15** (all `kg_schema/views.py`) —
    `dialect.json_extract(col, path)` / a `json_each`-equivalent. **Phase 1
    follow-up** — not in v1.2.0. The 30 `CREATE VIEW`s also use
    correlated-subquery "latest version" patterns and `REPLACE`/`||` string
    funcs; views are the least portable surface — consider marking the whole
    `views.py` as an explicitly-SQLite module with a single top-of-file
    note rather than dialectising every fragment.
11. **`LIMIT ? OFFSET ?`** (`api/routers/scores.py:42`) + hand-rolled `IN
    (?,?,…)` (`quant/panel.py:105`, `fundamental_agent/db.py:249`,
    `pricing_agent/db.py:181`, `entity_resolution/news_db.py:52`) — switch
    the hand-rolled ones to `portfolio_common.db.in_clause`; keep `LIMIT ?
    OFFSET ?` marked.
12. **Migrations** (`kg_schema/migrations.py`) — the `PRAGMA foreign_keys =
    OFF` + create-`__new` + copy + drop + rename "12-step ALTER" is
    inherently a SQLite migration technique. Keep it, but wrap the
    `foreign_keys` toggle and the `sqlite_master` DDL reads in
    `portfolio-common` helpers so the module has no bare `PRAGMA` / `import
    sqlite3`. Migration bodies stay SQLite-specific by nature; that's
    acceptable (a Postgres migration path would be written fresh).
13. **Tests** — SQLite reference engine; `sqlite3.Row` annotations in
    helpers → `Row`; `Database(sqlite3.connect(...))` fixtures may stay
    (setup). `RETURNING id` appears in ~4 test seeds — those are test-only
    and SQLite ≥3.35 supports it; leave. `mypy src tests` + `pytest -q` +
    ruff must stay green.

**`.code_quality/ruff.toml` `target-version` is `py310`** (repo is 3.12) —
fix while there.

---

## Phase 5 — `portfolio-data-mining` (the SOURCE-store owner)

Explored. The crawler; owns `data/urls.db` (`discovered_urls`,
`discovery_progress`, `articles`) and a separate `data/universe.db`
(`universe_membership`). Connection is a single chokepoint
(`src/data_mining/db.py` `connect` → `Database.connect`); no raw
`sqlite3.connect` in `src/`. `DATABASE_URL` / `UNIVERSE_DB_PATH` are
filesystem paths — keep, route through `connect_url`.

Work:

1. **Pin** — `rev = "aa59a7ec…"` → `tag = "v1.2.0"`; `uv sync`; resolve v1.0
   API drift from whatever `aa59a7ec` predates (it's a pre-`v1.0.0` commit —
   check `Database.connect` kwargs).
2. **`sqlite3.Row` hints → `Row`** — `src/data_mining/queries.py:25,139`,
   `src/extractor/db.py:13,69,90`, `src/news_collector/storage/queue.py:8,421`.
3. **`articles` DDL** (`src/data_mining/schema.py:55-75`) — route the
   `discovered_urls` `id INTEGER PRIMARY KEY AUTOINCREMENT` through
   `dialect.autoincrement_pk`. **`articles.id` stays a plain
   caller-supplied `INTEGER PRIMARY KEY`** (no `AUTOINCREMENT`; it equals
   `discovered_urls.id`) — a rewrite must NOT turn it into an identity
   column. All 14 columns `portfolio-nlp` reads by name
   (`id, body_text, title, fetch_status, http_status_code, ticker, company,
   gics_sector, gics_sub_industry, pub_date, fetched_at, author, word_count,
   source_domain`) must keep their names and `TEXT`/`INTEGER` types. Add a
   cross-reference from `portfolio-nlp/docs/db-topology.md` to this schema.
4. **`.executescript()`** → `create_schema` — `schema.py:108-109`,
   `queries.py:86`.
5. **`PRAGMA user_version`** (`schema.py:110`) — needs a
   `portfolio-common` helper (`Database.schema_version` get/set). **Phase 1
   follow-up** — not in v1.2.0.
6. **`PRAGMA table_info`** (`schema.py:126`) → `table_columns`.
7. **`ALTER TABLE ... ADD/DROP COLUMN`** (`schema.py:130-134`,
   `_migrate_legacy_sector_column`) — `ADD COLUMN` → `ensure_columns`; the
   `DROP COLUMN sector` is SQLite ≥3.35 and destructive — keep as a
   documented one-time migration, but move the bare statement behind a
   `portfolio-common` `Database.drop_column` or leave it marked.
8. **`INSERT OR REPLACE INTO articles`** (`src/extractor/db.py:110`, the
   crawler write path `portfolio-nlp` reads) → `conn.dialect.upsert(...)`.
9. **`INSERT OR IGNORE`** (`queue.py:120`) → `conn.dialect.insert_or_ignore`.
10. **`ON CONFLICT (...) DO UPDATE SET ... excluded.`** (`queue.py:255-262`,
    `discovery_progress`) → `conn.dialect.upsert(..., update=[...])` — needs
    the same dialect extension as Phase 4 item 9.
11. **`PRAGMA foreign_keys = ON`** (`src/extractor/db.py:55`) — this is
    already `Database.connect(foreign_keys=True)` territory; move the
    explicit statement into the connect call.
12. **Row-value `(url, ticker) IN (...)`** (`queue.py:145`) — SQLite+Postgres
    only; keep marked (or switch to a compound-key OR-chain if portability
    matters later).
13. **Raw SQL in the app layer** — `apps/news_crawler_api.py:229-344` writes
    `SELECT * FROM articles WHERE …` / dynamic `discovered_urls` selects
    directly instead of through `extractor.db`. Move these into
    `extractor/db.py` (or a `queries.py`) as named functions so the app has
    no SQL — matches how `portfolio-nlp`'s API is structured.
14. **`universe.db` queries** (`src/data_mining/queries.py`) — the
    interpolated-column `INSERT` → `dialect.insert`; the rest is
    ANSI-portable (`MIN`, `COUNT(*)`, `WHERE valid_from <= ?`). `_connect()`
    → `connect_url`.
15. **Tests** — SQLite reference engine; `sqlite3.Row` hints → `Row`;
    `Database(sqlite3.connect(":memory:"))` fixtures may stay;
    `sqlite_master` / `PRAGMA` assertions in `test_schema.py` /
    `test_data_mining_db.py` / `test_db.py` stay (verification). No
    `conftest.py`; no CI (pre-commit only). `pytest -q` + `mypy src apps
    cli` + ruff must stay green.

---

## Phase 1 follow-ups surfaced by Phases 4 & 5

`portfolio-common` v1.2.0 as drafted covers `portfolio-nlp` fully, but the
other three repos need a few more seam pieces. Fold these into #9 before it
tags, or ship a fast `v1.2.1`:

- **`Dialect.upsert` must render `ON CONFLICT (...) DO UPDATE SET col =
  excluded.col`** given an `update=[...]` list, and support a
  `do_nothing=True` / `insert_or_ignore`-with-conflict-target form. Both
  `-financial-analysis` and `-data-mining` rely on real `ON CONFLICT DO
  UPDATE` semantics, not `INSERT OR REPLACE` (which deletes+reinserts).
- **A neutral "relation missing" exception** (`portfolio_common.db`) plus
  `Database.relation_exists(name)` — to replace ~12 `except
  sqlite3.OperationalError` "table absent in a partial DB" sites in
  `-financial-analysis`.
- **`Database.schema_version` get/set** — wraps `PRAGMA user_version`
  (`-data-mining`) so consumers don't write the pragma. (`-financial-analysis`
  uses a `schema_version` table instead — no change needed there.)
- **`Dialect.json_extract(col, path)`** (+ a `json_each` story) — for
  `-financial-analysis`'s `views.py`. Or accept `views.py` as an
  explicitly-SQLite module.
- **`Database.relation_ddl(name)`** — `-financial-analysis` migrations read
  `SELECT sql FROM sqlite_master` to gate CHECK-constraint widening.

## Per-phase PR shape

Each downstream phase = one PR in its repo: (a) pin bump to `v1.2.0` (or
`v1.2.1`), (b) mechanical seam adoption per the list above, (c) `Row`
annotation swaps, (d) docs coordination note, (e) full `uv run pytest` +
`ruff` + `mypy` green + a `grep -rn "import sqlite3" src apps cli` proving
it's gone from non-test code.
