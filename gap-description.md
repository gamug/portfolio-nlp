# Gap: `portfolio-nlp` kept back-compat facades instead of fully migrating to `portfolio-common`

## Context

`portfolio-common` centralizes the SQLite connection/schema/query code that used
to be duplicated across the Portfolio Thesis repos. Three repos have adopted it:

| Repo | How it migrated |
|---|---|
| `portfolio-data-mining` | PR #22 → #23 (commits `8d33db9`, `40eb273`): consolidated the duplicated DB code into a local `src/common/` package, then **deleted `src/common/` entirely** and rewired every call site (`extractor/db.py`, `news_collector/storage/queue.py`, `apps/pricing_api.py`, `cli/pricing_cli.py`) to import `portfolio_common.*` directly. |
| `portfolio-financial-analysis` | PR #22 → #28 (commits `e94d472`, `ea7bf70`): same shape — consolidated `src/kg_schema/` in-repo first, then **deleted it entirely** and rewired ~40 imports across the whole codebase to `portfolio_common.kg_schema.*`. |
| `portfolio-nlp` | PR #17 (`48d1e97`) + PR #18 (`5343f81`): added `portfolio-common` as a dependency and moved the DB layer's *implementation* over, but **kept `src/db.py`, `src/corrections.py`, `src/categories.py` as permanent thin re-export facades** instead of deleting them and rewiring callers. |

`portfolio-nlp` is the odd one out — it did not migrate "the same exact way" the
other two did, even though it depends on `portfolio-common` too.

## The gap, concretely

`src/db.py`, `src/corrections.py`, `src/categories.py` still exist and do
nothing but re-export names from `portfolio_common.news_nlp` (and its
`.taxonomy` / `.corrections` submodules). Their own docstrings say so plainly:

> `src/db.py`: *"Back-compat facade. The NLP database layer moved to
> `portfolio_common.news_nlp`... This module is kept thin so `import db` /
> `db.connect(...)` call sites (`pipeline.py`, `apps/news_nlp_api.py`, tests)
> keep working unchanged."*

> `src/corrections.py` / `src/categories.py`: *"Back-compat re-export... kept
> here so `import corrections`/`from categories import ...` call sites... don't
> churn."*

`src/db.py` additionally carries a `__getattr__` shim just to keep
`db.DB_PATH` / `db.SOURCE_DB_PATH` resolving as lazy module attributes, for
exactly one caller (`apps/news_nlp_api.py`'s `db.connect(db.DB_PATH)`).

None of the actual call sites were ever rewired to import
`portfolio_common.news_nlp` directly. They still do:

- `apps/news_nlp_api.py` — `import db`, `import corrections`,
  `from categories import CATEGORY_SLUGS, OTHER_LABEL`
- `src/pipeline.py` — `import db`, `from categories import CATEGORY_LABELS, OTHER_LABEL`
- `tests/news_nlp/conftest.py` — `import db as db_module`
- `tests/news_nlp/test_correction_endpoints.py` — `import db`
- `tests/news_nlp/test_summary_pipeline.py` — `import db`, `from categories import CATEGORY_SLUGS`
- `tests/news_nlp/test_category_pipeline.py` — `import db`
- `tests/news_nlp/test_two_tier.py` — `import db`
- `tests/news_nlp/test_query_endpoints.py` — `import db`

`pytest.ini`'s own comment acknowledges the arrangement is deliberate, not an
oversight: *"db/categories/corrections are now thin re-export facades over
portfolio_common.news_nlp, which is an installed dependency."* It's a real,
working design choice — just a different (lighter-weight) one than the other
two repos made, so this repo now carries an extra, permanent layer of
indirection they don't have.

## Why it's worth closing

- A reader of `pipeline.py` or `apps/news_nlp_api.py` sees `import db` /
  `import corrections` / `from categories import ...` and has no signal that
  every one of those names is 100% delegated to an external dependency — the
  facade hides exactly the fact the migration was supposed to make legible.
- The facades are extra maintenance surface: every time `portfolio_common`
  adds/renames/removes a `news_nlp` export, these three files need a matching
  edit just to keep re-exporting it, on top of whatever the real caller needed.
- It's inconsistent with the pattern this whole family of repos otherwise
  follows, which makes `portfolio-nlp` a worse template for the *next* repo
  that migrates onto `portfolio-common` (it would be ambiguous which of the
  two shapes to copy).

## Suggested fix (mirrors `portfolio-data-mining` PR #23 / `portfolio-financial-analysis` PR #28)

1. Rewire every call site above to import from `portfolio_common.news_nlp`
   directly:
   - `import db` → `from portfolio_common import news_nlp as db` (keeps
     `db.connect(...)`, `db.write_sentiment(...)`, etc. working unchanged at
     the call site, since those names are already exported at
     `portfolio_common.news_nlp`'s top level) — or import the specific names
     used, whichever reads better per file.
   - `import corrections` → `from portfolio_common.news_nlp import corrections`
     (the submodule), or import the specific `update_*`/`delete_*` names used.
   - `from categories import CATEGORY_LABELS, OTHER_LABEL` →
     `from portfolio_common.news_nlp.taxonomy import CATEGORY_LABELS, OTHER_LABEL`.
   - Replace `db.connect(db.DB_PATH)` in `apps/news_nlp_api.py` with
     `db.connect(results_db_path())`, importing `results_db_path` directly —
     this removes the need for `src/db.py`'s `__getattr__` shim entirely.
   - The handful of underscore-prefixed names the facade currently re-exports
     for the test suite (`_articles_rel`, `_ensure_article_row`,
     `_compute_lean_article_columns`, `_Connection`,
     `_PENDING_ARTICLE_TABLES`, `_migrate_sector_summary_schema`,
     `_CATEGORY_DISPLAY_NAMES`/`_CATEGORY_ORDER`/`_WEEK_START_EXPR`/
     `_WEEK_END_EXPR`) need to come from their real submodules
     (`portfolio_common.news_nlp.db`, `.queries`, `.schema`,
     `.sector_summary`) at each test's own import, the same way
     `portfolio_common.news_nlp.queries` itself already imports
     `_articles_rel`/`_ensure_article_row` from `.db` — internal-to-the-family
     imports like that are fine; the facade indirection is what should go.
2. Delete `src/db.py`, `src/corrections.py`, `src/categories.py`.
3. Update `pytest.ini`'s comment (currently describes the facades) and
   `docs/db-topology.md`'s "`src/db.py` is a thin re-export facade" line.
4. Optional, unrelated hygiene while in there: the pinned `portfolio-common`
   tag is `v0.3.0`; `v0.4.0` exists now (adds
   `news_nlp.fetch_processed_articles`, not needed by this repo, but no reason
   to stay behind).

Not in scope: any schema, behavior, or DB-contract change — this is purely an
import-path/structural cleanup, matching the shape `portfolio-data-mining` and
`portfolio-financial-analysis` already landed.

## Provenance

Surfaced 2026-09-04 while migrating `portfolio-knowledge-graph`'s `etl/` onto
`portfolio-common` (see that repo's PR #6 and `portfolio-common` PR #6) — this
file was written from that session's findings, not from a `portfolio-nlp`-side
investigation, so re-verify the exact call-site list above still matches
`portfolio-nlp`'s current `master` before acting on it.
