# Migration plan: portfolio-common v1.0.0 (DB engine + business_folders split)

## What changed upstream

`portfolio-common` v1.0.0 is a clean-break rewrite: the shared library used to
mix two concerns — a generic SQLite connection engine, and business/domain
code owned by specific downstream repos (`news_nlp` was ours, along with a
two-tier SOURCE/RESULTS ATTACH contract). As of v1.0.0:

- `portfolio-common` is **DB-engine-only**: `portfolio_common.db.Database` (a
  single connection class with generic `.attach()`/`.detach()` support,
  including the stale-WAL preflight check the old `news_nlp/db.py.attach_source`
  had), plus `portfolio_common.db.in_clause` / `portfolio_common.db.Allowlist`
  — reusable injection-prevention primitives.
- Everything under the old `portfolio_common.news_nlp` package has been
  extracted into `business_folders/news_nlp/news_nlp/` in the
  `portfolio-common` repo, staged for this repo to adopt directly — it is
  **not** part of the installed `portfolio-common` package anymore. The
  package there is renamed `NewsNlpDatabase(Database)` internally (a
  `Database` subclass, replacing the old `_Connection(sqlite3.Connection)`
  subclass) but keeps the same public function names
  (`connect`, `connect_pipeline`, `attach_source`, `detach_source`,
  `init_schema`, `fetch_*`, `write_*`, `update_*`, `delete_*`, ...) — the
  `__all__` contract this repo already relies on is unchanged in shape, only
  the import path moves.
- There is **no backward-compatible shim** — pinning our `portfolio-common`
  git tag to `v1.0.0` breaks every `from portfolio_common import news_nlp` /
  `from portfolio_common.news_nlp import X` import in this repo until this
  plan is executed.
- The design doc this repo already links to
  (`portfolio-common/docs/news-nlp-db-topology.md`) moved to
  `business_folders/news_nlp/docs/db-topology.md` — update
  `docs/db-topology.md` here (this repo already has a copy/reference) if it
  points at the old location.

See `portfolio-common`'s `business_folders/news_nlp/README.md` for the exact
file inventory and `CHANGELOG.md` for the full rationale.

## What to pull in

1. Copy `portfolio-common/business_folders/news_nlp/news_nlp/` into this repo
   as `src/news_nlp/` (this repo's `src/` currently has `pipeline.py` at the
   top level importing `from portfolio_common import news_nlp as db` —
   `src/news_nlp/` sits alongside it as a new local package).
2. Copy its `tests/` into this repo's `tests/news_nlp/`, merging with the
   existing `tests/news_nlp/conftest.py` (this repo already has one that
   mirrors the upstream fixtures — reconcile duplicates rather than keeping
   two).
3. Copy its `docs/db-topology.md` into this repo's `docs/db-topology.md`,
   replacing/merging with what's there now.
4. Delete `business_folders/news_nlp/` from `portfolio-common` once the copy
   is verified working here (a follow-up PR against `portfolio-common`, not
   part of this repo's change).

## Import updates

Grep this repo for `portfolio_common.news_nlp` / `from portfolio_common
import news_nlp` and replace with the new local `news_nlp` package. Known
current call sites (confirmed by search when this plan was written — re-grep
before executing, this list may be stale):

- `src/pipeline.py:27-28` — `from portfolio_common import news_nlp as db`,
  `from portfolio_common.news_nlp.taxonomy import CATEGORY_LABELS,
  OTHER_LABEL` → `import news_nlp as db`, `from news_nlp.taxonomy import
  CATEGORY_LABELS, OTHER_LABEL`
- `apps/news_nlp_api.py`
- Tests: `tests/news_nlp/test_two_tier.py`,
  `tests/news_nlp/test_summary_pipeline.py`,
  `tests/news_nlp/test_query_endpoints.py`,
  `tests/news_nlp/test_correction_endpoints.py`,
  `tests/news_nlp/test_category_pipeline.py`,
  `tests/news_nlp/conftest.py`

## `queries.py` convention

The vendored package already follows this: every query function lives in one
ordered, documented place (`news_nlp/queries/` or `news_nlp/queries.py`
depending on how the extraction split it — check
`business_folders/news_nlp/README.md`), separate from orchestration/pure
business logic (`sector_summary`'s composition functions were split out from
its query functions upstream — check the final layout). Keep following this
convention for any new news_nlp queries added here.

## Injection-safety helpers

Keep `portfolio-common` as a runtime dependency after vendoring `news_nlp`,
for `portfolio_common.db.Database` (the vendored `news_nlp.db.NewsNlpDatabase`
subclasses it) and `in_clause`/`Allowlist` (used inside the vendored
`corrections.py` and `queries` modules already — verify these made it through
the extraction correctly rather than reverting to hand-rolled dicts).

## Version pin

Once this repo's own tests pass against the vendored `news_nlp`, bump
`pyproject.toml`'s `[tool.uv.sources]` pin:

```toml
portfolio-common = { git = "https://github.com/gamug/portfolio-common", tag = "v1.0.0" }
```

## Downstream: portfolio-knowledge-graph

`portfolio-knowledge-graph`'s `src/etl/news_to_rdf.py` imports
`connect_pipeline`/`fetch_processed_articles` directly from
`portfolio_common.news_nlp` today. Once `news_nlp` is vendored here instead
of living in `portfolio-common`, that repo needs a new way to reach these two
functions — most likely a git dependency on this repo (`portfolio-nlp`)
instead of (or in addition to) `portfolio-common`, importing
`from news_nlp import connect_pipeline, fetch_processed_articles`. This is
flagged in `portfolio-knowledge-graph`'s own migration-plan doc as a decision
point; coordinate before executing either plan in isolation, since
knowledge-graph's plan currently assumes this repo will expose those two
functions somehow.

## Verification

- `uv sync`, `uv run pytest`, `uv run ruff check .`, `uv run mypy src` all
  pass with the vendored `news_nlp` and updated imports.
- Run the two-tier ATTACH test suite specifically
  (`tests/news_nlp/test_two_tier.py` or wherever it lands) and confirm the
  stale-WAL preflight, SOURCE-never-written, and FK-consistency guarantees
  still hold — these encode real production incidents, don't just skim the
  pass/fail count.
- Smoke-test `apps/news_nlp_api.py` and `src/pipeline.py` against a scratch
  DB pair (SOURCE + RESULTS) end to end.

---
🤖 Generated with [Claude Code](https://claude.com/claude-code) as part of the
portfolio-common v1.0.0 DB-engine/business_folders split.
Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
