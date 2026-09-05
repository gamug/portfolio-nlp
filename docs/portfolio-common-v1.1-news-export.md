# Coordination note: `portfolio_common.news_export` (v1.1.0)

## What this resolves

`docs/portfolio-common-v1-migration-plan.md` (the v1.0.0 migration executed
in [#21](https://github.com/gamug/portfolio-nlp/pull/21)) flagged one open
decision: `portfolio-knowledge-graph`'s `src/etl/news_to_rdf.py` imports
`connect_pipeline`/`fetch_processed_articles` and, with `news_nlp` vendored
into this repo instead of shipped by `portfolio-common`, needed a new place
to get them from. Its own migration-plan doc listed two options — depend on
`portfolio-nlp` directly, or duplicate the query locally — and picked the
duplication option first (no tagged `portfolio-nlp` release existed yet to
depend on), landed as `portfolio-knowledge-graph`'s
`refactor/portfolio-common-v1-local-queries` PR. That PR's own docstring
flagged the real cost of that choice: a second copy of the join query that
has to be hand-kept in sync with this repo's schema.

`portfolio-common` v1.1.0
([gamug/portfolio-common#8](https://github.com/gamug/portfolio-common/pull/8))
resolves this properly: it adds `portfolio_common.news_export`, carrying
*only* the read-only two-tier connect (`connect_readonly`) and the
`fetch_processed_articles` join — the narrow slice of the news_nlp read
contract that two separate repos actually need, not a re-merge of the whole
domain. `news_nlp.queries.fetch_processed_articles` here is now a thin
delegate to that shared implementation instead of carrying its own copy of
the SQL, and `portfolio-knowledge-graph`'s local duplicate is retired the
same way.

## Why not the other two options

- **Depend on `portfolio-nlp` directly** (the other option on the table):
  would have coupled `portfolio-knowledge-graph`'s dependency graph to this
  repo's release cadence for the sake of two read-only functions, and pulled
  in everything else `news_nlp` ships (torch-adjacent tooling, the whole
  write-side pipeline) that a pure reader has no use for.
- **Reintroduce all of `news_nlp` into `portfolio-common`**: would have
  undone the actual point of the v1.0.0 split (`news_nlp` has exactly one
  owner — this repo — for schema, corrections, taxonomy, and
  `sector_summary`). `news_export` only carries the one join that is
  genuinely shared by two repos, not this repo's whole domain.

## What changed here

- `src/news_nlp/queries.py`'s `fetch_processed_articles` now calls
  `portfolio_common.news_export.fetch_processed_articles(conn,
  _articles_rel(conn), limit=limit)` instead of running its own copy of the
  join. Same signature, same behavior, same tests — verified against
  `tests/news_nlp/test_queries.py`'s existing `fetch_processed_articles`
  coverage.
- `pyproject.toml`'s `portfolio-common` pin moves to the `v1.1.0` line (see
  that file's own comment for the interim commit pin used while
  `gamug/portfolio-common#8` is still unmerged/untagged).

## Companion PRs

- `portfolio-common`: [#8](https://github.com/gamug/portfolio-common/pull/8)
  — adds `news_export`.
- `portfolio-knowledge-graph`: retires `src/etl/queries.py`'s local copy,
  imports `portfolio_common.news_export` directly.

---
🤖 Generated with [Claude Code](https://claude.com/claude-code) as part of
coordinating the `portfolio-knowledge-graph` dependency switch across
`portfolio-common`, `portfolio-nlp`, and `portfolio-knowledge-graph`.
Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
