# SPEC section 5: the rows of the v1 tables, removed by T-220 PR C2 (2026-11-18), verbatim

The tables were dropped from the working database on 2026-11-18 (`depurate_db`, PR #215) and their DDL is no longer in `schema.py` (kept as `docs/legacy/schema_v1_tables_ddl_2026-11-18.sql`). Replaced, not annotated (INDEX rule 3).

## the banner above the table

```
**Rows for `article_sentiment`, `article_entities` (and its target columns), `article_category`, `article_summary`, `sector_summary` and `eval_judgement` describe tables that T-220 PR C drops; they stay here until it runs.**
```

```
| `article_sentiment` | `article_id` (PK) | `label`, `score`, `positive`/`negative`/`neutral`, `model_name`, `processed_at` | Stage 1 |
```

```
| `article_entities` | `id` (PK, autoincrement) | `article_id` (FK, indexed), `entity_type` (`PER`/`LOC`/`ORG`), `text`, `start_char`/`end_char`, `score` | Stage 2 |
```

```
| `article_entities` target columns (D39 (iii), approved 2026-11-10) | same key | adds nullable `sub_news_id` and `ticker`; **unused since 2026-11-13** (owner: step 3 does not store the raw NER spans, `news_company.mention_count` keeps the count); `score` stays the NER confidence | none |
```

```
| `article_category` | `article_id` (PK) | `label` (9 slugs or `other`), `score`, `group_label`/`group_score` (level-1 hierarchy), 9 raw per-slug NLI scores (audit trail) | Stage 3 |
```

```
| `article_summary` | `article_id` (PK) | `summary_text`, `num_chunks`, `model_name`, `processed_at` | Stage 4 (opt-in) |
```

```
| `sector_summary` | `id` (PK); `UNIQUE(gics_sector, gics_sub_industry, week_start)` | `summary_text`, `facts_json` (structured, non-narrative), `intro_text` (deterministic template as of 2026-09-14, `model_name` reads `"deterministic-template"`), `format_version` (self-heal marker), `num_articles`/`num_companies` | Stage 5 (opt-in) |
```

```
| `eval_judgement` | `id` (PK) | `run_id` (FK), `article_id` (not FK-constrained — snapshot semantics), `bucket`, `verdict_json`, `correct`, `severity`, `rationale` | `news_nlp.eval` |
```

## Section 7 "Self-healing schema drift" (verbatim, removed by T-220 PR C2)

- **Self-healing schema drift**: `sector_summary.format_version`,
  `eval_run.strata_json`'s additive migration, and `article_category`'s
  `group_label`/`group_score` additive migration all follow the same
  pattern — `Database.ensure_columns` (no-op if already current), with
  `sector_summary` additionally treating any row below the current
  `SECTOR_SUMMARY_FORMAT_VERSION` as stale and regenerating it via
  `INSERT OR REPLACE` on next run (no separate backfill script; contrast
  `article_category`'s additive-only migration, which does **not**
  auto-reprocess legacy rows — see §13).

## FR-006 row (verbatim, replaced by T-220 PR C2)

| **FR-006** | Every stage is idempotent and resumable: it processes only rows missing from its own result table (or below `format_version`/needing a schema migration self-heal for `sector_summary`/`eval_run`/`article_category`). | Running the pipeline twice on unchanged input produces no duplicate or changed rows; a `sector_summary` row below `SECTOR_SUMMARY_FORMAT_VERSION` regenerates via `INSERT OR REPLACE` on the next run with no separate backfill script. |
