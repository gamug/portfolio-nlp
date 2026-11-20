-- The DDL of the v1 result tables and of the legacy eval_judgement, as src/news_nlp/schema.py had it before T-220 PR C2 (2026-11-18). Kept as a record; nothing runs it.
-- {autoincrement_pk} is the dialect's auto-increment primary key (SQLite: INTEGER PRIMARY KEY AUTOINCREMENT).

-- ===== the five v1 result tables (article_sentiment, article_entities, article_summary, sector_summary, article_category) =====

CREATE TABLE IF NOT EXISTS article_sentiment (
    article_id INTEGER PRIMARY KEY REFERENCES articles(id),
    label TEXT NOT NULL,
    score REAL NOT NULL,
    positive REAL NOT NULL,
    negative REAL NOT NULL,
    neutral REAL NOT NULL,
    model_name TEXT NOT NULL,
    processed_at TEXT NOT NULL
);

-- On the working RESULTS store, `article_entities_v1` may exist alongside
-- this table -- not part of this DDL (nothing here creates or migrates it).
-- Checked 2026-11-20: `article_entities_v1` is on neither the 2026-11-18 live RESULTS file nor `nlp_backup.db`.
-- It's a one-time archival rename of the pre-2026-09-10-fix
-- `article_entities` (`scripts/resample_ner_2026_09_12.py`,
-- docs/evaluation.md's 2026-09-12 "T-025 executed" follow-up): the
-- pre-fix model's output for the ~439K articles not yet reprocessed under
-- the fixed merge_bio_predictions, preserved rather than deleted. Its own
-- index is `idx_article_entities_v1_article_id` (SQLite doesn't rename a
-- table's indexes on `ALTER TABLE ... RENAME TO`, so this had to be
-- recreated explicitly under a new name -- see that script for why).
CREATE TABLE IF NOT EXISTS article_entities (
    id {autoincrement_pk},
    article_id INTEGER NOT NULL REFERENCES articles(id),
    entity_type TEXT NOT NULL,   -- PER / LOC / ORG
    text TEXT NOT NULL,          -- surface span, e.g. "Apple Inc."
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    score REAL,
    model_name TEXT NOT NULL,
    processed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_article_entities_article_id
    ON article_entities(article_id);

CREATE TABLE IF NOT EXISTS article_summary (
    article_id   INTEGER PRIMARY KEY REFERENCES articles(id),
    summary_text TEXT NOT NULL,
    num_chunks   INTEGER NOT NULL,
    model_name   TEXT NOT NULL,
    processed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sector_summary (
    id                {autoincrement_pk},
    gics_sector       TEXT NOT NULL,
    gics_sub_industry TEXT NOT NULL,
    week_start        TEXT NOT NULL,
    week_end          TEXT NOT NULL,
    summary_text      TEXT NOT NULL,
    num_articles      INTEGER NOT NULL,
    num_companies     INTEGER NOT NULL,
    model_name        TEXT NOT NULL,
    -- facts_json: structured, non-narrative payload (sentiment/entity/category
    -- aggregates plus attributed per-company records) meant for programmatic
    -- consumers (e.g. knowledge-graph ingestion) -- see build_sector_facts.
    -- intro_text: the one model-generated sentence in summary_text, stored
    -- separately (and already run through clean_generated_text) so a
    -- consumer that only wants grounded facts can read facts_json and skip
    -- it entirely.
    facts_json        TEXT NOT NULL DEFAULT '{}',
    intro_text        TEXT NOT NULL DEFAULT '',
    format_version    INTEGER NOT NULL DEFAULT 0,
    processed_at      TEXT NOT NULL,
    UNIQUE (gics_sector, gics_sub_industry, week_start)
);

-- One row per article: the winning category (or 'other') plus the full
-- 9-way NLI score distribution, so low-confidence 'other' picks are
-- auditable and CATEGORY_CONFIDENCE_THRESHOLD can be retuned later without
-- reprocessing. See portfolio-nlp's docs/legacy/category-taxonomy-v1.md for what each
-- column means and where the taxonomy came from.
CREATE TABLE IF NOT EXISTS article_category (
    article_id INTEGER PRIMARY KEY REFERENCES articles(id),
    label TEXT NOT NULL,   -- winning category slug, or 'other'
    score REAL NOT NULL,   -- winning slug's NLI entailment probability (pre-threshold)
    group_label TEXT NOT NULL DEFAULT '',   -- winning level-1 group slug (taxonomy.CATEGORY_GROUPS);
                                             -- set even when label='other' via the flat-level-1 short-circuit
    group_score REAL NOT NULL DEFAULT 0.0,  -- winning group's NLI entailment probability (pre-floor)
    earnings_performance REAL NOT NULL,   -- 0.0 placeholder if this slug's group wasn't in the
                                           -- article's top-2 groups (see pipeline.run_category_stage)
    mergers_acquisitions REAL NOT NULL,
    leadership_governance REAL NOT NULL,
    legal_regulatory REAL NOT NULL,
    product_innovation REAL NOT NULL,
    capital_shareholder_returns REAL NOT NULL,
    labor_human_capital REAL NOT NULL,
    market_analyst_sentiment REAL NOT NULL,
    partnerships_business_dev REAL NOT NULL,
    model_name TEXT NOT NULL,
    processed_at TEXT NOT NULL
);

-- ===== eval_judgement (legacy) =====

-- Legacy: superseded 2026-09-18 by eval_inference/eval_verdict (below),
-- which split this row's two concerns and add task/experiment so more
-- than one experiment's data for the same article/task can coexist. Kept
-- as-is (DDL and any historical rows untouched) -- new eval runs no
-- longer write here. See docs/evaluation.md's 2026-09-18 follow-up.
CREATE TABLE IF NOT EXISTS eval_judgement (
    id {autoincrement_pk},
    run_id                INTEGER NOT NULL REFERENCES eval_run(id),
    article_id            INTEGER NOT NULL,
    bucket                TEXT NOT NULL,   -- low_conf | representative | stage-specific target_<x>
    model_prediction_json TEXT NOT NULL,
    verdict_json          TEXT NOT NULL,
    correct               INTEGER,         -- 0/1 for label stages; NULL for ner/c_summary
    severity              INTEGER,
    rationale             TEXT,
    judged_at             TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_eval_judgement_run_id
    ON eval_judgement(run_id);

-- ===== Python: the format version and the additive migrations of those tables =====

/*
# Bumped whenever sector_summary's generation logic changes shape (e.g. the
# category-grouped deterministic-roll-up rewrite, then the facts_json/
# intro_text split for knowledge-graph-friendly output, then 2026-09-14's
# switch to a deterministic intro_text template -- see pipeline.py's
# SECTOR_INTRO_METHOD comment and docs/evaluation.md's 2026-09-14 follow-up:
# the old model-paraphrase path hallucinated on 42-50% of rows). fetch_pending_sector_weeks()
# treats any row below this value as stale, so legacy rows self-heal via
# INSERT OR REPLACE on the next sector_summary run instead of needing a
# separate backfill script.
SECTOR_SUMMARY_FORMAT_VERSION = 3

_SECTOR_SUMMARY_ADDED_COLUMNS = {
    "format_version": "INTEGER NOT NULL DEFAULT 0",
    "facts_json": "TEXT NOT NULL DEFAULT '{}'",
    "intro_text": "TEXT NOT NULL DEFAULT ''",
}


def _migrate_sector_summary_schema(conn: Database) -> None:
    """Bring a pre-existing `sector_summary` table (created before
    format_version/facts_json/intro_text existed) up to the current schema.
    Idempotent -- safe to call on every startup, including against a table
    that's already current or was just freshly created by SCHEMA. Legacy
    rows land at format_version=0 (the column default), which
    fetch_pending_sector_weeks treats as stale/pending, so they self-heal
    via INSERT OR REPLACE on the next sector_summary run -- no separate
    backfill script needed.

    ``Database.ensure_columns`` is a no-op when the table is absent and only
    ADDs the columns that are missing, so this needs no existence guard of
    its own.
    """
    conn.ensure_columns("sector_summary", _SECTOR_SUMMARY_ADDED_COLUMNS)
*/

/*
_CATEGORY_ADDED_COLUMNS = {
    "group_label": "TEXT NOT NULL DEFAULT ''",
    "group_score": "REAL NOT NULL DEFAULT 0.0",
}


def _migrate_category_schema(conn: Database) -> None:
    """Bring a pre-existing `article_category` table (created before the
    two-level hierarchical classifier added group_label/group_score) up to
    the current schema. Additive only, same `ensure_columns`
    no-op-when-missing/already-current pattern as `_migrate_eval_run_schema`.
    Legacy rows read back group_label=''/group_score=0.0 (the column
    defaults). Unlike sector_summary's format_version self-heal,
    fetch_pending_category_articles only checks row-presence, so these
    legacy rows do NOT automatically get reprocessed -- see
    docs/legacy/category-taxonomy-v1.md for the (separate, not-yet-built) backfill
    this implies if one is ever wanted.
    """
    conn.ensure_columns("article_category", _CATEGORY_ADDED_COLUMNS)


# Unused by the stepped pipeline: the raw spans of step 3 are not stored (owner, 2026-11-13, PLAN D39 (iii)); the columns stay, nullable.
_ENTITIES_ADDED_COLUMNS = {"sub_news_id": "INTEGER", "ticker": "TEXT"}


def _migrate_entities_schema(conn: Database) -> None:
    """Add the nullable `sub_news_id` and `ticker` (D39 (iii)) to an existing `article_entities`.
    Additive and idempotent, like the other migrations; old rows read back NULL."""
    conn.ensure_columns("article_entities", _ENTITIES_ADDED_COLUMNS)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_article_entities_sub_news_id ON article_entities(sub_news_id)"
    )
*/

