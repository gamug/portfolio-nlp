"""Canonical DDL for the news-NLP RESULTS store: the five result tables, each
keyed by ``article_id`` and ``REFERENCES articles(id)``.

Does **not** create ``articles`` -- that table is owned by the crawler on the
SOURCE side; on the RESULTS side a lean, ``body_text``-free subset is
populated row-by-row by ``news_nlp.db._ensure_article_row``.

The one engine-specific token in the DDL -- the auto-increment primary key
spelling -- comes from ``conn.dialect`` (``portfolio_common.db``); the
multi-statement run and the ``sector_summary`` self-heal go through
``Database.create_schema`` / ``Database.ensure_columns``. Nothing here names a
database engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from portfolio_common.db import get_dialect

if TYPE_CHECKING:
    from portfolio_common.db import Database, Dialect

# ``{autoincrement_pk}`` is substituted from the dialect by :func:`build_schema`
# (str.replace, not str.format -- the ``DEFAULT '{}'`` below would trip
# ``.format``). Everything else is standard SQL: ``REFERENCES``, ``UNIQUE``,
# ``NOT NULL DEFAULT``, ``CREATE TABLE IF NOT EXISTS``.
_SCHEMA_TEMPLATE = """
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
-- reprocessing. See portfolio-nlp's docs/category-taxonomy.md for what each
-- column means and where the taxonomy came from.
CREATE TABLE IF NOT EXISTS article_category (
    article_id INTEGER PRIMARY KEY REFERENCES articles(id),
    label TEXT NOT NULL,   -- winning category slug, or 'other'
    score REAL NOT NULL,   -- winning slug's NLI entailment probability (pre-threshold)
    earnings_performance REAL NOT NULL,
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
"""


def build_schema(dialect: Dialect | None = None) -> str:
    """The RESULTS-store DDL, with the auto-increment PK token filled in from
    *dialect* (the SQLite one by default)."""
    d = dialect if dialect is not None else get_dialect()
    return _SCHEMA_TEMPLATE.replace("{autoincrement_pk}", d.autoincrement_pk)


#: The DDL string as rendered for the default (SQLite) dialect -- kept for the
#: ``news_nlp`` public API and tests that introspect it directly.
SCHEMA = build_schema()

# Bumped whenever sector_summary's generation logic changes shape (e.g. the
# category-grouped deterministic-roll-up rewrite, then the facts_json/
# intro_text split for knowledge-graph-friendly output). fetch_pending_sector_weeks()
# treats any row below this value as stale, so legacy rows self-heal via
# INSERT OR REPLACE on the next sector_summary run instead of needing a
# separate backfill script.
SECTOR_SUMMARY_FORMAT_VERSION = 2

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


def init_schema(conn: Database) -> None:
    conn.create_schema(build_schema(conn.dialect))
    _migrate_sector_summary_schema(conn)
    conn.commit()
