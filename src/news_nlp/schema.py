"""Canonical DDL for the news-NLP RESULTS store: the five result tables, each
keyed by ``article_id`` and ``REFERENCES articles(id)``, plus the run-log
tables written by ``news_nlp.eval`` (the LLM-as-judge accuracy evaluation --
see ``docs/evaluation.md``): ``eval_run``, ``eval_inference``/``eval_verdict``
(current), ``eval_confusion`` (sentiment/category only), and the now-legacy
``eval_judgement`` (superseded 2026-09-18, kept as-is for its historical
rows -- see each table's own DDL comment).

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

-- On the working RESULTS store, `article_entities_v1` may exist alongside
-- this table -- not part of this DDL (nothing here creates or migrates it).
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
-- reprocessing. See portfolio-nlp's docs/category-taxonomy.md for what each
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

-- LLM-as-judge accuracy evaluation (news_nlp.eval). One eval_run per
-- (stage, invocation); eval_inference/eval_verdict (below) hold the
-- per-sampled-row model inference and judge verdict, split so multiple
-- experiments' data for the same article/task can coexist (PLAN.md Work
-- item 10 step 4, TASKS.md T-090, SPEC.md FR-014). Not keyed to
-- articles(id) by a foreign key: a run's sample is a point-in-time
-- snapshot and rows can be re-processed/corrected after. The full metrics
-- blob is also logged to MLflow; metrics_json here is the queryable copy
-- behind GET /eval/latest. See docs/evaluation.md.
CREATE TABLE IF NOT EXISTS eval_run (
    id {autoincrement_pk},
    stage         TEXT NOT NULL,   -- sentiment | category | ner | c_summary | sector_summary
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    sample_size   INTEGER NOT NULL,
    low_conf_n    INTEGER NOT NULL,
    random_n      INTEGER NOT NULL,  -- sum of every non-low_conf stratum's drawn count
    seed          INTEGER,
    judge_model   TEXT NOT NULL,
    judge_url     TEXT NOT NULL,
    code_version  TEXT NOT NULL,
    mlflow_run_id TEXT,
    metrics_json  TEXT NOT NULL DEFAULT '{}',
    strata_json   TEXT NOT NULL DEFAULT '{}',  -- {bucket: {"population": N_h, "n": n_h}}; '{}' for pre-stratification runs
    status        TEXT NOT NULL DEFAULT 'running',   -- running | ok | error
    error         TEXT,
    experiment    TEXT NOT NULL DEFAULT 'base'  -- which model produced this run's inferences
                                                 -- ("base" for the pinned production model, or
                                                 -- --candidate-model/--run-name); 'base' for
                                                 -- pre-2026-09-18 rows -- see docs/evaluation.md
);

CREATE INDEX IF NOT EXISTS idx_eval_run_stage_started
    ON eval_run(stage, started_at);

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

-- The sampled model inference being evaluated -- one row per judged
-- article/task/experiment/run. `task` mirrors eval_run.stage verbatim
-- (kept as its own column so a row is self-describing without a join).
-- `experiment` is a free-form label identifying which model/candidate
-- produced this inference (`base` for the pinned production model,
-- `v2`/`v3`/... for a candidate scored via news_nlp.eval.candidate,
-- SPEC.md FR-013) -- lets two experiments' rows for the same
-- article_id/task coexist instead of colliding (the "separate scratch
-- database per candidate" friction PLAN.md Work item 10 names). The
-- UNIQUE constraint is a defensive anti-double-insert guard within one
-- run, not a reuse/lookup mechanism -- re-running the same experiment
-- still produces a new row per run_id (T-091 builds the reuse lookup on
-- top of this shape). For task='sector_summary', article_id is actually
-- a sector_summary.id, not an articles.id -- see
-- news_nlp.eval.sampling.EvalItem's own docstring; unchanged by this split.
CREATE TABLE IF NOT EXISTS eval_inference (
    id              {autoincrement_pk},
    run_id          INTEGER NOT NULL REFERENCES eval_run(id),
    article_id      INTEGER NOT NULL,
    task            TEXT NOT NULL,
    experiment      TEXT NOT NULL,
    bucket          TEXT NOT NULL,
    prediction_json TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE (article_id, task, experiment, run_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_inference_run_id ON eval_inference(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_inference_article_task_experiment
    ON eval_inference(article_id, task, experiment);

-- The LLM judge's verdict on one eval_inference row. article_id/task/
-- experiment are deliberately duplicated from eval_inference (not just
-- reachable via inference_id) so a query never needs a join to know what
-- a verdict row is about (SPEC.md FR-014's own requirement). The UNIQUE
-- constraint stays 4 columns (run_id included), not the 3-column
-- (article_id, task, experiment) PLAN.md Work item 5 also allows for --
-- this project keeps a full eval_inference+eval_verdict row every run
-- (history preserved), reusing a prior verdict's *content* to skip the
-- judge LLM call rather than skipping the row itself (TASKS.md T-091,
-- SPEC.md FR-015 -- see news_nlp.eval.store.find_verdict_json, the
-- idx_eval_verdict_article_task_experiment index below is that lookup's
-- own dedicated index, since the 4-column UNIQUE's own index, while
-- leftmost-prefix-compatible, isn't purpose-built for it).
CREATE TABLE IF NOT EXISTS eval_verdict (
    id             {autoincrement_pk},
    inference_id   INTEGER NOT NULL REFERENCES eval_inference(id),
    run_id         INTEGER NOT NULL REFERENCES eval_run(id),
    article_id     INTEGER NOT NULL,
    task           TEXT NOT NULL,
    experiment     TEXT NOT NULL,
    verdict_json   TEXT NOT NULL,
    correct        INTEGER,         -- 0/1 for label stages; NULL for ner/c_summary
    severity       INTEGER,
    rationale      TEXT,
    judged_at      TEXT NOT NULL,
    UNIQUE (article_id, task, experiment, run_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_verdict_run_id ON eval_verdict(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_verdict_inference_id ON eval_verdict(inference_id);
CREATE INDEX IF NOT EXISTS idx_eval_verdict_article_task_experiment
    ON eval_verdict(article_id, task, experiment);

-- One row per distinct (true_label, predicted_label) cell actually
-- observed in one run's own sample -- sentiment/category only (the only
-- two stages with a discrete predicted/ideal label shape; TASKS.md T-092,
-- SPEC.md FR-016). Sparse: a cell with zero occurrences this run simply
-- has no row. Scoped per run_id (full history, matching eval_inference/
-- eval_verdict's own precedent) -- a cumulative matrix across every
-- historical run for an experiment is `SUM(count) GROUP BY (experiment,
-- task, true_label, predicted_label)` at query time, not something this
-- table maintains at write time.
CREATE TABLE IF NOT EXISTS eval_confusion (
    id              {autoincrement_pk},
    run_id          INTEGER NOT NULL REFERENCES eval_run(id),
    task            TEXT NOT NULL,
    experiment      TEXT NOT NULL,
    true_label      TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    count           INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE (run_id, task, experiment, true_label, predicted_label)
);

CREATE INDEX IF NOT EXISTS idx_eval_confusion_run_id ON eval_confusion(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_confusion_experiment_task
    ON eval_confusion(experiment, task);
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


_EVAL_RUN_ADDED_COLUMNS = {
    "strata_json": "TEXT NOT NULL DEFAULT '{}'",
    "experiment": "TEXT NOT NULL DEFAULT 'base'",
}


def _migrate_eval_run_schema(conn: Database) -> None:
    """Bring a pre-existing `eval_run` table (created before the stratified-
    sampling redesign added strata_json, or before the 2026-09-18
    `experiment` column) up to the current schema. Additive
    only -- low_conf_n/random_n stay as columns, their meaning generalized
    (random_n = count of every non-low_conf stratum combined), not removed.
    Idempotent, same `ensure_columns` no-op-when-missing/already-current
    pattern as `_migrate_sector_summary_schema`. Legacy rows read back
    strata_json='{}'/experiment='base' (the column defaults) -- see docs/evaluation.md.
    """
    conn.ensure_columns("eval_run", _EVAL_RUN_ADDED_COLUMNS)


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
    docs/category-taxonomy.md for the (separate, not-yet-built) backfill
    this implies if one is ever wanted.
    """
    conn.ensure_columns("article_category", _CATEGORY_ADDED_COLUMNS)


def init_schema(conn: Database) -> None:
    conn.create_schema(build_schema(conn.dialect))
    _migrate_sector_summary_schema(conn)
    _migrate_eval_run_schema(conn)
    _migrate_category_schema(conn)
    conn.commit()
