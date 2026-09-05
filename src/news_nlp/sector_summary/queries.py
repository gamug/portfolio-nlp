"""The ``sector_summary`` stage's DB layer: fetch the week's c_summary'd
articles and their aggregate stats, and persist/list the composed rows.

Trace of capabilities -- every query function in this module, grouped
read/write:

Read:
    fetch_pending_sector_weeks              -- closed (sector, sub_industry, week) groups ready for
                                                generation, not yet in sector_summary at the current
                                                SECTOR_SUMMARY_FORMAT_VERSION
    fetch_company_summaries_for_sector_week -- the article_summary rows (+ company/ticker/category/
                                                sentiment) contributing to one such group
    fetch_sector_week_entity_stats          -- top mentioned entities for one such group
    list_sector_summaries                   -- filtered list of persisted sector_summary rows

Write:
    write_sector_summary                    -- upsert one (sector, sub_industry, week) summary row

The pure, no-SQL composition logic that turns these rows into a
``sector_summary`` (``compose_sector_summary``, ``build_sector_facts``,
``build_sector_intro_seed``, ``clean_generated_text``) lives in
``news_nlp.sector_summary.composition`` -- a separate module since none of it
takes a ``conn``/``db`` argument or contains SQL text.

The week-bucketing expressions, the "today" bound, the bare-digit entity
filter and the upsert verb come from ``conn.dialect`` -- nothing here names a
database engine.
"""

from __future__ import annotations

import json

from portfolio_common.db import Row

from news_nlp.db import NewsNlpDatabase
from news_nlp.queries import now_iso
from news_nlp.schema import SECTOR_SUMMARY_FORMAT_VERSION

# Weeks are bucketed off pub_date, falling back to fetched_at when pub_date is
# NULL. The Monday-start / Sunday-end expressions themselves are the dialect's
# (SQLite: date(col, 'weekday 0', ...)).
_WEEK_DATE_COL = "COALESCE(a.pub_date, a.fetched_at)"

_SECTOR_SUMMARY_COLS = (
    "gics_sector",
    "gics_sub_industry",
    "week_start",
    "week_end",
    "summary_text",
    "num_articles",
    "num_companies",
    "model_name",
    "facts_json",
    "intro_text",
    "format_version",
    "processed_at",
)
_SECTOR_SUMMARY_CONFLICT = ("gics_sector", "gics_sub_industry", "week_start")


def fetch_pending_sector_weeks(conn: NewsNlpDatabase, limit: int | None = None) -> list[Row]:
    """Return (gics_sector, gics_sub_industry, week_start, week_end) tuples
    ready for sector_summary generation: closed weeks (week_end already in
    the past, so a partial week is never summarized and later regenerated)
    with at least one c_summary'd article, not yet in sector_summary.
    Weeks are bucketed off pub_date, falling back to fetched_at when
    pub_date is NULL.
    """
    week_start_expr = conn.dialect.week_start_expr(_WEEK_DATE_COL)
    week_end_expr = conn.dialect.week_end_expr(_WEEK_DATE_COL)
    # S608: week_start_expr / week_end_expr / current_date_expr are dialect
    # fragments, not caller input; the format_version below is bound.
    sql = f"""
        SELECT
            a.gics_sector AS gics_sector,
            a.gics_sub_industry AS gics_sub_industry,
            {week_start_expr} AS week_start,
            {week_end_expr} AS week_end
        FROM article_summary asum
        JOIN articles a ON a.id = asum.article_id
        WHERE a.gics_sector IS NOT NULL AND a.gics_sub_industry IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM sector_summary ss
              WHERE ss.gics_sector = a.gics_sector
                AND ss.gics_sub_industry = a.gics_sub_industry
                AND ss.week_start = {week_start_expr}
                AND ss.format_version = ?
          )
        GROUP BY a.gics_sector, a.gics_sub_industry, week_start
        HAVING week_end < {conn.dialect.current_date_expr()}
        ORDER BY week_start, a.gics_sector, a.gics_sub_industry
    """  # noqa: S608
    # A row at an older format_version doesn't satisfy the NOT EXISTS check,
    # so it's treated as still-pending here -- the next sector_summary run
    # naturally regenerates and overwrites it via the dialect upsert.
    params: list = [SECTOR_SUMMARY_FORMAT_VERSION]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def fetch_company_summaries_for_sector_week(
    conn: NewsNlpDatabase, gics_sector: str, gics_sub_industry: str, week_start: str
) -> list[Row]:
    """Return the article_summary rows (with company/ticker/category/sentiment)
    contributing to one (gics_sector, gics_sub_industry, week_start)
    sector_summary. INNER JOINs to article_category and article_sentiment:
    an article whose c_summary exists but has no article_category row (e.g.
    historical data predating the category stage becoming mandatory, or a
    direct/partial stage invocation) is excluded entirely rather than
    bucketed as "uncategorized" -- a deliberate scope decision, not an
    oversight. The article_sentiment JOIN never drops a row in practice:
    c_summary generation itself already requires a sentiment row to exist
    (see fetch_pending_company_summaries's INNER JOIN), and sentiment rows
    are never deleted afterwards.
    """
    week_start_expr = conn.dialect.week_start_expr(_WEEK_DATE_COL)
    # S608: week_start_expr is a dialect fragment, not caller input;
    # gics_sector/sub_industry/week_start below are bound as query params.
    sql = f"""
        SELECT asum.article_id, asum.summary_text, a.ticker, a.company,
               c.label AS category_label, s.label AS sentiment_label
        FROM article_summary asum
        JOIN articles a ON a.id = asum.article_id
        JOIN article_category c ON c.article_id = a.id
        JOIN article_sentiment s ON s.article_id = a.id
        WHERE a.gics_sector = ? AND a.gics_sub_industry = ?
          AND {week_start_expr} = ?
        ORDER BY c.label, a.company, asum.article_id
    """  # noqa: S608
    return conn.execute(sql, (gics_sector, gics_sub_industry, week_start)).fetchall()


def fetch_sector_week_entity_stats(
    conn: NewsNlpDatabase,
    gics_sector: str,
    gics_sub_industry: str,
    week_start: str,
    top: int = 10,
) -> list[dict]:
    """Top mentioned entities for one (sector, sub_industry, week) group,
    scoped to the same c_summary'd articles fetch_company_summaries_for_sector_week
    draws from (same week-bucketing, joined through article_summary). Same
    qualifying-entity filter (score>0.8, non-numeric) used by
    fetch_pending_company_summaries."""
    week_start_expr = conn.dialect.week_start_expr(_WEEK_DATE_COL)
    # S608: week_start_expr and the bare-digit predicate are dialect
    # fragments; gics_sector/sub_industry/week_start/top below are bound.
    sql = f"""
        SELECT e.text, e.entity_type, COUNT(*) AS count
        FROM article_entities e
        JOIN articles a ON a.id = e.article_id
        JOIN article_summary asum ON asum.article_id = a.id
        WHERE a.gics_sector = ? AND a.gics_sub_industry = ?
          AND {week_start_expr} = ?
          AND e.score > 0.8 AND {conn.dialect.excludes_bare_digit("e.text")}
        GROUP BY e.text, e.entity_type
        ORDER BY count DESC
        LIMIT ?
    """  # noqa: S608
    params = (gics_sector, gics_sub_industry, week_start, top)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def write_sector_summary(
    conn: NewsNlpDatabase,
    gics_sector: str,
    gics_sub_industry: str,
    week_start: str,
    week_end: str,
    summary_text: str,
    num_articles: int,
    num_companies: int,
    model_name: str,
    facts: dict | None = None,
    intro_text: str = "",
    format_version: int = SECTOR_SUMMARY_FORMAT_VERSION,
) -> None:
    """`facts`/`intro_text` default to an empty dict/string -- matching the
    facts_json/intro_text columns' own schema defaults -- so callers that
    only care about the prose `summary_text` (e.g. older tests) don't need
    to pass them. See composition.build_sector_facts and
    composition.clean_generated_text."""
    conn.execute(
        conn.dialect.upsert(
            "sector_summary", _SECTOR_SUMMARY_COLS, conflict=_SECTOR_SUMMARY_CONFLICT
        ),
        (
            gics_sector,
            gics_sub_industry,
            week_start,
            week_end,
            summary_text,
            num_articles,
            num_companies,
            model_name,
            json.dumps(facts if facts is not None else {}),
            intro_text,
            format_version,
            now_iso(),
        ),
    )


def list_sector_summaries(
    conn: NewsNlpDatabase,
    sector: str | None = None,
    sub_industry: str | None = None,
    week_start: str | None = None,
) -> list[dict]:
    sql = "SELECT * FROM sector_summary WHERE 1=1"
    params: list = []
    if sector:
        sql += " AND gics_sector = ?"
        params.append(sector)
    if sub_industry:
        sql += " AND gics_sub_industry = ?"
        params.append(sub_industry)
    if week_start:
        sql += " AND week_start = ?"
        params.append(week_start)
    sql += " ORDER BY week_start DESC, gics_sector, gics_sub_industry"
    results = [dict(r) for r in conn.execute(sql, params).fetchall()]
    # facts_json is stored as a TEXT column (see write_sector_summary's
    # json.dumps) -- decode it back to a real object here so API consumers
    # get a nested JSON value, not a JSON string they'd have to parse a
    # second time themselves.
    for r in results:
        r["facts_json"] = json.loads(r["facts_json"])
    return results
