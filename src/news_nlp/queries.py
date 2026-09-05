"""Article-level read/write helpers for the news-NLP RESULTS store.

Trace of capabilities -- every query function in this module, grouped
read/write, one line each:

Read (pipeline "pending" fetchers -- drive the "what's left to process" loop):
    fetch_pending_articles              -- (id, body_text) rows missing from a given result table
    fetch_pending_category_articles     -- (id, title, body_text) rows missing from article_category
    fetch_pending_company_summaries     -- raw fields for articles ready for c_summary generation
    build_company_summary_input         -- (pure, no SQL) assembles one c_summary prompt from a
                                            fetch_pending_company_summaries row

Read (plain export join for consumers outside portfolio-nlp):
    fetch_processed_articles            -- every fully-processed article as flat rows, unpaginated
                                            (thin wrapper over the shared portfolio_common.news_export
                                            implementation -- see that function's docstring)

Read (FastAPI query endpoints -- paginated, dict-per-call):
    list_articles                       -- filtered/paginated article list with sentiment+category
    get_article_detail                  -- one article's full detail (sentiment/entities/summary/category)
    sentiment_stats                     -- 3-way sentiment pivot, optionally grouped by company/year/month
    entity_stats                        -- top mentioned entities by count
    category_stats                      -- per-category-label article counts

Write (pipeline result writers -- each first upserts the lean `articles` row):
    write_sentiment                     -- upsert one article's sentiment result
    write_category                      -- upsert one article's category result + full NLI distribution
    write_entities                      -- replace one article's extracted entities
    write_company_summary               -- upsert one article's c_summary result

Helper (no SQL):
    now_iso                             -- current UTC timestamp, ISO 8601

The ``body_text`` readers (the pending fetchers plus ``fetch_processed_articles``)
qualify ``articles`` with ``db._articles_rel(conn)`` so they read
``source.articles`` during a two-tier pipeline run and ``main.articles`` when
serving; every ``write_*`` first ``db._ensure_article_row`` copies the lean
``articles`` row from SOURCE so the ``REFERENCES articles(id)`` foreign key holds.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from portfolio_common.db import Allowlist
from portfolio_common.news_export import (
    fetch_processed_articles as _shared_fetch_processed_articles,
)

from news_nlp.db import NewsNlpDatabase, _articles_rel, _ensure_article_row

_PENDING_ARTICLE_TABLES = {"article_sentiment", "article_entities"}

# `group_by` selects which SQL expression `sentiment_stats` groups by --
# checked against this Allowlist (rather than a bare dict .get(), which
# silently produced `NULL AS group_key` -- an ungrouped result -- for any
# unrecognized `group_by` instead of telling the caller their argument was
# wrong) before it ever reaches the f-string below.
_SENTIMENT_STATS_GROUP_EXPR = {
    "company": "a.company",
    "year": "strftime('%Y', a.pub_date)",
    "month": "strftime('%Y-%m', a.pub_date)",
}
_SENTIMENT_STATS_GROUP_BY = Allowlist(*_SENTIMENT_STATS_GROUP_EXPR)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def fetch_pending_articles(
    conn: NewsNlpDatabase, table: str, limit: int | None = None
) -> list[sqlite3.Row]:
    """Return (id, body_text) rows from `articles` not yet present in `table`,
    restricted to successfully fetched, non-empty articles."""
    if table not in _PENDING_ARTICLE_TABLES:
        raise ValueError(f"table must be one of {sorted(_PENDING_ARTICLE_TABLES)}, got {table!r}")
    # S608: `table` is checked against the _PENDING_ARTICLE_TABLES allowlist
    # above; _articles_rel(conn) is only ever "main" / "source".
    sql = f"""
        SELECT a.id, a.body_text
        FROM {_articles_rel(conn)}.articles a
        LEFT JOIN {table} r ON r.article_id = a.id
        WHERE r.article_id IS NULL
          AND a.fetch_status = 'ok'
          AND a.body_text IS NOT NULL
          AND TRIM(a.body_text) != ''
        ORDER BY a.id
    """  # noqa: S608
    params: list = []
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    return conn.execute(sql, params).fetchall()


def fetch_pending_category_articles(
    conn: NewsNlpDatabase, limit: int | None = None
) -> list[sqlite3.Row]:
    """Return (id, title, body_text) rows from `articles` not yet present in
    article_category, same eligibility filter as fetch_pending_articles. A
    dedicated query (not a widened fetch_pending_articles) since that
    function's (id, body_text) two-tuple shape is unpacked directly at the
    sentiment/NER call sites -- widening it would break those."""
    # S608: _articles_rel(conn) is only ever "main" / "source"; `limit` is
    # bound as a parameter below, not interpolated.
    sql = f"""
        SELECT a.id, a.title, a.body_text
        FROM {_articles_rel(conn)}.articles a
        LEFT JOIN article_category r ON r.article_id = a.id
        WHERE r.article_id IS NULL
          AND a.fetch_status = 'ok'
          AND a.body_text IS NOT NULL
          AND TRIM(a.body_text) != ''
        ORDER BY a.id
    """  # noqa: S608
    params: list = []
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    return conn.execute(sql, params).fetchall()


def fetch_processed_articles(conn: NewsNlpDatabase, limit: int | None = None) -> list[sqlite3.Row]:
    """Every successfully-fetched article that has both a sentiment and a
    category result, as one flat row per article: ``id, ticker, pub_date,
    fetched_at, body_text, positive, negative, sent_processed_at, cat_label,
    cat_score, cat_processed_at``.

    Unlike the ``fetch_pending_*`` functions (which drive the pipeline's own
    "what's left to process" loop) or ``list_articles``/``get_article_detail``
    (shaped for the FastAPI query endpoints, paginated and dict-per-call), this
    is a plain read-only export join meant for a consumer outside
    ``portfolio-nlp`` entirely -- e.g. ``portfolio-knowledge-graph``'s ETL,
    which wants every processed article as rows, unpaginated.

    A thin wrapper, not a second copy of the SQL: the join itself lives in
    ``portfolio_common.news_export.fetch_processed_articles``, the one piece
    of this schema's read contract that's genuinely shared across repos (see
    that module's docstring). This wrapper's only job is resolving
    ``_articles_rel(conn)`` -- hiding the SOURCE/RESULTS split from callers
    the same way the pipeline's own readers do, so a caller here only needs
    ``connect_pipeline()`` and this function, no knowledge of ``ATTACH``.
    """
    return _shared_fetch_processed_articles(conn, _articles_rel(conn), limit=limit)


def write_sentiment(
    conn: NewsNlpDatabase,
    article_id: int,
    label: str,
    score: float,
    positive: float,
    negative: float,
    neutral: float,
    model_name: str,
) -> None:
    _ensure_article_row(conn, article_id)
    conn.execute(
        """INSERT OR REPLACE INTO article_sentiment
           (article_id, label, score, positive, negative, neutral, model_name, processed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (article_id, label, score, positive, negative, neutral, model_name, now_iso()),
    )


def write_category(
    conn: NewsNlpDatabase,
    article_id: int,
    label: str,
    score: float,
    scores: dict[str, float],
    model_name: str,
) -> None:
    """`scores` must have one entry per taxonomy.CATEGORY_SLUGS slug (the full
    9-way distribution) -- `label`/`score` are the winning slug (or 'other') and
    its probability, kept separately from the raw distribution so a human
    correction (see corrections.update_category) can change the winner without
    touching the audit trail."""
    _ensure_article_row(conn, article_id)
    conn.execute(
        """INSERT OR REPLACE INTO article_category
           (article_id, label, score, earnings_performance, mergers_acquisitions,
            leadership_governance, legal_regulatory, product_innovation,
            capital_shareholder_returns, labor_human_capital, market_analyst_sentiment,
            partnerships_business_dev, model_name, processed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            article_id,
            label,
            score,
            scores["earnings_performance"],
            scores["mergers_acquisitions"],
            scores["leadership_governance"],
            scores["legal_regulatory"],
            scores["product_innovation"],
            scores["capital_shareholder_returns"],
            scores["labor_human_capital"],
            scores["market_analyst_sentiment"],
            scores["partnerships_business_dev"],
            model_name,
            now_iso(),
        ),
    )


def write_entities(
    conn: NewsNlpDatabase, article_id: int, entities: list[dict], model_name: str
) -> None:
    _ensure_article_row(conn, article_id)
    # Idempotency: clear any prior entities for this article before inserting fresh ones.
    conn.execute("DELETE FROM article_entities WHERE article_id = ?", (article_id,))
    ts = now_iso()
    conn.executemany(
        """INSERT INTO article_entities
           (article_id, entity_type, text, start_char, end_char, score, model_name, processed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                article_id,
                e["entity_type"],
                e["text"],
                e["start_char"],
                e["end_char"],
                e.get("score"),
                model_name,
                ts,
            )
            for e in entities
        ],
    )


def fetch_pending_company_summaries(
    conn: NewsNlpDatabase, limit: int | None = None
) -> list[sqlite3.Row]:
    """Return raw fields for articles ready for c_summary generation: a
    successful fetch (http_status_code=200), a computed sentiment, at least
    one qualifying entity (score>0.8, non-numeric), and no article_summary
    row yet. Adapts query.sql's `source_text` CTE -- the two INNER JOINs mean
    an article with sentiment but zero qualifying entities is never selected
    here, same as the original query.

    Returns raw columns rather than the assembled template text: SQLite
    string literals don't interpret \\n as an escape (unlike Python), so
    template assembly happens in build_company_summary_input() instead.
    """
    # S608: _articles_rel(conn) is only ever "main" / "source"; every value is
    # bound as a parameter. The result-table joins stay in `main`.
    sql = f"""
        WITH entities AS (
            SELECT article_id, GROUP_CONCAT(text, ', ') AS entities
            FROM article_entities
            WHERE score > 0.8 AND text NOT GLOB '[0-9]'
            GROUP BY article_id
        )
        SELECT
            a.id AS article_id, a.ticker, a.company, a.gics_sector, a.gics_sub_industry,
            a.title, a.body_text, s.label AS sentiment_label, s.score AS sentiment_confidence,
            e.entities
        FROM {_articles_rel(conn)}.articles a
        INNER JOIN article_sentiment s ON s.article_id = a.id
        INNER JOIN entities e ON e.article_id = a.id
        LEFT JOIN article_summary asum ON asum.article_id = a.id
        WHERE a.http_status_code = 200 AND asum.article_id IS NULL
        ORDER BY a.id
    """  # noqa: S608
    params: list = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def build_company_summary_input(row: sqlite3.Row) -> str:
    """Assemble the METADATA/NLP FEATURES/TEXT BODY template (query.sql's
    intent) with real newlines, for one row from fetch_pending_company_summaries."""
    return (
        f"METADATA:\nTicker-{row['ticker']}\nCompany-{row['company']}\n\n"
        f"NLP FEATURES:\nSentiment-{row['sentiment_label']} "
        f"Confidence-{row['sentiment_confidence']}\nEntities-{row['entities']}\n\n"
        f"TEXT BODY:\nTitle-{row['title']}\nBody-{row['body_text']}"
    )


def write_company_summary(
    conn: NewsNlpDatabase, article_id: int, summary_text: str, num_chunks: int, model_name: str
) -> None:
    _ensure_article_row(conn, article_id)
    conn.execute(
        """INSERT OR REPLACE INTO article_summary
           (article_id, summary_text, num_chunks, model_name, processed_at)
           VALUES (?, ?, ?, ?, ?)""",
        (article_id, summary_text, num_chunks, model_name, now_iso()),
    )


def list_articles(
    conn: NewsNlpDatabase,
    company: str | None = None,
    ticker: str | None = None,
    sentiment: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    sql = """
        SELECT a.id, a.company, a.ticker, a.title, a.pub_date,
               s.label AS sentiment_label, s.score AS sentiment_score,
               c.label AS category_label,
               (SELECT COUNT(*) FROM article_entities e WHERE e.article_id = a.id) AS entity_count
        FROM articles a
        LEFT JOIN article_sentiment s ON s.article_id = a.id
        LEFT JOIN article_category c ON c.article_id = a.id
        WHERE 1=1
    """
    params: list = []
    if company:
        sql += " AND a.company = ?"
        params.append(company)
    if ticker:
        sql += " AND a.ticker = ?"
        params.append(ticker)
    if sentiment:
        sql += " AND s.label = ?"
        params.append(sentiment)
    if category:
        sql += " AND c.label = ?"
        params.append(category)
    if date_from:
        sql += " AND a.pub_date >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND a.pub_date <= ?"
        params.append(date_to)
    sql += " ORDER BY a.pub_date DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_article_detail(conn: NewsNlpDatabase, article_id: int) -> dict | None:
    article = conn.execute(
        """SELECT id, company, ticker, title, author, pub_date, word_count, source_domain
           FROM articles WHERE id = ?""",
        (article_id,),
    ).fetchone()
    if article is None:
        return None

    sentiment_row = conn.execute(
        """SELECT label, score, positive, negative, neutral, model_name
           FROM article_sentiment WHERE article_id = ?""",
        (article_id,),
    ).fetchone()

    entity_rows = conn.execute(
        """SELECT id, entity_type, text, start_char, end_char, score
           FROM article_entities WHERE article_id = ? ORDER BY start_char""",
        (article_id,),
    ).fetchall()

    summary_row = conn.execute(
        """SELECT summary_text, num_chunks, model_name
           FROM article_summary WHERE article_id = ?""",
        (article_id,),
    ).fetchone()

    category_row = conn.execute(
        """SELECT label, score, earnings_performance, mergers_acquisitions, leadership_governance,
                  legal_regulatory, product_innovation, capital_shareholder_returns,
                  labor_human_capital, market_analyst_sentiment, partnerships_business_dev, model_name
           FROM article_category WHERE article_id = ?""",
        (article_id,),
    ).fetchone()

    return {
        **dict(article),
        "sentiment": dict(sentiment_row) if sentiment_row else None,
        "entities": [dict(r) for r in entity_rows],
        "summary": dict(summary_row) if summary_row else None,
        "category": dict(category_row) if category_row else None,
    }


def sentiment_stats(
    conn: NewsNlpDatabase,
    company: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    group_by: str | None = None,
) -> list[dict]:
    """`group_by`, when given, must be one of `_SENTIMENT_STATS_GROUP_BY`
    ("company" / "year" / "month") -- checked up front so an unrecognized
    value raises rather than silently falling back to an ungrouped result
    (the pre-Allowlist behavior: a bare dict `.get(group_by)` returned `None`
    for any unknown key, which produced `NULL AS group_key` with no error)."""
    if group_by is not None:
        _SENTIMENT_STATS_GROUP_BY.check(group_by)
    # S608: `group_expr` only ever comes from the fixed
    # _SENTIMENT_STATS_GROUP_EXPR literals above (group_by is validated
    # against the same map's keys just above), never from `group_by` itself.
    group_expr = _SENTIMENT_STATS_GROUP_EXPR.get(group_by) if group_by else None
    select_group = f"{group_expr} AS group_key," if group_expr else "NULL AS group_key,"
    sql = f"""
        SELECT {select_group}
               SUM(CASE WHEN s.label = 'positive' THEN 1 ELSE 0 END) AS positive,
               SUM(CASE WHEN s.label = 'negative' THEN 1 ELSE 0 END) AS negative,
               SUM(CASE WHEN s.label = 'neutral' THEN 1 ELSE 0 END) AS neutral,
               COUNT(*) AS total
        FROM article_sentiment s
        JOIN articles a ON a.id = s.article_id
        WHERE 1=1
    """  # noqa: S608
    params: list = []
    if company:
        sql += " AND a.company = ?"
        params.append(company)
    if date_from:
        sql += " AND a.pub_date >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND a.pub_date <= ?"
        params.append(date_to)
    if group_expr:
        sql += f" GROUP BY {group_expr} ORDER BY group_key"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def entity_stats(
    conn: NewsNlpDatabase,
    company: str | None = None,
    entity_type: str | None = None,
    top: int = 20,
) -> list[dict]:
    sql = """
        SELECT e.text, e.entity_type, COUNT(*) AS count
        FROM article_entities e
        JOIN articles a ON a.id = e.article_id
        WHERE 1=1
    """
    params: list = []
    if company:
        sql += " AND a.company = ?"
        params.append(company)
    if entity_type:
        sql += " AND e.entity_type = ?"
        params.append(entity_type)
    sql += " GROUP BY e.text, e.entity_type ORDER BY count DESC LIMIT ?"
    params.append(top)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def category_stats(
    conn: NewsNlpDatabase,
    company: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Per-label article counts. Unlike sentiment_stats' fixed 3-way pivot
    (justified there by sentiment's permanently-fixed 3-class contract), 10
    label values read better as label/count rows -- same shape as
    entity_stats."""
    sql = """
        SELECT c.label, COUNT(*) AS count
        FROM article_category c
        JOIN articles a ON a.id = c.article_id
        WHERE 1=1
    """
    params: list = []
    if company:
        sql += " AND a.company = ?"
        params.append(company)
    if date_from:
        sql += " AND a.pub_date >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND a.pub_date <= ?"
        params.append(date_to)
    sql += " GROUP BY c.label ORDER BY count DESC"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]
