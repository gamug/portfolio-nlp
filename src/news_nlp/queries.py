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
    latest_eval_runs                    -- most recent news_nlp.eval run per stage (GET /eval/latest)

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

Nothing here names a database engine: the upsert verb, the ``INSERT`` column
list, ``GROUP_CONCAT``, the bare-digit predicate and the year/month grouping
expressions all come from ``conn.dialect`` (``portfolio_common.db``). The
remaining hand-assembled ``WHERE ... = ?`` / ``LIMIT ? OFFSET ?`` fragments use
the DB-API qmark marker directly -- a future non-qmark engine would route those
through ``conn.dialect.placeholder`` too; they are marked, not rewritten now.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from portfolio_common.db import Allowlist, Row
from portfolio_common.news_export import (
    fetch_processed_articles as _shared_fetch_processed_articles,
)

from news_nlp.db import NewsNlpDatabase, _articles_rel, _ensure_article_row

if TYPE_CHECKING:
    from portfolio_common.db import Dialect

_PENDING_ARTICLE_TABLES = {"article_sentiment", "article_entities"}

# Column tuples for the dialect-built INSERT / upsert statements -- kept next to
# nothing else so the accompanying params tuple stays in the same order.
_SENTIMENT_COLS = (
    "article_id",
    "label",
    "score",
    "positive",
    "negative",
    "neutral",
    "model_name",
    "processed_at",
)
_CATEGORY_COLS = (
    "article_id",
    "label",
    "score",
    "group_label",
    "group_score",
    "earnings_performance",
    "mergers_acquisitions",
    "leadership_governance",
    "legal_regulatory",
    "product_innovation",
    "capital_shareholder_returns",
    "labor_human_capital",
    "market_analyst_sentiment",
    "partnerships_business_dev",
    "model_name",
    "processed_at",
)
_ENTITY_COLS = (
    "article_id",
    "entity_type",
    "text",
    "start_char",
    "end_char",
    "score",
    "model_name",
    "processed_at",
)
_SUMMARY_COLS = ("article_id", "summary_text", "num_chunks", "model_name", "processed_at")

# `group_by` selects which grouping to apply in `sentiment_stats` -- checked
# against this Allowlist of keys (rather than a bare dict .get(), which silently
# produced an ungrouped result for any unrecognized `group_by`) before the
# matching SQL expression (from `_group_exprs`, dialect-built for year/month) is
# ever interpolated.
_SENTIMENT_STATS_GROUP_BY = Allowlist("company", "year", "month")


def _group_exprs(dialect: Dialect) -> dict[str, str]:
    """The SQL expression `sentiment_stats` groups by, per `group_by` key.
    ``company`` is a plain column; ``year`` / ``month`` are the dialect's
    date-truncation expressions (SQLite: ``strftime``)."""
    return {
        "company": "a.company",
        "year": dialect.year_expr("a.pub_date"),
        "month": dialect.year_month_expr("a.pub_date"),
    }


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def fetch_pending_articles(
    conn: NewsNlpDatabase, table: str, limit: int | None = None, *, sample_seed: int | None = None
) -> list[Row]:
    """Return (id, body_text) rows from `articles` not yet present in `table`,
    restricted to successfully fetched, non-empty articles.

    Plain call (`sample_seed=None`): the first `limit` pending rows,
    `ORDER BY a.id` -- the normal "work through the backlog in order" path
    every stage uses day to day.

    `sample_seed` given (`limit` then required, the sample size): a
    `random.Random(sample_seed)`-seeded, reproducible **random** sample of
    `limit` pending ids instead of the first `limit` by id order -- for a
    deliberately-random reprocessing pass (e.g. a targeted post-fix resample,
    docs/evaluation.md's 2026-09-12 NER follow-up / `PLAN.md` Work item 3
    T-025), not the routine backlog-order path. Two queries: ids only first
    (cheap -- no body_text pulled for the whole pending population), then
    body_text for just the sampled ids, mirroring `news_nlp.eval.sampling`'s
    own seeded-`random.Random` convention rather than SQL `ORDER BY RANDOM()`
    (not seedable/reproducible in SQLite).
    """
    if table not in _PENDING_ARTICLE_TABLES:
        raise ValueError(f"table must be one of {sorted(_PENDING_ARTICLE_TABLES)}, got {table!r}")
    # S608: `table` is checked against the _PENDING_ARTICLE_TABLES allowlist
    # above; _articles_rel(conn) is only ever "main" / "source".
    base_sql = f"""
        FROM {_articles_rel(conn)}.articles a
        LEFT JOIN {table} r ON r.article_id = a.id
        WHERE r.article_id IS NULL
          AND a.fetch_status = 'ok'
          AND a.body_text IS NOT NULL
          AND TRIM(a.body_text) != ''
    """

    if sample_seed is not None:
        if not limit:
            raise ValueError("sample_seed requires a positive limit (the sample size)")
        all_ids = [
            int(r[0]) for r in conn.execute(f"SELECT a.id {base_sql} ORDER BY a.id", []).fetchall()
        ]
        rng = random.Random(sample_seed)  # noqa: S311 -- sample selection, not cryptography
        sampled_ids = rng.sample(all_ids, k=min(limit, len(all_ids)))
        if not sampled_ids:
            return []
        placeholders = ",".join("?" for _ in sampled_ids)
        rows = conn.execute(
            f"SELECT a.id, a.body_text FROM {_articles_rel(conn)}.articles a "  # noqa: S608
            f"WHERE a.id IN ({placeholders})",
            sampled_ids,
        ).fetchall()
        by_id = {int(r[0]): r for r in rows}
        return [by_id[i] for i in sampled_ids if i in by_id]

    sql = f"SELECT a.id, a.body_text {base_sql} ORDER BY a.id"
    params: list = []
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    return conn.execute(sql, params).fetchall()


def fetch_pending_category_articles(conn: NewsNlpDatabase, limit: int | None = None) -> list[Row]:
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


def fetch_processed_articles(conn: NewsNlpDatabase, limit: int | None = None) -> list[Row]:
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
        conn.dialect.upsert("article_sentiment", _SENTIMENT_COLS, conflict=("article_id",)),
        (article_id, label, score, positive, negative, neutral, model_name, now_iso()),
    )


def write_category(
    conn: NewsNlpDatabase,
    article_id: int,
    label: str,
    score: float,
    scores: dict[str, float],
    model_name: str,
    group_label: str = "",
    group_score: float = 0.0,
) -> None:
    """`scores` must have one entry per taxonomy.CATEGORY_SLUGS slug (the full
    9-way distribution -- 0.0 placeholders for whichever group didn't make an
    article's top-2 at level 1, see pipeline.run_category_stage) --
    `label`/`score` are the winning leaf slug (or 'other') and its
    probability, kept separately from the raw distribution so a human
    correction (see corrections.update_category) can change the winner
    without touching the audit trail. `group_label`/`group_score` are the
    winning level-1 group and its probability, kept alongside for the
    two-level audit trail (docs/category-taxonomy.md) -- default to
    ''/0.0 (matching the column defaults) for callers that don't have a
    group decision to report (e.g. test fixtures, human corrections)."""
    _ensure_article_row(conn, article_id)
    conn.execute(
        conn.dialect.upsert("article_category", _CATEGORY_COLS, conflict=("article_id",)),
        (
            article_id,
            label,
            score,
            group_label,
            group_score,
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
        conn.dialect.insert("article_entities", _ENTITY_COLS),
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


def fetch_pending_company_summaries(conn: NewsNlpDatabase, limit: int | None = None) -> list[Row]:
    """Return raw fields for articles ready for c_summary generation: a
    successful fetch (http_status_code=200), a computed sentiment, at least
    one qualifying entity (score>0.8, non-numeric), and no article_summary
    row yet. Adapts query.sql's `source_text` CTE -- the two INNER JOINs mean
    an article with sentiment but zero qualifying entities is never selected
    here, same as the original query.

    Returns raw columns rather than the assembled template text: SQL string
    literals don't interpret \\n as an escape (unlike Python), so template
    assembly happens in build_company_summary_input() instead.
    """
    # S608: _articles_rel(conn) is only ever "main" / "source"; the
    # GROUP_CONCAT / bare-digit fragments come from conn.dialect; every value
    # is bound as a parameter. The result-table joins stay in `main`.
    sql = f"""
        WITH entities AS (
            SELECT article_id, {conn.dialect.group_concat("text", ", ")} AS entities
            FROM article_entities
            WHERE score > 0.8 AND {conn.dialect.excludes_bare_digit("text")}
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


def build_company_summary_input(row: Row) -> str:
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
        conn.dialect.upsert("article_summary", _SUMMARY_COLS, conflict=("article_id",)),
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
    # Constant fragments + bound `?` values only; a non-qmark engine would take
    # the marker from conn.dialect.placeholder here.
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
        """SELECT label, score, group_label, group_score, earnings_performance, mergers_acquisitions,
                  leadership_governance, legal_regulatory, product_innovation, capital_shareholder_returns,
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
    # S608: `group_expr` only ever comes from `_group_exprs` -- a fixed column
    # name or a dialect-built date expression, keyed by the already-validated
    # `group_by`, never `group_by`'s raw text.
    group_expr = _group_exprs(conn.dialect).get(group_by) if group_by else None
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


def latest_eval_runs(conn: NewsNlpDatabase) -> list[dict]:
    """The most recent ``eval_run`` row per stage (newest first), with
    ``metrics_json`` decoded into a ``metrics`` object. Backs ``GET /eval/latest``.
    Lives here (not in ``news_nlp.eval``) so the API path never imports the
    ``eval`` dependency group (``strands`` / ``mlflow``)."""
    rows = conn.execute(
        """
        SELECT r.stage, r.started_at, r.finished_at, r.status, r.sample_size,
               r.judge_model, r.code_version, r.mlflow_run_id, r.metrics_json
        FROM eval_run r
        JOIN (
            SELECT stage, MAX(started_at) AS mx FROM eval_run GROUP BY stage
        ) latest ON latest.stage = r.stage AND latest.mx = r.started_at
        ORDER BY r.started_at DESC
        """
    ).fetchall()
    out: list[dict] = []
    for row in rows:
        d = dict(row)
        d["metrics"] = json.loads(d.pop("metrics_json") or "{}")
        out.append(d)
    return out
