"""The news-NLP results-DB layer, owned by ``portfolio-nlp``.

Owns the RESULTS-store schema (five result tables), the two-tier SOURCE/RESULTS
connection machinery (read-only ``ATTACH`` of the crawl DB), the pipeline
read/write helpers, the deterministic ``sector_summary`` composition, the
read-only query helpers behind ``portfolio-nlp``'s FastAPI endpoints, the manual
result-row corrections, and the 10-category taxonomy.

``portfolio-nlp`` imports this package directly (``import news_nlp`` /
``from news_nlp import ...``). See ``docs/db-topology.md``.
"""

from __future__ import annotations

from news_nlp.db import (
    LATEST_KNOWN_SOURCE_SCHEMA_VERSION,
    MIN_SOURCE_SCHEMA_VERSION,
    SOURCE_REQUIRED_COLUMNS,
    NewsNlpDatabase,
    SourceSchemaError,
    attach_source,
    check_schema,
    check_source,
    connect,
    connect_pipeline,
    detach_source,
    exit_on_schema_error,
    require_source_text,
)
from news_nlp.env import (
    RESULTS_DB_ENV_VAR,
    SOURCE_DB_ENV_VAR,
    results_db_path,
    source_db_path,
)
from news_nlp.queries import (
    build_company_summary_input,
    category_stats,
    entity_stats,
    fetch_pending_articles,
    fetch_pending_category_articles,
    fetch_pending_company_summaries,
    fetch_pending_sentiment_articles,
    fetch_processed_articles,
    get_article_detail,
    latest_eval_runs,
    list_articles,
    now_iso,
    sentiment_stats,
    write_category,
    write_company_summary,
    write_entities,
    write_sentiment,
)
from news_nlp.schema import (
    SCHEMA,
    SECTOR_SUMMARY_FORMAT_VERSION,
    init_schema,
)
from news_nlp.taxonomy import (
    CATEGORY_LABELS,
    CATEGORY_SLUGS,
    OTHER_LABEL,
)

__all__ = [
    "CATEGORY_LABELS",
    "CATEGORY_SLUGS",
    "LATEST_KNOWN_SOURCE_SCHEMA_VERSION",
    "MIN_SOURCE_SCHEMA_VERSION",
    "OTHER_LABEL",
    "RESULTS_DB_ENV_VAR",
    "SCHEMA",
    "SECTOR_SUMMARY_FORMAT_VERSION",
    "SOURCE_DB_ENV_VAR",
    "SOURCE_REQUIRED_COLUMNS",
    "NewsNlpDatabase",
    "SourceSchemaError",
    "attach_source",
    "build_company_summary_input",
    "category_stats",
    "check_schema",
    "check_source",
    "connect",
    "connect_pipeline",
    "detach_source",
    "entity_stats",
    "exit_on_schema_error",
    "fetch_pending_articles",
    "fetch_pending_category_articles",
    "fetch_pending_company_summaries",
    "fetch_pending_sentiment_articles",
    "fetch_processed_articles",
    "get_article_detail",
    "init_schema",
    "latest_eval_runs",
    "list_articles",
    "now_iso",
    "require_source_text",
    "results_db_path",
    "sentiment_stats",
    "source_db_path",
    "write_category",
    "write_company_summary",
    "write_entities",
    "write_sentiment",
]
