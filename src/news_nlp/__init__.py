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

from news_nlp.corrections import (
    delete_category,
    delete_entities_for_article,
    delete_entity,
    delete_sentiment,
    update_category,
    update_entity,
    update_sentiment,
)
from news_nlp.db import (
    NewsNlpDatabase,
    attach_source,
    connect,
    connect_pipeline,
    detach_source,
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
    fetch_processed_articles,
    get_article_detail,
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
from news_nlp.sector_summary import (
    build_sector_facts,
    build_sector_intro_seed,
    clean_generated_text,
    compose_sector_summary,
    fetch_company_summaries_for_sector_week,
    fetch_pending_sector_weeks,
    fetch_sector_week_entity_stats,
    list_sector_summaries,
    write_sector_summary,
)
from news_nlp.taxonomy import (
    CATEGORY_LABELS,
    CATEGORY_SLUGS,
    OTHER_LABEL,
)

__all__ = [
    "CATEGORY_LABELS",
    "CATEGORY_SLUGS",
    "OTHER_LABEL",
    "RESULTS_DB_ENV_VAR",
    "SCHEMA",
    "SECTOR_SUMMARY_FORMAT_VERSION",
    "SOURCE_DB_ENV_VAR",
    "NewsNlpDatabase",
    "attach_source",
    "build_company_summary_input",
    "build_sector_facts",
    "build_sector_intro_seed",
    "category_stats",
    "clean_generated_text",
    "compose_sector_summary",
    "connect",
    "connect_pipeline",
    "delete_category",
    "delete_entities_for_article",
    "delete_entity",
    "delete_sentiment",
    "detach_source",
    "entity_stats",
    "fetch_company_summaries_for_sector_week",
    "fetch_pending_articles",
    "fetch_pending_category_articles",
    "fetch_pending_company_summaries",
    "fetch_pending_sector_weeks",
    "fetch_processed_articles",
    "fetch_sector_week_entity_stats",
    "get_article_detail",
    "init_schema",
    "list_articles",
    "list_sector_summaries",
    "now_iso",
    "require_source_text",
    "results_db_path",
    "sentiment_stats",
    "source_db_path",
    "update_category",
    "update_entity",
    "update_sentiment",
    "write_category",
    "write_company_summary",
    "write_entities",
    "write_sector_summary",
    "write_sentiment",
]
