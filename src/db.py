"""Back-compat facade. The NLP database layer moved to
``portfolio_common.news_nlp`` (see docs/db-topology.md). This module is kept
thin so ``import db`` / ``db.connect(...)`` call sites (``pipeline.py``,
``apps/news_nlp_api.py``, tests) keep working unchanged, and so ``.env`` is
loaded for every entrypoint that imports it.
"""

from pathlib import Path

from dotenv import load_dotenv

# Loaded here (not just in apps/news_nlp_api.py) so DATABASE_URL /
# SOURCE_DATABASE_URL are honored by every entrypoint that imports this module,
# including the standalone `python -m setup` / `python -m pipeline` paths that
# never go through the FastAPI app. Safe to call more than once.
load_dotenv()

# Explicit re-exports (not `import *`) so type-checkers see real signatures.
from portfolio_common.news_nlp import (  # noqa: E402, F401
    CATEGORY_LABELS,
    CATEGORY_SLUGS,
    OTHER_LABEL,
    SCHEMA,
    SECTOR_SUMMARY_FORMAT_VERSION,
    attach_source,
    build_company_summary_input,
    build_sector_facts,
    build_sector_intro_seed,
    category_stats,
    clean_generated_text,
    compose_sector_summary,
    connect,
    connect_pipeline,
    delete_category,
    delete_entities_for_article,
    delete_entity,
    delete_sentiment,
    detach_source,
    entity_stats,
    fetch_company_summaries_for_sector_week,
    fetch_pending_articles,
    fetch_pending_category_articles,
    fetch_pending_company_summaries,
    fetch_pending_sector_weeks,
    fetch_sector_week_entity_stats,
    get_article_detail,
    init_schema,
    list_articles,
    list_sector_summaries,
    now_iso,
    require_source_text,
    results_db_path,
    sentiment_stats,
    source_db_path,
    update_category,
    update_entity,
    update_sentiment,
    write_category,
    write_company_summary,
    write_entities,
    write_sector_summary,
    write_sentiment,
)
from portfolio_common.news_nlp.db import (  # noqa: E402, F401
    _articles_rel,
    _compute_lean_article_columns,
    _Connection,
    _ensure_article_row,
)
from portfolio_common.news_nlp.queries import _PENDING_ARTICLE_TABLES  # noqa: E402, F401
from portfolio_common.news_nlp.schema import _migrate_sector_summary_schema  # noqa: E402, F401
from portfolio_common.news_nlp.sector_summary import (  # noqa: E402, F401
    _CATEGORY_DISPLAY_NAMES,
    _CATEGORY_ORDER,
    _WEEK_END_EXPR,
    _WEEK_START_EXPR,
)


def __getattr__(name: str) -> Path:
    """`db.DB_PATH` / `db.SOURCE_DB_PATH` used to be module-level constants
    computed at import; they are now resolved per call from the environment.
    Kept as lazy module attributes so `apps/news_nlp_api.py`'s
    `db.connect(db.DB_PATH)` and any leftover reads still work."""
    if name == "DB_PATH":
        return results_db_path()
    if name == "SOURCE_DB_PATH":
        path = source_db_path()
        if path is None:
            raise AttributeError("SOURCE_DB_PATH is unset ($SOURCE_DATABASE_URL not configured)")
        return path
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
