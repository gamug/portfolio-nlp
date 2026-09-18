"""The ``sector_summary`` pipeline stage, split into a DB layer
(:mod:`news_nlp.sector_summary.queries`), pure composition logic
(:mod:`news_nlp.sector_summary.composition`), and the orchestration
entrypoint (:mod:`news_nlp.sector_summary.stage`, TASKS.md T-087) --
re-exported flat here so ``news_nlp.sector_summary.compose_sector_summary``
etc. keep working the same as before the split. See each submodule's own
docstring for its trace of capabilities.
"""

from __future__ import annotations

from news_nlp.sector_summary.composition import (
    build_sector_facts,
    build_sector_intro_seed,
    clean_generated_text,
    compose_sector_summary,
)
from news_nlp.sector_summary.queries import (
    fetch_company_summaries_for_sector_week,
    fetch_pending_sector_weeks,
    fetch_sector_week_entity_stats,
    list_sector_summaries,
    write_sector_summary,
)
from news_nlp.sector_summary.stage import SECTOR_INTRO_METHOD, run_sector_summary_stage

__all__ = [
    "SECTOR_INTRO_METHOD",
    "build_sector_facts",
    "build_sector_intro_seed",
    "clean_generated_text",
    "compose_sector_summary",
    "fetch_company_summaries_for_sector_week",
    "fetch_pending_sector_weeks",
    "fetch_sector_week_entity_stats",
    "list_sector_summaries",
    "run_sector_summary_stage",
    "write_sector_summary",
]
