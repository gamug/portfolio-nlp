"""The `sector_summary` stage's orchestration entrypoint (PLAN.md Work item
10 / TASKS.md T-087, SPEC.md FR-012): ties this package's own
`composition`/`queries` modules together into `run_sector_summary_stage`.

Fully decoupled from the FTI hierarchy (`src/fti.py`) -- nothing here
imports from it, and nothing here has a Feature/Train/Inference shape to
share with the four ML stages, since `sector_summary` never loads a model
or touches the GPU (see `SECTOR_INTRO_METHOD`'s own comment below for the
2026-09-14 history behind that). `pipeline.run_pipeline` still calls
`run_sector_summary_stage` the same way it calls the four FTI-based
stages -- only where the function lives has changed.
"""

from collections.abc import Callable

from tqdm import tqdm

from news_nlp.db import NewsNlpDatabase
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
    write_sector_summary,
)

# (stage_name, processed_count, total_count) -> None -- same shape as
# pipeline.py's own ProgressCallback, duplicated here rather than imported
# to avoid a circular import (pipeline.py imports this module).
ProgressCallback = Callable[[str, int, int], None]

# sector_summary's intro_text used to run its aggregate-stats seed
# (build_sector_intro_seed) through SUMMARY_MODEL via
# hierarchical_summarize_batch, the same as c_summary. Found 2026-09-14
# (docs/evaluation.md's follow-up) to hallucinate on ~42-50% of rows --
# a fabricated source attribution ("...according to CNN.com's weekly
# Newsquiz") or a self-contradicting repeated percentage -- because a
# news-article summarizer was being asked to paraphrase a synthetic,
# templated stats sentence it was never trained on. Confirmed the same
# day with fresh, fully-corrected sentiment data: the pattern is
# independent of the underlying numbers, so re-running couldn't have
# fixed it. build_sector_intro_seed's own output is already a complete,
# fully-grounded sentence (see its docstring) -- so as of the same fix,
# intro_text IS that seed, verbatim (through clean_generated_text for
# whitespace normalization only), never run through a model. Zero
# hallucination risk by construction, not by mitigation -- the same
# "structural guarantee over probabilistic mitigation" principle this
# stage's cross-company-blending design already used. This also means
# run_sector_summary_stage never loads a model or touches the GPU at all.
SECTOR_INTRO_METHOD = "deterministic-template"


def run_sector_summary_stage(
    conn: NewsNlpDatabase, limit: int | None = None, on_progress: ProgressCallback | None = None
) -> None:
    """No model load, no GPU -- intro_text is now build_sector_intro_seed's
    own deterministic output, not a model paraphrase of it. See
    SECTOR_INTRO_METHOD's comment above for why."""
    groups = fetch_pending_sector_weeks(conn, limit=limit)
    total = len(groups)
    print(f"\n=== Sector summary stage ({SECTOR_INTRO_METHOD}) ===")
    print(f"{total} sector/week group(s) pending sector_summary")
    if on_progress:
        on_progress("sector_summary", 0, total)
    if total == 0:
        return

    idx = 0
    with tqdm(total=total, desc="sector_summary") as pbar:
        for group in groups:
            # A group with nothing to summarize (all its articles excluded,
            # see fetch_company_summaries_for_sector_week) is skipped, same
            # as the old per-group `if rows:` guard.
            group_rows = fetch_company_summaries_for_sector_week(
                conn, group["gics_sector"], group["gics_sub_industry"], group["week_start"]
            )
            if group_rows:
                entity_stats = fetch_sector_week_entity_stats(
                    conn, group["gics_sector"], group["gics_sub_industry"], group["week_start"]
                )
                intro_text = clean_generated_text(
                    build_sector_intro_seed(
                        group["gics_sector"],
                        group["gics_sub_industry"],
                        group["week_start"],
                        group["week_end"],
                        group_rows,
                    )
                )
                summary_text = compose_sector_summary(
                    group["gics_sector"],
                    group["gics_sub_industry"],
                    group["week_start"],
                    group["week_end"],
                    intro_text,
                    group_rows,
                    entity_stats,
                )
                facts = build_sector_facts(
                    group["gics_sector"],
                    group["gics_sub_industry"],
                    group["week_start"],
                    group["week_end"],
                    group_rows,
                    entity_stats,
                )
                write_sector_summary(
                    conn,
                    group["gics_sector"],
                    group["gics_sub_industry"],
                    group["week_start"],
                    group["week_end"],
                    summary_text,
                    num_articles=len(group_rows),
                    num_companies=len({r["company"] for r in group_rows}),
                    model_name=SECTOR_INTRO_METHOD,
                    facts=facts,
                    intro_text=intro_text,
                )
                conn.commit()
            idx += 1
            pbar.update(1)
            if on_progress:
                on_progress("sector_summary", idx, total)
