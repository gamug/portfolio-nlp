"""Score a candidate model over a sample of SOURCE articles for `news_nlp.eval`
(PLAN.md Work item 10 step 3 / TASKS.md T-089, SPEC.md FR-013).

Today, `sampling._prediction` reads whatever a *prior* `pipeline.py` run
already wrote into `article_sentiment`/`article_category`/`article_entities`/
`article_summary` -- there is no way to evaluate a model that hasn't already
been run over the whole corpus, short of the destructive, manual
`scripts/resample_sentiment_v{3,4,5}_2026_09_15.py` scratch-DB-copy-and-
restore workaround (PLAN.md Work item 10's own stated motivation for this
task).

`candidate_scored_connection` fixes that by reusing the exact same FTI
`Inference` subclass `pipeline.py` uses for that stage (parameterized with a
different `model_name`/`revision`, its constructor's own documented
extension point) against a fresh, empty, throwaway RESULTS file -- never the
real one. `sample_for_stage` (unmodified) can then be pointed at the
returned connection instead of the production one, so "low confidence"
stratification is computed from the *candidate's own* freshly-written
scores.

`sector_summary` is deliberately absent from `_STAGE_CLASSES` -- it has no
Feature/Train/Inference shape (SPEC.md FR-012), so "candidate model" is
meaningless for it.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import news_nlp as db
from category_stage import CategoryFeature, CategoryInference
from fti import Feature, Inference
from ner_stage import NerFeature, NerInference
from news_nlp.db import NewsNlpDatabase
from sentiment_stage import SentimentFeature, SentimentInference
from summary_stage import SummaryFeature, SummaryInference

_STAGE_CLASSES: dict[str, tuple[type[Feature[Any, Any]], type[Inference[Any, Any]]]] = {
    "sentiment": (SentimentFeature, SentimentInference),
    "ner": (NerFeature, NerInference),
    "category": (CategoryFeature, CategoryInference),
    "c_summary": (SummaryFeature, SummaryInference),
}


@contextmanager
def candidate_scored_connection(
    source_db: str,
    stage: str,
    *,
    model_name: str,
    revision: str,
    limit: int,
    sample_seed: int | None,
) -> Iterator[NewsNlpDatabase]:
    """Score up to `limit` SOURCE articles for `stage` with (`model_name`,
    `revision`) -- NOT the pinned production model -- via that stage's own
    FTI `Inference` subclass, writing into a fresh, empty, throwaway RESULTS
    file (never the real one). Yields the open two-tier connection (SOURCE
    still attached) so a caller can read both the article text and the
    freshly-written candidate predictions from it; deletes the scratch file
    on exit regardless of success or failure.

    `revision` is required, not optional -- `Inference.revision`'s own
    fallback (`type(self).MODEL_REVISIONS[model_name]`) only knows each
    stage's *production* model, so an unrecognized `model_name` raises
    `KeyError` there instead of silently loading an unpinned checkpoint,
    which would violate this project's own pin-every-model convention
    (SPEC.md SS13 item 4). For a local checkpoint path, pass any
    placeholder (`from_pretrained` ignores `revision` for a local
    directory entirely) -- mirrors `scripts/resample_sentiment_v4_2026_09_15.py`'s
    own `pipeline.MODEL_REVISIONS[_V4_CHECKPOINT] = "local"` convention.

    `sample_seed` is silently ignored for `category`/`c_summary` -- neither
    stage's own `fetch_pending` override supports it today (a pre-existing
    limitation of those two stages' fetch helpers, documented on `fti.
    Inference.fetch_pending`, not something this function works around).
    """
    if stage not in _STAGE_CLASSES:
        raise ValueError(f"stage must be one of {sorted(_STAGE_CLASSES)}, got {stage!r}")
    if not revision:
        raise ValueError("revision is required (pass any placeholder for a local checkpoint path)")
    feature_cls, inference_cls = _STAGE_CLASSES[stage]

    fd, scratch_path = tempfile.mkstemp(suffix=".db", prefix="news_nlp_eval_candidate_")
    os.close(fd)
    try:
        conn = db.connect_pipeline(results_db=scratch_path, source_db=source_db)
        try:
            # `connect_pipeline` always opens with `foreign_keys=True` (the
            # right default for a real RESULTS file) -- but this scratch
            # file's cloned `articles` table (below) carries whatever FK
            # constraints the real SOURCE schema happens to declare (e.g.
            # portfolio-data-mining's own `articles` REFERENCES
            # `discovered_urls`), and this scratch file deliberately never
            # clones `discovered_urls` or any other SOURCE-only table --
            # only `articles` itself is needed here (see below). Disabled
            # before any write, not worked around per-statement, so every
            # insert into the cloned table (via `copy_row_lean`) doesn't
            # fail on a table this throwaway file was never going to have.
            # A no-op if called mid-transaction (SQLite's own restriction);
            # called immediately after connecting, before any statement
            # runs, so that never applies here.
            conn.execute("PRAGMA foreign_keys = OFF")
            # `init_schema` deliberately never creates `articles` -- a real
            # RESULTS file already has it (crawler-owned, migrated once
            # historically; see news_nlp/schema.py's own docstring). This
            # scratch file starts from nothing, so clone SOURCE's own
            # `articles` DDL verbatim (via sqlite_master.sql, not a `CREATE
            # TABLE AS SELECT`, which would drop the `id` PRIMARY KEY that
            # every result table's `REFERENCES articles(id)` foreign key
            # needs) before any result-table write needs it via
            # `copy_row_lean`'s `main.articles` INSERT OR IGNORE target --
            # `body_text` tags along harmlessly (copy_row_lean's own
            # `exclude=("body_text",)` already keeps it out of every real
            # copied row; nothing in this scratch file ever reads it back,
            # since article text is always read from `source.articles`).
            (source_articles_ddl,) = conn.execute(
                "SELECT sql FROM source.sqlite_master WHERE type = 'table' AND name = 'articles'"
            ).fetchone()
            conn.execute(source_articles_ddl)
            db.init_schema(conn)
            inference = inference_cls(feature_cls(), model_name=model_name, revision=revision)
            inference.run(conn, limit=limit, sample_seed=sample_seed)
            yield conn
        finally:
            db.detach_source(conn)
            conn.close()
    finally:
        if os.path.exists(scratch_path):
            os.remove(scratch_path)
