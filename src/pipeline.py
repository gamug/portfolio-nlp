"""One-shot batch pipeline: FinBERT sentiment + fine-tuned NER + zero-shot
category classification over `articles`.

Run manually whenever new articles need processing:
    .venv/Scripts/python.exe -m pipeline

Idempotent/resumable: only processes articles missing from the results
tables. Loads one model onto the GPU at a time (sentiment, then NER, then
category) to stay well within a 6GB VRAM budget, and frees each model before
loading the next. If a stage has nothing pending, it skips loading that
stage's model entirely.

Two-tier DB: run_pipeline reads article text from the read-only SOURCE store
($SOURCE_DATABASE_URL, required) and writes results to the RESULTS store
($DATABASE_URL). See docs/db-topology.md.
"""

import gc
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from dotenv import load_dotenv

import news_nlp as db
from category_stage import CategoryFeature, CategoryInference
from ner_stage import NerFeature, NerInference
from news_nlp.sector_summary import run_sector_summary_stage
from pipeline_config import load_pipeline_models_config
from sentiment_stage import SentimentFeature, SentimentInference
from summary_stage import SummaryFeature, SummaryInference

# Loaded here (every real entrypoint -- apps/news_nlp_api.py, cli/news_nlp_cli.py,
# `python -m pipeline`, src/setup.py -- imports this module) so DATABASE_URL /
# SOURCE_DATABASE_URL are honored wherever they're set via .env. Used to live in
# the since-deleted src/db.py facade; news_nlp.env reads only real
# env vars, so something in this repo has to load .env into them. Safe to call
# more than once.
load_dotenv()

# Production model selection (name/revision/batch size, one entry per ML
# stage) lives in the git-tracked config/pipeline_models.json, not here --
# schema/validation in pipeline_config.py (SPEC.md FR-018, PLAN.md Work
# item 14; supersedes the literal constants + MODEL_REVISIONS dict this
# block used to define directly, SPEC.md SS13 item 4). Deliberately loaded
# at import time, with no CLI/env override: per constitution AI-behavior
# #1, swapping a model is a spec-level change that goes through normal PR
# review either way, so there is nothing to gain from a runtime flag and
# real risk in adding one. This does mean an invalid or missing
# config/pipeline_models.json now fails on `import pipeline` itself --
# every real entrypoint (apps/news_nlp_api.py, cli/*.py, `python -m
# pipeline`, setup.py, this file's own test suite) imports this module, so
# a broken pin file fails loudly and immediately, everywhere, including in
# CI (pytest collection), rather than silently shipping and only surfacing
# the next time someone happens to run the real pipeline. Uncaught/
# unreworded on purpose -- pydantic's own ValidationError already names
# exactly what's wrong, matching cli/run_experiment.py's convention for
# the same JSON+pydantic pattern.
#
# Design history preserved here since JSON has no comment syntax:
# - Sentiment (`gamug/FinBERT-financial-news`): selected 2026-09-13 as the
#   production sentiment design, after real-data evaluation of four
#   candidates (base/fine-tuned FinBERT x chunk-level/title-only
#   aggregation) against the same 2,000-article pool + LLM judge -- full
#   comparison, diagnosis of the remaining precision gap, and the
#   rejected-fix evidence (confidence threshold, subject-coverage gate,
#   zero-shot materiality gate) in docs/evaluation.md's 2026-09-13
#   follow-ups; decision recorded in SPEC.md SS13 item 1 / PLAN.md Work
#   item 4. The entity-scoped chunk-weighting logic itself
#   (`_sentiment_chunk_weights`, `_text_mentions_subject`,
#   `_normalize_company_name`) lives in `sentiment_stage.py` (PLAN.md Work
#   item 10 / TASKS.md T-083). Current pin: v4 (class-weighted retrain,
#   PLAN.md Work item 9, TASKS.md T-073) -- published + pinned 2026-09-19,
#   adopted for its recall_negative gain (0.808->0.832); see
#   docs/evaluation.md's 2026-09-15 follow-ups and the model card at
#   https://huggingface.co/gamug/FinBERT-financial-news.
# - NER batch size: articles per forward pass, not chunks per forward
#   pass -- every chunk of every article in one ner.batch_size-sized group
#   of articles is flattened into a single padded tokenizer call (see
#   ner_stage.NerFeature), so the actual forward-pass batch dimension is
#   the *total chunk count* across those articles, not this constant
#   itself -- unlike category.batch_size (a fixed 9 pairs/article, so its
#   forward-pass width is exactly category.batch_size * 9 every time).
#   NER's per-article chunk count varies with article length
#   (docs/evaluation.md's 2026-09-12 follow-up found one 156,053-char
#   outlier alone worth dozens of chunks). Empirically tune against
#   SPEC.md NR-001's 6GB VRAM budget before trusting this number on a
#   full-corpus run (PLAN.md Work item 7 step 5 / TASKS.md T-062 -- needs
#   a real GPU, not yet re-measured past this starting value).
# - Category batch size: articles classified per forward pass, not just
#   labels-per-article -- each article already batches its own 9
#   (premise, hypothesis) pairs in one call, but 9 rows is too small a
#   batch to keep a GPU busy. Grouping category.batch_size articles'
#   pairs into one call (8 * 9 = 72 rows) gets real throughput out of the
#   GPU without materially raising peak VRAM -- still one model on the
#   card at a time, just a wider batch through it.
# - Summary batch size: generate() with beam search is far more
#   memory-intensive per row than a single classification forward pass
#   (category's forward-only batching), so this stays smaller despite the
#   same one-model-at-a-time VRAM budget -- tune down further if a 6GB
#   card OOMs.
# - Revision pins: fetched from the HF Hub API (GET /api/models/<repo_id>,
#   the "sha" field) at pin time, not guessed -- resolving by repo name
#   alone means an upstream push to any of these four repos changes
#   results silently, with no signal, undermining the SPEC.md SS9 accuracy
#   baseline every one of these numbers was measured against. Passed as
#   `revision=` at every from_pretrained call site below AND in
#   setup.py's download_models() -- pinning only the pre-download and
#   leaving from_pretrained(name) unpinned would not actually fix
#   anything, since HF's local cache resolution isn't guaranteed to serve
#   the pinned snapshot for an unpinned call. Bumping a pin is a
#   deliberate, reviewed, one-line diff against config/pipeline_models.json,
#   not silent drift.
_MODELS_CONFIG = load_pipeline_models_config()

SENTIMENT_MODEL = _MODELS_CONFIG.sentiment.model
NER_MODEL = _MODELS_CONFIG.ner.model
CATEGORY_MODEL = _MODELS_CONFIG.category.model
SUMMARY_MODEL = _MODELS_CONFIG.c_summary.model

# A real, mutable dict (not e.g. a frozen mapping): scripts/resample_
# sentiment_v{3,4,5}_2026_09_15.py mutate this in-process at runtime
# (pipeline.MODEL_REVISIONS[candidate] = "local") to score a candidate
# checkpoint without touching disk -- that pattern only works if this
# stays a plain dict, regardless of how it was populated.
MODEL_REVISIONS: dict[str, str] = {
    SENTIMENT_MODEL: _MODELS_CONFIG.sentiment.revision,
    NER_MODEL: _MODELS_CONFIG.ner.revision,
    CATEGORY_MODEL: _MODELS_CONFIG.category.revision,
    SUMMARY_MODEL: _MODELS_CONFIG.c_summary.revision,
}

NER_BATCH_SIZE = _MODELS_CONFIG.ner.batch_size

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (stage_name, processed_count, total_count) -> None
ProgressCallback = Callable[[str, int, int], None]


def _warn_if_cpu() -> None:
    """Print a hard-to-miss banner when DEVICE resolved to CPU. Each stage's
    own "=== ... on {DEVICE} ===" banner already says so, but that's easy to
    miss in the moment -- the usual symptom is just "the pipeline seems to be
    hanging", discovered hours into a run, not a line read at the top. Called
    once from run_pipeline() (not at import time) so importing this module
    without running it stays silent.

    A CPU fallback here isn't a driver/GPU problem -- torch.cuda.is_available()
    is False whenever the installed torch build has no CUDA support compiled
    in at all, which is what a plain `pip install torch` (or a transitive
    dependency pulling it in) gives you. The fix is reinstalling torch from
    the CUDA-specific index documented in requirements.txt, not anything
    driver-side.
    """
    if DEVICE.type == "cpu":
        print(
            "\n" + "!" * 78 + "\n! WARNING: CUDA is not available -- this run will use the CPU.\n"
            "! Sentiment/NER/category/summarization models are dramatically slower\n"
            "! on CPU. If this machine has an NVIDIA GPU, torch is very likely\n"
            "! installed as the plain CPU-only wheel instead of a CUDA build --\n"
            "! reinstall it with:\n"
            "!     .venv\\Scripts\\python.exe -m pip install torch "
            "--index-url https://download.pytorch.org/whl/cu124\n" + "!" * 78 + "\n"
        )


# See the design-history comment above _MODELS_CONFIG for the rationale
# behind each of these batch sizes.
CATEGORY_BATCH_SIZE = _MODELS_CONFIG.category.batch_size
SUMMARY_BATCH_SIZE = _MODELS_CONFIG.c_summary.batch_size


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sentiment_stage(
    conn: db.NewsNlpDatabase,
    limit: int | None = None,
    on_progress: ProgressCallback | None = None,
    *,
    sample_seed: int | None = None,
) -> None:
    """Entity-scoped, chunk-level aggregation (PLAN.md Work item 4 step 1,
    chosen 2026-09-12, revised to chunk granularity 2026-09-13; migrated
    onto the FTI hierarchy 2026-09-16, PLAN.md Work item 10 / TASKS.md
    T-083 -- see `sentiment_stage.SentimentFeature`/`SentimentInference`
    for the actual chunking/weighting/forward-pass logic, unchanged in
    behavior from before this migration).

    Kept here as a real, settable module-level function (not inlined into
    `run_pipeline`) rather than migrated away entirely, for two reasons:
    `tests/news_nlp/test_pipeline_run.py` monkeypatches
    `pipeline.run_sentiment_stage` itself to stub out the stage from
    `run_pipeline`, and `scripts/resample_sentiment_v{3,4,5}_2026_09_15.py`
    (PLAN.md Work item 9) monkeypatch `pipeline.SENTIMENT_MODEL`/
    `pipeline.MODEL_REVISIONS[...]` *before* calling this function to score
    a candidate model without touching this file on disk -- both module
    globals are read fresh here on every call (not at import time), so
    that pattern keeps working exactly as before this migration.

    `sample_seed` (with `limit` as the sample size): a reproducible random
    sample of pending articles instead of the normal backlog-order first
    `limit` -- see `db.fetch_pending_sentiment_articles`'s docstring. For a
    deliberate targeted reprocessing pass (mirrors NER's T-025 resample,
    `TASKS.md` T-034/T-035), not routine pipeline runs. **Always pass a
    seed for a reprocessing pass** -- an unseeded (backlog/id-order) run
    can badly skew the sample if some property of interest correlates with
    id order, as `TASKS.md` T-037 found the hard way (a data-quality bug
    concentrated in early-crawled ids made an unseeded resample 94%
    unrepresentative on that axis).
    """
    inference = SentimentInference(
        SentimentFeature(),
        model_name=SENTIMENT_MODEL,
        revision=MODEL_REVISIONS[SENTIMENT_MODEL],
    )
    inference.run(conn, limit, on_progress, sample_seed=sample_seed)


def run_ner_stage(
    conn: db.NewsNlpDatabase,
    limit: int | None = None,
    on_progress: ProgressCallback | None = None,
    *,
    sample_seed: int | None = None,
) -> None:
    """`sample_seed` (with `limit` as the sample size): a reproducible random
    sample of pending articles instead of the normal backlog-order first
    `limit` -- see `db.fetch_pending_articles`'s docstring. For a deliberate
    targeted reprocessing pass (docs/evaluation.md's 2026-09-12 NER
    follow-up), not routine pipeline runs.

    A thin wrapper around `ner_stage.NerInference` (PLAN.md Work item 10 /
    TASKS.md T-084) -- `merge_bio_predictions`/the cross-article batching
    logic now live there. Passes `NER_MODEL`/`MODEL_REVISIONS`/
    `NER_BATCH_SIZE` through explicitly, read fresh from this module's own
    globals on every call, so a resample script's or test's
    `pipeline.NER_MODEL = ...`/`pipeline.NER_BATCH_SIZE = ...` monkeypatch
    keeps working exactly as before. Batched `NER_BATCH_SIZE` articles at a
    time -- see that constant's comment for why its forward-pass width
    isn't fixed the way `CATEGORY_BATCH_SIZE`'s is."""
    inference = NerInference(
        NerFeature(),
        model_name=NER_MODEL,
        revision=MODEL_REVISIONS[NER_MODEL],
        batch_size=NER_BATCH_SIZE,
    )
    inference.run(conn, limit, on_progress, sample_seed=sample_seed)


def run_category_stage(
    conn: db.NewsNlpDatabase, limit: int | None = None, on_progress: ProgressCallback | None = None
) -> None:
    """Two-level hierarchical zero-shot classification (docs/category-taxonomy.md):
    level 1 picks (up to) the top-2 CATEGORY_GROUPS for each article via a
    3-way softmax; level 2 classifies only the survivors -- articles whose
    winning group cleared CATEGORY_GROUP_FLOOR -- against those top-2
    groups' 6 combined children. Replaces a single flat 9-way softmax, which
    empirically starved real signal for several labels by making them
    compete against 8 others in one softmax (see docs/category-taxonomy.md's
    "Hierarchical classification" section for the eval data that motivated
    this).

    A thin wrapper around `category_stage.CategoryInference` (PLAN.md Work
    item 10 / TASKS.md T-085) -- the level-1/level-2 classification logic
    now lives there. Passes `CATEGORY_MODEL`/`MODEL_REVISIONS`/
    `CATEGORY_BATCH_SIZE` through explicitly, read fresh from this module's
    own globals on every call, matching `run_sentiment_stage`/
    `run_ner_stage`'s own thin-wrapper shape. No `sample_seed` support --
    `db.fetch_pending_category_articles` doesn't take one, matching today's
    real signature exactly."""
    inference = CategoryInference(
        CategoryFeature(),
        model_name=CATEGORY_MODEL,
        revision=MODEL_REVISIONS[CATEGORY_MODEL],
        batch_size=CATEGORY_BATCH_SIZE,
    )
    inference.run(conn, limit, on_progress)


def run_company_summary_stage(
    conn: db.NewsNlpDatabase, limit: int | None = None, on_progress: ProgressCallback | None = None
) -> None:
    """A thin wrapper around `summary_stage.SummaryInference` (PLAN.md Work
    item 10 / TASKS.md T-086) -- the batched chunk-then-reduce summarization
    logic now lives there. Passes `SUMMARY_MODEL`/`MODEL_REVISIONS`/
    `SUMMARY_BATCH_SIZE` through explicitly, read fresh from this module's
    own globals on every call, matching every other stage's thin-wrapper
    shape. No `sample_seed` support -- `db.fetch_pending_company_summaries`
    doesn't take one, matching today's real signature exactly."""
    inference = SummaryInference(
        SummaryFeature(),
        model_name=SUMMARY_MODEL,
        revision=MODEL_REVISIONS[SUMMARY_MODEL],
        batch_size=SUMMARY_BATCH_SIZE,
    )
    inference.run(conn, limit, on_progress)


def run_pipeline(
    limit: int | None = None,
    summarize: bool = False,
    on_progress: ProgressCallback | None = None,
    results_db: Path | None = None,
    source_db: Path | None = None,
) -> None:
    """Run every stage against the RESULTS store, reading article text from the
    read-only SOURCE store (ATTACHed by db.connect_pipeline). `results_db` /
    `source_db` override $DATABASE_URL / $SOURCE_DATABASE_URL; SOURCE is
    required -- db.connect_pipeline raises if none is configured. See
    docs/db-topology.md.
    """
    _warn_if_cpu()
    conn = db.connect_pipeline(results_db=results_db, source_db=source_db)
    try:
        db.init_schema(conn)
        db.require_source_text(conn)  # fail fast, before any model loads
        run_sentiment_stage(conn, limit=limit, on_progress=on_progress)
        run_ner_stage(conn, limit=limit, on_progress=on_progress)
        run_category_stage(conn, limit=limit, on_progress=on_progress)
        if summarize:
            run_company_summary_stage(conn, limit=limit, on_progress=on_progress)
            run_sector_summary_stage(conn, limit=limit, on_progress=on_progress)
    finally:
        db.detach_source(conn)
        conn.close()
    print("\nPipeline run complete.")


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_pipeline(limit=limit)


if __name__ == "__main__":
    main()
