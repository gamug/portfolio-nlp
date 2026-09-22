#!/usr/bin/env python
"""CLI entrypoint: run the sentiment + NER + category (+ optional
summarization) batch pipeline directly, no FastAPI/uvicorn involved (for
that, see apps/news_nlp_api.py instead). Wraps pipeline.run_pipeline()
directly with real --limit/--summarize flags; see docs/modules/news-nlp.md.

Sentiment (FinBERT) -> NER (SEC-BERT) -> Category (zero-shot DeBERTa-v3
against a fixed 10-category taxonomy; see docs/category-taxonomy.md) always
run. Pass --summarize to also run c_summary (one summary per article) ->
sector_summary (one summary per gics_sub_industry per closed calendar week,
reduced from that week's c_summary rows) -- off by default, since
summarization loads its own model on top of the 6GB VRAM budget the other
three stages already use.

(src/pipeline.py also has its own bare `if __name__ == "__main__":`
usable via `python -m pipeline [limit]` — kept for backward
compatibility, but this is the documented entrypoint going forward.)

Usage:
    .venv\\Scripts\\python.exe cli\\news_nlp_cli.py --limit 50
    .venv\\Scripts\\python.exe cli\\news_nlp_cli.py   # process every pending article
    .venv\\Scripts\\python.exe cli\\news_nlp_cli.py --summarize   # also run c_summary/sector_summary
"""

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tqdm import tqdm

from pipeline import run_pipeline


class _CliProgress:
    """tqdm-backed `on_progress` handler -- restores the per-stage console
    progress bar sentiment/NER/category/company_summary each used to print
    directly before the FTI migration (PLAN.md Work item 10, TASKS.md
    T-083-T-086): every stage's own hand-written loop called
    `tqdm(rows, desc=<stage>)` unconditionally, but that migration replaced
    all four with the shared `fti.Inference.run()` template method, which
    only reports progress through the `on_progress` callback -- and this
    CLI never wired one up. An undisclosed regression, not a deliberate
    trade-off (see `docs/evaluation.md`/PR history for T-083-T-086: none
    mention dropping the printed bar).

    Skips `"sector_summary"`: that stage isn't on the FTI hierarchy and
    already prints its own `tqdm` bar directly inside
    `news_nlp/sector_summary/stage.py`, independent of `on_progress` --
    wiring a second bar here would double it, not restore it.

    One bar at a time, matching `run_pipeline`'s own strictly sequential
    stage order -- a new stage name closes the previous stage's bar before
    opening its own.
    """

    def __init__(self) -> None:
        self._stage: str | None = None
        self._bar: Any = None

    def __call__(self, stage: str, processed: int, total: int) -> None:
        if stage == "sector_summary":
            return
        if stage != self._stage:
            if self._bar is not None:
                self._bar.close()
            self._stage = stage
            self._bar = tqdm(total=total, desc=stage) if total else None
        if self._bar is None:
            return
        self._bar.n = processed
        self._bar.refresh()
        if processed >= total:
            self._bar.close()
            self._bar = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to process per stage (articles for sentiment/NER/category/c_summary; "
        "sector/week groups for sector_summary). Default: all pending.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Also run the c_summary/sector_summary stages. Off by default -- "
        "these stages load their own summarization model in addition to "
        "the sentiment/NER models.",
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=None,
        help="Read-only SOURCE database with articles.body_text. Overrides "
        "$SOURCE_DATABASE_URL (the usual mechanism). Required by the "
        "text-reading stages; see docs/db-topology.md.",
    )
    parser.add_argument(
        "--results-db",
        type=Path,
        default=None,
        help="RESULTS/serving database to write to. Overrides $DATABASE_URL "
        "(default: <repo>/data/nlp.db).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        limit=args.limit,
        summarize=args.summarize,
        source_db=args.source_db,
        results_db=args.results_db,
        on_progress=_CliProgress(),
    )


if __name__ == "__main__":
    main()
