#!/usr/bin/env python
"""CLI entrypoint of the stepped pipeline: one step at a time, never the whole chain (D39 (vii), D40; T-162, T-163).

Usage:
    uv run cli/news_nlp_steps.py --step scope --limit 500     # classify up to 500 unique bodies not yet done
    uv run cli/news_nlp_steps.py --step scope --status        # counts by state and by scope, no model loaded
    uv run cli/news_nlp_steps.py --step scope --fetch --status   # first: download the pinned model into models/scope_final
    uv run cli/news_nlp_steps.py --step scope --status --titles   # also the bodies with no title (scans the corpus)
    uv run cli/news_nlp_steps.py --step cut --limit 500       # cut up to 500 multi bodies into sub-news (step 1.2)
    uv run cli/news_nlp_steps.py --step cut --status          # sub-news written and bodies by state
    uv run cli/news_nlp_steps.py --step summary --limit 200   # step 2: summarise single bodies and sub-news over 2,000 characters
    uv run cli/news_nlp_steps.py --step summary --items sub_news --cut-revision units5-...   # only the sub-news of one cut revision
    uv run cli/news_nlp_steps.py --step summary --status      # summaries and originals stored, items by state

A rerun adds nothing: the bodies already done at the model's revision are skipped, failed ones are retried. A new model
revision starts a new set of rows and leaves the old ones. ``--model-dir`` is the folder written by
``scripts/train_scope_final_2026_11_11.py`` (default ``models/scope_final``). The ``cut`` step reads the ``multi`` bodies
step 1 wrote (``--scope-revision``, default the latest) and writes ``sub_news`` with the ``units`` extractor; its model is the
attribution checkpoint (default ``data/models/attribution_2026_11_08``). The ``summary`` step (step 2) reads the ``single`` bodies of step 1
and the sub-news of the cut (``--cut-revision``, default the latest); an item over 2,000 characters is summarised by the pinned local
summariser, a shorter one is stored as it is (``news_summary.text_source`` = ``original``); see ``src/news_summary_stage.py``.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

import news_nlp as db
from news_nlp import steps
from news_nlp.db import exit_on_schema_error

STEPS = (steps.STEP_SCOPE, steps.STEP_CUT, steps.STEP_SUMMARY)
BATCH = 16
SUMMARY_BATCH = 8  # texts summarised together; the model's own batch (4) is applied inside
MAX_FAILED_BATCHES = 3  # consecutive batches in which every body failed: the cause is not one body
MAX_FAILED_BODIES = 10  # the same for the cut, which goes body by body
SCOPE_ONLY_FLAGS = ("fetch", "allow_unpinned", "titles")


class RunAborted(RuntimeError):
    """The run stopped because batch after batch failed completely (out of memory, a lost GPU, a corrupt weight file)."""


class Classifier(Protocol):
    """What ``run_scope`` needs of a scope classifier (``scope_stage.ScopeClassifier`` or a test stub)."""

    revision: str
    chars: int

    def classify(self, items: Sequence[tuple[str, str]]) -> list[Any]: ...


def _classify_batch(
    conn: db.NewsNlpDatabase,
    classifier: Classifier,
    batch: list[steps.Body],
    counts: dict[str, int],
) -> None:
    """Classify and store one batch; a batch that fails is redone body by body, so one bad body fails alone."""
    try:
        results = classifier.classify([(b.title, b.text) for b in batch])
    except Exception as exc:
        if len(batch) > 1:
            for body in batch:
                _classify_batch(conn, classifier, [body], counts)
            return
        # a single body: the reason is stored with it
        steps.record_failure(
            conn, batch[0], steps.STEP_SCOPE, classifier.revision, steps.error_text(exc)
        )
        counts["failed"] += 1
        return
    for body, result in zip(batch, results, strict=True):
        steps.record_scope(
            conn,
            body,
            classifier.revision,
            result.scope,
            result.scores,
            f"title+{classifier.chars}",
        )
        counts["done"] += 1
        counts[result.scope] = counts.get(result.scope, 0) + 1


def run_scope(
    conn: db.NewsNlpDatabase, classifier: Classifier, limit: int | None
) -> dict[str, int]:
    """Classify up to ``limit`` unique bodies not yet done at the classifier's revision; return what was done."""
    todo = list(steps.pending_bodies(conn, steps.STEP_SCOPE, classifier.revision, limit))
    counts: dict[str, int] = {"done": 0, "failed": 0}
    failed_batches = 0
    with tqdm(total=len(todo), desc="scope", unit="body") as bar:  # progress and time left
        for i in range(0, len(todo), BATCH):
            batch = todo[i : i + BATCH]
            done_before = counts["done"]
            _classify_batch(conn, classifier, batch, counts)
            bar.update(len(batch))
            failed_batches = failed_batches + 1 if counts["done"] == done_before else 0
            if failed_batches >= MAX_FAILED_BATCHES:
                raise RunAborted(
                    f"{failed_batches} batches in a row failed completely after {counts['done']} done "
                    f"(see the error of a failed body with --status); the failed bodies are retried by the next run"
                )
    return counts


class Cutter(Protocol):
    """What ``run_cut`` needs of a cut extractor (``cut_stage.UnitsExtractor`` or a test stub)."""

    revision: str

    def cut(self, body: str) -> list[str]: ...


def run_cut(
    conn: db.NewsNlpDatabase, extractor: Cutter, scope_revision: str, limit: int | None
) -> dict[str, int]:
    """Cut up to ``limit`` multi bodies not yet done at the extractor's revision; return what was done.

    A body that fails is stored ``failed`` with its reason and the run goes on; ``MAX_FAILED_BODIES`` failures in a row
    stop it (the cause is then not one body: a broken checkpoint, a lost GPU), as step 1 does with its batches.
    """
    todo = list(steps.pending_multi_bodies(conn, scope_revision, extractor.revision, limit))
    counts: dict[str, int] = {"done": 0, "no_company_text": 0, "failed": 0, "sub_news": 0}
    in_a_row = 0
    with tqdm(total=len(todo), desc="cut", unit="body") as bar:
        for body in todo:
            try:
                texts = extractor.cut(body.text)
            except Exception as exc:
                steps.record_failure(
                    conn, body, steps.STEP_CUT, extractor.revision, steps.error_text(exc)
                )
                counts["failed"] += 1
                in_a_row += 1
                if in_a_row >= MAX_FAILED_BODIES:
                    raise RunAborted(
                        f"{in_a_row} bodies in a row failed after {counts['done'] + counts['no_company_text']} done "
                        f"(see the error with --status); the failed bodies are retried by the next run"
                    ) from exc
            else:
                in_a_row = 0
                steps.record_cut(conn, body, extractor.revision, texts)
                counts["done" if texts else "no_company_text"] += 1
                counts["sub_news"] += len(texts)
            bar.update(1)
    return counts


class Summarizing(Protocol):
    """What ``run_summary`` needs of a summariser (``news_summary_stage.Summarizer`` or a test stub)."""

    revision: str
    min_chars: int

    def summarize(self, texts: Sequence[str]) -> list[tuple[str, int]]: ...


class SummaryItem(NamedTuple):
    """One thing the summary step reads: a body (its hash) or a sub-news (its id), and the text the summariser would read."""

    kind: str
    item_id: str
    text: str


def _summary_items(
    conn: db.NewsNlpDatabase,
    revision: str,
    scope_revision: str | None,
    cut_revision: str | None,
    items: Sequence[str],
    limit: int | None,
) -> list[SummaryItem]:
    """The single bodies, then the sub-news, still to do at ``revision``; at most ``limit`` items in all."""
    from news_summary_stage import input_of_body  # noqa: PLC0415 -- loads torch

    todo: list[SummaryItem] = []
    if "single" in items and scope_revision is not None:
        for body in steps.pending_single_bodies(conn, scope_revision, revision, limit):
            todo.append(
                SummaryItem(steps.ITEM_BODY, body.body_hash, input_of_body(body.title, body.text))
            )
    if "sub_news" in items and cut_revision is not None:
        left = None if limit is None else max(limit - len(todo), 0)
        if left != 0:
            for sub in steps.pending_sub_news(conn, cut_revision, revision, left):
                todo.append(SummaryItem(steps.ITEM_SUB_NEWS, str(sub.sub_news_id), sub.text))
    return todo


def _summarize_items(
    conn: db.NewsNlpDatabase,
    summarizer: Summarizing,
    batch: list[SummaryItem],
    counts: dict[str, int],
) -> None:
    """Summarise and store one batch; a batch that fails is redone item by item, so one bad item fails alone."""
    try:
        results = summarizer.summarize([item.text for item in batch])
    except Exception as exc:
        if len(batch) > 1:
            for item in batch:
                _summarize_items(conn, summarizer, [item], counts)
            return
        steps.record_item_failure(
            conn,
            batch[0].kind,
            batch[0].item_id,
            steps.STEP_SUMMARY,
            summarizer.revision,
            steps.error_text(exc),
        )
        counts["failed"] += 1
        return
    for item, (summary, chunks) in zip(batch, results, strict=True):
        if not summary.strip():  # an empty text would be read by the next steps as the item's text
            steps.record_item_failure(
                conn,
                item.kind,
                item.item_id,
                steps.STEP_SUMMARY,
                summarizer.revision,
                "ValueError: empty summary",
            )
            counts["failed"] += 1
            continue
        steps.record_summary(
            conn, item.kind, item.item_id, summarizer.revision, summary, "summary", chunks
        )
        counts["summary"] += 1


def run_summary(
    conn: db.NewsNlpDatabase,
    summarizer: Summarizing,
    scope_revision: str | None,
    cut_revision: str | None,
    items: Sequence[str],
    limit: int | None,
) -> dict[str, int]:
    """Step 2: summarise up to ``limit`` items not yet done at the summariser's revision; return what was done.

    An item over ``summarizer.min_chars`` goes to the model in batches; a shorter one is stored as it is
    (``text_source = original``, no chunks). A batch that fails is redone item by item, an item that fails is stored
    ``failed`` with its reason and retried by the next run, and ``MAX_FAILED_BATCHES`` batches in a row that fail
    completely stop the run (the cause is then not one text: out of memory, a lost GPU).
    """
    from news_summary_stage import needs_summary  # noqa: PLC0415 -- loads torch

    todo = _summary_items(conn, summarizer.revision, scope_revision, cut_revision, items, limit)
    counts: dict[str, int] = {"summary": 0, "original": 0, "failed": 0}
    failed_batches = 0
    waiting: list[SummaryItem] = []

    def flush() -> None:
        nonlocal failed_batches
        if not waiting:
            return
        done_before = counts["summary"]
        _summarize_items(conn, summarizer, list(waiting), counts)
        failed_batches = failed_batches + 1 if counts["summary"] == done_before else 0
        waiting.clear()
        if failed_batches >= MAX_FAILED_BATCHES:
            raise RunAborted(
                f"{failed_batches} batches in a row failed completely after {counts['summary'] + counts['original']} done "
                f"(see the error of a failed item with --status); the failed items are retried by the next run"
            )

    with tqdm(total=len(todo), desc="summary", unit="item") as bar:
        for item in todo:
            if needs_summary(item.text, summarizer.min_chars):
                waiting.append(item)
                if len(waiting) >= SUMMARY_BATCH:
                    flush()
            else:
                steps.record_summary(
                    conn, item.kind, item.item_id, summarizer.revision, item.text, "original", 0
                )
                counts["original"] += 1
            bar.update(1)
        flush()
    return counts


def default_model_dir(step: str) -> Path:
    """The model folder a step uses when ``--model-dir`` is not given (both are in the repository)."""
    if step == steps.STEP_CUT:
        return ROOT / "data" / "models" / "attribution_2026_11_08"
    return ROOT / "models" / "scope_final"


def main_cut(args: argparse.Namespace, conn: db.NewsNlpDatabase) -> None:
    if args.status:
        # the revision depends on the model and the universe, so it is not known without loading the model: the latest
        # one that has state is shown, and the others are listed
        latest = steps.latest_revision(conn, steps.STEP_CUT)
        report = steps.status(conn, steps.STEP_CUT, latest)
        report["revisions"] = steps.revisions(conn, steps.STEP_CUT)
        print(json.dumps(report, indent=2))
        return
    scope_revision = args.scope_revision or steps.latest_classifier_revision(conn)
    if scope_revision is None:
        sys.exit("body_scope is empty: run --step scope first")
    if not conn.table_columns("articles", schema="main"):
        # sub_news references articles(id); the lean articles table is the crawler's, init_schema does not create it
        sys.exit(
            "the RESULTS store has no articles table: sub_news cannot reference an article (see schema.py)"
        )
    from cut_stage import (  # noqa: PLC0415 -- loads torch
        DEFAULT_READING,
        UnitsExtractor,
        companies_of,
    )

    if not args.model_dir.is_dir():
        sys.exit(
            f"{args.model_dir} is not a folder: the attribution checkpoint is local (T-148b12), copy it there or pass --model-dir"
        )
    try:
        extractor = UnitsExtractor.load(
            args.model_dir, companies_of(conn), args.reading or DEFAULT_READING
        )
    except ValueError as exc:  # the folder lacks a model file: say so, not a stack
        sys.exit(str(exc))
    try:
        counts = run_cut(conn, extractor, scope_revision, args.limit)
    except RunAborted as exc:
        print(json.dumps(steps.status(conn, steps.STEP_CUT, extractor.revision), indent=2))
        sys.exit(f"stopped: {exc}")
    print(json.dumps({"scope_revision": scope_revision, **counts}, indent=2))
    print(json.dumps(steps.status(conn, steps.STEP_CUT, extractor.revision), indent=2))


def main_summary(args: argparse.Namespace, conn: db.NewsNlpDatabase) -> None:
    if args.status:
        latest = steps.latest_revision(conn, steps.STEP_SUMMARY)
        report = steps.status(conn, steps.STEP_SUMMARY, latest)
        report["revisions"] = steps.revisions(conn, steps.STEP_SUMMARY)
        print(json.dumps(report, indent=2))
        return
    items = (args.items,) if args.items else ("single", "sub_news")
    scope_revision = args.scope_revision or steps.latest_classifier_revision(conn)
    cut_revision = args.cut_revision
    if "single" in items and scope_revision is None:
        sys.exit("body_scope is empty: run --step scope first")
    if "sub_news" in items:
        # a reader of sub_news names the cut it reads, never "the latest" (docs/scope_training.md 8.1)
        ran = ", ".join(steps.revisions(conn, steps.STEP_CUT)) or "none"
        if cut_revision is None:
            sys.exit(
                f"sub-news need the cut they come from: pass --cut-revision (the cuts that ran: {ran}), "
                "or --items single"
            )
        found = conn.execute(
            "SELECT 1 FROM sub_news WHERE extractor_revision = ? LIMIT 1", (cut_revision,)
        ).fetchone()
        if found is None:
            sys.exit(
                f"the cut wrote no sub-news at revision {cut_revision} (the cuts that ran: {ran})"
            )
    from news_summary_stage import Summarizer  # noqa: PLC0415 -- loads torch

    summarizer = Summarizer()
    print(
        f"summariser {summarizer.model_name}@{summarizer.model_revision[:8]} on {summarizer.device}, size {summarizer.min_chars}",
        file=sys.stderr,
    )
    try:
        counts = run_summary(conn, summarizer, scope_revision, cut_revision, items, args.limit)
    except RunAborted as exc:
        print(json.dumps(steps.status(conn, steps.STEP_SUMMARY, summarizer.revision), indent=2))
        sys.exit(f"stopped: {exc}")
    print(
        json.dumps(
            {"scope_revision": scope_revision, "cut_revision": cut_revision, **counts}, indent=2
        )
    )
    print(json.dumps(steps.status(conn, steps.STEP_SUMMARY, summarizer.revision), indent=2))


def main_scope(args: argparse.Namespace, conn: db.NewsNlpDatabase) -> None:
    revision = None
    if (args.model_dir / "training_record.json").exists():
        from scope_stage import revision_of  # noqa: PLC0415 -- no torch import here

        revision = revision_of(args.model_dir)
    if args.status:
        report = steps.status(conn, args.step, revision)
        if args.titles:
            report["missing_titles"] = steps.missing_titles(conn)
        print(json.dumps(report, indent=2))
        return
    if revision is None:
        sys.exit(f"{args.model_dir}/training_record.json is missing: train the models first")
    from scope_stage import ScopeClassifier, is_pinned, pin_status  # noqa: PLC0415 -- loads torch

    print(pin_status(revision), file=sys.stderr)
    if is_pinned(revision) is False and not args.allow_unpinned:
        sys.exit(
            f"refused: {revision} is not the pinned model (config/scope_model.json). "
            "Use --fetch into an empty --model-dir, or --allow-unpinned for a development copy"
        )

    try:
        counts = run_scope(conn, ScopeClassifier(args.model_dir), args.limit)
    except RunAborted as exc:
        print(json.dumps(steps.status(conn, args.step, revision), indent=2))
        sys.exit(f"stopped: {exc}")
    print(json.dumps(counts, indent=2))
    print(json.dumps(steps.status(conn, args.step, revision), indent=2))


#: Step 2 reads ``sub_news`` and keys its rows by ``sub_news.id``: before a revision of the cut is cut again, its summaries go first.
RECUT_SQL = (
    "DELETE FROM news_summary WHERE item_kind = 'sub_news' AND item_id IN "
    "(SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<revision>');",
    "DELETE FROM pipeline_step_state WHERE step = 'summary' AND item_kind = 'sub_news' AND item_id IN "
    "(SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<revision>');",
)
#: To summarise again at one revision of the summary step (the state is keyed by revision, the rows are written with INSERT OR IGNORE).
RESUMMARIZE_SQL = (
    "DELETE FROM pipeline_step_state WHERE step = 'summary' AND model_revision = '<revision>';",
    "DELETE FROM news_summary WHERE model_revision = '<revision>';",
)
_RERUN_TEMPLATE = """\
To run a step again on bodies it already did: a finished body is found in `pipeline_step_state`, not in the result table,
so `DELETE FROM sub_news` alone does not make the cut run again. At the same revision (the code, the model and the reading
are the same), in the RESULTS database:
  DELETE FROM pipeline_step_state WHERE step = 'cut' AND model_revision = '<revision>';
  DELETE FROM sub_news WHERE extractor_revision = '<revision>';   -- the rows are written with INSERT OR IGNORE
Before the `DELETE FROM sub_news` above, delete that revision's summaries: step 2 reads `sub_news` and keys its rows by `sub_news.id`, so new ids would leave them pointing at nothing:
  {recut}
To summarise again at one revision of the summary step (`--status` prints it):
  {resummarize}
A new revision (another reading, model or code) needs neither: it starts new rows and leaves the old ones.
`--status` prints the revisions (the latest first). The recipe: docs/scope_training.md, section 8.1.
"""
RERUN_HELP = _RERUN_TEMPLATE.format(
    recut="\n  ".join(RECUT_SQL), resummarize="\n  ".join(RESUMMARIZE_SQL)
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=RERUN_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--step", choices=STEPS, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="touch at most this many items (bodies; for summary also sub-news)",
    )
    parser.add_argument("--status", action="store_true", help="print the counts and exit")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download the pinned model (config/scope_model.json, public repository) into an empty --model-dir first",
    )
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="run on a model whose revision is not the pinned one (a refit, a development copy); the rows are not traceable to the Hub",
    )
    parser.add_argument(
        "--titles",
        action="store_true",
        help="with --status: also count the unique bodies with no title (reads the whole corpus: minutes)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="scope: models/scope_final; cut: data/models/attribution_2026_11_08 (both in the repository)",
    )
    parser.add_argument(
        "--scope-revision",
        default=None,
        help="cut and summary: the classifier revision whose multi bodies are cut / whose single bodies are summarised (default the latest)",
    )
    parser.add_argument(
        "--reading",
        choices=("units", "units4", "units5"),
        default=None,
        help="cut: how the body is read (default `units4`, T-176; `units` is the cut as it was before; `units5`, T-177, is a candidate)",
    )
    parser.add_argument(
        "--items",
        choices=("single", "sub_news"),
        default=None,
        help="summary: only the single bodies or only the sub-news (default both)",
    )
    parser.add_argument(
        "--cut-revision",
        default=None,
        help="summary: the cut revision whose sub-news are read; required with sub-news (the revisions the cut has are listed), never the latest",
    )
    parser.add_argument("--source-db", default=None)
    parser.add_argument("--results-db", default=None)
    return parser


def check_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse an option the chosen step does not read, so it is never silently ignored."""
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.step != steps.STEP_SCOPE:
        used = [f"--{n.replace('_', '-')}" for n in SCOPE_ONLY_FLAGS if getattr(args, n)]
        if used:
            parser.error(f"{', '.join(used)} only apply to --step scope")
    if args.step != steps.STEP_CUT and args.reading is not None:
        parser.error("--reading only applies to --step cut")
    if args.step != steps.STEP_SUMMARY and (
        args.items is not None or args.cut_revision is not None
    ):
        parser.error("--items and --cut-revision only apply to --step summary")
    if args.step == steps.STEP_SUMMARY and args.model_dir is not None:
        parser.error(
            "--model-dir does not apply to --step summary (its model is the pin of config/pipeline_models.json)"
        )
    if args.items == "single" and args.cut_revision is not None:
        parser.error("--cut-revision does not apply to --items single")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    check_options(parser, args)
    args.model_dir = args.model_dir or default_model_dir(args.step)
    db.env.load_env_file()
    if args.fetch:
        from scope_stage import fetch_pinned  # noqa: PLC0415 -- the download only

        print(f"fetched the pinned model into {args.model_dir}: {fetch_pinned(args.model_dir)}")
    with exit_on_schema_error():
        conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    db.init_schema(conn)
    if args.step == steps.STEP_CUT:
        main_cut(args, conn)
        return
    if args.step == steps.STEP_SUMMARY:
        main_summary(args, conn)
        return
    main_scope(args, conn)


if __name__ == "__main__":
    main()
