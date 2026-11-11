#!/usr/bin/env python
"""CLI entrypoint of the stepped pipeline: one step at a time, never the whole chain (D39 (vii), D40; T-162, T-163).

Usage:
    uv run cli/news_nlp_steps.py --step scope --limit 500     # classify up to 500 unique bodies not yet done
    uv run cli/news_nlp_steps.py --step scope --status        # counts by state and by scope, no model loaded
    uv run cli/news_nlp_steps.py --step scope --fetch --status   # first: download the pinned model into models/scope_final
    uv run cli/news_nlp_steps.py --step scope --status --titles   # also the bodies with no title (scans the corpus)
    uv run cli/news_nlp_steps.py --step cut --limit 500       # cut up to 500 multi bodies into sub-news (step 1.2)
    uv run cli/news_nlp_steps.py --step cut --status          # sub-news written and bodies by state

A rerun adds nothing: the bodies already done at the model's revision are skipped, failed ones are retried. A new model
revision starts a new set of rows and leaves the old ones. ``--model-dir`` is the folder written by
``scripts/train_scope_final_2026_11_11.py`` (default ``models/scope_final``). The ``cut`` step reads the ``multi`` bodies
step 1 wrote (``--scope-revision``, default the latest) and writes ``sub_news`` with the ``units`` extractor; its model is the
attribution checkpoint (default ``data/models/attribution_2026_11_08``).
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

import news_nlp as db
from news_nlp import steps
from news_nlp.db import exit_on_schema_error

STEPS = (steps.STEP_SCOPE, steps.STEP_CUT)
BATCH = 16
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
    from cut_stage import UnitsExtractor, companies_of  # noqa: PLC0415 -- loads torch

    if not args.model_dir.is_dir():
        sys.exit(
            f"{args.model_dir} is not a folder: the attribution checkpoint is local (T-148b12), copy it there or pass --model-dir"
        )
    try:
        extractor = UnitsExtractor.load(args.model_dir, companies_of(conn), args.reading)
    except ValueError as exc:  # the folder lacks a model file: say so, not a stack
        sys.exit(str(exc))
    try:
        counts = run_cut(conn, extractor, scope_revision, args.limit)
    except RunAborted as exc:
        print(json.dumps(steps.status(conn, steps.STEP_CUT, extractor.revision), indent=2))
        sys.exit(f"stopped: {exc}")
    print(json.dumps({"scope_revision": scope_revision, **counts}, indent=2))
    print(json.dumps(steps.status(conn, steps.STEP_CUT, extractor.revision), indent=2))


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", choices=STEPS, required=True)
    parser.add_argument("--limit", type=int, default=None, help="touch at most this many bodies")
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
        help="cut: the classifier revision whose multi bodies are cut (default the latest)",
    )
    parser.add_argument(
        "--reading",
        choices=("units", "units4"),
        default="units",
        help="cut: how the body is read; `units` is the owner's selection, `units4` the candidate of T-176",
    )
    parser.add_argument("--source-db", default=None)
    parser.add_argument("--results-db", default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.step == steps.STEP_CUT:
        used = [f"--{n.replace('_', '-')}" for n in SCOPE_ONLY_FLAGS if getattr(args, n)]
        if used:
            parser.error(f"{', '.join(used)} only apply to --step scope")
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
    main_scope(args, conn)


if __name__ == "__main__":
    main()
