#!/usr/bin/env python
"""CLI entrypoint of the stepped pipeline: one step at a time, never the whole chain (D39 (vii), D40; T-162, T-163).

Usage:
    uv run cli/news_nlp_steps.py --step scope --limit 500     # classify up to 500 unique bodies not yet done
    uv run cli/news_nlp_steps.py --step scope --status        # counts by state and by scope, no model loaded
    uv run cli/news_nlp_steps.py --step scope --fetch --status   # first: download the pinned model into models/scope_final
    uv run cli/news_nlp_steps.py --step scope --status --titles   # also the bodies with no title (scans the corpus)
    uv run cli/news_nlp_steps.py --step cut --fetch --status  # first: download the pinned attribution model into an empty --model-dir (private: HF_TOKEN)
    uv run cli/news_nlp_steps.py --step cut --limit 500       # cut up to 500 multi bodies into sub-news (step 1.2); a folder that is not the pinned model is refused (--allow-unpinned)
    uv run cli/news_nlp_steps.py --step cut --status          # sub-news written and bodies by state
    uv run cli/news_nlp_steps.py --step summary --items single --limit 200   # step 2: summarise the single bodies over 2,000 characters
    uv run cli/news_nlp_steps.py --step summary --cut-revision units5-... --limit 200   # the single bodies and the sub-news of that cut
    uv run cli/news_nlp_steps.py --step summary --items sub_news --cut-revision units5-...   # only the sub-news of one cut revision
    uv run cli/news_nlp_steps.py --step summary --items sector --sector-revision sector-... --limit 200   # the sector bodies the sector step assigned
    uv run cli/news_nlp_steps.py --step summary --status      # summaries and originals stored, items by state
    uv run cli/news_nlp_steps.py --step companies --summary-revision summary-... --cut-revision units5-... --threshold 0.5 --candidates ner+text --limit 500   # step 3
    uv run cli/news_nlp_steps.py --step companies --status    # companies found, main rows, items on the review list
    uv run cli/news_nlp_steps.py --step sector --limit 200    # the sector step: the GICS sector of up to 200 `sector` bodies (or out_of_scope)
    uv run cli/news_nlp_steps.py --step sector --status       # bodies assigned / sent out_of_scope, by sector
    uv run cli/news_nlp_steps.py --step category --summary-revision summary-... --companies-revision companies-... --limit 500   # the category of each main company (provisional model)
    uv run cli/news_nlp_steps.py --step category --status     # categories written, by label and by group
    uv run cli/news_nlp_steps.py --step sentiment --summary-revision summary-... --companies-revision companies-... --limit 500   # the sentiment of each main company (pair model + relevance stack)
    uv run cli/news_nlp_steps.py --step sentiment --status    # sentiments written, by label
    uv run cli/news_nlp_steps.py --step snapshot --summary-revision summary-... --companies-revision companies-... --cut-revision units5-... --category-revision category-... --sentiment-revision sentiment-... --limit 500   # step 8: one snapshot per (text, main company)
    uv run cli/news_nlp_steps.py --step snapshot --status     # snapshots written by revision (with the five revisions: the pairs still waiting and why)

A rerun adds nothing: the bodies already done at the model's revision are skipped, failed ones are retried. A new model
revision starts a new set of rows and leaves the old ones. ``--model-dir`` is the folder written by
``scripts/train_scope_final_2026_11_11.py`` (default ``models/scope_final``). The ``cut`` step reads the ``multi`` bodies
step 1 wrote (``--scope-revision``, default the latest) and writes ``sub_news`` with the ``units`` extractor; its model is the
attribution checkpoint (default ``data/models/attribution_2026_11_08``). The ``summary`` step (step 2) reads the ``single`` bodies of step 1
and the sub-news of the cut (``--cut-revision``, default the latest); an item over 2,000 characters is summarised by the pinned local
summariser, a shorter one is stored as it is (``news_summary.text_source`` = ``original``); see ``src/news_summary_stage.py``.
The ``companies`` step (step 3) reads the texts step 2 left (``--summary-revision``, the single bodies of ``--scope-revision`` and the sub-news of
``--cut-revision``), finds the universe companies each names with NER and the matcher, scores each pair with the relevance model and marks every
company at ``--threshold`` or over it as main; an item with none goes to the review list; see ``src/news_company_stage.py``.
The ``sector`` step (T-167) reads the ``sector`` bodies of step 1 and writes ``body_sector``: the GICS sector (2 digits) when the
embedding score is confident or the companies named agree with it, else ``out_of_scope``; its model is the pin of
``config/sector_step.json``; see ``src/sector_stage.py`` and ``docs/sector_step.md``.
The ``category`` step (T-168, **provisional**: the pinned zero-shot model, not yet the fine-tuned one of T-174) reads the texts step 2 left
(``--summary-revision``) that have a main company at ``--companies-revision`` and writes one ``news_category`` row per (text, main company): one of
the 14 labels of D15 and its group; see ``src/news_category_stage.py`` and ``docs/category_step.md``.
The ``sentiment`` step (T-168) reads the same texts and main companies and writes one ``news_sentiment`` row per (text, main company): positive, negative or
neutral and the three probabilities of the stack of the pair model (``config/sentiment_step.json``, the local folder) and the relevance score step 3 stored;
nothing is downloaded; see ``src/news_sentiment_stage.py`` and ``docs/sentiment_step.md``.
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

STEPS = (
    steps.STEP_SCOPE,
    steps.STEP_CUT,
    steps.STEP_SUMMARY,
    steps.STEP_COMPANIES,
    steps.STEP_SECTOR,
    steps.STEP_CATEGORY,
    steps.STEP_SENTIMENT,
    steps.STEP_SNAPSHOT,
)
BATCH = 16
# texts stored per transaction, redone one by one when a batch fails (the model scores one pair at a time: 14 hypotheses each);
# `batch_size` of config/pipeline_models.json `category` belongs to the v1 stage and is not read here
CATEGORY_BATCH = 4
SENTIMENT_BATCH = 8  # texts stored per transaction; the pair model reads one chunk at a time
COMPANY_BATCH = 16  # texts read together by NER and the relevance model
SUMMARY_BATCH = 8  # texts summarised together; the model's own batch (4) is applied inside
MAX_FAILED_BATCHES = 3  # consecutive batches in which every body failed: the cause is not one body
MAX_FAILED_BODIES = 10  # the same for the cut, which goes body by body
SCOPE_ONLY_FLAGS = ("titles",)


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


class SectorClassifying(Protocol):
    """What ``run_sector`` needs of the sector classifier (``sector_stage.SectorClassifier`` or a test stub)."""

    revision: str

    def classify(self, items: Sequence[tuple[str, str]]) -> list[Any]: ...


def _sector_batch(
    conn: db.NewsNlpDatabase,
    classifier: SectorClassifying,
    batch: list[steps.Body],
    counts: dict[str, int],
) -> None:
    """Classify and store one batch; a batch that fails is redone body by body, so one bad body fails alone."""
    try:
        results = classifier.classify([(b.title, b.text) for b in batch])
    except Exception as exc:
        if len(batch) > 1:
            for body in batch:
                _sector_batch(conn, classifier, [body], counts)
            return
        steps.record_failure(
            conn, batch[0], steps.STEP_SECTOR, classifier.revision, steps.error_text(exc)
        )
        counts["failed"] += 1
        return
    for body, r in zip(batch, results, strict=True):
        scores = dict(r.scores)
        if (
            r.truncated_chunks
        ):  # recorded with the body, not only counted (constitution, Pipeline models 3)
            scores[steps.TRUNCATED_KEY] = r.truncated_chunks
            counts["truncated_chunks"] = counts.get("truncated_chunks", 0) + r.truncated_chunks
        steps.record_sector(
            conn,
            body,
            classifier.revision,
            r.status,
            r.sector,
            r.candidate,
            r.margin,
            r.vote_sector,
            scores,
            r.votes,
        )
        counts["done"] += 1
        counts[r.status] = counts.get(r.status, 0) + 1


def run_sector(
    conn: db.NewsNlpDatabase,
    classifier: SectorClassifying,
    scope_revision: str,
    limit: int | None,
) -> dict[str, int]:
    """Give a sector to up to ``limit`` ``sector`` bodies not yet done at the classifier's revision; return what was done."""
    print(
        "reading SOURCE for the sector bodies still to do (it scans the corpus: minutes on the full one)",
        file=sys.stderr,
    )
    todo = list(steps.pending_sector_bodies(conn, scope_revision, classifier.revision, limit))
    print(f"pending {len(todo):,} sector bodies at revision {classifier.revision}", file=sys.stderr)
    prepare = getattr(classifier, "prepare", None)
    if todo and callable(prepare):
        prepare()  # the model is loaded here, not in the first batch: the bar's rate and time left count bodies only
    counts: dict[str, int] = {"done": 0, "failed": 0}
    failed_batches = 0
    with tqdm(total=len(todo), desc="sector", unit="body") as bar:
        for i in range(0, len(todo), BATCH):
            batch = todo[i : i + BATCH]
            done_before = counts["done"]
            _sector_batch(conn, classifier, batch, counts)
            bar.update(len(batch))
            failed_batches = failed_batches + 1 if counts["done"] == done_before else 0
            if failed_batches >= MAX_FAILED_BATCHES:
                raise RunAborted(
                    f"{failed_batches} batches in a row failed completely after {counts['done']} done "
                    f"(see the error of a failed body with --status); the failed bodies are retried by the next run"
                )
    if counts.get(
        "truncated_chunks"
    ):  # a chunk of 120 words never fills the window: say it when one does
        print(
            f"warning: {counts['truncated_chunks']} chunk(s) were cut at the model's window (kept in body_sector.scores)",
            file=sys.stderr,
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
    sector_revision: str | None = None,
) -> list[SummaryItem]:
    """The single bodies, then the sub-news, then the sector bodies, still to do at ``revision``; at most ``limit`` items in all."""
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
    if "sector" in items and sector_revision is not None:
        left = None if limit is None else max(limit - len(todo), 0)
        if left != 0:
            for body in steps.pending_assigned_sector_bodies(conn, sector_revision, revision, left):
                todo.append(
                    SummaryItem(
                        steps.ITEM_BODY, body.body_hash, input_of_body(body.title, body.text)
                    )
                )
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
    sector_revision: str | None = None,
) -> dict[str, int]:
    """Step 2: summarise up to ``limit`` items not yet done at the summariser's revision; return what was done.

    An item over ``summarizer.min_chars`` goes to the model in batches; a shorter one is stored as it is
    (``text_source = original``, no chunks). A batch that fails is redone item by item, an item that fails is stored
    ``failed`` with its reason and retried by the next run, and ``MAX_FAILED_BATCHES`` batches in a row that fail
    completely stop the run (the cause is then not one text: out of memory, a lost GPU).
    """
    from news_summary_stage import needs_summary  # noqa: PLC0415 -- loads torch

    todo = _summary_items(
        conn, summarizer.revision, scope_revision, cut_revision, items, limit, sector_revision
    )
    over: list[SummaryItem] = []
    kept: list[SummaryItem] = []
    for item in todo:
        (over if needs_summary(item.text, summarizer.min_chars) else kept).append(item)
    _say_the_plan(todo, len(over), len(kept), summarizer.min_chars)
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

    # two phases, two bars: the items under the size cost a write each (286 a second measured on a local disk), the ones over it cost
    # the model (seconds each). One bar over both made its rate and its time left meaningless.
    if kept:
        with tqdm(total=len(kept), desc="original", unit="item") as bar:
            for item in kept:
                steps.record_summary(
                    conn, item.kind, item.item_id, summarizer.revision, item.text, "original", 0
                )
                counts["original"] += 1
                bar.update(1)
    if over:
        with tqdm(total=len(over), desc="summary", unit="item") as bar:
            for item in over:
                waiting.append(item)
                if len(waiting) >= SUMMARY_BATCH:
                    size = len(waiting)
                    flush()
                    bar.update(size)
            size = len(waiting)
            flush()
            bar.update(size)
    return counts


def _say_the_plan(todo: Sequence[SummaryItem], over: int, kept: int, min_chars: int) -> None:
    """Say before the work what the bars will count: how many items there are, and how many of them need the model."""
    if not todo:
        print("nothing pending", file=sys.stderr)
        return
    bodies = sum(1 for item in todo if item.kind == steps.ITEM_BODY)
    print(
        f"pending {len(todo):,} items ({bodies:,} bodies, {len(todo) - bodies:,} sub-news): "
        f"{over:,} to summarise with the model (over {min_chars:,} characters), {kept:,} stored as they are",
        file=sys.stderr,
    )


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
        folder_identity,
        is_pinned,
        pin_status,
    )

    if not args.model_dir.is_dir():
        sys.exit(
            f"{args.model_dir} is not a folder: run with --fetch to download the pinned attribution model "
            "(config/attribution_model.json, private: needs HF_TOKEN) or pass --model-dir"
        )
    try:
        identity = folder_identity(args.model_dir)  # once: the weights are large
        print(pin_status(args.model_dir, identity=identity), file=sys.stderr)
        pinned = is_pinned(args.model_dir, identity=identity)
    except ValueError as exc:  # the folder lacks a model file: say so, not a stack
        sys.exit(str(exc))
    if pinned is not True and not args.allow_unpinned:
        sys.exit(
            "refused: config/attribution_model.json is missing, so no folder can be checked against the pin"
            if pinned is None
            else "refused: the folder is not the pinned attribution model (config/attribution_model.json). "
            "Use --fetch into an empty --model-dir, or --allow-unpinned for a development copy"
        )
    try:
        extractor = UnitsExtractor.load(
            args.model_dir, companies_of(conn), args.reading or DEFAULT_READING
        )
    except ValueError as exc:  # the folder lacks a model file: say so, not a stack
        sys.exit(str(exc))
    print(
        f"cutting with the reading {args.reading or DEFAULT_READING}, revision {extractor.revision}",
        file=sys.stderr,
    )
    try:
        counts = run_cut(conn, extractor, scope_revision, args.limit)
    except RunAborted as exc:
        print(json.dumps(steps.status(conn, steps.STEP_CUT, extractor.revision), indent=2))
        sys.exit(f"stopped: {exc}")
    print(json.dumps({"scope_revision": scope_revision, **counts}, indent=2))
    print(json.dumps(steps.status(conn, steps.STEP_CUT, extractor.revision), indent=2))


def main_sector(args: argparse.Namespace, conn: db.NewsNlpDatabase) -> None:
    if args.status:
        # the revision depends on the universe, so it is not known without reading it: the latest one with state is shown
        latest = steps.latest_revision(conn, steps.STEP_SECTOR)
        report = steps.status(conn, steps.STEP_SECTOR, latest)
        report["revisions"] = steps.revisions(conn, steps.STEP_SECTOR)
        print(json.dumps(report, indent=2))
        return
    scope_revision = args.scope_revision or steps.latest_classifier_revision(conn)
    if scope_revision is None:
        sys.exit("body_scope is empty: run --step scope first")
    from cut_stage import companies_of  # noqa: PLC0415 -- the universe, as the cut reads it
    from sector_stage import SectorClassifier  # noqa: PLC0415 -- numpy; torch on the 1st body

    companies = companies_of(conn)
    if not companies:
        sys.exit(
            "the SOURCE store has no companies: the sector step needs the universe (see docs/db-topology.md)"
        )
    classifier = SectorClassifier(companies)
    print(f"sector step, revision {classifier.revision}", file=sys.stderr)
    try:
        counts = run_sector(conn, classifier, scope_revision, args.limit)
    except RunAborted as exc:
        print(json.dumps(steps.status(conn, steps.STEP_SECTOR, classifier.revision), indent=2))
        sys.exit(f"stopped: {exc}")
    print(json.dumps({"scope_revision": scope_revision, **counts}, indent=2))
    print(json.dumps(steps.status(conn, steps.STEP_SECTOR, classifier.revision), indent=2))


def _sector_revision_read(conn: db.NewsNlpDatabase, sector_revision: str | None) -> str:
    """The sector revision a run of sector summaries reads: named, never "the latest", and with bodies assigned at it."""
    ran = ", ".join(steps.revisions(conn, steps.STEP_SECTOR)) or "none"
    if sector_revision is None:
        sys.exit(
            f"sector bodies need the sector step they come from: pass --sector-revision (the sector revisions that ran: {ran})"
        )
    found = conn.execute(
        "SELECT 1 FROM body_sector WHERE classifier_revision = ? AND status = 'assigned' LIMIT 1",
        (sector_revision,),
    ).fetchone()
    if found is None:
        sys.exit(
            f"the sector step assigned no body at revision {sector_revision} (the sector revisions that ran: {ran})"
        )
    return sector_revision


def main_summary(args: argparse.Namespace, conn: db.NewsNlpDatabase) -> None:
    if args.status:
        latest = steps.latest_revision(conn, steps.STEP_SUMMARY)
        report = steps.status(conn, steps.STEP_SUMMARY, latest)
        report["revisions"] = steps.revisions(conn, steps.STEP_SUMMARY)
        print(json.dumps(report, indent=2))
        return
    items = (args.items,) if args.items else ("single", "sub_news")
    # only the revisions this run reads: the scope revision for the single bodies, the named cut for the sub-news
    scope_revision = None
    if "single" in items:
        scope_revision = args.scope_revision or steps.latest_classifier_revision(conn)
        if scope_revision is None:
            sys.exit("body_scope is empty: run --step scope first")
    cut_revision = args.cut_revision
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
    sector_revision = args.sector_revision
    if "sector" in items:
        sector_revision = _sector_revision_read(conn, sector_revision)
    from news_summary_stage import Summarizer  # noqa: PLC0415 -- loads torch

    summarizer = Summarizer()
    print(
        f"summariser {summarizer.model_name}@{summarizer.model_revision[:8]} on {summarizer.device}, size {summarizer.min_chars}",
        file=sys.stderr,
    )
    try:
        counts = run_summary(
            conn, summarizer, scope_revision, cut_revision, items, args.limit, sector_revision
        )
    except RunAborted as exc:
        print(json.dumps(steps.status(conn, steps.STEP_SUMMARY, summarizer.revision), indent=2))
        sys.exit(f"stopped: {exc}")
    read = {
        "scope_revision": scope_revision,
        "cut_revision": cut_revision,
        "sector_revision": sector_revision,
    }
    print(json.dumps({**{k: v for k, v in read.items() if v is not None}, **counts}, indent=2))
    print(json.dumps(steps.status(conn, steps.STEP_SUMMARY, summarizer.revision), indent=2))


class CompanyFinding(Protocol):
    """What ``run_companies`` needs of a finder (``news_company_stage.CompanyFinder`` or a test stub)."""

    revision: str
    threshold: float

    def find(self, texts: Sequence[str]) -> list[list[steps.CompanyScore]]: ...


def _find_companies(
    conn: db.NewsNlpDatabase,
    finder: CompanyFinding,
    batch: list[steps.TextItem],
    counts: dict[str, int],
) -> None:
    """Find and store the companies of one batch; a batch that fails is redone item by item, so one bad text fails alone."""
    try:
        results = finder.find([item.text for item in batch])
    except Exception as exc:
        if len(batch) > 1:
            for item in batch:
                _find_companies(conn, finder, [item], counts)
            return
        steps.record_item_failure(
            conn,
            batch[0].kind,
            batch[0].item_id,
            steps.STEP_COMPANIES,
            finder.revision,
            steps.error_text(exc),
        )
        counts["failed"] += 1
        return
    for item, found in zip(batch, results, strict=True):
        state = steps.record_companies(
            conn, item.kind, item.item_id, finder.revision, found, finder.threshold
        )
        counts[state] += 1
        counts["main_rows"] += sum(
            1 for c in found if steps.is_main(c.relevance_score, finder.threshold)
        )


def run_companies(
    conn: db.NewsNlpDatabase,
    finder: CompanyFinding,
    summary_revision: str,
    scope_revision: str | None,
    cut_revision: str | None,
    items: Sequence[str],
    limit: int | None,
) -> dict[str, int]:
    """Step 3: find the companies of up to ``limit`` texts not yet finished at the finder's revision; return what was done.

    ``done`` is an item with a main company, ``review`` an item with none (it waits for the owner and is not retried at this revision),
    ``failed`` an item the models could not read (retried by the next run), ``main_rows`` the ``is_main`` rows written. A batch that fails is
    redone item by item, and ``MAX_FAILED_BATCHES`` batches in a row that fail completely stop the run.
    """
    todo = list(
        steps.pending_texts(
            conn,
            summary_revision,
            finder.revision,
            scope_revision=scope_revision,
            cut_revision=cut_revision,
            items=tuple(items),
            limit=limit,
        )
    )
    if not todo:
        print("nothing pending", file=sys.stderr)
    counts: dict[str, int] = {"done": 0, "review": 0, "failed": 0, "main_rows": 0}
    failed_batches = 0
    with tqdm(total=len(todo), desc="companies", unit="item") as bar:
        for start in range(0, len(todo), COMPANY_BATCH):
            batch = todo[start : start + COMPANY_BATCH]
            failed_before = counts["failed"]
            _find_companies(conn, finder, batch, counts)
            failed_batches = (
                failed_batches + 1 if counts["failed"] - failed_before == len(batch) else 0
            )
            if failed_batches >= MAX_FAILED_BATCHES:
                raise RunAborted(
                    f"{failed_batches} batches in a row failed completely after {counts['done'] + counts['review']} done "
                    f"(see the error of a failed item with --status); the failed items are retried by the next run"
                )
            bar.update(len(batch))
    return counts


def main_companies(args: argparse.Namespace, conn: db.NewsNlpDatabase) -> None:
    import news_company_stage  # noqa: PLC0415 -- imports torch, loads no model

    if args.status:
        latest = steps.latest_revision(conn, steps.STEP_COMPANIES)
        report = steps.status(conn, steps.STEP_COMPANIES, latest)
        report["revisions"] = steps.revisions(conn, steps.STEP_COMPANIES)
        report["models"] = news_company_stage.models_of_the_step()
        print(json.dumps(report, indent=2))
        return
    items = (args.items,) if args.items else ("single", "sub_news")
    ran_summaries = ", ".join(steps.revisions(conn, steps.STEP_SUMMARY)) or "none"
    found = conn.execute(
        "SELECT 1 FROM news_summary WHERE model_revision = ? LIMIT 1", (args.summary_revision,)
    ).fetchone()
    if found is None:
        sys.exit(
            f"step 2 left no text at the summary revision {args.summary_revision} (the revisions that ran: {ran_summaries})"
        )
    scope_revision = None
    if "single" in items:
        scope_revision = args.scope_revision or steps.latest_classifier_revision(conn)
        if scope_revision is None:
            sys.exit("body_scope is empty: run --step scope first")
    cut_revision = args.cut_revision
    if "sub_news" in items:
        ran = ", ".join(steps.revisions(conn, steps.STEP_CUT)) or "none"
        if cut_revision is None:
            sys.exit(
                f"sub-news need the cut they come from: pass --cut-revision (the cuts that ran: {ran}), "
                "or --items single"
            )
        if (
            conn.execute(
                "SELECT 1 FROM sub_news WHERE extractor_revision = ? LIMIT 1", (cut_revision,)
            ).fetchone()
            is None
        ):
            sys.exit(
                f"the cut wrote no sub-news at revision {cut_revision} (the cuts that ran: {ran})"
            )
    from cut_stage import companies_of  # noqa: PLC0415 -- loads torch
    from news_company_stage import CompanyFinder  # noqa: PLC0415 -- loads torch

    finder = CompanyFinder(companies_of(conn), args.threshold, args.candidates)
    print(
        f"NER {finder.spans.model_name}@{finder.spans.model_revision[:8]}, relevance {finder.scorer.repo}@{finder.scorer.revision[:8]} "
        f"on {finder.scorer.device}, threshold {finder.threshold}, candidates {finder.candidates}, revision {finder.revision}",
        file=sys.stderr,
    )
    try:
        counts = run_companies(
            conn, finder, args.summary_revision, scope_revision, cut_revision, items, args.limit
        )
    except RunAborted as exc:
        print(json.dumps(steps.status(conn, steps.STEP_COMPANIES, finder.revision), indent=2))
        sys.exit(f"stopped: {exc}")
    read = {
        "summary_revision": args.summary_revision,
        "scope_revision": scope_revision,
        "cut_revision": cut_revision,
        "threshold": finder.threshold,
        "candidates": finder.candidates,
    }
    print(json.dumps({**{k: v for k, v in read.items() if v is not None}, **counts}, indent=2))
    print(json.dumps(steps.status(conn, steps.STEP_COMPANIES, finder.revision), indent=2))


class CategoryClassifying(Protocol):
    """What ``run_category`` needs of a finder (``news_category_stage.CategoryFinder`` or a test stub)."""

    revision: str
    model_name: str

    def classify(self, pairs: Sequence[tuple[str, str]]) -> list[steps.CategoryScore]: ...


def _gpu_lost(exc: BaseException) -> bool:
    """Whether the error is a CUDA runtime fault (``CUDA error: unknown error``, a device-side assert): after it the context of the process
    is unusable and every later call fails too. An out-of-memory error is not one, in either wording (``torch.cuda.OutOfMemoryError``, "CUDA out of
    memory", the driver's "CUDA error: out of memory"): a smaller input can survive it."""
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
        return False
    return isinstance(exc, RuntimeError) and "CUDA error" in str(exc)


def _classify_items(
    conn: db.NewsNlpDatabase,
    finder: CategoryClassifying,
    names: dict[str, str],
    batch: list[steps.CategoryItem],
    counts: dict[str, int],
) -> None:
    """Classify and store one batch; a batch that fails is redone item by item, so one bad text fails alone. A blank text is not
    classified: the item is ``skipped`` with the reason, a label of nothing would look as sure as any other."""
    for blank in [item for item in batch if not item.text.strip()]:
        steps.record_item_skipped(
            conn, blank.kind, blank.item_id, steps.STEP_CATEGORY, finder.revision, "blank text"
        )
        counts["skipped"] += 1
    batch = [item for item in batch if item.text.strip()]
    if not batch:
        return
    pairs = [(item.text, names.get(t, t)) for item in batch for t in item.tickers]
    nameless = sum(1 for item in batch for t in item.tickers if t not in names)
    try:
        results = finder.classify(pairs)
    except Exception as exc:
        if _gpu_lost(exc):
            raise RunAborted(
                f"the GPU failed ({steps.error_text(exc).splitlines()[0]}): the CUDA context of this process is unusable. "
                "Rerun the same command: the finished items are kept and none was marked failed"
            ) from exc
        if len(batch) > 1:
            for item in batch:
                _classify_items(conn, finder, names, [item], counts)
            return
        steps.record_item_failure(
            conn,
            batch[0].kind,
            batch[0].item_id,
            steps.STEP_CATEGORY,
            finder.revision,
            steps.error_text(exc),
        )
        counts["failed"] += 1
        return
    taken = iter(results)
    for item in batch:
        found = {ticker: next(taken) for ticker in item.tickers}
        steps.record_category(
            conn, item.kind, item.item_id, finder.revision, found, finder.model_name
        )
        counts["done"] += 1
        counts["category_rows"] += len(found)
    counts["name_missing"] += nameless


def run_category(
    conn: db.NewsNlpDatabase,
    finder: CategoryClassifying,
    names: dict[str, str],
    summary_revision: str,
    companies_revision: str,
    limit: int | None,
) -> dict[str, int]:
    """Step 7: classify up to ``limit`` texts with a main company not yet finished at the finder's revision; return what was done.

    ``done`` is an item with all its main companies classified, ``category_rows`` the ``news_category`` rows written, ``failed`` an item the model
    could not read (retried by the next run), ``skipped`` an item with a blank text, ``name_missing`` the pairs whose ticker is not in the universe (the ticker is read as the name). A batch that fails is redone item by item, and ``MAX_FAILED_BATCHES`` batches in a row that fail
    completely stop the run.
    """
    todo = list(
        steps.pending_category_items(
            conn, summary_revision, companies_revision, finder.revision, limit=limit
        )
    )
    if not todo:
        print("nothing pending", file=sys.stderr)
    else:
        pairs = sum(len(item.tickers) for item in todo)
        print(f"{len(todo)} items ({pairs} pairs) to do", file=sys.stderr)
        warm_up = getattr(finder, "warm_up", None)
        if warm_up is not None:
            warm_up()  # the model loads before the progress bar, not inside its first step
    counts: dict[str, int] = {
        "done": 0,
        "failed": 0,
        "skipped": 0,
        "category_rows": 0,
        "name_missing": 0,
    }
    failed_batches = 0
    with tqdm(total=len(todo), desc="category", unit="item") as bar:
        for start in range(0, len(todo), CATEGORY_BATCH):
            batch = todo[start : start + CATEGORY_BATCH]
            failed_before = counts["failed"]
            _classify_items(conn, finder, names, batch, counts)
            failed_batches = (
                failed_batches + 1 if counts["failed"] - failed_before == len(batch) else 0
            )
            if failed_batches >= MAX_FAILED_BATCHES:
                raise RunAborted(
                    f"{failed_batches} batches in a row failed completely after {counts['done']} done "
                    f"(see the error of a failed item with --status); the failed items are retried by the next run"
                )
            bar.update(len(batch))
    return counts


def main_category(args: argparse.Namespace, conn: db.NewsNlpDatabase) -> None:
    import news_category_stage  # noqa: PLC0415 -- imports torch, loads no model

    if args.status:
        summary_revision = getattr(args, "summary_revision", None)
        companies_revision = getattr(args, "companies_revision", None)
        # what this code would write for those inputs; compare it with "revisions" (no model is loaded for it)
        code_revision = (
            news_category_stage.code_revision(summary_revision, companies_revision)
            if summary_revision and companies_revision
            else None
        )
        # with the inputs the counts are those of the revision this code writes (the one a run continues); without them the latest by time,
        # which can be a partial revision of an older code
        shown = code_revision or steps.latest_revision(conn, steps.STEP_CATEGORY)
        report = steps.status(conn, steps.STEP_CATEGORY, shown)
        report["revisions"] = steps.revisions(conn, steps.STEP_CATEGORY)
        report["models"] = news_category_stage.models_of_the_step()
        report["code_revision"] = (
            code_revision
            or "give --summary-revision and --companies-revision to see the revision this code would write"
        )
        print(json.dumps(report, indent=2))
        return
    ran_summaries = ", ".join(steps.revisions(conn, steps.STEP_SUMMARY)) or "none"
    if (
        conn.execute(
            "SELECT 1 FROM news_summary WHERE model_revision = ? LIMIT 1", (args.summary_revision,)
        ).fetchone()
        is None
    ):
        sys.exit(
            f"step 2 left no text at the summary revision {args.summary_revision} (the revisions that ran: {ran_summaries})"
        )
    ran_companies = ", ".join(steps.revisions(conn, steps.STEP_COMPANIES)) or "none"
    if (
        conn.execute(
            "SELECT 1 FROM news_company WHERE model_revision = ? AND is_main = 1 LIMIT 1",
            (args.companies_revision,),
        ).fetchone()
        is None
    ):
        sys.exit(
            f"step 3 found no main company at the companies revision {args.companies_revision} (the revisions that ran: {ran_companies})"
        )
    from cut_stage import companies_of  # noqa: PLC0415 -- the universe, as the cut reads it
    from news_category_stage import CategoryFinder  # noqa: PLC0415 -- loads torch

    names = {c.ticker: c.name for c in companies_of(conn)}
    no_text = steps.category_items_without_text(
        conn, args.summary_revision, args.companies_revision
    )
    if no_text:
        print(
            f"warning: {no_text} items with a main company at {args.companies_revision} have no text at {args.summary_revision} and are skipped "
            "(did the companies step read another summary revision?)",
            file=sys.stderr,
        )
    finder = CategoryFinder(args.summary_revision, args.companies_revision)
    print(
        f"category model {finder.model_name}@{finder.model_revision[:8]} (provisional, zero-shot) on {finder.scorer.device}, revision {finder.revision}",
        file=sys.stderr,
    )
    try:
        counts = run_category(
            conn, finder, names, args.summary_revision, args.companies_revision, args.limit
        )
    except RunAborted as exc:
        print(json.dumps(steps.status(conn, steps.STEP_CATEGORY, finder.revision), indent=2))
        sys.exit(f"stopped: {exc}")
    read = {
        "summary_revision": args.summary_revision,
        "companies_revision": args.companies_revision,
        "items_without_text": no_text,
    }
    print(json.dumps({**read, **counts}, indent=2))
    print(json.dumps(steps.status(conn, steps.STEP_CATEGORY, finder.revision), indent=2))


class SentimentClassifying(Protocol):
    """What ``run_sentiment`` needs of a finder (``news_sentiment_stage.SentimentFinder`` or a test stub)."""

    revision: str
    model_name: str

    def classify(
        self, triples: Sequence[tuple[str, str, float]]
    ) -> list[steps.SentimentScore | None]: ...


def _named_items(
    conn: db.NewsNlpDatabase,
    finder: SentimentClassifying,
    names: dict[str, str],
    batch: list[steps.SentimentItem],
    counts: dict[str, int],
) -> list[steps.SentimentItem]:
    """The items of ``batch`` with only the tickers the universe names. A ticker it does not name is not scored: the model was trained on the
    universe's names, and a ticker read as a name would store a row as sure as any other from an input it never saw. An item with no named ticker is
    skipped with the reason (rerun it once the universe is fixed)."""
    named: list[steps.SentimentItem] = []
    for item in batch:
        known = tuple((ticker, rel) for ticker, rel in item.tickers if ticker in names)
        left = tuple(ticker for ticker, _ in item.tickers if ticker not in names)
        counts["name_missing"] += len(left)
        if known:
            named.append(item._replace(tickers=known, dropped=item.dropped + left))
        else:
            steps.record_item_skipped(
                conn,
                item.kind,
                item.item_id,
                steps.STEP_SENTIMENT,
                finder.revision,
                "company not in the universe",
            )
            counts["skipped"] += 1
    return named


def _score_items(
    conn: db.NewsNlpDatabase,
    finder: SentimentClassifying,
    names: dict[str, str],
    batch: list[steps.SentimentItem],
    counts: dict[str, int],
) -> None:
    """Score and store one batch; a batch that fails is redone item by item, so one bad text fails alone. A blank text, or one with nothing to read
    (no sentence), is not scored: the item is ``skipped`` with the reason, a label of nothing would look as sure as any other."""
    for blank in [item for item in batch if not item.text.strip()]:
        steps.record_item_skipped(
            conn, blank.kind, blank.item_id, steps.STEP_SENTIMENT, finder.revision, "blank text"
        )
        counts["skipped"] += 1
    batch = [item for item in batch if item.text.strip()]
    batch = _named_items(conn, finder, names, batch, counts)
    if not batch:
        return
    triples = [
        (item.text, names[ticker], relevance)
        for item in batch
        for ticker, relevance in item.tickers
    ]
    try:
        results = finder.classify(triples)
    except Exception as exc:
        if _gpu_lost(exc):
            raise RunAborted(
                f"the GPU failed ({steps.error_text(exc).splitlines()[0]}): the CUDA context of this process is unusable. "
                "Rerun the same command: the finished items are kept and none was marked failed"
            ) from exc
        if len(batch) > 1:
            for item in batch:
                _score_items(conn, finder, names, [item], counts)
            return
        steps.record_item_failure(
            conn,
            batch[0].kind,
            batch[0].item_id,
            steps.STEP_SENTIMENT,
            finder.revision,
            steps.error_text(exc),
        )
        counts["failed"] += 1
        return
    taken = iter(results)
    for item in batch:
        scored = {ticker: next(taken) for ticker, _ in item.tickers}
        found = {ticker: score for ticker, score in scored.items() if score is not None}
        if not found:  # every pair of an item shares its text: the text has no sentence to read
            steps.record_item_skipped(
                conn,
                item.kind,
                item.item_id,
                steps.STEP_SENTIMENT,
                finder.revision,
                "no sentence to read",
            )
            counts["skipped"] += 1
            continue
        note = (
            "not scored, company not in the universe: " + ", ".join(item.dropped)
            if item.dropped
            else None
        )
        steps.record_sentiment(
            conn, item.kind, item.item_id, finder.revision, found, finder.model_name, note
        )
        counts["done"] += 1
        counts["sentiment_rows"] += len(found)


def run_sentiment(
    conn: db.NewsNlpDatabase,
    finder: SentimentClassifying,
    names: dict[str, str],
    summary_revision: str,
    companies_revision: str,
    limit: int | None,
) -> dict[str, int]:
    """Step 6: score up to ``limit`` texts with a main company not yet finished at the finder's revision; return what was done.

    ``done`` is an item with its main companies of the universe scored, ``sentiment_rows`` the ``news_sentiment`` rows written, ``failed`` an item the model could
    not read (retried by the next run), ``skipped`` an item with a blank text, no sentence or no company of the universe, ``name_missing`` the pairs whose ticker is not in the
    universe (not scored; an item with none left is skipped). A batch that fails is redone item by item, and ``MAX_FAILED_BATCHES`` batches in a row that fail
    completely stop the run.
    """
    todo = list(
        steps.pending_sentiment_items(
            conn, summary_revision, companies_revision, finder.revision, limit=limit
        )
    )
    if not todo:
        print("nothing pending", file=sys.stderr)
    else:
        pairs = sum(len(item.tickers) for item in todo)
        print(f"{len(todo)} items ({pairs} pairs) to do", file=sys.stderr)
        warm_up = getattr(finder, "warm_up", None)
        if warm_up is not None:
            warm_up()  # the model loads before the progress bar, not inside its first step
    counts: dict[str, int] = {
        "done": 0,
        "failed": 0,
        "skipped": 0,
        "sentiment_rows": 0,
        "name_missing": 0,
    }
    failed_batches = 0
    with tqdm(total=len(todo), desc="sentiment", unit="item") as bar:
        for start in range(0, len(todo), SENTIMENT_BATCH):
            batch = todo[start : start + SENTIMENT_BATCH]
            failed_before = counts["failed"]
            _score_items(conn, finder, names, batch, counts)
            failed_batches = (
                failed_batches + 1 if counts["failed"] - failed_before == len(batch) else 0
            )
            if failed_batches >= MAX_FAILED_BATCHES:
                raise RunAborted(
                    f"{failed_batches} batches in a row failed completely after {counts['done']} done "
                    f"(see the error of a failed item with --status); the failed items are retried by the next run"
                )
            bar.update(len(batch))
    return counts


def _sentiment_status(args: argparse.Namespace, conn: db.NewsNlpDatabase) -> None:
    """``--status`` of the sentiment step: the counts, the revisions that ran, the model, and the revision this code would write for the inputs given
    (from the pin and the names of the main companies; no weight is read), to compare with "revisions"."""
    import news_sentiment_stage  # noqa: PLC0415 -- imports torch, loads no model
    from cut_stage import companies_of  # noqa: PLC0415 -- the universe, as the cut reads it

    summary_revision = getattr(args, "summary_revision", None)
    companies_revision = getattr(args, "companies_revision", None)
    code_revision = None
    unknown = (
        "give --summary-revision and --companies-revision to see the revision this code would write"
    )
    if summary_revision and companies_revision:
        try:
            known = {c.ticker: c.name for c in companies_of(conn)}
        except (
            Exception
        ) as exc:  # a store with no source articles: the counts and the revisions are still printed
            known = {}
            unknown = f"not computed: the universe could not be read ({exc}); give --source-db"
        if known:
            code_revision = news_sentiment_stage.code_revision(
                summary_revision,
                companies_revision,
                news_sentiment_stage.names_identity(conn, companies_revision, known),
            )
        elif unknown.startswith("give"):
            unknown = "not computed: the company universe is empty; give --source-db"
    shown = code_revision or steps.latest_revision(conn, steps.STEP_SENTIMENT)
    report = steps.status(conn, steps.STEP_SENTIMENT, shown)
    report["revisions"] = steps.revisions(conn, steps.STEP_SENTIMENT)
    report["models"] = news_sentiment_stage.models_of_the_step()
    if summary_revision and companies_revision:
        # a run is refused when this is not 0 (step 3 read another summary revision)
        report["items_without_text"] = steps.category_items_without_text(
            conn, summary_revision, companies_revision
        )
    report["code_revision"] = code_revision or unknown
    print(json.dumps(report, indent=2))


def main_sentiment(args: argparse.Namespace, conn: db.NewsNlpDatabase) -> None:
    import news_sentiment_stage  # noqa: PLC0415 -- imports torch, loads no model

    if args.status:
        _sentiment_status(args, conn)
        return
    ran_summaries = ", ".join(steps.revisions(conn, steps.STEP_SUMMARY)) or "none"
    if (
        conn.execute(
            "SELECT 1 FROM news_summary WHERE model_revision = ? LIMIT 1", (args.summary_revision,)
        ).fetchone()
        is None
    ):
        sys.exit(
            f"step 2 left no text at the summary revision {args.summary_revision} (the revisions that ran: {ran_summaries})"
        )
    ran_companies = ", ".join(steps.revisions(conn, steps.STEP_COMPANIES)) or "none"
    if (
        conn.execute(
            "SELECT 1 FROM news_company WHERE model_revision = ? AND is_main = 1 LIMIT 1",
            (args.companies_revision,),
        ).fetchone()
        is None
    ):
        sys.exit(
            f"step 3 found no main company at the companies revision {args.companies_revision} (the revisions that ran: {ran_companies})"
        )
    # step 3 does not record which summary revision it read, so the check is what can be seen: the relevance score is the one of the text step 6
    # reads only if every main item of the companies revision has a text at the summary revision. A gap means another summary revision.
    no_text = steps.category_items_without_text(
        conn, args.summary_revision, args.companies_revision
    )
    if no_text:
        sys.exit(
            f"refused: {no_text} items with a main company at {args.companies_revision} have no text at {args.summary_revision}: step 3 read "
            "another summary revision, and its relevance scores are not those of these texts"
        )
    from cut_stage import companies_of  # noqa: PLC0415 -- the universe, as the cut reads it
    from news_sentiment_stage import SentimentFinder  # noqa: PLC0415 -- loads torch

    names = {c.ticker: c.name for c in companies_of(conn)}
    if not names:
        sys.exit(
            "refused: the company universe is empty (give --source-db, a store with 'ok' articles): every item would be skipped as "
            "'company not in the universe'"
        )
    folder = news_sentiment_stage.model_dir()
    if not folder.is_dir():
        sys.exit(
            f"{folder} is not a folder: the pair model of the step is local (git-ignored) and is never downloaded; restore it "
            "(its Hub copy is recorded in config/sentiment_step.json)"
        )
    try:
        finder = SentimentFinder(
            args.summary_revision,
            args.companies_revision,
            news_sentiment_stage.names_identity(conn, args.companies_revision, names),
        )
    except ValueError as exc:  # the folder lacks a model file: say so, not a stack
        sys.exit(str(exc))
    pinned = news_sentiment_stage.pin()["identity"]
    if finder.model_identity != pinned:
        sys.exit(
            f"refused: {folder} has the identity {finder.model_identity}, not the pinned {pinned} (config/sentiment_step.json). "
            "Restore the pinned folder, or change the pin on purpose: it is another revision of the step"
        )
    print(
        f"sentiment model {finder.model_name} (local {folder.name}, identity {finder.model_identity}) on {finder.scorer.device}, revision {finder.revision}",
        file=sys.stderr,
    )
    try:
        counts = run_sentiment(
            conn, finder, names, args.summary_revision, args.companies_revision, args.limit
        )
    except RunAborted as exc:
        print(json.dumps(steps.status(conn, steps.STEP_SENTIMENT, finder.revision), indent=2))
        sys.exit(f"stopped: {exc}")
    read = {
        "summary_revision": args.summary_revision,
        "companies_revision": args.companies_revision,
    }
    print(json.dumps({**read, **counts}, indent=2))
    print(json.dumps(steps.status(conn, steps.STEP_SENTIMENT, finder.revision), indent=2))


def main_snapshot(args: argparse.Namespace, conn: db.NewsNlpDatabase) -> None:
    """Step 8 (T-195): join the texts, categories and sentiments into ``news_snapshot``. No model; the state is derived from the tables."""
    import news_snapshot_stage as snap  # noqa: PLC0415

    summary, companies, cut = args.summary_revision, args.companies_revision, args.cut_revision
    category, sentiment = args.category_revision, args.sentiment_revision
    sector_rows = (
        snap.sector_revision_of(
            snap.revision(summary, companies, cut, category, sentiment), args.sector_revision
        )
        if all((summary, companies, cut, category, sentiment)) and args.sector_revision
        else None
    )
    revision = (
        snap.revision(summary, companies, cut, category, sentiment)
        if all((summary, companies, cut, category, sentiment))
        else None
    )
    if args.status:
        print(
            json.dumps(
                snap.status(
                    conn,
                    revision,
                    summary_revision=summary,
                    companies_revision=companies,
                    category_revision=category,
                    sentiment_revision=sentiment,
                    sector_rows_revision=sector_rows,
                ),
                indent=2,
            )
        )
        return
    if (
        revision is None
    ):  # check_options asks for the cut, category and sentiment revisions; these two are checked here
        sys.exit("--step snapshot needs --summary-revision and --companies-revision")
    problems = snap.check_inputs(
        conn, summary, companies, cut, category, sentiment, args.sector_revision
    )
    if problems:
        sys.exit("refused: " + "; ".join(problems))
    print(f"snapshot revision {revision}", file=sys.stderr)
    sector = (sector_rows, args.sector_revision) if sector_rows and args.sector_revision else None
    result = {
        "revision": revision,
        **snap.run(conn, revision, summary, companies, category, sentiment, args.limit, sector),
    }
    if sector:
        result["sector_revision"] = sector_rows
    print(json.dumps(result, indent=2))
    print(
        json.dumps(
            snap.status(
                conn,
                revision,
                summary_revision=args.summary_revision,
                companies_revision=args.companies_revision,
                category_revision=args.category_revision,
                sentiment_revision=args.sentiment_revision,
                sector_rows_revision=sector_rows,
                waiting=result["waiting_by_reason"],
            ),
            indent=2,
        )
    )


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
    "(SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');",
    "DELETE FROM pipeline_step_state WHERE step = 'summary' AND item_kind = 'sub_news' AND item_id IN "
    "(SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');",
    # step 3 keys its rows by the same ids
    "DELETE FROM news_company WHERE item_kind = 'sub_news' AND item_id IN "
    "(SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');",
    "DELETE FROM pipeline_step_state WHERE step = 'companies' AND item_kind = 'sub_news' AND item_id IN "
    "(SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');",
    # step 7 too
    "DELETE FROM news_category WHERE item_kind = 'sub_news' AND item_id IN "
    "(SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');",
    "DELETE FROM pipeline_step_state WHERE step = 'category' AND item_kind = 'sub_news' AND item_id IN "
    "(SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');",
    # and step 6
    "DELETE FROM news_sentiment WHERE item_kind = 'sub_news' AND item_id IN "
    "(SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');",
    "DELETE FROM pipeline_step_state WHERE step = 'sentiment' AND item_kind = 'sub_news' AND item_id IN "
    "(SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');",
    # and step 8: a snapshot of a sub-news is keyed by the same id
    "DELETE FROM news_snapshot WHERE item_kind = 'sub_news' AND item_id IN "
    "(SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');",
)
#: To run the companies step again at one revision (the items on the review list included: they are not retried by themselves).
RECOMPANIES_SQL = (
    "DELETE FROM pipeline_step_state WHERE step = 'companies' AND model_revision = '<companies revision>';",
    "DELETE FROM news_company WHERE model_revision = '<companies revision>';",
)
#: To run the category step again at one revision (the rows are written with INSERT OR IGNORE, the state is keyed by revision).
RECATEGORY_SQL = (
    "DELETE FROM pipeline_step_state WHERE step = 'category' AND model_revision = '<category revision>';",
    "DELETE FROM news_category WHERE model_revision = '<category revision>';",
)
#: To run the sentiment step again at one revision (the rows are written with INSERT OR IGNORE, the state is keyed by revision).
RESENTIMENT_SQL = (
    "DELETE FROM pipeline_step_state WHERE step = 'sentiment' AND model_revision = '<sentiment revision>';",
    "DELETE FROM news_sentiment WHERE model_revision = '<sentiment revision>';",
)
#: To write the snapshots again at one revision (the rows are written with INSERT OR IGNORE; the state is derived from the rows, so there is none to delete).
RESNAPSHOT_SQL = ("DELETE FROM news_snapshot WHERE model_revision = '<snapshot revision>';",)
#: To summarise again at one revision of the summary step (the state is keyed by revision, the rows are written with INSERT OR IGNORE).
RESUMMARIZE_SQL = (
    "DELETE FROM pipeline_step_state WHERE step = 'summary' AND model_revision = '<summary revision>';",
    "DELETE FROM news_summary WHERE model_revision = '<summary revision>';",
)
_RERUN_TEMPLATE = """\
To run a step again on bodies it already did: a finished body is found in `pipeline_step_state`, not in the result table,
so `DELETE FROM sub_news` alone does not make the cut run again. At the same revision (the code, the model and the reading
are the same), in the RESULTS database:
  DELETE FROM pipeline_step_state WHERE step = 'cut' AND model_revision = '<cut revision>';
  DELETE FROM sub_news WHERE extractor_revision = '<cut revision>';   -- the rows are written with INSERT OR IGNORE
Before the `DELETE FROM sub_news` above, delete that revision's summaries, companies and categories: steps 2, 3 and 7 read `sub_news` and key their rows by `sub_news.id`, so new ids would leave them pointing at nothing:
  {recut}
To summarise again at one revision of the summary step (`--status` prints it):
  {resummarize}
To run the companies step again at one revision (`--step companies --status` prints it); an item on the review list is not retried by a plain rerun:
  {recompanies}
To run the category step again at one revision (`--step category --status` prints it):
  {recategory}
To run the sentiment step again at one revision (`--step sentiment --status` prints it):
  {resentiment}
To write the snapshots again at one revision (`--step snapshot` prints it; sector rows have their own, printed as `sector_revision`):
  {resnapshot}
A new revision (another reading, model or code) needs neither: it starts new rows and leaves the old ones.
`--status` prints the revisions (the latest first). The recipe: docs/scope_training.md, section 8.1.
"""
RERUN_HELP = _RERUN_TEMPLATE.format(
    recut="\n  ".join(RECUT_SQL),
    resummarize="\n  ".join(RESUMMARIZE_SQL),
    recompanies="\n  ".join(RECOMPANIES_SQL),
    recategory="\n  ".join(RECATEGORY_SQL),
    resentiment="\n  ".join(RESENTIMENT_SQL),
    resnapshot="\n  ".join(RESNAPSHOT_SQL),
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
        help="touch at most this many items (bodies; for summary, companies and category also sub-news)",
    )
    parser.add_argument("--status", action="store_true", help="print the counts and exit")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download the pinned model (scope: config/scope_model.json; cut: config/attribution_model.json, private, needs HF_TOKEN) into an empty --model-dir first",
    )
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="scope or cut: run on a model that is not the pinned one (a refit, a development copy); the rows are not traceable to the Hub",
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
        help="cut and summary: the classifier revision whose multi bodies are cut / whose single bodies are summarised; sector: whose sector bodies get a sector (default the latest)",
    )
    parser.add_argument(
        "--reading",
        choices=("units", "units4", "units5"),
        default=None,
        help="cut: how the body is read (default `units5`, T-177; `units4` is the default of 2026-11-11, `units` the cut as it was before)",
    )
    parser.add_argument(
        "--items",
        choices=("single", "sub_news", "sector"),
        default=None,
        help="summary: only the single bodies, only the sub-news or only the sector bodies the sector step assigned (default single and sub-news)",
    )
    parser.add_argument(
        "--cut-revision",
        default=None,
        help="summary: the cut revision whose sub-news are read; required with sub-news (the revisions the cut has are listed), never the latest",
    )
    parser.add_argument(
        "--summary-revision",
        default=None,
        help="companies, category and sentiment: the summary revision whose texts are read (required to run; `--step summary --status` lists them), never the latest",
    )
    parser.add_argument(
        "--companies-revision",
        default=None,
        help="category and sentiment: the companies revision whose main companies are classified (required to run; `--step companies --status` lists them), never the latest",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="companies: a company whose relevance score is at this value or over it is main, 0 < t <= 1 (required to run: it is part of the revision, D39 (vi))",
    )
    parser.add_argument(
        "--candidates",
        choices=("ner", "ner+text"),
        default=None,
        help="companies: where the candidate companies of a text come from, `ner` (the NER spans, D39 (3.1)) or `ner+text` (also the matcher over the text); required to run: it is part of the revision",
    )
    parser.add_argument(
        "--sector-revision",
        default=None,
        help="summary, with --items sector: the sector revision whose assigned bodies are summarised; required, never the latest (the revisions the sector step has are listed)",
    )
    parser.add_argument(
        "--category-revision",
        default=None,
        help="snapshot: the category revision whose labels are joined (required to run, never the latest)",
    )
    parser.add_argument(
        "--sentiment-revision",
        default=None,
        help="snapshot: the sentiment revision whose labels are joined (required to run, never the latest)",
    )
    parser.add_argument("--source-db", default=None)
    parser.add_argument("--results-db", default=None)
    return parser


def _check_companies_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """The options of the companies step: the two it needs to run, and the ones no other step reads."""
    if (
        args.step
        not in (
            steps.STEP_COMPANIES,
            steps.STEP_CATEGORY,
            steps.STEP_SENTIMENT,
            steps.STEP_SNAPSHOT,
        )
        and args.summary_revision is not None
    ):
        parser.error(
            "--summary-revision only applies to --step companies, --step category, --step sentiment and --step snapshot"
        )
    if args.step != steps.STEP_COMPANIES and args.threshold is not None:
        parser.error("--threshold only applies to --step companies")
    if args.step != steps.STEP_COMPANIES and args.candidates is not None:
        parser.error("--candidates only applies to --step companies")
    if (
        args.step not in (steps.STEP_CATEGORY, steps.STEP_SENTIMENT, steps.STEP_SNAPSHOT)
        and args.companies_revision is not None
    ):
        parser.error(
            "--companies-revision only applies to --step category, --step sentiment and --step snapshot"
        )
    if args.step == steps.STEP_COMPANIES and not args.status:
        if args.summary_revision is None:
            parser.error(
                "--step companies needs --summary-revision (the texts it reads; `--step summary --status` lists the revisions)"
            )
        if args.threshold is None:
            parser.error(
                "--step companies needs --threshold: the value is the owner's (D39 (i), T-169) and part of the revision"
            )
        if args.candidates is None:
            parser.error(
                "--step companies needs --candidates: `ner` or `ner+text` (docs/companies_step.md); it is part of the revision"
            )
        if not 0 < args.threshold <= 1:
            parser.error("--threshold must be over 0 and at most 1")


def _check_category_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """The options of the category and the sentiment steps: the two they need to run, and the ones they do not read."""
    if args.step not in (steps.STEP_CATEGORY, steps.STEP_SENTIMENT, steps.STEP_SNAPSHOT):
        return
    if args.scope_revision is not None:
        parser.error(
            f"--scope-revision does not apply to --step {args.step} (the items are those with a main company at --companies-revision)"
        )
    if not args.status:
        if args.summary_revision is None:
            parser.error(
                f"--step {args.step} needs --summary-revision (the texts it reads; `--step summary --status` lists the revisions)"
            )
        if args.companies_revision is None:
            parser.error(
                f"--step {args.step} needs --companies-revision (the main companies it reads; `--step companies --status` lists the revisions)"
            )


def _check_snapshot_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """The options of the snapshot step (step 8): the five revisions it joins to run; the two it alone reads are refused elsewhere."""
    if args.step != steps.STEP_SNAPSHOT:
        if args.category_revision is not None or args.sentiment_revision is not None:
            parser.error(
                "--category-revision and --sentiment-revision only apply to --step snapshot"
            )
        return
    if args.items is not None or args.threshold is not None or args.candidates is not None:
        parser.error("--items, --threshold and --candidates do not apply to --step snapshot")
    if args.model_dir is not None:
        parser.error("--model-dir does not apply to --step snapshot (it reads no model)")
    if not args.status:
        for flag in ("cut", "category", "sentiment"):
            if getattr(args, f"{flag}_revision") is None:
                parser.error(
                    f"--step snapshot needs --{flag}-revision (never the latest; `--step {flag} --status` lists the revisions)"
                )


def _check_scope_cut_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """``--titles`` is the scope's; ``--fetch`` and ``--allow-unpinned`` are the scope's and the cut's (T-147b, T-194)."""
    if args.step != steps.STEP_SCOPE:
        used = [f"--{n.replace('_', '-')}" for n in SCOPE_ONLY_FLAGS if getattr(args, n)]
        if used:
            parser.error(f"{', '.join(used)} only apply to --step scope")
    if args.step not in (steps.STEP_SCOPE, steps.STEP_CUT):
        used = [f"--{n.replace('_', '-')}" for n in ("fetch", "allow_unpinned") if getattr(args, n)]
        if used:
            parser.error(f"{', '.join(used)} only apply to --step scope and --step cut")


def check_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse an option the chosen step does not read, so it is never silently ignored."""
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    _check_scope_cut_flags(parser, args)
    if args.step != steps.STEP_CUT and args.reading is not None:
        parser.error("--reading only applies to --step cut")
    if args.step not in (steps.STEP_SUMMARY, steps.STEP_COMPANIES, steps.STEP_SNAPSHOT) and (
        args.items is not None or args.cut_revision is not None
    ):
        parser.error(
            "--items and --cut-revision only apply to --step summary, --step companies and --step snapshot (--items not to snapshot)"
        )
    if (
        args.step not in (steps.STEP_SUMMARY, steps.STEP_SNAPSHOT)
        and args.sector_revision is not None
    ):
        parser.error("--sector-revision only applies to --step summary and --step snapshot")
    _check_snapshot_options(parser, args)
    _check_companies_options(parser, args)
    _check_category_options(parser, args)
    if (
        args.step
        in (steps.STEP_SUMMARY, steps.STEP_COMPANIES, steps.STEP_CATEGORY, steps.STEP_SENTIMENT)
        and args.model_dir is not None
    ):
        parser.error(
            f"--model-dir does not apply to --step {args.step} (its models are the pins of config/)"
        )
    if args.step == steps.STEP_SENTIMENT and args.model_dir is not None:
        parser.error(
            "--model-dir does not apply to --step sentiment (its model is the folder of config/sentiment_step.json, never downloaded)"
        )
    if args.step == steps.STEP_SECTOR and args.model_dir is not None:
        parser.error(
            "--model-dir does not apply to --step sector (its model is the pin of config/sector_step.json)"
        )
    _check_summary_items(parser, args)


def _check_summary_items(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """The revisions each kind of summary item reads: an option of another kind is refused, never ignored."""
    if args.items == "sector" and args.step != steps.STEP_SUMMARY:
        parser.error(
            "--items sector only applies to --step summary (the companies step reads single bodies and sub-news)"
        )
    if args.items in ("single", "sector") and args.cut_revision is not None:
        parser.error(f"--cut-revision does not apply to --items {args.items}")
    if args.sector_revision is not None and args.items != "sector":
        parser.error("--sector-revision only applies to --items sector")
    if args.items == "sector" and args.scope_revision is not None:
        parser.error(
            "--scope-revision names the single bodies to read: it does not apply to --items sector (--sector-revision does)"
        )
    if (
        args.step in (steps.STEP_SUMMARY, steps.STEP_COMPANIES)
        and args.items == "sub_news"
        and args.scope_revision is not None
    ):
        parser.error(
            "--scope-revision names the single bodies to read: it does not apply to --items sub_news"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    check_options(parser, args)
    args.model_dir = args.model_dir or default_model_dir(args.step)
    db.env.load_env_file()
    if args.fetch:
        if args.step == steps.STEP_CUT:
            from cut_stage import fetch_pinned  # noqa: PLC0415 -- the download only
        else:
            from scope_stage import fetch_pinned  # noqa: PLC0415

        try:
            identity = fetch_pinned(args.model_dir)
        except (
            OSError,
            RuntimeError,
        ) as exc:  # not empty, no token, no access, a wrong copy: one line, not a stack
            sys.exit(f"--fetch: {exc}")
        print(f"fetched the pinned model into {args.model_dir}: {identity}")
    with exit_on_schema_error():
        conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    db.init_schema(conn)
    runners = {
        steps.STEP_CUT: main_cut,
        steps.STEP_SUMMARY: main_summary,
        steps.STEP_COMPANIES: main_companies,
        steps.STEP_SECTOR: main_sector,
        steps.STEP_CATEGORY: main_category,
        steps.STEP_SENTIMENT: main_sentiment,
        steps.STEP_SNAPSHOT: main_snapshot,
    }
    runners.get(args.step, main_scope)(args, conn)


if __name__ == "__main__":
    main()
