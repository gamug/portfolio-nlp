"""Summarise some stored items again and compare (step 2, T-166; constitution, Pipeline models 2: inference is deterministic).

    uv run scripts/summary_determinism_check_2026_11_12.py --results-db <nlp.db copy> --source-db <urls.db copy> \
        [--kind body|sub_news] [-n 6] [--revision summary-...]

The idempotency of the summary step rests on one property: the same text, summarised with the same revision, gives the same summary. A GPU can flip a
beam near a tie between runs or between devices, so this is measured, not assumed. For the first ``-n`` items of ``--kind`` that the step stored as a
**summary** at the revision (default the latest with state) it summarises the same text twice in this process and reports how many summaries differ
(a) from the stored ones, which another process wrote (and, on another device, another machine), and (b) between the two runs. Exit code 1 if any
differs. Loads the pinned model (a GPU is advised; the CPU works). Reads the databases only: pass copies.

**The stored rows must be at the revision of this code.** The revision hashes the code that summarises (``news_summary_stage.hashed_sources``): rows written by
other code are not a like-for-like reference. A mismatch is an outcome of its own: the check prints both revisions and **exits 2**, which is not a pass;
``--allow-other-revision`` goes on anyway (the result says so) for a deliberate comparison across versions, and then exits 0 or 1 by the differences.

**It must compare every text it was asked for.** A stored summary whose text is not in SOURCE (a source copy without the bodies, a deleted one) is not skipped: fewer texts than
stored, or none, is **exit 2** with the count found (``found 1 of 2``), never a pass. Exit codes: 0 clean, 1 a summary differs, 2 not comparable.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def compare(
    summarize: Callable[[Sequence[str]], list[tuple[str, int]]],
    items: Sequence[tuple[str, str]],
    stored: Mapping[str, str],
) -> dict[str, int]:
    """``items`` are (id, text); count the summaries that differ from ``stored`` and between two runs of ``summarize``."""
    texts = [text for _, text in items]
    first = [summary for summary, _ in summarize(texts)]
    second = [summary for summary, _ in summarize(texts)]
    return {
        "items": len(items),
        "differ_from_stored": sum(
            1 for (key, _), s in zip(items, first, strict=True) if s != stored[key]
        ),
        "differ_between_two_runs": sum(1 for a, b in zip(first, second, strict=True) if a != b),
    }


def revision_problem(stored: str, current: str) -> str | None:
    """None when the stored rows are at the revision of this code, else why the comparison is not like for like."""
    if stored == current:
        return None
    return f"the stored rows are at {stored} but this code is {current}: not like for like"


def coverage_problem(requested: int, found: int) -> str | None:
    """None when every stored summary asked for had its text in SOURCE; else how many were found. A check that compared fewer texts
    than it was asked to (a source copy without the bodies, a deleted one) says nothing about the rest, and one that compared none says nothing."""
    if found >= requested and found > 0:
        return None
    return f"found {found} of {requested} texts in SOURCE: not everything asked for was compared"


def exit_code(
    result: Mapping[str, int],
    *,
    same_revision: bool,
    allow_other_revision: bool,
    requested: int | None = None,
) -> int:
    """0 clean, 1 a summary differs, 2 not comparable and never a pass: the revisions differ and that was not asked for, or fewer texts
    than ``requested`` were found (none included)."""
    if not same_revision and not allow_other_revision:
        return 2
    if requested is not None and coverage_problem(requested, result["items"]) is not None:
        return 2
    return 1 if result["differ_from_stored"] or result["differ_between_two_runs"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--kind", choices=("body", "sub_news"), default="body")
    parser.add_argument("-n", type=int, default=6)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--allow-other-revision",
        action="store_true",
        help="compare although the stored rows are at another revision than this code (printed in the result)",
    )
    args = parser.parse_args()
    import news_nlp as db  # noqa: PLC0415 -- after the path set-up

    db.env.load_env_file()
    from news_nlp import steps, triage  # noqa: PLC0415
    from news_summary_stage import Summarizer, input_of_body  # noqa: PLC0415 -- loads torch

    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    revision = args.revision or steps.latest_revision(conn, steps.STEP_SUMMARY)
    if revision is None:
        sys.exit("the summary step has not run: nothing to compare")
    stored = {
        str(r[0]): str(r[1])
        for r in conn.execute(
            "SELECT item_id, summary_text FROM news_summary "
            "WHERE item_kind = ? AND model_revision = ? AND text_source = 'summary' ORDER BY rowid LIMIT ?",
            (args.kind, revision, args.n),
        )
    }
    if not stored:
        sys.exit(f"no stored summaries of kind {args.kind} at {revision}")
    if args.kind == "sub_news":
        texts = {
            str(r[0]): str(r[1])
            for r in conn.execute(
                f"SELECT id, text FROM sub_news WHERE id IN ({', '.join('?' for _ in stored)})",  # noqa: S608 -- placeholders only
                [int(k) for k in stored],
            )
        }
    else:
        texts = {}
        marks = ", ".join("?" for _ in steps.PROPER_STATUSES)
        for title, text in conn.execute(
            f"SELECT title, body_text FROM {conn.articles_rel}.articles "  # noqa: S608 -- the schema name is main or source, never input
            f"WHERE fetch_status IN ({marks}) AND body_text IS NOT NULL AND body_text != '' ORDER BY id",
            steps.PROPER_STATUSES,
        ):
            key = triage.body_hash(text)
            if key in stored and key not in texts:
                texts[key] = input_of_body(title or "", text)
                if len(texts) == len(stored):
                    break
    summarizer = Summarizer()
    problem = revision_problem(revision, summarizer.revision)
    if problem and not args.allow_other_revision:
        # sys.exit(<message>) would exit 1, which reads as "a summary differs": a refusal is its own code
        print(
            f"{problem} (--allow-other-revision to compare anyway; exit 2 is not a pass)",
            file=sys.stderr,
        )
        sys.exit(
            exit_code(
                {"differ_from_stored": 0, "differ_between_two_runs": 0},
                same_revision=False,
                allow_other_revision=False,
            )
        )
    print(f"summariser on {summarizer.device}")
    items = [(key, texts[key]) for key in stored if key in texts]
    gap = coverage_problem(len(stored), len(items))
    # before the model runs: a source without the texts is a wrong --source-db, not a result
    if gap:
        print(f"{gap} (check --source-db; exit 2 is not a pass)", file=sys.stderr)
        sys.exit(2)
    result = compare(summarizer.summarize, items, stored)
    print(f"revision {revision}: {result}" + (f"; {problem}" if problem else ""))
    sys.exit(
        exit_code(
            result,
            same_revision=problem is None,
            allow_other_revision=args.allow_other_revision,
            requested=len(stored),
        )
    )


if __name__ == "__main__":
    main()
