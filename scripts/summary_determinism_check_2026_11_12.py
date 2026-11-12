"""Summarise some stored items again and compare (step 2, T-166; constitution, Pipeline models 2: inference is deterministic).

    uv run scripts/summary_determinism_check_2026_11_12.py --results-db <nlp.db copy> --source-db <urls.db copy> \
        [--kind body|sub_news] [-n 6] [--revision summary-...]

The idempotency of the summary step rests on one property: the same text, summarised with the same revision, gives the same summary. A GPU can flip a
beam near a tie between runs or between devices, so this is measured, not assumed. For the first ``-n`` items of ``--kind`` that the step stored as a
**summary** at the revision (default the latest with state) it summarises the same text twice in this process and reports how many summaries differ
(a) from the stored ones, which another process wrote (and, on another device, another machine), and (b) between the two runs. Exit code 1 if any
differs. Loads the pinned model (a GPU is advised; the CPU works). Reads the databases only: pass copies.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--kind", choices=("body", "sub_news"), default="body")
    parser.add_argument("-n", type=int, default=6)
    parser.add_argument("--revision", default=None)
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
    if summarizer.revision != revision:
        print(
            f"warning: the stored rows are at {revision}, this code is {summarizer.revision}: not like for like",
            file=sys.stderr,
        )
    print(f"summariser on {summarizer.device}")
    items = [(key, texts[key]) for key in stored if key in texts]
    result = compare(summarizer.summarize, items, stored)
    print(f"revision {revision}: {result}")
    sys.exit(1 if result["differ_from_stored"] or result["differ_between_two_runs"] else 0)


if __name__ == "__main__":
    main()
