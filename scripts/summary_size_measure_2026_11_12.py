"""The measure behind the size of the summary step (T-166; owner's choice 2,000 characters, 2026-11-12): how many items each size would send to the summariser.

    uv run scripts/summary_size_measure_2026_11_12.py --results-db <nlp.db copy> --source-db <urls.db copy> \
        [--scope-revision REV] [--cut-revision REV] [--first-bodies N] [--seconds-per-body 0.84]

Read-only. For the ``single`` bodies of step 1 (at ``--scope-revision``, default the latest) it takes the text the summariser would read
(``Title-<title>\\nBody-<body>``, ``news_summary_stage.input_of_body``), and for the sub-news of ``--cut-revision`` (named, never the latest; skipped
if not given) their own text. It prints the length percentiles and, for a list of sizes, how many items are over each and the GPU hours that many
bodies would take at ``--seconds-per-body`` (the earlier summary run's 0.84 s a body: an extrapolation, not a measure of this model on this machine).
The figures of ``docs/summary_step.md`` are this output on the 93,034 single bodies of the corpus run and the 3,134 sub-news of the first 500 multi
bodies (``units4``). With ``--first-bodies N`` the sub-news of the cut are also measured on their first N bodies (by article id), next to the whole
cut: the table of the doc is such a sample, and it read 1.6% over 2,000 where the whole ``units`` cut has 3.05%. Pass copies of the databases (the owner's mount fails a read-only open).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SIZES = (500, 1000, 1500, 2000, 3000, 4000, 6000, 10000)
PERCENTILES = (10, 25, 50, 75, 90, 95, 99)


def percentiles(lengths: Sequence[int], qs: Sequence[int] = PERCENTILES) -> dict[int, int]:
    """The length at each percentile (the value at index ``n * q / 100`` of the sorted lengths)."""
    ordered = sorted(lengths)
    return {q: ordered[min(len(ordered) - 1, int(len(ordered) * q / 100))] for q in qs}


def over_table(
    lengths: Sequence[int], sizes: Sequence[int] = SIZES
) -> list[tuple[int, int, float]]:
    """For each size: how many lengths are over it (the size itself is not) and their share."""
    table = []
    for size in sizes:
        over = sum(1 for n in lengths if n > size)
        table.append((size, over, over / len(lengths) if lengths else 0.0))
    return table


def hours(items: int, seconds_per_item: float) -> float:
    return items * seconds_per_item / 3600


def sub_news_lengths(conn: Any, revision: str, first_bodies: int | None = None) -> list[int]:
    """The lengths of the sub-news of a cut; with ``first_bodies`` only those of its first N bodies (by article id, among the bodies the cut wrote sub-news for)."""
    if first_bodies is None:
        rows = conn.execute("SELECT text FROM sub_news WHERE extractor_revision = ?", (revision,))
    else:
        rows = conn.execute(
            "SELECT text FROM sub_news WHERE extractor_revision = ? AND article_id IN "
            "(SELECT DISTINCT article_id FROM sub_news WHERE extractor_revision = ? ORDER BY article_id LIMIT ?)",
            (revision, revision, first_bodies),
        )
    return [len(str(r[0])) for r in rows]


def report(name: str, lengths: Sequence[int], seconds_per_body: float | None) -> None:
    if not lengths:
        print(f"\n{name}: none")
        return
    print(
        f"\n{name}: n={len(lengths)} min={min(lengths)} mean={sum(lengths) // len(lengths)} max={max(lengths)}"
    )
    print("  percentiles (characters):", percentiles(lengths))
    for size, over, share in over_table(lengths):
        gpu = (
            f"  ~{hours(over, seconds_per_body):.1f} h at {seconds_per_body} s"
            if seconds_per_body
            else ""
        )
        print(f"  over {size:>6}: {over:>7} ({share:.1%}){gpu}")


def positive_int(text: str) -> int:
    """An argparse type for a count of bodies: a whole number of 1 or more (SQLite reads ``LIMIT -1`` as no limit at all)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is not 1 or more")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--scope-revision", default=None)
    parser.add_argument("--cut-revision", default=None)
    parser.add_argument("--first-bodies", type=positive_int, default=None)
    parser.add_argument("--seconds-per-body", type=float, default=0.84)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import news_nlp as db  # noqa: PLC0415 -- after the path set-up

    db.env.load_env_file()
    from news_nlp import steps, triage  # noqa: PLC0415
    from news_summary_stage import input_of_body  # noqa: PLC0415 -- loads torch

    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    scope = args.scope_revision or steps.latest_classifier_revision(conn)
    if scope is None:
        sys.exit("body_scope is empty: run --step scope first")
    wanted = {
        str(r[0])
        for r in conn.execute(
            "SELECT body_hash FROM body_scope WHERE classifier_revision = ? AND scope = 'single'",
            (scope,),
        )
    }
    marks = ", ".join("?" for _ in steps.PROPER_STATUSES)
    seen: set[str] = set()
    single: list[int] = []
    for title, text in conn.execute(
        f"SELECT title, body_text FROM {conn.articles_rel}.articles "  # noqa: S608 -- the schema name is main or source, never input
        f"WHERE fetch_status IN ({marks}) AND body_text IS NOT NULL AND body_text != '' ORDER BY id",
        steps.PROPER_STATUSES,
    ):
        key = triage.body_hash(text)
        if key in wanted and key not in seen:
            seen.add(key)
            single.append(len(input_of_body(title or "", text)))
            if len(seen) == len(wanted):
                break
    print(f"scope revision {scope}: {len(wanted)} single bodies, {len(single)} found in SOURCE")
    report("single bodies (Title-/Body- input)", single, args.seconds_per_body)
    if args.cut_revision:
        subs = sub_news_lengths(conn, args.cut_revision)
        report(f"sub-news (cut {args.cut_revision})", subs, None)
        if args.first_bodies is not None:
            sample = sub_news_lengths(conn, args.cut_revision, args.first_bodies)
            report(
                f"sub-news (cut {args.cut_revision}, its first {args.first_bodies} bodies)",
                sample,
                None,
            )
    else:
        print("\nno --cut-revision: the sub-news are not measured")


if __name__ == "__main__":
    main()
