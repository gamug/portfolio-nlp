"""Cut the same bodies again and compare with the rows already stored (step 1.2, T-163; manager's review of #168).

    uv run scripts/cut_determinism_check_2026_11_11.py --results-db <nlp.db copy> --source-db <urls.db copy> \
        [--model-dir data/models/attribution_2026_11_08] [--revision units-...] [-n 30]

The idempotency of the cut rests on one property: the same body, cut with the same revision, gives the same texts. A GPU scorer
can flip a score near the threshold between runs, so this is measured, not assumed. For the first ``-n`` bodies with stored
``sub_news`` at the revision (default: the latest with state) it reports how many differ (a) from the stored rows, which another
process wrote, and (b) between two cuts in this process. Exit code 1 if any differs. Needs the checkpoint and a GPU; reads the
results file only (pass copies of the databases: the owner's mount fails a read-only open).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def compare(
    cut: Callable[[str], list[str]],
    bodies: Sequence[tuple[str, str]],
    stored: Mapping[str, list[str]],
) -> dict[str, int]:
    """``bodies`` are (hash, text); count the bodies whose cut differs from the stored rows and between two cuts."""
    from_stored = between = 0
    for body_hash, text in bodies:
        first = cut(text)
        from_stored += first != stored[body_hash]
        between += first != cut(text)
    return {
        "bodies": len(bodies),
        "differ_from_stored": from_stored,
        "differ_between_two_runs": between,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/models/attribution_2026_11_08",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("-n", type=int, default=30)
    args = parser.parse_args()
    import news_nlp as db  # noqa: PLC0415 -- after the path set-up

    db.env.load_env_file()
    from cut_stage import UnitsExtractor, companies_of  # noqa: PLC0415 -- loads torch
    from news_nlp import steps  # noqa: PLC0415

    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    extractor = UnitsExtractor.load(args.model_dir, companies_of(conn))
    revision = args.revision or extractor.revision
    if revision != extractor.revision:
        sys.exit(
            f"the loaded model gives revision {extractor.revision}, not {revision}: the comparison would mean nothing"
        )
    stored: dict[str, list[str]] = {}
    for body_hash, text in conn.execute(
        "SELECT body_hash, text FROM sub_news WHERE extractor_revision = ? ORDER BY body_hash, segment_index",
        (revision,),
    ):
        stored.setdefault(str(body_hash), []).append(str(text))
    scope = steps.latest_classifier_revision(conn)
    if scope is None or not stored:
        sys.exit(f"no sub_news at revision {revision} in this file: run `--step cut` first")
    todo = [
        (b.body_hash, b.text)
        for b in steps.pending_multi_bodies(conn, scope, "units-none", None)
        if b.body_hash in stored
    ]
    result = compare(extractor.cut, todo[: args.n], stored)
    print(f"revision {revision}: {result}")
    sys.exit(0 if not result["differ_from_stored"] and not result["differ_between_two_runs"] else 1)


if __name__ == "__main__":
    main()
