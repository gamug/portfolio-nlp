"""Score the owner's validation of the scope step 1 gave to a sample of `quarantined` bodies (T-147/T-163; owner 2026-11-11).

    uv run python scripts/score_scope_validation_2026_11_11.py [--file gold/scope_quarantined_validation.csv ...] \
        [--draw main|multi_extra|all] [--key final|first] [--sizes out_of_scope=146348,single=63051,sector=31182,multi=19775]

Reads the tags: by default the tracked ``gold/scope_quarantined_validation.csv`` (ids and tags, no text; the working files of
``draw_scope_quarantined_check_2026_11_11.py`` in ``data/`` have the same ``validator`` / ``correct_scope`` columns and work too;
``--file`` may be repeated and the rows are pooled). ``--draw`` picks the draw of the gold file (``main`` = the 200 drawn per predicted
scope, the default and the headline; ``multi_extra`` = the extra ``multi`` bodies; ``all`` = both pooled). ``--key`` picks the tags:
``final`` (after the owner's audits of 2026-11-11) or ``first`` (the first reading, kept in ``validator_first`` / ``correct_first``).
Rows with an empty ``validator`` are left out and counted. Prints, per predicted scope, the
bodies read, the share of ``yes`` (the precision of that class on quarantined bodies) with its 95% Wilson interval, what the
``no`` rows should have been, and the share of all quarantined bodies the step tags right (with a rough 95% interval) when the classes are weighted by their
size in the corpus (``--sizes``: the quarantined bodies per predicted scope; the defaults are the pool of the corpus run of revision
``scope-nli-8ff21e50796b`` on 2026-11-11, 260,356 bodies, and belong to that revision only).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from news_nlp import gold_metrics

SCOPES = ("out_of_scope", "single", "multi", "sector")
DEFAULT_SIZES = "out_of_scope=146348,single=63051,sector=31182,multi=19775"
VALID = {"yes", "no"}


def parse_sizes(text: str) -> dict[str, int]:
    return {k.strip(): int(v) for k, v in (part.split("=") for part in text.split(","))}


def weighted_precision(
    counts: Mapping[str, tuple[int, int]], sizes: Mapping[str, int]
) -> tuple[float, float, float]:
    """The share right over all bodies, the classes weighted by ``sizes``, with a 95% interval.

    ``counts[scope]`` is ``(right, read)``. The interval is a stratified normal approximation
    (variance ``sum w_i^2 p_i (1 - p_i) / (n_i - 1)``), so it is a rough one for classes of 25.
    """
    total = sum(sizes.values())
    point = sum(sizes[s] * counts[s][0] / counts[s][1] for s in sizes) / total
    variance = sum(
        (sizes[s] / total) ** 2
        * (counts[s][0] / counts[s][1])
        * (1 - counts[s][0] / counts[s][1])
        / max(counts[s][1] - 1, 1)
        for s in sizes
    )
    half = 1.96 * variance**0.5
    return point, point - half, point + half


def summarise(
    rows: Sequence[Mapping[str, str]], sizes: Mapping[str, int]
) -> tuple[list[str], dict[str, gold_metrics.Figure]]:
    """The report lines and the precision of every predicted scope that has at least one validated row."""
    by_class: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    skipped = 0
    for r in rows:
        answer = (r.get("validator") or "").strip().lower()
        if not answer:
            skipped += 1
            continue
        if answer not in VALID:
            raise ValueError(
                f"row {r.get('review_order')}: validator must be yes or no, got {answer!r}"
            )
        by_class[r["predicted_scope"]].append(r)
    lines = [f"{sum(map(len, by_class.values()))} bodies validated, {skipped} still empty"]
    figures: dict[str, gold_metrics.Figure] = {}
    counts: dict[str, tuple[int, int]] = {}
    for scope in SCOPES:
        got = by_class.get(scope, [])
        if not got:
            continue
        yes = sum((r["validator"].strip().lower() == "yes") for r in got)
        figures[scope] = gold_metrics.wilson(yes, len(got))
        counts[scope] = (yes, len(got))
        should = Counter(
            (r.get("correct_scope") or "?").strip()
            for r in got
            if r["validator"].strip().lower() == "no"
        )
        lines.append(
            f"{scope:>13}: {len(got):3d} read | right {yes:3d} = {figures[scope]}"
            + (
                f" | the {len(got) - yes} wrong ones should be {dict(should)}"
                if len(got) > yes
                else ""
            )
        )
    if figures and all(s in figures for s in sizes):
        total = sum(sizes.values())
        right, low, high = weighted_precision(counts, sizes)
        lines.append(
            f"weighted by the {total:,} quarantined bodies: the step tags {right:.1%} right "
            f"[{low:.1%}, {high:.1%}] (stratified normal approximation; the per-class intervals above are Wilson)"
        )
    return lines, figures


def load_rows(
    paths: Sequence[Path], draw: str = "main", key: str = "final"
) -> list[dict[str, str]]:
    """The rows of every file, pooled, with the tags of ``key`` in ``validator`` / ``correct_scope``.

    A file with a ``draw`` column (the tracked gold file) is filtered by ``draw`` (``all`` keeps everything); a working file
    from ``data/`` has none and is taken whole. ``key="first"`` needs the ``validator_first`` / ``correct_first`` columns.
    """
    if draw not in {"main", "multi_extra", "all"} or key not in {"final", "first"}:
        raise ValueError(
            f"--draw is main, multi_extra or all and --key is final or first, got {draw!r} / {key!r}"
        )
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("draw") and draw != "all" and row["draw"] != draw:
                    continue
                if key == "first":
                    if "validator_first" not in row:
                        raise ValueError(f"{path}: no validator_first column for --key first")
                    rows.append(
                        {
                            **row,
                            "validator": row["validator_first"],
                            "correct_scope": row["correct_first"],
                        }
                    )
                    continue
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, action="append", default=[])
    parser.add_argument("--draw", default="main", choices=("main", "multi_extra", "all"))
    parser.add_argument("--key", default="final", choices=("final", "first"))
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    args = parser.parse_args()
    files = args.file or [Path("gold/scope_quarantined_validation.csv")]
    rows = load_rows(files, args.draw, args.key)
    print("\n".join(summarise(rows, parse_sizes(args.sizes))[0]))


if __name__ == "__main__":
    main()
