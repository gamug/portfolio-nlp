"""Score the owner's validation of the scope step 1 gave to a sample of `quarantined` bodies (T-147/T-163; owner 2026-11-11).

    uv run python scripts/score_scope_validation_2026_11_11.py [--file data/scope_quarantined_to_validate.csv] \
        [--sizes out_of_scope=146348,single=63051,sector=31182,multi=19775]

Reads the file written by ``draw_scope_quarantined_check_2026_11_11.py`` after the owner filled ``validator`` (``yes`` / ``no``)
and, for a ``no``, ``correct_scope``. Rows with an empty ``validator`` are left out and counted. Prints, per predicted scope, the
bodies read, the share of ``yes`` (the precision of that class on quarantined bodies) with its 95% Wilson interval, what the
``no`` rows should have been, and the share of all quarantined bodies the step tags right (with a rough 95% interval) when the classes are weighted by their
size in the corpus (``--sizes``: the quarantined bodies per predicted scope; the defaults are the ones of the 2026-11-11 run).
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=Path("data/scope_quarantined_to_validate.csv"))
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    args = parser.parse_args()
    with args.file.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    print("\n".join(summarise(rows, parse_sizes(args.sizes))[0]))


if __name__ == "__main__":
    main()
