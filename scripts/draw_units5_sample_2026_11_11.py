"""Draw the sample of `units5` texts the owner reads blind, and list which changed texts are read (T-177, review of #172).

    uv run scripts/compare_cut_readings_2026_11_11.py ... --a units4 --b units5 --changed-out changed.csv
    uv run scripts/draw_units5_sample_2026_11_11.py --changed changed.csv --marks gold/cut_owner_marks_2026_11_11.csv \
        [--n 30] [--seed 172] [--sample-out sample.csv] [--list-out gold/units5_changed_texts_2026_11_11.csv]

``changed.csv`` is the output of ``compare_cut_readings`` (kind, body_hash, title, ticker, text of A, text of B). The texts that are
**gained or changed** under B are the pool; those that already have a mark in ``--marks`` (a (ticker, text) pair) are left out; the
rest is shuffled with ``--seed`` and the first ``--n`` distinct (ticker, text) pairs are the sample. The sample file has no
recommendation column: it is for reading blind. ``--list-out`` writes every gained or changed text with ``read`` = ``marked``
(in ``--marks`` before the draw), ``sample`` (in the sample, so read once the owner answers) or ``no`` (not read), so a later reading
can continue from the rest. The seed 172 draw of 2026-11-11 gave the 30 texts of ``gold/cut_owner_marks_2026_11_11.csv`` marked
"blind sample S1..S30". Pure file work: no database and no model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

POOL_KINDS = ("gained", "changed")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def draw(
    rows: list[dict[str, str]], marked: set[tuple[str, str]], n: int, seed: int
) -> list[dict[str, str]]:
    """The first ``n`` distinct (ticker, text) pairs of the pool after a seeded shuffle; the pool is gained and changed texts
    under B that have no mark."""
    pool = [
        r
        for r in rows
        if r["kind"] in POOL_KINDS
        and r["text_units5"].strip()
        and (r["ticker"], r["text_units5"].strip()) not in marked
    ]
    random.Random(seed).shuffle(pool)  # noqa: S311 -- a reproducible draw, not security
    seen: set[tuple[str, str]] = set()
    sample: list[dict[str, str]] = []
    for r in pool:
        key = (r["ticker"], r["text_units5"])
        if key in seen:
            continue
        seen.add(key)
        sample.append(r)
        if len(sample) == n:
            break
    return sample


def status_list(
    rows: list[dict[str, str]], marked: set[tuple[str, str]], sample: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Every gained or changed text with whether it is ``marked``, in the ``sample`` or not read (``no``)."""
    drawn = {(r["ticker"], r["text_units5"]) for r in sample}
    out = []
    for r in rows:
        if r["kind"] not in POOL_KINDS or not r["text_units5"].strip():
            continue
        key = (r["ticker"], r["text_units5"])
        read = (
            "marked"
            if (r["ticker"], r["text_units5"].strip()) in marked
            else "sample"
            if key in drawn
            else "no"
        )
        out.append(
            {
                "kind": r["kind"],
                "body_hash": r["body_hash"],
                "ticker": r["ticker"],
                "text_hash": text_hash(r["text_units5"]),
                "read": read,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--changed", type=Path, required=True)
    parser.add_argument("--marks", type=Path, required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=172)
    parser.add_argument("--sample-out", type=Path, default=None)
    parser.add_argument("--list-out", type=Path, default=None)
    args = parser.parse_args()
    rows = read_rows(args.changed)
    marked = {(r["ticker"], r["text"].strip()) for r in read_rows(args.marks)}
    sample = draw(rows, marked, args.n, args.seed)
    print(
        f"pool {sum(r['kind'] in POOL_KINDS for r in rows)}; sample {len(sample)} (seed {args.seed})"
    )
    if args.sample_out:
        with args.sample_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["n", "kind", "body_hash", "ticker", "text_units4", "text_units5"]
            )
            writer.writeheader()
            for i, r in enumerate(sample, 1):
                writer.writerow(
                    {k: r[k] for k in ("kind", "body_hash", "ticker", "text_units4", "text_units5")}
                    | {"n": str(i)}
                )
    if args.list_out:
        listing = status_list(rows, marked, sample)
        with args.list_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["kind", "body_hash", "ticker", "text_hash", "read"]
            )
            writer.writeheader()
            writer.writerows(listing)
        from collections import Counter  # noqa: PLC0415

        print(dict(Counter(r["read"] for r in listing)))


if __name__ == "__main__":
    main()
