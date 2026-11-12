"""The figures of the cut sample the owner marked (T-183; owner 2026-11-12).

    uv run scripts/cut_sample_metrics_2026_11_12.py [--marked data/cut_sample_to_tag.csv] [--ids data/cut_sample_ids.csv] \
        [--seed 20261112] [--draws 2000] [--out docs/cut_sample_metrics_2026-11-12.md]

Reads the file the owner marked (no database). It stops when a mark is not yes / no, or when a row's text is not the one the ids file
ties to that ``review_order`` (a mark belongs to a text). Three levels, all printed:

- ``all``: every piece of the sample;
- ``right_tag``: without the pieces whose note says the tagging, not the cut, is wrong: the company was read from the title
  (``wrong_tag``: "company missunderstood", "company is lost") or the body was scoped wrongly (the note says ``investment_general`` or
  "misslabeled");
- ``right_tag_unambiguous``: also without the pieces whose note says the body mixes companies or stories so that no cut reads well
  (``ambiguous``: "mix", "two companies", "two different stories", "market_economy", "can understand"). This level is the most
  favourable reading of the cut, not a figure to quote alone: the ambiguous bodies are real input of the cut.

A note with both kinds counts as ``wrong_tag`` (the cause). The kind is read from the owner's free text by the words above, and
the report lists every excluded row with its note, so the call can be checked row by row. Per question: the share of yes with a Wilson 95%
interval over pieces, and the interval of a bootstrap over bodies (the pieces of a body are not independent: the sample is of bodies). ``clean``
is a piece with yes on the three questions; a clean body has only clean pieces. The bar of the extractor was >= 80% faithful (D32).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

QUESTIONS = ("faithful", "one_story", "complete")
BAR = 0.80
WRONG_TAG_WORDS = (
    "company missunderstood",
    "company is lost",
    "tagged from the title",
    "investment_general",
    "misslabeled",
    "mislabeled",
)
AMBIGUOUS_WORDS = (
    "mix",
    "two companies",
    "two different stories",
    "market_economy",
    "can understand",
)
Row = dict[str, str]


def text_hash(text: str) -> str:
    """SHA-256 of the piece, outer whitespace stripped (the draw's own rule)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def kind_of(note: str) -> str:
    """``wrong_tag``, ``ambiguous`` or ``kept``: what the owner's note says about the row (the wrong tag wins)."""
    low = note.lower()
    if any(w in low for w in WRONG_TAG_WORDS):
        return "wrong_tag"
    if any(w in low for w in AMBIGUOUS_WORDS):
        return "ambiguous"
    return "kept"


def wilson(yes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """The share and its Wilson interval (never outside 0..1)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = yes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _yes(row: Row, q: str) -> bool:
    return row[q].strip().lower() == "yes"


def _clean(row: Row) -> bool:
    return all(_yes(row, q) for q in QUESTIONS)


@dataclass(frozen=True)
class Level:
    name: str
    pieces: int
    bodies: int
    yes: dict[str, int]
    clean_pieces: int
    clean_bodies: int
    both: int  # faithful and one_story


def _level(name: str, rows: list[Row]) -> Level:
    bodies: dict[str, list[Row]] = {}
    for r in rows:
        bodies.setdefault(r["article_id"], []).append(r)
    return Level(
        name=name,
        pieces=len(rows),
        bodies=len(bodies),
        yes={q: sum(_yes(r, q) for r in rows) for q in QUESTIONS},
        clean_pieces=sum(_clean(r) for r in rows),
        clean_bodies=sum(all(_clean(r) for r in rs) for rs in bodies.values()),
        both=sum(_yes(r, "faithful") and _yes(r, "one_story") for r in rows),
    )


def levels(rows: list[Row]) -> list[Level]:
    right = [r for r in rows if kind_of(r["notes"]) != "wrong_tag"]
    unamb = [r for r in right if kind_of(r["notes"]) != "ambiguous"]
    return [_level("all", rows), _level("right_tag", right), _level("right_tag_unambiguous", unamb)]


def body_bootstrap(rows: list[Row], what: str, seed: int, draws: int = 2000) -> tuple[float, float]:
    """95% interval of the share of yes (``what`` is a question or ``clean``), resampling whole bodies."""
    bodies: dict[str, list[Row]] = {}
    for r in rows:
        bodies.setdefault(r["article_id"], []).append(r)
    groups = list(bodies.values())
    if not groups:
        return 0.0, 0.0
    rng = random.Random(seed)  # noqa: S311 -- a seeded resample, not a secret
    shares = []
    for _ in range(draws):
        pick = [g for _ in groups for g in [rng.choice(groups)]]
        n = sum(len(g) for g in pick)
        y = sum((_clean(r) if what == "clean" else _yes(r, what)) for g in pick for r in g)
        shares.append(y / n)
    shares.sort()
    return shares[int(0.025 * (draws - 1))], shares[math.ceil(0.975 * (draws - 1))]


def check_marks(rows: list[Row]) -> None:
    for r in rows:
        for q in QUESTIONS:
            if r[q].strip().lower() not in ("yes", "no"):
                raise SystemExit(f"review_order {r['review_order']}: {q} is {r[q]!r}, not yes / no")


def check_ids(rows: list[Row], ids_path: Path) -> str:
    """The cut revision of the ids file, after checking that every row's text is the one tied to its ``review_order``."""
    with ids_path.open(encoding="utf-8-sig", newline="") as f:
        ids = {r["review_order"]: r for r in csv.DictReader(f)}
    revisions = {r["extractor_revision"] for r in ids.values()}
    if len(revisions) != 1:
        raise SystemExit(f"the ids file names {len(revisions)} cut revisions, expected one")
    for r in rows:
        want = ids.get(r["review_order"])
        if want is None or want["text_hash"] != text_hash(r["text"]):
            raise SystemExit(
                f"review_order {r['review_order']}: the text is not the one the ids file ties to it"
            )
    return revisions.pop()


def _fmt(yes: int, n: int) -> str:
    p, lo, hi = wilson(yes, n)
    return f"{yes}/{n} = {p:.1%} [{lo:.1%}, {hi:.1%}]"


def report(rows: list[Row], revision: str, seed: int, draws: int) -> str:
    out = [f"cut `{revision}`, {len(rows)} pieces marked", ""]
    kinds = [kind_of(r["notes"]) for r in rows]
    out.append(
        f"notes read as: wrong_tag: {kinds.count('wrong_tag')}, ambiguous: {kinds.count('ambiguous')}, kept: {kinds.count('kept')}"
    )
    out.append("")
    for lv, sub in zip(levels(rows), _subsets(rows), strict=True):
        out += [f"## {lv.name}: {lv.pieces} pieces of {lv.bodies} bodies", ""]
        out += ["| measure | pieces (Wilson 95%) | bodies resampled (95%) |", "|---|---|---|"]
        for q in QUESTIONS:
            lo, hi = body_bootstrap(sub, q, seed, draws)
            out.append(f"| {q} yes | {_fmt(lv.yes[q], lv.pieces)} | [{lo:.1%}, {hi:.1%}] |")
        lo, hi = body_bootstrap(sub, "clean", seed, draws)
        out.append(f"| clean (3 yes) | {_fmt(lv.clean_pieces, lv.pieces)} | [{lo:.1%}, {hi:.1%}] |")
        out.append(f"| faithful and one_story | {_fmt(lv.both, lv.pieces)} | |")
        out.append(f"| bodies with every piece clean | {_fmt(lv.clean_bodies, lv.bodies)} | |")
        verdict = "above" if wilson(lv.yes["faithful"], lv.pieces)[1] >= BAR else "below"
        out += [
            "",
            f"faithful against the 80% bar of the extractor: the lower end of its interval is {verdict} the bar.",
            "",
        ]
    out += [
        "## Rows left out of a level, with the owner's note",
        "",
        "| review_order | kind | faithful | one_story | complete | note |",
        "|---|---|---|---|---|---|",
    ]
    for r, k in zip(rows, kinds, strict=True):
        if k != "kept":
            note = r["notes"].replace("|", "/").strip()
            out.append(
                f"| {r['review_order']} | {k} | {r['faithful'].strip()} | {r['one_story'].strip()} | {r['complete'].strip()} | {note} |"
            )
    return "\n".join(out) + "\n"


def _subsets(rows: list[Row]) -> list[list[Row]]:
    right = [r for r in rows if kind_of(r["notes"]) != "wrong_tag"]
    return [rows, right, [r for r in right if kind_of(r["notes"]) != "ambiguous"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--marked", type=Path, default=Path("data/cut_sample_to_tag.csv"))
    ap.add_argument("--ids", type=Path, default=Path("data/cut_sample_ids.csv"))
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None, help="also write the report here")
    args = ap.parse_args(argv)
    with args.marked.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    check_marks(rows)
    revision = check_ids(rows, args.ids)
    text = report(rows, revision, args.seed, args.draws)
    sys.stdout.write(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
