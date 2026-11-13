"""The figures of the cut sample the owner marked (T-183; owner 2026-11-12).

    uv run scripts/cut_sample_metrics_2026_11_12.py [--marks gold/cut_sample_marks.csv] [--texts gold/texts/cut_sample_texts.csv] \
        [--seed 20261112] [--draws 2000] [--out FILE]

Reads the marks the owner gave and ``build_cut_sample_gold_2026_11_12.py`` stored in ``gold/`` (no database). It stops when a mark is not yes / no, when a
``kind`` is unknown, or when a piece's text is not the one its ``text_hash`` ties to (a mark belongs to a text). Three levels, all printed, **the first
being the pre-set measure** (every piece of the sample):

- ``all``: every piece;
- ``right_scope``: without the pieces whose body should have been scoped another way (``wrong_scope``: ``investment_general``, "misslabeled"). That is step
  1.1's error, not the cut's. The company read from the title (``wrong_company``) is **not** left out: the three answers judge the piece against its
  article, so a wrong company tag does not touch them, and the cut's own "no" in such a row counts;
- ``right_scope_unambiguous``: also without the pieces whose body mixes companies or stories so that no cut reads well (``ambiguous``). This level is the most
  favourable reading of the cut, not a figure to quote alone: those bodies are real input of the cut.

The levels other than ``all`` were chosen after the owner saw the misses; the bar of the extractor, >= 80% faithful (D32), is read on ``all`` first. Per
question: the share of yes with a Wilson 95% interval over pieces and the interval of a bootstrap over bodies (the pieces of a body are not independent: the
sample is of bodies). ``clean`` is a piece with yes on the three questions; a clean body has only clean pieces. The report lists every piece whose kind is
not ``kept`` with the owner's note, so each call can be checked row by row.
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
KINDS = ("kept", "wrong_scope", "wrong_company", "ambiguous")
BAR = 0.80
Row = dict[str, str]


def text_hash(text: str) -> str:
    """SHA-256 of the piece, outer whitespace stripped (the draw's own rule)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


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


def _by_body(rows: list[Row]) -> dict[str, list[Row]]:
    bodies: dict[str, list[Row]] = {}
    for r in rows:
        bodies.setdefault(r["article_id"], []).append(r)
    return bodies


def _level(name: str, rows: list[Row]) -> Level:
    bodies = _by_body(rows)
    return Level(
        name=name,
        pieces=len(rows),
        bodies=len(bodies),
        yes={q: sum(_yes(r, q) for r in rows) for q in QUESTIONS},
        clean_pieces=sum(_clean(r) for r in rows),
        clean_bodies=sum(all(_clean(r) for r in rs) for rs in bodies.values()),
        both=sum(_yes(r, "faithful") and _yes(r, "one_story") for r in rows),
    )


def subsets(rows: list[Row]) -> list[tuple[str, list[Row]]]:
    right = [r for r in rows if r["kind"] != "wrong_scope"]
    return [
        ("all", rows),
        ("right_scope", right),
        ("right_scope_unambiguous", [r for r in right if r["kind"] != "ambiguous"]),
    ]


def levels(rows: list[Row]) -> list[Level]:
    return [_level(name, sub) for name, sub in subsets(rows)]


def body_bootstrap(rows: list[Row], what: str, seed: int, draws: int = 2000) -> tuple[float, float]:
    """95% interval of the share of yes (``what`` is a question or ``clean``), resampling whole bodies."""
    groups = list(_by_body(rows).values())
    if not groups:
        return 0.0, 0.0
    rng = random.Random(seed)  # noqa: S311 -- a seeded resample, not a secret
    shares = []
    for _ in range(draws):
        pick = [rng.choice(groups) for _ in groups]
        n = sum(len(g) for g in pick)
        y = sum((_clean(r) if what == "clean" else _yes(r, what)) for g in pick for r in g)
        shares.append(y / n)
    shares.sort()
    return shares[int(0.025 * (draws - 1))], shares[math.ceil(0.975 * (draws - 1))]


def check_rows(rows: list[Row]) -> None:
    for r in rows:
        for q in QUESTIONS:
            if r[q].strip().lower() not in ("yes", "no"):
                raise SystemExit(f"review_order {r['review_order']}: {q} is {r[q]!r}, not yes / no")
        if r.get("kind", "kept") not in KINDS:
            raise SystemExit(
                f"review_order {r['review_order']}: the kind {r['kind']!r} is not one of {', '.join(KINDS)}"
            )


def _read(path: Path) -> list[Row]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load(marks_path: Path, texts_path: Path) -> tuple[list[Row], str]:
    """The marked rows with their text, and the one cut revision they belong to; stops when a text is not the one its hash ties."""
    marks = _read(marks_path)
    texts = {t["review_order"]: t["text"] for t in _read(texts_path)}
    revisions = {m["extractor_revision"] for m in marks}
    if len(revisions) != 1:
        raise SystemExit(f"the marks name {len(revisions)} cut revisions, expected one")
    for m in marks:
        text = texts.get(m["review_order"])
        if text is None or text_hash(text) != m["text_hash"]:
            raise SystemExit(
                f"review_order {m['review_order']}: the text is not the one its mark is tied to"
            )
        m["text"] = text
    check_rows(marks)
    return marks, revisions.pop()


def _fmt(yes: int, n: int) -> str:
    p, lo, hi = wilson(yes, n)
    return f"{yes}/{n} = {p:.1%} [{lo:.1%}, {hi:.1%}]"


def report(rows: list[Row], revision: str, seed: int, draws: int) -> str:
    out = [f"cut `{revision}`, {len(rows)} pieces marked", ""]
    kinds = [r["kind"] for r in rows]
    out.append("kinds of the owner's notes: " + ", ".join(f"{k}: {kinds.count(k)}" for k in KINDS))
    out.append("")
    for (name, sub), lv in zip(subsets(rows), levels(rows), strict=True):
        out += [f"## {name}: {lv.pieces} pieces of {lv.bodies} bodies", ""]
        out += ["| measure | pieces (Wilson 95%) | bodies resampled (95%) |", "|---|---|---|"]
        for q in QUESTIONS:
            lo, hi = body_bootstrap(sub, q, seed, draws)
            out.append(f"| {q} yes | {_fmt(lv.yes[q], lv.pieces)} | [{lo:.1%}, {hi:.1%}] |")
        lo, hi = body_bootstrap(sub, "clean", seed, draws)
        out.append(f"| clean (3 yes) | {_fmt(lv.clean_pieces, lv.pieces)} | [{lo:.1%}, {hi:.1%}] |")
        out.append(f"| faithful and one_story | {_fmt(lv.both, lv.pieces)} | |")
        out.append(f"| bodies with every piece clean | {_fmt(lv.clean_bodies, lv.bodies)} | |")
        side = "above" if wilson(lv.yes["faithful"], lv.pieces)[1] >= BAR else "below"
        measure = (
            "pre-set measure, every piece: "
            if name == "all"
            else "a level chosen after the misses: "
        )
        out += [
            "",
            f"faithful against the 80% bar of the extractor ({measure}the lower end of the interval is {side} the bar).",
            "",
        ]
    out += [
        "## Pieces with a kind other than kept, with the owner's note",
        "",
        "| review_order | kind | faithful | one_story | complete | note |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["kind"] != "kept":
            note = r["notes"].replace("|", "/").strip()
            out.append(
                f"| {r['review_order']} | {r['kind']} | {r['faithful'].strip()} | {r['one_story'].strip()} | {r['complete'].strip()} | {note} |"
            )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--marks", type=Path, default=Path("gold/cut_sample_marks.csv"))
    ap.add_argument("--texts", type=Path, default=Path("gold/texts/cut_sample_texts.csv"))
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None, help="also write the report here")
    args = ap.parse_args(argv)
    rows, revision = load(args.marks, args.texts)
    text = report(rows, revision, args.seed, args.draws)
    sys.stdout.write(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
