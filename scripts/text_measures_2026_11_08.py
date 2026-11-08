"""Text measures of the company texts the extractors write (Work item 18, T-148b10).

    uv run scripts/text_measures_2026_11_08.py --out data [--out data/faithfulness_confirmation]

Reads ``faithfulness_ids.csv`` and ``faithfulness_to_tag.csv`` of each ``--out`` directory (the texts
the owner read; the marks are not used) and prints, per extractor, characters per text and how the
sentences of a text divide. Each sentence of a text for one company is put in one group, with the names
the plain ``CompanyResolver`` finds in it (``skip_attribution=False``):

* **names the company**: the company's own ticker is among the names;
* **mixed**: it names the company and another universe company (a subset of the line above);
* **names nobody**: no universe company is named.

A sentence naming only other companies cannot occur in a text (a sentence goes only to the companies it
names, or to those of the previous sentence), so that group is zero and is not shown. A text that the
guard or the header rule of ``units2``/``units3`` gave to a company by a short form shows its lines as
"names nobody": the plain resolver does not read short forms there, so ``own%`` is not comparable
between ``units`` and those two on header lines.

The paired lines compare two extractors on the same company of the same body, with a 95% bootstrap
interval over the bodies (4,000 redraws, seed 1). It reads files and writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import retag_gold_2026_10_29 as retag_cli

from chunking import split_sentences
from news_nlp.triage import CompanyResolver

EXTRACTORS = ("baseline", "units", "units2", "units3")
PAIRS = (
    ("baseline", "units"),
    ("units2", "units"),
    ("units3", "units"),
    ("units3", "units2"),
)
DRAWS = 4000
SEED = 1


class Measure(NamedTuple):
    """One text of one extractor for one company of one body."""

    extractor: str
    article: str
    ticker: str
    chars: int
    sentences: int
    own: int
    mixed: int
    nameless: int
    others: int

    @property
    def mixed_share(self) -> float:
        return self.mixed / max(1, self.sentences)


def measure_text(
    text: str,
    ticker: str,
    sentences: Iterable[str],
    resolve: Callable[[str], Iterable[str]],
) -> tuple[int, int, int, int, int, int]:
    """``(chars, sentences, own, mixed, nameless, others)`` of *text* written for *ticker*.

    *sentences* is the text already split and *resolve* gives the tickers a sentence names; both are
    parameters so that the counting is testable without the resolver."""
    parts = list(sentences)
    own = mixed = nameless = 0
    others: set[str] = set()
    for sentence in parts:
        named = set(resolve(sentence))
        if ticker in named:
            own += 1
            if len(named) > 1:
                mixed += 1
                others |= named - {ticker}
        elif not named:
            nameless += 1
    return len(text), len(parts), own, mixed, nameless, len(others)


def read_measures(
    out: Path,
    split: Callable[[str], Iterable[str]],
    resolve: Callable[[str], Iterable[str]],
) -> list[Measure]:
    """The measures of every (extractor, text) of the draw in *out*; a text shared by several
    extractors is counted once for each, as the report counts it."""
    ids = _read(out / "faithfulness_ids.csv")
    texts = {r["review_order"]: r["text"] for r in _read(out / "faithfulness_to_tag.csv")}
    rows: list[Measure] = []
    for row in ids:
        text = texts[row["review_order"]]
        chars, n, own, mixed, nameless, others = measure_text(
            text, row["ticker"], split(text), resolve
        )
        for extractor in row["extractors"].split("|"):
            rows.append(
                Measure(
                    extractor,
                    row["article_id"],
                    row["ticker"],
                    chars,
                    n,
                    own,
                    mixed,
                    nameless,
                    others,
                )
            )
    return rows


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summary_lines(rows: list[Measure]) -> list[str]:
    """The table, one line per extractor present."""
    head = (
        f"{'':9}{'texts':>6}{'mean ch':>9}{'median':>8}{'sent/text':>10}{'own%':>7}"
        f"{'nameless%':>10}{'mixed sent%':>12}{'texts w/ mixed%':>16}{'others/text':>12}"
    )
    lines = [head]
    for name in EXTRACTORS:
        mine = [r for r in rows if r.extractor == name]
        if not mine:
            continue
        total = sum(r.sentences for r in mine) or 1
        lines.append(
            f"{name:9}{len(mine):6}{statistics.mean(r.chars for r in mine):9.0f}"
            f"{statistics.median(r.chars for r in mine):8.0f}{total / len(mine):10.1f}"
            f"{100 * sum(r.own for r in mine) / total:7.1f}"
            f"{100 * sum(r.nameless for r in mine) / total:10.1f}"
            f"{100 * sum(r.mixed for r in mine) / total:12.1f}"
            f"{100 * sum(r.mixed > 0 for r in mine) / len(mine):16.1f}"
            f"{statistics.mean(r.others for r in mine):12.2f}"
        )
    return lines


def paired(
    rows: list[Measure], a: str, b: str, field: Callable[[Measure], float]
) -> tuple[int, float, float, float] | None:
    """``(n, mean, low, high)`` of ``field(a) - field(b)`` over the companies both wrote a text for,
    with a bootstrap over the bodies; ``None`` when there is no such company."""
    by: dict[tuple[str, str], dict[str, Measure]] = defaultdict(dict)
    for r in rows:
        by[(r.article, r.ticker)][r.extractor] = r
    groups: dict[str, list[float]] = defaultdict(list)
    for (article, _), pair in by.items():
        if a in pair and b in pair:
            groups[article].append(field(pair[a]) - field(pair[b]))
    if not groups:
        return None
    bodies = list(groups.values())
    rng = random.Random(SEED)  # noqa: S311 - a seeded bootstrap, not a secret
    means = sorted(
        statistics.mean([x for g in (rng.choice(bodies) for _ in bodies) for x in g])
        for _ in range(DRAWS)
    )
    flat = [x for g in bodies for x in g]
    low, high = means[int(DRAWS * 0.025)], means[int(DRAWS * 0.975) - 1]
    return len(flat), statistics.mean(flat), low, high


def paired_lines(rows: list[Measure]) -> list[str]:
    lines = ["paired differences (same company, same body), 95% interval over the bodies"]
    for a, b in PAIRS:
        chars = paired(rows, a, b, lambda r: float(r.chars))
        share = paired(rows, a, b, lambda r: r.mixed_share)
        if chars:
            n, m, lo, hi = chars
            lines.append(f"  chars {a} - {b}: n={n} {m:+.0f} [{lo:+.0f}, {hi:+.0f}]")
        if share:
            n, m, lo, hi = share
            lines.append(
                f"  mixed-share {a} - {b}: n={n} {100 * m:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, action="append", required=True, metavar="DIR")
    args = parser.parse_args()

    resolver = CompanyResolver(retag_cli.companies())

    def split(text: str) -> list[str]:
        return [s for s, _, _ in split_sentences(text)]

    def resolve(sentence: str) -> list[str]:
        return list(resolver.resolve(sentence, skip_attribution=False))

    for out in args.out:
        rows = read_measures(out, split, resolve)
        print(f"== {out}")
        print("\n".join(summary_lines(rows)))
        print()
        print("\n".join(paired_lines(rows)))
        print()


if __name__ == "__main__":
    main()
