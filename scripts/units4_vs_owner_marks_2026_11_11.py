"""What the owner's faithfulness marks say about a cut reading (T-176, T-177): for the bodies the owner already read, cut with reading A and
reading B (default `units` and `units4`) and look up each text in the marks.

    uv run scripts/units4_vs_owner_marks_2026_11_11.py --source-db <urls.db copy> --data-dir data --data-dir data/faithfulness_confirmation \
        [--model-dir data/models/attribution_2026_11_08] [--a units] [--b units4] [--extra-marks gold/cut_owner_marks_2026_11_11.csv]

``--extra-marks`` adds the marks the owner made later on texts of a later reading (``gold/cut_owner_marks_2026_11_11.csv``: the 7 `units4` texts,
the 12 new `units5` texts and the blind sample of 30; the file says which marks are inferred). The output counts are labelled by A and B, not by
reading names: "A mark" is the mark of reading A's text, "B text UNREAD" a text of reading B without a mark. The figure compares the rates over the
texts that have a mark in each reading, not the same (body, company) pairs.

The owner marked (faithful yes/no) the texts of the faithfulness draws (``faithfulness_to_tag.csv``, with the text hash in
``faithfulness_ids.csv``). A mark belongs to a text, so a `units4` text equal to a marked text is already read; a company that
`units` gave a text and `units4` does not ("lost") shows whether the text left out was faithful (the owner said yes: a real loss)
or not (a text that should not exist). Reads the files and the source copy; writes nothing. The marks are those of 2026-11-08/10: a
`units` text the current code writes differently has no mark ("stale"), and is counted apart.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sqlite3
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TEXT_CAP = 6000  # what the owner read (draw_sector_extra.TEXT_CAP)
RESAMPLES = 10_000
Tally = dict[str, list[int]]  # body -> [faithful, marked] of one reading


def share(tally: Tally, bodies: list[str]) -> float:
    """Faithful share of the marked texts of ``bodies`` (a body may repeat: it is a bootstrap draw)."""
    yes = sum(tally[b][0] for b in bodies)
    marked = sum(tally[b][1] for b in bodies)
    return yes / marked if marked else float("nan")


def body_intervals(a: Tally, b: Tally, seed: int = 1) -> dict[str, tuple[float, float, float]]:
    """Point and 95% percentile interval of each reading's faithful share and of ``b - a``, redrawing **bodies** (the
    texts of one body are not independent, D32). The two readings are redrawn together, so the difference is paired."""
    bodies = sorted(set(a) | set(b))
    for t in (a, b):
        for body in bodies:
            t.setdefault(body, [0, 0])
    rng = random.Random(seed)  # noqa: S311 -- a reproducible interval, not security
    draws: dict[str, list[float]] = {"A": [], "B": [], "B-A": []}
    for _ in range(RESAMPLES):
        pick = [rng.choice(bodies) for _ in bodies]
        x, y = share(a, pick), share(b, pick)
        draws["A"].append(x)
        draws["B"].append(y)
        draws["B-A"].append(y - x)

    def interval(values: list[float]) -> tuple[float, float]:
        ordered = sorted(values)
        return ordered[round(0.025 * (len(ordered) - 1))], ordered[
            round(0.975 * (len(ordered) - 1))
        ]

    points = {"A": share(a, bodies), "B": share(b, bodies)}
    points["B-A"] = points["B"] - points["A"]
    return {k: (points[k], *interval(draws[k])) for k in draws}


def text_hash(text: str) -> str:
    """The identity of a text a mark belongs to (``retag.text_hash``)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def read_marks(data_dir: Path) -> dict[tuple[str, str, str], str]:
    """``{(article_id, ticker, text_hash): "yes" | "no"}`` of one draw's tagging file."""
    with (data_dir / "faithfulness_ids.csv").open(encoding="utf-8-sig") as f:
        ids = list(csv.DictReader(f))
    with (data_dir / "faithfulness_to_tag.csv").open(encoding="utf-8-sig") as f:
        tag = {r["review_order"]: r["faithful"].strip().lower() for r in csv.DictReader(f)}
    return {
        (r["article_id"], r["ticker"], r["text_hash"]): tag[r["review_order"]]
        for r in ids
        if tag.get(r["review_order"]) in ("yes", "no")
    }


def read_extra_marks(path: Path) -> dict[tuple[str, str, str], str]:
    """``{(article_id, ticker, text_hash): "yes" | "no"}`` of a CSV with the text itself (marks made on a later reading)."""
    with path.open(encoding="utf-8-sig") as f:
        return {
            (r["article_id"], r["ticker"], text_hash(r["text"][:TEXT_CAP])): r["faithful"]
            .strip()
            .lower()
            for r in csv.DictReader(f)
            if r["faithful"].strip().lower() in ("yes", "no")
        }


def load_marks(
    data_dirs: Sequence[Path], extra: Path | None = None
) -> list[tuple[str, dict[tuple[str, str, str], str]]]:
    """``[(folder name, marks)]``, one entry per draw folder. The marks of ``extra`` (``--extra-marks``) belong to **no draw**: they are
    added once, to the first folder. (Added to every folder they were counted once per folder: the figures of the first version of
    #172 were inflated that way, review round 2.)"""
    loaded = []
    for i, data_dir in enumerate(data_dirs):
        marks = read_marks(data_dir)
        if extra is not None and i == 0:
            marks |= read_extra_marks(extra)
        loaded.append((data_dir.name, marks))
    return loaded


def classify(
    a: str | None, b: str | None, marks: dict[tuple[str, str, str], str], article: str, ticker: str
) -> tuple[str, str | None, str | None]:
    """``(outcome, mark of A, mark of B)``: outcome is ``same``, ``lost``, ``gained`` or ``changed``."""

    def mark(text: str | None) -> str | None:
        return None if not text else marks.get((article, ticker, text_hash(text[:TEXT_CAP])))

    if a and not b:
        return "lost", mark(a), None
    if b and not a:
        return "gained", None, mark(b)
    if a == b:
        return "same", mark(a), mark(b)
    return "changed", mark(a), mark(b)


def add_mark(tally: Tally, body: str, mark: str | None) -> None:
    """Count one text of ``body`` with a mark (a text without one is not in the figure)."""
    if mark in ("yes", "no"):
        cell = tally.setdefault(body, [0, 0])
        cell[0] += mark == "yes"
        cell[1] += 1


def count_outcome(count: Counter[str], outcome: str, ma: str | None, mb: str | None) -> None:
    """One (body, company) pair of the comparison, by outcome and by the marks of the two texts."""
    count[outcome] += 1
    count[f"{outcome}: A mark {ma or 'none (stale or unread)'}"] += 1
    if outcome in ("changed", "gained"):
        count[f"{outcome}: B text {'marked ' + mb if mb else 'UNREAD'}"] += 1


def write_to_read(path: Path, rows: list[list[str]]) -> None:
    """The lost and the unread changed texts, for the owner to read (``text_units`` is reading A, ``text_units4`` reading B)."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["outcome", "article_id", "ticker", "units_mark", "text_units", "text_units4"]
        )
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def print_intervals(
    tally_a: Tally, tally_b: Tally, name_a: str = "units", name_b: str = "units4"
) -> None:
    """The faithful share of each reading on the marked texts, with the body bootstrap and the Wilson interval for contrast."""
    from news_nlp.gold_metrics import wilson  # noqa: PLC0415 -- after the path set-up

    print("== faithful share of the texts with a mark (bootstrap over bodies, 95%)")
    for name, tally in ((name_a, tally_a), (name_b, tally_b)):
        yes = sum(c[0] for c in tally.values())
        marked = sum(c[1] for c in tally.values())
        print(f"  {name}: {yes}/{marked}; if the texts were independent: {wilson(yes, marked)}")
    for key, (point, low, high) in body_intervals(tally_a, tally_b).items():
        label = {"A": name_a, "B": name_b, "B-A": f"{name_b} minus {name_a} (paired)"}[key]
        print(f"  {label}: {point:.1%} [{low:.1%}, {high:.1%}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-db", required=True)
    parser.add_argument(
        "--a", default="units", help="the reading the marks were made with (default units)"
    )
    parser.add_argument(
        "--b", default="units4", help="the reading to look up in the marks (default units4)"
    )
    parser.add_argument("--data-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--extra-marks",
        type=Path,
        default=None,
        help="a CSV of article_id, ticker, faithful, text: marks made later (gold/cut_owner_marks_2026_11_11.csv)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/models/attribution_2026_11_08",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the lost and the unread changed texts here, to read",
    )
    args = parser.parse_args()
    to_read: list[list[str]] = []
    from cut_stage import UnitsExtractor  # noqa: PLC0415 -- loads torch
    from news_nlp import attribution_model as am  # noqa: PLC0415
    from news_nlp.triage import Company, CompanyResolver  # noqa: PLC0415

    src = sqlite3.connect(f"file:{args.source_db}?mode=ro", uri=True)
    rows = src.execute(
        "SELECT DISTINCT ticker, company, gics_sector, gics_sub_industry FROM articles WHERE fetch_status = 'ok' ORDER BY ticker, company"
    ).fetchall()
    companies = [Company(r[0], r[1] or "", r[2] or "", r[3] or "") for r in rows]
    resolver = CompanyResolver(companies)
    first = UnitsExtractor.load(args.model_dir, companies, args.a)
    second = UnitsExtractor(first.scorer, companies, "local:x", args.b)
    tally_a: Tally = {}
    tally_b: Tally = {}
    for folder, marks in load_marks(args.data_dir, args.extra_marks):
        articles = sorted({k[0] for k in marks})
        count: Counter[str] = Counter()
        for article in articles:
            row = src.execute(
                "SELECT body_text FROM articles WHERE id = ?", (int(article),)
            ).fetchone()
            if not row or not row[0]:
                count["body_missing"] += 1
                continue
            body = row[0]
            ta = am.model_company_texts(
                body, first.resolver, first.names, first.scorer, reading=args.a
            )
            tb = am.model_company_texts(
                body, second.resolver, second.names, second.scorer, reading=args.b
            )
            for ticker in sorted(resolver.resolve(body, skip_attribution=False)):
                outcome, ma, mb = classify(ta.get(ticker), tb.get(ticker), marks, article, ticker)
                if not ta.get(ticker) and not tb.get(ticker):
                    continue
                add_mark(tally_a, f"{folder}/{article}", ma)
                add_mark(tally_b, f"{folder}/{article}", mb)
                count_outcome(count, outcome, ma, mb)
                if outcome == "lost" or (outcome in ("changed", "gained") and mb is None):
                    to_read.append(
                        [
                            outcome,
                            article,
                            ticker,
                            ma or "",
                            ta.get(ticker, "")[:TEXT_CAP],
                            tb.get(ticker, "")[:TEXT_CAP],
                        ]
                    )
        print(f"== {folder} ({len(articles)} bodies, {len(marks)} marks)")
        for k in sorted(count):
            print(f"  {k}: {count[k]}")

    print_intervals(tally_a, tally_b, args.a, args.b)

    if args.out:
        write_to_read(args.out, to_read)


if __name__ == "__main__":
    main()
