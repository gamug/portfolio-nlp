"""T-163: how the step-1 scope is spread over the corpus, by ``fetch_status`` and by year, and a sample of ``out_of_scope`` bodies to read.

    uv run python scripts/scope_corpus_breakdown_2026_11_11.py --results-db <nlp.db> --source-db <urls.db> \
        [--revision scope-nli-8ff21e50796b] [--sample 50] [--out data/scope_oos_quarantined_sample.csv]

Reads ``body_scope`` of the RESULTS file and the unique bodies of SOURCE (the properly fetched corpus, ``ok`` and
``quarantined``; a body is identified by its hash, as in the step, and takes the status and year of the first article that
carries it). Prints the share of every scope by status and by year and writes a random sample of the bodies of one status
the model called ``out_of_scope`` (id, title, first 300 characters) for a human read. No GPU, no writes to either database.
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from news_nlp import triage
from news_nlp.steps import PROPER_STATUSES, SCOPES

Row = tuple[int, str, str, str, str]  # article id, status, year, title, text


def unique_bodies(
    rows: Iterable[tuple[int, str, str | None, str | None, str | None]],
) -> dict[str, Row]:
    """The first article of every unique body, in id order: ``{body_hash: (id, status, year, title, text)}``."""
    out: dict[str, Row] = {}
    for article_id, status, pub_date, title, text in rows:
        if not text:
            continue
        out.setdefault(
            triage.body_hash(text),
            (int(article_id), status, (pub_date or "")[:4] or "?", title or "", text),
        )
    return out


def breakdown(
    bodies: Mapping[str, Row], scope_of: Mapping[str, str]
) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]]]:
    """Scope counts by status and by year, over the bodies that have a scope."""
    by_status: dict[str, Counter[str]] = defaultdict(Counter)
    by_year: dict[str, Counter[str]] = defaultdict(Counter)
    for key, (_, status, year, _, _) in bodies.items():
        if key in scope_of:
            by_status[status][scope_of[key]] += 1
            by_year[year][scope_of[key]] += 1
    return by_status, by_year


def lines(groups: Mapping[str, Counter[str]], title: str) -> list[str]:
    out = [f"{title}: bodies | " + " | ".join(SCOPES)]
    for name in sorted(groups):
        total = sum(groups[name].values())
        shares = " | ".join(f"{groups[name][s] / total:6.1%}" for s in SCOPES)
        out.append(f"  {name:>14}: {total:7,d} | {shares}")
    return out


def sample_oos(
    bodies: Mapping[str, Row], scope_of: Mapping[str, str], status: str, size: int, seed: int = 1
) -> list[Row]:
    pool = sorted(
        (
            row
            for key, row in bodies.items()
            if scope_of.get(key) == "out_of_scope" and row[1] == status
        ),
        key=lambda r: r[0],
    )
    return random.Random(seed).sample(pool, min(size, len(pool)))  # noqa: S311 -- a sample to read, not a secret


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--revision", default="scope-nli-8ff21e50796b")
    parser.add_argument("--status", default="quarantined", help="status of the sample to read")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("data/scope_oos_sample.csv"))
    parser.add_argument(
        "--immutable",
        action="store_true",
        help="open RESULTS without its write-ahead log: rows still in the -wal file are MISSING; prefer a copy of nlp.db with its -wal",
    )
    args = parser.parse_args()
    flags = "mode=ro&immutable=1" if args.immutable else "mode=ro"
    results = sqlite3.connect(f"file:{args.results_db}?{flags}", uri=True)
    scope_of = dict(
        results.execute(
            "SELECT body_hash, scope FROM body_scope WHERE classifier_revision = ?",
            (args.revision,),
        )
    )
    source = sqlite3.connect(f"file:{args.source_db}?mode=ro", uri=True)
    marks = ", ".join("?" for _ in PROPER_STATUSES)
    bodies = unique_bodies(
        source.execute(
            f"SELECT id, fetch_status, pub_date, title, body_text FROM articles "  # noqa: S608 -- fixed clauses
            f"WHERE fetch_status IN ({marks}) ORDER BY id",
            PROPER_STATUSES,
        )
    )
    by_status, by_year = breakdown(bodies, scope_of)
    print(
        f"{len(scope_of):,} bodies with a scope at {args.revision}; {len(bodies):,} unique bodies in the corpus"
    )
    print("\n".join([*lines(by_status, "by fetch_status"), "", *lines(by_year, "by year")]))
    rows = sample_oos(bodies, scope_of, args.status, args.sample)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["article_id", "status", "year", "title", "text_start"])
        for article_id, status, year, title, text in rows:
            writer.writerow([article_id, status, year, title, " ".join(text.split())[:300]])
    print(f"\n{len(rows)} {args.status} bodies called out_of_scope -> {args.out}")


if __name__ == "__main__":
    main()
