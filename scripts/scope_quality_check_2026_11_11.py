"""T-163: a read-only quality check of the step-1 results (integrity, gaps, lengths, unusable bodies, a sample to read).

    uv run python scripts/scope_quality_check_2026_11_11.py --results-db <nlp.db> --source-db <urls.db> [--immutable] [--seed 7]

Checks the ``body_scope`` rows of one revision against their state rows, their stored scores and the unique bodies of SOURCE
(``ok`` and ``quarantined``): duplicates, non-finite scores, a stored scope that is not what its scores give, state without
result, corpus bodies without a scope, the scope by body length, bodies that look like a paywall or bot page, bodies without a
title, the share with a small margin between the two best topics, and a random sample of single / multi / sector bodies to read.
No GPU and no write to either database.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sqlite3
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import scope_corpus_breakdown_2026_11_11 as corpus

from news_nlp.steps import PROPER_STATUSES, SCOPES

TOPICS = ("company", "sector", "not_related")
PAGE = re.compile(
    r"security verification|enable javascript|subscribe to (read|continue)|please enable cookies|access denied|are you a robot|captcha",
    re.IGNORECASE,
)
SMALL_MARGIN = 1.0


def scope_from_scores(scores: Mapping[str, float]) -> str:
    """The scope the stored scores give (the step's rule: topic, then single/multi for a company body)."""
    topic = max(TOPICS, key=lambda name: scores[f"topic:{name}"])
    if topic == "sector":
        return "sector"
    if topic == "not_related":
        return "out_of_scope"
    return max(("single", "multi"), key=lambda name: scores[f"count:{name}"])


def length_bucket(n: int) -> str:
    return "<300" if n < 300 else "<1000" if n < 1000 else "<3000" if n < 3000 else ">=3000"  # noqa: PLR2004


def integrity(rows: list[tuple[str, str, str]]) -> dict[str, int]:
    """Counts of the problems in ``(body_hash, scope, scores_json)`` rows; all should be 0."""
    bad = {
        "duplicates": len(rows) - len({r[0] for r in rows}),
        "non_finite": 0,
        "scope_not_from_scores": 0,
    }
    for _, scope, raw in rows:
        scores = json.loads(raw)
        if any(not math.isfinite(v) for v in scores.values()):
            bad["non_finite"] += 1
        elif scope_from_scores(scores) != scope:
            bad["scope_not_from_scores"] += 1
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--revision", default="scope-nli-8ff21e50796b")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--immutable",
        action="store_true",
        help="open RESULTS without its write-ahead log (the writer is stopped)",
    )
    args = parser.parse_args()
    results = sqlite3.connect(
        f"file:{args.results_db}?mode=ro{'&immutable=1' if args.immutable else ''}", uri=True
    )
    rows = results.execute(
        "SELECT body_hash, scope, scores FROM body_scope WHERE classifier_revision = ?",
        (args.revision,),
    ).fetchall()
    print(f"{len(rows):,} rows at {args.revision}; problems: {integrity(rows)}")
    print(
        "state:",
        results.execute(
            "SELECT status, COUNT(*) FROM pipeline_step_state WHERE step = 'scope' AND model_revision = ? GROUP BY 1",
            (args.revision,),
        ).fetchall(),
        "| done without result:",
        results.execute(
            "SELECT COUNT(*) FROM pipeline_step_state s WHERE step = 'scope' AND status = 'done' AND model_revision = ? "
            "AND item_id NOT IN (SELECT body_hash FROM body_scope WHERE classifier_revision = ?)",
            (args.revision, args.revision),
        ).fetchone()[0],
    )
    scope_of = {r[0]: r[1] for r in rows}
    source = sqlite3.connect(f"file:{args.source_db}?mode=ro", uri=True)
    marks = ", ".join("?" for _ in PROPER_STATUSES)
    bodies = corpus.unique_bodies(
        source.execute(
            f"SELECT id, fetch_status, pub_date, title, body_text FROM articles WHERE fetch_status IN ({marks}) ORDER BY id",  # noqa: S608 -- fixed clauses
            PROPER_STATUSES,
        )
    )
    missing = [k for k in bodies if k not in scope_of]
    print(
        f"corpus bodies without a scope: {len(missing)} {dict(collections.Counter(bodies[k][1] for k in missing))}; "
        f"scopes for bodies outside the corpus: {len(set(scope_of) - set(bodies))}"
    )
    by_length: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for key, (_, _, _, _, text) in bodies.items():
        if key in scope_of:
            by_length[length_bucket(len(text))][scope_of[key]] += 1
    for bucket in ("<300", "<1000", "<3000", ">=3000"):
        n = sum(by_length[bucket].values())
        print(
            f"length {bucket:>6}: {n:7,d} | "
            + " ".join(f"{s} {by_length[bucket][s] / n:.0%}" for s in SCOPES)
        )
    pages = [k for k, v in bodies.items() if k in scope_of and PAGE.search(v[4][:600])]
    print(
        f"paywall or bot-page looking bodies: {len(pages)} {dict(collections.Counter(scope_of[k] for k in pages))}"
    )
    untitled = [k for k, v in bodies.items() if k in scope_of and not v[3].strip()]
    print(
        f"bodies without a title: {len(untitled)} {dict(collections.Counter(scope_of[k] for k in untitled))}"
    )
    small = 0
    for _, _, raw in rows:
        margins = sorted((json.loads(raw)[f"topic:{t}"] for t in TOPICS), reverse=True)
        small += margins[0] - margins[1] < SMALL_MARGIN
    print(f"topic margin under {SMALL_MARGIN}: {small:,} ({small / max(len(rows), 1):.1%})")
    rnd = random.Random(args.seed)  # noqa: S311 -- a sample to read, not a secret
    for scope in ("single", "multi", "sector"):
        pool = sorted(
            (v for k, v in bodies.items() if scope_of.get(k) == scope), key=lambda v: v[0]
        )
        print(f"\n## {scope}")
        for _, status, year, title, text in rnd.sample(pool, min(14, len(pool))):
            print(f"{status[:2]} {year} | {title[:90]} | {' '.join(text.split())[:90]}")


if __name__ == "__main__":
    main()
