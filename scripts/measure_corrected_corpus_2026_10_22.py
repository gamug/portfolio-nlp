#!/usr/bin/env python
"""One-shot, read-only: measure what `portfolio-data-mining`'s ticker re-tag did
to this repo's stored results (PLAN.md Work item 15, T-133 step 1 and the first
look at step 3).

Context: docs/evaluation.md's 2026-10-22 follow-up. Nothing here writes: both
databases are opened `mode=ro` with `PRAGMA query_only=ON`, so it is safe to run
while a collector holds SOURCE open (portfolio-common >= 1.2.2 allows that; this
script does its own plain read-only open, not `news_nlp`'s).

It reports, in order:

1. the schema version of SOURCE (`PRAGMA user_version`);
2. which category `eval_run`s have stored judgements at all (eval_run 21's do not);
3. how the RESULTS articles split by SOURCE `fetch_status` and `match_class`;
4. how many of each class already have stored `article_entities` (and an `ORG`
   span), i.e. what a NER-based relevance check would cost in GPU time;
5. on one eval run's `representative` bucket (uniform sampling weights, so the
   unweighted rates are population rates within it), how often the judge agreed
   with the stored category, by upstream class, with 95% Wilson intervals.

Item 5 is unweighted agreement only, not the complete constitution #12 metric
set; the complete set per partition is T-133's remaining work.

Safe to remove once the numbers are in docs/evaluation.md; keep it in git
history so a rerun after the next upstream change has a starting point.

Usage:
    uv run python scripts/measure_corrected_corpus_2026_10_22.py \\
        --source-db /path/to/urls.db --results-db /path/to/nlp.db --eval-run 43
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
from pathlib import Path

Z95 = 1.96


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (centre - half) / denom, 100 * (centre + half) / denom)


def rate(k: int, n: int) -> str:
    if n == 0:
        return "n=0"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% (95% interval {lo:.0f}-{hi:.0f}%)"


def connect(source: Path, results: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("ATTACH DATABASE ? AS r", (f"file:{results.as_posix()}?mode=ro",))
    return conn


def section(title: str) -> None:
    print(f"\n== {title}")


def report_versions_and_runs(conn: sqlite3.Connection) -> None:
    section("SOURCE schema version and stored category judgements")
    print("urls.db user_version:", conn.execute("PRAGMA user_version").fetchone()[0])
    stored = dict(
        conn.execute(
            "SELECT run_id, COUNT(*) FROM r.eval_verdict WHERE task = 'category' GROUP BY run_id"
        ).fetchall()
    )
    runs = conn.execute(
        "SELECT id, experiment, status, sample_size, started_at FROM r.eval_run "
        "WHERE stage = 'category' ORDER BY id"
    ).fetchall()
    for run_id, experiment, status, sample, started in runs:
        print(
            f"eval_run {run_id:>3} {experiment:<20} {status:<8} sample={sample:<5} "
            f"stored verdicts={stored.get(run_id, 0)}  ({started[:10]})"
        )


def report_split(conn: sqlite3.Connection) -> None:
    section("RESULTS articles by upstream class (article_category ids)")
    total = conn.execute("SELECT COUNT(*) FROM r.article_category").fetchone()[0]
    print("RESULTS article_category rows:", total)
    rows = conn.execute(
        "SELECT COALESCE(d.match_class, '(none)'), COALESCE(a.fetch_status, '(missing)'), COUNT(*) "
        "FROM r.article_category rc "
        "LEFT JOIN main.articles a ON a.id = rc.article_id "
        "LEFT JOIN main.discovered_urls d ON d.id = rc.article_id "
        "GROUP BY 1, 2 ORDER BY 3 DESC"
    ).fetchall()
    for match_class, status, n in rows:
        print(f"  {match_class:<8} {status:<12} {n:>8}  {100 * n / total:5.1f}%")


def report_entities(conn: sqlite3.Connection) -> None:
    section("Stored NER coverage by upstream class (zero GPU if present)")
    rows = conn.execute(
        "SELECT d.match_class, COUNT(*), "
        "SUM(EXISTS (SELECT 1 FROM r.article_entities e WHERE e.article_id = rc.article_id)), "
        "SUM(EXISTS (SELECT 1 FROM r.article_entities e "
        "WHERE e.article_id = rc.article_id AND e.entity_type = 'ORG')) "
        "FROM r.article_category rc JOIN main.discovered_urls d ON d.id = rc.article_id "
        "GROUP BY 1"
    ).fetchall()
    for match_class, n, with_entities, with_org in rows:
        print(
            f"  {match_class:<8} articles={n:>7} with entities={with_entities:>7} with ORG={with_org:>7}"
        )


_JUDGED = (
    "FROM r.eval_inference i JOIN r.eval_verdict v ON v.inference_id = i.id "
    "LEFT JOIN main.discovered_urls d ON d.id = i.article_id "
    "WHERE i.task = 'category' AND i.run_id = ?"
)


def report_eval(conn: sqlite3.Connection, run_id: int) -> None:
    section(f"eval_run {run_id}: judged items by upstream class (all buckets)")
    for match_class, n, agree in conn.execute(
        "SELECT COALESCE(d.match_class, '(none)'), COUNT(*), SUM(v.correct) "
        + _JUDGED
        + " GROUP BY 1 ORDER BY 2 DESC",
        (run_id,),
    ):
        print(f"  {match_class:<8} judged={n:>5} judge agrees {rate(agree, n)}")

    section(f"eval_run {run_id}: `representative` bucket (uniform weights), unweighted")
    rep = _JUDGED + " AND i.bucket = 'representative'"
    label = "json_extract(i.prediction_json, '$.label')"
    queries = {
        "any label": "",
        "model predicted other": f" AND {label} = 'other'",
        "model predicted a substantive label": f" AND {label} != 'other'",
    }
    for name, extra in queries.items():
        print(f"  {name}:")
        for match_class, n, agree in conn.execute(
            "SELECT COALESCE(d.match_class, '(none)'), COUNT(*), SUM(v.correct) "
            + rep
            + extra
            + " GROUP BY 1 ORDER BY 2 DESC",
            (run_id,),
        ):
            print(f"    {match_class:<8} {rate(agree, n)}")


def default_path(env: str) -> Path | None:
    value = os.environ.get(env)
    return Path(value) if value else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source-db", type=Path, default=default_path("SOURCE_DATABASE_URL"))
    parser.add_argument("--results-db", type=Path, default=default_path("DATABASE_URL"))
    parser.add_argument("--eval-run", type=int, default=43, help="category eval_run to partition")
    args = parser.parse_args()
    if args.source_db is None or args.results_db is None:
        parser.error("--source-db and --results-db are required (or set the env vars)")
    conn = connect(args.source_db, args.results_db)
    report_versions_and_runs(conn)
    report_split(conn)
    report_entities(conn)
    report_eval(conn, args.eval_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
