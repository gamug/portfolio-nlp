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
5. the share of RESULTS articles labelled `other` over the whole population, by
   upstream class and for the eight ordinary-word tickers (`ON`, `A`, `ARE`, `IT`,
   `ALL`, `HAS`, `TECH`, `NOW`) against the rest: is `other` concentrated on the
   wrong-tagged articles? (It is not: it follows each group's share of the corpus.)
6. how many `ok` SOURCE articles have no `article_category` row yet, when they were
   fetched, and when the last results were written;
7. which scoring path each stored category row took (level-1 short-circuit: no leaf
   scores; level 2: a softmax over exactly six evaluated leaves), i.e. what a re-label
   from stored scores can and cannot reproduce;
8. the `other` rate stratified by source and period, and the excess `other` that wrong
   tagging could explain at most (rejected and `review` articles at the confirmed
   articles' rate, overall and within strata);
9. whether the current decision rule applied to the stored scores reproduces every
   stored label (what a re-label from stored scores relies on);
10. the number of confirmed articles in each (predicted slug x score band) cell, which
    sizes the judged sample that per-slug calibration (T-121) needs;
11. on one eval run's `representative` bucket (uniform sampling weights, so the
   unweighted rates are population rates within it), how often the judge agreed
   with the stored category, by upstream class, with 95% Wilson intervals;
12. the same eval run through `aggregate_category`, split by upstream class with each
    stratum's population rescaled to a domain estimate (news_nlp.eval.partition):
    per-class precision, recall, F1 and one-vs-rest accuracy, overall accuracy and
    macro F1, and the judged counts.

Item 11 is unweighted agreement only. Item 12 is the complete constitution #12 metric
set per upstream class (T-133 steps 2-4), through `aggregate_category`.

Kept in the repository: docs/evaluation.md cites it as the command that reproduces
the 2026-10-22 figures, and a rerun after the next upstream change starts here. A dated
one-shot, like the other scripts/ one-shots (not part of the mypy gate, not run in CI).

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

# The shipped constants and slug order come from the repo, not from literals here, so a
# change to taxonomy.py cannot leave this check testing stale numbers. Only the pure
# `news_nlp.taxonomy` module is read (no model, no torch).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from news_nlp.db import SourceSchemaError, check_schema
from news_nlp.eval.partition import (
    bootstrap_interval,
    judge_label_counts,
    judged_counts,
    load_category_pairs,
    metrics_by_partition,
)
from news_nlp.taxonomy import (
    CATEGORY_CONFIDENCE_THRESHOLD,
    CATEGORY_GROUP_FLOOR,
    CATEGORY_SLUGS,
    OTHER_LABEL,
)

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
    try:  # the shared SOURCE guard (T-135): the minimum version plus the columns read here
        version = check_schema(
            conn,
            columns={"articles": ("id", "fetch_status"), "discovered_urls": ("id", "match_class")},
        )
    except SourceSchemaError as exc:
        raise SystemExit(f"{exc} (this script reads discovered_urls.match_class)") from exc
    print("urls.db user_version:", version)
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


WORD_TICKERS = ("ON", "A", "ARE", "IT", "ALL", "HAS", "TECH", "NOW")


def report_other_rates(conn: sqlite3.Connection) -> None:
    section("Population `other` rate (all RESULTS articles)")
    rows = conn.execute(
        "SELECT d.match_class, COUNT(*), SUM(rc.label = 'other') "
        "FROM r.article_category rc JOIN main.discovered_urls d ON d.id = rc.article_id "
        "GROUP BY 1 ORDER BY 1"
    ).fetchall()
    total = sum(n for _, n, _ in rows)
    total_other = sum(o for _, _, o in rows)
    for match_class, n, other in rows:
        print(
            f"  {match_class:<8} articles={n:>7} ({100 * n / total:4.1f}% of corpus) "
            f"other={other:>7} rate={100 * other / n:4.1f}% "
            f"share of all other={100 * other / total_other:4.1f}%"
        )
    print(f"  total    articles={total} other={total_other} ({100 * total_other / total:.1f}%)")
    base = (
        "SELECT COUNT(*), SUM(rc.label = 'other') FROM r.article_category rc "
        "JOIN r.articles a ON a.id = rc.article_id WHERE a.ticker "
    )
    members = "(?, ?, ?, ?, ?, ?, ?, ?)"  # one placeholder per entry of WORD_TICKERS
    assert members.count("?") == len(WORD_TICKERS)
    queries = {
        "word-like tickers": base + "IN " + members,
        "all other tickers": base + "NOT IN " + members,
    }
    for name, sql in queries.items():
        n, other = conn.execute(sql, WORD_TICKERS).fetchone()
        print(
            f"  {name:<18} articles={n:>7} ({100 * n / total:4.1f}%) other rate={100 * other / n:4.1f}%"
        )


def report_unprocessed(conn: sqlite3.Connection) -> None:
    section("`ok` SOURCE articles with no article_category row yet")
    ok, missing = conn.execute(
        "SELECT COUNT(*), SUM(NOT EXISTS (SELECT 1 FROM r.article_category rc WHERE rc.article_id = a.id)) "
        "FROM main.articles a WHERE a.fetch_status = 'ok'"
    ).fetchone()
    print(f"  ok articles={ok}  without a category row={missing}")
    first, last = conn.execute(
        "SELECT MIN(a.fetched_at), MAX(a.fetched_at) FROM main.articles a "
        "WHERE a.fetch_status = 'ok' AND NOT EXISTS "
        "(SELECT 1 FROM r.article_category rc WHERE rc.article_id = a.id)"
    ).fetchone()
    last_written = conn.execute("SELECT MAX(processed_at) FROM r.article_category").fetchone()[0]
    print(f"  those articles were fetched between {first} and {last}")
    print(f"  last article_category write: {last_written}")


# Rows with a level-1 group score below the floor took the short-circuit path.
GROUP_FLOOR = CATEGORY_GROUP_FLOOR
MIN_STRATUM = 50  # smallest confirmed-article count for a stratum to be used as a reference


def report_scoring_paths(conn: sqlite3.Connection) -> None:
    section("Scoring path of the stored category rows")
    rows = conn.execute(
        "SELECT d.match_class, COUNT(*), SUM(rc.group_score < ?), SUM(rc.group_score >= ?), "
        "AVG(CASE WHEN rc.group_score >= ? THEN rc.earnings_performance + rc.mergers_acquisitions "
        "+ rc.leadership_governance + rc.legal_regulatory + rc.product_innovation "
        "+ rc.capital_shareholder_returns + rc.labor_human_capital + rc.market_analyst_sentiment "
        "+ rc.partnerships_business_dev END), "
        "AVG(CASE WHEN rc.group_score >= ? THEN (rc.earnings_performance > 0) + (rc.mergers_acquisitions > 0) "
        "+ (rc.leadership_governance > 0) + (rc.legal_regulatory > 0) + (rc.product_innovation > 0) "
        "+ (rc.capital_shareholder_returns > 0) + (rc.labor_human_capital > 0) "
        "+ (rc.market_analyst_sentiment > 0) + (rc.partnerships_business_dev > 0) END) "
        "FROM r.article_category rc JOIN main.discovered_urls d ON d.id = rc.article_id GROUP BY 1",
        (GROUP_FLOOR, GROUP_FLOOR, GROUP_FLOOR, GROUP_FLOOR),
    ).fetchall()
    for match_class, n, short, level2, leaf_sum, evaluated in rows:
        print(
            f"  {match_class:<8} articles={n:>7} level-1 short-circuit (no leaf scores)={short:>6} "
            f"({100 * short / n:.1f}%) level 2={level2:>7}; on level-2 rows mean leaf sum="
            f"{leaf_sum:.4f}, mean evaluated leaves={evaluated:.2f}"
        )


CONFIDENCE_THRESHOLD = CATEGORY_CONFIDENCE_THRESHOLD
# Column order of the nine leaf scores in the SELECTs below; asserted against the repo's order.
LEAF_SLUGS = (
    "earnings_performance",
    "mergers_acquisitions",
    "leadership_governance",
    "legal_regulatory",
    "product_innovation",
    "capital_shareholder_returns",
    "labor_human_capital",
    "market_analyst_sentiment",
    "partnerships_business_dev",
)
assert LEAF_SLUGS == CATEGORY_SLUGS, "SELECT column order no longer matches taxonomy.CATEGORY_SLUGS"
BAND_EDGES = (0.4, 0.5, 0.6, 0.8)  # upper edges of the score bands; the last band is >= 0.8
BAND_NAMES = ("<0.4", "0.4-0.5", "0.5-0.6", "0.6-0.8", ">=0.8")
CELL_N = 60  # judged items per cell in the proposed calibration sample


def report_rule_consistency(conn: sqlite3.Connection) -> None:
    # category_stage's own functions take logits (classify_category_scores) or sit inline in
    # CategoryInference.predict_batch (the floor), so neither can be called on stored
    # probabilities. The decision is restated here with the shipped constants and slug order;
    # TASKS T-122 extracts one pure function that both predict_batch and the re-label use.
    section("Does the current rule, applied to the stored scores, reproduce every stored label?")
    rows = conn.execute(
        "SELECT rc.label, rc.group_score, rc.earnings_performance, rc.mergers_acquisitions, "
        "rc.leadership_governance, rc.legal_regulatory, rc.product_innovation, "
        "rc.capital_shareholder_returns, rc.labor_human_capital, rc.market_analyst_sentiment, "
        "rc.partnerships_business_dev FROM r.article_category rc"
    )
    total = short_circuit = mismatches = tied = 0
    for label, group_score, *leaf_scores in rows:
        total += 1
        if group_score < GROUP_FLOOR:
            short_circuit += 1
            expected = OTHER_LABEL
        else:
            best = max(leaf_scores)
            tied += leaf_scores.count(best) > 1
            expected = (
                LEAF_SLUGS[leaf_scores.index(best)] if best >= CONFIDENCE_THRESHOLD else OTHER_LABEL
            )
        mismatches += expected != label
    print(
        f"  rows={total} (level-1 short-circuit {short_circuit}); rows where the rule "
        f"(floor {GROUP_FLOOR}, argmax, threshold {CONFIDENCE_THRESHOLD}) differs from the stored label: {mismatches}"
    )
    print(f"  rows whose highest leaf score is an exact tie between two slugs: {tied}")


def report_calibration_cells(conn: sqlite3.Connection) -> None:
    section("Confirmed articles by predicted slug (argmax) and score band: calibration sample size")
    rows = conn.execute(
        "SELECT rc.group_score, rc.earnings_performance, rc.mergers_acquisitions, "
        "rc.leadership_governance, rc.legal_regulatory, rc.product_innovation, "
        "rc.capital_shareholder_returns, rc.labor_human_capital, rc.market_analyst_sentiment, "
        "rc.partnerships_business_dev FROM r.article_category rc "
        "JOIN main.discovered_urls d ON d.id = rc.article_id WHERE d.match_class = 'accept'"
    )
    cells: dict[tuple[str, int], int] = {}
    short_circuit = 0
    for group_score, *leaf_scores in rows:
        if group_score < GROUP_FLOOR:
            short_circuit += 1
            continue
        best = max(leaf_scores)
        band = sum(best >= edge for edge in BAND_EDGES)
        key = (LEAF_SLUGS[leaf_scores.index(best)], band)
        cells[key] = cells.get(key, 0) + 1
    print(f"  {'slug':<30}" + "".join(f"{name:>9}" for name in BAND_NAMES))
    for slug in LEAF_SLUGS:
        print(
            f"  {slug:<30}"
            + "".join(f"{cells.get((slug, b), 0):>9}" for b in range(len(BAND_NAMES)))
        )
    needed = sum(min(n, CELL_N) for n in cells.values()) + min(short_circuit, CELL_N)
    small = sum(1 for n in cells.values() if n < CELL_N)
    print(f"  level-1 short-circuit articles (no leaf scores): {short_circuit}")
    print(
        f"  cells={len(cells)} (+1 short-circuit); cells with fewer than {CELL_N} articles: {small}; "
        f"judged items for {CELL_N} per cell (all of a smaller cell): {needed}"
    )


def report_stratified_other(conn: sqlite3.Connection) -> None:
    section("`other` rate by upstream class, source and period; excess attributable to tagging")
    rows = conn.execute(
        "SELECT d.match_class, "
        "CASE WHEN a.source_domain LIKE '%cnbc%' THEN 'cnbc' ELSE 'other-domain' END, "
        "CASE WHEN a.pub_date < '2022-01-01' THEN 'pre-2022' ELSE '2022+' END, "
        "COUNT(*), SUM(rc.label = 'other') "
        "FROM r.article_category rc JOIN r.articles a ON a.id = rc.article_id "
        "JOIN main.discovered_urls d ON d.id = rc.article_id GROUP BY 1, 2, 3 ORDER BY 2, 3, 1"
    ).fetchall()
    cells: dict[tuple[str, str, str], tuple[int, int]] = {}
    for match_class, domain, period, n, other in rows:
        cells[(match_class, domain, period)] = (n, other)
        print(
            f"  {match_class:<8} {domain:<13} {period:<9} articles={n:>7} other rate={100 * other / n:4.1f}%"
        )
    total_other = sum(o for _, o in cells.values())
    overall = {"accept": [0, 0], "rest": [0, 0]}
    for (match_class, _, _), (n, other) in cells.items():
        key = "accept" if match_class == "accept" else "rest"
        overall[key][0] += n
        overall[key][1] += other
    ref = overall["accept"][1] / overall["accept"][0]
    excess = overall["rest"][1] - overall["rest"][0] * ref
    print(
        f"  unstratified: rejected + review articles at the confirmed rate ({100 * ref:.1f}%) "
        f"leave an excess of {excess:,.0f} `other` = {100 * excess / total_other:.1f}% of all `other`"
    )
    stratified = 0.0
    for (match_class, domain, period), (n, other) in cells.items():
        accept = cells.get(("accept", domain, period))
        if match_class != "accept" and accept is not None and accept[0] >= MIN_STRATUM:
            stratified += other - n * accept[1] / accept[0]
    print(
        f"  stratified by source and period (strata with >= {MIN_STRATUM} confirmed articles): excess "
        f"{stratified:,.0f} = {100 * stratified / total_other:.1f}% of all `other`"
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


def report_partitions(conn: sqlite3.Connection, results: Path, run_id: int) -> None:
    """T-133 steps 2-4: the stored judgements of one run, split by upstream class and
    run back through `aggregate_category` (the live-run code path, no judge call). Each
    partition's strata are reweighted to a domain estimate (news_nlp.eval.partition)."""
    section(f"eval_run {run_id}: complete metric set per upstream class (domain-reweighted)")
    stored = sqlite3.connect(f"file:{results.as_posix()}?mode=ro", uri=True)
    try:
        pairs = load_category_pairs(stored, run_id)
    finally:
        stored.close()
    partition_of = load_partition_of(conn, sorted({it.article_id for it, _ in pairs}))
    counts = judged_counts(pairs, partition_of)
    out = metrics_by_partition(pairs, partition_of)
    names = ["all", *[n for n in ("accept", "review", "reject") if n in out]]
    print("  judged items (parse ok) by bucket:")
    for name, by_bucket in counts.items():
        print(f"    {name:<8} total={sum(by_bucket.values()):>5}  {by_bucket}")
    print("  judge's own labels per partition (parse ok, outside low_conf, unweighted):")
    for name, labels in judge_label_counts(pairs, partition_of).items():
        print(f"    {name:<8} " + ", ".join(f"{k}={n}" for k, n in labels.items()))
    keys = sorted({k for o in out.values() for k in o})
    shown = [
        k
        for k in keys
        if not k.endswith("_naive_pooled")
        and not k.startswith("roc_auc_")
        and (
            k in {"n", "accuracy_vs_judge", "macro_f1", "other_rate_model", "other_rate_judge"}
            or k.startswith(("precision_", "recall_", "f1_", "accuracy_ovr_"))
        )
    ]
    print(f"  {'metric':<44}" + "".join(f"{n:>10}" for n in names))
    for k in shown:
        print(f"  {k:<44}" + "".join(f"{out[n].get(k, float('nan')):>10.3f}" for n in names))
    hidden = [k for k in keys if k not in shown]
    print(f"  not printed ({len(hidden)}, in metrics_by_partition): {', '.join(hidden)}")
    if "accept" in out and "reject" in out:
        print("  95% stratified-bootstrap intervals (500 replicates, seed 0):")
        for label, kwargs in (
            ("precision_other, all", {"partition": "all"}),
            ("precision_other, accept", {"partition": "accept"}),
            ("precision_other, reject", {"partition": "reject"}),
            ("precision_other, accept - reject", {"minus": ("accept", "reject")}),
        ):
            est, low, high = bootstrap_interval(pairs, partition_of, "precision_other", **kwargs)  # type: ignore[arg-type]
            print(f"    {label:<34} {est:>7.3f}  [{low:.3f}, {high:.3f}]")


def load_partition_of(conn: sqlite3.Connection, ids: list[int]) -> dict[int, str]:
    """``discovered_urls.match_class`` per article id (``(none)`` when NULL), in chunks."""
    partition_of: dict[int, str] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        marks = ",".join("?" * len(chunk))
        for aid, match_class in conn.execute(
            f"SELECT id, COALESCE(match_class, '(none)') FROM main.discovered_urls WHERE id IN ({marks})",  # noqa: S608 -- placeholders only
            chunk,
        ):
            partition_of[aid] = match_class
    return partition_of


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
    report_other_rates(conn)
    report_stratified_other(conn)
    report_scoring_paths(conn)
    report_rule_consistency(conn)
    report_calibration_cells(conn)
    report_unprocessed(conn)
    report_eval(conn, args.eval_run)
    report_partitions(conn, args.results_db, args.eval_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
