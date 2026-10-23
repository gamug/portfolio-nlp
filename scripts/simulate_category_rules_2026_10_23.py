#!/usr/bin/env python
"""One-shot, read-only: simulate candidate category decision rules offline (PLAN.md
Work item 15 step 4, TASKS T-121(a)). No model, no judge call, nothing written: both
databases are opened `mode=ro` with `PRAGMA query_only=ON`.

It reports, for a grid of uniform rules (one confidence threshold for every slug, and a
minimum margin of the best slug over the runner-up):

1. how many confirmed (`accept`) articles each rule re-labels, from the stored leaf scores
   alone, in total, by direction, by band, and (with `--cells`) by (slug x band) cell;
2. for the rules that only tighten the shipped one, the stored eval run's metrics on the
   `accept` partition (the judge's `ideal_slug` is the truth), a first coarse trade-off
   table with paired bootstrap intervals against the shipped rule;
3. the power of T-123's estimator (paired difference of the weighted `other` precision,
   within-cell bootstrap) at the planned report-half sample size, under a truth model
   with a swept share of articles whose truth is the argmax slug, and at the share where
   the model's own implied gain is +0.05, for each share shape in `--ramps`.

Usage:
    uv run python scripts/simulate_category_rules_2026_10_23.py \\
        --source-db /path/to/urls.db --results-db /path/to/nlp.db --eval-run 43
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from news_nlp.eval.calibration import (
    BAND_NAMES,
    BASELINE,
    RAMPS,
    ConfirmedArticle,
    PowerRow,
    RelabelRow,
    Rule,
    apply_rule,
    candidate_rules,
    cell_of,
    fit_truth_model,
    frame_for,
    paired_rule_interval,
    relabel_table,
    simulate_power,
    slug_share_for_gain,
    verdict_table,
)
from news_nlp.eval.partition import domain_pairs, load_category_pairs
from news_nlp.taxonomy import CATEGORY_SLUGS

POWER_TARGET = 0.80
GAIN = 0.05
SHARES = (0.1, 0.3, 0.5)
COLUMNS = ("group_score", *CATEGORY_SLUGS)


def connect(source: Path, results: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("ATTACH DATABASE ? AS r", (f"file:{results.as_posix()}?mode=ro",))
    return conn


def load_confirmed(conn: sqlite3.Connection) -> list[ConfirmedArticle]:
    cols = ", ".join(f"rc.{c}" for c in COLUMNS)
    rows = conn.execute(
        f"SELECT rc.article_id, rc.label, {cols} FROM r.article_category rc "  # noqa: S608 -- repo constants
        "JOIN main.discovered_urls d ON d.id = rc.article_id WHERE d.match_class = 'accept'"
    )
    return [ConfirmedArticle(i, label, g, tuple(leaves)) for i, label, g, *leaves in rows]


def section(title: str) -> None:
    print(f"\n== {title}")


def report_relabels(articles: list[ConfirmedArticle], rules: list[Rule], show_cells: bool) -> None:
    mismatches = sum(apply_rule(a.group_score, a.leaf_scores) != a.stored_label for a in articles)
    section(f"Re-labelled confirmed articles per rule (of {len(articles)})")
    print(f"  baseline {BASELINE.name} reproduces the stored label on all but {mismatches}")
    print(f"  {'rule':<12}{'changed':>9}{'share':>8}{'->other':>9}{'other->':>9}{'slug->slug':>12}")
    table = relabel_table(articles, rules)
    for row in table:
        print(
            f"  {row.rule.name:<12}{row.changed:>9}{100 * row.changed / len(articles):>7.1f}%"
            f"{row.to_other:>9}{row.from_other:>9}{row.slug_to_slug:>12}"
        )
    section("Re-labelled articles by band (summed over slugs)")
    print(f"  {'rule':<12}" + "".join(f"{b:>9}" for b in BAND_NAMES) + f"{'short':>9}")
    for row in table:
        by_band = [0] * len(BAND_NAMES)
        for cell, n in row.by_cell.items():
            if "|" in cell:
                by_band[int(cell.rsplit("|", 1)[1])] += n
        print(
            f"  {row.rule.name:<12}"
            + "".join(f"{n:>9}" for n in by_band)
            + f"{row.by_cell.get('short_circuit', 0):>9}"
        )
    if show_cells:
        report_cells(articles, table)


def report_cells(articles: list[ConfirmedArticle], table: list[RelabelRow]) -> None:
    """The slug x band grid of the confirmed population and, per distinct rule, of the
    articles it re-labels (the population and per-cell counts T-121(a) asks for)."""
    population = Counter(cell_of(a.group_score, a.leaf_scores) for a in articles)

    def grid(counts: Mapping[str, int]) -> None:
        print(f"  {'slug':<30}" + "".join(f"{b:>9}" for b in BAND_NAMES))
        for slug in CATEGORY_SLUGS:
            print(
                f"  {slug:<30}"
                + "".join(f"{counts.get(f'{slug}|{b}', 0):>9}" for b in range(len(BAND_NAMES)))
            )
        print(f"  short-circuit stratum: {counts.get('short_circuit', 0)}")

    section("Confirmed articles per cell (the population)")
    grid(population)
    seen: dict[tuple[tuple[str, int], ...], str] = {}
    for row in table:
        key = tuple(sorted(row.by_cell.items()))
        if not key:
            continue
        if key in seen:
            section(
                f"Re-labelled per cell: {row.rule.name} re-labels the same articles as {seen[key]}"
            )
            continue
        seen[key] = row.rule.name
        section(f"Re-labelled per cell: {row.rule.name} ({row.changed})")
        grid(row.by_cell)


def accept_ids(conn: sqlite3.Connection) -> set[int]:
    return {
        i
        for (i,) in conn.execute("SELECT id FROM main.discovered_urls WHERE match_class = 'accept'")
    }


def report_verdicts(
    conn: sqlite3.Connection, run_id: int, rules: list[Rule], reps: int, seed: int
) -> None:
    pairs = load_category_pairs(conn, run_id)
    accept = accept_ids(conn)
    domain = domain_pairs(pairs, lambda a: a in accept)
    section(f"eval_run {run_id}, `accept` partition: stored verdicts under the tightening rules")
    print(f"  judged items in the partition: {len(domain)}")
    print(
        f"  {'rule':<12}{'other_prec':>11}{'other_rec':>10}{'acc':>7}{'macroF1':>9}"
        f"{'mean slug prec':>16}{'other rate':>12}"
    )
    table = verdict_table(domain, rules)
    for name, m in table.items():
        print(
            f"  {name:<12}{m.get('precision_other', 0):>11.3f}{m.get('recall_other', 0):>10.3f}"
            f"{m['accuracy_vs_judge']:>7.3f}{m['macro_f1']:>9.3f}"
            f"{m['mean_slug_precision']:>16.3f}{m['other_rate_model']:>12.3f}"
        )
    section(
        f"Rule minus shipped rule, paired stratified bootstrap ({reps} reps, 95%): "
        "same items scored under both"
    )
    print(f"  {'rule':<12}{'metric':<22}{'difference':>11}  {'interval':<20}{'used':>10}")
    done: dict[tuple[float, ...], str] = {}
    for rule in rules:
        if not rule.tightens or rule == BASELINE:
            continue
        signature = tuple(
            round(table[rule.name][k], 9) for k in ("precision_other", "recall_other")
        )
        if signature in done:
            print(f"  {rule.name:<12}same labels as {done[signature]}")
            continue
        done[signature] = rule.name
        for key in ("precision_other", "recall_other", "accuracy_vs_judge"):
            iv = paired_rule_interval(domain, rule, key, reps=reps, seed=seed)
            print(
                f"  {rule.name:<12}{key:<22}{iv.difference:>+11.3f}  "
                f"[{iv.low:+.3f}, {iv.high:+.3f}]{iv.used:>6}/{iv.reps}"
            )


def report_power(
    conn: sqlite3.Connection,
    run_id: int,
    articles: list[ConfirmedArticle],
    rules: list[Rule],
    draws: int,
    boot_reps: int,
    seed: int,
    ramp: str,
) -> None:
    pairs = load_category_pairs(conn, run_id)
    accept = accept_ids(conn)
    truth = fit_truth_model(domain_pairs(pairs, lambda a: a in accept), slug_share=0.0, ramp=ramp)
    section(f"Truth model (ramp {ramp!r}: {RAMPS[ramp]}) measured on eval_run stored verdicts")
    print(f"  `other` agreement of the stored `other` label: {truth.other_agreement:.3f}")
    print(f"  share of wrong >= 0.6 labels whose truth is `other`: {truth.other_given_wrong:.3f}")
    print(
        f"  confident cells whose agreement fell back to their band's rate "
        f"(fewer than 10 judged): {len(truth.fallback_cells)} of {len(truth.agree)}"
    )
    section(
        f"Power ({ramp} ramp) of the paired weighted `other`-precision difference (report half; "
        f"{draws} draws x {boot_reps} bootstrap reps; need >= {POWER_TARGET:.0%})"
    )
    print(f"  {'rule':<12}{'share':>7}{'implied gain':>14}{'power':>8}{'defined':>9}  note")
    seen: dict[bytes, str] = {}
    for rule in rules:
        if rule == BASELINE:
            continue
        frame = frame_for(articles, rule)
        if frame.signature in seen:
            print(f"  {rule.name:<12} labels every article as {seen[frame.signature]} does")
            continue
        seen[frame.signature] = rule.name
        shares: list[tuple[float, str]] = [(s, "swept") for s in SHARES]
        star = slug_share_for_gain(frame, truth, GAIN)
        if star is not None:
            shares.append((star, f"share at which the implied gain is +{GAIN}"))
        for share, note in shares:
            row: PowerRow = simulate_power(
                frame,
                replace(truth, slug_share=share),
                draws=draws,
                boot_reps=boot_reps,
                seed=seed,
            )
            flag = "" if row.power >= POWER_TARGET else " (not detectable at this sample)"
            print(
                f"  {rule.name:<12}{share:>7.2f}{row.implied_gain:>+14.3f}{row.power:>8.2f}"
                f"{row.defined:>9}  {note}{flag}"
            )
        if star is None:
            print(f"  {rule.name:<12}{'':>7}  +{GAIN} is not reached for any share in [0, 1]")


def default_path(env: str) -> Path | None:
    value = os.environ.get(env)
    return Path(value) if value else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source-db", type=Path, default=default_path("SOURCE_DATABASE_URL"))
    ap.add_argument("--results-db", type=Path, default=default_path("DATABASE_URL"))
    ap.add_argument("--eval-run", type=int, default=43)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--bootstrap-reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--interval-reps", type=int, default=500, help="bootstrap reps, paired intervals"
    )
    ap.add_argument("--cells", action="store_true", help="print the per-cell re-label grids")
    ap.add_argument(
        "--ramps",
        default="graded,band,flat",
        help=f"comma-separated slug-share shapes ({', '.join(RAMPS)}); the first is the default "
        "and covers every rule, the others only the loosening rules (tightening ones lose "
        "under every shape)",
    )
    args = ap.parse_args()
    if args.source_db is None or args.results_db is None:
        ap.error(
            "--source-db and --results-db (or SOURCE_DATABASE_URL / DATABASE_URL) are required"
        )
    conn = connect(args.source_db, args.results_db)
    articles = load_confirmed(conn)
    rules = candidate_rules()
    report_relabels(articles, rules, args.cells)
    report_verdicts(conn, args.eval_run, rules, args.interval_reps, args.seed)
    for i, ramp in enumerate(args.ramps.split(",")):
        subset = rules if i == 0 else [r for r in rules if not r.tightens]
        report_power(
            conn, args.eval_run, articles, subset, args.draws, args.bootstrap_reps, args.seed, ramp
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
