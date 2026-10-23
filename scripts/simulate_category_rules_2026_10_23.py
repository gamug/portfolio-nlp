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
   with a swept probability that a 0.5-0.6 article is truly `other` (the average over the
   articles labelled `other` stays at the measured precision), and at the rate where the
   model's own implied gain is +0.05, for each shape in `--shapes`; the closed-form bound
   of T-123 (how many removed articles may be truly `other`) needs no truth model.

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
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from news_nlp.eval.calibration import (
    BAND_NAMES,
    BASELINE,
    POWER_TARGET,
    RECALL_TOLERANCE,
    SHAPES,
    ConfirmedArticle,
    Frame,
    RelabelRow,
    Rule,
    TruthModel,
    apply_rule,
    baseline_under,
    candidate_rules,
    cell_of,
    classify_power,
    confident_other_rates,
    fit_truth_model,
    frame_for,
    implied_changes,
    loosening_bound,
    low_band_counts,
    needs_refinement,
    null_critical,
    paired_rule_interval,
    rate_for_gain,
    rate_for_recall,
    relabel_table,
    shaped,
    simulate_power,
    verdict_table,
)
from news_nlp.eval.metrics import aggregate_category
from news_nlp.eval.partition import domain_pairs, load_category_pairs
from news_nlp.taxonomy import CATEGORY_SLUGS

GAIN = 0.05
RATES = (0.30, 0.20, 0.10, 0.05)  # P(a 0.5-0.6 article is truly `other`); measured average 0.319
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


SAME_RATE = 0.002  # a swept rate this close to the rate that reaches +GAIN is that row
CALIBRATION_TOLERANCE = 0.005  # the model's baseline `other` precision vs the measured one


class PowerSettings(NamedTuple):
    draws: int
    boot_reps: int
    seed: int
    refine_draws: int
    null_seeds: int


class Scenario(NamedTuple):
    """The measured truth model, the per-band counts of articles labelled `other` that
    :func:`shaped` calibrates against, and the shape of the swept rate."""

    truth: TruthModel
    counts: dict[int, int]
    shape: str
    recall: float  # the measured `other` recall of the stored run


def rate_rows(frame: Frame, sc: Scenario) -> tuple[list[tuple[float, bool]], bool]:
    """The rates (probability that a 0.5-0.6 article is truly `other`) to simulate for one
    rule, and whether +GAIN is reached at any rate: the sweep, with the row at which the
    implied gain is +GAIN marked (merged into the sweep row when it is the same rate)."""
    rows: list[tuple[float, bool]] = [(r, False) for r in RATES]
    star = rate_for_gain(frame, sc.truth, sc.counts, sc.shape, GAIN)
    if star is None:
        return rows, False
    near = [i for i, (r, _) in enumerate(rows) if abs(r - star) < SAME_RATE]
    if near:
        rows[near[0]] = (star, True)
    else:
        rows.append((star, True))
    return rows, True


def check_calibration(frame: Frame, shaped_truth: TruthModel, measured: float) -> None:
    """The shaped model must reproduce the measured baseline `other` precision, or its
    gains and recall losses describe a world the stored judgements contradict."""
    precision, _ = baseline_under(frame, shaped_truth)
    if abs(precision - measured) > CALIBRATION_TOLERANCE:
        raise SystemExit(
            f"truth model baseline `other` precision {precision:.3f} differs from the "
            f"measured {measured:.3f}"
        )


def power_row(
    frame: Frame,
    sc: Scenario,
    rate: float,
    at_target: bool,
    cfg: PowerSettings,
    null_cache: dict[tuple[bytes, int], float],
) -> None:
    shared = shaped(sc.truth, sc.counts, rate, sc.shape)
    check_calibration(frame, shared, sc.truth.other_agreement)
    row = simulate_power(frame, shared, draws=cfg.draws, boot_reps=cfg.boot_reps, seed=cfg.seed)
    rerun = cfg.refine_draws > cfg.draws and needs_refinement(row.power)
    if rerun:
        row = simulate_power(
            frame, shared, draws=cfg.refine_draws, boot_reps=cfg.boot_reps, seed=cfg.seed
        )
    implied = implied_changes(frame, shared)
    precision, recall = baseline_under(frame, shared)
    note = (
        f"+{GAIN} reached [{classify_power(row.power, row.power_se)}]"
        if at_target
        else "power at this gain"
    )
    if rerun:
        note += f"; {cfg.refine_draws} draws, one stream, same seed"
    print(
        f"  {frame.rule.name:<12}{rate:>6.3f}{shared.low_rate(0):>7.3f}{implied.precision:>+8.3f}"
        f"{implied.recall:>+9.3f}{implied.accuracy:>+7.3f}{row.power:>7.2f}{row.power_se:>7.3f}"
        f"{row.joint_power:>7.2f}{row.defined:>9}{row.sd_diff:>7.4f}"
        f"  base {precision:.3f}/{recall:.3f}  {note}"
    )
    if rerun and at_target:  # only the verdict row near the bar needs the calibrated check
        report_calibrated(frame, shared, sc.truth, cfg, null_cache)


def report_joint(
    articles: list[ConfirmedArticle], rules: list[Rule], sc: Scenario, cfg: PowerSettings
) -> None:
    section(
        f"Joint condition ({sc.shape} shape): the largest probability that a 0.5-0.6 article is "
        f"truly `other` at which the implied `other` recall stays within {RECALL_TOLERANCE}, what "
        "the rule gains there, and the joint power there and at rate 0 (the best case for recall)"
    )
    print(
        f"  {'rule':<12}{'rate':>6}{'d prec':>8}{'d recall':>9}{'d acc':>7}"
        f"{'joint@rate':>12}{'joint@0':>9}"
    )
    for rule in rules:
        frame = frame_for(articles, rule)
        rate = rate_for_recall(frame, sc.truth, sc.counts, sc.shape)
        if rate is None:
            print(f"  {rule.name:<12}  not within {RECALL_TOLERANCE} at any rate")
            continue
        at = implied_changes(frame, shaped(sc.truth, sc.counts, rate, sc.shape))
        joint = [
            simulate_power(
                frame,
                shaped(sc.truth, sc.counts, r, sc.shape),
                draws=cfg.draws,
                boot_reps=cfg.boot_reps,
                seed=cfg.seed,
            ).joint_power
            for r in (rate, 0.0)
        ]
        print(
            f"  {rule.name:<12}{rate:>6.3f}{at.precision:>+8.3f}{at.recall:>+9.3f}"
            f"{at.accuracy:>+7.3f}{joint[0]:>12.2f}{joint[1]:>9.2f}"
        )


def report_bounds(
    articles: list[ConfirmedArticle], rules: list[Rule], truth: TruthModel, recall: float
) -> None:
    """The closed-form bound of T-123 for each loosening rule: no truth model, only the
    measured `other` precision and recall of the stored run."""
    section(
        f"What a loosening rule must achieve to raise `other` precision by {GAIN} with `other` "
        f"recall within {RECALL_TOLERANCE} (measured precision {truth.other_agreement:.3f}, "
        f"recall {recall:.3f}; a necessary condition, the 95% lower bound needs more room)"
    )
    print(f"  {'rule':<12}{'removed':>9}{'max truly other':>17}{'max rate':>10}{'min removed':>13}")
    seen: set[bytes] = set()
    for rule in rules:
        frame = frame_for(articles, rule)
        if frame.signature in seen:
            continue
        seen.add(frame.signature)
        removed = int(frame.base_other.sum() - frame.cand_other.sum())
        pure = not (frame.cand_other & ~frame.base_other).any()
        if removed <= 0 or not pure:
            continue  # a rule that also adds articles to `other` is not a pure loosening
        bound = loosening_bound(
            int(frame.base_other.sum()), removed, truth.other_agreement, recall, gain=GAIN
        )
        print(
            f"  {rule.name:<12}{removed:>9}{bound.max_truth_other:>17.0f}"
            f"{100 * bound.max_rate:>9.1f}%{bound.min_removed:>13.0f}"
        )
    rates = confident_other_rates(truth, articles)
    print(
        "  for comparison, articles the shipped rule labels with a slug are truly `other` in "
        + " and ".join(f"{100 * rates[b]:.1f}% ({BAND_NAMES[b]})" for b in sorted(rates))
    )


def report_power(
    conn: sqlite3.Connection,
    run_id: int,
    articles: list[ConfirmedArticle],
    rules: list[Rule],
    cfg: PowerSettings,
    shape: str,
    null_cache: dict[tuple[bytes, int], float],
    *,
    bounds: bool,
) -> None:
    accept = accept_ids(conn)
    pairs = domain_pairs(load_category_pairs(conn, run_id), lambda a: a in accept)
    truth = fit_truth_model(pairs)
    measured = aggregate_category([it for it, _ in pairs], [v for _, v in pairs])
    sc = Scenario(truth, low_band_counts(articles), shape, measured["recall_other"])
    section(f"Truth model ({shape} shape) measured on eval_run {run_id}'s stored verdicts")
    print(
        f"  `other` precision of the stored label (the calibration target): {truth.other_agreement:.3f}"
    )
    print(f"  `other` recall: {sc.recall:.3f}; articles labelled `other` per low band: {sc.counts}")
    print(f"  share of wrong >= 0.6 labels whose truth is `other`: {truth.other_given_wrong:.3f}")
    print(
        f"  confident cells whose agreement fell back to their band's rate "
        f"(fewer than 10 judged): {len(truth.fallback_cells)} of {len(truth.agree)}"
    )
    print(f"    {', '.join(truth.fallback_cells) or '(none)'}")
    if bounds:
        report_bounds(articles, rules, truth, sc.recall)
    section(
        f"Power ({shape} shape) of the paired weighted `other`-precision difference (report half; "
        f"{cfg.draws} draws x {cfg.boot_reps} bootstrap reps; need >= {POWER_TARGET:.0%} at "
        f"+{GAIN}). `rate` is the probability that a 0.5-0.6 article is truly `other`, `rest` "
        f"that of the other low cells (calibrated so the average stays at the measured value); "
        f"`d recall` is the implied change in `other` recall (T-123: lower bound >= "
        f"{RECALL_TOLERANCE}); `joint` also requires the recall and accuracy lower bounds to "
        "stay within it; `base` is the model's baseline `other` precision/recall"
    )
    print(
        f"  {'rule':<12}{'rate':>6}{'rest':>7}{'d prec':>8}{'d recall':>9}{'d acc':>7}{'power':>7}"
        f"{'(se)':>7}{'joint':>7}{'defined':>9}{'sd':>7}  note"
    )
    seen: dict[bytes, str] = {}
    simulated: list[Rule] = []
    for rule in rules:
        if rule == BASELINE:
            continue
        frame = frame_for(articles, rule)
        if frame.signature in seen:
            print(f"  {rule.name:<12} labels every article as {seen[frame.signature]} does")
            continue
        seen[frame.signature] = rule.name
        simulated.append(rule)
        rows, reached = rate_rows(frame, sc)
        for rate, at_target in rows:
            power_row(frame, sc, rate, at_target, cfg, null_cache)
        if not reached:
            print(f"  {rule.name:<12}{'':>6}  +{GAIN} is not reached at any rate")
    report_joint(articles, [r for r in simulated if not r.tightens], sc, cfg)


def report_calibrated(
    frame: Frame,
    shared: TruthModel,
    truth: TruthModel,
    cfg: PowerSettings,
    null_cache: dict[tuple[bytes, int], float],
) -> None:
    """The power at the null-calibrated bound, once per null seed: the interval's
    false-positive rate is above nominal (see the null rows under the flat ramp), and the
    critical value is itself a 2.5% tail of *draws* draws, so its spread shows how far the
    calibrated verdict can be trusted."""
    draws = max(cfg.refine_draws, cfg.draws)
    powers = []
    for k in range(cfg.null_seeds):
        key = (frame.signature, k)
        if key not in null_cache:
            null_cache[key] = null_critical(
                frame, truth, draws=draws, boot_reps=cfg.boot_reps, seed=cfg.seed + 1000 + k
            )
        cal = simulate_power(
            frame,
            shared,
            draws=draws,
            boot_reps=cfg.boot_reps,
            seed=cfg.seed,
            critical=null_cache[key],
        )
        powers.append((cal, null_cache[key]))
    verdicts = {classify_power(c.power, c.power_se) for c, _ in powers}
    lo = min(c.power for c, _ in powers)
    hi = max(c.power for c, _ in powers)
    crits = ", ".join(f"{crit:+.4f}" for _, crit in powers)
    stable = next(iter(verdicts)) if len(verdicts) == 1 else "unstable across null seeds"
    print(
        f"  {'':<12}{'':>7}  null-calibrated power {lo:.2f}-{hi:.2f} "
        f"(critical values {crits}; {cfg.null_seeds} null seeds) [{stable}]"
    )


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
    ap.add_argument("--refine-draws", type=int, default=1000, help="draws for borderline powers")
    ap.add_argument(
        "--sections",
        default="relabel,verdicts,power",
        help="comma-separated parts to run: relabel, verdicts, power",
    )
    ap.add_argument("--null-seeds", type=int, default=3, help="null runs for the critical value")
    ap.add_argument("--cells", action="store_true", help="print the per-cell re-label grids")
    ap.add_argument(
        "--shapes",
        default="graded,band2,flat",
        help=f"comma-separated shapes of the truly-`other` rate ({', '.join(SHAPES)}); the first "
        "covers every rule, the others only the loosening rules (tightening ones lose under "
        "every shape)",
    )
    args = ap.parse_args()
    if args.source_db is None or args.results_db is None:
        ap.error(
            "--source-db and --results-db (or SOURCE_DATABASE_URL / DATABASE_URL) are required"
        )
    shapes = args.shapes.split(",")
    unknown = [r for r in shapes if r not in SHAPES]
    parts = args.sections.split(",")
    unknown_parts = [p for p in parts if p not in ("relabel", "verdicts", "power")]
    if unknown or unknown_parts:
        ap.error(
            f"unknown --shapes {unknown} (valid: {', '.join(SHAPES)}) or "
            f"--sections {unknown_parts} (valid: relabel, verdicts, power)"
        )
    conn = connect(args.source_db, args.results_db)
    articles = load_confirmed(conn)
    rules = candidate_rules()
    if "relabel" in parts:
        report_relabels(articles, rules, args.cells)
    if "verdicts" in parts:
        report_verdicts(conn, args.eval_run, rules, args.interval_reps, args.seed)
    null_cache: dict[tuple[bytes, int], float] = {}
    cfg = PowerSettings(
        args.draws, args.bootstrap_reps, args.seed, args.refine_draws, args.null_seeds
    )
    for i, shape in enumerate(shapes if "power" in parts else []):
        subset = rules if i == 0 else [r for r in rules if not r.tightens]
        report_power(conn, args.eval_run, articles, subset, cfg, shape, null_cache, bounds=i == 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
