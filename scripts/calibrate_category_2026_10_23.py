"""T-121(b): the paid, judged calibration sample of the category decision rule (PLAN.md Work
item 15 step 4; spend decision D10/T-142, cap 3,800 judge calls).

    uv run scripts/calibrate_category_2026_10_23.py plan              # free, read-only
    uv run scripts/calibrate_category_2026_10_23.py run --stage 1     # PAID: ~720 calls
    uv run scripts/calibrate_category_2026_10_23.py report            # free, reads stored verdicts
    uv run scripts/calibrate_category_2026_10_23.py run --stage 2     # PAID: the rest of the sample

The sample is ``calibration_sample.draw_sample`` (confirmed articles by slug x score band, the
0.5-0.6 band split by the candidate rules, a seeded fit/report split). Stage 1 judges the
sub-cells inside a candidate rule's removed set (A, U, V); ``report`` says which rules already
fail or are undecided, so the owner can stop before stage 2. ``run`` resumes the run of the same
seed, prompt and judge model, and never goes over ``remaining_calls`` (the D10 cap across runs).
Needs ``LLM_API_KEY`` / ``LLM_MODEL`` / ``LLM_URL`` and the ``eval`` dependency group for ``run``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import news_nlp as db
from news_nlp.eval.calibration import Rule
from news_nlp.eval.calibration_analysis import (
    judged_articles,
    report_rule,
    separation,
)
from news_nlp.eval.calibration_judge import (
    D10_CAP,
    TASK,
    judge_and_record,
    load_truths,
    remaining_calls,
    start_calibration_run,
    truth_experiment,
)
from news_nlp.eval.calibration_sample import (
    draw_sample,
    load_confirmed,
    sample_items,
    stage_of,
)
from news_nlp.eval.provenance import code_version

SEED = 20261023
RULES = (Rule(0.50, 0.20), Rule(0.55, 0.00), Rule(0.55, 0.20))


def open_store() -> db.NewsNlpDatabase:
    conn = db.connect_pipeline()
    db.init_schema(conn)  # idempotent; the eval tables exist already in the real store
    return conn


def population(conn: db.NewsNlpDatabase):  # type: ignore[no-untyped-def]
    return load_confirmed(conn, results_schema="main", urls_schema="source")


def cmd_plan(args: argparse.Namespace) -> None:
    conn = open_store()
    articles = population(conn)
    sample = draw_sample(articles, seed=args.seed)
    per_stage = Counter(stage_of(a.cell) for a in sample.drawn)
    print(f"confirmed articles: {len(articles)}; cells: {len(sample.populations)}")
    print(f"drawn: {len(sample.drawn)} (stage 1: {per_stage[1]}, stage 2: {per_stage[2]})")
    print(f"D10 cap {D10_CAP}; calls left across runs: {remaining_calls(conn)}")
    for half in ("fit", "report"):
        print(f"  {half}: {len(sample.half(half))} drawn")  # type: ignore[arg-type]


def find_run(conn: db.NewsNlpDatabase, model: str, seed: int) -> int | None:
    row = conn.execute(
        "SELECT id FROM eval_run WHERE stage = ? AND experiment = ? AND seed = ? "
        "ORDER BY id DESC LIMIT 1",
        (TASK, truth_experiment(model), seed),
    ).fetchone()
    return int(row[0]) if row else None


def cmd_run(args: argparse.Namespace) -> None:
    from news_nlp.eval.config import EvalSettings  # noqa: PLC0415 -- needs the LLM env
    from news_nlp.eval.model import calibration_agent_factory  # noqa: PLC0415 -- eval group

    settings = EvalSettings.load()
    conn = open_store()
    articles = population(conn)
    pairs, sample = sample_items(conn, draw_sample(articles, seed=args.seed))
    run_id = find_run(conn, settings.llm_model, args.seed)
    if run_id is None:
        run_id = start_calibration_run(
            conn,
            sample,
            judge_model=settings.llm_model,
            judge_url=settings.llm_url,
            code_version=code_version(),
            date=datetime.now(UTC).date().isoformat(),
        )
        print(f"started eval_run {run_id}")
    else:
        print(f"resuming eval_run {run_id}")
    items = [it for a, it in pairs if args.stage == "all" or stage_of(a.cell) == int(args.stage)]
    if args.limit:
        items = items[: args.limit]
    cap = min(remaining_calls(conn), args.max_calls)
    print(f"judging {len(items)} articles, at most {cap} calls (D10 left: {remaining_calls(conn)})")
    _, usage = judge_and_record(
        conn,
        run_id,
        items,
        agent_factory=calibration_agent_factory(settings),
        max_calls=cap,
        max_workers=settings.max_workers,
    )
    print(usage)


def cmd_report(args: argparse.Namespace) -> None:
    from news_nlp.eval.config import EvalSettings  # noqa: PLC0415

    conn = open_store()
    articles = population(conn)
    run_id = args.run_id or find_run(conn, EvalSettings.load().llm_model, args.seed)
    if run_id is None:
        sys.exit("no calibration run yet: run `run --stage 1` first")
    truths = load_truths(conn, run_id)
    judged = judged_articles(truths, articles)
    print(
        f"eval_run {run_id}: {len(truths.rows)} usable verdicts, {truths.parse_failed} "
        f"parse-failed ({100 * truths.parse_failed_rate:.1f}%), {truths.unjudged} unjudged"
    )
    print(f"calls spent {D10_CAP - remaining_calls(conn)} of {D10_CAP}")
    print(
        f"\n{'rule':<14}{'removed':>8}{'judged':>8}{'truly other':>13}{'rate':>7}"
        f"{'95%':>14}{'98.75% lo/hi':>16}{'cap':>7}{'deff':>6}  verdict"
    )
    for rule in RULES:
        r = report_rule(judged, articles, rule, final=args.final)
        if r is None:
            print(f"{rule.name:<14} removes nothing")
            continue
        print(
            f"{rule.name:<14}{r.removed:>8}{r.n:>8}{r.n_other:>13}{r.rate:>7.3f}"
            f"{f'{r.low:.3f}-{r.high:.3f}':>14}{f'{r.stop_low:.3f}/{r.stop_high:.3f}':>16}"
            f"{r.cap:>7.3f}{(r.design_effect or 0):>6.2f}  {r.verdict}"
        )
    print("\nseparation of truly-slug from truly-`other` among stored-`other` articles in 0.4-0.6")
    for s in separation(judged):
        print(
            f"  {s.feature:<12} AUC {s.auc:.3f} ({s.low:.3f}-{s.high:.3f}), "
            f"{s.n} judged, {s.n_other} truly other"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=SEED)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    run = sub.add_parser("run")
    run.add_argument("--stage", choices=("1", "2", "all"), default="1")
    run.add_argument("--limit", type=int, default=0, help="judge only the first N (smoke test)")
    run.add_argument("--max-calls", type=int, default=D10_CAP)
    report = sub.add_parser("report")
    report.add_argument("--run-id", type=int, default=0)
    report.add_argument("--final", action="store_true", help="decide on the 95% interval")
    args = parser.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
