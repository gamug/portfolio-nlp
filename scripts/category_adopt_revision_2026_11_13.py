"""Carry the finished work of an older revision of the category step over to the revision this code writes (T-168).

The revision of the step follows its inputs (the summary and the companies revisions) and its code, so a change of either starts a new, empty revision. The
rows of the old one are not lost, but the 3 hours of work look lost. When the scoring is the same (the pinned model, the definitions, the premise) and so are the
inputs, this copies the items the old revision finished under the new revision, and the next run does only what is missing. **Nothing is deleted**: the old
revision stays until `RECATEGORY_SQL` is run for it.

    uv run scripts/category_adopt_revision_2026_11_13.py --from-revision category-... --summary-revision summary-... --companies-revision companies-...
    ... --check 20      # first re-score 20 stored pairs with this code; refuse if a score differs (default 20, 0 skips it)
    ... --yes           # write; without it the script only reports

An item is adopted when it is finished at the old revision and not yet at the new one, has a text at the summary revision, and its stored companies are exactly
the main companies of the companies revision. The rest is left for the run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import news_nlp as db  # noqa: E402

TOLERANCE = 1e-3  # an entailment probability re-scored on another device or run may differ in the last digits


def adopt(
    conn: Any, old: str, new: str, summary_revision: str, companies_revision: str, *, execute: bool
) -> dict[str, int]:
    """Copy the finished items of revision ``old`` to ``new`` (rows and states, one transaction); with ``execute`` False only count them."""
    if old == new:
        raise ValueError("the old and the new revision are the same")
    finished = {
        (str(k), str(i)): str(s)
        for k, i, s in conn.execute(
            "SELECT item_kind, item_id, status FROM pipeline_step_state "
            "WHERE step = 'category' AND model_revision = ? AND status IN ('done', 'skipped')",
            (old,),
        )
    }
    at_new = {
        (str(k), str(i))
        for k, i in conn.execute(
            "SELECT item_kind, item_id FROM pipeline_step_state WHERE step = 'category' AND model_revision = ?",
            (new,),
        )
    }
    stored: dict[tuple[str, str], set[str]] = {}
    for k, i, t in conn.execute(
        "SELECT item_kind, item_id, ticker FROM news_category WHERE model_revision = ?", (old,)
    ):
        stored.setdefault((str(k), str(i)), set()).add(str(t))
    mains: dict[tuple[str, str], set[str]] = {}
    for k, i, t in conn.execute(
        "SELECT item_kind, item_id, ticker FROM news_company WHERE model_revision = ? AND is_main = 1",
        (companies_revision,),
    ):
        mains.setdefault((str(k), str(i)), set()).add(str(t))
    with_text = {
        (str(k), str(i))
        for k, i in conn.execute(
            "SELECT item_kind, item_id FROM news_summary WHERE model_revision = ?",
            (summary_revision,),
        )
    }
    counts = {
        "finished_at_old": len(finished),
        "already_at_new": 0,
        "no_text": 0,
        "companies_differ": 0,
        "adoptable": 0,
    }
    chosen: list[tuple[str, str]] = []
    for item, status in finished.items():
        if item in at_new:
            counts["already_at_new"] += 1
        elif item not in with_text:
            counts["no_text"] += 1
        elif (status == "done" and stored.get(item) != mains.get(item)) or (
            status == "skipped" and item not in mains
        ):
            counts["companies_differ"] += 1
        else:
            chosen.append(item)
    counts["adoptable"] = len(chosen)
    if not execute or not chosen:
        return counts
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _adopt (item_kind TEXT, item_id TEXT)")
    conn.execute("DELETE FROM _adopt")
    conn.executemany("INSERT INTO _adopt VALUES (?, ?)", chosen)
    conn.execute(
        "INSERT OR IGNORE INTO news_category "
        "(item_kind, item_id, ticker, model_revision, label, score, group_label, group_score, slug_scores, model_name, processed_at) "
        "SELECT c.item_kind, c.item_id, c.ticker, ?, c.label, c.score, c.group_label, c.group_score, c.slug_scores, c.model_name, c.processed_at "
        "FROM news_category c JOIN _adopt a ON a.item_kind = c.item_kind AND a.item_id = c.item_id WHERE c.model_revision = ?",
        (new, old),
    )
    conn.execute(
        "INSERT OR IGNORE INTO pipeline_step_state "
        "(item_kind, item_id, step, model_revision, status, attempts, error, started_at, finished_at) "
        "SELECT s.item_kind, s.item_id, s.step, ?, s.status, s.attempts, s.error, s.started_at, s.finished_at "
        "FROM pipeline_step_state s JOIN _adopt a ON a.item_kind = s.item_kind AND a.item_id = s.item_id "
        "WHERE s.step = 'category' AND s.model_revision = ?",
        (new, old),
    )
    conn.execute("DROP TABLE _adopt")
    conn.commit()
    return counts


def check_sample(
    conn: Any, old: str, summary_revision: str, companies_revision: str, n: int
) -> int:
    """Re-score ``n`` stored pairs of the old revision with this code and return how many differ by more than ``TOLERANCE`` in any label."""
    from cut_stage import companies_of  # noqa: PLC0415 -- the universe, as the step reads it
    from news_category_stage import CategoryFinder  # noqa: PLC0415 -- loads torch

    names = {c.ticker: c.name for c in companies_of(conn)}
    pairs = conn.execute(
        "SELECT c.item_kind, c.item_id, c.ticker, c.slug_scores, s.summary_text FROM news_category c "
        "JOIN news_summary s ON s.item_kind = c.item_kind AND s.item_id = c.item_id AND s.model_revision = ? "
        "WHERE c.model_revision = ? ORDER BY c.rowid",
        (summary_revision, old),
    ).fetchall()
    sample = random.Random(0).sample(pairs, min(n, len(pairs)))  # noqa: S311 -- a reproducible sample
    finder = CategoryFinder(summary_revision, companies_revision)
    again = finder.classify([(str(r[4]), names.get(str(r[2]), str(r[2]))) for r in sample])
    bad = 0
    for row, new in zip(sample, again, strict=True):
        was = {k: float(v) for k, v in json.loads(row[3]).items() if not k.startswith("_")}
        if any(abs(was[k] - v) > TOLERANCE for k, v in new.slug_scores.items() if k in was):
            bad += 1
    return bad


def main() -> None:
    db.env.load_env_file()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-revision", required=True)
    ap.add_argument("--summary-revision", required=True)
    ap.add_argument("--companies-revision", required=True)
    ap.add_argument(
        "--check", type=int, default=20, help="pairs to re-score first (0 skips the check)"
    )
    ap.add_argument("--yes", action="store_true", help="write; without it the script only reports")
    ap.add_argument("--source-db", default=None)
    ap.add_argument("--results-db", default=None)
    args = ap.parse_args()
    from news_category_stage import code_revision  # noqa: PLC0415 -- imports torch, loads no model

    new = code_revision(args.summary_revision, args.companies_revision)
    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    report: dict[str, Any] = {"from": args.from_revision, "to": new}
    if args.check:
        bad = check_sample(
            conn, args.from_revision, args.summary_revision, args.companies_revision, args.check
        )
        report["checked_pairs"], report["pairs_that_differ"] = args.check, bad
        if bad:
            print(json.dumps(report, indent=2))
            sys.exit(
                "nothing adopted: this code scores some of the stored pairs differently, so the old rows are not what it would write"
            )
    report.update(
        adopt(
            conn,
            args.from_revision,
            new,
            args.summary_revision,
            args.companies_revision,
            execute=args.yes,
        )
    )
    report["written"] = bool(args.yes)
    print(json.dumps(report, indent=2))
    if not args.yes:
        print("report only: add --yes to write (nothing is deleted either way)", file=sys.stderr)


if __name__ == "__main__":
    main()
