# ruff: noqa: E402  -- one-shot script
"""Does a body shared by several articles sit in two folds of the category cross-validation? (T-187, the caveat of ``docs/category_training.md`` section 8)

    uv run scripts/category_body_overlap_2026_11_15.py --source-db <urls.db> [--out docs/category_results_2026_11_15/body_overlap.md]

Read-only on the SOURCE database. The folds group by ``article_id``, which cannot see that a body is the text of several articles (about a third of the ``ok``
bodies are shared by several tickers, ``docs/category_step.md``). For the 1,216 rows of the selected run it reads the committed folds
(``docs/category_results_2026_11_15/scores/category_single/``), the body of each article from SOURCE, hashes it with ``triage.body_hash`` and counts the bodies
that are the text of more than one article and have rows in more than one fold; then the accuracy of the committed predictions with and without those rows.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import category_multilabel as ml
from news_nlp import triage

RESULTS = ROOT / "docs" / "category_results_2026_11_15"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-db", required=True)
    ap.add_argument("--out", default=str(RESULTS / "body_overlap.md"))
    args = ap.parse_args()
    folder = RESULTS / "scores" / "category_single"
    with (folder / "rows_single.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    folds = [int(x) for x in np.load(folder / "oof_single.npz")["fold"]]
    articles = [int(r["article_id"]) for r in rows]
    src = sqlite3.connect(f"file:{args.source_db}?mode=ro", uri=True)
    ids = sorted(set(articles))
    hashes: dict[int, str] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        marks = ",".join("?" * len(chunk))
        for a, body in src.execute(
            f"SELECT id, body_text FROM articles WHERE id IN ({marks})",  # noqa: S608 -- placeholders only
            chunk,
        ):
            hashes[int(a)] = triage.body_hash(body)
    missing = [a for a in ids if a not in hashes]
    if missing:
        sys.exit(f"{len(missing)} articles have no body in SOURCE (first: {missing[:5]})")
    res = ml.body_overlap(articles, hashes, folds)
    with (RESULTS / "single_predictions.csv").open(encoding="utf-8", newline="") as f:
        pred = list(csv.DictReader(f))
    bad = set(res["crossing_articles"])  # type: ignore[arg-type]
    rest = [p for p in pred if int(p["article_id"]) not in bad]
    hit = [p for p in pred if int(p["article_id"]) in bad]

    def acc(rs: list[dict[str, str]]) -> float:
        return sum(p["gold"] == p["top"] for p in rs) / len(rs) if rs else float("nan")

    lines = [
        "# Bodies shared by articles and the folds of the selected run (2026-11-15)",
        "",
        "`uv run scripts/category_body_overlap_2026_11_15.py --source-db <urls.db>`; the folds are the committed ones.",
        "",
        f"* articles: {res['articles']}; distinct bodies (`triage.body_hash`): {res['bodies']}",
        f"* bodies that are the text of more than one article: {res['shared_bodies']}",
        f"* of those, in more than one fold: {res['crossing_bodies']} (articles {res['crossing_articles']})",
        f"* rows in them: {len(hit)} of {len(pred)} ({len(hit) / len(pred):.1%}); accuracy of the committed predictions on them {acc(hit):.3f}",
        f"* accuracy on all rows {acc(pred):.4f}; without those rows ({len(rest)}) {acc(rest):.4f}",
        "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
