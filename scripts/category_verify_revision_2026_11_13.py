"""Check that the rows of a revision of the category step are what the current code writes (T-168): re-score stored pairs and compare. **Read only**: both
stores are opened with ``mode=ro``; nothing is written.

    uv run scripts/category_verify_revision_2026_11_13.py --revision category-65f33b55b26b --summary-revision summary-... --companies-revision companies-... \
        --results-db D:\\thesis\\data\\nlp.db --source-db D:\\thesis\\data\\urls.db [--per-label 72]

Takes up to ``--per-label`` stored pairs of each label (a seeded draw, so the rare labels are covered), re-scores them with this code and reports, per label, how
many labels differ and the largest difference of any of the 14 scores. Exits with 1 if any label differs or a score differs by more than ``TOLERANCE``.
This is the evidence behind ``LEGACY_REVISIONS`` in ``src/news_category_stage.py``: the figures are in ``docs/category_results_2026_11_13/README.md``.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TOLERANCE = 1e-3


def compare(stored: dict[str, Any], label: str, new: Any) -> tuple[bool, float]:
    """Whether the label differs and the largest difference of a score, stored (a JSON of the scores, ``_`` keys apart) against a ``CategoryScore``."""
    was = {k: float(v) for k, v in stored.items() if not k.startswith("_")}
    worst = max((abs(was[k] - v) for k, v in new.slug_scores.items() if k in was), default=0.0)
    return new.label != label, worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--revision", required=True)
    ap.add_argument("--summary-revision", required=True)
    ap.add_argument("--companies-revision", required=True)
    ap.add_argument("--results-db", required=True)
    ap.add_argument("--source-db", required=True)
    ap.add_argument("--per-label", type=int, default=72)
    args = ap.parse_args()
    from news_category_stage import CategoryFinder  # noqa: PLC0415 -- imports torch

    results = sqlite3.connect(f"file:{args.results_db}?mode=ro", uri=True)
    source = sqlite3.connect(f"file:{args.source_db}?mode=ro", uri=True)
    names = dict(
        source.execute(  # the universe as the step reads it: the last name of a ticker in order wins
            "SELECT DISTINCT ticker, company FROM articles WHERE fetch_status = 'ok' ORDER BY ticker, company"
        ).fetchall()
    )
    by_label: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for ticker, label, scores, text in results.execute(
        "SELECT c.ticker, c.label, c.slug_scores, s.summary_text FROM news_category c "
        "JOIN news_summary s ON s.item_kind = c.item_kind AND s.item_id = c.item_id AND s.model_revision = ? "
        "WHERE c.model_revision = ? ORDER BY c.rowid",
        (args.summary_revision, args.revision),
    ):
        by_label[str(label)].append((str(ticker), str(label), str(scores), str(text)))
    rng = random.Random(0)  # noqa: S311 -- a reproducible draw
    sample = [
        row
        for label in sorted(by_label)
        for row in rng.sample(by_label[label], min(args.per_label, len(by_label[label])))
    ]
    finder = CategoryFinder(args.summary_revision, args.companies_revision)
    again = finder.classify([(r[3], names.get(r[0], r[0])) for r in sample])
    per_label: dict[str, dict[str, Any]] = {}
    for row, new in zip(sample, again, strict=True):
        differs, worst = compare(json.loads(row[2]), row[1], new)
        entry = per_label.setdefault(
            row[1], {"pairs": 0, "labels_differ": 0, "largest_score_difference": 0.0}
        )
        entry["pairs"] += 1
        entry["labels_differ"] += int(differs)
        entry["largest_score_difference"] = max(entry["largest_score_difference"], worst)
    total = {
        "revision": args.revision,
        "pairs_rescored": len(sample),
        "of_stored_pairs": sum(len(v) for v in by_label.values()),
        "labels_differ": sum(e["labels_differ"] for e in per_label.values()),
        "largest_score_difference": max(
            (e["largest_score_difference"] for e in per_label.values()), default=0.0
        ),
        "per_label": per_label,
    }
    print(json.dumps(total, indent=2))
    if total["labels_differ"] or total["largest_score_difference"] > TOLERANCE:
        sys.exit("the stored rows are NOT what this code writes")


if __name__ == "__main__":
    main()
