"""Put the owner's label-check marks in the tracked gold file (T-155, set 3 of ``docs/gold_datasets.md``).

    uv run scripts/build_segment_check_gold_2026_11_06.py [--out data] [--gold gold/segment_check_gold.csv]

Reads ``data/segment_check_to_tag.csv`` (the marks) and ``data/segment_check_ids.csv`` (the draw: the
paragraph position and the year) and writes ``gold/segment_check_gold.csv``
(``retag.SEGMENT_CHECK_GOLD_COLUMNS``): the mark and the ids only, no news text. The run stops unless
every row is marked yes / no and the marked file holds exactly the paragraphs of the draw. The gold
file is rewritten from the two data files each time.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import gold_metrics, retag

TO_TAG_NAME = "segment_check_to_tag.csv"
IDS_NAME = "segment_check_ids.csv"


def build(marks: list[dict[str, str]], ids: list[dict[str, str]]) -> list[dict[str, str]]:
    """The gold rows, in the order of the draw. A paragraph is paired with its mark by
    ``review_order`` (the draw's own position: one article and company can have two paragraphs of
    the same kind) and the pair must agree on article, company and kind. Raises ``MarkedFileError``
    when the marks and the draw are not the same paragraphs."""
    orders = [r["review_order"] for r in marks]
    gold_metrics.require_unique(orders, TO_TAG_NAME)
    gold_metrics.require_same_keys(
        orders, [str(i) for i in range(1, len(ids) + 1)], f"{TO_TAG_NAME} against {IDS_NAME}"
    )
    by_order = {r["review_order"]: r for r in marks}
    out = []
    for position, drawn in enumerate(ids, 1):
        marked = by_order[str(position)]
        for field in ("article_id", "ticker", "kind"):
            if marked[field] != drawn[field]:
                raise gold_metrics.MarkedFileError(
                    f"row {position}: {field} is {marked[field]!r} in {TO_TAG_NAME} and "
                    f"{drawn[field]!r} in {IDS_NAME}"
                )
        out.append(
            {
                "key": drawn["key"],
                "article_id": drawn["article_id"],
                "segment_index": drawn["segment_index"],
                "ticker": drawn["ticker"],
                "kind": drawn["kind"],
                "year": drawn["year"],
                "correct": marked["correct"],
                "notes": marked["notes"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold/segment_check_gold.csv"))
    args = parser.parse_args()
    try:
        marks = gold_metrics.read_marked(
            args.out / TO_TAG_NAME,
            "correct",
            ("yes", "no"),
            required=("article_id", "ticker", "kind", "review_order"),
        )
        with (args.out / IDS_NAME).open(encoding="utf-8-sig", newline="") as fh:
            ids = list(csv.DictReader(fh))
        rows = build(marks, ids)
    except gold_metrics.MarkedFileError as exc:
        sys.exit(str(exc))
    args.gold.parent.mkdir(parents=True, exist_ok=True)
    with args.gold.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(retag.SEGMENT_CHECK_GOLD_COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(retag.spreadsheet_safe(rows))
    print(f"{args.gold}: {len(rows)} rows")


if __name__ == "__main__":
    main()
