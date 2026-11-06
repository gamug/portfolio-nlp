"""Put the owner's faithfulness marks in the tracked gold file (T-155, set 4 of ``docs/gold_datasets.md``).

    uv run scripts/build_faithfulness_gold_2026_11_06.py [--out data] [--gold gold/faithfulness_gold.csv]

Reads ``data/faithfulness_to_tag.csv`` (the marks and the text the owner read) and
``data/faithfulness_ids.csv`` (the draw: which extractor wrote each ``review_order``, the text hash,
the revisions) and writes ``gold/faithfulness_gold.csv`` (``retag.FAITHFULNESS_GOLD_COLUMNS``): the
mark and the ids only, no news text, keyed by (``article_id``, ``ticker``, ``text_hash``) and not by
``review_order``. The run stops unless

* every row is marked yes / no and the marked file holds exactly the rows of the draw;
* each pair agrees on article and company, and the hash of the text the owner read is the hash the
  draw recorded (so the mark belongs to that text and the file was not edited while tagging).

``text_hash``, ``extractors`` and ``revisions`` are copied from the ids, never recomputed. The gold
file is rewritten from the two data files each time.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import gold_metrics, retag

TO_TAG_NAME = "faithfulness_to_tag.csv"
IDS_NAME = "faithfulness_ids.csv"


def build(marks: list[dict[str, str]], ids: list[dict[str, str]]) -> list[dict[str, str]]:
    """The gold rows, in the order of the draw. Raises ``MarkedFileError`` when the marks and the
    draw are not the same texts."""
    orders = [r["review_order"] for r in marks]
    gold_metrics.require_unique(orders, TO_TAG_NAME)
    gold_metrics.require_unique([r["review_order"] for r in ids], IDS_NAME)
    gold_metrics.require_same_keys(
        orders, [r["review_order"] for r in ids], f"{TO_TAG_NAME} against {IDS_NAME}"
    )
    by_order = {r["review_order"]: r for r in marks}
    out = []
    for drawn in ids:
        marked = by_order[drawn["review_order"]]
        for field in ("article_id", "ticker"):
            if marked[field] != drawn[field]:
                raise gold_metrics.MarkedFileError(
                    f"review_order {drawn['review_order']}: {field} is {marked[field]!r} in "
                    f"{TO_TAG_NAME} and {drawn[field]!r} in {IDS_NAME}"
                )
        if retag.text_hash(marked["text"]) != drawn["text_hash"]:
            raise gold_metrics.MarkedFileError(
                f"review_order {drawn['review_order']}: the text in {TO_TAG_NAME} is not the text "
                "the draw recorded (it was edited after the draw); the mark would belong to "
                "another text"
            )
        out.append(
            {
                "key": drawn["key"],
                "article_id": drawn["article_id"],
                "ticker": drawn["ticker"],
                "text_hash": drawn["text_hash"],
                "extractors": drawn["extractors"],
                "revisions": drawn["revisions"],
                "pub_date": drawn["pub_date"],
                "faithful": marked["faithful"],
                "notes": marked["notes"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold/faithfulness_gold.csv"))
    args = parser.parse_args()
    try:
        marks = gold_metrics.read_marked(
            args.out / TO_TAG_NAME,
            "faithful",
            ("yes", "no"),
            required=("article_id", "ticker", "review_order", "text"),
        )
        with (args.out / IDS_NAME).open(encoding="utf-8-sig", newline="") as fh:
            ids = list(csv.DictReader(fh))
        rows = build(marks, ids)
    except gold_metrics.MarkedFileError as exc:
        sys.exit(str(exc))
    args.gold.parent.mkdir(parents=True, exist_ok=True)
    with args.gold.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(retag.FAITHFULNESS_GOLD_COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(retag.spreadsheet_safe(rows))
    print(f"{args.gold}: {len(rows)} rows")


if __name__ == "__main__":
    main()
