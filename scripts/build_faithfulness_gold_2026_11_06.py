"""Put the owner's faithfulness marks in the tracked gold file (T-155, set 4 of ``docs/gold_datasets.md``).

    uv run scripts/build_faithfulness_gold_2026_11_06.py [--out data] [--gold gold/faithfulness_gold.csv]
        [--overwrite-marks]

Reads ``data/faithfulness_to_tag.csv`` (the marks and the text the owner read) and
``data/faithfulness_ids.csv`` (the draw: which extractor wrote each ``review_order``, the text hash,
the revisions) and writes ``gold/faithfulness_gold.csv`` (``retag.FAITHFULNESS_GOLD_COLUMNS``): the
mark and the ids only, no news text, keyed by (``article_id``, ``ticker``, ``text_hash``) and not by
``review_order``. The run stops unless

* every row is marked yes / no and the marked file holds exactly the rows of the draw;
* each pair agrees on article and company, and the hash of the text the owner read is the hash the
  draw recorded (so the mark belongs to that text and the file was not edited while tagging).

``text_hash``, ``extractors`` and ``revisions`` are copied from the ids, never recomputed. The gold
file is rewritten from the two data files each time, but a gold file that already exists is the
reference: if a mark, a text hash or a row of it would change or go missing the run stops and names
how many (``--overwrite-marks`` to accept). Adding the model's texts (``extractors`` and
``revisions`` of a row gaining a name, new rows) changes none of that and goes through.

``faithfulness_coverage.csv`` (the companies an extractor named and wrote no text for, which the D32
share counts as unfaithful) is copied next to the gold file, so ``report_wi18 faithfulness --gold``
needs nothing from ``data/``. It must exist: an empty one (header only) says that none was left out.
Both files are written together (temporary files, then replaced), and a tracked coverage entry the new
file lacks stops the run like a changed mark.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import gold_metrics, retag

TO_TAG_NAME = "faithfulness_to_tag.csv"
IDS_NAME = "faithfulness_ids.csv"
COVERAGE_NAME = gold_metrics.COVERAGE_NAME
COVERAGE_COLUMNS = gold_metrics.COVERAGE_COLUMNS


def read_coverage(path: Path) -> list[dict[str, str]]:
    return gold_metrics.read_coverage(path, hint=" (draw_faithfulness writes it)")


def lost_coverage(before: list[dict[str, str]], after: list[dict[str, str]]) -> list[str]:
    """Companies the tracked coverage lists that the new one would not: each is a text the D32 share
    counts as unfaithful, so dropping one moves the figure up."""
    now = {(r["article_id"], r["ticker"], r["extractor"]) for r in after}
    return [
        f"{r['article_id']}#{r['ticker']}#{r['extractor']}"
        for r in before
        if (r["article_id"], r["ticker"], r["extractor"]) not in now
    ]


def changed_marks(before: list[dict[str, str]], after: list[dict[str, str]]) -> list[str]:
    """Keys of the existing gold that the new rows would change (mark or text hash) or drop."""
    now = {r["key"]: r for r in after}
    return [
        r["key"]
        for r in before
        if r["key"] not in now
        or now[r["key"]]["faithful"] != r["faithful"]
        or now[r["key"]]["text_hash"] != r["text_hash"]
    ]


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


def render(fields: tuple[str, ...], rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold/faithfulness_gold.csv"))
    parser.add_argument(
        "--overwrite-marks",
        action="store_true",
        help="accept that marks, text hashes or rows of the existing gold file change or go",
    )
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
        coverage = read_coverage(args.out / COVERAGE_NAME)
    except gold_metrics.MarkedFileError as exc:
        sys.exit(str(exc))
    coverage_path = args.gold.with_name(COVERAGE_NAME)
    if not args.overwrite_marks:
        problems = []
        if args.gold.exists():
            with args.gold.open(encoding="utf-8-sig", newline="") as fh:
                moved = changed_marks(list(csv.DictReader(fh)), rows)
            if moved:
                problems.append(
                    f"{len(moved)} rows of {args.gold} would change or go missing (first {moved[0]})"
                )
        if coverage_path.exists():
            try:
                gone = lost_coverage(gold_metrics.read_coverage(coverage_path), coverage)
            except gold_metrics.MarkedFileError as exc:
                sys.exit(f"the tracked {coverage_path} is not usable: {exc}")
            if gone:
                problems.append(
                    f"{len(gone)} companies without a text in {coverage_path} would go missing "
                    f"(first {gone[0]})"
                )
        if problems:
            sys.exit(
                "; ".join(problems)
                + ": the gold files are the reference; run with --overwrite-marks if this is intended"
            )
    args.gold.parent.mkdir(parents=True, exist_ok=True)
    try:
        gold_metrics.write_pair_atomically(
            {
                args.gold: render(retag.FAITHFULNESS_GOLD_COLUMNS, retag.spreadsheet_safe(rows)),
                coverage_path: render(COVERAGE_COLUMNS, coverage),
            }
        )
    except OSError as exc:
        sys.exit(f"cannot write {args.gold} or {coverage_path} ({exc}); both are as they were")
    print(
        f"{args.gold}: {len(rows)} rows; {coverage_path}: {len(coverage)} companies without a text"
    )


if __name__ == "__main__":
    main()
