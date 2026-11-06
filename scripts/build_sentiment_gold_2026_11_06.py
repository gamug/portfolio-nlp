"""Put the owner's sentiment marks in the tracked gold file (T-155, set 5 of ``docs/gold_datasets.md``).

    uv run scripts/build_sentiment_gold_2026_11_06.py [--out data] [--gold gold/sentiment_gold.csv]
        [--overwrite-marks]

Reads ``data/sentiment_gold_to_tag.csv`` (the marks and the text the owner read) and
``data/sentiment_gold_ids.csv`` (the draw: part, stratum, extractor, the model's prediction, the
stratum size, the text hash and the revision of the way the text was made) and writes
``gold/sentiment_gold.csv`` (``retag.SENTIMENT_GOLD_COLUMNS``): the mark and the ids only, no news
text. The two files are paired by key (``<article_id>#<ticker or kind>``, the draw's own key; there
is no ``review_order`` in the draw). The run stops unless

* every row is marked positive / negative / neutral and the marked file holds exactly the texts of
  the draw, once each;
* each pair agrees on article, company and kind, and the hash of the text the owner read is the
  hash the draw recorded (so the mark belongs to that text and the file was not edited while
  tagging);
* every row of the draw has the stratum, the prediction, the part and the stratum size the figures
  of D32 need, and the sample is the one drawn (``gold_metrics.check_sentiment_design``:
  ``gold_metrics.SENTIMENT_DESIGN`` texts of each predicted class in each part, all of a stratum
  that is smaller, one whole-number size per stratum: the same design ``report_wi18 --gold`` checks).

``text_hash`` and ``revision`` are copied from the ids, never recomputed; ``pub_date`` is the one the
draw wrote in the file the owner tagged (the draw's ids file has none). A gold file that already
exists is the reference: if a mark, a text hash or any column the figures depend on (``PROTECTED``)
would change, or a row would go missing, the run stops and names how many (``--overwrite-marks`` to accept). The file is replaced as a whole
(temporary file, then replace).

The texts cannot be rebuilt (a summary is generated, a company text depends on the extractor): the
only copy of them is ``data/sentiment_gold_ids.csv``. Back it up.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import gold_metrics, retag

TO_TAG_NAME = "sentiment_gold_to_tag.csv"
IDS_NAME = "sentiment_gold_ids.csv"
NEEDED_FROM_THE_DRAW = ("part", "stratum", "predicted", "population", "text_hash", "revision")
#: What a rebuild may not change in a tracked row without ``--overwrite-marks``: the mark, the text
#: it belongs to, and everything that weights it or says how the text was made.
PROTECTED = (
    "sentiment",
    "text_hash",
    "revision",
    "part",
    "stratum",
    "extractor",
    "predicted",
    "population",
    "pub_date",
)


def key_of(row: dict[str, str]) -> str:
    return f"{row['article_id']}#{row['ticker'] or row['kind']}"


def changed_marks(before: list[dict[str, str]], after: list[dict[str, str]]) -> list[str]:
    """Keys of the existing gold that the new rows would change (a ``PROTECTED`` column) or drop."""
    now = {r["key"]: r for r in after}
    return [
        r["key"]
        for r in before
        if r["key"] not in now or any(now[r["key"]].get(c) != r.get(c) for c in PROTECTED)
    ]


def build(
    marks: list[dict[str, str]],
    ids: list[dict[str, str]],
    design: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    """The gold rows, in the order of the draw. Raises ``MarkedFileError`` when the marks and the
    draw are not the same texts."""
    gold_metrics.require_unique([r["key"] for r in ids], IDS_NAME)
    keys = [key_of(r) for r in marks]
    gold_metrics.require_unique(keys, TO_TAG_NAME)
    gold_metrics.require_same_keys(
        keys, [r["key"] for r in ids], f"{TO_TAG_NAME} against {IDS_NAME}"
    )
    by_key = dict(zip(keys, marks, strict=True))
    out = []
    for drawn in ids:
        empty = [f for f in NEEDED_FROM_THE_DRAW if not (drawn.get(f) or "").strip()]
        if empty:
            raise gold_metrics.MarkedFileError(f"{drawn['key']}: {IDS_NAME} has no {empty}")
        marked = by_key[drawn["key"]]
        for field in ("article_id", "ticker", "kind"):
            if marked[field] != drawn[field]:
                raise gold_metrics.MarkedFileError(
                    f"{drawn['key']}: {field} is {marked[field]!r} in {TO_TAG_NAME} and "
                    f"{drawn[field]!r} in {IDS_NAME}"
                )
        if retag.text_hash(marked["text"]) != drawn["text_hash"]:
            raise gold_metrics.MarkedFileError(
                f"{drawn['key']}: the text in {TO_TAG_NAME} is not the text the draw recorded (it "
                "was edited after the draw); the mark would belong to another text"
            )
        pub_date = (marked.get("pub_date") or "").strip()
        try:
            date.fromisoformat(pub_date[:10])
        except ValueError:
            raise gold_metrics.MarkedFileError(
                f"{drawn['key']}: pub_date {pub_date!r} in {TO_TAG_NAME} is not an ISO date "
                "(YYYY-MM-DD...); a spreadsheet may have rewritten it"
            ) from None
        out.append(
            {
                **{f: drawn[f] for f in retag.SENTIMENT_GOLD_COLUMNS if f in drawn},
                "pub_date": pub_date,
                "sentiment": marked["sentiment"],
                "notes": marked["notes"],
            }
        )
    gold_metrics.check_sentiment_design(out, design)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold/sentiment_gold.csv"))
    parser.add_argument(
        "--overwrite-marks",
        action="store_true",
        help="accept that marks, text hashes or rows of the existing gold file change or go",
    )
    args = parser.parse_args()
    try:
        marks = gold_metrics.read_marked(
            args.out / TO_TAG_NAME,
            "sentiment",
            gold_metrics.SENTIMENT_CLASSES,
            required=("article_id", "ticker", "kind", "text", "pub_date"),
        )
        rows = build(marks, gold_metrics.read_rows(args.out / IDS_NAME))
    except gold_metrics.MarkedFileError as exc:
        sys.exit(str(exc))
    if args.gold.exists() and not args.overwrite_marks:
        with args.gold.open(encoding="utf-8-sig", newline="") as fh:
            moved = changed_marks(list(csv.DictReader(fh)), rows)
        if moved:
            sys.exit(
                f"{len(moved)} rows of {args.gold} would change or go missing (first {moved[0]}): "
                "the gold file is the reference; run with --overwrite-marks if this is intended"
            )
    args.gold.parent.mkdir(parents=True, exist_ok=True)
    try:
        gold_metrics.write_pair_atomically(
            {
                args.gold: gold_metrics.render_csv(
                    retag.SENTIMENT_GOLD_COLUMNS, retag.spreadsheet_safe(rows)
                )
            }
        )
    except OSError as exc:
        sys.exit(f"cannot write {args.gold} ({exc}); it is as it was")
    print(f"{args.gold}: {len(rows)} rows")


if __name__ == "__main__":
    main()
