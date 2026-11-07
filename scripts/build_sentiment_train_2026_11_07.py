"""Put the owner's marks on the training summaries in the tracked file (Work item 18, T-148c1, D35).

    uv run scripts/build_sentiment_train_2026_11_07.py [--out data] [--gold gold/sentiment_train.csv]
        [--overwrite-marks]

Reads ``data/sentiment_train_to_tag.csv`` (the marks the owner wrote) and
``data/sentiment_train_ids.csv`` (the draw: the texts, the model's prediction, the block, the batch
and the slice) and writes ``gold/sentiment_train.csv``: the mark and the ids only, no news text.
The two files are paired by key (``<article_id>#summary``). The run stops unless

* every row is marked positive / negative / neutral and the marked file holds exactly the texts of
  the draw, once each;
* each pair agrees on the article and the kind, and the text the owner read is the text the draw
  recorded (its hash; a text that a spreadsheet made safe with a leading apostrophe is compared
  without it, and the escaped text is never what is kept);
* the sample is the one drawn (``check_design``): the same number of each predicted class, blocks of
  ``BLOCK`` rows with the same number of each, and every batch and slice what D35 says (the last
  ``VALIDATION_BLOCKS`` blocks of a batch are the validation slice).

``text_hash``, the block, the batch and the slice are copied from the ids, never recomputed;
``pub_date`` is the one in the file the owner tagged. A gold file that already exists is the
reference: if a mark or a column the training depends on (``PROTECTED``) would change, or a row would
go missing, the run stops (``--overwrite-marks`` to accept). The file is replaced as a whole.

The training reads the marks from here and the texts from ``data/sentiment_train_ids.csv`` (by key,
checked by ``text_hash``): that file is the only copy of the texts the owner read, and a summary is
generated, so **back it up** (the cut's chunks hold the same summaries by body hash).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sentiment_train_2026_11_07 as draw

from news_nlp import gold_metrics, retag

TO_TAG_NAME = draw.TO_TAG_NAME
IDS_NAME = draw.IDS_NAME
CLASSES = gold_metrics.SENTIMENT_CLASSES
#: The columns of ``gold/sentiment_train.csv``: no news text.
TRAIN_COLUMNS = (
    "key",
    "article_id",
    "body_hash",
    "rule_scope",
    "revision",
    "predicted",
    "pool_class_size",
    "block",
    "batch",
    "slice",
    "text_hash",
    "pub_date",
    "sentiment",
    "notes",
)
NEEDED_FROM_THE_DRAW = tuple(
    c for c in TRAIN_COLUMNS if c not in ("pub_date", "sentiment", "notes")
)
#: What a rebuild may not change in a tracked row without ``--overwrite-marks``.
PROTECTED = (
    "sentiment",
    "text_hash",
    "revision",
    "predicted",
    "block",
    "batch",
    "slice",
    "pub_date",
)


def key_of(row: dict[str, str]) -> str:
    return f"{row['article_id']}#{row['kind']}"


def changed_marks(before: list[dict[str, str]], after: list[dict[str, str]]) -> list[str]:
    """Keys of the existing file that the new rows would change (a ``PROTECTED`` column) or drop."""
    now = {r["key"]: r for r in after}
    return [
        r["key"]
        for r in before
        if r["key"] not in now or any(now[r["key"]].get(c) != r.get(c) for c in PROTECTED)
    ]


def check_design(rows: list[dict[str, str]]) -> None:
    """The sample is the one ``draw_sentiment_train`` makes: equal predicted classes, blocks that hold
    the same number of each, and the batch and slice of every block as D35 says."""
    per_block = draw.BLOCK // len(CLASSES)
    counts = Counter(r["predicted"] for r in rows)
    if set(counts) != set(CLASSES) or len(set(counts.values())) != 1:
        raise gold_metrics.MarkedFileError(f"the predicted classes are not equal: {dict(counts)}")
    blocks: dict[int, Counter[str]] = {}
    for r in rows:
        try:
            number = int(r["block"])
        except ValueError:
            raise gold_metrics.MarkedFileError(
                f"{r['key']}: block {r['block']!r} is not a number"
            ) from None
        blocks.setdefault(number, Counter())[r["predicted"]] += 1
        in_batch = (number - 1) % draw.BLOCKS_PER_BATCH + 1
        want_slice = (
            "validation" if in_batch > draw.BLOCKS_PER_BATCH - draw.VALIDATION_BLOCKS else "train"
        )
        want_batch = str((number - 1) // draw.BLOCKS_PER_BATCH + 1)
        if (r["batch"], r["slice"]) != (want_batch, want_slice):
            raise gold_metrics.MarkedFileError(
                f"{r['key']}: block {number} is batch {want_batch}, {want_slice}; the file says "
                f"batch {r['batch']}, {r['slice']}"
            )
    if sorted(blocks) != list(range(1, len(blocks) + 1)):
        raise gold_metrics.MarkedFileError("the blocks are not 1..N in a row")
    odd = [n for n, c in blocks.items() if set(c.values()) != {per_block} or set(c) != set(CLASSES)]
    if odd:
        raise gold_metrics.MarkedFileError(f"blocks without {per_block} of each class: {odd[:5]}")
    if len({r["body_hash"] for r in rows}) != len(rows):
        raise gold_metrics.MarkedFileError("two rows share a body")


def read_the_same_text(marked: str, drawn_hash: str) -> bool:
    """Whether the text in the owner's file is the drawn one: its hash, or its hash without the one
    apostrophe a spreadsheet-safe file puts before a text that starts like a formula."""
    return retag.text_hash(marked) == drawn_hash or (
        marked.startswith("'") and retag.text_hash(marked[1:]) == drawn_hash
    )


def build(marks: list[dict[str, str]], ids: list[dict[str, str]]) -> list[dict[str, str]]:
    """The rows of ``gold/sentiment_train.csv``, in the order of the draw. Raises
    ``MarkedFileError`` when the marks and the draw are not the same texts."""
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
        for field in ("article_id", "kind"):
            if marked[field] != drawn[field]:
                raise gold_metrics.MarkedFileError(
                    f"{drawn['key']}: {field} is {marked[field]!r} in {TO_TAG_NAME} and "
                    f"{drawn[field]!r} in {IDS_NAME}"
                )
        if not read_the_same_text(marked["text"], drawn["text_hash"]):
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
                **{f: drawn[f] for f in TRAIN_COLUMNS if f in drawn},
                "pub_date": pub_date,
                "sentiment": marked["sentiment"],
                "notes": marked["notes"],
            }
        )
    check_design(out)
    return out


def summary(rows: list[dict[str, str]]) -> list[str]:
    """What the owner marked, by slice, and the model that balanced the draw against the owner (a
    row per predicted class, a column per mark): context for the training, not a figure."""
    lines = []
    for kind in ("train", "validation"):
        mine = [r for r in rows if r["slice"] == kind]
        lines.append(
            f"{kind}: {len(mine)} texts, marks {dict(Counter(r['sentiment'] for r in mine))}"
        )
    both = Counter((r["predicted"], r["sentiment"]) for r in rows)
    lines.append(
        "the model that balanced the draw (rows) against the owner (columns): "
        + " | ".join(CLASSES)
    )
    for predicted in CLASSES:
        lines.append(
            f"  {predicted:>8}: " + " | ".join(f"{both[(predicted, m)]:>3}" for m in CLASSES)
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold/sentiment_train.csv"))
    parser.add_argument(
        "--overwrite-marks",
        action="store_true",
        help="accept that marks, text hashes or rows of the existing file change or go",
    )
    args = parser.parse_args()
    try:
        marks = gold_metrics.read_marked(
            args.out / TO_TAG_NAME,
            "sentiment",
            CLASSES,
            required=("article_id", "kind", "text", "pub_date"),
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
                "the file is the reference; run with --overwrite-marks if this is intended"
            )
    args.gold.parent.mkdir(parents=True, exist_ok=True)
    try:
        gold_metrics.write_pair_atomically(
            {args.gold: gold_metrics.render_csv(TRAIN_COLUMNS, retag.spreadsheet_safe(rows))}
        )
    except OSError as exc:
        sys.exit(f"cannot write {args.gold} ({exc}); it is as it was")
    print(f"{args.gold}: {len(rows)} rows")
    print("\n".join(summary(rows)))


if __name__ == "__main__":
    main()
