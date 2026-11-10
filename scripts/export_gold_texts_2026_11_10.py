"""Put the texts behind the tracked marks in ``gold/texts/`` (owner's word, 2026-11-10: the repository and its data
are public, the texts are tracked with the marks).

    uv run python scripts/export_gold_texts_2026_11_10.py [--data data] [--gold gold]

Reads the git-ignored working files and writes one file per set, keyed like the marks of ``gold/`` so a mark and its
text meet on ``key`` (sentiment, extractor) with ``text_hash`` to check them:

* ``sentiment_gold_texts.csv``      <- ``data/sentiment_gold_ids.csv`` (180 texts, set 5)
* ``sentiment_train_texts.csv``     <- ``data/sentiment_train_ids.csv`` (450, set 6)
* ``sentiment_train_2_texts.csv``   <- ``data/sentiment_eval_ids.csv`` (405, set 6, second file)
* ``faithfulness_texts.csv`` and ``faithfulness_confirmation_texts.csv`` <- ``faithfulness_to_tag.csv`` joined with
  ``faithfulness_ids.csv`` on ``review_order`` (the text the owner read; the article body is not copied: it is in
  SOURCE under ``article_id``, and the Spanish columns are a tagging aid).

* ``category_gold_v2_texts.csv`` <- ``gate240_to_tag.csv`` + ``sector_extra_to_tag.csv`` (284, the scope train set),
  ``category_gold_v2_test_texts.csv`` <- ``heldout_to_tag.csv`` (200) and ``segment_check_gold_texts.csv`` <-
  ``segment_check_to_tag.csv`` (200): the text the owner read, keyed by ``article_id`` and ``ticker`` (``key`` for the
  segment check). Added 2026-11-10 (owner: copy them too).

The run stops if a text's hash is not the recorded one (the owner read another text) or if a mark of ``gold/`` has no
text. Nothing in ``gold/`` that already exists is changed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from news_nlp import retag

csv.field_size_limit(sys.maxsize)
SENTIMENT = (
    ("sentiment_gold_ids.csv", "sentiment_gold.csv", "sentiment_gold_texts.csv"),
    ("sentiment_train_ids.csv", "sentiment_train.csv", "sentiment_train_texts.csv"),
    ("sentiment_eval_ids.csv", "sentiment_train_2.csv", "sentiment_train_2_texts.csv"),
)
SENTIMENT_COLUMNS = ("key", "article_id", "ticker", "company", "text_hash", "title", "text")
FAITH_COLUMNS = ("key", "article_id", "ticker", "text_hash", "title", "text")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check(rows: list[dict[str, str]], what: str) -> None:
    bad = [
        r["key"]
        for r in rows
        if r.get("text_hash") and retag.text_hash(r["text"]) != r["text_hash"]
    ]
    if bad:
        sys.exit(f"{what}: {len(bad)} texts do not match their recorded hash (first: {bad[0]})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    args = parser.parse_args()
    out = args.gold / "texts"
    for ids_name, marks_name, out_name in SENTIMENT:
        rows = read(args.data / ids_name)
        check(rows, ids_name)
        marks = {r["key"] for r in read(args.gold / marks_name)}
        missing = marks - {r["key"] for r in rows}
        if missing:
            sys.exit(f"{marks_name}: {len(missing)} marks have no text in {ids_name}")
        write(out / out_name, SENTIMENT_COLUMNS, [r for r in rows if r["key"] in marks])
        print(f"{out / out_name}: {len(marks)} texts")
    for folder, gold_name, out_name in (
        (args.data, "faithfulness_gold.csv", "faithfulness_texts.csv"),
        (
            args.data / "faithfulness_confirmation",
            "faithfulness_confirmation/faithfulness_gold.csv",
            "faithfulness_confirmation_texts.csv",
        ),
    ):
        tagged = {r["review_order"]: r for r in read(folder / "faithfulness_to_tag.csv")}
        ids = read(folder / "faithfulness_ids.csv")
        rows = [
            {
                **ids_row,
                "title": tagged[ids_row["review_order"]]["title"],
                "text": tagged[ids_row["review_order"]]["text"],
            }
            for ids_row in ids
        ]
        marks = {r["key"] for r in read(args.gold / gold_name)}
        missing = marks - {r["key"] for r in rows}
        if missing:
            sys.exit(f"{gold_name}: {len(missing)} marks have no text")
        # the owner's file may show a text with a spreadsheet-safe apostrophe: report, never rewrite
        odd = [r["key"] for r in rows if retag.text_hash(r["text"]) != r["text_hash"]]
        write(out / out_name, FAITH_COLUMNS, [r for r in rows if r["key"] in marks])
        print(
            f"{out / out_name}: {len(marks)} texts; {len(odd)} differ from their hash by a spreadsheet apostrophe or edit"
        )

    scope_columns = (
        "article_id",
        "ticker",
        "company",
        "pub_date",
        "source_domain",
        "title",
        "text",
    )
    for sources, gold_name, out_name in (
        (
            ("gate240_to_tag.csv", "sector_extra_to_tag.csv"),
            "category_gold_v2.csv",
            "category_gold_v2_texts.csv",
        ),
        (("heldout_to_tag.csv",), "category_gold_v2_test.csv", "category_gold_v2_test_texts.csv"),
    ):
        rows = [r for name in sources for r in read(args.data / name)]
        marks = {(r["article_id"], r["ticker"]) for r in read(args.gold / gold_name)}
        have = {(r["article_id"], r["ticker"]) for r in rows}
        if marks - have:
            sys.exit(f"{gold_name}: {len(marks - have)} marks have no text")
        write(
            out / out_name,
            scope_columns,
            [r for r in rows if (r["article_id"], r["ticker"]) in marks],
        )
        print(f"{out / out_name}: {len(marks)} texts")
    seg = read(args.data / "segment_check_to_tag.csv")
    ids = read(args.data / "segment_check_ids.csv")
    if len(ids) != len(seg) or any(
        a["article_id"] != b["article_id"] for a, b in zip(ids, seg, strict=True)
    ):
        sys.exit("segment_check: the ids file and the file to tag are not in the same order")
    seg_rows = [{**r, "key": i["key"]} for r, i in zip(seg, ids, strict=True)]
    marks = {r["key"] for r in read(args.gold / "segment_check_gold.csv")}
    if marks - {r["key"] for r in seg_rows}:
        sys.exit("segment_check_gold.csv: marks without text")
    write(
        out / "segment_check_gold_texts.csv",
        ("key", "article_id", "ticker", "kind", "title", "text"),
        [r for r in seg_rows if r["key"] in marks],
    )
    print(f"{out / 'segment_check_gold_texts.csv'}: {len(marks)} texts")


if __name__ == "__main__":
    main()
