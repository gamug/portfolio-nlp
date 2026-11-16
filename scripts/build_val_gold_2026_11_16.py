"""Move the owner's two new sets of marks to ``gold/`` (review of PR #201: the marks and their texts are tracked, ``docs/gold_datasets.md``).

    uv run scripts/build_val_gold_2026_11_16.py [--data data] [--gold gold]

* ``gold/sentiment_val190.csv`` and ``gold/texts/sentiment_val190_texts.csv`` <- ``data/sentiment_val90/to_tag.csv`` (the owner's tag of each
  piece) joined with ``key.csv`` (the model's prediction, the revisions, the draw: ``batch`` 90 or 100) on ``key``. The Spanish columns are a
  tagging aid and are not copied.
* ``gold/main_company_marks.csv`` and ``gold/texts/main_company_marks_texts.csv`` <- the **marked rows** of ``data/main_company/to_tag.csv`` joined
  with the relevance score and band of ``to_tag.strata.json``.

Each mark is tied to its text by ``text_hash`` (``retag.text_hash``). Nothing already in ``gold/`` is changed. The run stops on an unmarked or unknown
value, a key without text or a duplicate key, and prints the SHA-256 of the files it read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from news_nlp import retag

csv.field_size_limit(sys.maxsize)

VAL_COLUMNS = (
    "key",
    "article_id",
    "item_id",
    "ticker",
    "batch",
    "text_source",
    "text_hash",
    "predicted",
    "predicted_score",
    "relevance_score",
    "pub_date",
    "company_revision",
    "sentiment_revision",
    "summary_revision",
    "sentiment",
    "notes",
)
VAL_TEXT_COLUMNS = ("key", "article_id", "ticker", "company", "text_hash", "title", "text")
MARK_COLUMNS = (
    "key",
    "item_id",
    "ticker",
    "band",
    "relevance_score",
    "text_hash",
    "es_principal",
    "falta_la_principal",
    "notes",
)
MARK_TEXT_COLUMNS = ("key", "item_id", "ticker", "company", "text_hash", "text")
CLASSES = ("positive", "negative", "neutral")
#: The revisions of the run both draws read (the second draw's key file does not carry them).
COMPANY_REV = "companies-f349536cf701"
SENTIMENT_REV = "sentiment-5999fe131d8f"
SUMMARY_REV = "summary-7a9bd2062bc4"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentiment_rows(data: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    folder = data / "sentiment_val90"
    keys = {r["key"]: r for r in read(folder / "key.csv")}
    marks, texts = [], []
    for r in read(folder / "to_tag.csv"):
        tag = r["sentiment"].strip()
        if tag not in CLASSES:
            raise SystemExit(f"{r['key']}: tag {tag!r} is not one of {CLASSES}")
        k = keys[r["key"]]
        h = retag.text_hash(r["text"])
        marks.append(
            {
                "key": r["key"],
                "article_id": k["article_id"],
                "item_id": k["item_id"],
                "ticker": r["ticker"],
                "batch": k.get("batch") or "90",
                "text_source": k["text_source"],
                "text_hash": h,
                "predicted": k["predicted"],
                "predicted_score": k["predicted_score"],
                "relevance_score": k["relevance_score"],
                "pub_date": r["pub_date"],
                "company_revision": k.get("company_revision") or COMPANY_REV,
                "sentiment_revision": k.get("sentiment_revision") or SENTIMENT_REV,
                "summary_revision": k.get("summary_revision") or SUMMARY_REV,
                "sentiment": tag,
                "notes": r["notes"].strip(),
            }
        )
        texts.append(
            {
                "key": r["key"],
                "article_id": k["article_id"],
                "ticker": r["ticker"],
                "company": r["company"],
                "text_hash": h,
                "title": r["title"],
                "text": r["text"],
            }
        )
    return marks, texts


def main_company_rows(data: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    folder = data / "main_company"
    strata = json.loads((folder / "to_tag.strata.json").read_text(encoding="utf-8"))["items"]
    marks, texts = [], []
    for r in read(folder / "to_tag.csv"):
        mark = r["es_principal"].strip().lower()
        if not mark:
            continue
        if mark not in ("yes", "no"):
            raise SystemExit(f"{r['item_id']}#{r['ticker']}: mark {mark!r} is not yes or no")
        item = strata[r["item_id"]]
        key = f"{r['item_id']}#{r['ticker']}"
        h = retag.text_hash(r["texto"])
        marks.append(
            {
                "key": key,
                "item_id": r["item_id"],
                "ticker": r["ticker"],
                "band": str(item["band"]),
                "relevance_score": f"{item['scores'][r['ticker']]:.6f}",
                "text_hash": h,
                "es_principal": mark,
                "falta_la_principal": r["falta_la_principal"].strip().lower(),
                "notes": r["notas"].strip(),
            }
        )
        texts.append(
            {
                "key": key,
                "item_id": r["item_id"],
                "ticker": r["ticker"],
                "company": r["empresa"],
                "text_hash": h,
                "text": r["texto"],
            }
        )
    return marks, texts


def check_unique(rows: list[dict[str, str]], name: str) -> None:
    keys = [r["key"] for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"{name}: duplicate keys")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    args = parser.parse_args()
    for name in ("sentiment_val190.csv", "main_company_marks.csv"):
        if (args.gold / name).exists():
            raise SystemExit(f"{args.gold / name} exists: nothing in gold/ is overwritten")
    s_marks, s_texts = sentiment_rows(args.data)
    m_marks, m_texts = main_company_rows(args.data)
    check_unique(s_marks, "sentiment_val190")
    check_unique(m_marks, "main_company_marks")
    write(args.gold / "sentiment_val190.csv", VAL_COLUMNS, s_marks)
    write(args.gold / "texts" / "sentiment_val190_texts.csv", VAL_TEXT_COLUMNS, s_texts)
    write(args.gold / "main_company_marks.csv", MARK_COLUMNS, m_marks)
    write(args.gold / "texts" / "main_company_marks_texts.csv", MARK_TEXT_COLUMNS, m_texts)
    print(f"sentiment_val190: {len(s_marks)} rows; main_company_marks: {len(m_marks)} rows")
    for path in (
        args.data / "sentiment_val90" / "to_tag.csv",
        args.data / "sentiment_val90" / "key.csv",
        args.data / "main_company" / "to_tag.csv",
        args.data / "main_company" / "to_tag.strata.json",
    ):
        print(f"read {path}: sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
