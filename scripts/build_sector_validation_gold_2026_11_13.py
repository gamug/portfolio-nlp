"""Put the owner's two yes/no reviews of the sector step in one tracked validation file (owner, 2026-11-13).

    uv run scripts/build_sector_validation_gold_2026_11_13.py [--data data] [--gold gold]

Reads ``data/sector_yesno_to_review.csv`` + ``sector_yesno_key.csv`` (the first review, 2026-11-12: 100 bodies picked by the small model
alone, 10 controls) and ``data/sector_fresh_to_review_es.csv`` + ``sector_fresh_key.csv`` (the clean test, 2026-11-13: 40 assigned, 12
sent out_of_scope, 5 controls, drawn with the step as built) and writes:

* ``gold/sector_validation.csv``: one row per reviewed body, ``review`` (``first`` / ``fresh``), what the model showed, the owner's
  ``correct`` / ``right_sector`` / ``notes``, and what the key said (``control``, ``margin``, ``sure`` or ``status`` / ``vote_sector``);
* ``gold/texts/sector_validation_texts.csv``: the texts the owner read (no Spanish aid), as the other tracked sets do.

The file is a measure, not training data: the sector step was tuned without it (``docs/sector_step.md``). Refuses a row with no answer.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)
COLUMNS = (
    "review", "review_order", "article_id", "shown_sector", "control", "correct", "right_sector", "notes",
    "status", "candidate", "margin", "vote_sector", "sure", "step_revision",
)  # fmt: skip
TEXT_COLUMNS = ("article_id", "ticker", "company", "pub_date", "source_domain", "title", "text")
FRESH_REVISION = "sector-cf81d9b32565"  # the revision the clean test was drawn with


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def answer(row: dict[str, str], where: str) -> tuple[str, str]:
    correct = row["correct"].strip().lower()
    if correct not in ("yes", "no"):
        sys.exit(
            f"{where}: row {row['review_order']} (article {row['article_id']}) has no yes/no answer"
        )
    return correct, re.sub(r"\D", "", row["right_sector"])


def first_rows(data: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    key = {k["review_order"]: k for k in read(data / "sector_yesno_key.csv")}
    out, texts = [], []
    for r in read(data / "sector_yesno_to_review.csv"):
        k = key[r["review_order"]]
        correct, right = answer(r, "first review")
        out.append(
            {
                "review": "first", "review_order": r["review_order"], "article_id": r["article_id"],
                "shown_sector": k["shown"], "control": k["control"], "correct": correct, "right_sector": right,
                "notes": r["notes"].strip(), "status": "", "candidate": k["model_best"], "margin": k["margin"],
                "vote_sector": "", "sure": k["sure"], "step_revision": "",
            }
        )  # fmt: skip
        texts.append({c: r[c] for c in TEXT_COLUMNS})
    return out, texts


def fresh_rows(data: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    key = {k["review_order"]: k for k in read(data / "sector_fresh_key.csv")}
    out, texts = [], []
    for r in read(data / "sector_fresh_to_review_es.csv"):
        k = key[r["review_order"]]
        correct, right = answer(r, "fresh review")
        out.append(
            {
                "review": "fresh", "review_order": r["review_order"], "article_id": r["article_id"],
                "shown_sector": k["shown"], "control": k["control"], "correct": correct, "right_sector": right,
                "notes": r["notes"].strip(), "status": k["status"], "candidate": k["candidate"], "margin": k["margin"],
                "vote_sector": k["vote_sector"], "sure": "", "step_revision": FRESH_REVISION,
            }
        )  # fmt: skip
        texts.append({c: r[c] for c in TEXT_COLUMNS})
    return out, texts


def write(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    args = parser.parse_args()
    first, first_texts = first_rows(args.data)
    fresh, fresh_texts = fresh_rows(args.data)
    ids = [r["article_id"] for r in first + fresh]
    if len(ids) != len(set(ids)):
        sys.exit(
            "an article is in both reviews: the clean test must not share a body with the first"
        )
    write(args.gold / "sector_validation.csv", COLUMNS, first + fresh)
    write(
        args.gold / "texts" / "sector_validation_texts.csv", TEXT_COLUMNS, first_texts + fresh_texts
    )
    print(f"{len(first)} first + {len(fresh)} fresh rows -> {args.gold / 'sector_validation.csv'}")


if __name__ == "__main__":
    main()
