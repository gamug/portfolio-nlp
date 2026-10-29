"""The owner's re-tag of the 720 judged articles under the new taxonomy (Work item 18, T-146b).

    uv run scripts/retag_gold_2026_10_29.py export [--out data]   # writes the file to tag
    uv run scripts/retag_gold_2026_10_29.py check  [--out data]   # validates the returned file

``export`` reads ``data/category_gold_review.csv`` and its key (tags v1) and writes
``category_gold_v2_review.csv`` (the article, the v1 label and note and the scope they imply as
guidance, EMPTY ``scope`` / ``companies`` / ``company_labels`` / ``sub_industry`` / ``notes``) and
``category_gold_v2_key.csv`` (the v1 key, where the LLM's label stays: the owner is not shown it).
It **refuses to overwrite a file that already has scopes filled in**. ``check`` lists the
untagged rows and every row that breaks the rules (``news_nlp.retag.check_row``).
Column format: ``companies`` = ``AAPL; MSFT``; ``company_labels`` = ``AAPL:product_launch; MSFT:other``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import retag
from news_nlp.gold_review import TaggedFileError, write_review_files


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def cmd_export(out: Path) -> None:
    review, key = retag.retag_rows(
        _read(out / "category_gold_review.csv"), _read(out / "category_gold_review_key.csv")
    )
    try:
        write_review_files(
            out,
            review,
            key,
            review_name=retag.REVIEW_NAME,
            key_name=retag.KEY_NAME,
            tag_column="scope",
        )
    except TaggedFileError as exc:
        sys.exit(f"{exc}. Copy it elsewhere or pass another --out directory.")
    print(f"{len(review)} articles -> {out / retag.REVIEW_NAME}")


def cmd_check(out: Path) -> None:
    good, bad, untagged = retag.load_gold_v2(out / retag.REVIEW_NAME)
    print(f"tagged and valid {len(good)}, with errors {len(bad)}, untagged {untagged}")
    print("scopes:", dict(Counter(g.scope for g in good)))
    for article_id, errors in sorted(bad.items()):
        print(f"  article {article_id}: {'; '.join(errors)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("export", "check"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    {"export": cmd_export, "check": cmd_check}[args.command](args.out)


if __name__ == "__main__":
    main()
