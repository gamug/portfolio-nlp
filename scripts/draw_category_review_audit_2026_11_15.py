"""A second reader for the review of the category sheet (T-200): 50 changed rows, drawn at random, blind.

    uv run scripts/draw_category_review_audit_2026_11_15.py [--n 50] [--seed 1] [--force]

The review of 2026-11-15 changed the tag of 226 of the 999 rows of ``gold/category_single_gold.csv`` (the first tag of the owner's cell, after folding
``competitive_actions`` and ``peer_comparison`` into ``other``, against the final tag; the 227th change of the PR text is row 166, left out). This draws
``--n`` of them with ``random.Random(seed)`` and writes, git-ignored:

* ``data/category_review_audit_to_read.csv``: the company, the title, the text, and two options in random order (``option_1``, ``option_2``), with empty
  ``pick`` (``1``, ``2`` or ``other``) and ``other_tag`` columns. The owner does not see which option is the review's.
* ``data/category_review_audit_key.csv``: which option is the owner's first tag and which the review's.

The agreement is the share of picks of the review's tag (``other`` counts against both). It is recorded in ``docs/category_training.md`` when the sheet is read.
Refuses to overwrite without ``--force``.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLD = {"competitive_actions": "other", "peer_comparison": "other"}


def first_tag(cell: str) -> str:
    tag = cell.split(",", maxsplit=1)[0].strip()
    return FOLD.get(tag, tag)


def changed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if first_tag(r["label_owner"]) != r["label"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out, key_out = (
        ROOT / "data/category_review_audit_to_read.csv",
        ROOT / "data/category_review_audit_key.csv",
    )
    if (out.exists() or key_out.exists()) and not args.force:
        sys.exit(
            f"{out} or {key_out} exists: they may hold the owner's picks; use --force to overwrite"
        )
    with (ROOT / "gold/category_single_gold.csv").open(encoding="utf-8", newline="") as f:
        changed = changed_rows(list(csv.DictReader(f)))
    with (ROOT / "gold/texts/category_single_gold_texts.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        texts = {r["review_order"]: r for r in csv.DictReader(f)}
    rng = random.Random(args.seed)  # noqa: S311 -- a sample, not security
    chosen = sorted(rng.sample(changed, args.n), key=lambda r: int(r["review_order"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        out.open("w", encoding="utf-8", newline="") as fo,
        key_out.open("w", encoding="utf-8", newline="") as fk,
    ):
        wo = csv.writer(fo)
        wk = csv.writer(fk)
        wo.writerow(
            [
                "review_order",
                "article_id",
                "company",
                "title",
                "text",
                "option_1",
                "option_2",
                "pick",
                "other_tag",
            ]
        )
        wk.writerow(["review_order", "owner_option", "review_option", "owner_tag", "review_tag"])
        for r in chosen:
            t = texts[r["review_order"]]
            owner, review = first_tag(r["label_owner"]), r["label"]
            pair = [owner, review]
            rng.shuffle(pair)
            wo.writerow(
                [
                    r["review_order"],
                    r["article_id"],
                    r["company"],
                    t["title"],
                    t["text"],
                    pair[0],
                    pair[1],
                    "",
                    "",
                ]
            )
            wk.writerow(
                [r["review_order"], pair.index(owner) + 1, pair.index(review) + 1, owner, review]
            )
    print(f"{len(changed)} changed rows; {args.n} drawn -> {out} (key: {key_out})")


if __name__ == "__main__":
    main()
