"""A second reader for the review of the category sheet (T-200): changed rows and unchanged rows, drawn at random, blind.

    uv run scripts/draw_category_review_audit_2026_11_15.py [--n 50] [--unchanged 25] [--seed 1] [--force]

The review of 2026-11-15 changed the tag of 226 of the 999 rows of ``gold/category_single_gold.csv`` (the first tag of the owner's cell, after folding
``competitive_actions`` and ``peer_comparison`` into ``other``, against the final tag; the 227th change of the PR text is row 166, left out). The other 773 keep
the owner's tag. This draws ``--n`` changed rows (``random.Random(seed)``, the same rows for the same seed whatever ``--unchanged`` is) and ``--unchanged``
unchanged rows (``random.Random(seed + 1)``) and writes, git-ignored:

* ``data/category_review_audit_to_read.csv``: the company, the title, the text and two options in random order (``option_1``, ``option_2``), with empty
  ``pick`` (``1``, ``2`` or ``other``) and ``other_tag`` columns, all rows mixed in review order. A changed row's options are the owner's first tag and the
  review's tag. An unchanged row's options are its tag and the label the out-of-fold model scored highest among the others
  (``docs/category_results_2026_11_15/single_predictions.csv``), a hard alternative. **The owner sees their own earlier tag as one of the options of a changed
  row and may lean to it; that is a bias of this reading, stated in ``docs/category_training.md`` section 8.**
* ``data/category_review_audit_key.csv``: the kind of row (``changed`` / ``unchanged``), the tag of each option, ``file_option`` (which option is the tag of
  ``gold/category_single_gold.csv``: the review's for a changed row) and, for a changed row, the owner's first tag.

Read the picks as: on changed rows, the share that pick the file's tag (``file_option``); on unchanged rows, the share that pick the row's tag. ``other`` counts
against both. The rule for what the numbers mean is in ``TASKS.md`` T-200. Refuses to overwrite without ``--force`` (the files may hold the owner's picks).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLD = {"competitive_actions": "other", "peer_comparison": "other"}


def first_tag(cell: str) -> str:
    tag = cell.split(",", maxsplit=1)[0].strip()
    return FOLD.get(tag, tag)


def changed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if first_tag(r["label_owner"]) != r["label"]]


def unchanged_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if first_tag(r["label_owner"]) == r["label"]]


def build(
    rows: list[dict[str, str]],
    texts: dict[str, dict[str, str]],
    alternative: Callable[[dict[str, str]], str],
    n_changed: int,
    n_unchanged: int,
    seed: int,
) -> tuple[list[list[str]], list[list[str]]]:
    """The sheet and the key. Pure: the same rows and seed give the same files; the changed rows do not depend on ``n_unchanged``."""
    rng = random.Random(seed)  # noqa: S311 -- a sample, not security
    chosen = sorted(rng.sample(changed_rows(rows), n_changed), key=lambda r: int(r["review_order"]))
    items: list[tuple[dict[str, str], str, str, str]] = []
    for r in chosen:
        pair = [first_tag(r["label_owner"]), r["label"]]
        rng.shuffle(pair)
        items.append((r, "changed", pair[0], pair[1]))
    rng2 = random.Random(seed + 1)  # noqa: S311 -- a sample, not security
    for r in sorted(
        rng2.sample(unchanged_rows(rows), n_unchanged), key=lambda r: int(r["review_order"])
    ):
        pair = [r["label"], alternative(r)]
        rng2.shuffle(pair)
        items.append((r, "unchanged", pair[0], pair[1]))
    items.sort(key=lambda it: int(it[0]["review_order"]))
    sheet, key = [], []
    for r, kind, o1, o2 in items:
        t = texts[r["review_order"]]
        sheet.append(
            [
                r["review_order"],
                r["article_id"],
                r["company"],
                t["title"],
                t["text"],
                o1,
                o2,
                "",
                "",
            ]
        )
        owner = first_tag(r["label_owner"]) if kind == "changed" else ""
        key.append([r["review_order"], kind, o1, o2, str((o1, o2).index(r["label"]) + 1), owner])
    return sheet, key


def alternative_from_scores(path: Path) -> Callable[[dict[str, str]], str]:
    """The label the out-of-fold model scored highest among the others, for a row of the 999 (the committed predictions, no text)."""
    with path.open(encoding="utf-8", newline="") as f:
        pred = {
            (p["article_id"], p["company"]): p for p in csv.DictReader(f) if p["source"] == "single"
        }

    def alt(row: dict[str, str]) -> str:
        p = pred[(row["article_id"], row["company"])]
        scores = {
            k[len("score_") :]: float(v)
            for k, v in p.items()
            if k.startswith("score_") and k[len("score_") :] != row["label"]
        }
        return max(scores, key=scores.__getitem__)

    return alt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--unchanged", type=int, default=25)
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
        rows = list(csv.DictReader(f))
    with (ROOT / "gold/texts/category_single_gold_texts.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        texts = {r["review_order"]: r for r in csv.DictReader(f)}
    alt = alternative_from_scores(ROOT / "docs/category_results_2026_11_15/single_predictions.csv")
    sheet, key = build(rows, texts, alt, args.n, args.unchanged, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        out.open("w", encoding="utf-8", newline="") as fo,
        key_out.open("w", encoding="utf-8", newline="") as fk,
    ):
        wo, wk = csv.writer(fo), csv.writer(fk)
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
        wk.writerow(
            ["review_order", "kind", "option_1", "option_2", "file_option", "owner_first_tag"]
        )
        wo.writerows(sheet)
        wk.writerows(key)
    print(
        f"{len(sheet)} rows ({args.n} changed, {args.unchanged} unchanged) -> {out} (key: {key_out})"
    )


if __name__ == "__main__":
    main()
