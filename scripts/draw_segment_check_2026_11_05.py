"""Draw the paragraphs the owner reads to check the automatic labels of the step-4 training set
(Work item 18, T-148b2, D27).

    uv run scripts/draw_segment_check_2026_11_05.py [--out data] [--per-kind 100] [--seed 20261106] [--yes]

``gold/company_segments_train.csv`` labels every paragraph of a body written as one segment per
company with a company: ``lead`` (the paragraph starts with that company) or ``continuation`` (it
starts with none and inherits the previous company). Nobody has checked the labels, and the
attribution model learns from them. This draws ``--per-kind`` paragraphs of each kind from the
``train`` split, at random, and writes a tagging file (``retag.SEGMENT_CHECK_COLUMNS``, the Spanish
pair right after ``text``): the owner reads the paragraph (with the one before it) and marks
``correct`` yes or no for the company the parser chose, shown in ``ticker`` / ``company``. This is a
check of an automatic label, not a gold tagging, so the parser's choice is shown on purpose.

Writes ``<out>/segment_check_ids.csv`` (drawn once) and ``<out>/segment_check_to_tag.csv`` (never
overwritten). A body whose hash no longer matches the gold file stops the run: the paragraph
positions would point at other text. The Spanish aid is the approved paid translation of the same
engine; nothing is bought without ``--yes``. Its cache is ``<out>/segment_check/``.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sector_extra_2026_11_05 as extra
import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import company_texts, retag, triage

IDS_NAME = "segment_check_ids.csv"
TO_TAG_NAME = "segment_check_to_tag.csv"
CACHE_DIR = "segment_check"
GOLD = Path("gold/company_segments_train.csv")
MAX_CHARS = 1500
IDS_FIELDS = ("key", "article_id", "segment_index", "ticker", "kind", "year", "body_hash")


def pick(rows: list[dict[str, str]], per_kind: int, seed: int) -> list[dict[str, str]]:
    """``per_kind`` paragraphs of each kind from the ``train`` split, seeded, shuffled so the order
    does not reveal the kind."""
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    chosen: list[dict[str, str]] = []
    for kind in ("lead", "continuation"):
        members = sorted(
            (r for r in rows if r["split"] == "train" and r["kind"] == kind),
            key=lambda r: (int(r["article_id"]), int(r["segment_index"])),
        )
        rng.shuffle(members)
        chosen += [dict(r) for r in members[:per_kind]]
    rng.shuffle(chosen)
    for r in chosen:
        r["key"] = f"{r['article_id']}#{r['segment_index']}"
    return chosen


def build_rows(
    chosen: list[dict[str, str]], names: dict[str, str], spanish: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    conn = db.connect_pipeline()
    rows: list[dict[str, str]] = []
    for order, r in enumerate(chosen, 1):
        src = conn.execute(
            "SELECT title, body_text, pub_date, source_domain FROM source.articles WHERE id = ?",
            (int(r["article_id"]),),
        ).fetchone()
        body = src["body_text"] or ""
        if triage.body_hash(body) != r["body_hash"]:
            sys.exit(f"article {r['article_id']}: the body no longer matches the gold file's hash")
        parts = company_texts.segments(body)
        index = int(r["segment_index"])
        es = spanish[r["key"]]
        rows.append(
            {
                "review_order": str(order),
                "article_id": r["article_id"],
                "ticker": r["ticker"],
                "company": names.get(r["ticker"], r["ticker"]),
                "pub_date": src["pub_date"] or "",
                "source_domain": src["source_domain"] or "",
                "title": src["title"] or "",
                "text": parts[index][:MAX_CHARS],
                "spanish_title": es["spanish_title"],
                "spanish_text": es["spanish_text"],
                "kind": r["kind"],
                "previous_segment": parts[index - 1][:MAX_CHARS] if index > 0 else "",
                "spanish_previous_segment": (
                    spanish[f"{r['key']}~prev"]["spanish_text"] if index > 0 else ""
                ),
                "correct": "",
                "notes": "",
            }
        )
    return rows


def texts_to_translate(chosen: list[dict[str, str]]) -> list[dict[str, str]]:
    """The paragraph (key ``<id>#<position>``) and the one before it (key ``<id>#<position>~prev``)."""
    conn = db.connect_pipeline()
    keyed: list[dict[str, str]] = []
    for r in chosen:
        src = conn.execute(
            "SELECT title, body_text FROM source.articles WHERE id = ?", (int(r["article_id"]),)
        ).fetchone()
        parts = company_texts.segments(src["body_text"] or "")
        index = int(r["segment_index"])
        title = src["title"] or ""
        keyed.append({"article_id": r["key"], "title": title, "text": parts[index][:MAX_CHARS]})
        if index > 0:
            keyed.append(
                {
                    "article_id": f"{r['key']}~prev",
                    "title": title,
                    "text": parts[index - 1][:MAX_CHARS],
                }
            )
    return keyed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--per-kind", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--yes", action="store_true", help="pay for the Spanish translation")
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    args = parser.parse_args()
    out: Path = args.out
    ids_path, target = out / IDS_NAME, out / TO_TAG_NAME
    if extra.has_rows(target):
        sys.exit(f"{target} exists: it is your tagging file and is not overwritten")
    if ids_path.exists():
        chosen = extra._read(ids_path)
        print(f"{ids_path} exists: the set is not drawn again (--per-kind and --seed are ignored)")
    else:
        with GOLD.open(encoding="utf-8", newline="") as fh:
            chosen = pick(list(csv.DictReader(fh)), args.per_kind, args.seed)
        extra._write(ids_path, [{f: t.get(f, "") for f in IDS_FIELDS} for t in chosen])
        print(f"{len(chosen)} drawn:", dict(Counter(r["kind"] for r in chosen)))
        print(
            "by period:",
            dict(Counter("2022+" if r["year"] >= "2022" else "before 2022" for r in chosen)),
        )
    cache = out / CACHE_DIR
    cache.mkdir(exist_ok=True)
    done = retag_cli.add_spanish(
        cache,
        texts_to_translate(chosen),
        retag_cli._translator(),
        yes=args.yes,
        max_new=args.max_new,
    )
    spanish = {r["article_id"]: r for r in done}
    names = {c.ticker: c.name for c in retag_cli.companies()}
    rows = build_rows(chosen, names, spanish)
    try:
        retag.require_tagging_format(rows, target.name, retag.SEGMENT_CHECK_COLUMNS)
        extra._write(target, retag.spreadsheet_safe(rows))
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    except PermissionError:
        sys.exit(f"cannot write {target}: close it in Excel and run again")
    print(f"{len(rows)} paragraphs -> {target}")


if __name__ == "__main__":
    main()
