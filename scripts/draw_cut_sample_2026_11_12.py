"""Draw the sample of the cut the owner reads, to answer yes / no questions (T-183; owner 2026-11-12).

    uv run scripts/draw_cut_sample_2026_11_12.py --results-db <nlp.db copy> --source-db <urls.db copy> \
        --cut-revision units-5f831c298bcb [--bodies 30] [--seed 20261112] [--out data] [--yes] [--max-new 800]

Read-only on both databases (pass **copies**: the owner's mount fails a read-only open). It takes ``--bodies`` bodies at random (seeded)
from the ones the cut wrote ``sub_news`` for at the **named** revision (never "the latest"), and writes one row per sub-news, the pieces
of a body together and in the order of the body, with the whole body next to each piece, the Spanish of title, piece and body in the
columns of the other tagging files, and the questions empty:

- ``faithful``: the piece is true to the article (read the rule in ``READING_RULE``);
- ``one_story``: the piece tells the news of one company or one story, with no other company's news inside it;
- ``complete``: the piece reads as a whole, not a clause cut in half or a heading alone.

``<out>/cut_sample_to_tag.csv`` is the owner's file; ``<out>/cut_sample_ids.csv`` ties each row to the exact text by SHA-256 and to the
cut revision (a mark belongs to a text, never to ``review_order``). Neither is overwritten. The Spanish is the approved paid translation:
nothing is bought without ``--yes`` (it prints the estimate and stops), and one already in the cache is not bought again.
Do not draw this set again once marking has begun: a new draw is another set.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sector_extra_2026_11_05 as extra
import retag_gold_2026_10_29 as retag_cli

from news_nlp import retag, triage

TO_TAG_NAME = "cut_sample_to_tag.csv"
IDS_NAME = "cut_sample_ids.csv"
CACHE_DIR = "cut_sample_translations"
TEXT_CAP = extra.TEXT_CAP
STAGE_TEXT = extra.STAGE_TEXT
QUESTIONS = ("faithful", "one_story", "complete")
COLUMNS = (
    "review_order",
    "article_id",
    "segment_index",
    "ticker",
    "company",
    "pub_date",
    "source_domain",
    "title",
    "text",
    "spanish_title",
    "spanish_text",
    "body",
    "spanish_body",
    *QUESTIONS,
    "notes",
)
IDS_COLUMNS = ("review_order", "article_id", "segment_index", "text_hash", "extractor_revision")
READING_RULE = (
    "Read the piece (`text`) with the whole article (`body`). `faithful`: yes if the piece says what the article says, about the company "
    "or the story it is about, with nothing added or changed (it may also name other companies); no if it says something the article "
    "does not, is mostly about something else, or is too broken to tell. `one_story`: yes if it tells one company's or one story's news; "
    "no if news of another company is inside it (a list written in one paragraph). `complete`: yes if it reads as a whole; no if it is a "
    "clause cut in half or a heading alone. `ticker` and `company` are the crawl's tag of the article, for context only: a piece keeps no "
    "ticker."
)


@dataclass(frozen=True)
class Piece:
    sub_news_id: int
    segment_index: int
    text: str


@dataclass(frozen=True)
class DrawnBody:
    article_id: int
    title: str
    body: str
    company: str
    ticker: str
    pub_date: str
    source_domain: str
    pieces: tuple[Piece, ...]


def text_hash(text: str) -> str:
    """SHA-256 of the text the owner reads, outer whitespace stripped (like ``retag.text_hash``)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def for_reading(body: str) -> str:
    """The body as the owner reads it: the first ``TEXT_CAP`` characters, and a mark when it was cut."""
    return body[:TEXT_CAP] + STAGE_TEXT if len(body) > TEXT_CAP else body


def draw(
    results: sqlite3.Connection,
    source: sqlite3.Connection,
    revision: str,
    *,
    bodies: int,
    seed: int,
) -> list[DrawnBody]:
    """``bodies`` bodies of the cut at ``revision``, seeded; a body whose text in SOURCE is no longer the one that was cut is left out."""
    by_body: dict[str, tuple[int, list[Piece]]] = {}
    for sub_id, body_hash, article_id, segment, text in results.execute(
        "SELECT id, body_hash, article_id, segment_index, text FROM sub_news WHERE extractor_revision = ? "
        "ORDER BY body_hash, segment_index",
        (revision,),
    ):
        by_body.setdefault(body_hash, (int(article_id), []))[1].append(
            Piece(int(sub_id), int(segment), str(text))
        )
    order = sorted(by_body)
    random.Random(seed).shuffle(order)  # noqa: S311 -- a reproducible draw, not security
    drawn: list[DrawnBody] = []
    changed = 0
    for body_hash in order:
        if len(drawn) >= bodies:
            break
        article_id, pieces = by_body[body_hash]
        row = source.execute(
            "SELECT title, body_text, company, ticker, pub_date, source_domain FROM articles WHERE id = ?",
            (article_id,),
        ).fetchone()
        if row is None or not row[1] or triage.body_hash(row[1]) != body_hash:
            changed += 1
            continue
        drawn.append(
            DrawnBody(
                article_id,
                row[0] or "",
                row[1],
                row[2] or "",
                row[3] or "",
                row[4] or "",
                row[5] or "",
                tuple(pieces),
            )
        )
    if changed:
        print(
            f"{changed} bodies left out: their text in SOURCE is not the one that was cut",
            flush=True,
        )
    return drawn


def build_rows(
    bodies: list[DrawnBody], revision: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """The owner's rows (Spanish and questions empty) and the ids file, in the same order."""
    rows: list[dict[str, str]] = []
    ids: list[dict[str, str]] = []
    for body in bodies:
        for piece in body.pieces:
            order = str(len(rows) + 1)
            rows.append(
                {
                    "review_order": order,
                    "article_id": str(body.article_id),
                    "segment_index": str(piece.segment_index),
                    "ticker": body.ticker,
                    "company": body.company,
                    "pub_date": body.pub_date,
                    "source_domain": body.source_domain,
                    "title": body.title,
                    "text": piece.text,
                    "spanish_title": "",
                    "spanish_text": "",
                    "body": for_reading(body.body),
                    "spanish_body": "",
                    **{q: "" for q in QUESTIONS},
                    "notes": "",
                }
            )
            ids.append(
                {
                    "review_order": order,
                    "article_id": str(body.article_id),
                    "segment_index": str(piece.segment_index),
                    "text_hash": text_hash(piece.text),
                    "extractor_revision": revision,
                }
            )
    return rows, ids


def to_translate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """What is sent to the translator: each piece (key ``<id>#<n>``, with the title) and each body once (key ``<id>#body``)."""
    keyed = [
        {
            "article_id": f"{r['article_id']}#{r['review_order']}",
            "title": r["title"],
            "text": r["text"],
        }
        for r in rows
    ]
    seen: set[str] = set()
    for r in rows:
        if r["article_id"] not in seen:
            seen.add(r["article_id"])
            keyed.append(
                {"article_id": f"{r['article_id']}#body", "title": r["title"], "text": r["body"]}
            )
    return keyed


def with_spanish(
    rows: list[dict[str, str]], translated: list[dict[str, str]]
) -> list[dict[str, str]]:
    by_key = {t["article_id"]: t for t in translated}
    out = []
    for r in rows:
        piece = by_key[f"{r['article_id']}#{r['review_order']}"]
        body = by_key[f"{r['article_id']}#body"]
        out.append(
            {
                **r,
                "spanish_title": piece["spanish_title"],
                "spanish_text": piece["spanish_text"],
                "spanish_body": body["spanish_text"],
            }
        )
    return out


def refuse_to_overwrite(out: Path) -> None:
    for name in (TO_TAG_NAME, IDS_NAME):
        if (out / name).exists():
            sys.exit(f"{out / name} exists: it is the owner's file and is not overwritten")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--cut-revision", required=True)
    parser.add_argument("--bodies", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--yes", action="store_true", help="pay for the Spanish translation")
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    args = parser.parse_args()
    from news_nlp import env  # noqa: PLC0415

    env.load_env_file()
    refuse_to_overwrite(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    results = sqlite3.connect(f"file:{args.results_db}?mode=ro", uri=True)
    source = sqlite3.connect(f"file:{args.source_db}?mode=ro", uri=True)
    bodies = draw(results, source, args.cut_revision, bodies=args.bodies, seed=args.seed)
    if not bodies:
        sys.exit(
            f"the cut {args.cut_revision} has no sub_news (or none of its bodies match SOURCE)"
        )
    rows, ids = build_rows(bodies, args.cut_revision)
    print(f"{len(bodies)} bodies, {len(rows)} sub-news (cut {args.cut_revision})", flush=True)
    done = retag_cli.add_spanish(
        args.out / CACHE_DIR,
        to_translate(rows),
        retag_cli._translator(),
        yes=args.yes,
        max_new=args.max_new,
    )
    rows = with_spanish(rows, done)
    extra._write(args.out / IDS_NAME, ids)
    extra._write(
        args.out / TO_TAG_NAME, retag.spreadsheet_safe(rows, tag_columns=(*QUESTIONS, "notes"))
    )
    print(f"{len(rows)} rows -> {args.out / TO_TAG_NAME}\n{READING_RULE}")


if __name__ == "__main__":
    main()
