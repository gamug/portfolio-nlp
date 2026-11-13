"""Move the marks of the cut sample to ``gold/`` (T-183; owner 2026-11-12).

    uv run scripts/build_cut_sample_gold_2026_11_12.py [--marked data/cut_sample_to_tag.csv] [--ids data/cut_sample_ids.csv] \
        [--marks gold/cut_sample_marks.csv] [--texts gold/texts/cut_sample_texts.csv]

Reads the file the owner marked and its ids file (no database) and writes the two files that stay in the repository: the marks (``review_order``,
``article_id``, ``segment_index``, ``text_hash``, ``extractor_revision``, the three answers, ``kind``, ``notes``) and the pieces' texts
(``review_order``, ``text``), apart as the other gold sets keep theirs. It stops when a mark is not yes / no, when a row's text is not the one the
ids file ties to its ``review_order``, or when the ids name more than one cut revision. Neither output is overwritten.

``kind`` is the one place where the owner's free-text note is read, by the words below, and it is **stored**: the metrics read the column, so they do not
depend on spelling, and the owner can correct a kind in the file. **The kinds are word matches: the owner has not read them row by row.** In this order, the
first that fits:

- ``wrong_scope``: the note says the body should have been another scope (``investment_general``, "misslabeled"): step 1.1's error, not the cut's. A note
  about **the news** ("this is a misslabeled news") is about the body: every piece of that article gets the kind, the ones with no note too. A note about a
  piece ("this fragment is more like investment_general") is about that piece only;
- ``wrong_company``: the company was read from the title or is lost ("company missunderstood", "company is lost"): step 3's matter; the three answers are
  about the piece, so these rows stay in every level of the figures;
- ``ambiguous``: the body mixes companies or stories so that no cut reads well ("mix", "two companies", "two different stories"). A macro section the cut
  did not split off, an unreadable text or a broken story are **not** ambiguity: they stay ``kept`` and count against the cut;
- ``kept``: any other note, or none.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

QUESTIONS = ("faithful", "one_story", "complete")
KINDS = ("kept", "wrong_scope", "wrong_company", "ambiguous")
MARK_COLUMNS = (
    "review_order",
    "article_id",
    "segment_index",
    "text_hash",
    "extractor_revision",
    *QUESTIONS,
    "kind",
    "notes",
)
TEXT_COLUMNS = ("review_order", "text")
WRONG_SCOPE_WORDS = ("investment_general", "misslabeled", "mislabeled")
WRONG_COMPANY_WORDS = (
    "company missunderstood",
    "company misunderstood",
    "company is lost",
    "tagged from the title",
)
AMBIGUOUS_WORDS = ("mix", "two companies", "two different stories")
BODY_SCOPE_WORDS = ("misslabeled news", "mislabeled news")
Row = dict[str, str]


def text_hash(text: str) -> str:
    """SHA-256 of the piece, outer whitespace stripped (the draw's own rule)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def kind_of(note: str) -> str:
    """The kind of a note: ``wrong_scope``, ``wrong_company``, ``ambiguous`` or ``kept``, in that order of precedence."""
    low = note.lower()
    for kind, words in (
        ("wrong_scope", WRONG_SCOPE_WORDS),
        ("wrong_company", WRONG_COMPANY_WORDS),
        ("ambiguous", AMBIGUOUS_WORDS),
    ):
        if any(w in low for w in words):
            return kind
    return "kept"


def build(rows: list[Row], ids: list[Row]) -> tuple[list[Row], list[Row]]:
    """The marks rows and the texts rows of a marked sample, after checking marks and ties."""
    by_order = {i["review_order"]: i for i in ids}
    revisions = {i["extractor_revision"] for i in ids}
    if len(revisions) != 1:
        raise SystemExit(f"the ids file names {len(revisions)} cut revisions, expected one")
    marks: list[Row] = []
    texts: list[Row] = []
    for r in rows:
        order = r["review_order"]
        answers = {q: r[q].strip().lower() for q in QUESTIONS}
        for q, a in answers.items():
            if a not in ("yes", "no"):
                raise SystemExit(f"review_order {order}: {q} is {r[q]!r}, not yes / no")
        want = by_order.get(order)
        if want is None or want["text_hash"] != text_hash(r["text"]):
            raise SystemExit(
                f"review_order {order}: the text is not the one the ids file ties to it"
            )
        note = r.get("notes", "").strip()
        marks.append(
            {
                "review_order": order,
                "article_id": r["article_id"],
                "segment_index": r["segment_index"],
                "text_hash": want["text_hash"],
                "extractor_revision": want["extractor_revision"],
                **answers,
                "kind": kind_of(note),
                "notes": note,
            }
        )
        texts.append({"review_order": order, "text": r["text"].strip()})
    whole = {
        m["article_id"]
        for m in marks
        if m["kind"] == "wrong_scope" and any(w in m["notes"].lower() for w in BODY_SCOPE_WORDS)
    }
    for m in marks:
        if m["article_id"] in whole:
            m["kind"] = "wrong_scope"
    return marks, texts


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        w.writerows(rows)


def write(marks: list[Row], texts: list[Row], marks_path: Path, texts_path: Path) -> None:
    for p in (marks_path, texts_path):
        if p.exists():
            raise SystemExit(f"{p} exists: it is not overwritten")
    _write_csv(marks_path, MARK_COLUMNS, marks)
    _write_csv(texts_path, TEXT_COLUMNS, texts)


def _read(path: Path) -> list[Row]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--marked", type=Path, default=Path("data/cut_sample_to_tag.csv"))
    ap.add_argument("--ids", type=Path, default=Path("data/cut_sample_ids.csv"))
    ap.add_argument("--marks", type=Path, default=Path("gold/cut_sample_marks.csv"))
    ap.add_argument("--texts", type=Path, default=Path("gold/texts/cut_sample_texts.csv"))
    args = ap.parse_args(argv)
    marks, texts = build(_read(args.marked), _read(args.ids))
    write(marks, texts, args.marks, args.texts)
    kinds = [m["kind"] for m in marks]
    sys.stdout.write(
        f"{len(marks)} marks -> {args.marks}; {len(texts)} texts -> {args.texts}; kinds: "
        + ", ".join(f"{k} {kinds.count(k)}" for k in KINDS)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
