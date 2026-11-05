"""Build the training set of the step-4 attribution model (Work item 18, T-148b2, D27).

    uv run scripts/build_company_segments_2026_11_05.py [--out data] [--gold gold/company_segments_train.csv]

Every unique body of the corpus written as one segment per company
(``news_nlp.company_texts.label_segments``) gives one row per labelled segment: the company its
paragraph is about. The labels are automatic, not the owner's tags: a ``lead`` segment names its
company at its start (clean), a ``continuation`` inherits the previous company (noisy). The
attribution model learns who a sentence is about with the company name masked, so it can attribute
the sentences of a narrative body that name nobody.

The repository is public, so the file holds no news text: the article, the position of the
segment in ``company_texts.segments(body)`` and the ticker. The text is rebuilt from SOURCE with
``segments`` (``segment_text``). A body that shares its text with any article the owner has tagged
(the 720, the 44 sector bodies, the held-out 200) is left out, so the set never trains on a
body that gold measures. ``split`` is ``val`` for one body in ten (by body hash), else ``train``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sector_extra_2026_11_05 as extra
import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import company_texts, retag, triage
from news_nlp.triage import CompanyResolver

COLUMNS = (
    "article_id",
    "body_hash",
    "year",
    "split",
    "segment_index",
    "ticker",
    "kind",
)
VAL_EVERY = 10


def split_of(body_hash: str) -> str:
    return "val" if int(body_hash[:8], 16) % VAL_EVERY == 0 else "train"


def tagged_ids(out: Path) -> set[int]:
    """Every article the owner has tagged or been given to tag: the 720, the sector extras, the
    held-out set."""
    ids: set[int] = set()
    for name in (retag.REVIEW_NAME, extra.TO_TAG_NAME, "heldout_ids.csv"):
        ids |= {int(r["article_id"]) for r in extra._read(out / name)}
    return ids


def build(out: Path) -> list[dict[str, str]]:
    conn = db.connect_pipeline()
    resolver = CompanyResolver(retag_cli.companies())
    skip = extra.excluded_hashes(conn, tagged_ids(out))
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    bodies = left_out = scanned = 0
    for article in conn.execute(
        "SELECT id, substr(pub_date, 1, 4) AS year, body_text FROM source.articles "
        "WHERE fetch_status = 'ok' AND body_text IS NOT NULL ORDER BY id"
    ):
        body_hash = triage.body_hash(article["body_text"])
        if body_hash in seen:
            continue
        seen.add(body_hash)
        scanned += 1
        labels = company_texts.label_segments(article["body_text"], resolver)
        if labels is None:
            continue
        if body_hash in skip:
            left_out += 1
            continue
        bodies += 1
        rows += [
            {
                "article_id": str(article["id"]),
                "body_hash": body_hash,
                "year": article["year"] or "",
                "split": split_of(body_hash),
                "segment_index": str(label.position),
                "ticker": label.ticker,
                "kind": label.kind,
            }
            for label in labels
        ]
    print(
        f"{scanned} unique bodies; {bodies} structured; {left_out} structured but tagged/held out"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold/company_segments_train.csv"))
    args = parser.parse_args()
    rows = build(args.out)
    with args.gold.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} segments -> {args.gold}")
    print("kind:", dict(Counter(r["kind"] for r in rows)))
    print("split (lead segments):", dict(Counter(r["split"] for r in rows if r["kind"] == "lead")))
    print(
        "years:",
        dict(
            sorted(Counter(r["year"] for r in {r["body_hash"]: r for r in rows}.values()).items())
        ),
    )


if __name__ == "__main__":
    main()
