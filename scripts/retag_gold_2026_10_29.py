"""The owner's re-tag of the 720 judged articles under the new taxonomy (Work item 18, T-146b).

    uv run scripts/retag_gold_2026_10_29.py export [--out data]   # writes the file to tag
    uv run scripts/retag_gold_2026_10_29.py normalise [--out data]  # names -> tickers, new file
    uv run scripts/retag_gold_2026_10_29.py check  [--out data] [--file NAME]  # validates a file

``export`` reads ``data/category_gold_review.csv`` and its key (tags v1) and writes
``category_gold_v2_review.csv`` (the article, the v1 label and note and the scope they imply as
guidance, EMPTY ``scope`` / ``companies`` / ``company_labels`` / ``mentions`` / ``sub_industry`` /
``notes``) and
``category_gold_v2_key.csv`` (the v1 key, where the LLM's label stays: the owner is not shown it).
It **refuses to overwrite a file that already has scopes filled in**. ``check`` lists the
untagged rows and every row that breaks the rules (``news_nlp.retag.check_row``).
Column format: ``companies`` = ``AAPL; MSFT``; ``company_labels`` = ``AAPL:product_innovation; MSFT:other``.
The owner may type names or nicknames instead of tickers (``Apple; Google``, ``Apple:other``):
``normalise`` writes ``category_gold_v2_review_tickers.csv`` with tickers and lists every text it
could not resolve or that fits several companies (left as typed, never guessed), and the matches
it inferred (a first word, a brand, a ticker that is also a word) for the owner to confirm; the
review file is not changed. ``check --file category_gold_v2_review_tickers.csv`` then validates that copy.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import company_names, retag
from news_nlp.gold_review import TaggedFileError, write_review_files
from news_nlp.triage import Company


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def cmd_export(out: Path, *, universe_companies: list[Company] | None = None) -> None:
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
    write_csv(out / "category_gold_v2_guide.csv", retag.guide_rows())
    write_csv(
        out / "gics_sub_industries.csv", retag.sub_industry_rows(universe_companies or companies())
    )
    print(
        f"{len(review)} articles -> {out / retag.REVIEW_NAME}; guide and sub-industry list written"
    )


def companies() -> list[Company]:
    import news_nlp as db  # noqa: PLC0415 -- only these commands need the SOURCE store

    conn = db.connect_pipeline()
    rows = conn.execute(
        "SELECT DISTINCT ticker, company, gics_sector, gics_sub_industry FROM source.articles "
        "WHERE fetch_status = 'ok' ORDER BY ticker, company"
    ).fetchall()
    return [Company(r[0], r[1] or "", r[2] or "", r[3] or "") for r in rows]


def load_universe() -> retag.Universe:
    return retag.Universe.of(companies())


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


TICKERS_NAME = "category_gold_v2_review_tickers.csv"


def cmd_normalise(
    out: Path, *, universe_companies: list[Company] | None = None, file: str | None = None
) -> None:
    """Write the review file with names and nicknames replaced by tickers."""
    firms = universe_companies or companies()
    book = company_names.NameBook(firms, retag.Universe.of(firms).tickers)
    rows = _read(out / (file or retag.REVIEW_NAME))
    result: list[dict[str, str]] = []
    problems: list[str] = []
    confirm: list[str] = []
    for row in rows:
        new, found, inferred = company_names.normalise_row(book, row)
        result.append(new)
        tag = f"article {row.get('article_id')}"
        problems += [f"  {tag}: {p}" for p in found]
        confirm += [f"  {tag}: {c}" for c in inferred]
    write_csv(out / TICKERS_NAME, result)
    if problems:
        print("left as typed (fix them in the review file):", *problems, sep="\n")
    if confirm:
        print("converted by inference (please confirm):", *confirm, sep="\n")
    print(
        f"{len(result)} rows -> {out / TICKERS_NAME}; {len(problems)} texts left as typed, "
        f"{len(confirm)} to confirm"
    )


def cmd_check(
    out: Path, *, universe: retag.Universe | None = None, file: str | None = None
) -> None:
    expected = [int(r["article_id"]) for r in _read(out / retag.KEY_NAME)]
    report = retag.load_gold_v2(
        out / (file or retag.REVIEW_NAME), expected, universe or load_universe()
    )
    print(
        f"tagged and valid {len(report.good)}, with errors {len(report.bad)}, "
        f"untagged {report.untagged}, expected {len(expected)}"
    )
    print("scopes:", dict(Counter(g.scope for g in report.good)))
    for label, ids in (
        ("missing ids", report.missing),
        ("duplicated ids", report.duplicated),
        ("unknown ids", report.unknown),
    ):
        if ids:
            print(f"  {label}: {ids}")
    for article_id, errors in sorted(report.bad.items()):
        print(f"  article {article_id}: {'; '.join(errors)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("export", "normalise", "check"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--file", help="review file name inside --out (check, normalise)")
    args = parser.parse_args()
    if args.command == "export":
        cmd_export(args.out)
    elif args.command == "normalise":
        cmd_normalise(args.out, file=args.file)
    else:
        cmd_check(args.out, file=args.file)


if __name__ == "__main__":
    main()
