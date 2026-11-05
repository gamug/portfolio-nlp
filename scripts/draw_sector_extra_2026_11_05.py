"""Draw extra sector-like bodies for the sub-industry gate (Work item 18, T-146c), free of LLM calls.

    uv run scripts/draw_sector_extra_2026_11_05.py [--out data] [--per-sector 4] [--seed 20261105] [--yes]

The 240-body gate holds only 16 sector bodies, too few to score the sub-industry. This draws
``--per-sector`` more for each of the 11 GICS sectors from the stored corpus (SOURCE, read-only),
by free rules: the headline names an industry (a noun such as *airlines*, *chipmakers*, *banks*)
and no universe company, the body repeats that noun at least ``MIN_HITS`` times and names at most
``MAX_SUBJECTS`` companies, and neither the body nor any pair of it is among the 720 judged
articles. Within a sector the pick is random (seeded), not the best scoring, so the sample is not
only the obvious cases. The rules only find candidates: the owner decides the scope.

Writes ``<out>/sector_extra_ids.csv`` (drawn once; a second run does not draw again) and
``<out>/sector_extra_to_tag.csv``, a tagging file in the review file's columns (``spanish_title`` /
``spanish_text`` after ``text``; the v1 columns are empty) to fill in ``scope``, ``sub_industry``
and ``notes`` (and the others where they apply). The Spanish aid is the approved paid translation
of the same engine as the other files (about 1 USD per 720 articles); nothing is bought without
``--yes``. Neither file is overwritten once it exists.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import retag, triage
from news_nlp.company_names import NameBook, _key
from news_nlp.triage import CompanyResolver

IDS_NAME = "sector_extra_ids.csv"
TO_TAG_NAME = "sector_extra_to_tag.csv"
MIN_HITS = 3  # the industry noun appears at least this often in the body
MAX_SUBJECTS = 2  # companies named twice or in the title: more than this reads as a company story
TEXT_CAP = 6000  # what the judge and the owner read (title + first 6,000 characters)
STAGE_TEXT = "\n[... truncated ...]"

#: Plural actors of an industry: a headline naming them is about the industry (``airlines``).
AGENT_NOUNS: dict[str, tuple[str, ...]] = {
    "Energy": ("refiners", "drillers", "oil majors"),
    "Materials": ("miners", "chemical makers", "gold miners"),
    "Industrials": ("airlines", "railroads", "truckers", "defense contractors"),
    "Consumer Discretionary": (
        "automakers",
        "carmakers",
        "retailers",
        "homebuilders",
        "restaurants",
    ),
    "Consumer Staples": ("supermarkets", "grocers", "food makers"),
    "Health Care": ("drugmakers", "hospitals", "insurers"),
    "Financials": ("banks", "lenders", "brokerages", "asset managers"),
    "Information Technology": ("chipmakers", "chip makers", "software makers"),
    "Communication Services": ("telecoms", "broadcasters", "publishers"),
    "Utilities": ("utilities",),
    "Real Estate": ("REITs", "landlords"),
}
#: Words that name an industry only when followed by one of :data:`INDUSTRY_TAILS` (``oil stocks``).
GENERIC_NOUNS: dict[str, tuple[str, ...]] = {
    "Energy": ("oil", "natural gas", "energy", "oilfield services"),
    "Materials": ("chemicals", "steel", "mining", "gold", "fertilizer", "lithium"),
    "Industrials": ("aerospace", "defense", "shipping", "airline", "trucking", "freight"),
    "Consumer Discretionary": ("auto", "retail", "apparel", "restaurant", "homebuilding", "travel"),
    "Consumer Staples": ("grocery", "beverage", "tobacco", "packaged food", "food"),
    "Health Care": ("biotech", "pharma", "pharmaceutical", "medical device", "health care"),
    "Financials": ("bank", "banking", "insurance", "payments", "fintech"),
    "Information Technology": ("software", "semiconductor", "chip", "cloud", "tech", "hardware"),
    "Communication Services": ("telecom", "media", "streaming", "advertising", "social media"),
    "Utilities": ("utility", "power", "electric utility", "renewable energy"),
    "Real Estate": ("real estate", "housing", "commercial real estate", "office"),
}
INDUSTRY_TAILS = (
    r"(?:stocks?|industry|industries|sector|companies|firms|makers|market|business|giants)"
)
#: Brokers and banks a sector headline cites as its source (``... says JPMorgan``).
SOURCES = frozenset(
    {
        "jpmorgan",
        "jp morgan",
        "goldman sachs",
        "goldman",
        "morgan stanley",
        "citi",
        "citigroup",
        "bank of america",
        "wells fargo",
        "ubs",
        "barclays",
        "guggenheim",
        "bernstein",
        "cramer",
    }
)


def _pattern(noun: str, tail: str = "") -> re.Pattern[str]:
    return re.compile(rf"(?<![\w&]){re.escape(noun)}{tail}(?![\w&])", re.IGNORECASE)


NOUN_PATTERNS = {
    **{(sector, n): _pattern(n) for sector, ns in AGENT_NOUNS.items() for n in ns},
    **{
        (sector, n): _pattern(n, rf"\s+{INDUSTRY_TAILS}")
        for sector, ns in GENERIC_NOUNS.items()
        for n in ns
    },
}
INDUSTRY_NOUNS = {s: (*AGENT_NOUNS.get(s, ()), *GENERIC_NOUNS.get(s, ())) for s in GENERIC_NOUNS}
GENERIC = re.compile(r"(?<![\w&])(?:sector|industry|industries)(?![\w&])", re.IGNORECASE)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def excluded_hashes(conn: db.NewsNlpDatabase, ids: set[int]) -> set[str]:
    """Body hashes of the given articles (the 720 judged ones): no pair of those bodies is drawn."""
    out: set[str] = set()
    for (body,) in conn.execute(
        "SELECT body_text FROM source.articles WHERE id IN (SELECT value FROM json_each(?))",
        (str(sorted(ids)),),
    ):
        out.add(triage.body_hash(body or ""))
    return out


def names_company(book: NameBook, title: str) -> bool:
    """True when the headline names a universe company by its name, nickname, brand or ticker
    (``GM``, ``Exxon``), not counting a bank cited as the source. First words alone do not count
    (``Energy`` would match every energy headline)."""
    words = re.findall(r"[\w&.'-]+", title)
    for size in (4, 3, 2, 1):
        for i in range(len(words) - size + 1):
            gram = " ".join(words[i : i + size])
            key = _key(gram)
            if key in SOURCES or book._inferred.get(key) == "first word":
                continue
            if size == 1 and gram.islower():
                continue  # an everyday word, not a ticker or a name
            if book.lookup(gram).ticker is not None and book.lookup(gram).note != "first word":
                return True
    return False


def candidates(
    conn: db.NewsNlpDatabase, resolver: CompanyResolver, book: NameBook, skip: set[str]
) -> dict[str, list[dict[str, object]]]:
    """Per sector, the unique bodies that pass the rules. The headline is tested first, without
    the body, so only a few thousand bodies are read and resolved."""
    titled: list[tuple[int, str, str, tuple[str, str] | None]] = []
    for article_id, ticker, title in conn.execute(
        "SELECT id, ticker, title FROM source.articles "
        "WHERE fetch_status = 'ok' AND body_text IS NOT NULL AND body_text != ''"
    ):
        if not title or resolver.resolve(title) or names_company(book, title):
            continue  # a headline that names a company is a company story
        hit = next((key for key, pat in NOUN_PATTERNS.items() if pat.search(title)), None)
        if hit or GENERIC.search(title):
            titled.append((int(article_id), ticker, title, hit))
    print(f"headlines naming an industry and no company: {len(titled):,}", flush=True)
    seen: set[str] = set()
    pool: dict[str, list[dict[str, object]]] = defaultdict(list)
    for article_id, ticker, title, title_hit in titled:
        hit = title_hit
        row = conn.execute(
            "SELECT company, pub_date, source_domain, body_text FROM source.articles WHERE id = ?",
            (article_id,),
        ).fetchone()
        key = triage.body_hash(row["body_text"])
        if key in skip or key in seen:
            continue
        body = triage.strip_boilerplate(row["body_text"])
        if hit is None:  # a generic headline: the sector is the noun the body repeats most
            counts = {k: len(p.findall(body)) for k, p in NOUN_PATTERNS.items()}
            best = max(counts, key=lambda k: counts[k])
            if counts[best] < MIN_HITS:
                continue
            hit = best
        hits = len(NOUN_PATTERNS[hit].findall(body))
        if hits < MIN_HITS:
            continue
        in_title = resolver.resolve(title)
        named = triage.subjects(resolver.resolve(f"{title}\n{body}"), in_title)
        if len(named) > MAX_SUBJECTS:
            continue
        seen.add(key)
        cut = row["body_text"][:TEXT_CAP]
        text = cut + STAGE_TEXT if len(row["body_text"]) > TEXT_CAP else cut
        pool[hit[0]].append(
            {
                "article_id": article_id,
                "ticker": ticker,
                "company": row["company"] or "",
                "pub_date": row["pub_date"] or "",
                "source_domain": row["source_domain"] or "",
                "title": title,
                "text": text,
                "noun": hit[1],
                "hits": hits,
                "subjects": len(named),
            }
        )
    return pool


def draw(
    pool: dict[str, list[dict[str, object]]], per_sector: int, seed: int
) -> list[dict[str, object]]:
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    chosen: list[dict[str, object]] = []
    for sector in INDUSTRY_NOUNS:
        found = sorted(pool.get(sector, []), key=lambda r: int(str(r["article_id"])))
        rng.shuffle(found)
        # one noun is not allowed to fill a sector: take round-robin over the nouns
        by_noun: dict[str, list[dict[str, object]]] = defaultdict(list)
        for r in found:
            by_noun[str(r["noun"])].append(r)
        picked: list[dict[str, object]] = []
        while len(picked) < per_sector and any(by_noun.values()):
            for noun in sorted(by_noun):
                if by_noun[noun] and len(picked) < per_sector:
                    picked.append({**by_noun[noun].pop(), "sector": sector})
        chosen += picked
    # a sector with few candidates leaves its places to the others, at random, up to the target
    target = per_sector * len(INDUSTRY_NOUNS)
    taken = {str(r["article_id"]) for r in chosen}
    rest = [
        {**r, "sector": sector}
        for sector, found in pool.items()
        for r in found
        if str(r["article_id"]) not in taken
    ]
    rest.sort(key=lambda r: int(str(r["article_id"])))
    rng.shuffle(rest)
    return chosen + rest[: max(0, target - len(chosen))]


def tag_rows(chosen: list[dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for order, r in enumerate(chosen, 1):
        row = {
            "review_order": str(order),
            **{
                k: str(r[k])
                for k in (
                    "article_id",
                    "ticker",
                    "company",
                    "pub_date",
                    "source_domain",
                    "title",
                    "text",
                )
            },
            **dict.fromkeys(("v1_label", "v1_notes", "v1_scope_derived"), ""),
            **dict.fromkeys(retag.TAG_COLUMNS, ""),
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--per-sector", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--yes", action="store_true", help="pay for the Spanish translation")
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    args = parser.parse_args()
    out: Path = args.out
    if (out / TO_TAG_NAME).exists():
        sys.exit(f"{out / TO_TAG_NAME} exists: it is your tagging file and is not overwritten")
    if (out / IDS_NAME).exists():
        chosen: list[dict[str, object]] = [dict(r) for r in _read(out / IDS_NAME)]
        print(f"{out / IDS_NAME} exists: the set is not drawn again")
        by_id = {}
        conn = db.connect_pipeline()
        for r in chosen:
            row = conn.execute(
                "SELECT ticker, company, pub_date, source_domain, title, body_text "
                "FROM source.articles WHERE id = ?",
                (int(str(r["article_id"])),),
            ).fetchone()
            body = row["body_text"]
            cut = body[:TEXT_CAP]
            by_id[str(r["article_id"])] = {
                "article_id": r["article_id"],
                "ticker": row["ticker"],
                "company": row["company"] or "",
                "pub_date": row["pub_date"] or "",
                "source_domain": row["source_domain"] or "",
                "title": row["title"] or "",
                "text": cut + STAGE_TEXT if len(body) > TEXT_CAP else cut,
            }
        chosen = [by_id[str(r["article_id"])] for r in chosen]
    else:
        conn = db.connect_pipeline()
        universe = retag_cli.companies()
        reviewed = {int(r["article_id"]) for r in _read(out / retag.REVIEW_NAME)}
        skip = excluded_hashes(conn, reviewed)
        book = NameBook(universe, retag.Universe.of(universe).tickers)
        pool = candidates(conn, CompanyResolver(universe), book, skip)
        print("candidates per sector:", {s: len(v) for s, v in pool.items()})
        chosen = draw(pool, args.per_sector, args.seed)
        _write(
            out / IDS_NAME,
            [
                {
                    k: str(r[k])
                    for k in ("article_id", "sector", "noun", "hits", "subjects", "title")
                }
                for r in chosen
            ],
        )
        print(f"{len(chosen)} drawn:", dict(Counter(str(r["sector"]) for r in chosen)))
    rows = tag_rows(chosen)
    rows = retag_cli.add_spanish(
        out, rows, retag_cli._translator(), yes=args.yes, max_new=args.max_new
    )
    try:
        _write(out / TO_TAG_NAME, rows)
    except PermissionError:
        sys.exit(f"cannot write {out / TO_TAG_NAME}: close it in Excel and run again")
    print(f"{len(rows)} bodies -> {out / TO_TAG_NAME}")


if __name__ == "__main__":
    main()
