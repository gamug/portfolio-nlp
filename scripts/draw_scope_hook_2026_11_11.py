"""Draw the bodies where a company is only the hook, to tag (T-147; owner 2026-11-11): out_of_scope examples.

    uv run scripts/draw_scope_hook_2026_11_11.py [--out data] [--count 30] [--sample 10] [--seed 20261111] [--yes]

The model reads a title that names one company as ``single`` even when the story is about the market or the economy (the
9 held-out ``out_of_scope`` bodies it takes for ``single``: CSX as a read on the economy, Home Depot in a futures report,
KeyCorp's CEO on tariffs, Nvidia as the wait of the European session). This draws more of them from the stored corpus (SOURCE,
read-only): the title names exactly one universe company **and** the first 800 characters use at least ``--min-macro`` market or
macro terms. None is a body (or a twin of one) already tagged in any gold file, the 51 of ``scope_enrich`` included.

The rules only find candidates; the owner decides the scope (most should be ``market_economy`` or ``investing_general``, some
will turn out ``single``: that is what the model has to learn to tell apart). Writes ``<out>/scope_hook_to_tag.csv`` in the
tagging format (``spanish_title`` / ``spanish_text`` right after ``text``; nothing is bought without ``--yes``). Not overwritten.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_scope_enrich_2026_11_10 as enrich
import draw_sector_extra_2026_11_05 as extra
import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import retag, triage
from news_nlp.triage import CompanyResolver

TO_TAG_NAME = "scope_hook_to_tag.csv"
FRAGMENT = 800  # the characters the scope model reads
MACRO = re.compile(
    r"\b(?:stock futures|futures|dow|s&p 500|nasdaq|wall street|stock market|markets?|fed|federal reserve|rate (?:hike|cut)s?"
    r"|inflation|gdp|economy|economic|tariffs?|treasury|jobs report|unemployment|oil prices?|investors|traders|bellwether"
    r"|recession|interest rates?|consumer spending|earnings season)\b",
    re.IGNORECASE,
)


def macro_terms(text: str) -> int:
    return len(MACRO.findall(text))


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--sample", type=int, default=10, help="take 1 in this many articles")
    parser.add_argument("--min-macro", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--yes", action="store_true", help="pay for the Spanish translation")
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    args = parser.parse_args()
    out: Path = args.out
    if (out / TO_TAG_NAME).exists():
        sys.exit(f"{out / TO_TAG_NAME} exists: it is your tagging file and is not overwritten")
    conn = db.connect_pipeline()
    universe = retag_cli.companies()
    tagged = enrich.tagged_article_ids(args.gold) | {
        int(r["article_id"]) for r in extra._read(args.gold / "scope_enrich.csv")
    }
    skip = extra.excluded_hashes(conn, tagged)
    resolver = CompanyResolver(universe)
    pool: list[dict[str, object]] = []
    seen: set[str] = set()
    for article_id, ticker, title, body, company, pub_date, domain in conn.execute(
        "SELECT a.id, a.ticker, a.title, a.body_text, a.company, a.pub_date, a.source_domain "
        "FROM source.articles a WHERE a.fetch_status = 'ok' AND a.body_text IS NOT NULL AND a.body_text != '' "
        "AND a.id % ? = ? ORDER BY a.id",
        (args.sample, args.seed % args.sample),
    ):
        key = triage.body_hash(body)
        if key in seen or key in skip:
            continue
        seen.add(key)
        named = {t for _, t in resolver.counted_matches(title or "")}
        if len(named) != 1 or macro_terms(body[:FRAGMENT]) < args.min_macro:
            continue
        cut = body[: extra.TEXT_CAP]
        pool.append(
            {
                "article_id": int(article_id),
                "ticker": ticker,
                "company": company or "",
                "pub_date": pub_date or "",
                "source_domain": domain or "",
                "title": title or "",
                "text": cut + extra.STAGE_TEXT if len(body) > extra.TEXT_CAP else cut,
                "rule": f"hook:{macro_terms(body[:FRAGMENT])}",
            }
        )
    print(
        f"{len(pool)} candidates with a company in the title and macro words in the opening",
        flush=True,
    )
    rng = random.Random(args.seed)  # noqa: S311 -- a reproducible draw, not security
    pool.sort(key=lambda r: int(str(r["article_id"])))
    rng.shuffle(pool)
    chosen = pool[: args.count]
    rows = extra.tag_rows(chosen)
    rows = retag_cli.add_spanish(
        out, rows, retag_cli._translator(), yes=args.yes, max_new=args.max_new
    )
    retag.require_tagging_format(rows, TO_TAG_NAME)
    extra._write(out / TO_TAG_NAME, rows)
    print(f"{len(rows)} bodies -> {out / TO_TAG_NAME}")


if __name__ == "__main__":
    main()
