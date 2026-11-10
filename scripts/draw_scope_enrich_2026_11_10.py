"""Draw the bodies to tag so the rare scopes can be judged (T-147; owner 2026-11-10): sector and out-of-scope.

    uv run scripts/draw_scope_enrich_2026_11_10.py [--out data] [--per-sector 3] [--per-rule 6] [--seed 20261110] [--yes]

The held-out 200 has 9 `sector` and 26 `out_of_scope` bodies; D33 gates a scope's recall only from 40 gold bodies. This draws
about 31 more `sector` and 14 more `out_of_scope` candidates from the stored corpus (SOURCE, read-only), none of them a body
(or a twin of a body) already tagged in any gold file:

* sector candidates: ``draw_sector_extra`` rules (the headline names an industry and no company), ``--per-sector`` for each of
  the 11 GICS sectors;
* out-of-scope candidates: a seeded sample of the properly fetched bodies (1 in ``--sample``) classified by the triage rules;
  ``--per-rule`` are taken from each of ``market``, ``economy`` and ``no_company``.

The rules only find candidates; the owner decides the scope. The set is for **measuring** the rare scopes (class-conditional
recall, reported apart, unweighted: a rule-picked sample is easier than the corpus), not for the weighted held-out figures.
Writes ``<out>/scope_enrich_ids.csv`` (drawn once) and ``<out>/scope_enrich_to_tag.csv`` in the tagging format
(``spanish_title`` / ``spanish_text`` after ``text``; nothing is bought without ``--yes``). Neither file is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sector_extra_2026_11_05 as extra
import retag_gold_2026_10_29 as retag_cli
import size_triage_2026_10_28 as size

import news_nlp as db
from news_nlp import retag, triage
from news_nlp.company_names import NameBook
from news_nlp.triage import CompanyResolver

IDS_NAME = "scope_enrich_ids.csv"
TO_TAG_NAME = "scope_enrich_to_tag.csv"
RULE_GROUPS = ("market", "economy", "no_company")
#: Every file whose bodies are already tagged or drawn: none of them (nor a twin) is drawn again.
TAGGED = (
    "gold/category_gold_v2.csv",
    "gold/category_gold_v2_test.csv",
    "gold/sentiment_gold.csv",
    "gold/sentiment_train.csv",
    "gold/sentiment_train_2.csv",
    "gold/segment_check_gold.csv",
)


def tagged_article_ids(gold: Path) -> set[int]:
    ids: set[int] = set()
    for name in TAGGED:
        for row in extra._read(gold / Path(name).name):
            ids.add(int(row["article_id"]))
    return ids


def out_of_scope_pool(
    conn: db.NewsNlpDatabase, universe: list, skip: set[str], sample: int, seed: int
) -> dict[str, list[dict[str, object]]]:
    """Unique bodies of a 1-in-*sample* slice of the corpus that the rules call market, economy or no_company."""
    resolver = CompanyResolver(universe)
    terms = list(size.sector_terms(universe))
    pool: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[str] = set()
    frame: Counter[str] = Counter()
    for article_id, ticker, title, body, url, company, pub_date, domain in conn.execute(
        "SELECT a.id, a.ticker, a.title, a.body_text, d.url, a.company, a.pub_date, a.source_domain "
        "FROM source.articles a LEFT JOIN source.discovered_urls d ON d.id = a.id "
        "WHERE a.fetch_status = 'ok' AND a.body_text IS NOT NULL AND a.body_text != '' AND a.id % ? = ? "
        "ORDER BY a.id",
        (sample, seed % sample),
    ):
        key = triage.body_hash(body)
        if key in seen or key in skip:
            continue
        seen.add(key)
        scope, _ = size.classify(
            resolver, title or "", body, size.slug_of(url), [], terms, **size.FULL
        )
        frame[scope.scope] += 1
        if scope.scope in RULE_GROUPS:
            cut = body[: extra.TEXT_CAP]
            pool[scope.scope].append(
                {
                    "article_id": int(article_id),
                    "ticker": ticker,
                    "company": company or "",
                    "pub_date": pub_date or "",
                    "source_domain": domain or "",
                    "title": title or "",
                    "text": cut + extra.STAGE_TEXT if len(body) > extra.TEXT_CAP else cut,
                    "rule": scope.scope,
                }
            )
    print("rule scopes in the slice:", dict(frame), flush=True)
    return pool


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument("--per-sector", type=int, default=3)
    parser.add_argument("--per-rule", type=int, default=6)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--yes", action="store_true", help="pay for the Spanish translation")
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    args = parser.parse_args()
    out: Path = args.out
    if (out / TO_TAG_NAME).exists():
        sys.exit(f"{out / TO_TAG_NAME} exists: it is your tagging file and is not overwritten")
    conn = db.connect_pipeline()
    universe = retag_cli.companies()
    skip = extra.excluded_hashes(conn, tagged_article_ids(args.gold))
    print(f"{len(skip):,} bodies already tagged: left out", flush=True)
    book = NameBook(universe, retag.Universe.of(universe).tickers)
    sectors = extra.draw(
        extra.candidates(conn, CompanyResolver(universe), book, skip), args.per_sector, args.seed
    )
    for r in sectors:
        r["rule"] = f"sector:{r['sector']}"
    pool = out_of_scope_pool(conn, universe, skip, args.sample, args.seed)
    rng = random.Random(args.seed)  # noqa: S311 -- a reproducible draw, not security
    others: list[dict[str, object]] = []
    for group in RULE_GROUPS:
        found = sorted(pool.get(group, []), key=lambda r: int(str(r["article_id"])))
        rng.shuffle(found)
        others += found[: args.per_rule]
    chosen = sectors + others
    print(
        f"{len(sectors)} sector + {len(others)} out-of-scope candidates:",
        dict(Counter(str(r["rule"]).split(":")[0] for r in chosen)),
    )
    with (out / IDS_NAME).open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, ["article_id", "rule"])
        writer.writeheader()
        writer.writerows({"article_id": r["article_id"], "rule": r["rule"]} for r in chosen)
    rows = extra.tag_rows(chosen)
    rows = retag_cli.add_spanish(
        out, rows, retag_cli._translator(), yes=args.yes, max_new=args.max_new
    )
    retag.require_tagging_format(rows, TO_TAG_NAME)
    extra._write(out / TO_TAG_NAME, rows)
    print(f"{len(rows)} bodies -> {out / TO_TAG_NAME}")


if __name__ == "__main__":
    main()
