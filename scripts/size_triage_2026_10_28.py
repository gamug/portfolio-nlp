"""Size the triage-first pipeline on the stored corpus (PLAN.md Work item 18 P1, TASKS T-144).

    uv run scripts/size_triage_2026_10_28.py [size] [--sample 1000] [--out data]
    uv run scripts/size_triage_2026_10_28.py gold     # the rules against the owner's 720 tags

Read-only and free (no model, no LLM). Over the SOURCE articles with ``fetch_status = 'ok'`` it

* groups the (URL, ticker) pairs into unique bodies (``news_nlp.triage.body_hash``, D11);
* resolves the universe companies each body names (name, explicit ticker forms, the stored NER
  ``ORG`` spans of its pairs; no alias table yet, so the counts of single-company bodies are a
  floor when a company goes by another name);
* drops data credits and site furniture, and does not count a company quoted as an opinion's
  source ("Goldman Sachs analysts");
* keeps as subjects the companies named twice or in the title (``news_nlp.triage.subjects``);
* marks multi-company round-ups by their headline or URL slug, a title naming three companies or
  more, or five explicit ticker forms or more (``news_nlp.triage.is_roundup``), and
  gives each body a rule-based scope candidate (single, multi, multi with one dominant company,
  or no company: market / economy / sector / unknown) and a GICS guess;
* checks whether each pair's tagged ticker is among the companies its body names.

``gold`` scores each step of these rules against the scope the owner's tags and notes imply on
the 720 judged articles (``news_nlp.triage.gold_scope``): a development set, since the rules were
tuned while looking at it; the ~300 gold bodies of Work item 18 P2 are the held-out test.

Prints the tables for ``docs/evaluation.md`` and writes ``<out>/triage_sample_bodies.csv``: a
seeded random sample of unique bodies (title, the first 6,000 characters, the rule scope and the
companies) that ``scripts/laya_check_2026_10_28.py sample`` reads. ``data/`` is git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import news_nlp as db
from news_nlp import triage
from news_nlp.triage import Company, CompanyResolver

SEED = 20261028
TEXT_CAP = 6000  # what the judge and the human read (title + first 6,000 characters)


def load_universe(conn: db.NewsNlpDatabase) -> list[Company]:
    rows = conn.execute(
        "SELECT ticker, company, gics_sector, gics_sub_industry FROM source.articles "
        "WHERE fetch_status = 'ok' GROUP BY ticker ORDER BY ticker"
    ).fetchall()
    return [Company(r[0], r[1] or "", r[2] or "", r[3] or "") for r in rows]


def sector_terms(universe: list[Company]) -> dict[str, tuple[str, str]]:
    """Keyword -> (sector, sub-industry): each GICS sub-industry name, split on '&' / ',' / 'and',
    in singular and plural."""
    terms: dict[str, tuple[str, str]] = {}
    for c in universe:
        for raw in re.split(r"\s*(?:&|,|\band\b)\s*", c.gics_sub_industry.lower()):
            part = raw.strip()
            if len(part) < 4:  # noqa: PLR2004 -- skip fragments
                continue
            singular = part[:-1] if part.endswith("s") else part
            for term in {part, singular}:
                terms.setdefault(term, (c.gics_sector, c.gics_sub_industry))
    return terms


def load_org_spans(conn: db.NewsNlpDatabase) -> dict[int, list[str]]:
    spans: dict[int, list[str]] = defaultdict(list)
    for article_id, text in conn.execute(
        "SELECT e.article_id, e.text FROM article_entities e "
        "JOIN source.articles a ON a.id = e.article_id "
        "WHERE e.entity_type = 'ORG' AND a.fetch_status = 'ok'"
    ):
        spans[int(article_id)].append(text)
    return spans


#: Rule steps, cumulative, as the ``gold`` mode reports them.
STEPS: tuple[tuple[str, dict[str, bool]], ...] = (
    ("every named company", {"subjects": False, "clean": False, "roundup": False}),
    ("+ subjects (twice or in the title)", {"subjects": True, "clean": False, "roundup": False}),
    ("+ boilerplate, analyst attribution", {"subjects": True, "clean": True, "roundup": False}),
    ("+ round-up structure", {"subjects": True, "clean": True, "roundup": True}),
)
FULL = STEPS[-1][1]
#: Rule scope -> the scopes the owner's tags can be mapped to (``news_nlp.triage.GOLD_SCOPES``).
TO_GOLD = {
    "single": "single",
    "multi_dominated": "single",
    "multi": "multi",
    "roundup": "multi",
    "market": "market_economy",
    "economy": "market_economy",
    "sector": "sector",
    "no_company": "unrelated",
}


def slug_of(url: str | None) -> str:
    path = (url or "").split("?", 1)[0].rstrip("/")
    return path.rsplit("/", 1)[-1].removesuffix(".html")


def classify(
    resolver: CompanyResolver,
    title: str,
    body: str,
    slug: str,
    spans: list[str],
    terms: list[str],
    *,
    subjects: bool,
    clean: bool,
    roundup: bool,
) -> tuple[triage.Scope, Counter[str]]:
    text = f"{title}\n{triage.strip_boilerplate(body) if clean else body}"
    named = resolver.resolve(text, spans, skip_attribution=clean)
    in_title = resolver.resolve(title, skip_attribution=clean)
    mentions = triage.subjects(named, in_title) if subjects else named
    is_roundup = roundup and triage.is_roundup(
        title, slug, len(in_title), resolver.ticker_forms(text)
    )
    return triage.scope_candidate(mentions, text, terms, roundup=is_roundup), mentions


def bucket(n: int) -> str:
    for upper, name in ((0, "0"), (1, "1"), (2, "2"), (5, "3-5"), (10, "6-10")):
        if n <= upper:
            return name
    return "11+"


def pct(n: int, total: int) -> str:
    return f"{n:>8,} ({100 * n / total:5.1f}%)" if total else f"{n:>8,}"


def cmd_gold(args: argparse.Namespace) -> None:
    """Each rule step against the scope the owner's tags and notes imply on the 720 articles."""
    with (args.out / "category_gold_review.csv").open(encoding="utf-8-sig") as fh:
        review = list(csv.DictReader(fh))
    conn = db.connect_pipeline()
    universe = load_universe(conn)
    resolver = CompanyResolver(universe)
    terms = list(sector_terms(universe))
    ids = [int(r["article_id"]) for r in review]
    marks = ",".join("?" for _ in ids)
    full = {
        int(i): (t or "", b or "", slug_of(u))
        for i, t, b, u in conn.execute(
            "SELECT a.id, a.title, a.body_text, d.url FROM source.articles a "  # noqa: S608
            f"LEFT JOIN source.discovered_urls d ON d.id = a.id WHERE a.id IN ({marks})",
            ids,
        )
    }
    spans: dict[int, list[str]] = defaultdict(list)
    for i, text in conn.execute(
        f"SELECT article_id, text FROM article_entities WHERE entity_type = 'ORG' AND article_id IN ({marks})",  # noqa: S608
        ids,
    ):
        spans[int(i)].append(text)
    gold = [triage.gold_scope(r["human_label"], r["human_notes"]) for r in review]
    print(f"gold scopes on {len(review)} articles: {dict(Counter(gold))}")
    for name, switches in STEPS:
        pred = [
            TO_GOLD[classify(resolver, *full[i], spans[i], terms, **switches)[0].scope] for i in ids
        ]
        hits = sum(p == g for p, g in zip(pred, gold, strict=True))
        low, high = triage.wilson(hits, len(gold))
        multi_tp = sum(p == g == "multi" for p, g in zip(pred, gold, strict=True))
        print(
            f"\n{name:<38} agreement {hits / len(gold):.3f} ({low:.3f}-{high:.3f}); multi recall "
            f"{multi_tp}/{gold.count('multi')}, multi precision {multi_tp}/{pred.count('multi')}"
        )
        print("  gold -> rules:", Counter(zip(gold, pred, strict=True)).most_common(10))


def cmd_size(args: argparse.Namespace) -> None:  # noqa: PLR0915 -- a linear report
    started = time.time()

    conn = db.connect_pipeline()
    universe = load_universe(conn)
    by_ticker = {c.ticker: c for c in universe}
    resolver = CompanyResolver(universe)
    terms = sector_terms(universe)
    org = load_org_spans(conn)
    print(f"universe: {len(universe)} tickers; ORG spans for {len(org):,} articles", flush=True)

    members: dict[str, list[tuple[int, str]]] = defaultdict(list)
    first: dict[str, tuple[str, str, str]] = {}
    for article_id, ticker, title, body, url in conn.execute(
        "SELECT a.id, a.ticker, a.title, a.body_text, d.url FROM source.articles a "
        "LEFT JOIN source.discovered_urls d ON d.id = a.id "
        "WHERE a.fetch_status = 'ok' AND a.body_text IS NOT NULL AND a.body_text != '' "
        "ORDER BY a.id"
    ):
        key = triage.body_hash(body)
        members[key].append((int(article_id), ticker))
        first.setdefault(key, (title or "", body, slug_of(url)))
    pairs = sum(len(m) for m in members.values())
    print(f"ok pairs with text: {pairs:,}; unique bodies: {len(members):,}", flush=True)

    terms_list = list(terms)
    scope_bodies: Counter[str] = Counter()
    scope_pairs: Counter[str] = Counter()
    companies_hist: Counter[str] = Counter()
    pairs_per_body: Counter[str] = Counter()
    gics_hits: Counter[tuple[str, str]] = Counter()
    multi_with_gics = 0
    tagged_found: Counter[str] = Counter()
    tagged_total: Counter[str] = Counter()
    twins = 0
    rows: list[dict[str, object]] = []
    for key, pair_list in members.items():
        title, body, slug = first[key]
        spans = [s for article_id, _ in pair_list for s in org.get(article_id, ())]
        scope, mentions = classify(resolver, title, body, slug, spans, terms_list, **FULL)
        scope_bodies[scope.scope] += 1
        scope_pairs[scope.scope] += len(pair_list)
        companies_hist[bucket(scope.companies)] += 1
        pairs_per_body[bucket(len(pair_list))] += 1
        canonical = {resolver.by_ticker.get(t.upper(), t) for _, t in pair_list}
        twins += len(pair_list) - len(canonical)
        for _, ticker in pair_list:
            tagged_total[scope.scope] += 1
            tagged_found[scope.scope] += resolver.by_ticker.get(ticker.upper(), ticker) in mentions
        guess = triage.gics_guess(mentions, by_ticker)
        if scope.scope in ("multi", "multi_dominated", "roundup") and guess:
            multi_with_gics += 1
            gics_hits[guess] += 1
        rows.append(
            {
                "body_hash": key,
                "article_ids": " ".join(str(a) for a, _ in pair_list),
                "tagged_tickers": " ".join(sorted({t for _, t in pair_list})),
                "rule_scope": scope.scope,
                "companies": " ".join(f"{t}:{n}" for t, n in mentions.most_common()),
                "gics_guess": " / ".join(guess) if guess else "",
                "title": title,
                "slug": slug,
                "text": body[:TEXT_CAP],
            }
        )

    bodies = len(members)
    print("\n== pairs per unique body")
    for name in ("1", "2", "3-5", "6-10", "11+"):
        print(f"  {name:>5}: {pct(pairs_per_body[name], bodies)}")
    print(f"  share-class twin pairs (same company, two tickers): {twins:,}")
    print("\n== subject companies per body (named twice or in the title; rules + NER ORG spans)")
    for name in ("0", "1", "2", "3-5", "6-10", "11+"):
        print(f"  {name:>5}: {pct(companies_hist[name], bodies)}")
    print("\n== rule-based scope candidate (all rule steps): bodies, and the pairs they hold")
    print(f"  {'scope':<16}{'bodies':>18}{'pairs':>20}{'tagged ticker named':>22}")
    for name, n in scope_bodies.most_common():
        found = tagged_found[name] / tagged_total[name] if tagged_total[name] else 0.0
        print(f"  {name:<16}{pct(n, bodies):>18}{pct(scope_pairs[name], pairs):>20}{found:>21.1%}")
    multi = scope_bodies["multi"] + scope_bodies["multi_dominated"] + scope_bodies["roundup"]
    print(
        f"\n== GICS guess: {multi_with_gics:,} of {multi:,} multi-company bodies share a sub-industry"
    )
    for (sector, sub), n in gics_hits.most_common(12):
        print(f"  {n:>6,}  {sector} / {sub}")

    args.out.mkdir(parents=True, exist_ok=True)
    random.Random(SEED).shuffle(rows)  # noqa: S311 -- sampling, not cryptography
    sample = rows[: args.sample]
    path = args.out / "triage_sample_bodies.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)
    print(f"\nwrote {len(sample)} random bodies to {path}; {time.time() - started:.0f} s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("mode", nargs="?", choices=("size", "gold"), default="size")
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    {"size": cmd_size, "gold": cmd_gold}[args.mode](args)


if __name__ == "__main__":
    main()
