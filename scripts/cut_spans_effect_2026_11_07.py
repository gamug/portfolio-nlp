"""How much does the rules' scope move when the stored NER spans are not there? (T-148a, PR #129)

    uv run scripts/cut_spans_effect_2026_11_07.py [--n 500] [--from 2026-03-01] [--seed 20261107]

The rules read the stored NER ``ORG`` spans; 14% of the backtest window's articles (the newest
months) have none, so the cut scopes them from the text alone, while the figures the cut quotes (the
rules' agreement on the held-out set) were measured with spans. This takes ``--n`` unique bodies of
the window that DO have spans (published ``--from`` or later, the months nearest to the ones without),
scopes each twice, with its spans and with none, and prints how often the scope changes, the
confusion by scope, and how often the *summary decision* (D25 step 3: summarise or not) changes.
Read-only; nothing is written.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cut_summaries_2026_11_07 as cut
import size_triage_2026_10_28 as size_triage

import news_nlp as db
from news_nlp import triage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--from", dest="since", default="2026-03-01")
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    conn = db.connect_pipeline()
    universe = size_triage.load_universe(conn)
    resolver = triage.CompanyResolver(universe)
    terms = list(size_triage.sector_terms(universe))
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM source.articles WHERE fetch_status = 'ok' AND body_text != '' "
            "AND pub_date >= ? ORDER BY id",
            (args.since,),
        )
    ]
    random.Random(args.seed).shuffle(ids)  # noqa: S311 -- a reproducible sample
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    checked = 0
    differs = 0
    for article_id in ids:
        if len(pairs) >= args.n:
            break
        spans = cut.first_spans(conn, [article_id])
        if not spans:
            continue
        checked += 1
        title, body, slug, headlines = cut.fetch_body(conn, [article_id])
        key = triage.body_hash(body)
        if key in seen:
            continue
        seen.add(key)
        headline, hslug = size_triage.roundup_headline(headlines) or (title, slug)
        with_spans = size_triage.classify(
            resolver, headline, body, hslug, spans, terms, **size_triage.FULL
        )[0].scope
        without = size_triage.classify(
            resolver, headline, body, hslug, [], terms, **size_triage.FULL
        )[0].scope
        pairs.append((with_spans, without))
        text = f"{headline}\n{triage.strip_boilerplate(body)}"
        differs += resolver.resolve(text, spans, skip_attribution=True) != resolver.resolve(
            text, [], skip_attribution=True
        )
    n = len(pairs)
    same = sum(a == b for a, b in pairs)
    gold_same = sum(triage.RULE_SCOPE_TO_GOLD[a] == triage.RULE_SCOPE_TO_GOLD[b] for a, b in pairs)
    decision_same = sum(
        (a in cut.SUMMARY_RULE_SCOPES) == (b in cut.SUMMARY_RULE_SCOPES) for a, b in pairs
    )
    print(f"{n} unique bodies with spans, published {args.since} or later (seed {args.seed})")
    print(f"same rule scope with and without spans: {same}/{n} = {same / n:.1%}")
    print(f"same gold scope (single/multi/sector/...): {gold_same}/{n} = {gold_same / n:.1%}")
    print(
        f"same summarise-or-not decision (D25 step 3): {decision_same}/{n} = {decision_same / n:.1%}"
    )
    print(f"bodies whose company mentions change at all without the spans: {differs}/{n}")
    print("with spans -> without spans, where they differ:")
    for (a, b), count in Counter(p for p in pairs if p[0] != p[1]).most_common():
        print(f"  {a:>16} -> {b:<16} {count}")
    print("scope with spans:", dict(Counter(a for a, _ in pairs).most_common()))
    print("scope without spans:", dict(Counter(b for _, b in pairs).most_common()))


if __name__ == "__main__":
    main()
