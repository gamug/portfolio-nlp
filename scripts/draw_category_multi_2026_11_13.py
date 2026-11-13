"""Draw 1,000 (article, company) rows from the structured multi bodies, to tag for the category model (T-174; owner 2026-11-13).

    uv run scripts/draw_category_multi_2026_11_13.py [--rows 1000] [--per-body 3] [--seed 20261113] [--out data]

Pool: bodies the scope step called `multi` (body_scope, newest revision) written as header paragraphs (`Company — text`, split
by `split_headers_2026_11_13`, which drops the segments of companies outside the universe); each company's segment is one row, so the row's text is only what the body says about that company.
Not drawn: a body (or a twin, same body hash) already in any gold file. At most ``--per-body`` rows from one body, so no body
dominates. Random, not picked for rare labels (an honest read of the label mix). Reads SOURCE and RESULTS, read-only.
Writes ``<out>/category_multi_to_tag.csv`` (you fill `label`) and ``<out>/category_multi_ids.csv``; neither is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import split_headers_2026_11_13 as hdr

import news_nlp as db
from news_nlp import triage
from news_nlp.triage import Company, CompanyResolver

TAGGED = (
    "category_gold_v2.csv",
    "category_gold_v2_test.csv",
    "sentiment_gold.csv",
    "sentiment_train.csv",
    "sentiment_train_2.csv",
    "segment_check_gold.csv",
)
TEXT_CAP = 3000


def main() -> None:
    db.env.load_env_file()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--per-body", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    a = ap.parse_args()
    to_tag, ids_out = a.out / "category_multi_to_tag.csv", a.out / "category_multi_ids.csv"
    if to_tag.exists():
        sys.exit(f"{to_tag} exists: it is your tagging file and is not overwritten")
    conn = db.connect_pipeline()
    revs = [
        tuple(r)
        for r in conn.execute(
            "SELECT classifier_revision, COUNT(*) FROM main.body_scope GROUP BY 1"
        )
    ]
    print("scope revisions:", revs, flush=True)
    rev = max(revs, key=lambda r: r[1])[0]
    multi = {
        r[0]
        for r in conn.execute(
            "SELECT body_hash FROM main.body_scope WHERE classifier_revision=? AND scope='multi'",
            (rev,),
        )
    }
    print(f"{len(multi):,} multi bodies at {rev}", flush=True)
    tagged: set[int] = set()
    for name in TAGGED:
        tagged |= set(pd.read_csv(ROOT / "gold" / name, usecols=["article_id"]).article_id)
    skip = {
        triage.body_hash(b or "")
        for (b,) in conn.execute(
            "SELECT body_text FROM source.articles WHERE id IN (SELECT value FROM json_each(?))",
            (str(sorted(tagged)),),
        )
    }
    uni = pd.read_csv(ROOT / "gold/universe.csv").fillna("")
    res = CompanyResolver(
        [
            Company(t, c, s, "")
            for t, c, s in zip(uni.ticker, uni.company, uni.gics_sector, strict=False)
        ]
    )
    names = dict(zip(uni.ticker, uni.company, strict=False))
    seen: set[str] = set()
    bodies: list[dict] = []
    stat: Counter[str] = Counter()
    for aid, ticker, title, body, pub, dom in conn.execute(
        "SELECT id, ticker, title, body_text, pub_date, source_domain FROM source.articles "
        "WHERE fetch_status='ok' AND body_text IS NOT NULL AND body_text != '' ORDER BY id"
    ):
        h = triage.body_hash(body)
        if h not in multi or h in seen or h in skip:
            stat["skipped (not multi / twin / already tagged)"] += 1
            continue
        seen.add(h)
        got = hdr.split(body, res)
        if got is None:
            stat["not header paragraphs"] += 1
            continue
        parts, heads = got
        if len(parts) < 2:
            stat["fewer than 2 universe companies"] += 1
            continue
        stat["structured"] += 1
        bodies.append(
            {
                "article_id": aid,
                "title": title or "",
                "pub": pub or "",
                "dom": dom or "",
                "hash": h,
                "parts": parts,
                "heads": heads,
            }
        )
    print(dict(stat), flush=True)
    rng = random.Random(a.seed)
    rows = []
    for b in bodies:
        tk = sorted(b["parts"])
        rng.shuffle(tk)
        for t in tk[: a.per_body]:
            rows.append(
                {
                    "article_id": b["article_id"],
                    "ticker": t,
                    "company": names.get(t, ""),
                    "pub_date": b["pub"],
                    "source_domain": b["dom"],
                    "title": b["title"],
                    "text": b["parts"][t][:TEXT_CAP],
                    "body_companies": len(b["parts"]),
                    "body_headers": b["heads"],
                    "body_hash": b["hash"],
                }
            )
    print(
        f"{len(bodies):,} structured bodies, {len(rows):,} rows after the per-body cap", flush=True
    )
    rng.shuffle(rows)
    rows = sorted(rows[: a.rows], key=lambda r: (r["article_id"], r["ticker"]))
    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["review_order"] = i
    cols = [
        "review_order",
        "article_id",
        "ticker",
        "company",
        "pub_date",
        "source_domain",
        "title",
        "text",
        "label",
        "notes",
    ]
    with to_tag.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows({**r, "label": "", "notes": ""} for r in rows)
    with ids_out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh,
            ["review_order", "article_id", "ticker", "body_companies", "body_hash"],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)
    two = sum(
        sum(hdr.header_name(p) is not None for p in hdr.paragraphs(r["text"])) > 1 for r in rows
    )
    print(f"rows whose text holds more than one header paragraph: {two}")
    print(
        f"{len(rows)} rows from {len({r['article_id'] for r in rows})} bodies, {len({r['ticker'] for r in rows})} companies -> {to_tag}"
    )
    print("rows per body companies count:", dict(Counter(r["body_companies"] for r in rows)))


if __name__ == "__main__":
    main()

# ruff: noqa: E402, PLR0915, PLR2004, S311, B007  -- one-shot research script
