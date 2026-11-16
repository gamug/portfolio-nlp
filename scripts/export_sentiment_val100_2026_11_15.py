# ruff: noqa: PLR2004, PLC0415, RUF005, S311  (one-off export and scoring script: thresholds are the owner's decisions, the draw is not cryptographic)
"""Export 100 clear, company-meaningful texts of the summary step to tag, with the Spanish title and text (owner, 2026-11-15: spend approved).

    uv run scripts/export_sentiment_val100_2026_11_15.py [--out data/sentiment_val100] [--seed 20261117]

Same pool as ``export_sentiment_val_2026_11_15.py`` (read-only on the results database; ``NLP_DB_IMMUTABLE=1`` reads the main file only), with
stricter filters so that the company is the subject of the text: relevance 0.9 or more, 300 to 1,500 characters, the company named at least twice and within the
first 100 characters, and no list of analyst calls (two or more upgrades, downgrades, reiterations or initiations, or "biggest calls").
Not in the 90-text set or in the stashed 200-text sample. A simple random draw (not stratified by the prediction). The prediction is in ``key.csv``.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_sentiment_val_2026_11_15 as base

COUNT = 100
CALLS = re.compile(r"\b(upgrad\w+|downgrad\w+|reiterat\w+|initiat\w+)\b", re.IGNORECASE)


def _clear(r: dict[str, str]) -> bool:
    text = r["text"]
    if float(r["relevance_score"]) < 0.9 or not 300 <= len(text) <= 1500:
        return False
    if len(CALLS.findall(text)) >= 2 or re.search(
        r"biggest calls|biggest moves", text, re.IGNORECASE
    ):
        return False
    names = base._names(r["company"]) + [r["ticker"]]
    lowered = text.lower()
    mentions = sum(lowered.count(n.lower()) for n in names if len(n) >= 3)
    return mentions >= 2 and any(n.lower() in lowered[:100] for n in names if len(n) >= 3)


def _articles(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {row["article_id"] for row in csv.DictReader(fh)}


def main() -> None:
    base.db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-db", default=base.DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=Path("data/sentiment_val100"))
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument("--no-translate", action="store_true")
    args = parser.parse_args()
    taken = _articles(Path("data/sentiment_val90/key.csv")) | _articles(
        Path("data/sentiment_val_stash_2026_11_15/val200/key.csv")
    )
    pool = [
        r
        for r in base.candidates(args.results_db, args.gold)
        if _clear(r) and r["article_id"] not in taken
    ]
    print(f"clear candidates: {len(pool)}")
    random.Random(args.seed).shuffle(pool)
    picked, seen = [], set()
    for r in pool:
        if r["article_id"] not in seen:
            seen.add(r["article_id"])
            picked.append(r)
        if len(picked) == COUNT:
            break
    if len(picked) < COUNT:
        sys.exit(f"drew {len(picked)}; not written")
    tag, key = [], []
    for n, r in enumerate(picked, 1):
        k = f"sn{r['item_id']}#{r['ticker']}"
        tag.append(
            {
                "review_order": str(n),
                "key": k,
                "article_id": k,
                "ticker": r["ticker"],
                "company": r["company"],
                "pub_date": r["pub_date"],
                "title": r["title"],
                "text": r["text"],
                "sentiment": "",
                "notes": "",
            }
        )
        key.append(
            {
                "review_order": str(n),
                "key": k,
                "item_id": r["item_id"],
                "article_id": r["article_id"],
                "ticker": r["ticker"],
                "text_source": r["text_source"],
                "predicted": r["predicted"],
                "predicted_score": r["predicted_score"],
                "relevance_score": r["relevance_score"],
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_translate:
        import retag_gold_2026_10_29 as retag_cli

        tag = retag_cli.add_spanish(
            args.out, tag, retag_cli._translator(), yes=True, max_new=len(tag)
        )
    for name, rows in (("to_tag.csv", tag), ("key.csv", key)):
        with (args.out / name).open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(tag)} rows -> {args.out}")


if __name__ == "__main__":
    main()
