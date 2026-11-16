# ruff: noqa: PLR2004, PLC0415, S311  (one-off export and scoring script: thresholds are the owner's decisions, the draw is not cryptographic)
"""Export 90 company-meaningful texts of the summary step to tag, to validate the sentiment model (owner, 2026-11-15: spend approved).

    uv run scripts/export_sentiment_val_2026_11_15.py [--results-db PATH] [--out data/sentiment_val90] [--seed 20261115] [--no-translate]

What it draws: the sub-news the summary step stored (``news_summary``, ``item_kind = sub_news``) for which the company step named a
main company (``news_company.is_main = 1``) and the sentiment step scored that pair. The text to tag is the text the sentiment step
reads (the summary, or the original text when it was short). The results database (``--results-db``, else ``$DATABASE_URL``) is opened read-only; nothing is
written to it. If another run holds it open in WAL mode and a read-only connection fails with a disk I/O error, ``NLP_DB_IMMUTABLE=1`` reads the main file
only and **ignores the WAL**: valid only for tables that no run is writing (these three were closed when the draws were made).

Texts that mean little are dropped: under 250 characters, fewer than two sentences, mostly not letters, lists of tickers, disclosure
lines, a main company that the text does not name, a relevance score under 0.7, a non-English article, and every article that is already in
``gold/sentiment_gold.csv`` or ``gold/sentiment_train*.csv``. One text per article. 30 texts per class of the sentiment model
(positive, negative, neutral), 10 of each from a stored summary (``text_source = summary``) and the rest from original text, so the set
can be scored one-vs-rest. The prediction is kept in a separate key file, not in the file to tag.

Files in ``--out``: ``to_tag.csv`` (what the owner reads and fills: ``sentiment`` of the company in the text, ``notes``) and ``key.csv``
(predictions and ids). The Spanish title and text come from the project translator (``news_nlp.translate``, cache inside ``--out``).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import news_nlp as db

COMPANY_REV = "companies-f349536cf701"
SENTIMENT_REV = "sentiment-5999fe131d8f"
SUMMARY_REV = "summary-7a9bd2062bc4"
PER_CLASS = 30
SUMMARY_PER_CLASS = 10
MIN_CHARS = 250
MIN_RELEVANCE = 0.7
EXCLUDE_FILES = ("sentiment_gold.csv", "sentiment_train.csv", "sentiment_train_2.csv")
DISCLOSURE = re.compile(
    r"disclosure|disclaimer|i own|i have no position|i am not receiving|positions? in|seeking alpha|"
    r"not investment advice|all rights reserved|click here|subscribe|sign up|follow us|"
    r"free report|zacks rank|\bto read more\b",
    re.IGNORECASE,
)
SUFFIX = re.compile(
    r"\b(inc|corp|corporation|company|co|ltd|plc|holdings|group|incorporated|the|class [a-c]|\(.*?\))\b\.?",
    re.IGNORECASE,
)


def _excluded_articles(gold: Path) -> set[str]:
    seen: set[str] = set()
    for name in EXCLUDE_FILES:
        path = gold / name
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as fh:
                seen.update(row["article_id"] for row in csv.DictReader(fh))
    return seen


def _names(company: str) -> list[str]:
    cleaned = " ".join(SUFFIX.sub(" ", company).replace(",", " ").split())
    names = [n for n in (cleaned, " ".join(cleaned.split()[:2])) if len(n) >= 3]
    return list(dict.fromkeys(names))


def _meaningful(text: str, company: str, ticker: str) -> bool:
    if len(text) < MIN_CHARS or DISCLOSURE.search(text):
        return False
    if len(re.findall(r"[.!?](?:\s|$)", text)) < 2:
        return False
    if sum(ch.isalpha() or ch.isspace() for ch in text) / len(text) < 0.85:
        return False
    if len(re.findall(r"\b[A-Z]{2,5}\b(?:,|\s+and\b)", text)) >= 4:
        return False
    lowered = text.lower()
    return any(n.lower() in lowered for n in _names(company)) or bool(
        re.search(rf"\b{re.escape(ticker)}\b", text)
    )


def candidates(results_db: str, gold: Path) -> list[dict[str, str]]:
    # NLP_DB_IMMUTABLE=1 reads the main file only (ignores the WAL): for a database another run holds open in WAL mode,
    # when a read-only connection cannot open its shared-memory file. Only for tables that no run is writing.
    suffix = "&immutable=1" if os.environ.get("NLP_DB_IMMUTABLE") else ""
    conn = sqlite3.connect(f"file:{results_db}?mode=ro{suffix}", uri=True)
    main = {
        (item, ticker): (score, mentions)
        for item, ticker, score, mentions in conn.execute(
            "SELECT item_id, ticker, relevance_score, mention_count FROM news_company "
            "WHERE item_kind='sub_news' AND model_revision=? AND is_main=1 AND relevance_score>=?",
            (COMPANY_REV, MIN_RELEVANCE),
        )
    }
    sentiment = {
        (item, ticker): (label, score)
        for item, ticker, label, score in conn.execute(
            "SELECT item_id, ticker, label, score FROM news_sentiment "
            "WHERE item_kind='sub_news' AND model_revision=?",
            (SENTIMENT_REV,),
        )
    }
    by_item: dict[str, list[str]] = {}
    for item, ticker in main:
        if (item, ticker) in sentiment:
            by_item.setdefault(item, []).append(ticker)
    texts = {
        item: (text, source)
        for item, text, source in conn.execute(
            "SELECT item_id, summary_text, text_source FROM news_summary "
            "WHERE item_kind='sub_news' AND model_revision=?",
            (SUMMARY_REV,),
        )
        if item in by_item and len(text) >= MIN_CHARS
    }
    articles = {
        str(sid): (str(aid), seg)
        for sid, aid, seg in conn.execute("SELECT id, article_id, segment_index FROM sub_news")
        if str(sid) in texts
    }
    meta: dict[str, tuple[str, str, str, str]] = {}
    company_of: dict[str, str] = {}
    for aid, title, ticker, company, pub, lang in conn.execute(
        "SELECT id, title, ticker, company, pub_date, language FROM articles"
    ):
        meta[str(aid)] = (title, company, pub, lang)
        if ticker and company:
            company_of.setdefault(ticker, company)
    banned = _excluded_articles(gold)
    rows: list[dict[str, str]] = []
    for item, (text, source) in texts.items():
        article_id, segment = articles.get(item, ("", ""))
        if not article_id or article_id in banned or article_id not in meta:
            continue
        title, _article_company, pub, lang = meta[article_id]
        if lang not in (None, "", "en"):
            continue
        ticker = max(by_item[item], key=lambda t: main[(item, t)][0])
        company = company_of.get(ticker, ticker)
        if not _meaningful(text, company, ticker):
            continue
        label, score = sentiment[(item, ticker)]
        rows.append(
            {
                "item_id": item,
                "article_id": article_id,
                "segment_index": str(segment),
                "ticker": ticker,
                "company": company,
                "pub_date": pub or "",
                "title": title or "",
                "text": text,
                "text_source": source,
                "predicted": label,
                "predicted_score": f"{score:.3f}",
                "relevance_score": f"{main[(item, ticker)][0]:.3f}",
            }
        )
    conn.close()
    return rows


def results_db(argument: str | None) -> str:
    """The RESULTS database to read: ``--results-db``, else ``$DATABASE_URL`` (``CLAUDE.md``, "Database"); never a guessed path."""
    path = argument or os.environ.get("DATABASE_URL")
    if not path:
        sys.exit(
            "set --results-db or DATABASE_URL: the RESULTS database holding news_summary, news_company and news_sentiment"
        )
    return path


def draw(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    seen: set[str] = set()
    picked: list[dict[str, str]] = []
    for label in ("positive", "negative", "neutral"):
        for source, want in (
            ("summary", SUMMARY_PER_CLASS),
            ("original", PER_CLASS - SUMMARY_PER_CLASS),
        ):
            got = 0
            for r in rows:
                if got == want:
                    break
                if (
                    r["predicted"] == label
                    and r["text_source"] == source
                    and r["article_id"] not in seen
                ):
                    picked.append(r)
                    seen.add(r["article_id"])
                    got += 1
    rng.shuffle(picked)
    return picked


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-db", default=None, help="default: $DATABASE_URL")
    parser.add_argument("--out", type=Path, default=Path("data/sentiment_val90"))
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument("--no-translate", action="store_true")
    args = parser.parse_args()

    pool = candidates(results_db(args.results_db), args.gold)
    print(f"candidates after the filters: {len(pool)}")
    for label in ("positive", "negative", "neutral"):
        sub = [r for r in pool if r["predicted"] == label]
        print(
            f"  {label}: {len(sub)} ({sum(r['text_source'] == 'summary' for r in sub)} from a summary)"
        )
    picked = draw(pool, args.seed)
    if len(picked) != 3 * PER_CLASS:
        sys.exit(f"drew {len(picked)} of {3 * PER_CLASS}; not written")
    args.out.mkdir(parents=True, exist_ok=True)

    tag_rows = []
    key_rows = []
    for n, r in enumerate(picked, 1):
        key = f"sn{r['item_id']}#{r['ticker']}"
        tag_rows.append(
            {
                "review_order": str(n),
                "key": key,
                "article_id": key,  # the translation cache is keyed by this id
                "ticker": r["ticker"],
                "company": r["company"],
                "pub_date": r["pub_date"],
                "title": r["title"],
                "text": r["text"],
                "sentiment": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "review_order": str(n),
                "key": key,
                "item_id": r["item_id"],
                "article_id": r["article_id"],
                "segment_index": r["segment_index"],
                "ticker": r["ticker"],
                "text_source": r["text_source"],
                "predicted": r["predicted"],
                "predicted_score": r["predicted_score"],
                "relevance_score": r["relevance_score"],
                "company_revision": COMPANY_REV,
                "sentiment_revision": SENTIMENT_REV,
                "summary_revision": SUMMARY_REV,
            }
        )
    if not args.no_translate:
        import retag_gold_2026_10_29 as retag_cli

        tag_rows = retag_cli.add_spanish(
            args.out, tag_rows, retag_cli._translator(), yes=True, max_new=len(tag_rows)
        )
    for name, rows in (("to_tag.csv", tag_rows), ("key.csv", key_rows)):
        with (args.out / name).open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(tag_rows)} rows -> {args.out}/to_tag.csv and key.csv")


if __name__ == "__main__":
    main()
