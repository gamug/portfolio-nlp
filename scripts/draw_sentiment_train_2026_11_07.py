"""Draw the summaries the owner tags to retrain sentiment on summaries (Work item 18, T-148c, D26).

    uv run scripts/draw_sentiment_train_2026_11_07.py [--out data] [--per-class 150] [--pool 6000]
        [--seed 20261107] [--yes]

The sentiment model was measured on the 180-text gold and missed the D32 bar (60.4% agreement
against 80%); the owner chose to fine-tune it on summaries (2026-11-07). The model has to learn the
owner's reading of a summary, so the owner tags a **training set**, apart from the gold and never
mixed with it.

The texts are the summaries of the summaries cut (``data/cut/summaries/``, ``--cut``): the bodies
the rules call single, multi_dominated, sector, market or economy. A random ``--pool`` of them
(seeded) is scored by the model of ``config/pipeline_models.json`` the way the pipeline's sentiment
stage reads a text, only to balance the draw: ``--per-class`` texts of each predicted class
(150: 450 in all), so the rare negative class is covered. The prediction is never written to the
tagging file, whose rows are shuffled. The ids file holds it and the size of the pool of each class.

**Never drawn:** a body that is (by body hash, so twins too) one of the 180 gold texts
(``gold/sentiment_gold.csv``) or one of the held-out bodies (``<out>/heldout_ids.csv``). A body
that is a twin of a text already drawn is not drawn twice.

Only summaries are here: the texts of the companies of multi bodies (T-148b) do not exist yet, and
the gold's 60 extractor texts are not represented in the training set.

Writes ``<out>/sentiment_train_ids.csv`` (drawn once, with the texts: a summary is generated, the
copy in the cut is the other) and ``<out>/sentiment_train_to_tag.csv``, a tagging file in the
sentiment format (``retag.SENTIMENT_TAGGING_COLUMNS``, the Spanish pair right after ``text``) that
is never overwritten. The Spanish aid is the approved paid translation of the same engine; nothing
is bought without ``--yes``. Its cache is ``<out>/sentiment_train/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cut_summaries_2026_11_07 as cut
import draw_sector_extra_2026_11_05 as extra
import draw_sentiment_gold_2026_11_05 as gold_draw
import retag_gold_2026_10_29 as retag_cli
import size_triage_2026_10_28 as size_triage

import news_nlp as db
from news_nlp import retag, triage

IDS_NAME = "sentiment_train_ids.csv"
TO_TAG_NAME = "sentiment_train_to_tag.csv"
CACHE_DIR = "sentiment_train"
GOLD = Path("gold/sentiment_gold.csv")
CUT_DIR = Path("data/cut/summaries")
MIN_TEXT_CHARS = gold_draw.MIN_TEXT_CHARS
MAX_TEXT_CHARS = gold_draw.MAX_COMPANY_CHARS  # what the owner can read in a cell
IDS_FIELDS = (
    "key",
    "article_id",
    "body_hash",
    "kind",
    "rule_scope",
    "revision",
    "predicted",
    "population",
    "text_hash",
    "title",
    "ticker",
    "company",
    "text",
)


def cut_rows(cut_dir: Path) -> list[dict[str, str]]:
    """The summarised rows of every chunk of the cut, in order. A missing chunk number stops the
    run: a draw from a cut still in progress would not be the one the owner tags later."""
    files = sorted(cut_dir.glob("chunk_*.csv"))
    numbers = [int(f.stem.rsplit("_", 1)[1]) for f in files]
    if not files or numbers != list(range(1, len(files) + 1)):
        sys.exit(f"{cut_dir}: the chunks are not 1..{len(files)} in a row: finish the cut first")
    rows: list[dict[str, str]] = []
    for path in files:
        with path.open(encoding="utf-8", newline="") as fh:
            rows += [
                r
                for r in csv.DictReader(fh)
                if r["status"] == cut.SUMMARISED and r["summary"].strip()
            ]
    return rows


def candidates(
    rows: list[dict[str, str]], never: set[str], pool: int, seed: int
) -> list[dict[str, str]]:
    """A seeded random ``pool`` of the summaries of a length the owner can read, minus the bodies
    in ``never`` (gold and held-out), one per body."""
    usable = sorted(
        (
            r
            for r in rows
            if r["body_hash"] not in never and MIN_TEXT_CHARS <= len(r["summary"]) <= MAX_TEXT_CHARS
        ),
        key=lambda r: r["body_hash"],
    )
    random.Random(seed).shuffle(usable)  # noqa: S311 -- a reproducible draw, not security
    return usable[:pool]


def draw(
    scored: list[dict[str, str]], per_class: int, seed: int
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """``per_class`` texts of each predicted class (all of a class that has fewer), shuffled so
    the order does not reveal the class. Also the size of each class in the scored pool."""
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    population = Counter(t["predicted"] for t in scored)
    chosen: list[dict[str, str]] = []
    for label in gold_draw.CLASSES:
        members = sorted(
            (t for t in scored if t["predicted"] == label), key=lambda t: t["body_hash"]
        )
        rng.shuffle(members)
        chosen += [{**t, "population": str(population[label])} for t in members[:per_class]]
    rng.shuffle(chosen)
    for t in chosen:
        t["text_hash"] = retag.text_hash(t["text"])
    return chosen, dict(population)


def with_subjects(conn: db.NewsNlpDatabase, chosen: list[dict[str, str]]) -> None:
    """Title, key, ticker and company of each drawn text, as the cut read the body: the company
    the rules found named most (single, multi_dominated), or a label for the other scopes."""
    universe = size_triage.load_universe(conn)
    names = {c.ticker: c.name for c in universe}
    resolver = triage.CompanyResolver(universe)
    terms = list(size_triage.sector_terms(universe))
    for t in chosen:
        ids = [int(i) for i in t["article_ids"].split()]
        title, body, slug, headlines = cut.fetch_body(conn, ids)
        headline, head_slug = size_triage.roundup_headline(headlines) or (title, slug)
        _, mentions = size_triage.classify(
            resolver,
            headline,
            body,
            head_slug,
            cut.first_spans(conn, ids),
            terms,
            **size_triage.FULL,
        )
        ticker = mentions.most_common(1)[0][0] if mentions else ""
        scope = t["rule_scope"]
        t["article_id"] = str(ids[0])
        t["key"] = f"{ids[0]}#summary"
        t["kind"] = "summary"
        t["title"] = title
        t["ticker"] = ticker if scope in ("single", "multi_dominated") else ""
        t["company"] = (
            names.get(ticker, ticker)
            if t["ticker"]
            else {
                "sector": "[sector]",
                "market": "[market / economy]",
                "economy": "[market / economy]",
            }.get(scope, "")
        )


def models() -> dict[str, dict[str, str]]:
    return json.loads(Path("config/pipeline_models.json").read_text(encoding="utf-8"))


def main() -> None:
    db.env.load_env_file()  # .env: these scripts do not import pipeline, which loads it elsewhere
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--cut", type=Path, default=CUT_DIR)
    parser.add_argument("--per-class", type=int, default=150)
    parser.add_argument("--pool", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--yes", action="store_true", help="pay for the Spanish translation")
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    args = parser.parse_args()
    out: Path = args.out
    ids_path, target = out / IDS_NAME, out / TO_TAG_NAME
    if extra.has_rows(target):
        sys.exit(f"{target} exists: it is your tagging file and is not overwritten")
    conn = db.connect_pipeline()
    if ids_path.exists():
        chosen = extra._read(ids_path)
        print(f"{ids_path} exists: the set is not drawn again")
    else:
        gold_ids = {int(r["article_id"]) for r in extra._read(GOLD)}
        held_ids = {int(r["article_id"]) for r in extra._read(out / "heldout_ids.csv")}
        never = extra.excluded_hashes(conn, gold_ids | held_ids)
        pool = candidates(cut_rows(args.cut), never, args.pool, args.seed)
        scored = gold_draw.predict([{**r, "text": r["summary"]} for r in pool])
        chosen, population = draw(scored, args.per_class, args.seed)
        with_subjects(conn, chosen)
        revision = f"{models()['c_summary']['model']}@{models()['c_summary']['revision']}"
        for t in chosen:
            t["revision"] = revision
        short = {k: n for k, n in population.items() if n < args.per_class}
        if short:
            print(f"classes with fewer texts than the draw takes (all drawn): {short}")
        extra._write(ids_path, [{f: t.get(f, "") for f in IDS_FIELDS} for t in chosen])
        print(f"{len(chosen)} drawn:", dict(Counter(t["predicted"] for t in chosen)))
        print("scored pool:", population, "excluded bodies:", len(never))
    cache = out / CACHE_DIR
    cache.mkdir(exist_ok=True)
    keyed = [{"article_id": t["key"], "title": t["title"], "text": t["text"]} for t in chosen]
    done = retag_cli.add_spanish(
        cache, keyed, retag_cli._translator(), yes=args.yes, max_new=args.max_new
    )
    spanish = {r["article_id"]: r for r in done}
    rows = gold_draw.tag_rows(chosen, spanish)
    try:
        retag.require_tagging_format(rows, target.name, retag.SENTIMENT_TAGGING_COLUMNS)
        extra._write(target, retag.spreadsheet_safe(rows))
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    except PermissionError:
        sys.exit(f"cannot write {target}: close it in Excel and run again")
    print(f"{len(rows)} texts -> {target}")


if __name__ == "__main__":
    main()
