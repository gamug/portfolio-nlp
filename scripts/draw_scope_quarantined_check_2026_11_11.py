"""Draw a sample of `quarantined` bodies for the owner to validate the scope step 1 gave them (T-147/T-163; owner 2026-11-11).

    uv run scripts/draw_scope_quarantined_check_2026_11_11.py --results-db <nlp.db copy> [--out data] \
        [--counts out_of_scope=100,single=50,multi=25,sector=25] [--seed 20261111] [--text-cap 2000] [--yes]

A second draw that adds to a file already read (owner 2026-11-11: ``multi`` had 25 bodies, interval 35 points wide) skips those bodies:

    uv run scripts/draw_scope_quarantined_check_2026_11_11.py --results-db <nlp.db copy> --out data/multi_extra \\
        --counts multi=75 --exclude data/scope_quarantined_to_validate.csv --seed 20261112 [--yes]

Why: the held-out gold that measured the scope model (83.6%) may not represent the `quarantined` bodies, which are 79% of the
corpus, and the model calls 59% of them `out_of_scope`. This draws, at random and per **predicted** scope, bodies of the pairs
D36 calls `quarantined`, none of them (or a twin) already tagged in a gold file. The owner reads the title and the text (the first
``--text-cap`` characters, with the Spanish reading aid) and writes ``validator`` = ``yes`` when the predicted scope is right and
``no`` when not (``correct_scope`` then says which). Per predicted class, the share of ``yes`` is the precision of that class on
quarantined bodies: for ``out_of_scope`` it is how many real company stories the step would stop; for the other three it is how many
non-company bodies would go on.

Reads the RESULTS file read-only: pass a **copy of ``nlp.db`` together with its ``-wal`` file** (rows still in the write-ahead log
are not in ``nlp.db`` alone). Writes ``<out>/scope_quarantined_to_validate.csv`` in the format of ``retag.SCOPE_CHECK_COLUMNS``
(``spanish_title`` / ``spanish_text`` right after ``text``); nothing is bought without ``--yes``; an existing file is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_scope_enrich_2026_11_10 as enrich
import draw_sector_extra_2026_11_05 as extra
import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import retag, triage

TO_VALIDATE_NAME = "scope_quarantined_to_validate.csv"
DEFAULT_COUNTS = "out_of_scope=100,single=50,multi=25,sector=25"
STATUS = "quarantined"


def parse_counts(text: str) -> dict[str, int]:
    counts = {k.strip(): int(v) for k, v in (part.split("=") for part in text.split(","))}
    unknown = set(counts) - {"single", "multi", "sector", "out_of_scope"}
    if unknown or any(n < 0 for n in counts.values()):
        raise ValueError(
            f"--counts wants single, multi, sector and out_of_scope with sizes >= 0, got {text!r}"
        )
    return counts


def draw(
    pool: dict[str, list[dict[str, str]]], counts: dict[str, int], seed: int
) -> list[dict[str, str]]:
    """``counts[scope]`` random bodies of every predicted scope, shuffled together so the order tells nothing."""
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    chosen: list[dict[str, str]] = []
    for scope in sorted(counts):
        ordered = sorted(pool.get(scope, []), key=lambda r: int(r["article_id"]))
        chosen += rng.sample(ordered, min(counts[scope], len(ordered)))
    rng.shuffle(chosen)
    return chosen


def article_ids_of(path: Path) -> set[int]:
    """The ``article_id`` column of an earlier draw (any BOM is handled), to keep its bodies out of this one."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {int(row["article_id"]) for row in csv.DictReader(handle)}


def check_rows(chosen: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "review_order": str(order),
            **{k: r[k] for k in retag.CARRIED[1:]},
            "predicted_scope": r["predicted_scope"],
            **dict.fromkeys(retag.SCOPE_CHECK_TAG_COLUMNS, ""),
        }
        for order, r in enumerate(chosen, 1)
    ]


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-db", required=True, help="a copy of nlp.db with its -wal file")
    parser.add_argument("--source-db", default="/workspaces/thesis/data/urls.db")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument("--revision", default="scope-nli-8ff21e50796b")
    parser.add_argument("--counts", default=DEFAULT_COUNTS)
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="an earlier draw (CSV with article_id): its bodies are not drawn again; repeat for several",
    )
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--text-cap", type=int, default=2000)
    parser.add_argument("--yes", action="store_true", help="pay for the Spanish translation")
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    args = parser.parse_args()
    if (args.out / TO_VALIDATE_NAME).exists():
        sys.exit(
            f"{args.out / TO_VALIDATE_NAME} exists: it is your validation file and is not overwritten"
        )
    counts = parse_counts(args.counts)
    results = sqlite3.connect(f"file:{args.results_db}?mode=ro", uri=True)
    scope_of = dict(
        results.execute(
            "SELECT body_hash, scope FROM body_scope WHERE classifier_revision = ?",
            (args.revision,),
        )
    )
    source = sqlite3.connect(f"file:{args.source_db}?mode=ro", uri=True)
    tagged = enrich.tagged_article_ids(args.gold)
    for earlier in args.exclude:
        tagged |= article_ids_of(earlier)
    skip = {
        triage.body_hash(body or "")
        for (body,) in source.execute(
            "SELECT body_text FROM articles WHERE id IN (SELECT value FROM json_each(?))",
            (str(sorted(tagged)),),
        )
    }
    pool: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    for article_id, ticker, company, pub_date, domain, title, body in source.execute(
        "SELECT id, ticker, company, pub_date, source_domain, title, body_text FROM articles "
        "WHERE fetch_status = ? AND body_text IS NOT NULL AND body_text != '' ORDER BY id",
        (STATUS,),
    ):
        key = triage.body_hash(body)
        if key in seen or key in skip or key not in scope_of:
            continue
        seen.add(key)
        cut = body[: args.text_cap]
        pool[scope_of[key]].append(
            {
                "article_id": str(article_id),
                "ticker": ticker or "",
                "company": company or "",
                "pub_date": pub_date or "",
                "source_domain": domain or "",
                "title": title or "",
                "text": cut + extra.STAGE_TEXT if len(body) > args.text_cap else cut,
                "predicted_scope": scope_of[key],
            }
        )
    print(
        {scope: len(rows) for scope, rows in sorted(pool.items())},
        "quarantined bodies to draw from",
        flush=True,
    )
    rows = check_rows(draw(pool, counts, args.seed))
    rows = retag_cli.add_spanish(
        args.out, rows, retag_cli._translator(), yes=args.yes, max_new=args.max_new
    )
    retag.require_tagging_format(rows, TO_VALIDATE_NAME, retag.SCOPE_CHECK_COLUMNS)
    extra._write(args.out / TO_VALIDATE_NAME, rows)
    print(f"{len(rows)} bodies -> {args.out / TO_VALIDATE_NAME}")


if __name__ == "__main__":
    main()
