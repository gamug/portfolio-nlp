"""Draw the narrative multi bodies of the extractor's faithfulness check and write the file the
owner marks (Work item 18, T-148b2, D27, D32).

    uv run scripts/draw_faithfulness_2026_11_06.py [--out data] [--n 30] [--seed 20261106]
        [--extractors baseline,model] [--yes]
    uv run scripts/draw_faithfulness_2026_11_06.py --add-extractor model [--out data] [--yes]

The bodies are the ones the owner tagged ``multi`` (``gold/category_gold_v2.csv``) that the
one-segment-per-company parser cannot split (``company_texts.split_company_texts`` is ``None``),
minus the held-out set; ``--n`` of them are drawn once (``<out>/faithfulness_bodies.csv``, never
drawn again, so the baseline and the model are judged on the same bodies).

Each extractor (``attribution_baseline.baseline_company_texts`` or, once T-148b3 exists, the trained
model) writes the text each company named in the body gets. The owner reads each text with its
article and marks ``faithful`` yes or no (``retag.FAITHFULNESS_COLUMNS``, the Spanish pair right
after ``text``, and the Spanish version of the article). The file is **blind to the extractor**: the
rows are shuffled, and a text two extractors wrote identically is one row. Which extractor wrote
which row (by ``review_order``) is in ``<out>/faithfulness_ids.csv``, to compute the figures with ``report_wi18``.

``--extractors baseline`` alone writes the file for the baseline only (the decision on the baseline,
D34); nothing is written for ``model`` until it is trained. The Spanish aid is the approved paid
translation of the same engine; nothing is bought without ``--yes``. Its cache is
``<out>/faithfulness/``.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sector_extra_2026_11_05 as extra
import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import attribution_baseline, company_texts, gold_metrics, retag, triage
from news_nlp.triage import CompanyResolver

BODIES_NAME = "faithfulness_bodies.csv"
IDS_NAME = "faithfulness_ids.csv"
TO_TAG_NAME = "faithfulness_to_tag.csv"
CACHE_DIR = "faithfulness"
GOLD = Path("gold/category_gold_v2.csv")
BODY_FIELDS = ("article_id", "segments")
COVERAGE_NAME = gold_metrics.COVERAGE_NAME
COVERAGE_FIELDS = gold_metrics.COVERAGE_COLUMNS
IDS_FIELDS = (
    "review_order",
    "key",
    "article_id",
    "ticker",
    "text_hash",
    "extractors",
    "revisions",
    "pub_date",
)
Extractor = Callable[[str, CompanyResolver], dict[str, str]]
EXTRACTORS: dict[str, Extractor] = {"baseline": attribution_baseline.baseline_company_texts}
extractor_revision = retag.extractor_revision  # a model extractor will bring its own (T-148b3)


def draw_bodies(candidates: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    """*n* of the candidate bodies (``article_id`` and ``segments``), seeded, in article order."""
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    pool = sorted(candidates, key=lambda r: int(r["article_id"]))
    rng.shuffle(pool)
    return sorted(pool[:n], key=lambda r: int(r["article_id"]))


def merge_texts(
    per_extractor: dict[str, dict[str, str]],
) -> list[tuple[str, tuple[str, ...]]]:
    """``(text, extractors)`` for one company, one entry per distinct text, extractors sorted."""
    by_text: dict[str, list[str]] = {}
    for name in sorted(per_extractor):
        text = per_extractor[name].get("text", "").strip()
        if text:
            by_text.setdefault(text, []).append(name)
    return [(t, tuple(v)) for t, v in by_text.items()]


def candidate_bodies(out: Path, resolver: CompanyResolver) -> list[dict[str, str]]:
    conn = db.connect_pipeline()
    held = extra.excluded_hashes(
        conn, {int(r["article_id"]) for r in extra._read(out / "heldout_ids.csv")}
    )
    found: list[dict[str, str]] = []
    for row in extra._read(GOLD):
        if row["scope"] != "multi":
            continue
        src = conn.execute(
            "SELECT body_text FROM source.articles WHERE id = ?", (int(row["article_id"]),)
        ).fetchone()
        body = src["body_text"] or ""
        if triage.body_hash(body) in held:  # the body, not the article id: twins share a body
            continue
        if company_texts.split_company_texts(body, resolver) is None:
            found.append(
                {
                    "article_id": row["article_id"],
                    "segments": str(len(company_texts.segments(body))),
                }
            )
    return found


def collect_items(
    bodies: list[dict[str, str]], names: list[str], resolver: CompanyResolver
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """One item per distinct text an extractor wrote for a company named in a body, with the
    extractors that wrote it (``extractors``) and the article it was taken from; and one row per
    named company an extractor left without a text (``COVERAGE_FIELDS``): the report counts each as
    unfaithful, so an extractor cannot gain by leaving the hard companies out."""
    conn = db.connect_pipeline()
    items: list[dict[str, str]] = []
    omissions: list[dict[str, str]] = []
    for b in bodies:
        src = conn.execute(
            "SELECT title, body_text, pub_date, source_domain FROM source.articles WHERE id = ?",
            (int(b["article_id"]),),
        ).fetchone()
        body = src["body_text"] or ""
        found = {name: EXTRACTORS[name](body, resolver) for name in names}
        for ticker in sorted(resolver.resolve(body, skip_attribution=False)):
            per = {n: {"text": found[n].get(ticker, "")} for n in names}
            omissions += [
                {
                    "article_id": b["article_id"],
                    "ticker": ticker,
                    "extractor": n,
                    "pub_date": src["pub_date"] or "",
                }
                for n in names
                if not per[n]["text"].strip()
            ]
            for number, (text, extractors) in enumerate(merge_texts(per), 1):
                items.append(
                    {
                        "key": f"{b['article_id']}#{ticker}#{number}",
                        "article_id": b["article_id"],
                        "ticker": ticker,
                        "text_hash": retag.text_hash(text[: extra.TEXT_CAP]),
                        "extractors": "|".join(extractors),
                        "revisions": "|".join(extractor_revision(e) for e in extractors),
                        "title": src["title"] or "",
                        "text": text[: extra.TEXT_CAP],
                        "pub_date": src["pub_date"] or "",
                        "source_domain": src["source_domain"] or "",
                        "body": body[: extra.TEXT_CAP],
                    }
                )
    return items, omissions


def to_translate(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """The texts (key ``<id>#<ticker>#<n>``) and each article once (key ``<id>#body``)."""
    keyed = [{"article_id": i["key"], "title": i["title"], "text": i["text"]} for i in items]
    articles = {i["article_id"]: i for i in items}
    keyed += [
        {"article_id": f"{a}#body", "title": i["title"], "text": i["body"]}
        for a, i in articles.items()
    ]
    return keyed


def write_coverage(path: Path, omissions: list[dict[str, str]]) -> None:
    """The companies left without a text, also when there are none (the report needs the file)."""
    gold_metrics.write_pair_atomically({path: gold_metrics.render_csv(COVERAGE_FIELDS, omissions)})


def tag_rows(
    items: list[dict[str, str]], spanish: dict[str, dict[str, str]], names_of: dict[str, str]
) -> list[dict[str, str]]:
    """The owner's rows; ``review_order`` is the item's own when it has one (rows added later for
    another extractor), else its position."""
    return [
        {
            "review_order": i.get("review_order", str(order)),
            "article_id": i["article_id"],
            "ticker": i["ticker"],
            "company": names_of.get(i["ticker"], i["ticker"]),
            "pub_date": i["pub_date"],
            "source_domain": i["source_domain"],
            "title": i["title"],
            "text": i["text"],
            "spanish_title": spanish[i["key"]]["spanish_title"],
            "spanish_text": spanish[i["key"]]["spanish_text"],
            "body": i["body"],
            "spanish_body": spanish[f"{i['article_id']}#body"]["spanish_text"],
            "faithful": "",
            "notes": "",
        }
        for order, i in enumerate(items, 1)
    ]


def merge_new_extractor(
    ids: list[dict[str, str]], items: list[dict[str, str]], seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Add the texts of one more extractor to a file the owner has begun to mark, touching no mark.

    A text that an earlier row already holds (same article, company and ``text_hash``) only gets the
    extractor's name and revision added to that row of the ids. Any other text is a new row, with the
    next ``review_order`` and a key past the ones in use, in a shuffled order. Returns the updated
    ids and the new items (``review_order`` and ``key`` set); the earlier rows are not reordered."""
    by_text = {(r["article_id"], r["ticker"], r["text_hash"]): r for r in ids}
    new: list[dict[str, str]] = []
    for item in items:
        row = by_text.get((item["article_id"], item["ticker"], item["text_hash"]))
        if row is None:
            new.append(item)
            continue
        have = dict(zip(row["extractors"].split("|"), row["revisions"].split("|"), strict=True))
        have.update(zip(item["extractors"].split("|"), item["revisions"].split("|"), strict=True))
        row["extractors"] = "|".join(sorted(have))
        row["revisions"] = "|".join(have[n] for n in sorted(have))
    used: dict[tuple[str, str], int] = {}
    for r in ids:
        article, ticker, number = r["key"].rsplit("#", 2)
        used[(article, ticker)] = max(used.get((article, ticker), 0), int(number))
    random.Random(seed).shuffle(new)  # noqa: S311 -- blind within the new rows
    order = max((int(r["review_order"]) for r in ids), default=0)
    out: list[dict[str, str]] = []
    for item in new:
        order += 1
        used[(item["article_id"], item["ticker"])] = number = (
            used.get((item["article_id"], item["ticker"]), 0) + 1
        )
        out.append(
            {
                **item,
                "review_order": str(order),
                "key": f"{item['article_id']}#{item['ticker']}#{number}",
            }
        )
    return ids + [{k: i[k] for k in IDS_FIELDS} for i in out], out


def add_extractor(args: argparse.Namespace) -> None:
    """``--add-extractor NAME``: bring one more extractor's texts into the file being marked."""
    out: Path = args.out
    name: str = args.add_extractor
    if name not in EXTRACTORS:
        sys.exit(f"no extractor {name!r}: only {sorted(EXTRACTORS)} exist today (T-148b3).")
    target, ids_path, bodies_path = out / TO_TAG_NAME, out / IDS_NAME, out / BODIES_NAME
    if not (extra.has_rows(target) and ids_path.exists() and bodies_path.exists()):
        sys.exit(
            f"--add-extractor needs {target.name} (with rows), {IDS_NAME} and {BODIES_NAME} from a "
            "first draw; without them draw normally."
        )
    universe = retag_cli.companies()
    items, omissions = collect_items(extra._read(bodies_path), [name], CompanyResolver(universe))
    ids, new = merge_new_extractor(extra._read(ids_path), items, args.seed)
    print(
        f"{len(items)} texts of {name}: {len(items) - len(new)} already in the file, {len(new)} new"
    )
    kept = extra._read(target)
    backup = target.with_name(f"{target.stem}.before_{name}{target.suffix}")
    if backup.exists():
        sys.exit(
            f"{backup} exists: the add of {name} was interrupted. Compare it with {target.name}: if "
            f"{target.name} has the same marks, delete the copy and run again; if marks are "
            f"missing, copy it over {target.name} first."
        )
    done = retag_cli.add_spanish(
        out / CACHE_DIR,
        to_translate(new),
        retag_cli._translator(),
        yes=args.yes,
        max_new=args.max_new,
    )
    names_of = {c.ticker: c.name for c in universe}
    added = tag_rows(new, {r["article_id"]: r for r in done}, names_of)
    coverage_path = out / COVERAGE_NAME
    try:
        old_coverage = gold_metrics.read_coverage(coverage_path)
    except gold_metrics.MarkedFileError as exc:
        sys.exit(str(exc))
    old_ids = extra._read(ids_path)
    known = {tuple(r[k] for k in COVERAGE_FIELDS) for r in old_coverage}
    coverage = old_coverage + [
        o for o in omissions if tuple(o[k] for k in COVERAGE_FIELDS) not in known
    ]
    try:
        retag.require_tagging_format(added, target.name, retag.FAITHFULNESS_COLUMNS)
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    try:
        shutil.copy2(target, backup)  # the owner's marks, as they were
        # The tagging file goes first: if it cannot be written (open in Excel) nothing else changed.
        extra._write(target, retag.spreadsheet_safe(kept + added))  # idempotent on earlier rows
    except OSError as exc:
        backup.unlink(
            missing_ok=True
        )  # the tagging file was not touched: the next run starts clean
        sys.exit(f"cannot write {target.name} ({exc}); close it in Excel and run again")
    try:
        extra._write(ids_path, ids)
        write_coverage(coverage_path, coverage)
    except OSError as exc:
        # The tagging file already has the new rows; put the three back as they were. If that fails
        # too, the copy stays and the message says what state each file is in.
        try:
            shutil.copy2(backup, target)
            extra._write(ids_path, old_ids)
            write_coverage(coverage_path, old_coverage)
        except OSError as again:
            sys.exit(
                f"cannot write {ids_path.name} or {COVERAGE_NAME} ({exc}) and could not undo "
                f"({again}). {target.name} may already hold the new rows; {ids_path.name} and "
                f"{COVERAGE_NAME} may be old or half-written. The marks as they were are in "
                f"{backup.name}: do not delete it. Fix the cause and run again."
            )
        backup.unlink(missing_ok=True)
        sys.exit(f"cannot write {ids_path.name} or {COVERAGE_NAME} ({exc}); nothing was added")
    print(f"{len(added)} new texts appended to {target} (marks untouched; copy in {backup.name})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--extractors", default="baseline,model")
    parser.add_argument(
        "--add-extractor",
        metavar="NAME",
        help="add this extractor's texts to the file already being marked (no mark is touched)",
    )
    parser.add_argument("--yes", action="store_true", help="pay for the Spanish translation")
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.add_extractor:
        add_extractor(args)
        return
    out: Path = args.out
    names = [e.strip() for e in args.extractors.split(",") if e.strip()]
    missing = [e for e in names if e not in EXTRACTORS]
    if missing:
        sys.exit(
            f"no extractor {missing}: only {sorted(EXTRACTORS)} exist today. The trained model "
            "is T-148b3; use --extractors baseline for the decision on the baseline (D34)."
        )
    target = out / TO_TAG_NAME
    if extra.has_rows(target):
        sys.exit(f"{target} exists: it is your tagging file and is not overwritten")
    universe = retag_cli.companies()
    resolver = CompanyResolver(universe)
    bodies_path = out / BODIES_NAME
    if bodies_path.exists():
        bodies = extra._read(bodies_path)
        print(f"{bodies_path} exists: the bodies are not drawn again")
    else:
        candidates = candidate_bodies(out, resolver)
        bodies = draw_bodies(candidates, args.n, args.seed)
        extra._write(bodies_path, bodies)
        one = sum(b["segments"] == "1" for b in bodies)
        print(
            f"{len(candidates)} narrative multi bodies; {len(bodies)} drawn ({one} one paragraph)"
        )
    items, omissions = collect_items(bodies, names, resolver)
    unextracted = Counter(o["extractor"] for o in omissions)
    random.Random(args.seed).shuffle(items)  # noqa: S311 -- blind: the order hides the extractor
    print("texts for the owner:", len(items), "| left without a text:", dict(unextracted))
    cache = out / CACHE_DIR
    cache.mkdir(exist_ok=True)
    done = retag_cli.add_spanish(
        cache, to_translate(items), retag_cli._translator(), yes=args.yes, max_new=args.max_new
    )
    rows = tag_rows(items, {r["article_id"]: r for r in done}, {c.ticker: c.name for c in universe})
    try:
        retag.require_tagging_format(rows, target.name, retag.FAITHFULNESS_COLUMNS)
        ids = [
            {"review_order": str(n), **{k: i[k] for k in IDS_FIELDS[1:]}}
            for n, i in enumerate(items, 1)
        ]
        extra._write(
            out / IDS_NAME, ids
        )  # no news text: the owner's file is matched on review_order
        write_coverage(out / COVERAGE_NAME, omissions)
        extra._write(target, retag.spreadsheet_safe(rows))
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    except PermissionError:
        sys.exit(f"cannot write {target}: close it in Excel and run again")
    print(f"{len(rows)} texts -> {target}")


if __name__ == "__main__":
    main()
