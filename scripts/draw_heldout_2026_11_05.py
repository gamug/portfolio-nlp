"""Draw the held-out test set of the triage gold (Work item 18, T-146e): bodies nobody has tagged.

    uv run scripts/draw_heldout_2026_11_05.py [--out data] [--n 200] [--seed 20261105] [--name heldout] [--yes]

The 720 judged articles, the 240-body gate drawn from them and the 44 sector bodies are all used to
write and tune the labelling prompt, so none of them can measure it. This draws ``--n`` bodies
from the 1,000-body random sample of the stored corpus (``triage_sample_bodies.csv``, written by
``size_triage_2026_10_28.py``), leaving out every body that shares an article with the 720 or the 44
(and with an earlier held-out draw). The rule scopes that are rare in the corpus (no company, sector,
market, economy) are all taken, because a random 200 would hold about eight of them; the rest is a
seeded random draw over the other rule scopes. ``stratum`` in the ids file is the rule scope, so a
rate over the whole corpus can be re-weighted by the stratum sizes (``population`` column).

``<out>/<name>_strata.csv`` holds, per rule scope, the size in the whole 1,000-body sample
(``frame``), the bodies left out because they share an article with the 720 or the 44 (``excluded``),
the bodies left to draw from (``free``) and the bodies drawn. The weight of a stratum when a rate is
re-weighted to the corpus is its ``frame`` size, not ``free``: the bodies left out were not a random
part of the sample. It is rewritten on every run (it is derived, not tagging work).

Writes ``<out>/<name>_ids.csv`` and ``<out>/<name>_to_tag.csv`` (drawn once, never overwritten): a
tagging file in the review file's columns, ``spanish_title`` / ``spanish_text`` after ``text``,
the v1 columns empty. The Spanish aid is the approved paid translation of the same engine as the
other files (about 1 USD per 720 articles); nothing is bought without ``--yes``. A later draw with
another ``--name`` leaves out this one's bodies.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sector_extra_2026_11_05 as extra
import retag_gold_2026_10_29 as retag_cli

import news_nlp as db
from news_nlp import retag

SAMPLE_NAME = "triage_sample_bodies.csv"
RARE = ("no_company", "sector", "market", "economy")  # all taken: too few to leave to chance


def tagged_ids(out: Path, name: str) -> set[str]:
    """Articles of every file already tagged or drawn for tagging: the review file, the sector
    extras and any earlier held-out draw (not ``name`` itself, whose set is fixed)."""
    ids = {r["article_id"] for r in extra._read(out / retag.REVIEW_NAME)}
    for path in (out / extra.TO_TAG_NAME, *out.glob("*_ids.csv")):
        if path.name in (f"{name}_ids.csv", "gate240_ids.csv") or not path.exists():
            continue
        ids |= {r["article_id"] for r in extra._read(path) if r.get("article_id")}
    return ids


def _ids(cell: str) -> list[str]:
    """The article ids of a sample row (the sample file separates them with spaces)."""
    return cell.replace(",", " ").split()


def draw(frame: list[dict[str, str]], n: int, seed: int, taken: set[str]) -> list[dict[str, str]]:
    free = [r for r in frame if not set(_ids(r["article_ids"])) & taken]
    population = Counter(r["rule_scope"] for r in free)
    rare = [r for r in free if r["rule_scope"] in RARE]
    rest = sorted((r for r in free if r["rule_scope"] not in RARE), key=lambda r: r["body_hash"])
    random.Random(seed).shuffle(rest)  # noqa: S311 -- a reproducible draw, not security
    picked = rare[:n] + rest[: max(0, n - len(rare))]
    return [
        {
            "article_id": _ids(r["article_ids"])[0],
            "body_hash": r["body_hash"],
            "stratum": r["rule_scope"],
            "population": str(population[r["rule_scope"]]),
        }
        for r in sorted(picked, key=lambda r: r["body_hash"])
    ]


def strata_rows(
    frame: list[dict[str, str]], taken: set[str], chosen: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Per rule scope: the size in the sample, the bodies left out, the bodies left and the drawn."""
    size = Counter(r["rule_scope"] for r in frame)
    free = Counter(r["rule_scope"] for r in frame if not set(_ids(r["article_ids"])) & taken)
    drawn = Counter(r["stratum"] for r in chosen)
    return [
        {
            "stratum": s,
            "frame": str(size[s]),
            "excluded": str(size[s] - free[s]),
            "free": str(free[s]),
            "drawn": str(drawn[s]),
        }
        for s in sorted(size)
    ]


def source_rows(chosen: list[dict[str, str]]) -> list[dict[str, object]]:
    conn = db.connect_pipeline()
    out: list[dict[str, object]] = []
    for r in chosen:
        row = conn.execute(
            "SELECT ticker, company, pub_date, source_domain, title, body_text "
            "FROM source.articles WHERE id = ?",
            (int(r["article_id"]),),
        ).fetchone()
        body = row["body_text"] or ""
        cut = body[: extra.TEXT_CAP]
        out.append(
            {
                "article_id": r["article_id"],
                "ticker": row["ticker"],
                "company": row["company"] or "",
                "pub_date": row["pub_date"] or "",
                "source_domain": row["source_domain"] or "",
                "title": row["title"] or "",
                "text": cut + extra.STAGE_TEXT if len(body) > extra.TEXT_CAP else cut,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--name", default="heldout")
    parser.add_argument("--yes", action="store_true", help="pay for the Spanish translation")
    parser.add_argument("--max-new", type=int, default=retag_cli.DEFAULT_MAX_NEW)
    args = parser.parse_args()
    out: Path = args.out
    ids_path, target = out / f"{args.name}_ids.csv", out / f"{args.name}_to_tag.csv"
    if target.exists():
        sys.exit(f"{target} exists: it is your tagging file and is not overwritten")
    frame = extra._read(out / SAMPLE_NAME)
    taken = tagged_ids(out, args.name)
    if ids_path.exists():
        chosen = extra._read(ids_path)
        print(f"{ids_path} exists: the set is not drawn again (--n and --seed are ignored)")
    else:
        chosen = draw(frame, args.n, args.seed, taken)
        extra._write(ids_path, chosen)
        print(f"{len(chosen)} drawn:", dict(Counter(r["stratum"] for r in chosen)))
    extra._write(out / f"{args.name}_strata.csv", strata_rows(frame, taken, chosen))
    rows = retag_cli.add_spanish(
        out,
        extra.tag_rows(source_rows(chosen)),
        retag_cli._translator(),
        yes=args.yes,
        max_new=args.max_new,
    )
    try:
        retag.require_tagging_format(rows, target.name)
        extra._write(target, rows)
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    except PermissionError:
        sys.exit(f"cannot write {target}: close it in Excel and run again")
    print(f"{len(rows)} bodies -> {target}")


if __name__ == "__main__":
    main()
