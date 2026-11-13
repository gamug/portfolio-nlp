"""Draw the clean test of the sector step: bodies the step (as built) gives a sector, or sends out_of_scope, that nobody has seen (2026-11-13).

    uv run scripts/draw_sector_fresh_2026_11_13.py [--out data] [--assigned 40] [--out-of-scope 12] [--controls 5]

The first yes/no review (2026-11-12) was drawn and balanced with the small model alone, so a change of the method could only look
worse there. This draw is made with the final configuration (``src/sector_stage.py``: the score, the companies named, the gate):

* ``assigned`` rows: the step gave the body a sector; the owner answers whether that sector is right (the precision that counts).
* ``out_of_scope`` rows: the gate sent the body out; the owner answers whether that was right (``yes``: it is not a clean sector
  body, or its sector is unclear) or says the sector it should have had (the cost of the gate).
* ``controls``: assigned rows shown with the sector the model likes least (``yes`` to them is a reviewer that agrees too soon; the
  marks are only in the key file).

None of the bodies is in the tagged gold or in the first review. ``data/sector_fresh_to_review.csv`` is what the owner reads (no
margin, no score, no hint of which rows are controls); ``data/sector_fresh_key.csv`` has the rest. Neither is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_scope_enrich_2026_11_10 as enrich
import draw_sector_extra_2026_11_05 as extra
import draw_sector_yesno_2026_11_12 as first

import news_nlp as db
from cut_stage import companies_of
from sector_stage import SectorClassifier, SectorResult

TO_REVIEW = "sector_fresh_to_review.csv"
KEY = "sector_fresh_key.csv"
SECTORS = first.SECTORS
Cand = dict[str, object]
BATCH = 16


def previously_seen(out: Path) -> set[int]:
    """The article ids of the first review: they are not drawn again."""
    path = out / "sector_yesno_to_review.csv"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {int(r["article_id"]) for r in csv.DictReader(fh)}


def verdict_of(result: SectorResult, shown: str | None = None) -> tuple[str, str, str]:
    """``(model_sector_code, model_verdict, sector_description)`` as the owner reads them."""
    if shown is None and result.status == "out_of_scope":
        guess = SECTORS.get(result.candidate)
        tail = f"best guess: {guess[0]}" if guess else f"looks like: {result.candidate}"
        return "", f"OUT OF SCOPE ({tail})", guess[1] if guess else ""
    code = shown or result.sector
    name, blurb = SECTORS[code]
    return code, f"SECTOR: {name}", blurb


def pick_assigned(by_sector: dict[str, list[Cand]], count: int) -> list[Cand]:
    """Round-robin over the sectors the step assigned, so no sector is left with one row."""
    chosen: list[Cand] = []
    while len(chosen) < count and any(by_sector.values()):
        for rows in by_sector.values():
            if rows and len(chosen) < count:
                chosen.append(rows.pop(0))
    return chosen


def build_rows(
    chosen: list[Cand], controls: set[int]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    review, key = [], []
    for n, c in enumerate(chosen, 1):
        result: SectorResult = c["result"]  # type: ignore[assignment]
        aid = int(c["article_id"])  # type: ignore[call-overload]
        is_control = aid in controls
        shown = str(c["worst"]) if is_control else None
        code, verdict, blurb = verdict_of(result, shown)
        review.append(
            {
                "review_order": str(n), "article_id": str(aid), "ticker": str(c["ticker"]),
                "company": str(c["company"]), "pub_date": str(c["pub_date"]), "source_domain": str(c["source_domain"]),
                "title": str(c["title"]), "text": str(c["text"]),
                "model_sector_code": code, "model_verdict": verdict, "sector_description": blurb,
                "correct": "", "right_sector": "", "notes": "",
            }
        )  # fmt: skip
        key.append(
            {
                "review_order": str(n), "article_id": str(aid), "status": result.status,
                "sector": result.sector, "candidate": result.candidate, "shown": code,
                "control": "yes" if is_control else "no", "margin": f"{result.margin:.4f}",
                "vote_sector": result.vote_sector,
            }
        )  # fmt: skip
    return review, key


def score_pool(
    classifier: SectorClassifier, pool: list[first.Row], need: int, need_out: int
) -> tuple[dict[str, list[Cand]], list[Cand]]:
    """Score the pool in batches until there are ``need`` assigned and ``need_out`` dropped bodies (enough to choose from)."""
    assigned: dict[str, list[Cand]] = {s: [] for s in SECTORS}
    dropped: list[Cand] = []
    for start in range(0, len(pool), BATCH):
        batch = pool[start : start + BATCH]
        cuts = [row[6][: extra.TEXT_CAP] for row in batch]
        results = classifier.classify([(row[5], cut) for row, cut in zip(batch, cuts, strict=True)])
        for (aid, ticker, company, pub_date, domain, title, body, _key), cut, result in zip(
            batch, cuts, results, strict=True
        ):
            order = sorted(result.scores.items(), key=lambda kv: kv[1])
            worst = next(k for k, _ in order if k in SECTORS)
            cand: Cand = {
                "article_id": aid, "ticker": ticker, "company": company, "pub_date": pub_date, "source_domain": domain,
                "title": title, "text": cut + extra.STAGE_TEXT if len(body) > extra.TEXT_CAP else cut,
                "result": result, "worst": worst,
            }  # fmt: skip
            (assigned[result.sector] if result.status == "assigned" else dropped).append(cand)
        kept = sum(map(len, assigned.values()))
        if (start // BATCH) % 10 == 0:
            print(f"  scored {start + len(batch)}: assigned {kept}, out {len(dropped)}", flush=True)
        if kept >= need and len(dropped) >= need_out:
            break
    return assigned, dropped


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument("--assigned", type=int, default=40)
    parser.add_argument("--out-of-scope", type=int, default=12)
    parser.add_argument("--controls", type=int, default=5)
    parser.add_argument("--sample", type=int, default=10, help="take 1 in this many articles")
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    out: Path = args.out
    for name in (TO_REVIEW, KEY):
        if (out / name).exists():
            sys.exit(f"{out / name} exists and is not overwritten")
    conn = db.connect_pipeline()
    tagged = enrich.tagged_article_ids(args.gold)
    for name in ("scope_enrich.csv", "scope_hook.csv"):
        tagged |= {int(r["article_id"]) for r in extra._read(args.gold / name)}
    tagged |= previously_seen(out)
    skip = extra.excluded_hashes(conn, tagged)
    sector_hashes = {
        str(r[0]) for r in conn.execute("SELECT body_hash FROM body_scope WHERE scope = 'sector'")
    }
    pool = first.read_pool(conn, args, skip, sector_hashes)
    print(f"{len(pool)} sector bodies in the pool", flush=True)
    classifier = SectorClassifier(companies_of(conn))
    print(f"revision {classifier.revision}", flush=True)
    rng = random.Random(args.seed)  # noqa: S311 -- a reproducible draw, not security
    rng.shuffle(pool)
    assigned, dropped = score_pool(
        classifier, pool, (args.assigned + args.controls) * 3, args.out_of_scope * 3
    )
    chosen_assigned = pick_assigned(assigned, args.assigned + args.controls)
    controls = {
        int(c["article_id"])  # type: ignore[call-overload]
        for c in rng.sample(chosen_assigned, args.controls)
    }
    chosen = chosen_assigned + rng.sample(dropped, min(args.out_of_scope, len(dropped)))
    rng.shuffle(chosen)
    review, key = build_rows(chosen, controls)
    first.write(out / TO_REVIEW, review)
    first.write(out / KEY, key)
    kept = [k for k in key if k["control"] == "no" and k["status"] == "assigned"]
    print(f"{len(review)} rows -> {out / TO_REVIEW}; key -> {out / KEY}")
    print(
        "assigned:",
        len(kept),
        "out_of_scope:",
        sum(k["status"] == "out_of_scope" for k in key),
        "controls:",
        len(controls),
    )
    print("by sector:", {SECTORS[s][0]: sum(k["sector"] == s for k in kept) for s in SECTORS})


if __name__ == "__main__":
    main()
