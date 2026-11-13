"""Draw the yes/no sample of the sector step (owner, 2026-11-12): bodies the scope step calls ``sector``, with the sector the model assigns.

    uv run scripts/draw_sector_yesno_2026_11_12.py [--out data] [--count 100] [--controls 10] [--sample 10] [--seed 20261112]

The pool is the stored ``sector`` bodies of the scope step (RESULTS ``body_scope``) among 1 in ``--sample`` articles of SOURCE, read-only,
none of them a body (or a twin of one) already tagged in any gold file. Each is scored with the variant chosen on the dev half
(``sector_embed_tune_2026_11_12``: descriptions v2, noise direction 0.5, title, mean of the top 3 chunks) and the sample is balanced by
the sector assigned, half from the bodies the model is sure of (margin at or above the 25th percentile of the dev half) and half from
the rest. ``--controls`` extra rows show the sector the model finds *least* like the body: they are the check on a "yes" said too soon,
are marked only in the key file and do not count in the precision.

Writes ``<out>/sector_yesno_to_review.csv`` (the owner fills ``correct`` with yes or no, and optionally ``right_sector``) and
``<out>/sector_yesno_key.csv`` (margin, sure or not, control or not: not shown to the owner). Neither is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_scope_enrich_2026_11_10 as enrich
import draw_sector_extra_2026_11_05 as extra
import sector_embed_baseline_2026_11_12 as base
import sector_embed_tune_2026_11_12 as tune

import news_nlp as db
from news_nlp import triage

TO_REVIEW = "sector_yesno_to_review.csv"
KEY = "sector_yesno_key.csv"
MIN_CHARS = 400  # a paywall stub or a headline list is not read
ALPHA = 0.5
SECTORS: dict[str, tuple[str, str]] = {
    "10": ("Energy", "Oil, gas and coal producers, drillers, pipelines and refiners."),
    "15": (
        "Materials",
        "Chemicals, miners and metals, steel, packaging, paper and construction materials.",
    ),
    "20": (
        "Industrials",
        "Aerospace and defense, airlines, rail and freight, machinery, construction and business services.",
    ),
    "25": (
        "Consumer Discretionary",
        "Automakers, retailers and e-commerce, restaurants, hotels and leisure, homebuilders.",
    ),
    "30": (
        "Consumer Staples",
        "Food, beverages, tobacco, household products, grocery and drugstore chains.",
    ),
    "35": (
        "Health Care",
        "Drugmakers and biotech, medical devices, hospitals and health insurers.",
    ),
    "40": (
        "Financials",
        "Banks, insurers, asset managers, exchanges, payments and consumer finance.",
    ),
    "45": (
        "Information Technology",
        "Software, semiconductors, hardware and networking, IT services.",
    ),
    "50": (
        "Communication Services",
        "Telecom carriers, media and entertainment, social media and internet platforms, video games.",
    ),
    "55": (
        "Utilities",
        "Electric, gas and water utilities, renewable and independent power producers.",
    ),
    "60": (
        "Real Estate",
        "REITs, property owners and developers, housing and real estate services.",
    ),
}


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class Scorer:
    """The chosen variant: descriptions v2, noise 0.5, the title in the first chunk, mean of the 3 best chunks per sector."""

    def __init__(self) -> None:
        self.emb = base.Embedder()
        self.sectors = list(SECTORS)
        self.desc = self.emb([d for s in self.sectors for d in tune.DESC_V2[s]])
        self.per = len(tune.DESC_V2[self.sectors[0]])
        self.noise = self.emb([tune.NOISE])[0]

    def __call__(self, title: str, text: str) -> np.ndarray:
        chunks = base.chunks_of(text)
        vecs = self.emb([f"{title}. {chunks[0]}", *chunks[1:]])
        vecs = unit(vecs - ALPHA * (vecs @ self.noise)[:, None] * self.noise[None, :])
        s = (vecs @ self.desc.T).reshape(len(vecs), len(self.sectors), self.per).max(-1)
        return np.sort(s, axis=0)[-3:].mean(0)


def margin_of(score: np.ndarray) -> float:
    srt = np.sort(score)
    return float(srt[-1] - srt[-2])


def dev_threshold(scorer: Scorer) -> float:
    """The 25th percentile of the margins of the dev half (even positions) of the 70 single-sector bodies."""
    rows, texts = base.load_inputs()
    dev = rows[0::2]
    return float(np.quantile([margin_of(scorer(*texts[r["article_id"]])) for r in dev], 0.25))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pick(
    buckets: dict[str, dict[bool, list[dict[str, object]]]], count: int
) -> list[dict[str, object]]:
    """Round-robin over the sectors assigned, alternating the sure and the unsure bodies, until ``count``."""
    chosen: list[dict[str, object]] = []
    turn = 0
    while len(chosen) < count and any(v[True] or v[False] for v in buckets.values()):
        for sector in buckets:
            for sure in ((turn % 2 == 0), (turn % 2 == 1)):
                if buckets[sector][sure] and len(chosen) < count:
                    chosen.append(buckets[sector][sure].pop(0))
                    break
        turn += 1
    return chosen


Row = tuple[int, str, str, str, str, str, str, str]


def read_pool(
    conn: db.NewsNlpDatabase, args: argparse.Namespace, skip: set[str], sector_hashes: set[str]
) -> list[Row]:
    """The sector bodies of 1 in ``--sample`` articles of SOURCE, one per body, none already tagged."""
    pool: list[Row] = []
    seen: set[str] = set()
    for article_id, ticker, title, body, company, pub_date, domain in conn.execute(
        "SELECT id, ticker, title, body_text, company, pub_date, source_domain FROM source.articles "
        "WHERE fetch_status = 'ok' AND body_text IS NOT NULL AND length(body_text) >= ? AND id % ? = ? ORDER BY id",
        (MIN_CHARS, args.sample, args.seed % args.sample),
    ):
        key = triage.body_hash(body)
        if key in seen or key in skip or key not in sector_hashes:
            continue
        seen.add(key)
        pool.append(
            (
                int(article_id),
                ticker or "",
                company or "",
                pub_date or "",
                domain or "",
                title or "",
                body,
                key,
            )
        )
    return pool


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--controls", type=int, default=10)
    parser.add_argument("--sample", type=int, default=10, help="take 1 in this many articles")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    out: Path = args.out
    for name in (TO_REVIEW, KEY):
        if (out / name).exists():
            sys.exit(f"{out / name} exists and is not overwritten")
    conn = db.connect_pipeline()
    tagged = enrich.tagged_article_ids(args.gold)
    for name in ("scope_enrich.csv", "scope_hook.csv"):
        tagged |= {int(r["article_id"]) for r in extra._read(args.gold / name)}
    skip = extra.excluded_hashes(conn, tagged)
    sector_hashes = {
        str(r[0]) for r in conn.execute("SELECT body_hash FROM body_scope WHERE scope = 'sector'")
    }
    pool = read_pool(conn, args, skip, sector_hashes)
    print(f"{len(pool)} sector bodies in the pool", flush=True)
    scorer = Scorer()
    threshold = dev_threshold(scorer)
    print(f"sure = margin >= {threshold:.4f} (25th percentile of the dev half)", flush=True)
    buckets: dict[str, dict[bool, list[dict[str, object]]]] = {
        s: {True: [], False: []} for s in SECTORS
    }
    rng = random.Random(args.seed)  # noqa: S311 -- a reproducible draw, not security
    rng.shuffle(pool)
    for n, (aid, ticker, company, pub_date, domain, title, body, _key) in enumerate(pool, 1):
        score = scorer(title, body[: extra.TEXT_CAP])
        order = np.argsort(-score)
        best, worst = scorer.sectors[order[0]], scorer.sectors[order[-1]]
        marg = margin_of(score)
        cut = body[: extra.TEXT_CAP]
        buckets[best][marg >= threshold].append(
            {
                "article_id": aid, "ticker": ticker, "company": company, "pub_date": pub_date, "source_domain": domain,
                "title": title, "text": cut + extra.STAGE_TEXT if len(body) > extra.TEXT_CAP else cut,
                "best": best, "worst": worst, "margin": marg, "sure": marg >= threshold,
            }
        )  # fmt: skip
        if n % 500 == 0:
            print(f"  scored {n}", flush=True)
    chosen = pick(buckets, args.count + args.controls)
    controls = {int(c["article_id"]) for c in rng.sample(chosen, min(args.controls, len(chosen)))}  # type: ignore[call-overload]
    rng.shuffle(chosen)
    review, key_rows = [], []
    for order_no, c in enumerate(chosen, 1):
        control = int(c["article_id"]) in controls  # type: ignore[call-overload]
        shown = str(c["worst"] if control else c["best"])
        name, blurb = SECTORS[shown]
        review.append(
            {
                "review_order": str(order_no), "article_id": str(c["article_id"]), "ticker": str(c["ticker"]),
                "company": str(c["company"]), "pub_date": str(c["pub_date"]), "source_domain": str(c["source_domain"]),
                "title": str(c["title"]), "text": str(c["text"]),
                "model_sector_code": shown, "model_sector": name, "sector_description": blurb,
                "correct": "", "right_sector": "", "notes": "",
            }
        )  # fmt: skip
        key_rows.append(
            {
                "review_order": str(order_no), "article_id": str(c["article_id"]), "model_best": str(c["best"]),
                "shown": shown, "control": "yes" if control else "no", "margin": f"{float(c['margin']):.4f}",  # type: ignore[arg-type]
                "sure": "yes" if c["sure"] else "no",
            }
        )  # fmt: skip
    write(out / TO_REVIEW, review)
    write(out / KEY, key_rows)
    kept = [r for r in key_rows if r["control"] == "no"]
    by = {s: sum(1 for r in kept if r["model_best"] == s) for s in SECTORS}
    print(f"{len(review)} rows ({len(controls)} controls) -> {out / TO_REVIEW}; key -> {out / KEY}")
    print("assigned (without controls):", {SECTORS[s][0]: n for s, n in by.items() if n})
    print(
        "sure:",
        sum(1 for r in kept if r["sure"] == "yes"),
        "unsure:",
        sum(1 for r in kept if r["sure"] == "no"),
    )


if __name__ == "__main__":
    main()
