"""Read-only quality check of the cut step (step 1.2, T-163): what the `units` extractor wrote into `sub_news`.

    uv run scripts/cut_quality_check_2026_11_11.py --results-db <nlp.db copy> --source-db <urls.db copy> \
        [--revision units-...] [--sample 12] [--out cut_sample.csv]

The same idea as ``scope_quality_check_2026_11_11.py``: numbers that show a surprise before the full run is paid for in hours.
It prints, for one extractor revision (default the latest):
- the bodies by state (done / skipped "no company text" / failed) and the share skipped;
- sub-news per body (min, median, mean, p90, max) and the bodies with a single sub-news;
- the length of a sub-news in characters (min, p10, median, p90, max) and how many are under 80 characters;
- the bodies whose longest sub-news covers more than 80% of the body (the cut did not cut);
``--sample N`` also writes N random done bodies with the body and each sub-news, for a person to read (not a tagging file: no
validator column, so no Spanish columns). Reads both files read-only: pass copies of ``nlp.db`` and ``urls.db``.
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import triage

SHORT = 80  # characters: a sub-news this short is a heading, not a story
WHOLE = 0.8  # a sub-news this share of the body or more: the body was not cut


def quantiles(values: list[int]) -> dict[str, float]:
    """min / p10 / median / mean / p90 / max of a list (empty list: all zero)."""
    if not values:
        return dict.fromkeys(("min", "p10", "median", "mean", "p90", "max"), 0.0)
    ordered = sorted(values)

    def at(q: float) -> float:
        return float(ordered[min(len(ordered) - 1, int(q * len(ordered)))])

    return {
        "min": float(ordered[0]),
        "p10": at(0.10),
        "median": float(statistics.median(ordered)),
        "mean": round(statistics.fmean(ordered), 1),
        "p90": at(0.90),
        "max": float(ordered[-1]),
    }


def summarise(
    per_body: dict[str, list[str]], body_len: dict[str, int], skipped: int
) -> dict[str, object]:
    """The breakdown of the sub-news of one revision; ``per_body`` maps a body hash to its sub-news texts."""
    counts = [len(t) for t in per_body.values()]
    lengths = [len(x) for t in per_body.values() for x in t]
    whole = [
        h
        for h, t in per_body.items()
        if body_len.get(h) and max(len(x) for x in t) >= WHOLE * body_len[h]
    ]
    return {
        "bodies_done": len(per_body),
        "bodies_skipped_no_company_text": skipped,
        "skipped_share": round(skipped / (skipped + len(per_body)), 3)
        if per_body or skipped
        else 0.0,
        "sub_news_total": len(lengths),
        "sub_news_per_body": quantiles(counts),
        "bodies_with_one_sub_news": sum(1 for c in counts if c == 1),
        "sub_news_chars": quantiles(lengths),
        f"sub_news_under_{SHORT}_chars": sum(1 for n in lengths if n < SHORT),
        f"bodies_not_cut_(longest>={int(WHOLE * 100)}%_of_body)": len(whole),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument(
        "--revision", default=None, help="extractor revision (default: the latest with state)"
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="also write this many random done bodies"
    )
    parser.add_argument("--out", default="cut_sample.csv")
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    res = sqlite3.connect(f"file:{args.results_db}?mode=ro", uri=True)
    revision = (
        args.revision
        or res.execute(
            "SELECT model_revision FROM pipeline_step_state WHERE step = 'cut' ORDER BY finished_at DESC, rowid DESC LIMIT 1"
        ).fetchone()[0]
    )
    per_body: dict[str, list[str]] = {}
    for body_hash, text in res.execute(
        "SELECT body_hash, text FROM sub_news WHERE extractor_revision = ? ORDER BY body_hash, segment_index",
        (revision,),
    ):
        per_body.setdefault(str(body_hash), []).append(str(text))
    skipped = res.execute(
        "SELECT COUNT(*) FROM pipeline_step_state WHERE step = 'cut' AND model_revision = ? AND status = 'skipped'",
        (revision,),
    ).fetchone()[0]
    src = sqlite3.connect(f"file:{args.source_db}?mode=ro", uri=True)
    wanted = set(per_body)
    bodies: dict[str, tuple[int, str, str]] = {}
    for article_id, title, text in src.execute(
        "SELECT id, title, body_text FROM articles WHERE fetch_status IN ('ok', 'quarantined') "
        "AND body_text IS NOT NULL AND body_text != '' ORDER BY id"
    ):
        key = triage.body_hash(text)
        if key in wanted and key not in bodies:
            bodies[key] = (int(article_id), title or "", text)
            if len(bodies) == len(wanted):
                break
    report = summarise(per_body, {h: len(b[2]) for h, b in bodies.items()}, int(skipped))
    print(f"revision {revision}")
    for key, value in report.items():
        print(f"  {key}: {value}")
    if args.sample:
        rng = random.Random(args.seed)  # noqa: S311 -- a reproducible draw, not security
        chosen = rng.sample(sorted(bodies), min(args.sample, len(bodies)))
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["article_id", "title", "body", "n_sub_news", "sub_news"])
            for key in chosen:
                article_id, title, text = bodies[key]
                writer.writerow(
                    [article_id, title, text, len(per_body[key]), "\n---\n".join(per_body[key])]
                )
        print(f"wrote {len(chosen)} bodies to {args.out}")


if __name__ == "__main__":
    main()
