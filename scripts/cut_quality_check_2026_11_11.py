"""Read-only quality check of the cut step (step 1.2, T-163): what the `units` extractor wrote into `sub_news`.

    uv run scripts/cut_quality_check_2026_11_11.py --results-db <nlp.db copy> --source-db <urls.db copy> \
        [--revision units-...] [--sample 12] [--out cut_sample.csv]

The same idea as ``scope_quality_check_2026_11_11.py``: numbers that show a surprise before the full run is paid for in hours.
It prints, for one extractor revision (default the latest):
- the bodies by state (done / skipped "no company text" / failed) and the share skipped;
- sub-news per body (min, median, mean, p90, max) and the bodies with a single sub-news;
- the length of a sub-news in characters (min, p10, median, p90, max) and how many are under 80 characters;
- the bodies whose longest sub-news covers more than 80% of the body (the cut did not cut);
- the pieces that end like a fragment (under 80 characters and no final . ! ? or quote: "Marvell Technology rallied 12%,");
- the pieces that hold two or more "Name —" headings (a list written in one paragraph that was not split: a company whose text
  carries other companies' news), and the bodies with at least one;
- the pieces under 30 characters and the pieces that are only a name and a dash ("Micron Technology —": a company left without
  its sentence), and the bodies with at least one of them.
``--sample N`` also writes N bodies with the body and each sub-news, for a person to read (not a tagging file: no validator
column, so no Spanish columns): up to a third of N for each kind of flag (not cut, tiny piece, several headings in a piece), then random bodies up to N,
with a ``flag`` column. Reads both files read-only: pass **copies** of ``nlp.db`` (with its ``-wal``) and ``urls.db``; the
owner's databases sit on a mount where a read-only open fails with "disk I/O error".
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import triage

SHORT = 80  # characters: a sub-news this short is a heading, not a story
WHOLE = 0.8  # a sub-news this share of the body or more: the body was not cut
SEVERAL = 2  # headings in one piece: the piece is more than one company's news
TINY = 30  # characters: a piece this short carries no news
HEADING = re.compile(
    r"^[^.!?]{1,60}\s[\u2014\u2013-]\s*$"
)  # "Micron Technology —": the company's name and nothing else


# a heading inside a piece: at its start, after a sentence end or after a run of spaces, "Name (TICKER) - " with a dash
INNER_HEADING = re.compile(r"(?:^|[.!?]\s+|\s{2,})[A-Z][^.!?\u2014\u2013]{1,50}\s[\u2014\u2013]\s")
SENTENCE_END = re.compile(r"[.!?\"\u201d]\s*$")


def is_fragment(piece: str) -> bool:
    """Under 80 characters and not ending like a sentence: a clause cut in half, a heading, a list of names."""
    return len(piece) < SHORT and not SENTENCE_END.search(piece.strip())


def holds_several_headings(piece: str) -> bool:
    return len(INNER_HEADING.findall(piece)) >= SEVERAL


def is_tiny(piece: str) -> bool:
    return len(piece.strip()) < TINY or bool(HEADING.match(piece.strip()))


def flag_of(pieces: list[str], body_len: int) -> str:
    """Why a body is worth reading: ``not_cut``, ``tiny_piece``, both joined by ``+``, or empty."""
    flags = []
    if body_len and max(len(x) for x in pieces) >= WHOLE * body_len:
        flags.append("not_cut")
    if any(is_tiny(x) for x in pieces):
        flags.append("tiny_piece")
    if any(holds_several_headings(x) for x in pieces):
        flags.append("several_headings")
    return "+".join(flags)


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
        f"pieces_under_{SHORT}_chars_not_ending_like_a_sentence": sum(
            1 for t in per_body.values() for x in t if is_fragment(x)
        ),
        "pieces_with_several_headings": sum(
            1 for t in per_body.values() for x in t if holds_several_headings(x)
        ),
        "bodies_with_a_piece_of_several_headings": sum(
            1 for t in per_body.values() if any(holds_several_headings(x) for x in t)
        ),
        f"pieces_under_{TINY}_chars": sum(1 for n in lengths if n < TINY),
        "heading_only_pieces": sum(
            1 for t in per_body.values() for x in t if HEADING.match(x.strip())
        ),
        "bodies_with_a_tiny_or_heading_piece": sum(
            1 for t in per_body.values() if any(is_tiny(x) for x in t)
        ),
    }


def latest_cut_revision(res: sqlite3.Connection) -> str:
    """The extractor revision of the most recent cut state; exits with a message when the file has none."""
    row = res.execute(
        "SELECT model_revision FROM pipeline_step_state WHERE step = 'cut' ORDER BY finished_at DESC, rowid DESC LIMIT 1"
    ).fetchone()
    if row is None:
        sys.exit(
            "no cut state in this file: run `--step cut` first (or pass the right --results-db)"
        )
    return str(row[0])


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
    revision = args.revision or latest_cut_revision(res)
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
        flags = {k: flag_of(per_body[k], len(b[2])) for k, b in bodies.items()}
        chosen = []
        for kind in (
            "not_cut",
            "tiny_piece",
            "several_headings",
        ):  # up to a third of N for each kind of flag
            pool = sorted(k for k, f in flags.items() if kind in f and k not in chosen)
            chosen += rng.sample(pool, min(len(pool), max(1, args.sample // 3)))
        rest = sorted(k for k, f in flags.items() if k not in chosen)  # then any body, to fill N
        chosen += rng.sample(rest, max(0, min(args.sample - len(chosen), len(rest))))
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["article_id", "title", "body", "flag", "n_sub_news", "sub_news"])
            for key in chosen:
                article_id, title, text = bodies[key]
                writer.writerow(
                    [
                        article_id,
                        title,
                        text,
                        flags[key],
                        len(per_body[key]),
                        "\n---\n".join(per_body[key]),
                    ]
                )
        print(f"wrote {len(chosen)} bodies to {args.out}")


if __name__ == "__main__":
    main()
