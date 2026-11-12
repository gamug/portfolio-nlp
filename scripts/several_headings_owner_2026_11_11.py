"""Read the company texts that still hold several headings (T-176, review of #169): overlap, or a line given to a company it does not name?

    uv run scripts/several_headings_owner_2026_11_11.py --results-db <nlp.db copy> --source-db <urls.db copy> \
        [--model-dir data/models/attribution_2026_11_08] [-n 232] [--reading units4|units5]

Cuts the first ``-n`` ``multi`` bodies with ``--reading`` (default `units4`; the model on the GPU) and, for each text that holds two or more "Name -"
headings, splits it at the headings and looks for the text's own company in each part. A text where **every** part names its
owner is *overlap* (a company named in other companies' lines is, by `units4`'s rule, that company's news). A text with a part
that does not is the *third fault*: a line the cut gave to a company it does not name, because its header was not read. The parts
that do not name the owner are printed with their head and a cause: a head in capitals ("SAP -"), a head with lowercase words
("Memory stocks -"), or a dash inside a sentence that the quality check takes for a heading. Nothing is written (pass copies).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cut_quality_check_2026_11_11 as quality

DASH = re.compile(r"\s[\u2014\u2013-]\s")
MAX_HEAD = 60  # characters: a longer "head" is a sentence with a dash in it
BRACKET_TICKER = re.compile(r"\s*\([A-Z.]+\)")


def parts_of(text: str) -> list[str]:
    """The text cut at each inner heading (the first part is what comes before the first one, when something does)."""
    starts = [m.start() for m in quality.INNER_HEADING.finditer(text)]
    return [text[a:b] for a, b in zip(starts, [*starts[1:], len(text)], strict=True)]


def cause_of(part: str) -> str:
    """Why the header of a part was not read: by the shape of its head."""
    stripped = part.lstrip(". ")
    head = BRACKET_TICKER.sub("", DASH.split(stripped, maxsplit=1)[0]).strip()
    if head == stripped.strip() or len(head) > MAX_HEAD:
        return "a dash inside a sentence (the check's false positive)"
    if head.replace(" ", "").replace("&", "").isupper():
        return "a head in capitals that names no universe company"
    if any(w[:1].islower() for w in head.split()):
        return "a head with lowercase words (a group label)"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/models/attribution_2026_11_08",
    )
    parser.add_argument("-n", type=int, default=232)
    parser.add_argument("--reading", choices=("units", "units4", "units5"), default="units4")
    args = parser.parse_args()
    import news_nlp as db  # noqa: PLC0415 -- after the path set-up

    db.env.load_env_file()
    from cut_stage import UnitsExtractor, companies_of  # noqa: PLC0415 -- loads torch
    from news_nlp import attribution_model as am  # noqa: PLC0415
    from news_nlp import steps  # noqa: PLC0415

    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    extractor = UnitsExtractor.load(args.model_dir, companies_of(conn), args.reading)
    scope = steps.latest_classifier_revision(conn)
    if scope is None:
        sys.exit("body_scope is empty: run --step scope first")
    texts = overlap = fault = 0
    flagged_bodies: set[str] = set()
    causes: Counter[str] = Counter()
    for body in steps.pending_multi_bodies(conn, scope, "units-none", args.n):
        cut = am.model_company_texts(
            body.text, extractor.resolver, extractor.names, extractor.scorer, reading=args.reading
        )
        for ticker, text in cut.items():
            texts += 1
            if not quality.holds_several_headings(text):
                continue
            flagged_bodies.add(body.body_hash)
            missing = [
                p
                for p in parts_of(text)
                if ticker not in {m[2] for m in extractor.resolver.mention_spans(p)}
            ]
            if not missing:
                overlap += 1
                continue
            fault += 1
            for part in missing:
                causes[cause_of(part)] += 1
                print(f"{ticker}: {cause_of(part)}: {part.lstrip('. ')[:90]}")
    print(
        f"bodies {args.n}; company texts {texts}; with several headings {overlap + fault} "
        f"({len(flagged_bodies)} bodies): overlap {overlap}, a line given to a company it does not name {fault}"
    )
    for cause, count in causes.most_common():
        print(f"  {count}: {cause}")


if __name__ == "__main__":
    main()
