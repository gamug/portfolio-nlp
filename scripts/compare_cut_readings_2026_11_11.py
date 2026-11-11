"""Cut the same bodies with two readings and compare the texts (T-176: is `units4` a better cut than `units`?).

    uv run scripts/compare_cut_readings_2026_11_11.py --results-db <nlp.db copy> --source-db <urls.db copy> \
        [--model-dir data/models/attribution_2026_11_08] [-n 232] [--a units] [--b units4] [--changed-out changed.csv]

Reads the first ``-n`` ``multi`` bodies of the latest ``body_scope`` revision, cuts each with reading ``--a`` and ``--b`` (one
model load, the GPU), and prints for each reading the faults of `docs/scope_training.md` §8.1: company texts that hold several
"Name -" headings, heading-only pieces, tiny pieces. Then what moved between the two: the (body, company) pairs that have a text
in A and none in B ("lost") and the other way ("gained"), and the texts that differ. ``--changed-out`` writes the pairs whose text
differs, with both texts, for the owner to read. Nothing is written to the databases (pass copies).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cut_quality_check_2026_11_11 as quality


def fault_counts(per_body: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """The faults of §8.1 in the company texts of one reading: ``per_body`` maps a body to its company texts."""
    texts = [x for t in per_body.values() for x in t]
    return {
        "company_texts": len(texts),
        "texts_with_several_headings": sum(1 for x in texts if quality.holds_several_headings(x)),
        "bodies_with_such_a_text": sum(
            1 for t in per_body.values() if any(quality.holds_several_headings(x) for x in t)
        ),
        "heading_only_texts": sum(1 for x in texts if quality.HEADING.match(x.strip())),
        f"texts_under_{quality.TINY}_chars": sum(1 for x in texts if len(x.strip()) < quality.TINY),
    }


def movement(
    a: Mapping[str, Mapping[str, str]], b: Mapping[str, Mapping[str, str]]
) -> dict[str, list[tuple[str, str]]]:
    """Per (body, ticker): ``lost`` (a text in A, none in B), ``gained`` (none in A) and ``changed`` (both, different)."""
    out: dict[str, list[tuple[str, str]]] = {"lost": [], "gained": [], "changed": []}
    for body in sorted(set(a) | set(b)):
        ta, tb = a.get(body, {}), b.get(body, {})
        for ticker in sorted(set(ta) | set(tb)):
            if ticker not in tb:
                out["lost"].append((body, ticker))
            elif ticker not in ta:
                out["gained"].append((body, ticker))
            elif ta[ticker] != tb[ticker]:
                out["changed"].append((body, ticker))
    return out


def compare(
    cut_a: Callable[[str], Mapping[str, str]],
    cut_b: Callable[[str], Mapping[str, str]],
    bodies: Sequence[tuple[str, str]],
) -> tuple[dict[str, Mapping[str, str]], dict[str, Mapping[str, str]]]:
    """``{body hash: {ticker: text}}`` for each reading."""
    return (
        {h: cut_a(text) for h, text in bodies},
        {h: cut_b(text) for h, text in bodies},
    )


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
    parser.add_argument("--a", default="units")
    parser.add_argument("--b", default="units4")
    parser.add_argument("--changed-out", default=None)
    args = parser.parse_args()
    import news_nlp as db  # noqa: PLC0415 -- after the path set-up

    db.env.load_env_file()
    from cut_stage import UnitsExtractor, companies_of  # noqa: PLC0415 -- loads torch
    from news_nlp import attribution_model as am  # noqa: PLC0415
    from news_nlp import steps  # noqa: PLC0415

    conn = db.connect_pipeline(results_db=args.results_db, source_db=args.source_db)
    companies = companies_of(conn)
    first = UnitsExtractor.load(args.model_dir, companies, args.a)
    second = UnitsExtractor(
        first.scorer, companies, f"local:{am.directory_identity(args.model_dir)}", args.b
    )
    scope = steps.latest_classifier_revision(conn)
    if scope is None:
        sys.exit("body_scope is empty: run --step scope first")
    todo = [
        (b.body_hash, b.text, b.title)
        for b in steps.pending_multi_bodies(conn, scope, "units-none", args.n)
    ]

    def texts_of(extractor: UnitsExtractor) -> Callable[[str], Mapping[str, str]]:
        return lambda body: am.model_company_texts(
            body, extractor.resolver, extractor.names, extractor.scorer, reading=extractor.reading
        )

    texts_a, texts_b = compare(texts_of(first), texts_of(second), [(h, t) for h, t, _ in todo])
    for name, got in ((args.a, texts_a), (args.b, texts_b)):
        print(name, fault_counts({h: list(v.values()) for h, v in got.items()}))
    moved = movement(texts_a, texts_b)
    print(
        {k: len(v) for k, v in moved.items()},
        "of",
        sum(len(v) for v in texts_a.values()),
        "texts in",
        args.a,
    )
    if args.changed_out:
        titles = {h: title for h, _, title in todo}
        with open(args.changed_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["kind", "body_hash", "title", "ticker", f"text_{args.a}", f"text_{args.b}"]
            )
            for kind in ("changed", "lost", "gained"):
                for body, ticker in moved[kind]:
                    writer.writerow(
                        [
                            kind,
                            body,
                            titles[body],
                            ticker,
                            texts_a[body].get(ticker, ""),
                            texts_b[body].get(ticker, ""),
                        ]
                    )
        print(f"wrote {sum(len(v) for v in moved.values())} rows to {args.changed_out}")


if __name__ == "__main__":
    main()
