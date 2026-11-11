"""Put the owner's tagging of the rare-scope draw in tracked gold (T-147; owner 2026-11-10, set 7 of docs/gold_datasets.md).

    uv run scripts/build_scope_enrich_gold_2026_11_10.py [--data data] [--gold gold]

Reads ``data/scope_enrich_to_tag.csv`` (the 51 bodies of ``draw_scope_enrich_2026_11_10.py`` the owner tagged), refuses the file
unless every row is tagged and valid (``news_nlp.retag``, as ``build_gold_2026_11_05.py`` does), and writes
``gold/scope_enrich.csv`` (``retag.GOLD_COLUMNS`` + ``rule``, the rules' guess the body was drawn under) and, with the texts the
owner read, ``gold/texts/scope_enrich_texts.csv`` (the owner's approval of 2026-11-10: the texts are tracked with the marks).
This set is **not** part of the weighted held-out 200: it was drawn by rules to hold more sector and out-of-scope bodies, so its
recalls are class-conditional, unweighted and easier than the corpus; report it apart.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import retag_gold_2026_10_29 as retag_cli

from news_nlp import retag

csv.field_size_limit(sys.maxsize)
TEXT_COLUMNS = ("article_id", "ticker", "company", "pub_date", "source_domain", "title", "text")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument(
        "--set",
        default="scope_enrich",
        help="scope_enrich (the rare-scope draw) or scope_hook (the company-as-hook draw of 2026-11-11)",
    )
    args = parser.parse_args()
    name = f"{args.set}_to_tag.csv"
    rows = retag.read_tag_file(args.data / name)
    # a ';' between two labels of the same company is a format slip (the format wants a comma): normalised on a copy,
    # the owner's file is not edited, the scope is never touched, and every change is printed
    for r in rows:
        if ";" in r["company_labels"]:
            print(
                f"article {r['article_id']}: company_labels separator ';' read as ',' ({r['company_labels']!r})"
            )
            r["company_labels"] = r["company_labels"].replace(";", ",")
    with tempfile.TemporaryDirectory() as tmp:
        checked = Path(tmp) / name
        with checked.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        report = retag.load_gold_v2(
            checked, [int(r["article_id"]) for r in rows], retag_cli.load_universe()
        )
    if report.bad or report.untagged or report.duplicated or report.unknown:
        sys.exit(
            f"{name}: not ready (errors {sorted(report.bad)}, untagged {report.untagged}, "
            f"duplicated {report.duplicated}, unknown {report.unknown}); run the check command"
        )
    gold_rows = retag.gold_file_rows({args.set: rows}, {args.set: report.good})
    ids = (
        args.data / f"{args.set}_ids.csv"
    )  # the hook draw has no ids file: its rule is the set's own name
    rule = {r["article_id"]: r["rule"] for r in read(ids)} if ids.exists() else {}
    for r in gold_rows:
        r["rule"] = rule.get(r["article_id"], args.set)
    args.gold.mkdir(parents=True, exist_ok=True)
    with (args.gold / f"{args.set}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*retag.GOLD_COLUMNS, "rule"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(gold_rows)
    texts = {r["article_id"]: r for r in rows}
    (args.gold / "texts").mkdir(exist_ok=True)
    with (args.gold / "texts" / f"{args.set}_texts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(TEXT_COLUMNS), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(texts[r["article_id"]] for r in gold_rows)
    print(
        f"{args.gold / f'{args.set}.csv'}: {len(gold_rows)} rows, texts in {args.gold / 'texts' / f'{args.set}_texts.csv'}"
    )


if __name__ == "__main__":
    main()
