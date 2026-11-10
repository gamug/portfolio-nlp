"""What replaced the marked texts the code no longer writes (Work item 18, after PR #155).

    uv run scripts/dead_text_replacement_2026_11_10.py --out data --earlier DIR [--out ... --earlier ...]

``--out`` is a faithfulness directory (``faithfulness_ids.csv``, ``faithfulness_to_tag.csv``,
``faithfulness_stale.json``); ``--earlier`` is the directory that held the ids file before the owner marked
the new texts (only ``faithfulness_ids.csv`` is read). Give the two flags in pairs, in order.

* A row is **dead** when the code no longer writes its text for any extractor that held it (the stale
  record); **new** when its text hash is not in the earlier ids file; **kept** otherwise.
* Per extractor: its dead marks, how many have a **new row of the same extractor** for the same article and
  company (a dead text of one extractor sharing a company with a new row of another is not counted), the
  old mark against the new mark of those, and the texts the code writes that nobody has marked (the
  record's ``new`` list).
* A dead mark with no new row of its own means the code now writes no text for that company **or** writes
  a text that a row of another extractor already holds; the files cannot tell which (only running the
  extractors can).
* Per set, in rows: the share of ``no`` among the rows the code writes (kept + new).

It reads files and writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

Row = dict[str, str]


def _rows(path: Path) -> list[Row]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _extractors(row: Row) -> set[str]:
    return {x.strip() for x in row["extractors"].replace("|", ",").split(",") if x.strip()}


def replacement_lines(label: str, out: Path, earlier: Path) -> list[str]:
    ids = _rows(out / "faithfulness_ids.csv")
    marks = {
        r["review_order"]: (r.get("faithful") or "").strip().lower()
        for r in _rows(out / "faithfulness_to_tag.csv")
    }
    stale = json.loads((out / "faithfulness_stale.json").read_text(encoding="utf-8-sig"))[
        "extractors"
    ]
    seen = {r["text_hash"] for r in _rows(earlier / "faithfulness_ids.csv")}
    dead_for: dict[str, set[str]] = defaultdict(set)
    for name, entry in stale.items():
        for tag in entry["stale"]:
            dead_for[tag.rsplit(" (", 1)[0]].add(name)
    by_key = {r["key"]: r for r in ids}
    mark = lambda r: marks.get(r["review_order"], "")  # noqa: E731
    new_rows = [r for r in ids if r["text_hash"] not in seen]
    new_of: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in new_rows:
        new_of[(r["article_id"], r["ticker"])].append(r)
    kinds: dict[str, Counter[str]] = {"dead": Counter(), "kept": Counter(), "new": Counter()}
    for r in ids:
        if r["text_hash"] not in seen:
            kind = "new"
        elif r["key"] in dead_for and _extractors(r) <= dead_for[r["key"]]:
            kind = "dead"
        else:
            kind = "kept"
        kinds[kind][mark(r)] += 1
    written = kinds["kept"] + kinds["new"]
    lines = [
        f"{label}: {len(ids)} rows; dead {sum(kinds['dead'].values())} ({kinds['dead']['no']} no, {kinds['dead']['yes']} yes), "
        f"kept {sum(kinds['kept'].values())} ({kinds['kept']['no']} no), new {sum(kinds['new'].values())} "
        f"({kinds['new']['no']} no, {kinds['new']['yes']} yes)",
        f"{label}: the rows the code writes (kept + new): {sum(written.values())}, {written['no']} no "
        f"({100 * written['no'] / max(1, sum(written.values())):.1f}%)",
    ]
    for name, entry in stale.items():
        pairs: Counter[tuple[str, str]] = Counter()
        for tag in entry["stale"]:
            row = by_key[tag.rsplit(" (", 1)[0]]
            own = [
                n
                for n in new_of.get((row["article_id"], row["ticker"]), [])
                if name in _extractors(n)
            ]
            if own:
                pairs[(mark(row), "yes" if all(mark(n) == "yes" for n in own) else "no")] += 1
        replaced = sum(pairs.values())
        detail = ", ".join(f"{a} to {b}: {n}" for (a, b), n in sorted(pairs.items())) or "none"
        lines.append(
            f"{label} {name}: {len(entry['stale'])} dead marks, {replaced} with a new row of its own ({detail}); "
            f"{len(entry.get('new', []))} texts it writes that nobody has marked"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", action="append", type=Path, required=True)
    ap.add_argument("--earlier", action="append", type=Path, required=True)
    args = ap.parse_args(argv)
    if len(args.out) != len(args.earlier):
        ap.error("give --out and --earlier in pairs")
    for out, earlier in zip(args.out, args.earlier, strict=True):
        print("\n".join(replacement_lines(out.name or str(out), out, earlier)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
