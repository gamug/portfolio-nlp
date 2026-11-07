"""Put the owner's tagging sessions of the prompt gate in one tracked gold file (T-146c).

    uv run scripts/build_gold_2026_11_05.py [--set train|test] [--out data] [--gold PATH]

``--set train`` (the default) reads ``data/gate240_to_tag.csv`` (the fixed 240) and ``data/sector_extra_to_tag.csv`` (44 sector
bodies), checks every row with ``news_nlp.retag`` (the file is refused unless all rows are tagged
and valid, and no article is in both) and writes ``gold/category_gold_v2.csv``: the tags only,
sorted by ``article_id`` (``retag.GOLD_COLUMNS``). The article text and its Spanish aid stay out of
git (news text, public repository); read it back from the SOURCE store by ``article_id``. The gold
file is rewritten from the two tagging files each time.

``--set test`` does the same for the held-out set: it reads ``data/heldout_to_tag.csv`` (the 200 the
owner tags) and writes ``gold/category_gold_v2_test.csv`` (``session`` = ``heldout``, with the draw's
``stratum`` from ``data/heldout_ids.csv`` and ``frame`` = the stratum's size from
``data/heldout_strata.csv``, so a rate can be re-weighted from ``gold/`` alone). The set is fixed, so the run is refused unless the tagged articles are exactly the ones already in the tracked
test file. A tag already in that file is protected: a rebuild that would change one stops and names
the ids (``--overwrite-marks`` to accept, printing the ids that change; the reference is the file as
committed, ``git show HEAD:``, not a local edit); the rows of the shell, still untagged, are filled in freely.
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import retag_gold_2026_10_29 as retag_cli

from news_nlp import retag

SETS = {
    "train": (
        {"gate240": "gate240_to_tag.csv", "sector_extra": "sector_extra_to_tag.csv"},
        Path("gold/category_gold_v2.csv"),
    ),
    "test": ({"heldout": "heldout_to_tag.csv"}, Path("gold/category_gold_v2_test.csv")),
}


def fixed_ids(path: Path) -> set[str]:
    """The article ids already in a tracked gold file (empty when it does not exist)."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {r["article_id"] for r in csv.DictReader(fh)}


def changed_tags(old: list[dict[str, str]], new: list[dict[str, str]]) -> list[str]:
    """The ids of tracked rows that already carry a tag and that *new* changes (any column) or
    drops. An untagged row (the shell of the draw) may be filled in."""
    fresh = {r["article_id"]: r for r in new}
    return [
        r["article_id"]
        for r in old
        if (r.get("scope") or "").strip()
        and any(fresh.get(r["article_id"], {}).get(c) != r[c] for c in r)
    ]


def reference_rows(gold_path: Path) -> list[dict[str, str]]:
    """The tags a rebuild is compared with: the file as committed (``git show HEAD:``), so a local
    edit of the file is not the reference; the file on disk when it is not in a repository."""
    if gold_path.exists():
        shown = subprocess.run(  # noqa: S603
            ["git", "-C", str(gold_path.parent), "show", f"HEAD:./{gold_path.name}"],  # noqa: S607
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if shown.returncode == 0:
            return list(csv.DictReader(io.StringIO(shown.stdout.lstrip("\ufeff"), newline="")))
        return _read(gold_path)
    return []


def protect_tags(gold_path: Path, out_rows: list[dict[str, str]], accept: bool = False) -> None:
    """Stop if rebuilding *gold_path* from *out_rows* would change a tag it already holds; with
    *accept* (``--overwrite-marks``) say which ones change instead."""
    moved = changed_tags(reference_rows(gold_path), out_rows)
    if moved and accept:
        print(f"--overwrite-marks: {len(moved)} tags of {gold_path} change (ids {moved})")
    elif moved:
        sys.exit(
            f"{len(moved)} tags of {gold_path} would change (first ids {moved[:5]}): the held-out "
            "set is fixed (D30); run with --overwrite-marks if this is intended"
        )


def add_draw_weights(out_rows: list[dict[str, str]], out: Path) -> None:
    """Give each held-out row the draw's ``stratum`` and ``frame`` (the stratum's size)."""
    stratum_of = {r["article_id"]: r["stratum"] for r in _read(out / "heldout_ids.csv")}
    frame = {r["stratum"]: r["frame"] for r in _read(out / "heldout_strata.csv")}
    if missing := sorted({r["article_id"] for r in out_rows} - stratum_of.keys()):
        sys.exit(f"heldout_ids.csv lacks the stratum of {missing[:5]}")
    if unsized := sorted({stratum_of[r["article_id"]] for r in out_rows} - frame.keys()):
        sys.exit(f"heldout_strata.csv lacks the frame size of the strata {unsized}")
    for r in out_rows:
        r["stratum"] = stratum_of[r["article_id"]]
        r["frame"] = frame[r["stratum"]]  # D30: the size in the whole sample, not "free"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--set", choices=sorted(SETS), default="train")
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument(
        "--overwrite-marks",
        action="store_true",
        help="--set test: accept a change to a tag already in the tracked held-out file",
    )
    args = parser.parse_args()
    sessions_files, default_gold = SETS[args.set]
    gold_path: Path = args.gold or default_gold
    universe = retag_cli.load_universe()
    sessions: dict[str, list[dict[str, str]]] = {}
    golds: dict[str, list[retag.Gold]] = {}
    for session, name in sessions_files.items():
        try:
            rows = retag.read_tag_file(args.out / name)
        except retag.TagFileError as exc:
            sys.exit(str(exc))
        ids = [int(r["article_id"]) for r in rows]
        report = retag.load_gold_v2(args.out / name, ids, universe)
        if report.bad or report.untagged or report.duplicated or report.unknown:
            sys.exit(
                f"{name}: not ready for the gold file (errors {sorted(report.bad)}, "
                f"untagged {report.untagged}, duplicated {report.duplicated}, "
                f"unknown {report.unknown}); run the check command"
            )
        sessions[session], golds[session] = rows, report.good
        print(f"{name}: {len(report.good)} rows tagged and valid")
    try:
        out_rows = retag.gold_file_rows(sessions, golds)
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    fixed = fixed_ids(gold_path) if args.set == "test" else set()
    if fixed and {r["article_id"] for r in out_rows} != fixed:
        sys.exit(f"the tagged articles are not the {len(fixed)} fixed in {gold_path}")
    columns = retag.GOLD_COLUMNS
    if args.set == "test":
        columns = retag.TEST_GOLD_COLUMNS
        add_draw_weights(out_rows, args.out)
    if args.set == "test":
        protect_tags(gold_path, out_rows, args.overwrite_marks)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with gold_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"{gold_path}: {len(out_rows)} rows")


if __name__ == "__main__":
    main()
