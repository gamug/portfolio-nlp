"""Draw the fixed 240-body set of the prompt gate (Work item 18, T-146c) from the review file.

    uv run scripts/draw_gate240_2026_11_04.py [draw] [--out data] [--file NAME] [--seed 20261104]
                                              [--tracked gold]
    uv run scripts/draw_gate240_2026_11_04.py merge [--out data] [--file NAME] [--gate-wins]

``draw`` writes ``data/gate240_ids.csv`` (the 240 with their stratum; **drawn once**, a second run
does not draw again) and ``data/gate240_to_tag.csv``: a tagging file of its own with the 240 rows
in the same columns as the review file (title, text, the Spanish ``spanish_title`` /
``spanish_text`` after ``text``, the v1 columns and the columns to tag), the tags the owner has
already made included. That file is never overwritten once it holds rows. Nothing is written until
the draw, the quotas and the ids have been checked (a stratum short of its quota, a gate id missing
from the review file, or more tagged rows than a quota all stop the run).

The set is also recorded in git, because ``data/`` is not: ``gold/gate240_ids.csv`` (article id and
stratum only, no news text) and ``gold/gate240_draw.json`` (seed, quotas and the hash of the review
file). The draw depends on which rows were tagged when it ran, so the same seed on a later review
file would give another set: the recorded ids are the gate, not the seed.

``merge`` copies the tags made in the gate file into the review file as a new file
(``<stem>_merged_<timestamp>.csv``); nothing is replaced and a row with different tags in both is
listed with the columns that differ, and every differing column is written to
``<stem>_conflicts_<timestamp>.csv`` (review value, gate value, kept or replaced). While any row
is still in conflict the command exits 1: the merged file is not complete. ``--gate-wins`` takes
the gate file's tags for those rows (the owner tags in the gate file only, so its tags are the later
ones); the audit file shows what was replaced. The **newest** ``_merged_`` file is the current review file:
``draw`` and ``merge`` both start from it, so tags accumulate in one chain and the original file
stays as it was.

Titles are copied as they are. A title that starts with ``=``, ``+``, ``-`` or ``@`` would be read as
a formula by a spreadsheet; the titles are public news headlines the owner already reads in the
review file, so they are not rewritten here (rewriting them would make this file differ from the
review file it merges back into).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import retag

IDS_NAME = "gate240_ids.csv"
TO_TAG_NAME = "gate240_to_tag.csv"
DRAW_NAME = "gate240_draw.json"
TRACKED_COLUMNS = ("review_order", "article_id", "stratum")


def _write(
    path: Path, rows: Sequence[Mapping[str, str]], columns: Sequence[str] | None = None
) -> None:
    """Write through a temporary file and rename, so a crash never leaves a truncated CSV."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(columns or rows[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except PermissionError:
        sys.exit(
            f"cannot write {path}: it is open in another program (Excel?); close it and run again"
        )


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _current_review(out: Path, file: str) -> Path:
    """The newest ``<stem>_merged_*.csv`` (the names sort by time), else the review file itself."""
    merged = sorted(out.glob(f"{Path(file).stem}_merged_*.csv"))
    return merged[-1] if merged else out / file


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    tracked: Path,
    review: Path,
    chosen: Sequence[dict[str, str]],
    seed: int,
    *,
    after: bool,
    quotas: Mapping[str, int],
) -> None:
    """The set and how it was drawn, in a directory git tracks. Written once."""
    tracked.mkdir(parents=True, exist_ok=True)
    if not (tracked / IDS_NAME).exists():
        _write(tracked / IDS_NAME, chosen, TRACKED_COLUMNS)
    if not (tracked / DRAW_NAME).exists():
        info = {
            "seed": seed,
            "quotas": dict(quotas),
            "review_file": review.name,  # the file the rows were read from
            "review_sha256": _sha256(review),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "recorded_after_draw": after,  # True: the hash is of the file at recording, not at the draw
        }
        (tracked / DRAW_NAME).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def _check_tracked(chosen: Sequence[Mapping[str, str]], tracked_ids: Path) -> None:
    """The local set and the one in git must be the same articles in the same strata."""
    if not tracked_ids.exists():
        return
    key = lambda rows: [(r["review_order"], r["article_id"], r["stratum"]) for r in rows]  # noqa: E731
    if key(chosen) != key(_read(tracked_ids)):
        sys.exit(
            f"{tracked_ids} and the local set differ: the gate is the recorded set, "
            "restore whichever file was edited"
        )


def cmd_draw(
    out: Path, file: str, seed: int, tracked: Path, quotas: Mapping[str, int] = retag.GATE_QUOTAS
) -> None:
    review = _current_review(out, file)
    rows = retag.read_tag_file(review)
    ids_path = out / IDS_NAME
    existing = ids_path.exists()
    if existing:  # the set is fixed
        chosen = _read(ids_path)
        print(f"{ids_path} exists: the set is not drawn again")
        _check_tracked(chosen, tracked / IDS_NAME)
        recorded = tracked / DRAW_NAME
        if recorded.exists() and json.loads(recorded.read_text("utf-8"))["seed"] != seed:
            print(f"  --seed {seed} ignored: the set was drawn with another seed ({recorded})")
    else:
        try:
            chosen = retag.draw_gate(rows, quotas, seed=seed)
        except ValueError as exc:
            sys.exit(str(exc))
        short, unknown = retag.gate_shortfalls(rows, chosen, quotas)
        if short or unknown:
            detail = ", ".join(f"{s}: {got} of {q}" for s, (got, q) in short.items())
            sys.exit(
                f"the draw is not {sum(quotas.values())} bodies ({len(chosen)}); "
                f"short strata: {detail or 'none'}; rows outside every stratum: {unknown}. "
                "Nothing was written."
            )
    try:
        gate = retag.gate_rows(chosen, rows)  # checked before the first write
    except ValueError as exc:
        sys.exit(str(exc))
    if not existing:
        _write(ids_path, chosen)
    _record(tracked, review, chosen, seed, after=existing, quotas=quotas)
    target = out / TO_TAG_NAME
    if target.exists() and _read(target):
        done = sum(1 for r in _read(target) if retag.tagged(r))
        sys.exit(
            f"{target} exists and is your tagging file ({done} tagged): it is not overwritten. "
            "Use 'merge' to bring its tags into the review file."
        )
    _write(target, gate)
    tagged = sum(1 for r in gate if (r.get("scope") or "").strip())
    print(f"{len(gate)} bodies -> {target} ({tagged} already tagged, {len(gate) - tagged} to tag)")
    print("by stratum:", dict(Counter(r["stratum"] for r in chosen)))


def cmd_merge(out: Path, file: str, *, gate_wins: bool = False) -> None:
    """Copy the tags made in the gate file into the current review file, as a NEW file (timestamped
    name): neither input is touched. A review row that already holds different tags keeps them
    (listed, exit 1) unless *gate_wins* takes the gate file's tags (listed). Every differing
    column goes to ``<stem>_conflicts_<timestamp>.csv``."""
    source = _current_review(out, file)
    review = retag.read_tag_file(source)
    try:
        gate = retag.read_tag_file(out / TO_TAG_NAME)
        merged, copied, conflicts, replaced = retag.merge_tags(review, gate, source_wins=gate_wins)
        differences = retag.tag_differences(review, gate)
    except retag.TagFileError as exc:
        sys.exit(str(exc))
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    target = out / f"{Path(file).stem}_merged_{stamp}.csv"
    _write(target, merged)
    print(
        f"{len(merged)} rows -> {target} (from {source.name}): {copied} rows copied from {TO_TAG_NAME}"
    )
    print("this is now the current review file (the newest *_merged_* file)")
    if differences:
        took = {item.split(" ")[0] for item in replaced}
        audit = [
            {**d, "resolution": "took_gate" if d["article_id"] in took else "kept_review"}
            for d in differences
        ]
        audit_path = out / f"{Path(file).stem}_conflicts_{stamp}.csv"
        _write(audit_path, audit)
        print(f"  every differing column -> {audit_path}")
    for item in replaced:
        print(f"  replaced by the gate file's tags (--gate-wins): {item}")
    for item in conflicts:
        print(f"  not overwritten (different tags in both files): {item}")
    if conflicts:
        sys.exit(
            f"{len(conflicts)} rows still differ: the merged file is NOT complete. "
            "Review them, then run again with --gate-wins to take the gate file's tags."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("draw", "merge"), default="draw")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--file", default=retag.REVIEW_NAME)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--tracked", type=Path, default=Path("gold"))
    parser.add_argument(
        "--gate-wins",
        action="store_true",
        help="merge: the gate file's tags replace the different ones in the review file",
    )
    args = parser.parse_args()
    if args.command == "merge":
        cmd_merge(args.out, args.file, gate_wins=args.gate_wins)
    else:
        cmd_draw(args.out, args.file, args.seed, args.tracked)


if __name__ == "__main__":
    main()
