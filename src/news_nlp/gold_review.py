"""Write the gold-review CSVs without destroying the owner's work (Work item 18 P2).

The review file is what a human fills in (``human_label``, ``human_notes``) over days; the key
file joins it back to the LLM's and the classifier's labels. :func:`write_review_files` refuses to
overwrite a review file that already holds a label (``human_label`` by default; ``tag_column`` for
another file, e.g. ``scope`` in the gold-v2 re-tag), and replaces the review and key files together
(both written to temporary names first, renamed only when both are complete), so a failure or a
second run never leaves files of different runs paired.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HUMAN_COLUMNS = ("human_label", "human_notes")


class TaggedFileError(RuntimeError):
    """The review file already holds human labels, or cannot be read back (another delimiter or
    code page): it is not overwritten."""


def tagged_rows(path: Path, column: str = "human_label") -> int:
    """How many rows of the CSV at *path* have a non-empty *column* (0 if the file is absent).
    Raises :class:`TaggedFileError` if the file exists but has no such column or is not UTF-8:
    a file re-saved by a spreadsheet must not read as 'untagged'."""
    if not path.exists():
        return 0
    _require_readable(path, column)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return sum(bool((row.get(column) or "").strip()) for row in csv.DictReader(fh))


def _require_readable(path: Path, tag_column: str) -> None:
    """An existing review file must still parse into its tag column. A file re-saved by a
    spreadsheet with another delimiter or code page has no such column, reads as 'untagged' and
    would be overwritten: refuse instead."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            header = next(csv.reader(fh), [])
    except UnicodeDecodeError as exc:
        raise TaggedFileError(
            f"{path} is not UTF-8 ({exc.reason}); refusing to overwrite it"
        ) from exc
    if tag_column not in header:
        raise TaggedFileError(
            f"{path} has no {tag_column!r} column (saved with another delimiter or code page?); "
            "refusing to overwrite it"
        )


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_review_files(
    out_dir: Path,
    review: Sequence[Mapping[str, Any]],
    key: Sequence[Mapping[str, Any]],
    *,
    review_name: str = "category_gold_review.csv",
    key_name: str = "category_gold_review_key.csv",
    tag_column: str = "human_label",
) -> None:
    """Write *review* and *key* into *out_dir*, both or neither. Raises :class:`TaggedFileError`
    if the review file exists and has rows with a non-empty *tag_column* (move or copy it first)."""
    review_path, key_path = out_dir / review_name, out_dir / key_name
    done = tagged_rows(review_path, tag_column)
    if done:
        raise TaggedFileError(
            f"{review_path} already holds {done} human labels; refusing to overwrite it"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_review, tmp_key = review_path.with_suffix(".tmp"), key_path.with_suffix(".tmp")
    try:
        _write(tmp_review, review)
        _write(tmp_key, key)
        os.replace(tmp_review, review_path)
        os.replace(tmp_key, key_path)
    finally:
        tmp_review.unlink(missing_ok=True)
        tmp_key.unlink(missing_ok=True)
