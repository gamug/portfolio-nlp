"""Put the owner's marks on the training summaries in the tracked file (Work item 18, T-148c1, D35).

    uv run scripts/build_sentiment_train_2026_11_07.py [--out data] [--gold gold/sentiment_train.csv]
        [--overwrite-marks]

Reads ``data/sentiment_train_to_tag.csv`` (the marks the owner wrote) and
``data/sentiment_train_ids.csv`` (the draw: the texts, the model's prediction, the block, the batch
and the slice) and writes ``gold/sentiment_train.csv`` (``retag.SENTIMENT_TRAIN_COLUMNS``): the mark
and the ids only, no news text; the ``ticker`` and ``company`` of the row are in it because the mark is
about that company (D35). The two files are paired by key (``<article_id>#summary``). The run
stops unless

* the marked file holds exactly the texts of the draw, once each;
* every row is marked positive / negative / neutral;
* each pair agrees on the article and the kind, and the text the owner read is the text the draw
  recorded (its hash; a text that a spreadsheet made safe with a leading apostrophe is compared
  without it, and the escaped text is never what is kept);
* ``pub_date`` is an ISO date;
* the sample is the one drawn (``check_design``): the same number of each predicted class, blocks
  of ``sentiment_train.BLOCK`` rows with the same number of each, and every batch and slice what D35
  says.

**Every problem is reported at once** (a spreadsheet round trip fails in the same way on many rows:
an encoding change in the text, a date in another format): the error names each kind of problem
with its count, the first keys, and for a text that differs the position of the first difference
and whether it looks like an encoding change. Warnings (printed, nothing stops): a class under
``MIN_CLASS_SHARE`` of the training slice, and ``MAX_RUN`` or more rows in a row with the same
mark (a fill-down in the spreadsheet); the rows are shuffled by block, so a long run is not natural.

``text_hash``, the block, the batch and the slice are copied from the ids, never recomputed;
``pub_date`` is the one in the file the owner tagged. A gold file that already exists is the
reference: if a mark or a column the training depends on (``PROTECTED``) would change, or a row would
go missing, the run stops (``--overwrite-marks`` to accept: how a corrected mark gets in). The file
is replaced as a whole.

The training reads the marks from here and the texts from ``data/sentiment_train_ids.csv`` (by key,
checked by ``text_hash``): that file is the only copy of the texts the owner read, and a summary is
generated, so **back it up** (the cut's chunks hold the same summaries by body hash).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from news_nlp import gold_metrics, retag, sentiment_train

TO_TAG_NAME = "sentiment_train_to_tag.csv"
IDS_NAME = "sentiment_train_ids.csv"
CLASSES = gold_metrics.SENTIMENT_CLASSES
TRAIN_COLUMNS = retag.SENTIMENT_TRAIN_COLUMNS
#: A ticker may be blank (26 of the 450 rows: the company has none the draw knew); the company may not,
#: it is what the mark is about.
NEEDED_FROM_THE_DRAW = tuple(
    c for c in TRAIN_COLUMNS if c not in ("pub_date", "sentiment", "notes", "ticker")
)
NEEDED_FROM_THE_OWNER = ("article_id", "kind", "text", "pub_date", "sentiment", "notes")
#: The company the mark is about (the label is anchored to it, D35) and its ticker.
COMPANY_COLUMNS = ("ticker", "company")
#: What a rebuild may not change in a tracked row without ``--overwrite-marks``.
PROTECTED = (
    "sentiment",
    "text_hash",
    "revision",
    "predicted",
    "block",
    "batch",
    "slice",
    "pub_date",
)
SHOWN = 5  # keys named per kind of problem
MIN_CLASS_SHARE = 0.10
MAX_RUN = 20
#: What the owner is likeliest to type instead of a class (the builder does not translate); the
#: hint is given when the first refused mark is one of these.
SPANISH_MARKS = frozenset({"positivo", "negativo", "neutro", "positiva", "negativa"})
#: Text a UTF-8 file turns into when it is read or saved in a Windows code page.
MOJIBAKE = ("Ã", "â€", "Â", "�")


def key_of(row: dict[str, str]) -> str:
    return f"{row['article_id']}#{row['kind']}"


def changed_marks(before: list[dict[str, str]], after: list[dict[str, str]]) -> list[str]:
    """Keys of the existing file that the new rows would change (a ``PROTECTED`` column) or drop."""
    now = {r["key"]: r for r in after}
    # the company and ticker are protected once the file has them (the file of 2026-11-08 first did not)
    protected = PROTECTED + tuple(c for c in COMPANY_COLUMNS if before and c in before[0])
    return [
        r["key"]
        for r in before
        if r["key"] not in now or any(now[r["key"]].get(c) != r.get(c) for c in protected)
    ]


def check_design(rows: list[dict[str, str]]) -> None:
    """The sample is the one ``draw_sentiment_train`` makes: equal predicted classes, blocks that hold
    the same number of each, and the batch and slice of every block as D35 says."""
    per_block = sentiment_train.BLOCK // len(CLASSES)
    counts = Counter(r["predicted"] for r in rows)
    if set(counts) != set(CLASSES) or len(set(counts.values())) != 1:
        raise gold_metrics.MarkedFileError(f"the predicted classes are not equal: {dict(counts)}")
    blocks: dict[int, Counter[str]] = {}
    for r in rows:
        try:
            number = int(r["block"])
        except ValueError:
            raise gold_metrics.MarkedFileError(
                f"{r['key']}: block {r['block']!r} is not a number"
            ) from None
        blocks.setdefault(number, Counter())[r["predicted"]] += 1
        want = (str(sentiment_train.batch_of(number)), sentiment_train.slice_of(number))
        if (r["batch"], r["slice"]) != want:
            raise gold_metrics.MarkedFileError(
                f"{r['key']}: block {number} is batch {want[0]}, {want[1]}; the file says "
                f"batch {r['batch']}, {r['slice']}"
            )
    if sorted(blocks) != list(range(1, len(blocks) + 1)):
        raise gold_metrics.MarkedFileError("the blocks are not 1..N in a row")
    odd = [n for n, c in blocks.items() if set(c.values()) != {per_block} or set(c) != set(CLASSES)]
    if odd:
        raise gold_metrics.MarkedFileError(f"blocks without {per_block} of each class: {odd[:5]}")
    if len({r["body_hash"] for r in rows}) != len(rows):
        raise gold_metrics.MarkedFileError("two rows share a body")


def read_the_same_text(marked: str, drawn_hash: str) -> bool:
    """Whether the text in the owner's file is the drawn one: its hash, or its hash without the one
    apostrophe a spreadsheet-safe file puts before a text that starts like a formula."""
    return retag.text_hash(marked) == drawn_hash or (
        marked.startswith("'") and retag.text_hash(marked[1:]) == drawn_hash
    )


def looks_like_an_encoding_change(drawn: str, marked: str) -> bool:
    """The marked text has the marks of UTF-8 read as a Windows code page (or replacement
    characters) that the drawn text does not have, or lost non-ASCII characters to ``?``."""
    if any(m in marked and m not in drawn for m in MOJIBAKE):
        return True
    return marked.count("?") > drawn.count("?") and not drawn.isascii()


def first_difference(drawn: str, marked: str) -> str:
    """Where two texts part: the position and a few characters of each."""
    drawn, marked = drawn.strip(), marked.strip()
    at = next((i for i, (a, b) in enumerate(zip(drawn, marked, strict=False)) if a != b), None)
    if at is None:
        at = min(len(drawn), len(marked))
    return f"at character {at}: drawn {drawn[at : at + 12]!r}, marked {marked[at : at + 12]!r}"


class Problems:
    """What is wrong with the owner's file, by kind, so that one error names all of it."""

    def __init__(self) -> None:
        self.found: dict[str, list[str]] = {}
        self.notes: dict[str, str] = {}

    def add(self, kind: str, key: str, note: str = "") -> None:
        self.found.setdefault(kind, []).append(key)
        if note:
            self.notes.setdefault(kind, note)

    def lines(self) -> list[str]:
        return [
            f"{len(keys)} {'row' if len(keys) == 1 else 'rows'} {kind} (first: {keys[:SHOWN]})"
            + (f"; {self.notes[kind]}" if kind in self.notes else "")
            for kind, keys in self.found.items()
        ]


def check_keys(
    problems: Problems,
    keys: list[str],
    by_key: dict[str, dict[str, str]],
    ids: list[dict[str, str]],
) -> None:
    """The owner's file holds the drawn texts, each once: a repeated key, a drawn text that is not
    in the file and a row that is not in the draw are named, not stopped on."""
    for key in sorted(k for k, n in Counter(keys).items() if n > 1):
        problems.add("whose key is repeated in the tagged file", key)
    for key in (r["key"] for r in ids if r["key"] not in by_key):
        problems.add("of the draw missing from the tagged file", key)
    for key in sorted(set(by_key) - {r["key"] for r in ids}):
        problems.add("of the tagged file missing from the draw", key)


def check_row(problems: Problems, drawn: dict[str, str], marked: dict[str, str]) -> str:
    """One pair: what is wrong with the mark, the ids, the text and the date goes to *problems*;
    returns the mark (normalised) and leaves the date to the caller."""
    key = drawn["key"]
    mark = (marked["sentiment"] or "").strip().lower()
    if not mark:
        problems.add("without a mark", key)
    elif mark not in CLASSES:
        hint = (
            "; write the English word: positive, negative or neutral (the builder does not "
            "translate)"
            if mark in SPANISH_MARKS
            else ""
        )
        problems.add(f"with a mark outside {list(CLASSES)}", key, f"for example {mark!r}{hint}")
    if any(marked[f] != drawn[f] for f in ("article_id", "kind")):
        problems.add("whose article or kind is not the drawn one", key)
    if not read_the_same_text(marked["text"], drawn["text_hash"]):
        hint = (
            "; it looks like an encoding change: save the file as CSV UTF-8"
            if looks_like_an_encoding_change(drawn["text"], marked["text"])
            else ""
        )
        problems.add(
            "whose text is not the text the draw recorded (edited, or saved in another encoding)",
            key,
            first_difference(drawn["text"], marked["text"]) + hint,
        )
    pub_date = (marked.get("pub_date") or "").strip()
    try:
        date.fromisoformat(pub_date[:10])
    except ValueError:
        problems.add(
            "with a pub_date that is not ISO (YYYY-MM-DD...); a spreadsheet may have rewritten it",
            key,
            f"for example {pub_date!r}",
        )
    return mark


def build(marks: list[dict[str, str]], ids: list[dict[str, str]]) -> list[dict[str, str]]:
    """The rows of ``gold/sentiment_train.csv``, in the order of the draw. Raises
    ``MarkedFileError`` naming every problem found (the rows missing or repeated included, and the
    problems of the rows that are there) when the marks and the draw are not the same texts."""
    gold_metrics.require_unique([r["key"] for r in ids], IDS_NAME)  # the draw's own file
    keys = [key_of(r) for r in marks]
    by_key: dict[str, dict[str, str]] = {}
    for key, row in zip(keys, marks, strict=True):
        by_key.setdefault(key, row)  # a repeated key is reported; its first row is checked
    problems = Problems()
    check_keys(problems, keys, by_key, ids)
    out = []
    for drawn in ids:
        marked = by_key.get(drawn["key"])
        if marked is None:
            continue  # reported above
        if empty := [f for f in NEEDED_FROM_THE_DRAW if not (drawn.get(f) or "").strip()]:
            problems.add(f"of {IDS_NAME} without {empty}", drawn["key"])
            continue
        mark = check_row(problems, drawn, marked)
        out.append(
            {
                **{f: drawn[f] for f in TRAIN_COLUMNS if f in drawn},
                "pub_date": (marked.get("pub_date") or "").strip(),
                "sentiment": mark,
                "notes": marked["notes"],
            }
        )
    if problems.found:
        raise gold_metrics.MarkedFileError(f"{TO_TAG_NAME}:\n  " + "\n  ".join(problems.lines()))
    check_design(out)
    return out


def warnings(rows: list[dict[str, str]]) -> list[str]:
    """What looks wrong in the marks without making the build fail: a class that is almost
    absent from the training slice, and a long run of the same mark in the order of the file."""
    found = []
    train = [r for r in rows if r["slice"] == sentiment_train.TRAIN]
    for mark, n in sorted(Counter(r["sentiment"] for r in train).items()):
        if n < MIN_CLASS_SHARE * len(train):
            found.append(f"only {n} of {len(train)} training marks are {mark}")
    run, longest, at = 0, 0, 0
    for i, r in enumerate(rows):
        run = run + 1 if i and r["sentiment"] == rows[i - 1]["sentiment"] else 1
        if run > longest:
            longest, at = run, i
    if longest >= MAX_RUN:
        found.append(
            f"{longest} rows in a row (to row {at + 1}) have the mark {rows[at]['sentiment']}: "
            "a fill-down in the spreadsheet?"
        )
    for kind in sorted({"positive", "negative", "neutral"} - {r["sentiment"] for r in train}):
        found.append(f"no training mark is {kind}")
    return found


def summary(rows: list[dict[str, str]]) -> list[str]:
    """What the owner marked, by slice, and the model that balanced the draw against the owner (a
    row per predicted class, a column per mark): context for the training, not a figure."""
    lines = []
    for kind in (sentiment_train.TRAIN, sentiment_train.VALIDATION):
        mine = [r for r in rows if r["slice"] == kind]
        lines.append(
            f"{kind}: {len(mine)} texts, marks {dict(Counter(r['sentiment'] for r in mine))}"
        )
    both = Counter((r["predicted"], r["sentiment"]) for r in rows)
    lines.append(
        "the model that balanced the draw (rows) against the owner (columns): "
        + " | ".join(CLASSES)
    )
    for predicted in CLASSES:
        cells = " | ".join(f"{both[(predicted, m)]:>3}" for m in CLASSES)
        lines.append(f"  {predicted:>8}: {cells}")
    return lines


def read_marks(path: Path) -> list[dict[str, str]]:
    """The rows of the owner's file, which must be UTF-8 and have the columns the build reads. A
    file in a Windows code page (Excel's plain "CSV") and one separated by ``;`` (a Spanish-locale
    Excel) each get a message that says what to do, not a traceback or a list of "missing"
    columns that are there."""
    if not path.exists():
        raise gold_metrics.MarkedFileError(
            f"{path.name} not found: save your tagged copy at {path} (CSV UTF-8, comma-separated); "
            "the draw writes only the untagged one"
        )
    data = path.read_bytes()
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise gold_metrics.MarkedFileError(
            f"{path.name} is not UTF-8 (the first byte that is not: line "
            f"{data[: exc.start].count(b'\n') + 1}, byte {exc.start}): save it again as "
            "CSV UTF-8 (in Excel: Save as, CSV UTF-8)"
        ) from None
    marks = gold_metrics.read_rows(path)
    missing = [c for c in NEEDED_FROM_THE_OWNER if marks and c not in marks[0]]
    if not marks or missing:
        # the header line itself: the rows may hold commas (a text), which would add fields
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        header = lines[0] if lines else ""  # an empty file has none
        separator = next((s for s in (";", "\t") if "," not in header and s in header), None)
        if separator:
            name = "a semicolon" if separator == ";" else "a tab"
            raise gold_metrics.MarkedFileError(
                f"{path.name} looks separated by {name}, not a comma: save it again with commas "
                "(in Excel: Save as, CSV UTF-8)"
            )
        why = "has no rows" if not marks else f"lacks the columns {missing}"
        raise gold_metrics.MarkedFileError(f"{path.name} {why}")
    return marks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--gold", type=Path, default=Path("gold/sentiment_train.csv"))
    parser.add_argument(
        "--overwrite-marks",
        action="store_true",
        help="accept that marks, text hashes or rows of the existing file change or go",
    )
    args = parser.parse_args()
    try:
        marks = read_marks(args.out / TO_TAG_NAME)
        rows = build(marks, gold_metrics.read_rows(args.out / IDS_NAME))
    except gold_metrics.MarkedFileError as exc:
        sys.exit(str(exc))
    if args.gold.exists() and not args.overwrite_marks:
        with args.gold.open(encoding="utf-8-sig", newline="") as fh:
            moved = changed_marks(list(csv.DictReader(fh)), rows)
        if moved:
            sys.exit(
                f"{len(moved)} rows of {args.gold} would change or go missing (first {moved[0]}): "
                "the file is the reference; run with --overwrite-marks if this is intended"
            )
    args.gold.parent.mkdir(parents=True, exist_ok=True)
    try:
        gold_metrics.write_pair_atomically(
            {args.gold: gold_metrics.render_csv(TRAIN_COLUMNS, retag.spreadsheet_safe(rows))}
        )
    except OSError as exc:
        sys.exit(f"cannot write {args.gold} ({exc}); it is as it was")
    print(f"{args.gold}: {len(rows)} rows")
    print("\n".join(summary(rows)))
    for line in warnings(rows):
        print(f"warning: {line}")


if __name__ == "__main__":
    main()
