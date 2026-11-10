"""Figures from the owner's tagged files of Work item 18 (PLAN D27, D32): intervals, the
re-weighted sentiment figures of the stratified 180, the label check and the faithfulness check.

The 180-text sentiment gold is drawn 60 per *predicted* class, so its raw agreement is not the
production agreement: every row weighs ``population / drawn`` of its stratum, and the interval of
each figure is a bootstrap that resamples inside each stratum. Everything is seeded.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from news_nlp import triage

Z95 = 1.96
RESAMPLES = 2000
LATE_FROM_YEAR = 2022  # the backtest window (D34)
CONTINUATION_BAR = 0.80  # D27: at most this share right -> train on `lead` only
SENTIMENT_CLASSES = ("positive", "negative", "neutral")
#: Texts the sentiment draw takes per predicted class, by part (``draw_sentiment_gold``: 40 of the
#: summaries and structured company texts, 20 of the extractor's texts): 180 in all.
SENTIMENT_DESIGN = {"checked": 40, "extractor": 20}
COVERAGE_NAME = "faithfulness_coverage.csv"  # companies an extractor left without a text (D32)
COVERAGE_COLUMNS = ("article_id", "ticker", "extractor", "pub_date")
# Companies a body names that an extractor of the ``units`` family is not asked for a text of (T-148b4,
# T-148b5): the ones named only in the footer (``units``, ``units2``, ``units3``) and the ones named
# only by the first word of their name (``units2``, ``units3``, the guard). They are not in the
# coverage, so the share of the bar leaves them out; the report also gives the bar with them counted
# as unfaithful, and (once the owner has read them) with only the ones that are real companies.
# The file is ``{"rules": <revision of the rules>, "companies": [<coverage rows>]}``: the companies
# depend on the rules, so the report skips a file written by other rules.
EXEMPT_NAME = "faithfulness_exempt.json"
# the companies above, one row each, for the owner to read: is it a company? (``real_company``)
EXEMPT_READ_NAME = "faithfulness_exempt_to_read.csv"
EXEMPT_READ_COLUMNS = (
    "article_id",
    "ticker",
    "company",
    "why",
    "extractors",
    "title",
    "text",
    "spanish_title",
    "spanish_text",
    "real_company",
)
# what ``draw_faithfulness --stale`` found: the marked rows the current code no longer writes
STALE_NAME = "faithfulness_stale.json"
# what each extractor writes now, by company (the same run): the texts as hashes, and the companies it
# writes nothing for
NOW_NAME = "faithfulness_now.json"
KNOWN_EXTRACTORS = ("baseline", "model", "units", "units2", "units3")


class MarkedFileError(ValueError):
    """An owner's file is not usable: a missing column, an unknown value or an unmarked row."""


@dataclass(frozen=True)
class Figure:
    """A point estimate with its 95% interval."""

    point: float
    low: float
    high: float

    def judge(self, bar: float) -> str:
        """``pass`` when the whole interval clears *bar* (its low end is at or above it), ``fail``
        when the whole interval is below it, else ``not decided``: an interval that straddles the
        bar or only touches it from below (``high == bar``) is not a pass or a fail. The point
        estimate decides nothing alone (D32: reported with the interval, the cut still goes, D34).
        ``undefined`` when the interval does not exist (NaN: a precision with no text predicted, a
        share of no texts): that is not an interval that straddles the bar, and it must not read
        as "not decided", which still goes to the cut."""
        if math.isnan(self.low) or math.isnan(self.high):
            return "undefined"
        if self.low >= bar:
            return "pass"
        if self.high < bar:
            return "fail"
        return "not decided"

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"


def wilson(successes: int, n: int, z: float = Z95) -> Figure:
    """Wilson score interval of a proportion; ``0 of 0`` has no estimate (NaN)."""
    low, high = triage.wilson(successes, n, z)
    return Figure(successes / n if n else math.nan, low, high)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def _bootstrap(
    strata: Mapping[str, list[Mapping[str, str]]],
    statistic: Callable[[Mapping[str, list[Mapping[str, str]]]], float],
    seed: int,
    resamples: int,
    fixed: frozenset[str] = frozenset(),
) -> Figure:
    """Seeded percentile interval of *statistic*, resampling inside each stratum. A stratum in
    *fixed* was drawn whole (a census): it has no sampling error and is not resampled."""
    point = statistic(strata)
    rng = random.Random(seed)  # noqa: S311 -- a reproducible interval, not security
    draws: list[float] = []
    for _ in range(resamples):
        sample = {k: v if k in fixed else [rng.choice(v) for _ in v] for k, v in strata.items()}
        value = statistic(sample)
        if not math.isnan(value):
            draws.append(value)
    if not draws or math.isnan(point):
        return Figure(point, math.nan, math.nan)
    return Figure(point, _percentile(draws, 0.025), _percentile(draws, 0.975))


@dataclass(frozen=True)
class SentimentReport:
    agreement: Figure
    negative_precision: Figure
    negative_recall: Figure
    raw_agreement: Figure  # not re-weighted: for comparison only
    drawn: dict[str, int]
    census: tuple[str, ...] = ()  # strata drawn whole: held fixed in the bootstrap


def sentiment_report(
    rows: Sequence[Mapping[str, str]],
    populations: Mapping[str, int],
    *,
    seed: int = 20261106,
    resamples: int = RESAMPLES,
) -> SentimentReport:
    """*rows* carry ``predicted`` and ``gold`` (the owner's class), and ``stratum`` when the draw has
    more than one part (otherwise ``predicted`` is the stratum). *populations* is the size of each
    stratum in the pool the draw came from."""
    key = "stratum" if rows and "stratum" in rows[0] else "predicted"
    r = stratified_report(
        rows, populations, label="negative", stratum_key=key, seed=seed, resamples=resamples
    )
    return SentimentReport(r.agreement, r.precision, r.recall, r.raw_agreement, r.drawn, r.census)


@dataclass(frozen=True)
class StratifiedReport:
    agreement: Figure
    precision: Figure  # of *label*: of the rows predicted *label*, the share gold calls *label*
    recall: Figure  # of *label*: of the rows gold calls *label*, the share predicted *label*
    raw_agreement: Figure
    drawn: dict[str, int]
    gold_count: int  # rows the owner calls *label* in the sample (D33: under 40, not gated)
    census: tuple[str, ...] = ()  # strata drawn whole (drawn >= population): not resampled


def stratified_report(
    rows: Sequence[Mapping[str, str]],
    populations: Mapping[str, int],
    *,
    label: str,
    stratum_key: str,
    seed: int = 20261106,
    resamples: int = RESAMPLES,
) -> StratifiedReport:
    """Agreement and the precision / recall of *label* of a draw stratified on *stratum_key*, each
    row weighing ``population / drawn`` of its stratum; the intervals are a seeded bootstrap that
    resamples inside each stratum, except a stratum drawn whole (a census: no sampling error there).
    *rows* carry ``predicted`` and ``gold``."""
    strata: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        strata.setdefault(row[stratum_key], []).append(row)
    unknown = sorted(set(strata) - set(populations))
    if unknown:
        raise MarkedFileError(f"no population for the strata {unknown}")

    def weighted(
        s: Mapping[str, list[Mapping[str, str]]], keep: Callable[[Mapping[str, str]], bool]
    ) -> float:
        return sum(populations[k] / len(v) for k, v in s.items() for r in v if keep(r))

    def agreement(s: Mapping[str, list[Mapping[str, str]]]) -> float:
        total = weighted(s, lambda r: True)
        return weighted(s, lambda r: r["gold"] == r["predicted"]) / total if total else math.nan

    def precision(s: Mapping[str, list[Mapping[str, str]]]) -> float:
        predicted = weighted(s, lambda r: r["predicted"] == label)
        hit = weighted(s, lambda r: r["predicted"] == label and r["gold"] == label)
        return hit / predicted if predicted else math.nan

    def recall(s: Mapping[str, list[Mapping[str, str]]]) -> float:
        actual = weighted(s, lambda r: r["gold"] == label)
        hit = weighted(s, lambda r: r["predicted"] == label and r["gold"] == label)
        return hit / actual if actual else math.nan

    raw = wilson(sum(r["gold"] == r["predicted"] for r in rows), len(rows))
    census = frozenset(k for k, v in strata.items() if len(v) >= populations[k])
    return StratifiedReport(
        agreement=_bootstrap(strata, agreement, seed, resamples, census),
        precision=_bootstrap(strata, precision, seed + 1, resamples, census),
        recall=_bootstrap(strata, recall, seed + 2, resamples, census),
        raw_agreement=raw,
        drawn={k: len(v) for k, v in strata.items()},
        gold_count=sum(r["gold"] == label for r in rows),
        census=tuple(sorted(census)),
    )


def paired_stratified_gain(
    rows: Sequence[Mapping[str, str]],
    populations: Mapping[str, int],
    *,
    candidate: str = "predicted",
    baseline: str = "baseline",
    metric: str = "agreement",
    label: str = "negative",
    stratum_key: str = "stratum",
    seed: int = 20261106,
    resamples: int = RESAMPLES,
) -> Figure:
    """The candidate's *metric* minus the baseline's, on the same rows of a stratified draw, with the
    weights and the resampling of :func:`stratified_report` (the same resample feeds both, so the
    interval is of the difference). *metric* is ``agreement``, ``precision`` or ``recall`` (of
    *label*). *rows* carry ``gold``, the candidate's column and the baseline's. Every row of the gold
    is its own body, so the bootstrap is inside the strata, as the report's. The result goes to
    :func:`gain_verdict`: ``better`` only when the whole interval is above 0. A resample in which the
    metric is undefined (no row predicted *label*, for the precision) is left out, as in the report's
    own intervals, so the interval of a precision gain can come from fewer resamples than asked when
    the baseline predicts few of *label*; a point that is undefined makes the whole figure NaN
    (``undefined`` for the verdict)."""
    if metric not in {"agreement", "precision", "recall"}:
        raise ValueError(f"metric {metric!r}: one of agreement, precision, recall")
    strata: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        strata.setdefault(row[stratum_key], []).append(row)
    unknown = sorted(set(strata) - set(populations))
    if unknown:
        raise MarkedFileError(f"no population for the strata {unknown}")

    def value(s: Mapping[str, list[Mapping[str, str]]], column: str) -> float:
        def weighted(keep: Callable[[Mapping[str, str]], bool]) -> float:
            return sum(populations[k] / len(v) for k, v in s.items() for r in v if keep(r))

        if metric == "agreement":
            total = weighted(lambda r: True)
            return weighted(lambda r: r["gold"] == r[column]) / total if total else math.nan
        hit = weighted(lambda r: r[column] == label and r["gold"] == label)
        if metric == "precision":
            predicted = weighted(lambda r: r[column] == label)
            return hit / predicted if predicted else math.nan
        actual = weighted(lambda r: r["gold"] == label)
        return hit / actual if actual else math.nan

    def gain(s: Mapping[str, list[Mapping[str, str]]]) -> float:
        return value(s, candidate) - value(s, baseline)

    census = frozenset(k for k, v in strata.items() if len(v) >= populations[k])
    return _bootstrap(strata, gain, seed, resamples, census)


def _cluster_ratio(clusters: Sequence[tuple[float, int]], seed: int, resamples: int) -> Figure:
    """The ratio ``sum(first) / sum(second)`` over *clusters*, with a bootstrap interval that
    resamples whole clusters: the units that are independent (the bodies), not their texts."""
    total = sum(n for _, n in clusters)
    if not total:
        return Figure(math.nan, math.nan, math.nan)
    point = sum(x for x, _ in clusters) / total
    rng = random.Random(seed)  # noqa: S311 -- a reproducible interval, not security
    draws: list[float] = []
    for _ in range(resamples):
        sample = rng.choices(clusters, k=len(clusters))
        n = sum(c for _, c in sample)
        if n:
            draws.append(sum(x for x, _ in sample) / n)
    return Figure(point, _percentile(draws, 0.025), _percentile(draws, 0.975))


def paired_difference(
    a: Sequence[bool],
    b: Sequence[bool],
    *,
    clusters: Sequence[str] | None = None,
    seed: int = 20261106,
    resamples: int = RESAMPLES,
) -> Figure:
    """Share of items *a* gets right minus the share *b* gets right, on the same items (the
    extractor against the baseline). With *clusters* (the body of each item) the interval is a
    bootstrap over the bodies, because the company texts of one body are not independent;
    without it, over the items."""
    if len(a) != len(b) or not a:
        raise ValueError("two non-empty lists of the same length are needed")
    if clusters is not None and len(clusters) != len(a):
        raise ValueError("one cluster per item is needed")
    diffs = [int(x) - int(y) for x, y in zip(a, b, strict=True)]
    names = list(clusters) if clusters is not None else [str(i) for i in range(len(diffs))]
    grouped: dict[str, list[int]] = {}
    for name, d in zip(names, diffs, strict=True):
        grouped.setdefault(name, []).append(d)
    return _cluster_ratio([(sum(v), len(v)) for v in grouped.values()], seed, resamples)


def read_marked(
    path: Path, column: str, allowed: Sequence[str], required: Sequence[str] = ("article_id",)
) -> list[dict[str, str]]:
    """The rows of an owner's tagging file with *column* checked: every row marked, every value one
    of *allowed* (case and spaces ignored). Raises :class:`MarkedFileError` naming the first rows
    at fault, so a half-tagged or mistyped file is never turned into a figure."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    if not rows:
        raise MarkedFileError(f"{path.name} has no rows")
    missing = [c for c in (column, *required) if c not in rows[0]]
    if missing:
        raise MarkedFileError(f"{path.name} lacks the columns {missing}")
    empty, wrong = [], []
    for r in rows:
        value = (r[column] or "").strip().lower()
        r[column] = value
        if not value:
            empty.append(r["article_id"])
        elif value not in allowed:
            wrong.append(f"{r['article_id']}={value!r}")
    if empty or wrong:
        raise MarkedFileError(
            f"{path.name}: {len(empty)} rows without {column} (first: {empty[:5]}), "
            f"{len(wrong)} with a value outside {list(allowed)} (first: {wrong[:5]})"
        )
    return rows


def read_marked_prefix(
    path: Path,
    column: str,
    allowed: Sequence[str],
    minimum: int,
    required: Sequence[str] = ("article_id", "review_order"),
) -> tuple[list[dict[str, str]], int]:
    """The marked rows of a file the owner has not finished, and the number of rows in the file.
    Allowed only when the marked rows are exactly ``review_order`` 1..k (the rows are shuffled, so a
    prefix is a random sample; a hand-picked subset is not) and k is at least *minimum*. A value
    outside *allowed* stops the run as in :func:`read_marked`."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    if not rows:
        raise MarkedFileError(f"{path.name} has no rows")
    missing = [c for c in (column, *required) if c not in rows[0]]
    if missing:
        raise MarkedFileError(f"{path.name} lacks the columns {missing}")
    marked = [r for r in rows if (r[column] or "").strip()]
    for r in marked:
        r[column] = r[column].strip().lower()
        if r[column] not in allowed:
            raise MarkedFileError(
                f"{path.name}: {r['review_order']}={r[column]!r} is not in {list(allowed)}"
            )
    orders = [r["review_order"].strip() for r in marked]
    if not all(o.isdigit() for o in orders):
        raise MarkedFileError(
            f"{path.name}: review_order must be a whole number in every marked row"
        )
    if minimum < 1:
        raise MarkedFileError("a partial report needs a minimum of at least 1 marked row")
    if sorted(int(o) for o in orders) != list(range(1, len(marked) + 1)):
        raise MarkedFileError(
            f"{path.name}: a partial report needs the rows marked in order, review_order 1 to k "
            "with no gap (the rows are shuffled, so that prefix is a random sample)"
        )
    if len(marked) < minimum:
        raise MarkedFileError(f"{path.name}: {len(marked)} rows marked, the rule needs {minimum}")
    return marked, len(rows)


def read_coverage(path: Path, *, hint: str = "") -> list[dict[str, str]]:
    """The companies left without a text, checked to have the columns the faithfulness figures read,
    only known extractors and no repeated (article, ticker, extractor): a repeat would count one
    company twice as unfaithful. One reader for the draw's file, the builder and the tracked gold."""
    if not path.exists():
        raise MarkedFileError(
            f"{path.name} not found: it lists the companies an extractor left without a text, "
            f"which the D32 share counts as unfaithful{hint}"
        )
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != COVERAGE_COLUMNS:
            raise MarkedFileError(f"{path.name} must have the columns {COVERAGE_COLUMNS}")
        rows = list(reader)
    foreign = sorted({r["extractor"] for r in rows} - set(KNOWN_EXTRACTORS))
    if foreign:
        raise MarkedFileError(f"{path.name} has extractors the draw does not have: {foreign}")
    require_unique([f"{r['article_id']}#{r['ticker']}#{r['extractor']}" for r in rows], path.name)
    return rows


class ExemptRulesChanged(MarkedFileError):
    """The exempt-companies file was written by other rules than the current ones."""


def read_exempt(path: Path, rules: str) -> list[dict[str, str]]:
    """The companies an extractor of the ``units`` family is not asked about, from the file the
    draw wrote with ``--exempt``. Refused when it was written by other *rules* (the revision of
    the footer and guard code): the companies would be those of a different cut."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("rules") != rules:
        raise ExemptRulesChanged(
            f"{path.name} was written by the rules {data.get('rules')!r}, not the current "
            f"{rules!r}: draw it again (--exempt)"
        )
    rows: list[dict[str, str]] = data["companies"]
    if any(tuple(r) != COVERAGE_COLUMNS for r in rows):
        raise MarkedFileError(f"{path.name} rows must have the columns {COVERAGE_COLUMNS}")
    return rows


EXEMPT_READ_OWNER_COLUMNS = ("notes",)  # the owner's free text, which the figures never read


def exempt_read_header_ok(header: Sequence[str]) -> bool:
    """The reading file has every column of ``EXEMPT_READ_COLUMNS`` (a spreadsheet may reorder them)
    and, besides, only the owner's own ``notes``."""
    return (
        set(EXEMPT_READ_COLUMNS)
        <= set(header)
        <= {*EXEMPT_READ_COLUMNS, *EXEMPT_READ_OWNER_COLUMNS}
    )


def read_exempt_marks(path: Path) -> dict[str, str]:
    """The owner's reading of the exempt companies: ``<article>#<ticker>`` -> ``yes`` (a real
    company), ``no`` (not a company: a person, a place, a common word) or ``""`` (not read)."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not exempt_read_header_ok(reader.fieldnames or ()):
            raise MarkedFileError(f"{path.name} must have the columns {EXEMPT_READ_COLUMNS}")
        rows = list(reader)
    marks: dict[str, str] = {}
    for row in rows:
        mark = row["real_company"].strip().lower()
        if mark not in ("", "yes", "no"):
            raise MarkedFileError(f"{path.name}: real_company must be yes or no, not {mark!r}")
        marks[f"{row['article_id']}#{row['ticker']}"] = mark
    return marks


class StaleEntry(TypedDict):
    """What the last ``--stale`` found for one extractor."""

    date: str
    revision: str  # the extractor's revision when it was checked
    stale: list[str]  # marked rows (``<key> (<extractor>)``) on texts it no longer writes
    new: list[str]  # texts it writes now that no row holds


class NowEntry(TypedDict):
    """What the last ``--stale`` run found each extractor writes now."""

    date: str
    revision: str
    texts: dict[str, str]  # ``<article_id>#<ticker>`` -> the hash of the text written now
    without: list[str]  # ``<article_id>#<ticker>`` it is asked about and writes nothing for


def read_now(path: Path) -> dict[str, NowEntry]:
    """The result of the last ``--stale`` per extractor (``{}`` if the file does not exist)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("extractors") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        raise MarkedFileError(f"{path.name} is not in the current format: run --stale again")
    out: dict[str, NowEntry] = {}
    for name, entry in entries.items():
        ok = (
            isinstance(entry, dict)
            and isinstance(entry.get("date"), str)
            and isinstance(entry.get("revision"), str)
            and isinstance(entry.get("texts"), dict)
            and isinstance(entry.get("without"), list)
        )
        if not ok:
            raise MarkedFileError(
                f"{path.name}: the entry of {name!r} is malformed: run --stale again"
            )
        out[name] = NowEntry(
            date=entry["date"],
            revision=entry["revision"],
            texts=entry["texts"],
            without=entry["without"],
        )
    return out


def read_stale(path: Path) -> dict[str, StaleEntry]:
    """The result of ``draw_faithfulness --stale`` per extractor (``{}`` if it was never run). The
    file keeps the last check of each extractor: a later run of other extractors does not erase it."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("extractors") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        raise MarkedFileError(f"{path.name} is not in the current format: run --stale again")
    out: dict[str, StaleEntry] = {}
    for name, entry in entries.items():
        ok = (
            isinstance(entry, dict)
            and isinstance(entry.get("date"), str)
            and isinstance(entry.get("revision"), str)
            and isinstance(entry.get("stale"), list)
            and isinstance(entry.get("new"), list)
        )
        if not ok:
            raise MarkedFileError(
                f"{path.name}: the entry of {name!r} is malformed: run --stale again"
            )
        out[name] = StaleEntry(
            date=entry["date"], revision=entry["revision"], stale=entry["stale"], new=entry["new"]
        )
    return out


def merge_stale(
    old: Mapping[str, StaleEntry], ran: Mapping[str, StaleEntry]
) -> dict[str, dict[str, StaleEntry]]:
    """The file's content after a run: the entries of the extractors that were run replace the old
    ones, the others stay as they were."""
    return {"extractors": {**old, **ran}}


class PairWriteError(OSError):
    """A replace failed after another file of the group was already replaced and could not be put back."""


def render_csv(
    fields: Sequence[str], rows: Sequence[Mapping[str, str]], *, lineterminator: str = "\n"
) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator=lineterminator)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)  # not possible on Windows: the replace stays atomic there
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_rows(path: Path) -> list[dict[str, str]]:
    """The rows of a draw's helper file (``faithfulness_bodies.csv``); a clean error when absent."""
    if not path.exists():
        raise MarkedFileError(
            f"{path.name} not found: it is written by the draw script of that set"
        )
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def check_coverage_against_draw(
    coverage: Sequence[Mapping[str, str]],
    *,
    drawn_bodies: set[str] | None,
    texts: Sequence[tuple[str, str, Sequence[str]]],
) -> None:
    """A company left without a text must belong to a drawn body (when the list of bodies is at hand)
    and must not have a text from the same extractor: both would shift the D32 share."""
    if drawn_bodies is not None:
        foreign = sorted({c["article_id"] for c in coverage} - drawn_bodies)
        if foreign:
            raise MarkedFileError(
                f"{COVERAGE_NAME} has bodies the draw does not have: {foreign[:5]}"
            )
    written = {(a, t, e) for a, t, owners in texts for e in owners}
    both = sorted(
        f"{c['article_id']}#{c['ticker']}#{c['extractor']}"
        for c in coverage
        if (c["article_id"], c["ticker"], c["extractor"]) in written
    )
    if both:
        raise MarkedFileError(
            f"{COVERAGE_NAME} lists {len(both)} companies that the same extractor wrote a text "
            f"for (first {both[0]}): a company with a text is not left without one"
        )


def _put_back(
    replaced: Sequence[Path], old: Mapping[Path, bytes | None], temps: Mapping[Path, Path]
) -> list[str]:
    """Restore the files already replaced; the names of those that could not be restored."""
    failed = []
    for path in replaced:
        try:
            before = old[path]
            if before is None:
                path.unlink()
            else:
                temps[path].write_bytes(before)
                os.replace(temps[path], path)
        except OSError:
            failed.append(path.name)
    return failed


def write_pair_atomically(files: Mapping[Path, str]) -> None:
    """Write every file to a temporary neighbour first (flushed to disk), then replace them: a write
    that fails (disk full, file open in Excel) leaves all the old files as they were. If a replace
    fails after another file was replaced, the replaced ones are put back from their old bytes; when
    that fails too, ``PairWriteError`` names the files that are new. Temporary files of a killed
    earlier run are deleted first."""
    temps = {path: path.with_name(f"{path.name}.{os.getpid()}.tmp") for path in files}
    for path in files:
        for stale in path.parent.glob(f"{path.name}.*.tmp"):
            stale.unlink(missing_ok=True)
    old = {path: path.read_bytes() if path.exists() else None for path in files}
    replaced: list[Path] = []
    try:
        for path, text in files.items():
            with temps[path].open("w", newline="", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
        for path, tmp in temps.items():
            os.replace(tmp, path)
            replaced.append(path)
    except OSError:
        left_new = _put_back(replaced, old, temps)
        if left_new:
            raise PairWriteError(
                f"{', '.join(left_new)} was replaced and could not be put back; "
                "the other files of the group are the old ones"
            ) from None
        raise
    finally:
        for tmp in temps.values():
            tmp.unlink(missing_ok=True)
    for directory in {p.parent for p in files}:
        with contextlib.suppress(
            OSError
        ):  # the files are in place; only the rename's durability is open
            _fsync_dir(directory)


def _check_sentiment_row(r: Mapping[str, str], design: Mapping[str, int]) -> None:
    if r["part"] not in design:
        raise MarkedFileError(f"{r['key']}: part {r['part']!r} is not one of {list(design)}")
    if r["predicted"] not in SENTIMENT_CLASSES:
        raise MarkedFileError(f"{r['key']}: predicted {r['predicted']!r} is not a class")
    if r["stratum"] != f"{r['part']}/{r['predicted']}":
        raise MarkedFileError(
            f"{r['key']}: stratum {r['stratum']!r} is not {r['part']}/{r['predicted']}"
        )


def check_sentiment_design(
    rows: Sequence[Mapping[str, str]], design: Mapping[str, int] | None = None
) -> dict[str, int]:
    """The sentiment draw's rows (``key``, ``part``, ``predicted``, ``stratum``, ``population``) are
    the sample the figures assume: a known part, a predicted class, the stratum named
    ``<part>/<predicted>``, one whole-number size per stratum, and in every stratum of every part the
    number the draw takes (all of them when the stratum is smaller). Returns the size of each
    stratum; raises :class:`MarkedFileError` naming what is off, so a short or edited file never gives
    a figure."""
    design = SENTIMENT_DESIGN if design is None else design
    sizes: dict[str, set[str]] = {}
    drawn: Counter[str] = Counter()
    for r in rows:
        _check_sentiment_row(r, design)
        sizes.setdefault(r["stratum"], set()).add(r["population"])
        drawn[r["stratum"]] += 1
    split = sorted(s for s, v in sizes.items() if len(v) > 1)
    if split:
        raise MarkedFileError(f"the stratum size is not the same on every row of {split}")
    try:
        population = {s: int(next(iter(v))) for s, v in sizes.items()}
    except ValueError:
        raise MarkedFileError("population is not a whole number on every row") from None
    off = []
    for part, per_class in design.items():
        for label in SENTIMENT_CLASSES:
            stratum = f"{part}/{label}"
            want = min(per_class, population.get(stratum, per_class))
            if drawn[stratum] != want:
                off.append(f"{stratum}: {drawn[stratum]} rows, the draw takes {want}")
    if off:
        raise MarkedFileError(f"the sample is not the one drawn: {'; '.join(off)}")
    return population


def require_unique(keys: Sequence[str], what: str) -> None:
    """Stop when a key repeats: a repeated key would give two rows one weight and one mark."""
    repeated = sorted(k for k, n in Counter(keys).items() if n > 1)
    if repeated:
        raise MarkedFileError(f"{what}: repeated keys {repeated[:5]}")


def require_same_keys(found: Sequence[str], expected: Sequence[str], what: str) -> None:
    """Stop when the owner's file does not hold exactly the rows of the draw. A deleted row would
    silently change the weights (``population / rows tagged``) or drop a text from a share."""
    missing = sorted(set(expected) - set(found))
    if missing:
        raise MarkedFileError(
            f"{what}: {len(missing)} rows of the draw are not in the tagged file "
            f"(first: {missing[:5]})"
        )


def year_of(row: Mapping[str, str]) -> int | None:
    """The publication year of a row (``pub_date`` as ``YYYY-...``, or the ``year`` column of a gold
    file, which keeps the year and not the date), ``None`` when it has none."""
    head = (row.get("pub_date") or row.get("year") or "").strip()[:4]
    return int(head) if head.isdigit() else None


def rows_without_year(rows: Sequence[Mapping[str, str]]) -> int:
    """Rows the ``2022+`` figures cannot place because their ``pub_date`` is empty or malformed."""
    return sum(year_of(r) is None for r in rows)


def label_check_report(rows: Sequence[Mapping[str, str]]) -> dict[str, Figure]:
    """Precision of the automatic labels per kind (``lead`` / ``continuation``) from the marked
    ``correct`` column (yes / no), with Wilson intervals; ``<kind> 2022+`` repeats it for the rows
    published from 2022 on (``pub_date``), the window the backtest cut serves. A file without the
    ``pub_date`` column is refused: its ``2022+`` figures would silently be ``0 of 0``."""
    if rows and "pub_date" not in rows[0] and "year" not in rows[0]:
        raise MarkedFileError("the label-check rows lack the column 'pub_date' (or 'year')")
    out: dict[str, Figure] = {}
    for kind in ("lead", "continuation"):
        part = [r for r in rows if r["kind"] == kind]
        out[kind] = wilson(sum(r["correct"] == "yes" for r in part), len(part))
        late = [r for r in part if (year_of(r) or 0) >= LATE_FROM_YEAR]
        out[f"{kind} 2022+"] = wilson(sum(r["correct"] == "yes" for r in late), len(late))
    return out


def train_on_lead_only(report: Mapping[str, Figure]) -> bool:
    """D27's rule, fixed before the owner read the sample: the `continuation` labels are dropped
    from training when at most :data:`CONTINUATION_BAR` of them are right. The rule is on the point
    estimate, as the owner fixed it; the interval is printed beside it, not used. With no
    `continuation` row marked there is no estimate and no rule: that raises, it never defaults."""
    if math.isnan(report["continuation"].point):
        raise MarkedFileError(
            "no `continuation` row in the label check: the D27 rule has no figure"
        )
    return report["continuation"].point <= CONTINUATION_BAR


def gain_verdict(gain: Figure) -> str:
    """D32: the extractor is ``better`` than the baseline when the whole interval of its gain is
    above 0, ``not better`` when the whole interval is below 0, else ``not decided``. The edges
    are those of :meth:`Figure.judge`: an interval that only touches 0 decides nothing; ``undefined``
    when the interval does not exist (NaN)."""
    if math.isnan(gain.low) or math.isnan(gain.high):
        return "undefined"
    if gain.low > 0:
        return "better"
    if gain.high < 0:
        return "not better"
    return "not decided"


def body_of_key(key: str) -> str:
    """The body (article) of a faithfulness text key ``<article>#<ticker>#<n>``."""
    return key.split("#", 1)[0]


def faithfulness_report(
    marks: Mapping[str, bool],
    extractors_of: Mapping[str, Sequence[str]],
    *,
    cluster_of: Callable[[str], str] = body_of_key,
    omitted: Mapping[str, Sequence[str]] | None = None,
    seed: int = 20261106,
    resamples: int = RESAMPLES,
) -> dict[str, Figure]:
    """Share of faithful texts per extractor. *marks* is the owner's yes / no by text key and
    *extractors_of* says which extractors produced each text (one text can be both's). The point
    is the share of texts; the interval is a bootstrap over bodies (*cluster_of*), because the
    company texts of one body are not independent (D32). *omitted* maps an extractor to the keys
    of the companies it named but wrote no text for; each counts as an unfaithful text, so an
    extractor cannot raise its share by leaving the hard companies out."""
    omitted = omitted or {}
    names = sorted({e for v in extractors_of.values() for e in v} | set(omitted))
    out: dict[str, Figure] = {}
    for name in names:
        grouped: dict[str, list[int]] = {}
        for key, owners in extractors_of.items():
            if name in owners:
                grouped.setdefault(cluster_of(key), []).append(int(marks[key]))
        for key in omitted.get(name, ()):
            grouped.setdefault(cluster_of(key), []).append(0)
        out[name] = _cluster_ratio([(sum(v), len(v)) for v in grouped.values()], seed, resamples)
    return out
