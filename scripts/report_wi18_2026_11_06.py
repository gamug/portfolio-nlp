"""Turn the owner's tagged files of Work item 18 into the figures of the acceptance bars (PLAN D27,
D30, D32, D33).

    uv run scripts/report_wi18_2026_11_06.py scope|label-check|sentiment|faithfulness [--out data] [--gold] [--partial MIN]

* ``scope``: the rules' agreement with the owner's tags on the held-out 200 (``heldout_to_tag.csv``),
  re-weighted to the corpus with the sample sizes of ``heldout_strata.csv`` (D30), and the recall of
  each scope, gated only for a scope with at least 40 gold bodies (D33).
* ``label-check``: the precision of the `lead` / `continuation` labels (``segment_check_to_tag.csv``)
  and D27's rule: whether the `continuation` labels are dropped from training.
* ``sentiment``: agreement and negative precision / recall on the 180 (``sentiment_gold_to_tag.csv``
  with ``sentiment_gold_ids.csv``), re-weighted by the stratum sizes, bootstrap intervals (D32).
* ``faithfulness``: the share of faithful texts per extractor (``faithfulness_to_tag.csv`` with
  ``faithfulness_ids.csv``) and the model's gain over the baseline on the same company texts.

``label-check --gold --out gold`` reads the tracked file instead (it keeps the year);
``sentiment --gold --out gold`` reads ``sentiment_gold.csv`` (the marks and the draw's weights);
``faithfulness --gold --out gold`` reads ``faithfulness_gold.csv`` and ``faithfulness_coverage.csv``. ``faithfulness --partial MIN``
reports a file the owner has not finished, if its first MIN rows or more are marked in order (PLAN D32).

A file that is not completely and correctly marked stops the run, naming the rows. Nothing is
written; this only prints.
"""

from __future__ import annotations

import argparse
import functools
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sector_extra_2026_11_05 as extra

from news_nlp import attribution_model, attribution_units, gold_metrics, retag, triage
from news_nlp.gold_metrics import MarkedFileError

AGREEMENT_BAR = 0.80  # D32 sentiment overall agreement
NEGATIVE_BAR = 0.70  # D32 negative precision and recall
FAITHFUL_BAR = 0.80  # D32 extractor
SCOPE_BAR = 0.90  # D20 overall agreement (reported for the rules, gated for the classifier)
RECALL_BAR = 0.80  # D20, only for a scope with MIN_GOLD bodies (D33)
MIN_GOLD = 40
COVERAGE_NAME = gold_metrics.COVERAGE_NAME
KNOWN_EXTRACTORS = gold_metrics.KNOWN_EXTRACTORS


def scope_report(out: Path) -> list[str]:
    tagged = gold_metrics.read_marked(
        out / "heldout_to_tag.csv", "scope", triage.GOLD_SCOPES, ("article_id",)
    )
    stratum_of = {r["article_id"]: r["stratum"] for r in extra._read(out / "heldout_ids.csv")}
    unknown = sorted(r["article_id"] for r in tagged if r["article_id"] not in stratum_of)
    if unknown:
        raise MarkedFileError(f"heldout_to_tag.csv has ids not in heldout_ids.csv: {unknown[:5]}")
    gold_metrics.require_unique([r["article_id"] for r in tagged], "heldout_to_tag.csv")
    gold_metrics.require_same_keys(
        [r["article_id"] for r in tagged], list(stratum_of), "heldout_to_tag.csv"
    )
    populations = {r["stratum"]: int(r["frame"]) for r in extra._read(out / "heldout_strata.csv")}
    rows = [
        {
            "stratum": stratum_of[r["article_id"]],
            "predicted": triage.RULE_SCOPE_TO_GOLD[stratum_of[r["article_id"]]],
            "gold": r["scope"],
        }
        for r in tagged
    ]
    lines: list[str] = []
    overall = gold_metrics.stratified_report(
        rows, populations, label="single", stratum_key="stratum"
    )
    lines.append(
        f"agreement, weighted to the corpus: {overall.agreement} "
        f"(raw {overall.raw_agreement}; bar {SCOPE_BAR:.0%} is the classifier's, reported here)"
    )
    if overall.census:
        lines.append(f"(census, not resampled: {list(overall.census)})")
    for scope in triage.GOLD_SCOPES:
        r = gold_metrics.stratified_report(rows, populations, label=scope, stratum_key="stratum")
        gated = r.gold_count >= MIN_GOLD
        verdict = r.recall.judge(RECALL_BAR) if gated else "not gated (< 40 gold bodies, D33)"
        lines.append(f"{scope:>18}: {r.gold_count:3d} gold | recall {r.recall} | {verdict}")
    return lines


def label_check_report(out: Path, gold: bool = False) -> list[str]:
    """From the owner's file in *out*, or with *gold* from the tracked ``gold/`` file (``out`` is
    then its folder), which keeps the year and not the date."""
    rows = gold_metrics.read_marked(
        out / ("segment_check_gold.csv" if gold else "segment_check_to_tag.csv"),
        "correct",
        ("yes", "no"),
        required=("article_id", "year" if gold else "pub_date", "kind"),
    )
    report = gold_metrics.label_check_report(rows)
    lines = [f"{name:>18}: {figure}" for name, figure in report.items()]
    lead_only = gold_metrics.train_on_lead_only(report)
    undated = gold_metrics.rows_without_year(rows)
    if undated:
        lines.append(f"{undated} rows have no usable pub_date: they are not in the 2022+ figures")
    cont = report["continuation"]
    if cont.low <= gold_metrics.CONTINUATION_BAR <= cont.high:
        lines.append(
            f"note: the continuation interval {cont} straddles the "
            f"{gold_metrics.CONTINUATION_BAR:.0%} bar; the owner's rule is on the point estimate"
        )
    lines.append(
        "D27 rule: continuation precision "
        f"{report['continuation'].point:.1%} -> "
        + (
            f"at most {gold_metrics.CONTINUATION_BAR:.0%}: train on `lead` only"
            if lead_only
            else "above the bar: train on `lead` and `continuation`"
        )
    )
    return lines


def _period_lines(rows: list[dict[str, str]]) -> list[str]:
    """Per stratum: the texts drawn and how many are from 2022 on (the window of the cut, D34)."""
    lines = []
    for stratum in sorted({r["stratum"] for r in rows}):
        mine = [r for r in rows if r["stratum"] == stratum]
        late = sum((gold_metrics.year_of(r) or 0) >= gold_metrics.LATE_FROM_YEAR for r in mine)
        lines.append(
            f"  {stratum}: {len(mine)} drawn, {late} from {gold_metrics.LATE_FROM_YEAR} on"
        )
    undated = gold_metrics.rows_without_year(rows)
    if undated:
        lines.append(f"  {undated} texts have no usable pub_date")
    return lines


def _confusion_lines(rows: list[dict[str, str]], populations: dict[str, int]) -> list[str]:
    """Where the model and the owner part: a row per class the model predicted, a column per class
    the owner marked. Raw counts of the sample, then the same rows as shares re-weighted by stratum
    size (the sample draws the same number per class, the corpus does not), and per part."""
    classes = gold_metrics.SENTIMENT_CLASSES
    counts = Counter((r["predicted"], r["gold"]) for r in rows)
    lines = ["[all] model (rows) against owner (columns), raw counts: " + " | ".join(classes)]
    for predicted in classes:
        cells = " | ".join(f"{counts[(predicted, gold)]:>3}" for gold in classes)
        lines.append(f"[all]   {predicted:>8}: {cells}")
    size = Counter(r["stratum"] for r in rows)
    weighted: Counter[tuple[str, str]] = Counter()
    for r in rows:
        weighted[(r["predicted"], r["gold"])] += populations[r["stratum"]] / size[r["stratum"]]
    lines.append("[all] the same rows re-weighted by stratum size (share of each predicted class)")
    for predicted in classes:
        total = sum(weighted[(predicted, gold)] for gold in classes)
        cells = " | ".join(
            f"{100 * weighted[(predicted, gold)] / total:5.1f}%" if total else "  n/a "
            for gold in classes
        )
        lines.append(f"[all]   {predicted:>8}: {cells}")
    for part in sorted({r["part"] for r in rows}):
        mine = Counter((r["predicted"], r["gold"]) for r in rows if r["part"] == part)
        for predicted in classes:
            cells = " | ".join(f"{mine[(predicted, gold)]:>3}" for gold in classes)
            lines.append(f"[{part}]   {predicted:>8}: {cells}")
    return lines


def _shared_text_note(rows: list[dict[str, str]]) -> str:
    """The intervals treat every row as independent; a text two companies of one article share is
    the same words marked twice."""
    hashes = Counter(r["text_hash"] for r in rows if r.get("text_hash"))
    shared = sum(n for n in hashes.values() if n > 1)
    return (
        f"{shared} of {len(rows)} texts are shared by two companies or more of one article, so "
        "the intervals are a little narrow."
        if shared
        else "no text is shared by two companies."
    )


def _sentiment_block(
    label: str, rows: list[dict[str, str]], populations: dict[str, int]
) -> list[str]:
    rep = gold_metrics.sentiment_report(rows, populations)
    return [
        f"[{label}] drawn per stratum: {rep.drawn}"
        + (f" | (census, not resampled): {list(rep.census)}" if rep.census else ""),
        f"[{label}] overall agreement (weighted): {rep.agreement} -> "
        f"{rep.agreement.judge(AGREEMENT_BAR)} (raw {rep.raw_agreement})",
        f"[{label}] negative precision: {rep.negative_precision} -> "
        f"{rep.negative_precision.judge(NEGATIVE_BAR)}",
        f"[{label}] negative recall: {rep.negative_recall} -> "
        f"{rep.negative_recall.judge(NEGATIVE_BAR)}",
    ]


def _sentiment_rows_from_data(out: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    tagged = gold_metrics.read_marked(
        out / "sentiment_gold_to_tag.csv", "sentiment", gold_metrics.SENTIMENT_CLASSES
    )
    drawn = extra._read(out / "sentiment_gold_ids.csv")
    gold_metrics.require_unique([r["key"] for r in drawn], "sentiment_gold_ids.csv")
    ids = {r["key"]: r for r in drawn}
    keys = [f"{r['article_id']}#{r['ticker'] or r['kind']}" for r in tagged]
    gold_metrics.require_unique(keys, "sentiment_gold_to_tag.csv")
    for key in keys:
        if key not in ids:
            raise MarkedFileError(f"{key} is not in sentiment_gold_ids.csv")
    gold_metrics.require_same_keys(keys, list(ids), "sentiment_gold_to_tag.csv")
    rows = []
    for key, r in zip(keys, tagged, strict=True):
        row = ids[key]
        rows.append(
            {
                "predicted": row["predicted"],
                "stratum": row.get("stratum") or row["predicted"],
                "part": row.get("part") or "checked",
                "gold": r["sentiment"],
                "pub_date": r.get("pub_date", ""),
                "text_hash": row.get("text_hash", ""),
            }
        )
    populations = {(r.get("stratum") or r["predicted"]): int(r["population"]) for r in ids.values()}
    return rows, populations


def _sentiment_rows_from_gold(
    out: Path, predictions: dict[str, str] | None = None
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """From the tracked ``sentiment_gold.csv`` in *out*: the marks and the draw's weights, no
    ``data/``. The sample must be the one the draw took (``gold_metrics.check_sentiment_design``).
    With *predictions* (key -> class) the model compared with the owner is that one, not the one
    the sample was drawn by: the strata (the weights) stay those of the draw."""
    path = out / "sentiment_gold.csv"
    tagged = gold_metrics.read_marked(
        path,
        "sentiment",
        gold_metrics.SENTIMENT_CLASSES,
        required=("key", "part", "stratum", "predicted", "population", "text_hash", "pub_date"),
    )
    gold_metrics.require_unique([r["key"] for r in tagged], path.name)
    populations = gold_metrics.check_sentiment_design(tagged)
    if predictions is not None:
        unscored = [r["key"] for r in tagged if r["key"] not in predictions]
        if unscored:
            raise MarkedFileError(f"{len(unscored)} gold texts have no prediction: {unscored[:3]}")
    rows = [
        {
            "predicted": r["predicted"] if predictions is None else predictions[r["key"]],
            "stratum": r["stratum"],
            "part": r["part"],
            "gold": r["sentiment"],
            "pub_date": r["pub_date"],
            "text_hash": r["text_hash"],
        }
        for r in tagged
    ]
    return rows, populations


def sentiment_report(
    out: Path, gold: bool = False, predictions: dict[str, str] | None = None
) -> list[str]:
    """From the owner's files in *out*, or with *gold* from the tracked ``gold/sentiment_gold.csv``
    (with *predictions*, of another model than the one the sample was drawn by)."""
    rows, populations = (
        _sentiment_rows_from_gold(out, predictions) if gold else _sentiment_rows_from_data(out)
    )
    lines = [
        "weights: the size of each stratum in the pool of the owner's tagged bodies "
        "(gold/category_gold_v2.csv), not in the corpus or in the 2022+ window (D32)"
    ]
    lines += _period_lines(rows)
    lines += _sentiment_block("all", rows, populations)
    lines += _confusion_lines(rows, populations)
    parts = sorted({r["part"] for r in rows})
    if len(parts) > 1:
        for part in parts:
            mine = [r for r in rows if r["part"] == part]
            pops = {k: v for k, v in populations.items() if k in {r["stratum"] for r in mine}}
            lines += _sentiment_block(part, mine, pops)
        lines.append(
            "part `extractor` = the texts the baseline / model wrote for narrative multi bodies; "
            "`checked` = summaries and structured multi texts"
        )
    lines.append(
        "bars are judged on the whole interval; 'not decided' still goes to the cut with the "
        "interval printed (D34); 'undefined' is a figure that does not exist (nothing was "
        "predicted), not an interval that straddles the bar. The intervals resample texts inside each stratum, as if each were "
        "independent: " + _shared_text_note(rows)
    )
    return lines


def _faithfulness_block(
    label: str,
    marks: dict[str, bool],
    owners: dict[str, list[str]],
    omitted: dict[str, list[str]],
) -> list[str]:
    """The per-extractor shares and the gain over the baseline for one set of texts. *omitted* is,
    per extractor, the companies (``<article>#<ticker>``) it named but wrote no text for: each
    counts as an unfaithful text, in the share and in the gain."""
    if not marks and not any(omitted.values()):
        return [f"[{label}] no texts"]
    report = gold_metrics.faithfulness_report(marks, owners, omitted=omitted)
    lines = []
    for name, fig in report.items():
        texts = [k for k, owner in owners.items() if name in owner]
        right = sum(marks[k] for k in texts)
        skipped = len(omitted.get(name, ()))
        lines.append(
            f"[{label}] {name:>10}: {fig} -> {fig.judge(FAITHFUL_BAR)} (the bar: of the companies "
            f"named, those with a faithful text; {right} faithful of {len(texts)} texts + "
            f"{skipped} companies left without one)"
        )
        if texts:
            only = gold_metrics.wilson(right, len(texts))
            lines.append(
                f"[{label}] {name:>10}: of the texts it wrote, {only} (informative; omissions not "
                "counted)"
            )
    for challenger, reference in (
        ("model", "baseline"),
        ("units", "baseline"),
        ("units", "model"),
        ("units2", "units"),
        ("units2", "baseline"),
        ("units3", "units2"),
        ("units3", "units"),
    ):
        if {challenger, reference} <= set(report):
            lines += _gain_lines(label, marks, owners, omitted, challenger, reference)
    return lines


def _gain_lines(
    label: str,
    marks: dict[str, bool],
    owners: dict[str, list[str]],
    omitted: dict[str, list[str]],
    challenger: str = "model",
    reference: str = "baseline",
) -> list[str]:
    """*challenger* minus *reference* (by default model minus baseline) on the companies at least
    one of them wrote a text for; a company one
    skipped counts as unfaithful for it, and a company both skipped says nothing on which is
    better (it stays in the shares only)."""
    pairs: dict[str, dict[str, bool]] = {}
    for key, names in owners.items():
        pair = key.rsplit("#", 1)[0]
        for name in names:
            pairs.setdefault(pair, {})[name] = marks[key]
    written = set(pairs)
    for name, pair_keys in omitted.items():
        for pair in pair_keys:
            if pair in written:
                pairs[pair][name] = False
    both = {p: v for p, v in pairs.items() if {challenger, reference} <= set(v)}
    if not both:
        return []
    gain = gold_metrics.paired_difference(
        [v[challenger] for v in both.values()],
        [v[reference] for v in both.values()],
        clusters=[gold_metrics.body_of_key(p) for p in both],
    )
    bodies = len({gold_metrics.body_of_key(p) for p in both})
    return [
        f"[{label}] {challenger} minus {reference} on {len(both)} company texts of {bodies} "
        f"bodies: {gain} -> {gold_metrics.gain_verdict(gain)}"
    ]


def _faithfulness_marks(out: Path, partial: int | None) -> tuple[list[dict[str, str]], list[str]]:
    """The owner's marked rows and, for a partial report, the line that says so."""
    path = out / "faithfulness_to_tag.csv"
    if partial is None:
        return gold_metrics.read_marked(
            path, "faithful", ("yes", "no"), ("article_id", "review_order")
        ), []
    tagged, total = gold_metrics.read_marked_prefix(path, "faithful", ("yes", "no"), partial)
    if len(tagged) == total:
        return tagged, []  # finished: it is the full report, nothing provisional
    return tagged, [
        f"PARTIAL: {len(tagged)} of {total} texts marked (the first {len(tagged)} of the shuffled "
        "order); the companies an extractor left without a text are counted in full, so the "
        "share is a floor. Provisional: the full figure replaces it."
    ]


def faithfulness_report(out: Path, partial: int | None = None) -> list[str]:
    """With *partial* (the least number of marked rows the rule accepts, PLAN D32) the owner may not
    have finished: the marked rows must be ``review_order`` 1..k, the omissions are counted in full
    (conservative) and the output says it is partial."""
    tagged, note = _faithfulness_marks(out, partial)
    ids = {r["review_order"]: r for r in extra._read(out / "faithfulness_ids.csv")}
    if ids and "pub_date" not in next(iter(ids.values())):
        raise MarkedFileError("faithfulness_ids.csv lacks pub_date: the 2022+ split needs it")
    unknown = sorted(r["review_order"] for r in tagged if r["review_order"] not in ids)
    if unknown:
        raise MarkedFileError(
            f"faithfulness_to_tag.csv has rows not in the ids file: {unknown[:5]}"
        )
    gold_metrics.require_unique([r["review_order"] for r in tagged], "faithfulness_to_tag.csv")
    gold_metrics.require_same_keys(
        [r["review_order"] for r in tagged],
        [k for k in ids if partial is None or int(k) <= len(tagged)],
        "faithfulness_to_tag.csv",
    )
    coverage = gold_metrics.read_coverage(out / COVERAGE_NAME, hint=". Draw the file again.")
    drawn = {r["article_id"] for r in extra._read(out / "faithfulness_bodies.csv")}
    foreign = sorted({c["article_id"] for c in coverage if c["article_id"] not in drawn})
    if foreign:
        raise MarkedFileError(f"{COVERAGE_NAME} has bodies the draw does not have: {foreign[:5]}")
    marks: dict[str, bool] = {}
    owners: dict[str, list[str]] = {}
    dates: dict[str, dict[str, str]] = {}
    for r in tagged:
        row = ids[r["review_order"]]
        marks[row["key"]] = r["faithful"] == "yes"
        owners[row["key"]] = row["extractors"].split("|")
        dates[row["key"]] = row
    notes = list(note)
    try:
        footer_only = gold_metrics.read_footer_only(
            out / gold_metrics.FOOTER_ONLY_NAME, retag.code_revision(attribution_units)
        )
    except gold_metrics.FooterRulesChanged as changed:
        footer_only = []
        notes.append(f"{changed}; the second reading of the bar is skipped")
    notes += _stale_lines(out, {n for names in owners.values() for n in names})
    return _faithfulness_lines(notes, marks, owners, dates, coverage, footer_only)


def _footer_only_lines(
    label: str,
    marks: dict[str, bool],
    owners: dict[str, list[str]],
    omitted: dict[str, list[str]],
    footer_only: Sequence[dict[str, str]],
) -> list[str]:
    """``units`` on the bar with the companies named only in a footer counted as unfaithful."""
    extra_pairs = [f"{c['article_id']}#{c['ticker']}" for c in footer_only]
    if not extra_pairs or "units" not in {n for o in owners.values() for n in o}:
        return []
    strict = {**omitted, "units": [*omitted.get("units", []), *extra_pairs]}
    fig = gold_metrics.faithfulness_report(marks, owners, omitted=strict)["units"]
    return [
        f"[{label}] units: with the {len(extra_pairs)} companies named only in a footer counted "
        f"as unfaithful (second reading of the bar): {fig} -> {fig.judge(FAITHFUL_BAR)}"
    ]


STALE_SHOWN = 6  # marked rows named in the one-line staleness note


def _current_code(name: str) -> str | None:
    """The code part of the revision of extractor *name* as it is now (the baseline's whole
    revision), or ``None`` for an extractor this branch does not define. No model is needed."""
    if name == "baseline":
        return retag.extractor_revision("baseline")
    modules = {
        "model": attribution_model.CODE_MODULES,
        "units": attribution_model.UNITS_CODE_MODULES,
    }
    return retag.code_revision(*modules[name]) if name in modules else None


def _checked_with_current_code(name: str, revision: str) -> bool | None:
    """Whether the check of *name* was made with the code it has now (``None``: cannot tell)."""
    code = _current_code(name)
    if code is None:
        return None
    return revision == code if name == "baseline" else revision.endswith(f"@{code}")


def _stale_lines(out: Path, marked: set[str]) -> list[str]:
    """What ``draw_faithfulness --stale`` last found, per extractor, in a few lines: the marked rows
    on texts the current code no longer writes (the figures above include those marks), the
    extractors of the marks it never checked, and a warning for each check that was made with other
    code than the current one (the code part of a revision needs no model; the check does)."""
    entries = gold_metrics.read_stale(out / gold_metrics.STALE_NAME)
    if not entries:
        return ["no staleness check recorded: run draw_faithfulness --stale (needs the model)"]
    lines: list[str] = []
    rows: list[str] = []
    for name, entry in sorted(entries.items()):
        if _checked_with_current_code(name, entry["revision"]) is False:
            lines.append(
                f"warning: the staleness check of {name} ({entry['date']}) was made with other "
                "code than the current one: run it again"
            )
        rows += entry["stale"]
    said = (
        f"staleness check ({', '.join(f'{n} {e["date"]}' for n, e in sorted(entries.items()))}): "
    )
    if rows:
        shown = ", ".join(rows[:STALE_SHOWN]) + (", ..." if len(rows) > STALE_SHOWN else "")
        said += f"{len(rows)} marked rows are on texts the current code no longer writes "
        said += f"({shown}); the figures include their marks"
    else:
        said += "no marked row is on a text the current code no longer writes"
    missing = sorted(marked - set(entries))
    if missing:
        said += f"; never checked: {', '.join(missing)}"
    return [*lines, said]


def _faithfulness_lines(
    note: list[str],
    marks: dict[str, bool],
    owners: dict[str, list[str]],
    dates: dict[str, dict[str, str]],
    coverage: list[dict[str, str]],
    footer_only: Sequence[dict[str, str]] = (),
) -> list[str]:
    """The figures from marks by text key, who wrote each text, the row (with ``pub_date``) of
    each key and the companies an extractor left without a text. Shared by the owner's file and
    the tracked gold, so both give the same figures. *footer_only* are the companies a body names
    only in its footer, which ``units`` is not asked about: a second reading of the bar counts
    them as unfaithful for ``units`` (the owner decides which reading is the bar's)."""
    omitted: dict[str, list[str]] = {}
    omitted_late: dict[str, list[str]] = {}
    for c in coverage:
        pair = f"{c['article_id']}#{c['ticker']}"
        omitted.setdefault(c["extractor"], []).append(pair)
        if (gold_metrics.year_of(c) or 0) >= gold_metrics.LATE_FROM_YEAR:
            omitted_late.setdefault(c["extractor"], []).append(pair)
    late = {
        k
        for k, row in dates.items()
        if (gold_metrics.year_of(row) or 0) >= gold_metrics.LATE_FROM_YEAR
    }
    undated = sum(gold_metrics.year_of(row) is None for row in dates.values())
    lines = [*note, "the intervals are a bootstrap over bodies, not over texts (D32)"]
    if undated:
        lines.append(f"{undated} texts have no usable pub_date: they are not in the 2022+ block")
    lines += _faithfulness_block("all", marks, owners, omitted)
    lines += _footer_only_lines("all", marks, owners, omitted, footer_only)
    late_marks = {k: v for k, v in marks.items() if k in late}
    late_owners = {k: v for k, v in owners.items() if k in late}
    lines += _faithfulness_block("2022+", late_marks, late_owners, omitted_late)
    late_footer = [
        c for c in footer_only if (gold_metrics.year_of(c) or 0) >= gold_metrics.LATE_FROM_YEAR
    ]
    lines += _footer_only_lines("2022+", late_marks, late_owners, omitted_late, late_footer)
    return lines


def faithfulness_gold_report(out: Path) -> list[str]:
    """The same figures from the tracked ``faithfulness_gold.csv`` and ``faithfulness_coverage.csv``
    in *out* (``--gold --out gold``): they need neither the git-ignored ``data/`` nor the draw."""
    rows = gold_metrics.read_marked(
        out / "faithfulness_gold.csv",
        "faithful",
        ("yes", "no"),
        required=("key", "article_id", "ticker", "extractors", "pub_date"),
    )
    gold_metrics.require_unique([r["key"] for r in rows], "faithfulness_gold.csv")
    coverage = gold_metrics.read_coverage(out / COVERAGE_NAME)
    owned = sorted({n for r in rows for n in r["extractors"].split("|")} - set(KNOWN_EXTRACTORS))
    if owned:
        raise MarkedFileError(
            f"faithfulness_gold.csv has extractors the draw does not have: {owned}"
        )
    gold_metrics.check_coverage_against_draw(
        coverage,
        drawn_bodies=None,  # the list of bodies is not tracked: a body with no text has no row
        texts=[(r["article_id"], r["ticker"], r["extractors"].split("|")) for r in rows],
    )
    marks = {r["key"]: r["faithful"] == "yes" for r in rows}
    owners = {r["key"]: r["extractors"].split("|") for r in rows}
    note = [
        f"from the tracked gold: {len(rows)} texts, {len(coverage)} companies left without a text "
        f"({COVERAGE_NAME})"
    ]
    return _faithfulness_lines(note, marks, owners, {r["key"]: r for r in rows}, coverage)


COMMANDS: dict[str, Callable[[Path], list[str]]] = {
    "scope": scope_report,
    "label-check": label_check_report,
    "sentiment": sentiment_report,
    "faithfulness": faithfulness_report,
}


def _at_least_one(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument(
        "--gold",
        action="store_true",
        help="label-check / sentiment / faithfulness: read the tracked gold/ files in --out",
    )
    parser.add_argument(
        "--partial",
        type=_at_least_one,
        metavar="MIN",
        help="faithfulness: the file is not finished; accept the first rows marked, at least MIN",
    )
    args = parser.parse_args()
    if args.gold and args.command not in ("label-check", "sentiment", "faithfulness"):
        parser.error("--gold is for label-check, sentiment and faithfulness only")
    if args.gold and args.partial is not None:
        parser.error("--partial reads the owner's file in progress: not with --gold")
    if args.partial is not None and args.command != "faithfulness":
        parser.error("--partial is for faithfulness only")
    command = COMMANDS[args.command]
    if args.command == "label-check":
        command = functools.partial(label_check_report, gold=args.gold)
    elif args.command == "sentiment":
        command = functools.partial(sentiment_report, gold=args.gold)
    elif args.command == "faithfulness":
        command = (
            faithfulness_gold_report
            if args.gold
            else functools.partial(faithfulness_report, partial=args.partial)
        )
    try:
        print("\n".join(command(args.out)))
    except MarkedFileError as exc:
        sys.exit(str(exc))
    except FileNotFoundError as exc:
        sys.exit(
            f"{exc.filename} not found: it is written by the draw script of that set "
            "(scripts/draw_*; see the docstring of this script) or is the owner's tagged file"
        )


if __name__ == "__main__":
    main()
