"""Turn the owner's tagged files of Work item 18 into the figures of the acceptance bars (PLAN D27,
D30, D32, D33).

    uv run scripts/report_wi18_2026_11_06.py scope|label-check|sentiment|faithfulness [--out data]

* ``scope``: the rules' agreement with the owner's tags on the held-out 200 (``heldout_to_tag.csv``),
  re-weighted to the corpus with the sample sizes of ``heldout_strata.csv`` (D30), and the recall of
  each scope, gated only for a scope with at least 40 gold bodies (D33).
* ``label-check``: the precision of the `lead` / `continuation` labels (``segment_check_to_tag.csv``)
  and D27's rule: whether the `continuation` labels are dropped from training.
* ``sentiment``: agreement and negative precision / recall on the 180 (``sentiment_gold_to_tag.csv``
  with ``sentiment_gold_ids.csv``), re-weighted by the stratum sizes, bootstrap intervals (D32).
* ``faithfulness``: the share of faithful texts per extractor (``faithfulness_to_tag.csv`` with
  ``faithfulness_ids.csv``) and the model's gain over the baseline on the same company texts.

A file that is not completely and correctly marked stops the run, naming the rows. Nothing is
written; this only prints.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sector_extra_2026_11_05 as extra

from news_nlp import gold_metrics, triage
from news_nlp.gold_metrics import MarkedFileError

AGREEMENT_BAR = 0.80  # D32 sentiment overall agreement
NEGATIVE_BAR = 0.70  # D32 negative precision and recall
FAITHFUL_BAR = 0.80  # D32 extractor
SCOPE_BAR = 0.90  # D20 overall agreement (reported for the rules, gated for the classifier)
RECALL_BAR = 0.80  # D20, only for a scope with MIN_GOLD bodies (D33)
MIN_GOLD = 40


def scope_report(out: Path) -> list[str]:
    tagged = gold_metrics.read_marked(
        out / "heldout_to_tag.csv", "scope", triage.GOLD_SCOPES, ("article_id",)
    )
    stratum_of = {r["article_id"]: r["stratum"] for r in extra._read(out / "heldout_ids.csv")}
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
    for scope in triage.GOLD_SCOPES:
        r = gold_metrics.stratified_report(rows, populations, label=scope, stratum_key="stratum")
        gated = r.gold_count >= MIN_GOLD
        verdict = r.recall.judge(RECALL_BAR) if gated else "not gated (< 40 gold bodies, D33)"
        lines.append(f"{scope:>18}: {r.gold_count:3d} gold | recall {r.recall} | {verdict}")
    return lines


def label_check_report(out: Path) -> list[str]:
    rows = gold_metrics.read_marked(
        out / "segment_check_to_tag.csv",
        "correct",
        ("yes", "no"),
        required=("article_id", "pub_date", "kind"),
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


def sentiment_report(out: Path) -> list[str]:
    tagged = gold_metrics.read_marked(
        out / "sentiment_gold_to_tag.csv", "sentiment", gold_metrics.SENTIMENT_CLASSES
    )
    ids = {r["key"]: r for r in extra._read(out / "sentiment_gold_ids.csv")}
    rows = []
    for r in tagged:
        key = f"{r['article_id']}#{r['ticker'] or r['kind']}"
        if key not in ids:
            raise MarkedFileError(f"{key} is not in sentiment_gold_ids.csv")
        rows.append({"predicted": ids[key]["predicted"], "gold": r["sentiment"]})
    populations = {r["predicted"]: int(r["population"]) for r in ids.values()}
    rep = gold_metrics.sentiment_report(rows, populations)
    return [
        f"drawn per predicted class: {rep.drawn}",
        f"overall agreement (weighted): {rep.agreement} -> {rep.agreement.judge(AGREEMENT_BAR)}"
        f" (raw {rep.raw_agreement})",
        f"negative precision: {rep.negative_precision} -> "
        f"{rep.negative_precision.judge(NEGATIVE_BAR)}",
        f"negative recall: {rep.negative_recall} -> {rep.negative_recall.judge(NEGATIVE_BAR)}",
        "bars are judged on the whole interval; 'not decided' still goes to the cut with the "
        "interval printed (D34)",
    ]


def faithfulness_report(out: Path) -> list[str]:
    tagged = gold_metrics.read_marked(out / "faithfulness_to_tag.csv", "faithful", ("yes", "no"))
    ids = {r["review_order"]: r for r in extra._read(out / "faithfulness_ids.csv")}
    marks: dict[str, bool] = {}
    owners: dict[str, list[str]] = {}
    for r in tagged:
        row = ids[r["review_order"]]
        marks[row["key"]] = r["faithful"] == "yes"
        owners[row["key"]] = row["extractors"].split("|")
    report = gold_metrics.faithfulness_report(marks, owners)
    lines = [f"{name:>10}: {fig} -> {fig.judge(FAITHFUL_BAR)}" for name, fig in report.items()]
    if {"model", "baseline"} <= set(report):
        pairs: dict[str, dict[str, bool]] = {}
        for key, names in owners.items():
            pair = key.rsplit("#", 1)[0]
            for name in names:
                pairs.setdefault(pair, {})[name] = marks[key]
        both = [v for v in pairs.values() if {"model", "baseline"} <= set(v)]
        gain = gold_metrics.paired_difference(
            [v["model"] for v in both], [v["baseline"] for v in both]
        )
        lines.append(f"model minus baseline on {len(both)} company texts: {gain}")
    return lines


COMMANDS: dict[str, Callable[[Path], list[str]]] = {
    "scope": scope_report,
    "label-check": label_check_report,
    "sentiment": sentiment_report,
    "faithfulness": faithfulness_report,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        print("\n".join(COMMANDS[args.command](args.out)))
    except MarkedFileError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
