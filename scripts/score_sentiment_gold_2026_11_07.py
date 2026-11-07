"""Score a candidate sentiment model on the 180-text gold (Work item 18, T-148c2, D35).

    uv run scripts/score_sentiment_gold_2026_11_07.py --model <path or Hub id> [--revision R]
        [--label baseline] [--gold gold] [--texts data/sentiment_gold_ids.csv]
        [--log gold/sentiment_gold_scoring_log.csv]

``report_wi18 sentiment --gold`` reads the marks and the prediction of the model the sample was
drawn by; it runs no model. This runs one: it reads the texts the owner read (``--texts``, the draw's
ids file: the only copy, ``gold/`` has no news text) and the marks and weights of
``<gold>/sentiment_gold.csv``, checks every text against the hash the owner marked, scores each as
the pipeline's sentiment stage reads a text (sentence-packed chunks of at most 510 tokens, the
probabilities averaged), and prints the report of ``report_wi18 sentiment --gold`` for *that*
model: the strata (the weights) stay those of the draw, the model compared with the owner is the
candidate. Then, apart, the agreement for 2022 on and for before (D35: the training texts are 2022
and later, the gold is not), raw and not re-weighted, with Wilson intervals.

**Every run is listed (D35).** The gold is not a tuning set: each run appends a row to ``--log``
(tracked, no news text): the run number, the time, the label, the model and what identifies it (the
revision, or for a local path the hash of its weight files), the digest of the scoring code, the
hash of the gold file that was scored, the role of the run (``--role``: ``baseline``, ``final`` or
``other``; the first two are one each unless ``--redo-role``: the report quotes the ``final`` row), the labels the model uses and the device, then the agreement with its interval (as text and as
numbers) and what the bar says, and the negative precision and recall. A run is logged only when
every output exists, and a model whose labels are not the three classes stops the run before
anything is written. The report
that quotes a figure lists every run made, not only the best. A run of the pinned model reproduces
the stored predictions (the script says how many differ), which checks the scorer itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import types
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sector_extra_2026_11_05 as extra
import draw_sentiment_gold_2026_11_05 as gold_draw
import report_wi18_2026_11_06 as report

import chunking
from news_nlp import gold_metrics, retag
from news_nlp.gold_metrics import MarkedFileError

LOG_FIELDS = retag.SENTIMENT_SCORING_LOG_COLUMNS
PLANNED_RUNS = 2  # D35: the baseline and the final run
ROLES = ("baseline", "final", "other")
ONCE_ONLY = ("baseline", "final")  # a second one needs --redo-role


def texts_of_the_gold(
    gold_rows: list[dict[str, str]], ids: list[dict[str, str]]
) -> list[dict[str, str]]:
    """The text of each gold row, from the draw's ids, checked against the hash the owner marked.
    Raises ``MarkedFileError`` when a text is missing or is not the one that was read."""
    by_key = {r["key"]: r for r in ids}
    items = []
    for row in gold_rows:
        drawn = by_key.get(row["key"])
        if drawn is None:
            raise MarkedFileError(f"{row['key']} is not in the ids file: its text is not here")
        if retag.text_hash(drawn["text"]) != row["text_hash"]:
            raise MarkedFileError(
                f"{row['key']}: the text in the ids file is not the text the owner marked"
            )
        items.append({"key": row["key"], "text": drawn["text"]})
    return items


def score(
    items: list[dict[str, str]], cfg: dict[str, str] | None = None, id2label: str = ""
) -> dict[str, str]:
    """key -> the model's class, for every text. A text the stage cannot chunk stops the run (the
    figures need all 180), and so does a model that answers with anything but the three classes (a
    head whose ``id2label`` was never set says ``label_0``: every text would disagree and the run
    would be logged as 0%)."""
    scored = gold_draw.predict([dict(i) for i in items], cfg)
    predictions = {t["key"]: t["predicted"] for t in scored}
    missing = [i["key"] for i in items if i["key"] not in predictions]
    if missing:
        raise MarkedFileError(f"{len(missing)} texts have no chunk to score: {missing[:3]}")
    answers = Counter(predictions.values())
    if outside := sorted(set(answers) - set(gold_metrics.SENTIMENT_CLASSES)):
        raise MarkedFileError(
            f"the model answers {outside}, not {list(gold_metrics.SENTIMENT_CLASSES)} "
            f"(its id2label: {id2label or 'unknown'}): nothing was scored or logged; set "
            "id2label on the checkpoint (as train_sentiment does) and score again"
        )
    return predictions


def weights_hash(path: Path) -> str:
    """SHA-256 over the weight files of a local checkpoint (name and content), so a log row can be
    tied to the weights later; empty when *path* is not a directory with weights (a Hub id)."""
    if not path.is_dir():
        return ""
    files = sorted(
        p
        for p in path.glob("*")
        if p.suffix == ".safetensors" or (p.name.startswith("pytorch_model") and p.suffix == ".bin")
    )
    if not files:
        return ""
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        with file.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def describe_model(model: str, revision: str) -> dict[str, str]:
    """What identifies the model that is scored: the hash of its weights (a local path), its labels
    and the device it runs on."""
    import torch  # noqa: PLC0415 -- only a run needs the model
    from transformers import AutoConfig  # noqa: PLC0415

    labels = AutoConfig.from_pretrained(model, revision=revision or None).id2label
    return {
        "weights_sha256": weights_hash(Path(model)),
        "id2label": json.dumps({str(k): v for k, v in labels.items()}),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def period_figures(rows: list[dict[str, str]]) -> dict[str, tuple[int, gold_metrics.Figure]]:
    """Agreement with the owner for 2022 on and before, raw (not re-weighted: the strata are of the
    whole gold) with Wilson intervals, so a model trained on 2022-on texts is not read as one
    figure. Keyed ``2022 on`` and ``before 2022``; a text with no usable date is in neither."""
    late_from = gold_metrics.LATE_FROM_YEAR
    groups = {
        f"{late_from} on": [r for r in rows if (gold_metrics.year_of(r) or 0) >= late_from],
        f"before {late_from}": [
            r for r in rows if (gold_metrics.year_of(r) or late_from) < late_from
        ],
    }
    return {
        name: (
            len(mine),
            gold_metrics.wilson(sum(r["predicted"] == r["gold"] for r in mine), len(mine)),
        )
        for name, mine in groups.items()
    }


def period_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = ["agreement by period (raw, not re-weighted; D35):"]
    for name, (n, figure) in period_figures(rows).items():
        lines.append(f"  {name}: {n} texts, {figure}")
    undated = gold_metrics.rows_without_year(rows)
    if undated:
        lines.append(f"  {undated} texts have no usable pub_date (in neither)")
    return lines


def scoring_code_revision() -> str:
    """The version of the code that reads a text and builds the figures: the chunking, the draw's
    ``predict``, the report and this script (``retag.code_revision``). A change to any of them
    changes figures, and the log row says which version made them."""
    this = types.ModuleType("score_sentiment_gold")
    this.__file__ = __file__
    return retag.code_revision(chunking, gold_draw, report, this)


def require_pinned(model: str, revision: str) -> None:
    """A Hub model needs a revision, as the pipeline pins every model: without one the run scores
    whatever the repository points to that day and the row cannot say which model it was."""
    if not Path(model).is_dir() and not revision:
        raise MarkedFileError(
            f"{model} is not a local directory, so it needs --revision (a commit of the Hub "
            "repository): every model is pinned, and a row without a revision identifies nothing"
        )


def check_role(role: str, earlier: list[dict[str, str]], redo: bool) -> None:
    """The baseline and the final run are one each (D35): a second needs ``--redo-role``."""
    if role in ONCE_ONLY and not redo and any(r.get("role") == role for r in earlier):
        raise MarkedFileError(
            f"the log already has a {role} run: D35 has one, and the report quotes it; run with "
            "--redo-role if this is meant to replace it as the one that counts"
        )


def previous_runs(path: Path) -> list[dict[str, str]]:
    """The rows of the log, which must have the columns this script writes."""
    if not path.exists():
        return []
    rows = extra._read(path)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        header = tuple(next(csv.reader(fh), ()))
    if header != tuple(LOG_FIELDS):
        raise MarkedFileError(f"{path} does not have the columns this script writes: {header}")
    return rows


def log_run(path: Path, row: dict[str, str]) -> int:
    """Append the run to the log (created with its header) and return its number."""
    number = len(previous_runs(path)) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LOG_FIELDS))
        if fresh:
            writer.writeheader()
        writer.writerow({**row, "run": str(number)})
    return number


def figure_columns(name: str, figure: gold_metrics.Figure) -> dict[str, str]:
    """A figure as the log keeps it: the text of the report and its three numbers."""
    return {
        name: str(figure),
        f"{name}_point": f"{figure.point:.4f}",
        f"{name}_low": f"{figure.low:.4f}",
        f"{name}_high": f"{figure.high:.4f}",
    }


def log_row(
    args: argparse.Namespace,
    identity: dict[str, str],
    gold_sha: str,
    rep: gold_metrics.SentimentReport,
    rows: list[dict[str, str]],
) -> dict[str, str]:
    by_period = period_figures(rows)
    late, early = (
        by_period[f"{gold_metrics.LATE_FROM_YEAR} on"],
        by_period[f"before {gold_metrics.LATE_FROM_YEAR}"],
    )
    return {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": args.label,
        "role": args.role,
        "model": args.model,
        "revision": args.revision,
        "weights_sha256": identity["weights_sha256"],
        "code_revision": scoring_code_revision(),
        "gold_sha256": gold_sha,
        "id2label": identity["id2label"],
        "device": identity["device"],
        **figure_columns("agreement", rep.agreement),
        "agreement_judgement": rep.agreement.judge(report.AGREEMENT_BAR),
        "agreement_2022_on": f"{late[1].point:.4f}",
        "agreement_2022_on_low": f"{late[1].low:.4f}",
        "agreement_2022_on_high": f"{late[1].high:.4f}",
        "texts_2022_on": str(late[0]),
        "agreement_before_2022": f"{early[1].point:.4f}",
        "agreement_before_2022_low": f"{early[1].low:.4f}",
        "agreement_before_2022_high": f"{early[1].high:.4f}",
        "texts_before_2022": str(early[0]),
        **figure_columns("negative_precision", rep.negative_precision),
        "negative_precision_judgement": rep.negative_precision.judge(report.NEGATIVE_BAR),
        **figure_columns("negative_recall", rep.negative_recall),
        "negative_recall_judgement": rep.negative_recall.judge(report.NEGATIVE_BAR),
        "texts": str(len(rows)),
    }


def earlier_lines(earlier: list[dict[str, str]]) -> list[str]:
    """The runs already on the gold, and a warning past the two D35 plans (the baseline and the
    final run): the log is the count of looks at the exam."""
    lines = [f"earlier runs on the gold: {len(earlier)}"]
    lines += [f"  run {r['run']} {r['label']}: {r['agreement']}" for r in earlier]
    if len(earlier) + 1 > PLANNED_RUNS:
        lines.append(
            f"warning: this is run {len(earlier) + 1}; D35 plans the baseline and the final run, "
            "and the report that quotes a figure lists every run"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="a local path or a Hub id")
    parser.add_argument("--revision", default="", help="a Hub revision (none for a local path)")
    parser.add_argument("--label", default="candidate", help="a name for the log")
    parser.add_argument(
        "--role",
        choices=ROLES,
        default="other",
        help="which run this is (D35): the baseline and the final run are the ones that count",
    )
    parser.add_argument(
        "--redo-role",
        action="store_true",
        help="accept a second baseline or final run in the log",
    )
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument("--texts", type=Path, default=Path("data/sentiment_gold_ids.csv"))
    parser.add_argument("--log", type=Path, default=Path("gold/sentiment_gold_scoring_log.csv"))
    args = parser.parse_args()
    gold_file = args.gold / "sentiment_gold.csv"
    try:
        tagged = gold_metrics.read_marked(
            gold_file,
            "sentiment",
            gold_metrics.SENTIMENT_CLASSES,
            required=("key", "part", "stratum", "predicted", "population", "text_hash", "pub_date"),
        )
        items = texts_of_the_gold(tagged, extra._read(args.texts))
        require_pinned(args.model, args.revision)
        earlier = previous_runs(args.log)
        check_role(args.role, earlier, args.redo_role)
        identity = describe_model(args.model, args.revision)
        cfg = {"model": args.model, "revision": args.revision}
        predictions = score(items, cfg, identity["id2label"])
        rows, populations = report._sentiment_rows_from_gold(args.gold, predictions)
        lines = report.sentiment_report(args.gold, gold=True, predictions=predictions)
    except MarkedFileError as exc:
        sys.exit(str(exc))
    stored = {r["key"]: r["predicted"] for r in tagged}
    moved = sum(predictions[k] != stored[k] for k in stored)
    rep = gold_metrics.sentiment_report(rows, populations)
    gold_sha = hashlib.sha256(gold_file.read_bytes()).hexdigest()
    output = [
        *earlier_lines(earlier),
        f"model {args.model} {args.revision}: {moved} of {len(stored)} predictions differ from",
        "the model the sample was drawn by (0 for the pinned model: the scorer is the same)",
        *lines,
        *period_lines(rows),
    ]
    number = log_run(args.log, log_row(args, identity, gold_sha, rep, rows))  # last: all built
    print(f"run {number} on the gold ({args.log}): every run is listed (D35)")
    print("\n".join(output))


if __name__ == "__main__":
    main()
