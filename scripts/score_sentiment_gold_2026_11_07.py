"""Score a candidate sentiment model on the 180-text gold (Work item 18, T-148c2, D35).

    uv run scripts/score_sentiment_gold_2026_11_07.py --model <path or Hub id> [--revision R]
        [--label baseline] [--gold gold] [--texts data/sentiment_gold_ids.csv]
        [--log gold/sentiment_gold_scoring_log.csv]

``report_wi18 sentiment --gold`` reads the marks and the prediction of the model the sample was
drawn by; it runs no model. This runs one: it reads the texts the owner read (``--texts``, the draw's
ids file: the only copy, ``gold/`` has no news text) and the marks and weights of
``<gold>/sentiment_gold.csv``, checks every text against the hash the owner marked, scores each as
the pipeline's sentiment stage reads a text (sentence-packed chunks of at most 510 tokens, the
probabilities averaged; with ``--pair`` the model reads the pair company + chunk, the chunk cut to the
room the company leaves, ``sentiment_pair``), and prints the report of ``report_wi18 sentiment --gold`` for *that*
model: the strata (the weights) stay those of the draw, the model compared with the owner is the
candidate. Then, apart, the agreement for 2022 on and for before (D35: the training texts are 2022
and later, the gold is not), raw and not re-weighted, with Wilson intervals.

**A pair model (``--pair``, T-148c, D35 as amended).** The company of a gold row is its ``ticker``'s
name; a row with no ticker (37 of the 180: summaries of sector, market or economy bodies) has no
company and is scored by the text-only model (``--company-less-model`` / ``--company-less-revision``,
by default the pinned sentiment model), as the stage will route it. The figure is that of the routed
system, and the log row says which model each kind went to.

**The gain over the baseline (D35).** A ``final`` run also prints and logs the paired gain of the
agreement and of the negative precision and recall over the **last** ``baseline`` row: the baseline's
predictions are the ``predicted`` column of the gold (the model the sample was drawn by), the gain is
refused if the agreement recomputed from them is not the one the baseline row logged, and its interval
resamples inside the strata of the draw, the same pairs feeding both (``gold_metrics.paired_stratified_gain``).
Its verdict (``better`` / ``not better`` / ``not decided``) is what the pin rule reads.

**Every run is listed (D35).** The gold is not a tuning set: each run appends a row to ``--log``
(tracked, no news text): the run number, the time, the label, the model and what identifies it (the
revision, or for a local path the hash of its weight files), the digest of the scoring code, the
hash of the gold file that was scored, the role of the run (``--role``: ``baseline``, ``final`` or
``other``; the first two are one each unless ``--redo-role``: the report quotes the last ``final`` row, and an earlier one stays as a record), the labels the model uses and the device, then the agreement with its interval (as text and as
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
import re
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
import retag_gold_2026_10_29 as retag_cli

import chunking
import sentiment_pair
from news_nlp import gold_metrics, retag
from news_nlp.gold_metrics import MarkedFileError

LOG_FIELDS = retag.SENTIMENT_SCORING_LOG_COLUMNS
ROLES = ("baseline", "final", "other")
ONCE_ONLY = ("baseline", "final")  # a second one needs --redo-role
LONG_TEXT_TOKENS = (
    128  # the training length of D35: a text over it is a distribution gap, not a defect
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
BASELINE_TOLERANCE = 5e-5  # the log keeps four decimals


def texts_of_the_gold(
    gold_rows: list[dict[str, str]],
    ids: list[dict[str, str]],
    company_of: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """The text of each gold row, from the draw's ids, checked against the hash the owner marked.
    With *company_of* (ticker -> name) each item also carries its ``company`` (empty when
    ``sentiment_pair.reads_as_pair`` says the text is not read as a pair: no ticker, or a company-less
    scope when the ids file knows it). Raises ``MarkedFileError`` when a text is missing or is not the one that was read, or a
    ticker has no name."""
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
        item = {"key": row["key"], "text": drawn["text"]}
        if company_of is not None:
            ticker = (row.get("ticker") or "").strip()
            scope = drawn.get("scope", "")  # the local ids file has it, the tracked gold does not
            if not sentiment_pair.agrees_with_ticker(scope, ticker):
                raise MarkedFileError(
                    f"{row['key']}: the scope {scope!r} and the ticker {ticker!r} disagree about "
                    "whether this text has a company: the routing needs one answer"
                )
            if ticker and ticker not in company_of:
                raise MarkedFileError(f"{row['key']}: the ticker {ticker} has no company name")
            item["company"] = (
                company_of[ticker] if sentiment_pair.reads_as_pair(scope, ticker) else ""
            )
        items.append(item)
    return items


def score(
    items: list[dict[str, str]],
    cfg: dict[str, str] | None = None,
    id2label: str = "",
    *,
    pair: bool = False,
    company_less_cfg: dict[str, str] | None = None,
) -> dict[str, str]:
    """key -> the model's class, for every text. A text the stage cannot chunk stops the run (the
    figures need all 180), and so does a model that answers with anything but the three classes (a
    head whose ``id2label`` was never set says ``label_0``: every text would disagree and the run
    would be logged as 0%). With *pair* the texts that have a ``company`` are read as the pair
    company + text by *cfg*'s model, and the others by the text-only model *company_less_cfg* (the
    routing of the stage)."""
    if pair:
        with_company = [dict(i) for i in items if i.get("company")]
        without = [dict(i) for i in items if not i.get("company")]
        scored = gold_draw.predict(with_company, cfg, pair=True) if with_company else []
        scored += gold_draw.predict(without, company_less_cfg) if without else []
    else:
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


def token_counts(items: list[dict[str, str]], model: str, revision: str) -> dict[str, int]:
    """key -> the tokens of the text (no special tokens), with the tokenizer of *model*."""
    from transformers import AutoTokenizer  # noqa: PLC0415 -- only a run needs the tokenizer

    tok = AutoTokenizer.from_pretrained(model, revision=revision or None)
    return {i["key"]: len(tok.encode(i["text"], add_special_tokens=False)) for i in items}


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


def length_figures(
    rows: list[dict[str, str]], tokens_by_hash: dict[str, int]
) -> dict[str, tuple[int, gold_metrics.Figure]]:
    """Agreement with the owner for the texts over ``LONG_TEXT_TOKENS`` tokens and for the rest, raw
    (not re-weighted) with Wilson intervals: a model trained on texts of at most 128 tokens that
    drops on the long ones has a distribution gap, not necessarily a defect (D35). Keyed
    ``over 128`` and ``up to 128``."""
    groups: dict[str, list[dict[str, str]]] = {
        f"over {LONG_TEXT_TOKENS}": [],
        f"up to {LONG_TEXT_TOKENS}": [],
    }
    for r in rows:
        long = tokens_by_hash[r["text_hash"]] > LONG_TEXT_TOKENS
        groups[f"over {LONG_TEXT_TOKENS}" if long else f"up to {LONG_TEXT_TOKENS}"].append(r)
    return {
        name: (
            len(mine),
            gold_metrics.wilson(sum(r["predicted"] == r["gold"] for r in mine), len(mine)),
        )
        for name, mine in groups.items()
    }


def length_lines(rows: list[dict[str, str]], tokens_by_hash: dict[str, int]) -> list[str]:
    lines = [
        f"agreement by length (raw, not re-weighted; D35, the training texts are at most {LONG_TEXT_TOKENS} tokens):"
    ]
    for name, (n, figure) in length_figures(rows, tokens_by_hash).items():
        lines.append(f"  {name} tokens: {n} texts, {figure}")
    return lines


def gain_columns(
    rows: list[dict[str, str]],
    populations: dict[str, int],
    baseline_row: dict[str, str],
) -> dict[str, str]:
    """The paired gain of the candidate over the last baseline row, on the same gold: the agreement
    and the negative precision and recall, each with its verdict. *rows* carry ``predicted`` (the
    candidate's), ``baseline`` (the stored prediction of the model the sample was drawn by), ``gold``
    and ``stratum``. Refused when the baseline's agreement recomputed from ``baseline`` is not the one
    its log row holds (another gold, or another model, would make the gain mean nothing)."""
    again = gold_metrics.sentiment_report(
        [{**r, "predicted": r["baseline"]} for r in rows], populations
    ).agreement.point
    logged = float(baseline_row["agreement_point"])
    if abs(again - logged) > BASELINE_TOLERANCE:
        raise MarkedFileError(
            f"the baseline row (run {baseline_row['run']}) logged an agreement of {logged:.4f}, but "
            f"the stored predictions of this gold give {again:.4f}: the gain would compare "
            "different things; score the baseline again (--redo-role with a reason)"
        )
    out: dict[str, str] = {}
    for name, metric in (
        ("gain_agreement", "agreement"),
        ("gain_negative_precision", "precision"),
        ("gain_negative_recall", "recall"),
    ):
        figure = gold_metrics.paired_stratified_gain(rows, populations, metric=metric)
        out[name] = str(figure)
        out[f"{name.replace('gain_agreement', 'gain')}_verdict"] = gold_metrics.gain_verdict(figure)
    return out


def require_baseline(role: str, earlier: list[dict[str, str]]) -> dict[str, str] | None:
    """The baseline row a final run is measured against (``None`` for any other role); a final run
    with none is refused."""
    baseline = last_baseline_run(earlier)
    if role == "final" and baseline is None:
        raise MarkedFileError(
            "a final run is measured against the last baseline run: the log has none; score "
            "the pinned model with --role baseline first"
        )
    return baseline if role == "final" else None


def gain_block(
    rows: list[dict[str, str]],
    tagged: list[dict[str, str]],
    populations: dict[str, int],
    baseline_row: dict[str, str] | None,
) -> tuple[dict[str, str], list[str]]:
    """The gain columns and the lines of the report; nothing unless this is a final run (the
    baseline row is only given then)."""
    if baseline_row is None:
        return {}, []
    paired = [{**r, "baseline": t["predicted"]} for r, t in zip(rows, tagged, strict=True)]
    gain = gain_columns(paired, populations, baseline_row)
    lines = [
        f"gain over the baseline (run {baseline_row['run']}), paired, same strata:",
        f"  agreement {gain['gain_agreement']} -> {gain['gain_verdict']}",
        f"  negative precision {gain['gain_negative_precision']} -> "
        f"{gain['gain_negative_precision_verdict']}",
        f"  negative recall {gain['gain_negative_recall']} -> {gain['gain_negative_recall_verdict']}",
    ]
    return gain, lines


def last_baseline_run(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """The last run with role ``baseline``: the row the gain of a final run is measured against."""
    baselines = [r for r in rows if r.get("role") == "baseline"]
    return baselines[-1] if baselines else None


def scoring_code_revision() -> str:
    """The version of the code that reads a text and builds the figures: the chunking, the draw's
    ``predict``, the report and this script (``retag.code_revision``). A change to any of them
    changes figures, and the log row says which version made them."""
    this = types.ModuleType("score_sentiment_gold")
    this.__file__ = __file__
    return retag.code_revision(chunking, gold_draw, report, this)


def require_pinned(model: str, revision: str) -> None:
    """A Hub model needs a revision that is a commit (40 hexadecimal characters), as the pipeline
    pins every model: ``main`` or a tag would score whatever the repository points to that day and
    the row could not say which model it was. A local directory needs none."""
    if Path(model).is_dir():
        return
    if not revision:
        raise MarkedFileError(
            f"{model} is not a local directory, so it needs --revision (a commit of the Hub "
            "repository): every model is pinned, and a row without a revision identifies nothing"
        )
    if not HEX40.match(revision):
        raise MarkedFileError(
            f"--revision {revision!r} is not a commit (40 hexadecimal characters): a branch or a "
            "tag moves, so the row would not identify the model that was scored"
        )


def check_role(role: str, earlier: list[dict[str, str]], redo: bool, reason: str = "") -> None:
    """The baseline and the final run are one each (D35): a second needs ``--redo-role`` and a
    ``--redo-reason`` (a defect in the run, never the figure: it goes to the log row)."""
    if redo and not reason.strip():
        raise MarkedFileError(
            "--redo-role needs --redo-reason: a second baseline or final run is accepted only for "
            "a defect in the run, and the reason is written to the log"
        )
    if reason.strip() and not redo:
        raise MarkedFileError("--redo-reason is only for a --redo-role run")
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
    tokens_by_hash: dict[str, int],
    gain: dict[str, str] | None = None,
    routed: tuple[int, int] = (0, 0),
    company_less_identity: dict[str, str] | None = None,
) -> dict[str, str]:
    by_period = period_figures(rows)
    by_length = length_figures(rows, tokens_by_hash)
    long_n, long_f = by_length[f"over {LONG_TEXT_TOKENS}"]
    short_n, short_f = by_length[f"up to {LONG_TEXT_TOKENS}"]
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
        "redo_reason": args.redo_reason.strip(),
        "input_format": "pair" if args.pair else "text",
        "company_less_model": args.company_less_model if args.pair else "",
        "company_less_revision": args.company_less_revision if args.pair else "",
        "company_less_weights_sha256": (company_less_identity or {}).get("weights_sha256", ""),
        "rows_pair": str(routed[0]),
        "rows_text_only": str(routed[1]),
        "agreement_over_128": f"{long_f.point:.4f}",
        "agreement_over_128_low": f"{long_f.low:.4f}",
        "agreement_over_128_high": f"{long_f.high:.4f}",
        "texts_over_128": str(long_n),
        "agreement_up_to_128": f"{short_f.point:.4f}",
        "agreement_up_to_128_low": f"{short_f.low:.4f}",
        "agreement_up_to_128_high": f"{short_f.high:.4f}",
        "texts_up_to_128": str(short_n),
        **(gain or {}),
    }


def looks_beyond_the_plan(earlier: list[dict[str, str]]) -> int:
    """How many runs on the gold are not the one baseline and the one final run D35 plans: every
    ``other`` run, and every baseline or final run past the first of its role (a ``--redo-role``)."""
    roles = Counter(r.get("role", "other") for r in earlier)
    return roles["other"] + sum(max(roles[role] - 1, 0) for role in ONCE_ONLY)


def last_final_run(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """The row the report quotes: the last run with role ``final``. An earlier ``final`` row (after a
    ``--redo-role``) stays in the log as a record and is not the figure."""
    finals = [r for r in rows if r.get("role") == "final"]
    return finals[-1] if finals else None


def earlier_lines(earlier: list[dict[str, str]], role: str, redo: bool) -> list[str]:
    """The runs already on the gold, and a warning when this one is an extra look at the exam (a run
    that is not the planned baseline or final run): the log is the count of looks."""
    lines = [f"earlier runs on the gold: {len(earlier)}"]
    lines += [f"  run {r['run']} {r['label']}: {r['agreement']}" for r in earlier]
    if role == "other" or redo:
        so_far = looks_beyond_the_plan(earlier)
        lines.append(
            f"warning: this is an extra look at the gold (role {role}{', redo' if redo else ''}); "
            f"looks beyond the plan so far: {so_far}, with this one {so_far + 1}. D35 plans the "
            "baseline and the final run, and the report that quotes a figure lists every run"
        )
    return lines


def check_options(args: argparse.Namespace) -> None:
    """Options that would make a run meaningless are refused before anything is loaded: the baseline
    is the pinned text-only model (a pair model logged as the baseline would become the model every
    later gain is measured against), and the company-less model belongs to a pair run."""
    if args.pair and args.role == "baseline":
        raise MarkedFileError(
            "--pair with --role baseline: the baseline is the pinned text-only model; a pair model "
            "is scored as --role final (or other)"
        )
    if not args.pair and (args.company_less_model or args.company_less_revision):
        raise MarkedFileError(
            "--company-less-model / --company-less-revision belong to a --pair run: a text-only "
            "model has no company-less texts to route"
        )


def require_class_labels(identity: dict[str, str], model: str) -> None:
    """The labels a model uses must be the three classes (a head whose ``id2label`` was never set says
    ``label_0``): refused before anything is scored, for the second model of a routed run, whose
    answers are only checked once both models have run."""
    labels = {v.lower() for v in json.loads(identity["id2label"]).values()}
    if outside := sorted(labels - set(gold_metrics.SENTIMENT_CLASSES)):
        raise MarkedFileError(
            f"the company-less model {model} uses the labels {outside}, not "
            f"{list(gold_metrics.SENTIMENT_CLASSES)}: set id2label on the checkpoint and score again"
        )


def company_less_setup(args: argparse.Namespace) -> tuple[dict[str, str] | None, dict[str, str]]:
    """The text-only model of a ``--pair`` run and what identifies it (its weights, its labels), checked
    before anything is scored; ``(None, {})`` for a text-only run."""
    config = resolve_company_less(args)
    if config is None:
        return None, {}
    identity = describe_model(config["model"], config["revision"])
    require_class_labels(identity, config["model"])
    return config, identity


def company_names() -> dict[str, str]:
    """ticker -> the company's name, from the universe the draws use."""
    return {c.ticker: c.name for c in retag_cli.companies()}


def resolve_company_less(args: argparse.Namespace) -> dict[str, str] | None:
    """The text-only model that scores the rows with no company of a ``--pair`` run: the one named,
    or the pinned sentiment model. It is pinned like any model (a commit, or a local directory)."""
    if not args.pair:
        return None
    if args.company_less_model:
        require_pinned(args.company_less_model, args.company_less_revision)
        return {"model": args.company_less_model, "revision": args.company_less_revision}
    pinned = gold_draw._models()["sentiment"]
    args.company_less_model, args.company_less_revision = pinned["model"], pinned["revision"]
    return {"model": pinned["model"], "revision": pinned["revision"]}


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
    parser.add_argument(
        "--redo-reason",
        default="",
        help="with --redo-role: why (a defect in the run, never the figure); written to the log",
    )
    parser.add_argument(
        "--pair",
        action="store_true",
        help="the model reads the pair company + text (sentiment_pair); a row with no ticker goes "
        "to --company-less-model",
    )
    parser.add_argument(
        "--company-less-model",
        default="",
        help="with --pair: the text-only model for rows with no company (default: the pinned one)",
    )
    parser.add_argument("--company-less-revision", default="")
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
        check_options(args)
        companies = company_names() if args.pair else None
        items = texts_of_the_gold(tagged, extra._read(args.texts), companies)
        require_pinned(args.model, args.revision)
        company_less, company_less_identity = company_less_setup(args)
        earlier = previous_runs(args.log)
        check_role(args.role, earlier, args.redo_role, args.redo_reason)
        baseline_row = require_baseline(args.role, earlier)
        identity = describe_model(args.model, args.revision)
        cfg = {"model": args.model, "revision": args.revision}
        predictions = score(
            items, cfg, identity["id2label"], pair=args.pair, company_less_cfg=company_less
        )
        tokens = token_counts(items, args.model, args.revision)
        rows, populations = report._sentiment_rows_from_gold(args.gold, predictions)
        lines = report.sentiment_report(args.gold, gold=True, predictions=predictions)
    except MarkedFileError as exc:
        sys.exit(str(exc))
    stored = {r["key"]: r["predicted"] for r in tagged}
    moved = sum(predictions[k] != stored[k] for k in stored)
    rep = gold_metrics.sentiment_report(rows, populations)
    tokens_by_hash = {t["text_hash"]: tokens[t["key"]] for t in tagged}
    try:
        gain, gain_lines = gain_block(rows, tagged, populations, baseline_row)
    except MarkedFileError as exc:
        sys.exit(str(exc))
    gold_sha = hashlib.sha256(gold_file.read_bytes()).hexdigest()
    with_company = sum(bool(i.get("company")) for i in items)
    routed = (with_company, len(items) - with_company)
    output = [
        *earlier_lines(earlier, args.role, args.redo_role),
        f"model {args.model} {args.revision}: {moved} of {len(stored)} predictions differ from",
        "the model the sample was drawn by (0 for the pinned model: the scorer is the same)",
        f"rows read as a pair: {routed[0]}, by the text-only model: {routed[1]}",
        *lines,
        *period_lines(rows),
        *length_lines(rows, tokens_by_hash),
        *gain_lines,
    ]
    number = log_run(
        args.log,
        log_row(
            args, identity, gold_sha, rep, rows, tokens_by_hash, gain, routed, company_less_identity
        ),
    )  # last: all built
    print(f"run {number} on the gold ({args.log}): every run is listed (D35)")
    print("\n".join(output))


if __name__ == "__main__":
    main()
