"""Fine-tune the sentiment model on the owner's summaries as the pair company + text (T-148c3, PLAN D35).

    uv run scripts/train_sentiment_pair_2026_11_09.py --time-one    # one run, timed (4 epochs, 2e-5, seed 1)
    uv run scripts/train_sentiment_pair_2026_11_09.py --grid        # the 27 runs of the grid (resumable)
    uv run scripts/train_sentiment_pair_2026_11_09.py --final       # the setting chosen, one run, the model kept
    uv run scripts/train_sentiment_pair_2026_11_09.py --baseline    # the pinned model on the validation slice, no training
    uv run scripts/train_sentiment_pair_2026_11_09.py --report      # the final run against the baseline: precision, recall, ROC OVR

The recipe and its checks are in ``src/sentiment_pair_train.py``. This script gives them the data: the 450
marks (``gold/sentiment_train.csv``) with their texts (``data/sentiment_train_ids.csv``, git-ignored), the
earlier balanced pool of sentences as the replay (``data/sentiment_finetune``, read for the company of each
sentence from the SOURCE), and the scorer's own reading of a saved model for the validation slice
(``draw_sentiment_gold_2026_11_05.predict_probs``, with the pair). It never reads the 180-text gold.

**The metrics (owner, 2026-11-09; PLAN D37): Precision, Recall and ROC OVR, per class, are recorded for every run and
are what model selection uses**: the validation score of a run is the mean of those nine figures (weighted by the
strata of the draw), and the grid's tie rule reads it. Accuracy is not a selection metric.

Read-only on the databases; writes ``data/sentiment_pair/`` (the replay as drawn, the results of the runs, and
``predictions/<tag>__<run>.csv``: the class and the class probabilities each run gave each validation text) and,
for ``--final``, the model to ``models/sentiment_pair/final`` (git-ignored). A run's model is deleted after its
validation score is computed, except the final one. ``--mlflow`` logs every run (params, the validation score,
the precision, recall and ROC OVR of each class, the epochs' losses) to the local MLflow store
(``uv sync --group eval``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw_sentiment_gold_2026_11_05 as gold_draw
import draw_sentiment_train_2026_11_07 as train_draw
import torch
from tqdm import tqdm

import news_nlp as db
import sentiment_pair_report as rep
import sentiment_pair_train as spt
from news_nlp import retag, triage
from sentiment_stage import _text_mentions_subject

MARKS = Path("gold/sentiment_train.csv")
IDS = Path("data/sentiment_train_ids.csv")
REPLAY_POOL = Path("data/sentiment_finetune/labeled_sentences_balanced_2026_09_14.jsonl")
OUT = Path("data/sentiment_pair")
MODELS = Path("models/sentiment_pair")
REPLAY_SEED = (
    20261109  # fixed here, before any run: the replay is the same for every run of the grid
)
TIMING_SETTING = spt.RunSettings(*spt.EARLIER_SETTING, seed=1)
#: How the SOURCE was read: written into the results file, so a result got from a snapshot can be told.
#: A run's score recomputed from its predictions must equal the one its row logged.
SCORE_TOLERANCE = 1e-9
#: Bootstrap resamples of the report's intervals (``gold_metrics.RESAMPLES``); a test lowers it.
REPORT_RESAMPLES = rep.RESAMPLES
PREDICTION_COLUMNS = ["key", "expected", "predicted", *[f"p_{c}" for c in rep.CLASSES]]
SOURCE_MODE = {"how": "attached read-only"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def article_infos(conn: db.NewsNlpDatabase, ids: set[int]) -> dict[int, spt.ArticleInfo]:
    """The body hash, the ticker and the company of each article id of the replay pool, from the SOURCE."""
    infos: dict[int, spt.ArticleInfo] = {}
    for article_id, ticker, company, body, title in conn.execute(
        "SELECT id, ticker, company, body_text, title FROM source.articles "
        "WHERE id IN (SELECT value FROM json_each(?))",
        (json.dumps(sorted(ids)),),
    ):
        if body and ticker and company:
            infos[int(article_id)] = spt.ArticleInfo(
                triage.body_hash(body), ticker, company, headline_key(title)
            )
    return infos


def headline_key(title: str | None) -> str:
    """The headline key of the draw (``title_key``), or empty if too short to mean anything."""
    key = train_draw.title_key(title or "")
    return key if len(key) >= train_draw.MIN_TITLE_CHARS else ""


def exam_keys(
    conn: db.NewsNlpDatabase, marked_hashes: set[str], marked_titles: set[str]
) -> tuple[set[str], set[str]]:
    """What the replay must not contain: the bodies **and the headlines** of the 180-text gold, the
    held-out set and the 450 drawn for tagging (the draw filters by headline too: the same story is stored
    under another body)."""
    gold_ids = {int(r["article_id"]) for r in read_csv(train_draw.GOLD)}
    held_ids = {int(r["article_id"]) for r in read_csv(Path("data/heldout_ids.csv"))}
    hashes, titles = train_draw.exam_keys(conn, gold_ids | held_ids)
    return hashes | marked_hashes, titles | marked_titles


def build_data(
    conn: db.NewsNlpDatabase,
) -> tuple[spt.MarkedSummaries, list[spt.Example], spt.ReplayReport]:
    """The marks, and the replay as large as the training summaries. Prints what was left out and why."""
    marked = spt.load_marked_summaries(read_csv(MARKS), read_csv(IDS))
    print(
        f"marks: train {len(marked.train)}, validation {len(marked.validation)}; "
        f"left out {marked.left_out or 'none'} (no company: they stay with the text-only model)"
    )
    sentences = [json.loads(line) for line in REPLAY_POOL.read_text(encoding="utf-8").splitlines()]
    infos = article_infos(conn, {int(s["article_id"]) for s in sentences})
    marked_titles = {
        key
        for r in read_csv(IDS)
        if r["body_hash"] in marked.body_hashes and (key := headline_key(r["title"]))
    }
    hashes, titles = exam_keys(conn, marked.body_hashes, marked_titles)
    replay, report = spt.replay_examples(
        sentences,
        infos.get,
        hashes,
        _text_mentions_subject,
        n=len(marked.train),
        seed=REPLAY_SEED,
        never_titles=titles,
    )
    print("\n".join(report.lines()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay.json").write_text(
        json.dumps([e._asdict() for e in replay], indent=1), encoding="utf-8"
    )
    return marked, replay, report


class ValidationScorer:
    """The scorer's own reading of a saved model on the validation slice: the **validation score** (the mean of
    the precision, recall and ROC OVR of the three classes, weighted by the strata of the draw: PLAN D37). After
    a call, ``predictions`` holds the per-text predictions (key, owner's class, predicted class, the class
    probabilities) and ``detail`` the figures of each class and their macros."""

    def __init__(self, validation: list[spt.Example], marks: list[dict[str, str]]) -> None:
        self.items = [{"key": e.key, "text": e.text, "company": e.company} for e in validation]
        self._expected = {e.key: spt.ID2LABEL[e.label] for e in validation}
        self.marks = marks
        self.predictions: list[dict[str, str]] = []
        self.detail: dict[str, dict[str, float | None]] = {}

    def __call__(self, model_dir: str) -> float:
        scored = gold_draw.predict_probs(
            [dict(i) for i in self.items], {"model": model_dir}, pair=True
        )
        self.predictions = self.collect(scored)
        self.detail, score = self.measure(self.predictions)
        return score

    def collect(self, scored: list[dict[str, Any]]) -> list[dict[str, str]]:
        """The predictions of a reading, one per validation text; stop if a text was not read."""
        if len(scored) != len(self.items):
            raise spt.MarkedFileError(
                f"{len(self.items) - len(scored)} validation texts have no chunk"
            )
        return [
            {
                "key": t["key"],
                "expected": self._expected[t["key"]],
                "predicted": t["predicted"],
                **{f"p_{c}": repr(float(t["probs"][c])) for c in rep.CLASSES},
            }
            for t in scored
        ]

    def measure(
        self, predictions: list[dict[str, str]]
    ) -> tuple[dict[str, dict[str, float | None]], float]:
        """The per-class figures and the validation score of a reading."""
        return measure(predictions, self.marks)


def measure(
    predictions: list[dict[str, str]], marks: list[dict[str, str]]
) -> tuple[dict[str, dict[str, float | None]], float]:
    """The precision, recall and ROC OVR of each class (weighted by the strata of the draw), as a run records
    them, and the validation score: their mean."""
    scored, populations = rep.scored_from_rows(predictions, marks)
    metrics = rep.class_metrics(scored, rep.row_weights(scored, populations))
    return rep.detail_of(metrics), rep.selection_score(metrics)


def validation_scorer(
    validation: list[spt.Example], marks: list[dict[str, str]]
) -> ValidationScorer:
    return ValidationScorer(validation, marks)


def prediction_name(tag: str, name: str) -> str:
    """The file of a run's predictions: qualified by its tag, because the settings alone do not make a run
    unique (the timing run has the settings, and so the name, of one of the grid's)."""
    return f"{tag}__{name}"


def write_predictions(name: str, predictions: list[dict[str, str]]) -> None:
    folder = OUT / "predictions"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)


def read_predictions(name: str) -> list[dict[str, str]] | None:
    path = OUT / "predictions" / f"{name}.csv"
    return sorted(read_csv(path), key=lambda r: r["key"]) if path.exists() else None


def connect_source() -> Any:
    """The SOURCE as schema ``source``. The normal read-only attach first; if SQLite cannot open it (the
    usual cause: another process, on Windows for instance, has the file open and the write-ahead log's
    shared memory cannot be mapped from here, "disk I/O error"), the SOURCE is read as an immutable
    snapshot: nothing is written, and nothing may write to it while this runs (the crawler is idle)."""
    try:
        return db.connect_pipeline()
    except RuntimeError as error:
        path = db.env.source_db_path()
        if path is None:
            raise
        print(f"{error}\nreading the SOURCE as an immutable snapshot instead", file=sys.stderr)
        SOURCE_MODE["how"] = "immutable snapshot (an older state if a write-ahead log was active)"
        conn = sqlite3.connect(":memory:", uri=True)
        conn.execute(
            "ATTACH DATABASE ? AS source", (f"file:{Path(path).as_posix()}?mode=ro&immutable=1",)
        )
        return conn


def code_revision() -> str:
    """The revision of the code that makes a number of the grid: the recipe, the pair's encoding, the
    chunking and the scorer."""
    import chunking  # noqa: PLC0415
    import sentiment_pair  # noqa: PLC0415

    return retag.code_revision(spt, sentiment_pair, chunking, gold_draw)


def base_model() -> tuple[str, str]:
    cfg = json.loads(Path("config/pipeline_models.json").read_text(encoding="utf-8"))["sentiment"]
    return cfg["model"], cfg["revision"]


def one_run(
    settings: spt.RunSettings,
    train: list[spt.Example],
    scorer: Any,
    *,
    keep: Path | None = None,
    mlflow_on: bool = False,
    tag: str = "grid",
    fingerprint: str = "",
) -> dict[str, Any]:
    """Train and score one run; delete its model unless *keep* is given. Returns the result row."""
    model, revision = base_model()
    target = keep or MODELS / "_run" / settings.name
    config = spt.SentimentPairTrainConfig(
        base_model=model,
        base_revision=revision,
        output_dir=str(target),
        settings=settings,
        train=train,
    )
    started = time.time()
    artifact = spt.SentimentPairTrainer(scorer, log=tqdm.write).train(config)
    row = {
        "name": settings.name,
        **(artifact.metrics or {}),
        "total_seconds": time.time() - started,
        "tag": tag,
    }
    if keep is None:
        shutil.rmtree(target, ignore_errors=True)
    if predictions := getattr(scorer, "predictions", None):
        write_predictions(prediction_name(tag, settings.name), predictions)
    if mlflow_on:
        log_mlflow(row, settings, len(train), fingerprint)
    return row


def log_mlflow(
    row: dict[str, Any], settings: spt.RunSettings, n_train: int, fingerprint: str = ""
) -> None:
    """One MLflow run per training run, in the local store (``mlruns/``)."""
    import mlflow  # noqa: PLC0415 -- the ``eval`` group, imported only when asked for

    mlflow.set_experiment("sentiment_pair_summaries")
    with mlflow.start_run(run_name=prediction_name(row["tag"], row["name"])):
        mlflow.set_tags(
            {
                "stage": "sentiment",
                "recipe": "D35 pair",
                "kind": row["tag"],
                "fingerprint": fingerprint[:12],  # the key of grid_results.json
            }
        )
        mlflow.log_params(
            {
                "epochs": settings.epochs,
                "learning_rate": settings.learning_rate,
                "seed": settings.seed,
                "batch_size": spt.BATCH_SIZE,
                "weight_decay": spt.WEIGHT_DECAY,
                "n_train": n_train,
                "base_model": "{}@{}".format(*base_model()),
            }
        )
        mlflow.log_metric("validation_score", row["validation_score"])
        for label, figures in (row.get("validation_detail") or {}).items():
            for metric, value in figures.items():
                if value is not None:  # an undefined figure is not logged
                    mlflow.log_metric(f"{metric}_{label}", value)
        mlflow.log_metric("train_seconds", row["train_seconds"])
        for epoch, loss in enumerate(row["epoch_losses"], 1):
            mlflow.log_metric("train_loss", loss, step=epoch)


def load_results() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The results file: its header (fingerprint, how the SOURCE was read) and the rows of the runs. A file
    of the first format (a bare list) has no fingerprint."""
    path = OUT / "grid_results.json"
    if not path.exists():
        return {}, []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {}, data
    return {k: v for k, v in data.items() if k != "rows"}, data["rows"]


def save_results(rows: list[dict[str, Any]], header: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "grid_results.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({**header, "rows": rows}, indent=1), encoding="utf-8")
    temporary.replace(path)


def set_aside(header: dict[str, Any]) -> Path:
    """Move the results file to a name of its own (its fingerprint; a number if that name is taken), never over
    an earlier set-aside file: those rows are the only record of that grid."""
    base = f"grid_results.previous-{str(header.get('fingerprint') or 'unfingerprinted')[:12]}"
    target, n = OUT / f"{base}.json", 1
    while target.exists():
        n += 1
        target = OUT / f"{base}-{n}.json"
    (OUT / "grid_results.json").replace(target)
    return target


def open_results(
    fingerprint: str, *, fresh: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The rows already made, if they belong to this data and recipe; otherwise stop (or, with *fresh*, set
    the old file aside and start a new one). The header records the SOURCE's reading."""
    header, rows = load_results()
    if rows and header.get("fingerprint") != fingerprint:
        if not fresh:
            sys.exit(
                "grid_results.json was made with other data or settings (fingerprint "
                f"{str(header.get('fingerprint'))[:12]} against {fingerprint[:12]}): its rows are not rows "
                "of this grid. Run --grid --fresh to set it aside and start again."
            )
        aside = set_aside(header)
        print(f"the earlier results are kept as {aside}")
        rows = []
    mode, earlier = SOURCE_MODE["how"], header.get("source") if rows else None
    if earlier is None or earlier == mode or mode in earlier:
        source = earlier or mode
    else:  # the rows already there were made with the SOURCE read another way
        source = f"{earlier}, then {mode}"
    return {"fingerprint": fingerprint, "source": source}, rows


def scores_by_setting(rows: list[dict[str, Any]]) -> dict[tuple[int, float], list[float]]:
    by: dict[tuple[int, float], list[float]] = {}
    for row in rows:
        if row["tag"] == "grid":
            s = row["settings"]
            by.setdefault((s["epochs"], s["learning_rate"]), []).append(row["validation_score"])
    return by


def run_timing(train: list[spt.Example], scorer: Any, mlflow_on: bool) -> None:
    print(f"timing one run: {TIMING_SETTING.name} on {len(train)} training texts", flush=True)
    row = one_run(TIMING_SETTING, train, scorer, mlflow_on=mlflow_on, tag="timing")
    epochs_in_grid = sum(s.epochs for s in spt.grid_settings())
    per_epoch = row["train_seconds"] / TIMING_SETTING.epochs
    overhead = row["total_seconds"] - row["train_seconds"]
    full = per_epoch * (epochs_in_grid + spt.EARLIER_SETTING[0]) + overhead * (
        len(spt.grid_settings()) + 1
    )
    print(
        f"validation score {row['validation_score']:.3f}; training {row['train_seconds']:.0f} s "
        f"({per_epoch:.1f} s an epoch), loading and scoring {overhead:.0f} s; the 27 runs and the final one: "
        f"about {full / 60:.0f} min ({epochs_in_grid} epochs + {spt.EARLIER_SETTING[0]})"
    )


def run_grid(
    train: list[spt.Example],
    scorer: Any,
    mlflow_on: bool,
    fingerprint: str,
    n_validation: int = 0,
    *,
    fresh: bool = False,
) -> None:
    header, rows = open_results(fingerprint, fresh=fresh)
    done = {r["name"] for r in rows if r["tag"] == "grid"}
    todo = [s for s in spt.grid_settings() if s.name not in done]
    print(f"grid: {len(done)} runs already done, {len(todo)} to run", flush=True)
    for settings in tqdm(todo, desc="grid", unit="run"):
        rows.append(one_run(settings, train, scorer, mlflow_on=mlflow_on, fingerprint=fingerprint))
        save_results(rows, header)  # after every run: a stop loses one run, not the grid
        tqdm.write(f"{settings.name}: validation score {rows[-1]['validation_score']:.3f}")
    summarise(rows, n_validation)


def summarise(rows: list[dict[str, Any]], n_validation: int = 0) -> spt.Choice:
    """The mean validation score of each setting (and, where the runs recorded them, the macro precision, recall
    and ROC OVR it is made of), and the choice by D35's tie rule on the score."""
    by = scores_by_setting(rows)
    macros: dict[tuple[int, float], dict[str, list[float]]] = {}
    for row in rows:
        if row["tag"] == "grid" and (row.get("validation_detail") or {}).get("macro"):
            s = row["settings"]
            for metric, value in row["validation_detail"]["macro"].items():
                macros.setdefault((s["epochs"], s["learning_rate"]), {}).setdefault(
                    metric, []
                ).append(value)
    print(
        "\nmean validation score by setting (3 seeds; the score is the mean of the macro precision, recall and ROC OVR):"
    )
    for (epochs, lr), values in sorted(by.items()):
        parts = macros.get((epochs, lr))
        made_of = (
            " (precision {:.3f}, recall {:.3f}, ROC OVR {:.3f})".format(
                *(statistics.fmean(parts[m]) for m in rep.CLASS_METRICS)
            )
            if parts
            else ""
        )
        print(
            f"  {epochs} epochs, lr {lr:g}: {statistics.fmean(values):.3f}{made_of} {[round(v, 3) for v in values]}"
        )
    choice = spt.choose_settings(by)
    print(f"chosen: {choice.epochs} epochs, lr {choice.learning_rate:g}: {choice.reason}")
    if n_validation:
        print(
            f"read with care: the SE above is the spread between seeds only; the validation slice has "
            f"{n_validation} texts and its sampling noise is shared by every setting and not shown by the seeds "
            "(the report's intervals show it for the final run), so a small gap between settings is not a "
            "difference the validation slice can show"
        )
    return choice


def run_final(
    train: list[spt.Example],
    scorer: Any,
    mlflow_on: bool,
    fingerprint: str,
    n_validation: int = 0,
) -> None:
    header, rows = open_results(fingerprint)
    if len({r["name"] for r in rows if r["tag"] == "grid"}) < len(spt.grid_settings()):
        sys.exit("the grid is not complete: run --grid first")
    target = MODELS / "final"
    has_row = any(r["tag"] == "final" for r in rows)
    if target.exists() or has_row:
        sys.exit(
            f"a final run is made once: the model directory {'exists' if target.exists() else 'is missing'} "
            f"and the results file has {'a' if has_row else 'no'} final row. If a run was cut half way, "
            f"delete {target} and the final row of {OUT / 'grid_results.json'} to make it again."
        )
    choice = summarise(rows, n_validation)
    settings = spt.RunSettings(choice.epochs, choice.learning_rate, spt.FINAL_SEED)
    partial = (
        MODELS / "final.partial"
    )  # renamed only after the row is saved: no model without a row
    shutil.rmtree(partial, ignore_errors=True)
    row = one_run(
        settings,
        train,
        scorer,
        keep=partial,
        mlflow_on=mlflow_on,
        tag="final",
        fingerprint=fingerprint,
    )
    seeds = scores_by_setting(rows)[(choice.epochs, choice.learning_rate)]
    rows.append({**row, "fingerprint": fingerprint, "code": code_revision()})
    save_results(rows, header)
    partial.replace(target)
    print(
        f"\nfinal run ({settings.name}): validation score {row['validation_score']:.3f}; "
        f"the three seeds of the same setting: mean {statistics.fmean(seeds):.3f}, "
        f"spread (sd) {statistics.stdev(seeds):.3f}\nmodel: {target} (not published, not scored on the gold)"
    )
    run_report(required=False)


def keys_hash(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()[:12]


def metric_lines(name: str, detail: dict[str, dict[str, float | None]], score: float) -> list[str]:
    """The precision, recall and ROC OVR of each class and their macro, and the score, as a run recorded them
    (weighted by the strata of the draw; points, without intervals)."""

    def fmt(v: float | None) -> str:
        return "undefined" if v is None else f"{v:.3f}"

    lines = [f"{name}: validation score {score:.3f} (weighted; the mean of the nine figures)"]
    for label in (*rep.CLASSES, "macro"):
        figures = detail[label]
        lines.append(
            f"  {label:<8} precision {fmt(figures['precision'])}, recall {fmt(figures['recall'])}, "
            f"ROC OVR {fmt(figures['roc_ovr'])}"
        )
    return lines


def run_baseline(validation: list[spt.Example], marks: list[dict[str, str]]) -> dict[str, Any]:
    """The pinned model on the validation slice, as the scorer reads it, with the pair and without: what the
    grid's numbers are to be set against (no training). The per-text predictions (with the class
    probabilities) are kept; a text the model did not read stops the run, as it does for the grid.
    ``baseline.json`` records the model and revision, the code revision, the validation keys and each
    reading's score, so a report can tell a stale baseline."""
    model, revision = base_model()
    scorer = ValidationScorer(validation, marks)
    items = scorer.items
    result: dict[str, Any] = {
        "model": f"{model}@{revision}",
        "code": code_revision(),
        "keys": keys_hash([e.key for e in validation]),
        "n": len(items),
    }
    for name, pair in (("pair", True), ("text_only", False)):
        scored = gold_draw.predict_probs(
            [dict(i) for i in items], {"model": model, "revision": revision}, pair=pair
        )
        predictions = scorer.collect(scored)
        write_predictions(f"baseline_{name}", predictions)
        detail, score = scorer.measure(predictions)
        result[f"{name}_score"], result[f"{name}_detail"] = score, detail
        print("\n".join(metric_lines(f"pinned, {name}", detail, score)))
        print("\n".join(rep.report_lines(f"pinned, {name}", *_columns(predictions))))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(
        f"pinned {result['model']} on {result['n']} validation texts: validation score as the pair "
        f"{result['pair_score']:.3f}; text only {result['text_only_score']:.3f}"
    )
    return result


def _columns(predictions: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    return [p["expected"] for p in predictions], [p["predicted"] for p in predictions]


def stale_baseline(final_row: dict[str, Any], keys: list[str]) -> str | None:
    """Why the baseline cannot be set against this final run, or None: another model or revision, another
    code revision (the baseline's, the final run's, today's), or other validation texts."""
    path = OUT / "baseline.json"
    if not path.exists():
        return "there is no baseline.json: run --baseline"
    baseline = json.loads(path.read_text(encoding="utf-8"))
    model, revision = base_model()
    if baseline.get("model") != f"{model}@{revision}":
        return f"the baseline is of {baseline.get('model')}, the pinned model is now {model}@{revision}"
    if baseline.get("keys") != keys_hash(keys):
        return "the baseline read other validation texts than the final run"
    now = code_revision()
    if not baseline.get("code") == final_row.get("code") == now:
        return (
            f"the code differs: baseline {baseline.get('code')}, final run {final_row.get('code')}, "
            f"now {now}: run --baseline again (and the grid and --final if the change is to the recipe)"
        )
    return None


def print_weighted(scored: dict[str, list[rep.Scored]], populations: dict[str, int]) -> None:
    """The figures of each reading, weighted, with intervals: precision, recall and ROC OVR of each class, their
    macros and the validation score; D32's verdicts on the final run's negative class; then the paired weighted
    gain of the final run over the pinned model as text, with the gold's verdicts, and the pin rule on it. The
    two tests are different: a model can pin and still be below the bar."""
    for name, rows in scored.items():
        figures = rep.weighted_figures(rows, populations, resamples=REPORT_RESAMPLES)
        print(f"  {name}: validation score {figures['score']}")
        for label in rep.CLASSES:
            print(
                f"    {label:<8} "
                + "; ".join(f"{m} {figures[f'{label} {m}']}" for m in rep.CLASS_METRICS)
            )
        print(
            "    macro    " + "; ".join(f"{m} {figures[f'macro {m}']}" for m in rep.CLASS_METRICS)
        )
        if name == "final":
            verdicts = rep.bar_verdicts(figures)
            print(
                "    D32's bar on the final run's negative class (precision and recall 70%; ROC OVR has no bar): "
                + "; ".join(f"{k} {v}" for k, v in verdicts.items())
            )
    gains = rep.weighted_gains(
        scored["final"], scored["pinned_text"], populations, resamples=REPORT_RESAMPLES
    )
    print("  paired weighted gain of the final run over the pinned model as text (D35's pin rule):")
    for name in ("score", *(f"macro {m}" for m in rep.CLASS_METRICS)):
        gain, verdict = gains[name]
        print(f"    {name}: {gain} -> {verdict}")
    for name in ("negative precision", "negative recall", "negative roc_ovr"):
        gain, verdict = gains[name]
        print(f"    {name}: {gain} -> {verdict}")
    pins, reason = rep.pin_decision({m: gains[m][1] for m in rep.PIN_STATISTICS})
    print(
        f"  the pin rule on these figures: {'would pin' if pins else 'would not pin'}: {reason} "
        "(validation; the rule is applied once, to the gold)"
    )


def check_run_scores(marks: list[dict[str, str]], rows: list[dict[str, Any]]) -> None:
    """Every run's score recomputed from its predictions file must equal the one its row logged (the predictions
    and the row are of the same run). A mismatch is a WARNING, and says which run."""
    for row in rows:
        if row["tag"] not in ("grid", "final"):
            continue
        predictions = read_predictions(prediction_name(row["tag"], row["name"]))
        if predictions is None:
            print(f"  (the predictions of {row['name']} are not here: its score is not checked)")
            continue
        recomputed = measure(predictions, marks)[1]
        if abs(recomputed - row["validation_score"]) > SCORE_TOLERANCE:
            print(
                f"  WARNING: the predictions of {row['name']} give a score of {recomputed:.4f}, its row logged "
                f"{row['validation_score']:.4f}: they are not of the same run"
            )


def run_report(*, required: bool = True) -> None:
    """The final run against the pinned model's two readings, from the predictions kept. **Weighted** (the
    figures the bar and D35's pin rule are read in: precision, recall and ROC OVR per class, the validation
    score, and the paired gain over the pinned model read as text, with the gold's verdicts), then the **raw**
    description of the 83 (confusion, per class, McNemar). The slice is drawn per class the pinned model
    predicted, so raw shares are not the population's."""
    final_row = next((r for r in reversed(load_results()[1]) if r["tag"] == "final"), None)
    final = read_predictions(prediction_name("final", final_row["name"])) if final_row else None
    base = {n: read_predictions(f"baseline_{n}") for n in ("text_only", "pair")}
    if final is None or None in base.values():
        message = (
            "the report needs --baseline and --final to have run (their predictions are not here)"
        )
        if required:
            sys.exit(message)
        print(f"\n{message}: run --baseline, then --report")
        return
    assert final_row is not None
    keys = [p["key"] for p in final]
    for name, rows in base.items():
        assert rows is not None
        if [p["key"] for p in rows] != keys:
            sys.exit(f"the final run and the baseline {name} did not read the same texts")
    if problem := stale_baseline(final_row, keys):
        if required:
            sys.exit(f"the report cannot be made: {problem}")
        print(f"\nthe report is not made: {problem}")
        return
    marks = read_csv(MARKS)
    readings = {
        "final": final,
        "pinned_text": base["text_only"] or [],
        "pinned_pair": base["pair"] or [],
    }
    scored = {name: rep.scored_from_rows(preds, marks)[0] for name, preds in readings.items()}
    populations = rep.scored_from_rows(final, marks)[1]
    drawn = sum(s.predicted == s.stratum for s in scored["pinned_text"])
    print(
        f"\nWEIGHTED (each text weighs pool size / drawn of its stratum; strata = the class the pinned model "
        f"predicted when the text was drawn: {populations}; drawn "
        f"{dict(Counter(s.stratum for s in scored['final']))}). The weights stand for the pair-readable "
        f"texts of each stratum: the 7 company-less rows of the 90 are out of the slice and stay with the "
        f"text-only model, which the gold includes. Precision, recall and ROC OVR, per class (PLAN D37).\n"
        f"(check: the pinned model as text agrees with its stored draw prediction on {drawn} of {len(keys)} texts)"
    )
    if drawn != len(keys):
        print(
            f"WARNING: the baseline is not the reading the strata were drawn from ({len(keys) - drawn} texts "
            "differ): the weighted figures, the gain and the verdicts are withheld; the raw description follows"
        )
    else:
        print_weighted(scored, populations)
        check_run_scores(marks, load_results()[1])
    print("\n" + "\n".join(rep.report_lines(f"final ({final_row['name']})", *_columns(final))))
    print(
        "RAW is the description of this draw, not an estimate for the population; negatives are few"
    )
    for name, other in base.items():
        assert other is not None
        paired = rep.paired_report(
            [p["expected"] == p["predicted"] for p in final],
            [p["expected"] == p["predicted"] for p in other],
        )
        print(
            f"final against pinned {name}, RAW: both right {paired.both}, only the final {paired.only_a}, "
            f"only the pinned {paired.only_b}, neither {paired.neither}; exact McNemar p = "
            f"{paired.p_value:.4f}; raw gain {paired.gain} (no verdict: the pin rule is the weighted gain)"
        )


def main() -> None:
    db.env.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--time-one", action="store_true")
    mode.add_argument("--grid", action="store_true")
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="with --grid: set aside the results made with other data (kept as grid_results.previous-<fingerprint>.json; the predictions files of the old runs are overwritten as the new runs are made)",
    )
    parser.add_argument(
        "--mlflow", action="store_true", help="log every run to the local MLflow store"
    )
    args = parser.parse_args()
    if args.fresh and not args.grid:
        parser.error("--fresh only applies to --grid")
    if not torch.cuda.is_available():
        print("no CUDA device: the runs will be slow", file=sys.stderr)
    if args.report:  # the predictions kept: no model, no SOURCE
        run_report()
        return
    if args.baseline:  # marks only: no SOURCE, no replay
        marks = read_csv(MARKS)
        run_baseline(spt.load_marked_summaries(marks, read_csv(IDS)).validation, marks)
        return
    conn = connect_source()
    marked, replay, _report = build_data(conn)
    train = [*marked.train, *replay]
    scorer = validation_scorer(marked.validation, read_csv(MARKS))
    fingerprint = spt.train_fingerprint(
        train, marked.validation, *base_model(), code=code_revision()
    )
    n = len(marked.validation)
    if args.time_one:
        run_timing(train, scorer, args.mlflow)
    elif args.grid:
        run_grid(train, scorer, args.mlflow, fingerprint, n, fresh=args.fresh)
    else:
        run_final(train, scorer, args.mlflow, fingerprint, n)


if __name__ == "__main__":
    main()
