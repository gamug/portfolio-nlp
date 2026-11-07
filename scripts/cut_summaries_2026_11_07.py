"""Summaries of the backtest window, in chunks that can be stopped and resumed (T-148a, T-154, D25
step 3, D34).

    uv run scripts/cut_summaries_2026_11_07.py [--out data/cut/summaries] [--since 2021-12-31]
        [--chunk-size 250] [--max-chunks N] [--require-cuda] [--status]

Over the stored corpus (``fetch_status = 'ok'``, published ``--since`` or later) the unique bodies
(``triage.body_hash``) are listed once, in order of their first article id, in ``<out>/index.csv``.
The list is then cut into chunks of ``--chunk-size`` bodies. For each chunk the rules give every body
its scope (the same function that sized the corpus and drew the held-out set:
``size_triage_2026_10_28.classify``, with the stored NER ``ORG`` spans) and the pipeline's own local
summariser (``config/pipeline_models.json`` ``c_summary``) summarises the bodies of the scopes D25
step 3 names: single (also the single-company-dominated ones), sector, market and economy. A
``multi`` body, a round-up or a body with no company gets its scope and no summary. The chunk goes to
``<out>/chunk_NNNNN.csv``, one row per body.

**Settings.** ``SOURCE_DATABASE_URL`` (the crawl DB with ``articles.body_text``) and ``DATABASE_URL`` come
from the repo's ``.env`` or from the shell (the shell wins); see ``.env.example``.

**What you see.** The first line says whether the summariser will use CUDA (the GPU's name) or the CPU;
``--require-cuda`` stops before anything is written if there is no CUDA. A progress bar counts the
bodies of the whole cut (chunks already on disk count as done) and moves when a chunk is saved, with
the device beside it; it is off when stderr is not a terminal, and the per-chunk lines still print.
The model is loaded inside the first chunk (download included), so the bar does not move until that
chunk is saved, several minutes in: that is not a hang. The device line, the bar, the model and the
manifest all take the one decision ``main`` makes.

**It can be stopped at any moment and run again with the same command.**

* A chunk file is written to a temporary name and renamed when complete, so a file that exists is a
  whole chunk; a leftover temporary file is deleted at the start. A kill loses at most the chunk
  being made (``--chunk-size`` bodies: about 3.5 minutes at 0.84 s per body).
* ``Ctrl-C`` (or ``SIGTERM``) finishes the chunk in progress, then stops.
* A run skips every chunk that exists, so running again never summarises a body twice and a
  finished job does nothing. A chunk whose bodies are not the ones of its slice of the index stops
  the run.
* ``<out>/manifest.json`` records the conditions of the job: the window, the chunk size, the
  summariser ``model@revision``, the input format, a digest of the index and of the rules' source. A
  run under other conditions stops instead of mixing two kinds of summary in one cut: move the folder
  away to start a new cut.
* Only one run at a time: ``<out>/run.lock`` holds the pid of the run in progress; a second start
  while it is alive stops (after Ctrl-C the first run still finishes its chunk: wait for it). A lock
  whose process is gone is taken over. Temporary files carry the pid of their writer.
* A second ``Ctrl-C`` aborts at once (the atomic write makes that safe).
* If the window's stored NER spans change between runs (D25 runs NER on the window), the run is
  refused; ``--accept-new-spans`` resumes and keeps the chunks made (every row carries ``spans``),
  recording the old value in the manifest. ``--break-lock`` removes ``run.lock`` first (a pid reused
  after a restart).
* A row's ``status`` is ``summarised``, ``no_summary_by_scope``, ``summary_empty`` (a summary that
  came back empty twice) or ``summary_failed`` (the model raised for that body alone; ``error`` says
  what): never silent, counted by ``--status``. One bad body never stops the job. The chunk file is
  fsynced before its name appears, and a chunk with a missing summary and no such status is refused.
* The manifest also holds the torch / transformers versions and the device; a change refuses, and
  ``--accept-environment-change`` keeps the chunks made.
* ``read_cut_summaries(out)`` is the one way to read a cut (manifest, pinned summariser, index and
  chunks all checked). Columns: ``body_hash``, ``article_ids``, ``rule_scope``, ``gold_scope``,
  ``spans``, ``status``, ``summary``, ``num_chunks``.
* ``--redo-failed`` makes again the chunks that hold a ``summary_failed`` / ``summary_empty`` row (the
  old file waits in ``redo/`` until the new one is written). The lock covers the manifest too: a
  second start can neither change it nor the chunks while a run is in progress.
* The manifest is written atomically like the chunks, and an index with no manifest beside it (a first
  run killed in between) is listed again, unless chunks exist (also in ``redo/``): then the run stops, and so it does when the
  index is missing and chunks exist; the missing file has to be restored. ``--status`` slices the index by the manifest's chunk size.
* ``--status`` reads only (no lock, no manifest, no index built); it prints the chunks done and the scope counts so far, and loads no model. The scopes
  the rules never summarise (``multi``, ``roundup``, ``no_company``) are named there, with the bodies
  that had no stored NER span: read it after the first chunks, before leaving the job to run.

The input of the summariser is ``Title-<title>\\nBody-<body>``, as in ``draw_sentiment_gold``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
import signal
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import size_triage_2026_10_28 as size_triage

import news_nlp as db
from news_nlp import retag, triage

CONFIG = Path("config/pipeline_models.json")
INDEX_NAME = "index.csv"
MANIFEST_NAME = "manifest.json"
INDEX_FIELDS = ("body_hash", "article_ids")
CHUNK_FIELDS = (
    "body_hash",
    "article_ids",
    "rule_scope",
    "gold_scope",
    "spans",
    "status",
    "summary",
    "num_chunks",
    "error",
)
#: ``status`` of a row: what became of its summary.
SUMMARISED, BY_SCOPE, EMPTY, FAILED = (
    "summarised",
    "no_summary_by_scope",
    "summary_empty",
    "summary_failed",
)
#: Keys of the manifest that are bookkeeping, not conditions of the job.
BOOKKEEPING = ("ner_newest_id", "ner_spans_history")
LOCK_NAME = "run.lock"
#: Rule scopes whose bodies get a summary (D25 step 3; ``multi_dominated`` is single + mentions, D23).
SUMMARY_RULE_SCOPES = frozenset({"single", "multi_dominated", "sector", "market", "economy"})
INPUT_FORMAT = "Title-{title}\nBody-{body}"
WINDOW_SQL = (
    "SELECT id, body_text FROM source.articles WHERE fetch_status = 'ok' AND body_text IS NOT NULL "
    "AND body_text != '' AND pub_date >= ? ORDER BY id"
)

NO_DATE_SQL = (
    "SELECT COUNT(*) FROM source.articles WHERE fetch_status = 'ok' AND body_text IS NOT NULL "
    "AND body_text != '' AND (pub_date IS NULL OR pub_date NOT GLOB '[0-9][0-9][0-9][0-9]-*')"
)
#: What the scope reads besides the code: how many stored NER ``ORG`` spans the window has, and the
#: last one. If the NER stage writes for the window between two runs, the scopes would differ.
#: The newest NER row of the whole table: instant. If it is the one in the manifest nothing was
#: written since, and the slow count below (about two minutes) is not repeated.
NEWEST_ENTITY_SQL = "SELECT COALESCE(MAX(id), 0) FROM article_entities"
ENTITIES_SQL = (
    "SELECT COUNT(*), COALESCE(MAX(e.id), 0) FROM article_entities e "
    "JOIN source.articles a ON a.id = e.article_id "
    "WHERE e.entity_type = 'ORG' AND a.fetch_status = 'ok' AND a.pub_date >= ?"
)

Row = dict[str, str]
#: Summarises texts: returns (summary, number of leaf chunks) in the same order.
Summariser = Callable[[list[str]], list[tuple[str, int]]]


def chunk_name(number: int) -> str:
    return f"chunk_{number:05d}.csv"


def atomic_write(path: Path, fields: Sequence[str], rows: Iterable[Row]) -> None:
    """Write *rows* so that *path* either does not exist or is whole: a temporary file first, then
    ``os.replace``."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())  # the content is on disk before the name appears
    _publish(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    """:func:`atomic_write` for a text file (the manifest): whole or absent, never truncated."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    _publish(tmp, path)


def _publish(tmp: Path, path: Path) -> None:
    os.replace(tmp, path)
    with contextlib.suppress(OSError):  # a directory cannot be synced on every file system
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def load_manifest(path: Path) -> dict[str, Any]:
    """The manifest, or a stop that says what to do if the file cannot be read (it is written
    atomically now; a file from before, or damaged by hand, can still be truncated)."""
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{path} cannot be read ({exc}). If the chunks next to it are yours, restore it from a "
            "copy. Do not just delete it: with chunks beside it the run refuses to start without "
            "it, because the index they were made from cannot be told from a new listing."
        ) from exc
    return loaded


def read_rows(path: Path) -> list[Row]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def build_index(pairs: Iterable[tuple[int, str]]) -> list[Row]:
    """The unique bodies of ``(article_id, body)`` pairs, in order of their first article id, with
    every article id that holds the body."""
    members: dict[str, list[int]] = defaultdict(list)
    for article_id, body in pairs:
        if body.strip():
            members[triage.body_hash(body)].append(int(article_id))
    ordered = sorted(members.items(), key=lambda item: item[1][0])
    return [
        {"body_hash": key, "article_ids": " ".join(str(i) for i in ids)} for key, ids in ordered
    ]


def slices(index: Sequence[Row], size: int) -> list[list[Row]]:
    return [list(index[start : start + size]) for start in range(0, len(index), size)]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def environment(device_name: str | None = None) -> str:
    """The versions and device the summaries are made with: beam search over padded batches can
    differ between GPU and CPU and between releases, so a change is a condition of the job.
    *device_name* is the one ``main`` printed and gave the model; ``None`` reads it from torch."""
    import torch  # noqa: PLC0415 -- only to read the device

    versions = ", ".join(f"{p} {importlib.metadata.version(p)}" for p in ("torch", "transformers"))
    device = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    return f"{versions}, {device}"


def device_banner() -> tuple[bool, str]:
    """Whether the summariser will run on CUDA, and the line that says so (printed before anything
    slow starts, and shown beside the progress bar). The summariser picks its device the same way
    (``torch.cuda.is_available()``), so this is what it will use."""
    import torch  # noqa: PLC0415 -- only to read the device

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return True, f"device: CUDA ({name}, CUDA {torch.version.cuda}, torch {torch.__version__})"
    return False, (
        f"device: CPU, no CUDA (torch {torch.__version__}): about 0.84 s per body is the GPU "
        "figure; the CPU is far slower. --require-cuda stops instead of running on it"
    )


def conditions(
    args: argparse.Namespace, index_path: Path, entities: str, device_name: str | None = None
) -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["c_summary"]
    return {
        "since": args.since,
        "chunk_size": args.chunk_size,
        "summariser": f"{cfg['model']}@{cfg['revision']}",
        "input_format": INPUT_FORMAT,
        "index": file_digest(index_path),
        "rules": retag.code_revision(triage, size_triage),
        "ner_spans": entities,
        "environment": environment(device_name),
    }


def entities_basis(conn: db.NewsNlpDatabase, since: str, manifest: Path) -> tuple[str, str]:
    """``(ner_spans, newest_id)``: the count and last id of the window's stored NER ``ORG`` spans,
    and the newest row of the table. The count is slow, so it is read again only if the table has
    a newer row than the manifest's."""
    newest = str(conn.execute(NEWEST_ENTITY_SQL).fetchone()[0])
    if manifest.exists():
        before = load_manifest(manifest)
        if before.get("ner_newest_id") == newest:
            return before["ner_spans"], newest
    return "{}:{}".format(*conn.execute(ENTITIES_SQL, (since,)).fetchone()), newest


def check_manifest(
    path: Path,
    now: dict[str, Any],
    newest: str = "",
    *,
    accept_new_spans: bool = False,
    accept_environment_change: bool = False,
    chunks_done: int = 0,
) -> None:
    """Write the manifest of a new job, or stop when an existing job was made under other
    conditions. ``ner_newest_id`` moves when NER works on other articles and is refreshed, not
    compared. Two conditions may be accepted by the owner, and only when nothing else changed: the
    window's NER spans (D25 runs NER on the window) and the environment (versions, device). The
    chunks made are kept (every row carries ``spans``), and the old value and how many chunks were
    made under it go to ``ner_spans_history``; every other change refuses (summaries of two kinds
    in one cut)."""
    if not path.exists():
        atomic_text(path, json.dumps({**now, "ner_newest_id": newest}, indent=2))
        return
    before = load_manifest(path)
    history = before.get("ner_spans_history", [])
    core = {k: v for k, v in before.items() if k not in BOOKKEEPING}
    if core != now:
        changed = {k: (core.get(k), now.get(k)) for k in now if core.get(k) != now.get(k)}
        allowed = {
            key
            for key, ok in (
                ("ner_spans", accept_new_spans),
                ("environment", accept_environment_change),
            )
            if ok
        }
        if set(changed) <= allowed:
            history = [
                *history,
                {**{k: core.get(k) for k in changed}, "chunks_made_under_it": chunks_done},
            ]
        else:
            flags = {
                "ner_spans": "--accept-new-spans",
                "environment": "--accept-environment-change",
            }
            hint = (
                f" Only {' and '.join(flags[k] for k in changed)} would keep the chunks made."
                if set(changed) <= set(flags)
                else ""
            )
            sys.exit(
                f"{path} was made under other conditions {changed}: summaries of two kinds would "
                f"be mixed in one cut.{hint} Otherwise move the folder away to start a new cut."
            )
    elif not newest or before.get("ner_newest_id") == newest:
        return
    kept = {"ner_spans_history": history} if history else {}
    atomic_text(path, json.dumps({**now, "ner_newest_id": newest, **kept}, indent=2))


def row_is_sound(row: Row) -> bool:
    """A row whose scope asks for a summary has one, or says why not; any other row has none."""
    if row["rule_scope"] in SUMMARY_RULE_SCOPES:
        if row["status"] == SUMMARISED:
            return bool(row["summary"])
        return row["status"] in (EMPTY, FAILED)
    return row["status"] == BY_SCOPE and not row["summary"]


def done_chunks(out: Path, parts: Sequence[Sequence[Row]]) -> set[int]:
    """The numbers (from 1) of the chunks already written; stops if one is not its slice of the
    index."""
    done = set()
    for number, part in enumerate(parts, 1):
        path = out / chunk_name(number)
        if not path.exists():
            continue
        rows = read_rows(path)
        if [r["body_hash"] for r in rows] != [r["body_hash"] for r in part]:
            sys.exit(f"{path} does not hold the bodies of its slice of {INDEX_NAME}: look at it")
        broken = [r["body_hash"] for r in rows if not row_is_sound(r)]
        if broken:
            sys.exit(
                f"{path}: {len(broken)} rows are not sound (a summary missing without "
                f"{EMPTY} / {FAILED}, or a summary on a scope that gets none; first {broken[0]}): "
                "the file is damaged; delete it (and its copy in redo/, if any) to make the chunk again"
            )
        done.add(number)
    return done


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def locked(out: Path, *, break_lock: bool = False) -> Iterator[None]:
    """One run at a time in *out*: ``run.lock`` holds the pid of the run in progress. A second
    start while that process is alive stops; a lock left by a dead process is taken over. The pid is
    written to a private file that is then hard-linked to ``run.lock``, so the lock never exists
    without its content. ``break_lock`` removes a lock first (a pid reused after a restart)."""
    path = out / LOCK_NAME
    private = out / f"{LOCK_NAME}.{os.getpid()}"
    for stale in out.glob(f"{LOCK_NAME}.*"):  # left by a run killed between write and unlink
        suffix = stale.name.rsplit(".", 1)[1]
        if suffix.isdigit() and int(suffix) != os.getpid() and not _alive(int(suffix)):
            stale.unlink(missing_ok=True)
    private.write_text(str(os.getpid()), encoding="utf-8")
    try:
        if break_lock:
            path.unlink(missing_ok=True)
        for _ in range(2):
            try:
                os.link(private, path)
            except FileExistsError:
                text = path.read_text(encoding="utf-8").strip()
                if text.isdigit() and _alive(int(text)) and int(text) != os.getpid():
                    sys.exit(
                        f"another run (pid {text}) holds {path}. After Ctrl-C it still finishes "
                        f"its chunk: wait for it to end, then run again. If you are sure none is "
                        f"running (a pid reused after a restart), run with --break-lock."
                    )
                path.unlink(missing_ok=True)  # left by a process that is gone
                continue
            break
        else:
            sys.exit(f"could not take {path}")
    finally:
        private.unlink(missing_ok=True)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def run_chunks(
    out: Path,
    parts: Sequence[Sequence[Row]],
    process: Callable[[Sequence[Row]], list[Row]],
    *,
    max_chunks: int | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = tqdm.write,  # does not break the bar
    break_lock: bool = False,
    hold_lock: bool = True,
    redo_failed: bool = False,
    device_label: str = "",
) -> int:
    """Make every chunk that has no file, in order, and return how many were made. Stops after
    *max_chunks* or when *should_stop* says so, always between two chunks. The lock is taken here
    unless the caller already holds it (``hold_lock=False``: ``main`` locks before the manifest).
    With *redo_failed* the chunks that hold a failed or empty summary are moved to ``redo/`` first
    and made again. *device_label* (``cuda`` / ``cpu``) is shown on the progress bar."""
    if not hold_lock:
        return _make_chunks(
            out, parts, process, max_chunks, should_stop, log, redo_failed, device_label
        )
    with locked(out, break_lock=break_lock):
        return _make_chunks(
            out, parts, process, max_chunks, should_stop, log, redo_failed, device_label
        )


def set_aside_failed(out: Path, parts: Sequence[Sequence[Row]], log: Callable[[str], None]) -> int:
    """Move every chunk that holds a ``summary_failed`` / ``summary_empty`` row to ``<out>/redo/``
    so the loop makes it again; the old file stays there until the new one is written (a kill in
    between loses nothing). Returns the number of bodies to try again."""
    moved = bodies = 0
    for number in done_chunks(out, parts):
        path = out / chunk_name(number)
        gaps = sum(r["status"] in (EMPTY, FAILED) for r in read_rows(path))
        if gaps:
            (out / "redo").mkdir(exist_ok=True)
            os.replace(path, out / "redo" / chunk_name(number))
            moved += 1
            bodies += gaps
    if moved:
        log(f"{bodies} bodies with no summary in {moved} chunks: the chunks are made again")
    return bodies


def _make_chunks(
    out: Path,
    parts: Sequence[Sequence[Row]],
    process: Callable[[Sequence[Row]], list[Row]],
    max_chunks: int | None,
    should_stop: Callable[[], bool],
    log: Callable[[str], None],
    redo_failed: bool = False,
    device_label: str = "",
) -> int:
    for leftover in out.glob("*.tmp"):  # safe: the lock says no other run is writing
        leftover.unlink()
    if redo_failed:
        set_aside_failed(out, parts, log)
    done = done_chunks(out, parts)
    pending = [n for n in range(1, len(parts) + 1) if n not in done]
    log(f"{len(done)} of {len(parts)} chunks done, {len(pending)} to make")
    made = 0
    started = time.monotonic()
    # one bar over the bodies of the whole cut, counting the chunks already on disk; it moves when a
    # chunk is written (a chunk is the unit that is saved), and is off when stderr is not a terminal
    bar = tqdm(
        total=sum(len(part) for part in parts),
        initial=sum(len(parts[n - 1]) for n in done),
        desc="summaries",
        unit="body",
        disable=None,
        postfix={"device": device_label} if device_label else None,
    )
    try:
        for number in pending:
            if (max_chunks is not None and made >= max_chunks) or should_stop():
                log("stopped between chunks; run the same command to resume")
                break
            rows = _remake(out, number, parts[number - 1], process)
            atomic_write(out / chunk_name(number), CHUNK_FIELDS, rows)
            (out / "redo" / chunk_name(number)).unlink(missing_ok=True)  # the new file replaces it
            made += 1
            bar.update(len(parts[number - 1]))
            rate = (time.monotonic() - started) / made
            log(
                f"chunk {number}/{len(parts)} written ({len(rows)} bodies, "
                f"{rate / max(len(rows), 1):.2f} s per body); about "
                f"{rate * (len(pending) - made):.0f} s left"
            )
    finally:
        bar.close()
    return made


def _remake(
    out: Path,
    number: int,
    part: Sequence[Row],
    process: Callable[[Sequence[Row]], list[Row]],
) -> list[Row]:
    """The rows of a chunk. If an earlier version waits in ``redo/``, only the bodies without a
    summary are processed again and the others keep the rows they had."""
    old_path = out / "redo" / chunk_name(number)
    if not old_path.exists():
        return process(part)
    old = {r["body_hash"]: r for r in read_rows(old_path)}
    if list(old) != [e["body_hash"] for e in part]:
        sys.exit(f"{old_path} does not hold the bodies of its slice of {INDEX_NAME}: look at it")
    todo = [e for e in part if old[e["body_hash"]]["status"] in (EMPTY, FAILED)]
    fresh = {r["body_hash"]: r for r in process(todo)} if todo else {}
    return [fresh.get(e["body_hash"], old[e["body_hash"]]) for e in part]


def status(out: Path, parts: Sequence[Sequence[Row]]) -> list[str]:
    done = done_chunks(out, parts)
    scopes: Counter[str] = Counter()
    summarised = 0
    no_spans = 0
    empty = 0
    failed = 0
    for number in done:
        for r in read_rows(out / chunk_name(number)):
            scopes[r["rule_scope"]] += 1
            summarised += bool(r["summary"])
            no_spans += r["spans"] == "0"
            empty += r["status"] == EMPTY
            failed += r["status"] == FAILED
    bodies = sum(len(parts[n - 1]) for n in done)
    gaps = (
        [f"{empty + failed:,} bodies have no summary: run again with --redo-failed to retry them"]
        if empty + failed
        else []
    )
    return [
        *gaps,
        f"{len(done)} of {len(parts)} chunks done ({bodies:,} of {sum(map(len, parts)):,} bodies)",
        f"scopes so far: {dict(scopes.most_common())}",
        f"summaries so far: {summarised:,}; without a summary by design (rules' scope): "
        f"{ {k: v for k, v in scopes.items() if k not in SUMMARY_RULE_SCOPES} }",
        f"bodies with no stored NER span (scope read from the text alone): {no_spans:,}",
        f"rows whose scope asks for a summary but came back empty: {empty:,}; failed: {failed:,}",
    ]


def load_summariser(device_name: str | None = None) -> Summariser:
    """The summariser on *device_name* (``cuda`` / ``cpu``: the one ``main`` printed and put on the
    bar); ``None`` picks CUDA when it is there."""
    import torch  # noqa: PLC0415 -- the model is only needed to summarise
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

    from summary_stage import hierarchical_summarize_batch  # noqa: PLC0415

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["c_summary"]
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = model.to(device).eval()
    tqdm.write(f"summariser {cfg['model']}@{cfg['revision'][:8]} loaded on {device.type}")

    def summarise(texts: list[str]) -> list[tuple[str, int]]:
        try:
            return hierarchical_summarize_batch(
                texts, tokenizer, model, device, batch_size=int(cfg["batch_size"])
            )
        except RuntimeError:  # an out-of-memory leaves blocks cached: free them for the retry
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise

    return summarise


def summarise_with_retry(summarise: Summariser, texts: list[str]) -> list[tuple[str, int, str]]:
    """Summarise *texts*; returns ``(summary, leaf chunks, error)`` for each. The ones that come
    back empty are tried once more together. If the model raises for the batch, every text is tried
    on its own, so one bad body costs only itself: it gets an empty summary and the error text
    (``Exception`` only: a Ctrl-C or ``sys.exit`` is never swallowed)."""
    try:
        done = [(s, n, "") for s, n in summarise(texts)]
    except Exception:
        done = []
        for text in texts:
            try:
                ((summary, n),) = summarise([text])
                done.append((summary, n, ""))
            except Exception as exc:
                done.append(("", 0, f"{type(exc).__name__}: {str(exc)[:160]}"))
    again = [i for i, (summary, _, error) in enumerate(done) if not summary and not error]
    if again:
        for i, (summary, n) in zip(again, summarise([texts[i] for i in again]), strict=True):
            done[i] = (summary, n, "")
    return done


def make_processor(
    conn: db.NewsNlpDatabase, device_name: str | None = None
) -> Callable[[Sequence[Row]], list[Row]]:
    """The function that turns a slice of the index into chunk rows: scope by the rules, a summary
    where D25 step 3 asks for one. The model is loaded the first time a chunk needs it."""
    universe = size_triage.load_universe(conn)
    resolver = triage.CompanyResolver(universe)
    terms = list(size_triage.sector_terms(universe))
    state: dict[str, Summariser] = {}

    def process(part: Sequence[Row]) -> list[Row]:
        rows: list[Row] = []
        to_summarise: list[tuple[int, str]] = []
        for entry in part:
            ids = [int(i) for i in entry["article_ids"].split()]
            own_title, body, own_slug, headlines = fetch_body(conn, ids)
            if triage.body_hash(body) != entry["body_hash"]:
                sys.exit(f"article {ids[0]}: the body no longer matches the index")
            # the round-up test reads the first headline that looks like one; the summary reads the
            # article's own title (as draw_sentiment_gold does)
            headline, slug = size_triage.roundup_headline(headlines) or (own_title, own_slug)
            spans = first_spans(conn, ids)
            scope, _ = size_triage.classify(
                resolver, headline, body, slug, spans, terms, **size_triage.FULL
            )
            rows.append(
                {
                    "body_hash": entry["body_hash"],
                    "article_ids": entry["article_ids"],
                    "rule_scope": scope.scope,
                    "gold_scope": triage.RULE_SCOPE_TO_GOLD[scope.scope],
                    "spans": str(len(spans)),
                    "status": SUMMARISED if scope.scope in SUMMARY_RULE_SCOPES else BY_SCOPE,
                    "summary": "",
                    "num_chunks": "",
                    "error": "",
                }
            )
            if scope.scope in SUMMARY_RULE_SCOPES:
                to_summarise.append(
                    (len(rows) - 1, INPUT_FORMAT.format(title=own_title, body=body))
                )
        if to_summarise:
            if "run" not in state:
                state["run"] = load_summariser(device_name)
            done = summarise_with_retry(state["run"], [text for _, text in to_summarise])
            for (position, _), (summary, n_chunks, error) in zip(to_summarise, done, strict=True):
                rows[position]["summary"] = summary
                rows[position]["num_chunks"] = str(n_chunks) if summary else ""
                rows[position]["error"] = error
                if error:
                    rows[position]["status"] = FAILED  # never silent: counted by --status
                elif not summary:
                    rows[position]["status"] = EMPTY
        return rows

    return process


def fetch_body(
    conn: db.NewsNlpDatabase, ids: Sequence[int]
) -> tuple[str, str, str, list[tuple[str, str]]]:
    """The title, body and slug of the first article of a body, and every (title, slug) the body
    sits under."""
    headlines: list[tuple[str, str]] = []
    first: tuple[str, str, str] | None = None
    for article_id in ids:
        row = conn.execute(
            "SELECT a.title, a.body_text, d.url FROM source.articles a "
            "LEFT JOIN source.discovered_urls d ON d.id = a.id WHERE a.id = ?",
            (article_id,),
        ).fetchone()
        item = (row[0] or "", row[1] or "", size_triage.slug_of(row[2]))
        first = first or item
        headlines.append((item[0], item[2]))
    assert first is not None
    return first[0], first[1], first[2], headlines


def first_spans(conn: db.NewsNlpDatabase, ids: Sequence[int]) -> list[str]:
    """The stored NER ``ORG`` spans of ONE article of the body (its pairs share the text, so
    merging them would count every mention once per pair), as ``size_triage`` does."""
    for article_id in ids:
        spans = [
            text
            for (text,) in conn.execute(
                "SELECT text FROM article_entities WHERE entity_type = 'ORG' AND article_id = ?",
                (article_id,),
            )
        ]
        if spans:
            return spans
    return []


class CutSummariesError(Exception):
    """The cut cannot be read as a whole and valid set of summaries."""


def _read_valid_chunk(path: Path, part: Sequence[Row]) -> list[Row]:
    """One chunk file of a cut, checked: its columns, its slice of the index, every row sound."""
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != CHUNK_FIELDS:
            raise CutSummariesError(f"{path.name} does not have the columns {CHUNK_FIELDS}")
        chunk = list(reader)
    if [r["body_hash"] for r in chunk] != [r["body_hash"] for r in part]:
        raise CutSummariesError(f"{path.name} is not its slice of {INDEX_NAME}")
    if broken := [r["body_hash"] for r in chunk if not row_is_sound(r)]:
        raise CutSummariesError(
            f"{path.name}: {len(broken)} rows are not sound (first {broken[0]}): damaged"
        )
    return chunk


def read_cut_summaries(
    out: Path, *, allow_partial: bool = False, allow_gaps: bool = False
) -> dict[str, Row]:
    """The rows of a cut by ``body_hash``, after checking that they can be trusted: a manifest
    exists, its summariser is the one pinned in ``config/pipeline_models.json`` today, the index is
    the one the manifest digests, every chunk file has the columns of :data:`CHUNK_FIELDS` and holds
    its slice of the index, and (unless *allow_partial*) every chunk exists. Raises
    :class:`CutSummariesError` naming the first problem. Every row must be sound (the writer's own
    test, :func:`row_is_sound`); rows whose summary failed or came back empty are refused unless
    *allow_gaps*, which returns them with their ``status`` so the caller has to look at it."""
    manifest_path = out / MANIFEST_NAME
    if not manifest_path.exists():
        raise CutSummariesError(f"{manifest_path} not found: not a cut made by this script")
    try:
        manifest = load_manifest(manifest_path)
    except SystemExit as exc:
        raise CutSummariesError(str(exc)) from exc
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["c_summary"]
    pinned = f"{cfg['model']}@{cfg['revision']}"
    if manifest["summariser"] != pinned:
        raise CutSummariesError(f"made with {manifest['summariser']}, the pipeline pins {pinned}")
    index_path = out / INDEX_NAME
    if file_digest(index_path) != manifest["index"]:
        raise CutSummariesError(f"{INDEX_NAME} is not the one the manifest digests")
    parts = slices(read_rows(index_path), int(manifest["chunk_size"]))
    rows: dict[str, Row] = {}
    for number, part in enumerate(parts, 1):
        path = out / chunk_name(number)
        if not path.exists():
            if allow_partial:
                continue
            raise CutSummariesError(f"{path.name} is missing: the cut is not finished")
        rows.update({r["body_hash"]: r for r in _read_valid_chunk(path, part)})
    gaps = [r for r in rows.values() if r["status"] in (EMPTY, FAILED)]
    if gaps and not allow_gaps:
        raise CutSummariesError(
            f"{len(gaps)} bodies have no summary ({EMPTY} / {FAILED}); run the job again with "
            "--redo-failed, or read with allow_gaps=True and look at each row's status"
        )
    return rows


def chunk_files_exist(out: Path) -> bool:
    """Any chunk of the cut, also one set aside in ``redo/``."""
    return any(out.glob("chunk_*.csv")) or any((out / "redo").glob("chunk_*.csv"))


def load_or_build_index(
    conn: db.NewsNlpDatabase,
    path: Path,
    since: str,
    *,
    manifest_exists: bool = True,
    chunks_exist: bool = False,
) -> list[Row]:
    """The index of the job. One with no manifest next to it (a first run killed in between) does
    not say which ``--since`` it was listed for, so it is listed again, but only when no chunk
    exists: chunks without their manifest were made from an index that a new listing could
    replace by another one, so the run stops and the manifest has to be restored."""
    if chunks_exist and not (manifest_exists and path.exists()):
        sys.exit(
            f"chunks exist in {path.parent} but {MANIFEST_NAME} or {INDEX_NAME} does not: the index "
            "and the conditions they were made under are not known, and a new listing could differ. "
            "Restore the missing file from a copy, or move the folder away to start a new cut. "
            "Nothing was changed."
        )
    if path.exists() and manifest_exists:
        return read_rows(path)
    print(f"listing the unique bodies published {since} or later (a few minutes, once) ...")
    index = build_index(conn.execute(WINDOW_SQL, (since,)))
    atomic_write(path, INDEX_FIELDS, index)
    undated = conn.execute(NO_DATE_SQL).fetchone()[0]
    print(f"{len(index):,} unique bodies -> {path}")
    print(f"left out of the window: {undated:,} ok articles with text and no usable pub_date")
    return index


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_environment(env_file: Path = ENV_FILE) -> None:
    """``SOURCE_DATABASE_URL`` / ``DATABASE_URL`` from the repo's ``.env``, as every other entrypoint
    gets them (``pipeline`` loads it at import; this script does not import ``pipeline``). A
    variable already set in the shell wins."""
    load_dotenv(env_file, override=False)


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/cut/summaries"))
    parser.add_argument("--since", default="2021-12-31")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--accept-new-spans",
        action="store_true",
        help="resume although the window's stored NER spans changed (the old chunks are kept)",
    )
    parser.add_argument(
        "--accept-environment-change",
        action="store_true",
        help="resume although torch / transformers / the device changed (the chunks are kept)",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="stop before anything is written if CUDA is not available (the device is a condition "
        "of the job: a first run on the CPU would fix the cut to it)",
    )
    parser.add_argument(
        "--break-lock",
        action="store_true",
        help="remove run.lock first (a pid reused after a restart)",
    )
    parser.add_argument(
        "--redo-failed",
        action="store_true",
        help="make again the chunks that hold a failed or empty summary (the old file is kept in redo/)",
    )
    args = parser.parse_args()
    out: Path = args.out
    index_path = out / INDEX_NAME
    if args.status:  # reads only: no lock, no manifest, no index built
        if not index_path.exists():
            print(f"no cut in {out} yet")
            return
        manifest = out / MANIFEST_NAME
        size = int(load_manifest(manifest)["chunk_size"]) if manifest.exists() else args.chunk_size
        print("\n".join(status(out, slices(read_rows(index_path), size))))
        return
    stop = {"now": False}

    def ask_stop(signum: int, _frame: object) -> None:
        stop["now"] = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_DFL)  # a second signal aborts at once
        tqdm.write(
            f"signal {signum}: finishing the chunk in progress, then stopping "
            "(send it again to abort now)"
        )

    has_cuda, banner = device_banner()
    print(banner, flush=True)
    if args.require_cuda and not has_cuda:  # before the folder is made: nothing is written
        sys.exit("--require-cuda: CUDA is not available here, nothing was written")
    device_name = "cuda" if has_cuda else "cpu"  # the one decision: banner, bar and model
    out.mkdir(parents=True, exist_ok=True)
    with locked(out, break_lock=args.break_lock):  # everything that changes the cut is under it
        conn = db.connect_pipeline()
        index = load_or_build_index(
            conn,
            index_path,
            args.since,
            manifest_exists=(out / MANIFEST_NAME).exists(),
            chunks_exist=chunk_files_exist(out),
        )
        entities, newest = entities_basis(conn, args.since, out / MANIFEST_NAME)
        parts = slices(index, args.chunk_size)
        check_manifest(
            out / MANIFEST_NAME,
            conditions(args, index_path, entities, device_name),
            newest,
            accept_new_spans=args.accept_new_spans,
            accept_environment_change=args.accept_environment_change,
            chunks_done=len(done_chunks(out, parts)),
        )
        signal.signal(signal.SIGINT, ask_stop)
        signal.signal(signal.SIGTERM, ask_stop)
        made = run_chunks(
            out,
            parts,
            make_processor(conn, device_name),
            max_chunks=args.max_chunks,
            should_stop=lambda: stop["now"],
            hold_lock=False,
            redo_failed=args.redo_failed,
            device_label=device_name,
        )
    print("\n".join(status(out, parts)))
    print(f"{made} chunks made in this run")


if __name__ == "__main__":
    main()
