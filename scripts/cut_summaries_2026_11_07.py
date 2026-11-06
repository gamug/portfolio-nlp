"""Summaries of the backtest window, in chunks that can be stopped and resumed (T-148a, T-154, D25
step 3, D34).

    uv run scripts/cut_summaries_2026_11_07.py [--out data/cut/summaries] [--since 2021-12-31]
        [--chunk-size 250] [--max-chunks N] [--status]

Over the stored corpus (``fetch_status = 'ok'``, published ``--since`` or later) the unique bodies
(``triage.body_hash``) are listed once, in order of their first article id, in ``<out>/index.csv``.
The list is then cut into chunks of ``--chunk-size`` bodies. For each chunk the rules give every body
its scope (the same function that sized the corpus and drew the held-out set:
``size_triage_2026_10_28.classify``, with the stored NER ``ORG`` spans) and the pipeline's own local
summariser (``config/pipeline_models.json`` ``c_summary``) summarises the bodies of the scopes D25
step 3 names: single (also the single-company-dominated ones), sector, market and economy. A
``multi`` body, a round-up or a body with no company gets its scope and no summary. The chunk goes to
``<out>/chunk_NNNNN.csv``, one row per body.

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
* ``--status`` prints the chunks done and the scope counts so far, and loads no model. The scopes
  the rules never summarise (``multi``, ``roundup``, ``no_company``) are named there, with the bodies
  that had no stored NER span: read it after the first chunks, before leaving the job to run.

The input of the summariser is ``Title-<title>\\nBody-<body>``, as in ``draw_sentiment_gold``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import signal
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

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
    "summary",
    "num_chunks",
)
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
    os.replace(tmp, path)


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


def conditions(args: argparse.Namespace, index_path: Path, entities: str) -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["c_summary"]
    return {
        "since": args.since,
        "chunk_size": args.chunk_size,
        "summariser": f"{cfg['model']}@{cfg['revision']}",
        "input_format": INPUT_FORMAT,
        "index": file_digest(index_path),
        "rules": retag.code_revision(triage, size_triage),
        "ner_spans": entities,
    }


def entities_basis(conn: db.NewsNlpDatabase, since: str, manifest: Path) -> tuple[str, str]:
    """``(ner_spans, newest_id)``: the count and last id of the window's stored NER ``ORG`` spans,
    and the newest row of the table. The count is slow, so it is read again only if the table has
    a newer row than the manifest's."""
    newest = str(conn.execute(NEWEST_ENTITY_SQL).fetchone()[0])
    if manifest.exists():
        before = json.loads(manifest.read_text(encoding="utf-8"))
        if before.get("ner_newest_id") == newest:
            return before["ner_spans"], newest
    return "{}:{}".format(*conn.execute(ENTITIES_SQL, (since,)).fetchone()), newest


def check_manifest(path: Path, now: dict[str, Any], newest: str = "") -> None:
    """Write the manifest of a new job, or stop when an existing job was made under other
    conditions. ``ner_newest_id`` is bookkeeping (it moves when NER works on other articles) and
    is refreshed, not compared."""
    if not path.exists():
        path.write_text(json.dumps({**now, "ner_newest_id": newest}, indent=2), encoding="utf-8")
        return
    before = json.loads(path.read_text(encoding="utf-8"))
    if (
        newest
        and before.get("ner_newest_id") != newest
        and {k: v for k, v in before.items() if k != "ner_newest_id"} == now
    ):
        path.write_text(json.dumps({**now, "ner_newest_id": newest}, indent=2), encoding="utf-8")
        return
    before.pop("ner_newest_id", None)
    if before != now:
        changed = {k: (before.get(k), now.get(k)) for k in now if before.get(k) != now.get(k)}
        sys.exit(
            f"{path} was made under other conditions {changed}: summaries of two kinds would be "
            "mixed in one cut. Move the folder away to start a new cut."
        )


def done_chunks(out: Path, parts: Sequence[Sequence[Row]]) -> set[int]:
    """The numbers (from 1) of the chunks already written; stops if one is not its slice of the
    index."""
    done = set()
    for number, part in enumerate(parts, 1):
        path = out / chunk_name(number)
        if not path.exists():
            continue
        found = [r["body_hash"] for r in read_rows(path)]
        if found != [r["body_hash"] for r in part]:
            sys.exit(f"{path} does not hold the bodies of its slice of {INDEX_NAME}: look at it")
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
def locked(out: Path) -> Iterator[None]:
    """One run at a time in *out*: ``run.lock`` holds the pid of the run in progress. A second
    start while that process is alive stops; a lock left by a dead process is taken over."""
    path = out / LOCK_NAME
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            text = path.read_text(encoding="utf-8").strip()
            if text.isdigit() and _alive(int(text)) and int(text) != os.getpid():
                sys.exit(
                    f"another run (pid {text}) holds {path}. After Ctrl-C it still finishes its "
                    "chunk: wait for it to end, then run again."
                )
            path.unlink(missing_ok=True)  # left by a process that is gone
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
        break
    else:
        sys.exit(f"could not take {path}")
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
    log: Callable[[str], None] = print,
) -> int:
    """Make every chunk that has no file, in order, and return how many were made. Stops after
    *max_chunks* or when *should_stop* says so, always between two chunks."""
    with locked(out):
        return _make_chunks(out, parts, process, max_chunks, should_stop, log)


def _make_chunks(
    out: Path,
    parts: Sequence[Sequence[Row]],
    process: Callable[[Sequence[Row]], list[Row]],
    max_chunks: int | None,
    should_stop: Callable[[], bool],
    log: Callable[[str], None],
) -> int:
    for leftover in out.glob("*.tmp"):  # safe: the lock says no other run is writing
        leftover.unlink()
    done = done_chunks(out, parts)
    pending = [n for n in range(1, len(parts) + 1) if n not in done]
    log(f"{len(done)} of {len(parts)} chunks done, {len(pending)} to make")
    made = 0
    started = time.monotonic()
    for number in pending:
        if (max_chunks is not None and made >= max_chunks) or should_stop():
            log("stopped between chunks; run the same command to resume")
            break
        rows = process(parts[number - 1])
        atomic_write(out / chunk_name(number), CHUNK_FIELDS, rows)
        made += 1
        rate = (time.monotonic() - started) / made
        log(f"chunk {number}/{len(parts)} written; about {rate * (len(pending) - made):.0f} s left")
    return made


def status(out: Path, parts: Sequence[Sequence[Row]]) -> list[str]:
    done = done_chunks(out, parts)
    scopes: Counter[str] = Counter()
    summarised = 0
    no_spans = 0
    for number in done:
        for r in read_rows(out / chunk_name(number)):
            scopes[r["rule_scope"]] += 1
            summarised += bool(r["summary"])
            no_spans += r["spans"] == "0"
    bodies = sum(len(parts[n - 1]) for n in done)
    return [
        f"{len(done)} of {len(parts)} chunks done ({bodies:,} of {sum(map(len, parts)):,} bodies)",
        f"scopes so far: {dict(scopes.most_common())}",
        f"summaries so far: {summarised:,}; without a summary by design (rules' scope): "
        f"{ {k: v for k, v in scopes.items() if k not in SUMMARY_RULE_SCOPES} }",
        f"bodies with no stored NER span (scope read from the text alone): {no_spans:,}",
    ]


def load_summariser() -> Summariser:
    import torch  # noqa: PLC0415 -- the model is only needed to summarise
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

    from summary_stage import hierarchical_summarize_batch  # noqa: PLC0415

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["c_summary"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = model.to(device).eval()

    def summarise(texts: list[str]) -> list[tuple[str, int]]:
        return hierarchical_summarize_batch(
            texts, tokenizer, model, device, batch_size=int(cfg["batch_size"])
        )

    return summarise


def make_processor(conn: db.NewsNlpDatabase) -> Callable[[Sequence[Row]], list[Row]]:
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
                    "summary": "",
                    "num_chunks": "",
                }
            )
            if scope.scope in SUMMARY_RULE_SCOPES:
                to_summarise.append(
                    (len(rows) - 1, INPUT_FORMAT.format(title=own_title, body=body))
                )
        if to_summarise:
            if "run" not in state:
                state["run"] = load_summariser()
            done = state["run"]([text for _, text in to_summarise])
            for (position, _), (summary, n_chunks) in zip(to_summarise, done, strict=True):
                rows[position]["summary"] = summary
                rows[position]["num_chunks"] = str(n_chunks)
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


def load_or_build_index(conn: db.NewsNlpDatabase, path: Path, since: str) -> list[Row]:
    if path.exists():
        return read_rows(path)
    print(f"listing the unique bodies published {since} or later (a few minutes, once) ...")
    index = build_index(conn.execute(WINDOW_SQL, (since,)))
    atomic_write(path, INDEX_FIELDS, index)
    undated = conn.execute(NO_DATE_SQL).fetchone()[0]
    print(f"{len(index):,} unique bodies -> {path}")
    print(f"left out of the window: {undated:,} ok articles with text and no usable pub_date")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/cut/summaries"))
    parser.add_argument("--since", default="2021-12-31")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    conn = db.connect_pipeline()
    index_path = out / INDEX_NAME
    index = load_or_build_index(conn, index_path, args.since)
    entities, newest = entities_basis(conn, args.since, out / MANIFEST_NAME)
    check_manifest(out / MANIFEST_NAME, conditions(args, index_path, entities), newest)
    parts = slices(index, args.chunk_size)
    if args.status:
        print("\n".join(status(out, parts)))
        return
    stop = {"now": False}

    def ask_stop(signum: int, _frame: object) -> None:
        stop["now"] = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_DFL)  # a second signal aborts at once
        print(
            f"signal {signum}: finishing the chunk in progress, then stopping "
            "(send it again to abort now)",
            flush=True,
        )

    signal.signal(signal.SIGINT, ask_stop)
    signal.signal(signal.SIGTERM, ask_stop)
    made = run_chunks(
        out,
        parts,
        make_processor(conn),
        max_chunks=args.max_chunks,
        should_stop=lambda: stop["now"],
    )
    print("\n".join(status(out, parts)))
    print(f"{made} chunks made in this run")


if __name__ == "__main__":
    main()
