"""The LLM-as-judge measure of the summary step (step 2): how faithful, how complete and how compact the summaries are, for the single
bodies and for the sub-news, by the length of the source.

    # from a COPY of the databases, made once the summary run has ended (a live file refuses a read-only open while a step writes):
    #   python -c "import sqlite3; s = sqlite3.connect('/path/nlp.db'); d = sqlite3.connect('nlp_copy.db'); s.backup(d)"
    uv run python scripts/summary_judge_2026_11_13.py draw --results-db RESULTS.db --source-db SOURCE.db \\
        --summary-revision summary-... --scope-revision scope-nli-... --cut-revision units5-... [--per-group 150]
    uv run python scripts/summary_judge_2026_11_13.py judge          # needs LLM_API_KEY, LLM_MODEL and LLM_URL (.env)
    uv run python scripts/summary_judge_2026_11_13.py report
    uv run python scripts/summary_judge_2026_11_13.py tag-sheet --verdicts data/summary_judge/verdicts.jsonl   # the owner's slice, the judge hidden
    uv run python scripts/summary_judge_2026_11_13.py agree --tags data/summary_judge/to_tag.csv

**The frame.** Only the items the summariser actually summarised (``news_summary.text_source = 'summary'``): the ones under the size are
stored as they are, so they are trivially faithful and would inflate every rate. Two groups: ``single`` (the single bodies of the named scope
revision) and ``multi`` (the sub-news of the named cut revision). Each group is cut in three buckets by the characters of the source the summariser
read (terciles of the population; ``--bucket-by chunks`` cuts them by the chunk count instead, which collapses on this corpus), and the draw takes the same
number from each bucket. A bucket's weight is its population over its sample, so the overall rate is weighted and says it.

**The judge** reads the source exactly as the summariser did (a single body as ``Title-<title>\\nBody-<body>``, a sub-news alone, because its
title is the whole round-up's) and the summary, splits the summary into claims and marks each ``supported``, ``unsupported`` or
``contradicted`` by the source alone, and says whether the main event of the source is in the summary. The rates are per summary (no
unsupported or contradicted claim, any contradicted claim, the main event covered) and per claim (the share of claims that are unsupported).
Two measures need no judge: whether the summary keeps the companies the source names (the matcher of step 3, ``direct_mentions``), and its
compression. The figures are the judge's: ``tag-sheet`` writes a slice for the owner to mark, without the judge's verdicts, and ``agree``
compares the two. A random slice cannot measure the judge on a rare event (if 95% of the summaries are clean, 40 of them hold two bad ones), so the
slice holds the items the judge flagged and a random control of clean ones; ``agree`` then gives the owner's rate of unfaithful summaries among the
flagged ones (the judge's precision) and among the clean ones, and from them the judge's sensitivity, weighed back to the sample.

Every verdict carries the judge model, the hash of the prompt and the summary revision; a verdict made under another model or prompt is judged again by a
resumed run, and ``report`` refuses a file that holds two judges. A provider error (a wrong key, a rate limit, a network drop) is not a verdict: it is not
written, and ``MAX_CONSECUTIVE_ERRORS`` in a row stop the run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sqlite3
import statistics
import sys
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
csv.field_size_limit(sys.maxsize)

from news_nlp import triage  # noqa: E402
from news_nlp.eval.judges import JudgeAgent, coerce, repair_prompt  # noqa: E402

DEFAULT_DIR = ROOT / "data" / "summary_judge"
GROUPS = ("single", "multi")
BUCKETS = (0, 1, 2)
BUCKET_BY = ("chunks", "chars")
DEFAULT_SEED = 20261113
MAX_CLAIMS = 12  # the prompt says "At most 12 claims" (a test ties the two): a summary that reaches it may have more
SAMPLE_COLUMNS = (
    "id",
    "group",
    "item_kind",
    "item_id",
    "bucket",
    "num_chunks",
    "pop_n",
    "sample_n",
    "weight",
    "source_text",
    "summary_text",
    "source_chars",
    "summary_chars",
    "language",
    "compression",
    "source_tickers",
    "summary_tickers",
    "summary_revision",
)

# --- the judge -----------------------------------------------------------------------------------------------------------------

PROMPT = """\
You are a careful fact checker acting as an evaluation judge for an automated summariser. You are given a SOURCE TEXT and a SUMMARY of it.

## What to do

1. Split the SUMMARY into its atomic claims: one fact each (who did what, a figure, a date, a reason). Ignore filler such as "the company said" with no content.
   At most 12 claims; if there are more, take the 12 that carry the most information.
2. Mark each claim by ONLY the source text. What is true in the world does not matter:
   - `supported`: the source states it, or it follows directly from what the source states.
   - `unsupported`: the source does not say it (an invented figure, name, cause or date; a detail from elsewhere), even if it may be true.
   - `contradicted`: the source says something different (a figure, a direction, who did what).
   For `unsupported` and `contradicted`, quote the evidence or say what the source says in `evidence`; for `supported`, a short quote.
3. `main_event`: in one sentence, the main event or fact of the SOURCE TEXT (what happened, to whom). `main_event_covered`: true if the SUMMARY states it, false if it
   misses it or states something else.
4. `rationale`: one or two sentences.

## Output

Reply with ONLY a raw JSON object, no prose, no markdown fences, with exactly these keys:

{"claims": [{"text": string, "verdict": "supported" | "unsupported" | "contradicted", "evidence": string}, ...],
 "main_event": string, "main_event_covered": true | false, "rationale": string}
"""


class Claim(BaseModel):
    text: str = ""
    verdict: Literal["supported", "unsupported", "contradicted"]
    evidence: str = ""


class ClaimsVerdict(BaseModel):
    """The judge's read of one summary, claim by claim."""

    # no defaults on the two keys the rates are made of: a reply without them is not a verdict (a default of "covered" raised the rate)
    claims: list[Claim]
    main_event: str = ""
    main_event_covered: bool
    rationale: str = ""
    parse_failed: bool = False


def user_prompt(source: str, summary: str) -> str:
    return f"SOURCE TEXT\n{source}\n\nSUMMARY\n{summary}\n\nReturn your JSON verdict."


PROMPT_SHA = hashlib.sha256((PROMPT + user_prompt("{source}", "{summary}")).encode()).hexdigest()[
    :16
]
MAX_CONSECUTIVE_ERRORS = 5
DEFAULT_MAX_CALLS = 1000  # requests: the planned sample is 300 rows, at most two requests each (the repair prompt); a retry inside the provider is not seen


class CallCap(RuntimeError):
    """The cap of requests of the run was reached before this request was made."""


class CallBudget:
    """The requests of a run, counted where they are made (a row can take two: the prompt and the repair prompt), with the tokens the provider reported."""

    def __init__(self, max_calls: int | None) -> None:
        self.max_calls = max_calls
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                return False
            self.calls += 1
            return True

    def add_tokens(self, tin: int, tout: int) -> None:
        with self._lock:
            self.tokens_in += tin
            self.tokens_out += tout


def usage_of(result: object) -> tuple[int, int]:
    """The tokens ``(in, out)`` a reply reports, ``(0, 0)`` when the agent does not say (a stub). The strands result keeps ``metrics.accumulated_usage``
    (``inputTokens`` and ``outputTokens``), accumulated over the agent's calls."""
    usage = getattr(getattr(result, "metrics", None), "accumulated_usage", None)
    if not isinstance(usage, dict):
        return 0, 0
    return int(usage.get("inputTokens", 0) or 0), int(usage.get("outputTokens", 0) or 0)


class CountedAgent:
    """Wraps a judge agent: every request is taken from the budget first (``CallCap`` when it is spent) and its tokens are recorded, for this row and for the run."""

    def __init__(self, agent: JudgeAgent, budget: CallBudget) -> None:
        self.agent = agent
        self.budget = budget
        self.tokens_in = 0
        self.tokens_out = 0

    def __call__(self, prompt: str) -> object:
        if not self.budget.take():
            raise CallCap("the cap of requests was reached")
        result = self.agent(prompt)
        total_in, total_out = usage_of(
            result
        )  # accumulated over this agent's calls: take the difference
        delta_in, delta_out = max(0, total_in - self.tokens_in), max(0, total_out - self.tokens_out)
        self.tokens_in, self.tokens_out = (
            max(total_in, self.tokens_in),
            max(total_out, self.tokens_out),
        )
        self.budget.add_tokens(delta_in, delta_out)
        return result


class ProviderError(RuntimeError):
    """The judge's provider failed (a wrong key, a rate limit, a dropped connection): not a verdict, and never written as one."""


class JudgeOutage(RuntimeError):
    """``MAX_CONSECUTIVE_ERRORS`` provider errors in a row: the cause is not one row."""


def judge_one(agent: JudgeAgent, source: str, summary: str) -> ClaimsVerdict:
    """The judge's verdict on one summary: one repair prompt for a reply that cannot be read, then ``parse_failed``. A provider error raises
    ``ProviderError``."""
    for prompt in (user_prompt(source, summary), repair_prompt()):
        try:
            reply = str(agent(prompt))
        except CallCap:
            raise
        except Exception as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc
        parsed = coerce(reply, ClaimsVerdict)
        if parsed is not None:
            return parsed
    return ClaimsVerdict(
        claims=[],
        main_event_covered=False,
        rationale="judge reply could not be parsed",
        parse_failed=True,
    )


def read_verdicts(path: Path) -> dict[str, dict[str, Any]]:
    """The verdicts of a file by item id; a later line for the same id replaces an earlier one."""
    out: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                out[record["id"]] = record
    return out


def judge_identity(verdicts: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """The one judge (model, prompt hash) a file of verdicts was made by; a file with none recorded or with two is refused."""
    seen = {(v.get("judge_model"), v.get("prompt_sha")) for v in verdicts.values()}
    if any(m is None or p is None for m, p in seen):
        raise ValueError(
            "the verdicts carry no provenance (judge_model, prompt_sha): judge them again"
        )
    if len(seen) > 1:
        raise ValueError(
            f"more than one judge in the verdicts, judge the rest again or remove the file: {sorted(seen)}"
        )
    return next(iter(seen)) if seen else ("no verdicts", "")


def judge_of(verdicts: dict[str, dict[str, Any]]) -> str:
    """The model of the one judge of a file of verdicts (see ``judge_identity``)."""
    return judge_identity(verdicts)[0]


def run_judge(
    rows: Sequence[dict[str, Any]],
    agent_factory: Callable[[], JudgeAgent],
    out: Path,
    workers: int = 4,
    limit: int | None = None,
    judge_model: str = "",
    sample_id: str | None = None,
    max_calls: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Judge the rows without a good verdict **of this judge** in ``out`` (the model and the prompt are compared), appending each as it arrives
    (a run can stop and go on). A reply that arrived and could not be read is stored with ``parse_failed`` and judged again by the next run; a
    provider error is not stored, is counted, and ``MAX_CONSECUTIVE_ERRORS`` in a row stop the run with ``JudgeOutage``.
    ``max_calls`` is a hard cap on the **requests** made in this run (a row takes one, or two when its first reply cannot be read; an error counts): the rows left
    over are counted in ``not_called`` and a rerun goes on. ``calls``, ``tokens_in`` and ``tokens_out`` are the run's own figures (the tokens only when the agent reports them)."""
    done = {
        i
        for i, v in read_verdicts(out).items()
        if not v.get("parse_failed")
        and v.get("judge_model") == judge_model
        and v.get("prompt_sha") == PROMPT_SHA
        and v.get("sample_id") == sample_id
    }
    todo = [r for r in rows if r["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    lock = threading.Lock()
    stop = threading.Event()
    counts = {
        "judged": 0,
        "parse_failed": 0,
        "provider_errors": 0,
        "longest_verdict_chars": 0,
        "calls": 0,
        "not_called": 0,
        "tokens_in": 0,
        "tokens_out": 0,
    }
    budget = CallBudget(max_calls)
    streak = {"n": 0, "last": ""}
    out.parent.mkdir(parents=True, exist_ok=True)

    def one(row: dict[str, Any]) -> None:
        if stop.is_set():
            return
        agent = CountedAgent(agent_factory(), budget)
        try:
            verdict = judge_one(agent, row["source_text"], row["summary_text"])
        except CallCap:
            with lock:
                counts["not_called"] += 1
            return
        except ProviderError as exc:
            with lock:
                counts["provider_errors"] += 1
                streak["n"] += 1
                streak["last"] = str(exc)
                if streak["n"] >= MAX_CONSECUTIVE_ERRORS:
                    stop.set()
            return
        content = (
            "" if verdict.parse_failed else json.dumps(verdict.model_dump(), ensure_ascii=False)
        )
        record = {
            "id": row["id"],
            **verdict.model_dump(),
            "verdict_chars": len(content),
            "judge_model": judge_model,
            "prompt_sha": PROMPT_SHA,
            "sample_id": sample_id,
            "summary_revision": row.get("summary_revision"),
            "tokens_in": agent.tokens_in,
            "tokens_out": agent.tokens_out,
            **(extra or {}),
        }
        with lock:
            streak["n"] = 0
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["parse_failed" if verdict.parse_failed else "judged"] += 1
            counts["longest_verdict_chars"] = max(counts["longest_verdict_chars"], len(content))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    counts["calls"], counts["tokens_in"], counts["tokens_out"] = (
        budget.calls,
        budget.tokens_in,
        budget.tokens_out,
    )
    if stop.is_set():
        raise JudgeOutage(
            f"{streak['n']} provider errors in a row after {counts['judged'] + counts['parse_failed']} verdicts "
            f"(the last: {streak['last']}); nothing was written for them, a rerun continues"
        )
    return counts


# --- the frame and the draw --------------------------------------------------------------------------------------------------


def bucket_edges(chunks: Sequence[int]) -> tuple[int, int]:
    """The two cut points of the terciles of the chunk counts: a count at or under the first is bucket 0, at or under the second bucket 1."""
    ordered = sorted(chunks)
    if not ordered:
        return (0, 0)
    n = len(ordered)
    return ordered[(n - 1) // 3], ordered[2 * (n - 1) // 3]


def bucket_of(chunks: int, edges: tuple[int, int]) -> int:
    return 0 if chunks <= edges[0] else 1 if chunks <= edges[1] else 2


def allocate(population: dict[int, int], n: int) -> dict[int, int]:
    """``n`` items shared equally across the buckets, a bucket giving no more than it has and the rest going where there is room; if there
    are fewer than ``n`` items, all of them."""
    take = dict.fromkeys(BUCKETS, 0)
    left = min(n, sum(population.values()))
    while left > 0:
        room = [b for b in BUCKETS if take[b] < population.get(b, 0)]
        if (
            not room
        ):  # cannot happen with left capped by the population, and a loop that could spin must not
            break
        for b in room:
            if left == 0:
                break
            take[b] += 1
            left -= 1
    return take


def eligible(
    results: sqlite3.Connection, summary_revision: str, scope_revision: str, cut_revision: str
) -> dict[str, list[tuple[str, str, int, str | None]]]:
    """The summarised items of each group as ``(item_id, summary, chunks, source text or None)``; the items stored as they are (under the size)
    are not in the frame. A single body's source text is looked for in SOURCE afterwards."""
    single = results.execute(
        "SELECT s.item_id, s.summary_text, s.num_chunks FROM news_summary s "
        "JOIN body_scope b ON b.body_hash = s.item_id AND b.classifier_revision = ? AND b.scope = 'single' "
        "WHERE s.item_kind = 'body' AND s.model_revision = ? AND s.text_source = 'summary' ORDER BY s.item_id",
        (scope_revision, summary_revision),
    ).fetchall()
    multi = results.execute(
        "SELECT s.item_id, s.summary_text, s.num_chunks, n.text FROM news_summary s "
        "JOIN sub_news n ON CAST(n.id AS TEXT) = s.item_id AND n.extractor_revision = ? "
        "WHERE s.item_kind = 'sub_news' AND s.model_revision = ? AND s.text_source = 'summary' ORDER BY n.id",
        (cut_revision, summary_revision),
    ).fetchall()
    return {
        "single": [(str(r[0]), str(r[1]), int(r[2]), None) for r in single],
        "multi": [(str(r[0]), str(r[1]), int(r[2]), str(r[3])) for r in multi],
    }


def fetch_sources(source: sqlite3.Connection, hashes: set[str]) -> dict[str, tuple[str, str, str]]:
    """``(title, body, language)`` of the first article that carries each body hash, the one step 2 read; one pass over SOURCE, stopped when all are found."""
    found: dict[str, tuple[str, str, str]] = {}
    for title, text, language in source.execute(
        "SELECT title, body_text, language FROM articles WHERE fetch_status IN ('ok', 'quarantined') "
        "AND body_text IS NOT NULL AND body_text != '' ORDER BY id"
    ):
        key = triage.body_hash(text)
        if key in hashes and key not in found:
            found[key] = (title or "", text, language or "")
            if len(found) == len(hashes):
                break
    return found


def source_lengths(source: sqlite3.Connection, hashes: set[str]) -> dict[str, int]:
    """The characters of the text the summariser read (``Title-...\\nBody-...``) for each body hash, from one pass over SOURCE that keeps no text."""
    from news_summary_stage import input_of_body  # noqa: PLC0415 -- loads torch

    found: dict[str, int] = {}
    for title, text in source.execute(
        "SELECT title, body_text FROM articles WHERE fetch_status IN ('ok', 'quarantined') "
        "AND body_text IS NOT NULL AND body_text != '' ORDER BY id"
    ):
        key = triage.body_hash(text)
        if key in hashes and key not in found:
            found[key] = len(input_of_body(title or "", text))
            if len(found) == len(hashes):
                break
    return found


def sub_news_languages(
    results: sqlite3.Connection, source: sqlite3.Connection, ids: Sequence[int]
) -> dict[int, str]:
    """The language of the article each sub-news was cut from (``''`` when SOURCE does not say): the summariser reads English only."""
    out: dict[int, str] = {}
    for i in range(0, len(ids), 500):
        chunk = list(ids[i : i + 500])
        marks = ",".join("?" * len(chunk))
        sql = f"SELECT id, article_id FROM sub_news WHERE id IN ({marks})"  # noqa: S608 -- placeholders only
        article_of = dict(results.execute(sql, chunk).fetchall())
        wanted = sorted({a for a in article_of.values() if a is not None})
        langs: dict[int, str] = {}
        for j in range(0, len(wanted), 500):
            part = wanted[j : j + 500]
            sql = f"SELECT id, language FROM articles WHERE id IN ({','.join('?' * len(part))})"  # noqa: S608 -- placeholders only
            for a, lang in source.execute(sql, part):
                langs[int(a)] = lang or ""
        for sid, art in article_of.items():
            out[int(sid)] = langs.get(int(art), "") if art is not None else ""
    return out


def draw(  # noqa: C901 -- one pipeline: frame, buckets, draw, sources, rows
    results: sqlite3.Connection,
    source: sqlite3.Connection,
    summary_revision: str,
    scope_revision: str,
    cut_revision: str,
    per_group: int,
    seed: int,
    bucket_by: str = "chars",
) -> list[dict[str, Any]]:
    """The sample: ``per_group`` summarised items of each group, the same number from each length bucket, with the deterministic measures.
    ``bucket_by`` is ``chars`` (the default: terciles of the characters of the source the summariser read, which exists for any summariser, such as an API one,
    T-206) or ``chunks`` (terciles of the number of chunks the chunk-and-reduce summariser used; **on this corpus a discrete count collapses**: 78% of the single
    bodies and 93% of the sub-news have one chunk, so the middle bucket is empty and the draw is one chunk against more)."""
    if bucket_by not in BUCKET_BY:
        raise ValueError(f"bucket_by must be one of {BUCKET_BY}, not {bucket_by!r}")
    import news_company_stage as ncs  # noqa: PLC0415 -- loads torch
    from news_summary_stage import input_of_body  # noqa: PLC0415 -- loads torch

    frame = eligible(results, summary_revision, scope_revision, cut_revision)
    lengths: dict[str, int] = {}
    if (
        bucket_by == "chars"
    ):  # the length of a single body is known only from SOURCE: one pass over all of them, keeping the lengths and not the texts
        lengths = source_lengths(source, {item[0] for item in frame["single"]})
        lacking = len(frame["single"]) - len(lengths)
        if lacking:
            raise ValueError(
                f"the SOURCE copy lacks {lacking} of {len(frame['single'])} bodies of the single summaries: copy RESULTS and SOURCE at the same time"
            )

    def size_of(group: str, item: tuple[str, str, int, str | None]) -> int:
        if bucket_by == "chunks":
            return item[2]
        if group == "multi":
            return len(item[3] or "")
        return lengths[item[0]]

    universe = [
        triage.Company(*(x or "" for x in r))
        for r in source.execute(
            "SELECT DISTINCT ticker, company, gics_sector, gics_sub_industry FROM articles WHERE fetch_status = 'ok'"
        )
    ]
    resolver = triage.CompanyResolver(universe)
    picked: list[tuple[str, int, int, int, tuple[str, str, int, str | None]]] = []
    for group in GROUPS:
        items = frame[group]
        edges = bucket_edges([size_of(group, item) for item in items])
        by_bucket: dict[int, list[tuple[str, str, int, str | None]]] = {b: [] for b in BUCKETS}
        for item in items:
            by_bucket[bucket_of(size_of(group, item), edges)].append(item)
        take = allocate({b: len(v) for b, v in by_bucket.items()}, per_group)
        for b in BUCKETS:
            chosen = random.Random(f"{seed}:{group}:{b}").sample(by_bucket[b], take[b])  # noqa: S311 -- a reproducible draw, not security
            picked.extend((group, b, len(by_bucket[b]), take[b], item) for item in chosen)  # type: ignore[misc]
    wanted = {item[0] for group, _, _, _, item in picked if group == "single"}  # type: ignore[index]
    sources = fetch_sources(source, wanted)
    sub_ids = [int(item[0]) for group, _, _, _, item in picked if group == "multi"]  # type: ignore[index]
    language_of_sub = sub_news_languages(results, source, sub_ids)
    if len(sources) < len(wanted):
        raise ValueError(
            f"the SOURCE copy lacks {len(wanted) - len(sources)} of {len(wanted)} bodies of the sampled single summaries: "
            "copy RESULTS and SOURCE at the same time (docstring), or the first article of those bodies is no longer ok / quarantined"
        )
    rows: list[dict[str, Any]] = []
    for group, bucket, pop_n, sample_n, item in picked:  # type: ignore[misc]
        item_id, summary, chunks, text = item
        if group == "single":
            title, body, language = sources[item_id]
            text = input_of_body(title, body)
        else:
            language = language_of_sub.get(int(item_id), "")
        assert text is not None
        src_tickers = ncs.direct_mentions(resolver, text)
        sum_tickers = ncs.direct_mentions(resolver, summary)
        rows.append(
            {
                "id": f"{group}:{item_id}",
                "group": group,
                "item_kind": "body" if group == "single" else "sub_news",
                "item_id": item_id,
                "bucket": bucket,
                "num_chunks": chunks,
                "pop_n": pop_n,
                "sample_n": sample_n,
                "weight": pop_n / sample_n,
                "source_text": text,
                "summary_text": summary,
                "source_chars": len(text),
                "summary_chars": len(summary),
                "language": language,
                "compression": len(summary) / len(text) if text else 0.0,
                "source_tickers": json.dumps(src_tickers, sort_keys=True),
                "summary_tickers": json.dumps(sum_tickers, sort_keys=True),
                "summary_revision": summary_revision,
            }
        )
    return rows


def _file_identity(path: Path) -> dict[str, Any]:
    """Name, size and modification time of a database copy (not a hash: the files are gigabytes)."""
    if not path.exists():
        return {"name": path.name}
    stat = path.stat()
    return {"name": path.name, "bytes": stat.st_size, "mtime": int(stat.st_mtime)}


def pinned_summariser() -> str:
    """The model of the summary step as pinned (``config/pipeline_models.json``, ``c_summary``): the default author of the summaries of a sample."""
    pin = json.loads((ROOT / "config" / "pipeline_models.json").read_text(encoding="utf-8"))
    return str(pin["c_summary"]["model"])


def is_self_judge(summariser_model: str, judge_model: str) -> bool:
    """Is the judge the model that wrote the summaries? Compared by name without the owner and in lower case (``sshleifer/distilbart-cnn-12-6`` and
    ``distilbart-cnn-12-6`` are the same model; ``deepseek-chat`` writing and judging is the case this exists for). ``summariser_model`` may hold several names
    separated by commas (the model asked for and the one the API named): the judge is a self-judge if it is any of them."""

    def norm(name: str) -> str:
        return name.strip().lower().rsplit("/", 1)[-1]

    names = {norm(n) for n in summariser_model.split(",")} - {""}
    return norm(judge_model) in names


API_REVISION_PREFIX = "summary-llm-"


def summariser_of(results_db: Path, summary_revision: str, given: str | None) -> str:
    """The model(s) that wrote the summaries of a revision. ``--summariser-model`` wins. A revision of the API summariser (``summary-llm-``) names them itself:
    the model asked for and the one the API answered with, from ``news_summary_call`` (``deepseek-chat`` is answered as ``deepseek-flash``), so the guard against a
    self-judge does not depend on the flag being typed; with no call recorded the sample cannot be drawn. Any other revision is the pinned local summariser."""
    if given:
        return given
    if not summary_revision.startswith(API_REVISION_PREFIX):
        return pinned_summariser()
    with closing(sqlite3.connect(f"file:{results_db}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT model_asked, api_model FROM news_summary_call WHERE model_revision = ?",
            (summary_revision,),
        ).fetchall()
    names = sorted({str(n) for row in rows for n in row if n})
    if not names:
        sys.exit(
            f"{summary_revision} is an API revision with no call recorded in news_summary_call: pass --summariser-model with the model that wrote it"
        )
    return ",".join(names)


def make_meta(
    rows: Sequence[dict[str, Any]],
    summary_revision: str,
    scope_revision: str,
    cut_revision: str,
    *,
    per_group: int,
    seed: int,
    results_db: Path,
    source_db: Path,
    bucket_by: str = "chars",
    summariser_model: str | None = None,
) -> dict[str, Any]:
    """What a sample was drawn from: the three revisions (the ids of the sub-news mean nothing without the cut's), the seed, the size, the
    population of each group, the two database copies, and the identity ``sample_id`` (a hash of all that and of the ids drawn, not of the date)."""
    populations = {
        g: int(sum(float(r["weight"]) for r in rows if r["group"] == g) + 0.5) for g in GROUPS
    }
    core = {
        "summary_revision": summary_revision,
        "scope_revision": scope_revision,
        "cut_revision": cut_revision,
        "seed": seed,
        "per_group": per_group,
        "bucket_by": bucket_by,
        "summariser_model": summariser_model or pinned_summariser(),
        "populations": populations,
        "ids": sorted(r["id"] for r in rows),
    }
    sample_id = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12]
    core.pop("ids")
    return {
        "sample_id": sample_id,
        **core,
        "rows": len(rows),
        "results_db": _file_identity(results_db),
        "source_db": _file_identity(source_db),
        "drawn_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def meta_path(sample_path: Path) -> Path:
    return sample_path.with_suffix(".meta.json")


def read_meta(sample_path: Path) -> dict[str, Any] | None:
    path = meta_path(sample_path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def check_sample(verdicts: dict[str, dict[str, Any]], sample_id: str | None) -> None:
    """Refuse verdicts that are not of this sample (or that do not say which sample they are of): the ids of a sample mean nothing in another."""
    if sample_id is None:
        return
    seen = {v.get("sample_id") for v in verdicts.values()}
    if seen - {sample_id}:
        raise ValueError(
            f"the verdicts are of another sample than {sample_id} (they say {sorted(map(str, seen))}): judge this sample, or use its own verdicts file"
        )


def write_sample(
    path: Path, rows: Sequence[dict[str, Any]], meta: dict[str, Any] | None = None
) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists: the sample is drawn once; remove it to draw another")
    path.parent.mkdir(parents=True, exist_ok=True)
    if meta is not None:
        meta_path(path).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_sample(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --- the figures -------------------------------------------------------------------------------------------------------------


def wilson(hits: int, n: int) -> tuple[float, float]:
    return triage.wilson(hits, n)


def item_measures(verdict: dict[str, Any]) -> dict[str, Any]:
    """The per-summary measures of a verdict; ``clean`` is None when there is no claim to check (nothing was said, so nothing is faithful)."""
    claims = verdict.get("claims", [])
    unsupported = sum(1 for c in claims if c["verdict"] == "unsupported")
    contradicted = sum(1 for c in claims if c["verdict"] == "contradicted")
    return {
        "claims": len(claims),
        "unsupported": unsupported,
        "contradicted": contradicted,
        "at_cap": len(claims) >= MAX_CLAIMS,
        "clean": None if not claims else unsupported + contradicted == 0,
        "any_contradicted": contradicted > 0,
        "covered": bool(verdict.get("main_event_covered", False)),
    }


def entity_measures(source: dict[str, int], summary: dict[str, int]) -> dict[str, Any]:
    """Whether the summary keeps the companies the source names: the one named most often, and the share of all of them."""
    if not source:
        return {"has_company": False, "main_kept": None, "recall": None}
    main = sorted(source, key=lambda t: (-source[t], t))[0]
    return {
        "has_company": True,
        "main_kept": main in summary,
        "recall": sum(1 for t in source if t in summary) / len(source),
    }


def weighted_rate(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    return (
        sum(v * w for v, w in zip(values, weights, strict=True)) / total if total else float("nan")
    )


def _bootstrap(
    values: Sequence[float], weights: Sequence[float], buckets: Sequence[str], seed: int, draws: int
) -> tuple[float, float]:
    """The 95% interval of the weighted rate, resampling within each bucket (the draw was stratified)."""
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    by_bucket: dict[str, list[int]] = {}
    for i, b in enumerate(buckets):
        by_bucket.setdefault(b, []).append(i)
    rates = []
    for _ in range(draws):
        idx = [rng.choice(members) for members in by_bucket.values() for _ in members]
        rates.append(weighted_rate([values[i] for i in idx], [weights[i] for i in idx]))
    rates.sort()
    return rates[int(0.025 * len(rates))], rates[min(int(0.975 * len(rates)), len(rates) - 1)]


def _rate(
    flags: Sequence[bool], weights: Sequence[float], buckets: Sequence[str], seed: int, draws: int
) -> dict[str, Any]:
    values = [1.0 if f else 0.0 for f in flags]
    if not values:
        return {"estimate": None, "low": None, "high": None, "n": 0}
    low, high = _bootstrap(values, weights, buckets, seed, draws)
    return {"estimate": weighted_rate(values, weights), "low": low, "high": high, "n": len(values)}


def _plain(flags: Sequence[bool]) -> dict[str, Any]:
    n = len(flags)
    if n == 0:
        return {"estimate": None, "low": None, "high": None, "n": 0}
    low, high = wilson(sum(flags), n)
    return {"estimate": sum(flags) / n, "low": low, "high": high, "n": n}


def _ratio_rate(
    nums: Sequence[float],
    dens: Sequence[float],
    weights: Sequence[float],
    buckets: Sequence[str],
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """A pooled share (claims that are unsupported over all the claims, each summary weighted by its bucket's weight) with a bootstrap interval
    that resamples **summaries** within buckets: claims are clustered in their summary, so resampling claims would be too narrow."""
    total = sum(w * d for w, d in zip(weights, dens, strict=True))
    if not total:
        return {"estimate": None, "low": None, "high": None, "n": 0}
    estimate = sum(w * x for w, x in zip(weights, nums, strict=True)) / total
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    by_bucket: dict[str, list[int]] = {}
    for i, b in enumerate(buckets):
        by_bucket.setdefault(b, []).append(i)
    shares = []
    for _ in range(draws):
        idx = [rng.choice(members) for members in by_bucket.values() for _ in members]
        den = sum(weights[i] * dens[i] for i in idx)
        if den:
            shares.append(sum(weights[i] * nums[i] for i in idx) / den)
    shares.sort()
    low = shares[int(0.025 * len(shares))] if shares else None
    high = shares[min(int(0.975 * len(shares)), len(shares) - 1)] if shares else None
    return {"estimate": estimate, "low": low, "high": high, "n": int(sum(dens))}


def summarise(
    rows: Sequence[dict[str, Any]],
    verdicts: dict[str, dict[str, Any]],
    seed: int = DEFAULT_SEED,
    draws: int = 2000,
) -> dict[str, Any]:
    """The figures by group, with the weighted rate and a bootstrap interval (resampled within buckets), and by bucket with a Wilson interval."""
    report: dict[str, Any] = {}
    for group in GROUPS:
        mine = [r for r in rows if r["group"] == group]
        if not mine:
            continue
        judged = [
            (r, item_measures(verdicts[r["id"]]))
            for r in mine
            if r["id"] in verdicts and not verdicts[r["id"]].get("parse_failed")
        ]
        checkable = [(r, m) for r, m in judged if m["clean"] is not None]
        cw = [float(r["weight"]) for r, _ in checkable]
        cb = [str(r["bucket"]) for r, _ in checkable]
        entities = [
            (r, entity_measures(json.loads(r["source_tickers"]), json.loads(r["summary_tickers"])))
            for r in mine
        ]
        with_company = [(r, e) for r, e in entities if e["has_company"]]
        unreadable = [
            r for r in mine if r["id"] in verdicts and verdicts[r["id"]].get("parse_failed")
        ]
        out: dict[str, Any] = {
            "n": len(mine),
            "judged": len(judged),
            "parse_failed": sum(
                1 for r in mine if r["id"] in verdicts and verdicts[r["id"]].get("parse_failed")
            ),
            "not_judged": sum(1 for r in mine if r["id"] not in verdicts),
            "no_claims": len(judged) - len(checkable),
            "at_cap": sum(1 for _, m in judged if m["at_cap"]),
            "clean": _rate([m["clean"] for _, m in checkable], cw, cb, seed, draws),
            # the three summary rates share one denominator: the summaries with a claim to check (one without is counted apart)
            "any_contradicted": _rate(
                [m["any_contradicted"] for _, m in checkable], cw, cb, seed, draws
            ),
            "main_event_covered": _rate([m["covered"] for _, m in checkable], cw, cb, seed, draws),
            "unsupported_claim_share": _ratio_rate(
                [m["unsupported"] for _, m in judged],
                [m["claims"] for _, m in judged],
                [float(r["weight"]) for r, _ in judged],
                [str(r["bucket"]) for r, _ in judged],
                seed,
                draws,
            ),
            "contradicted_claim_share": _ratio_rate(
                [m["contradicted"] for _, m in judged],
                [m["claims"] for _, m in judged],
                [float(r["weight"]) for r, _ in judged],
                [str(r["bucket"]) for r, _ in judged],
                seed,
                draws,
            ),
            # every unreadable reply counted as an unfaithful summary: the floor of the clean rate if the failures hide bad summaries
            "clean_worst_case": _rate(
                [m["clean"] for _, m in checkable] + [False] * len(unreadable),
                cw + [float(r["weight"]) for r in unreadable],
                cb + [str(r["bucket"]) for r in unreadable],
                seed,
                draws,
            ),
            "main_company_kept": _rate(
                [bool(e["main_kept"]) for _, e in with_company],
                [float(r["weight"]) for r, _ in with_company],
                [str(r["bucket"]) for r, _ in with_company],
                seed,
                draws,
            ),
            "company_recall": (
                weighted_rate(
                    [e["recall"] for _, e in with_company],
                    [float(r["weight"]) for r, _ in with_company],
                )
                if with_company
                else None
            ),
            "items_with_company": len(with_company),
            "compression_median": statistics.median(float(r["compression"]) for r in mine),
            "longer_than_source": sum(1 for r in mine if float(r["compression"]) > 1) / len(mine),
            "by_bucket": {},
            "by_language": {},
        }
        for language in sorted({str(r.get("language", "")) for r in mine}):
            in_language = [(r, m) for r, m in checkable if str(r.get("language", "")) == language]
            out["by_language"][language or "unknown"] = {
                "n": sum(1 for r in mine if str(r.get("language", "")) == language),
                "clean": _plain([bool(m["clean"]) for _, m in in_language]),
                "main_event_covered": _plain([m["covered"] for _, m in in_language]),
            }
        for bucket in sorted({str(r["bucket"]) for r in mine}):
            in_bucket = [(r, m) for r, m in checkable if str(r["bucket"]) == bucket]
            out["by_bucket"][bucket] = {
                "n": sum(1 for r in mine if str(r["bucket"]) == bucket),
                "population": int(
                    float(next(r for r in mine if str(r["bucket"]) == bucket)["pop_n"])
                ),
                "median_chars": statistics.median(
                    int(r["source_chars"]) for r in mine if str(r["bucket"]) == bucket
                ),
                "unreadable": sum(1 for r in unreadable if str(r["bucket"]) == bucket),
                "clean": _plain([bool(m["clean"]) for _, m in in_bucket]),
                "any_contradicted": _plain([m["any_contradicted"] for _, m in in_bucket]),
                "main_event_covered": _plain([m["covered"] for _, m in in_bucket]),
            }
        report[group] = out
    return report


def _pct(rate: dict[str, Any] | None) -> str:
    if not rate or rate.get("estimate") is None:
        return "n/a"
    return f"{100 * rate['estimate']:.1f}% [{100 * rate['low']:.1f}, {100 * rate['high']:.1f}]"


def render(report: dict[str, Any], judge: str) -> str:
    lines = [
        f"Summary judge: `{judge}`. Rates are per summary, weighted by the bucket's share of its group (interval: bootstrap within buckets, 95%).",
        "These are the judge's figures, **not checked against the owner's marks** (`tag-sheet`, `agree`).",
        "The three buckets are terciles of the chunk count or of the source characters (`bucket_by` in `sample.meta.json`); a discrete count can collapse them (a bucket with no sampled item is not listed). The summariser reads English only: the language table is not weighted.",
        "",
    ]
    for group, g in report.items():
        lines += [
            f"### {group}: {g['n']} summaries, {g['judged']} judged, {g['parse_failed']} unreadable, {g['not_judged']} not judged, {g['no_claims']} with no claim",
            "",
            f"- no unsupported or contradicted claim: **{_pct(g['clean'])}**",
            f"- at least one contradicted claim: {_pct(g['any_contradicted'])}",
            f"- the main event is in the summary: {_pct(g['main_event_covered'])}",
            f"- claims that are unsupported: {_pct(g['unsupported_claim_share'])}; contradicted: {_pct(g['contradicted_claim_share'])} (interval: resampling summaries)",
            f"- worst case, every unreadable reply counted as an unfaithful summary: clean {_pct(g['clean_worst_case'])}",
            f"- summaries at the {MAX_CLAIMS}-claim cap: {g['at_cap']}"
            + (
                " (partly checked: the judge lists no more claims than the cap)"
                if g["at_cap"]
                else ""
            ),
            f"- the company the source names most is kept (of {g['items_with_company']} that name one): {_pct(g['main_company_kept'])}; share of the companies kept: {_share(g['company_recall'])}",
            f"- compression (summary over source), median {g['compression_median']:.2f}; longer than its source: {100 * g['longer_than_source']:.1f}%",
            "",
            "| bucket | in the group | sampled | unreadable | median source chars | clean | contradicted | main event |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for bucket, v in g["by_bucket"].items():
            lines.append(
                f"| {bucket} | {v['population']} | {v['n']} | {v['unreadable']} | {v['median_chars']:.0f} | {_pct(v['clean'])} | {_pct(v['any_contradicted'])} | {_pct(v['main_event_covered'])} |"
            )
        lines += [
            "",
            "| language | sampled | clean | main event |",
            "|---|---|---|---|",
        ]
        for language, v in g["by_language"].items():
            lines.append(
                f"| {language} | {v['n']} | {_pct(v['clean'])} | {_pct(v['main_event_covered'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


# --- the owner's check -------------------------------------------------------------------------------------------------------

TAG_RULE = """\
Regla de lectura (las dos columnas se responden con `si` o `no`):
- afirmaciones_sin_respaldo: ¿hay alguna afirmación del resumen que el texto fuente NO respalda o que contradice? (una cifra, un nombre, una causa o una fecha
  inventados; lo que sea cierto en el mundo no cuenta, solo lo que dice el texto fuente). `no` = el resumen es fiel.
- evento_principal: ¿el resumen dice el hecho principal del texto fuente (qué pasó y a quién)? `si` = lo dice.
Un resumen puede ser fiel y no decir el hecho principal. Escribe en `notas` lo que quieras recordar."""
TAG_COLUMNS = (
    "id",
    "grupo",
    "texto_fuente",
    "resumen",
    "afirmaciones_sin_respaldo",
    "evento_principal",
    "notas",
)


def write_tag_sheet(
    path: Path,
    rows: Sequence[dict[str, Any]],
    per_group: int,
    seed: int,
    verdicts: dict[str, dict[str, Any]] | None = None,
    sample_id: str | None = None,
) -> list[str]:
    """The owner's slice: ``per_group`` summaries of each group, shuffled, with the source, the summary and blank columns, and **no verdict**.

    With the verdicts, at most half of a group's slice are the summaries the judge flagged (an unsupported or contradicted claim) and the rest a
    random control of clean ones; a sidecar ``<name>.strata.json`` records which is which and the weighted size of each stratum in the sample, for
    ``agree``. Without them the slice is random, which cannot measure the judge on a rare event."""
    if path.exists():
        raise FileExistsError(
            f"{path} exists: the sheet is written once; remove it to write another"
        )
    judge = (
        judge_identity(verdicts) if verdicts is not None else None
    )  # refuses verdicts of two judges, before anything is written
    if verdicts is not None:
        check_sample(verdicts, sample_id)  # and verdicts of another sample
    chosen: list[dict[str, Any]] = []
    stratum: dict[str, str] = {}
    population: dict[str, dict[str, float]] = {}
    for group in GROUPS:
        mine = sorted((r for r in rows if r["group"] == group), key=lambda r: r["id"])
        rng = random.Random(f"{seed}:tag:{group}")  # noqa: S311 -- a reproducible draw, not security
        if verdicts is None:
            picked = rng.sample(mine, min(per_group, len(mine)))
            stratum.update(dict.fromkeys((r["id"] for r in picked), "random"))
        else:
            measured = [
                (r, item_measures(verdicts[r["id"]]))
                for r in mine
                if r["id"] in verdicts and not verdicts[r["id"]].get("parse_failed")
            ]
            flagged = [r for r, m in measured if m["clean"] is False]
            clean = [r for r, m in measured if m["clean"] is True]
            population[group] = {
                "flagged": sum(float(r["weight"]) for r in flagged),
                "clean": sum(float(r["weight"]) for r in clean),
            }
            take_flagged = rng.sample(flagged, min(per_group // 2, len(flagged)))
            take_clean = rng.sample(clean, min(per_group - len(take_flagged), len(clean)))
            picked = take_flagged + take_clean
            stratum.update({r["id"]: "flagged" for r in take_flagged})
            stratum.update({r["id"]: "control" for r in take_clean})
        chosen += picked
    random.Random(f"{seed}:order").shuffle(chosen)  # noqa: S311 -- a reproducible order, not security
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TAG_COLUMNS)
        writer.writeheader()
        for r in chosen:
            writer.writerow(
                {
                    "id": r["id"],
                    "grupo": r["group"],
                    "texto_fuente": r["source_text"],
                    "resumen": r["summary_text"],
                    "afirmaciones_sin_respaldo": "",
                    "evento_principal": "",
                    "notas": "",
                }
            )
    path.with_suffix(".strata.json").write_text(
        json.dumps(
            {
                "judge": {"model": judge[0], "prompt_sha": judge[1]} if judge else None,
                "sample_id": sample_id,
                "stratum": stratum,
                "population": population,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return [r["id"] for r in chosen]


def _weighted_agreement(
    pairs: Sequence[tuple[bool, bool]], weights: Sequence[float] | None = None
) -> dict[str, Any]:
    """Raw agreement and Cohen's kappa of (judge, owner) pairs, each pair counted by its weight (1 when there are none)."""
    w = list(weights) if weights is not None else [1.0] * len(pairs)
    total = sum(w)
    if not pairs or not total:
        return {"raw": None, "kappa": None}
    po = sum(x for x, (a, b) in zip(w, pairs, strict=True) if a == b) / total
    pa = sum(x for x, (a, _) in zip(w, pairs, strict=True) if a) / total
    pb = sum(x for x, (_, b) in zip(w, pairs, strict=True) if b) / total
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {"raw": po, "kappa": None if pe == 1 else (po - pe) / (1 - pe)}


def _check_marks(tags: Sequence[dict[str, str]], yes: dict[str, bool]) -> None:
    """Stop at a blank or unreadable mark: an unanswered question is not an answer."""
    for t in tags:
        for column in ("afirmaciones_sin_respaldo", "evento_principal"):
            if (t.get(column) or "").strip().lower() not in yes:
                raise ValueError(
                    f"blank or unreadable mark in {column} for {t['id']}: answer `si` or `no`"
                )


def _stratum_rate(marks: Sequence[bool]) -> dict[str, Any]:
    n = len(marks)
    low, high = wilson(sum(marks), n) if n else (None, None)
    return {
        "n": n,
        "owner_unfaithful": sum(marks),
        "rate": sum(marks) / n if n else None,
        "low": low,
        "high": high,
    }


def _sensitivity(size: dict[str, float], flagged_rate: float, clean_rate: float) -> float | None:
    caught = size["flagged"] * flagged_rate
    missed = size["clean"] * clean_rate
    return caught / (caught + missed) if caught + missed else None


def _strata_figures(
    size: dict[str, float] | None, flagged_marks: Sequence[bool], clean_marks: Sequence[bool]
) -> dict[str, Any]:
    """The owner's rate of unfaithful summaries in the judge's flagged stratum and in its clean control, the judge's precision (the first) and its
    sensitivity (flagged share of all the unfaithful, the two strata weighed by their size in the sample). The sensitivity's interval is the
    extremes over the Wilson intervals of the two rates (the flagged rate at its low end with the clean rate at its high end, and the reverse):
    wide on purpose, because a control of twenty clean items stands for the whole clean population."""
    flagged, clean = _stratum_rate(flagged_marks), _stratum_rate(clean_marks)
    figures: dict[str, Any] = {"flagged": flagged, "clean": clean}
    if flagged["rate"] is not None:
        figures["precision"] = flagged["rate"]
        figures["precision_interval"] = [flagged["low"], flagged["high"]]
    if size and flagged["rate"] is not None and clean["rate"] is not None:
        figures["sensitivity"] = _sensitivity(size, flagged["rate"], clean["rate"])
        low = _sensitivity(size, flagged["low"], clean["high"])
        high = _sensitivity(size, flagged["high"], clean["low"])
        figures["sensitivity_interval"] = (
            [low, high] if low is not None and high is not None else None
        )
    return figures


STRATIFIED_NOTE = (
    "the slice is half flagged by construction: quote the precision, the sensitivity and the weighed figures with their intervals, "
    "not the figures of the slice (under `slice`)"
)


def _interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values.sort()
    return [
        values[int(0.025 * len(values))],
        values[min(int(0.975 * len(values)), len(values) - 1)],
    ]


def _agreement_intervals(
    pairs: Sequence[tuple[bool, bool]],
    kinds: Sequence[str | None],
    weights: Sequence[float],
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """The 95% interval of the weighed agreement and kappa: the marked items are resampled within their stratum (a stratum keeps its size, so
    it keeps its weight). One control mark can weigh sixty times a flagged one, so the figures swing on it and the interval says how much."""
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    members: dict[str | None, list[int]] = {}
    for i, k in enumerate(kinds):
        members.setdefault(k, []).append(i)
    raws: list[float] = []
    kappas: list[float] = []
    for _ in range(draws):
        idx = [rng.choice(group) for group in members.values() for _ in group]
        got = _weighted_agreement([pairs[i] for i in idx], [weights[i] for i in idx])
        if got["raw"] is not None:
            raws.append(got["raw"])
        if got["kappa"] is not None:
            kappas.append(got["kappa"])
    return {"raw_interval": _interval(raws), "kappa_interval": _interval(kappas)}


def _block(
    pairs: Sequence[tuple[bool, bool]],
    weights: Sequence[float] | None,
    kinds: Sequence[str | None],
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """One question's figures. On a stratified slice (half of it flagged by construction) ``raw`` and ``kappa`` are weighed back to the sample and
    have an interval; the figures of the slice itself are under ``slice`` and a note says not to quote them. On a random slice there is one pair."""
    n = len(pairs)
    block: dict[str, Any] = {"n": n, "agree": sum(1 for a, b in pairs if a == b)}
    if weights is None:
        block.update(_weighted_agreement(pairs))
        return block
    block.update(_weighted_agreement(pairs, weights))
    block.update(_agreement_intervals(pairs, kinds, weights, seed, draws))
    block["slice"] = _weighted_agreement(pairs)
    block["note"] = STRATIFIED_NOTE
    return block


def _stratum_weights(
    kinds: Sequence[str | None], size: dict[str, float] | None
) -> list[float] | None:
    """The weight of each marked item: its stratum's size in the sample over the number of its stratum marked (flagged and control)."""
    if not size:
        return None
    count = Counter(kinds)
    unit = {
        "flagged": size["flagged"] / count["flagged"] if count["flagged"] else 0.0,
        "control": size["clean"] / count["control"] if count["control"] else 0.0,
    }
    return [unit.get(k or "", 1.0) for k in kinds]


def _collect(
    read: Sequence[dict[str, str]],
    verdicts: dict[str, dict[str, Any]],
    strata: dict[str, Any] | None,
    yes: dict[str, bool],
) -> tuple[
    list[tuple[bool, bool]],
    list[tuple[bool, bool]],
    list[str | None],
    list[str | None],
    list[bool],
    list[bool],
]:
    """The (judge, owner) pairs of the readable verdicts for each question, the stratum of each pair, and the owner's marks in each stratum."""
    faithful: list[tuple[bool, bool]] = []
    event: list[tuple[bool, bool]] = []
    kinds_f: list[str | None] = []
    kinds_e: list[str | None] = []
    flagged_marks: list[bool] = []
    clean_marks: list[bool] = []
    for t in read:
        m = item_measures(verdicts[t["id"]])
        owner_unfaithful = yes[t["afirmaciones_sin_respaldo"].strip().lower()]
        kind = (strata or {}).get("stratum", {}).get(t["id"])
        if m["clean"] is not None:
            faithful.append((bool(m["clean"]), not owner_unfaithful))
            kinds_f.append(kind)
            if kind == "flagged":
                flagged_marks.append(owner_unfaithful)
            elif kind == "control":
                clean_marks.append(owner_unfaithful)
        if m["clean"] is not None or strata is None:
            event.append((bool(m["covered"]), yes[t["evento_principal"].strip().lower()]))
            kinds_e.append(kind)
    return faithful, event, kinds_f, kinds_e, flagged_marks, clean_marks


def agreement(
    tags: Sequence[dict[str, str]],
    verdicts: dict[str, dict[str, Any]],
    strata: dict[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    draws: int = 1000,
) -> dict[str, Any]:
    """The judge against the owner, on the owner's marks: for 'faithful' (no unsupported or contradicted claim) and 'main event'. A blank mark stops
    the check; a summary the judge could not read or found no claim in is left out (``outside_population`` counts them: the sensitivity is
    conditional on a readable verdict with a claim).

    With the strata of the slice (``write_tag_sheet`` with the verdicts) the slice is half flagged by construction, so the raw agreement and
    kappa are weighed back to the sample (``raw``, ``kappa``, each with an interval resampled within strata) and the figures of the slice itself
    are under ``slice``; the
    'faithful' question also gives the owner's rate of unfaithful summaries among the flagged ones (``precision``, with its interval) and
    among the clean control, and the judge's ``sensitivity`` with its interval. Without them the slice is random and says so."""
    yes = {"si": True, "sí": True, "no": False}
    out: dict[str, Any] = {}
    _check_marks(tags, yes)
    chosen_by = (strata or {}).get("judge")
    if chosen_by and (chosen_by["model"], chosen_by["prompt_sha"]) != judge_identity(verdicts):
        raise ValueError(
            "the verdicts were made by another judge than the one that chose the slice "
            f"({chosen_by['model']}, {chosen_by['prompt_sha']}): the precision and the sensitivity would be of nobody"
        )
    if strata:
        check_sample(verdicts, strata.get("sample_id"))
    for group in GROUPS:
        mine = [t for t in tags if t["grupo"] == group and t["id"] in verdicts]
        read = [t for t in mine if not verdicts[t["id"]].get("parse_failed")]
        if not read:
            continue
        faithful, event, kinds_f, kinds_e, flagged_marks, clean_marks = _collect(
            read, verdicts, strata, yes
        )
        size = (strata or {}).get("population", {}).get(group)
        out[group] = {
            "faithful": _block(faithful, _stratum_weights(kinds_f, size), kinds_f, seed, draws),
            "main_event": _block(event, _stratum_weights(kinds_e, size), kinds_e, seed, draws),
        }
        out[group]["faithful"]["outside_population"] = len(mine) - len(faithful)
        if strata is None:
            out[group]["faithful"]["note"] = (
                "random slice: it cannot measure the judge on a rare event (see tag-sheet --verdicts)"
            )
            continue
        out[group]["faithful"].update(_strata_figures(size, flagged_marks, clean_marks))
    return out


# --- the command -------------------------------------------------------------------------------------------------------------


def _ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("draw")
    p.add_argument("--results-db", required=True)
    p.add_argument("--source-db", required=True)
    p.add_argument("--summary-revision", required=True)
    p.add_argument("--scope-revision", required=True)
    p.add_argument("--cut-revision", required=True)
    p.add_argument("--per-group", type=int, default=150)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--bucket-by",
        choices=BUCKET_BY,
        default="chars",
        help="terciles of the source characters (default; any summariser) or of the chunk count (the chunk-and-reduce summariser; collapses on this corpus)",
    )
    p.add_argument(
        "--summariser-model",
        default=None,
        help="the model that wrote the summaries (default: for a summary-llm- revision the models recorded in news_summary_call, else c_summary of config/pipeline_models.json); recorded, and the judge must not be it",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_DIR / "sample.csv")
    j = sub.add_parser("judge")
    j.add_argument("--sample", type=Path, default=DEFAULT_DIR / "sample.csv")
    j.add_argument("--out", type=Path, default=DEFAULT_DIR / "verdicts.jsonl")
    j.add_argument("--limit", type=int, default=None)
    j.add_argument(
        "--allow-self-judge",
        action="store_true",
        help="judge summaries written by the judge's own model (recorded in every verdict)",
    )
    j.add_argument(
        "--max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help="a hard cap on the calls of this run, errors and repeats included",
    )
    j.add_argument("--workers", type=int, default=None)
    r = sub.add_parser("report")
    r.add_argument("--sample", type=Path, default=DEFAULT_DIR / "sample.csv")
    r.add_argument("--verdicts", type=Path, default=DEFAULT_DIR / "verdicts.jsonl")
    t = sub.add_parser("tag-sheet")
    t.add_argument("--sample", type=Path, default=DEFAULT_DIR / "sample.csv")
    t.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="the verdicts: the slice then holds the flagged items and a clean control (the sheet shows no verdict)",
    )
    t.add_argument("--per-group", type=int, default=40)
    t.add_argument("--seed", type=int, default=DEFAULT_SEED)
    t.add_argument("--out", type=Path, default=DEFAULT_DIR / "to_tag.csv")
    a = sub.add_parser("agree")
    a.add_argument("--tags", type=Path, default=DEFAULT_DIR / "to_tag.csv")
    a.add_argument("--verdicts", type=Path, default=DEFAULT_DIR / "verdicts.jsonl")
    a.add_argument(
        "--strata",
        type=Path,
        default=None,
        help="the sidecar of tag-sheet (default <tags>.strata.json when it exists)",
    )
    return parser


def _sample_id(sample: Path) -> str | None:
    meta = read_meta(sample)
    return meta["sample_id"] if meta else None


def _identity_line(sample: Path) -> str:
    meta = read_meta(sample)
    if not meta:
        return "sample: no meta file (drawn before the sample recorded what it came from)"
    return (
        f"sample {meta['sample_id']}: summary {meta['summary_revision']}, scope {meta['scope_revision']}, cut {meta['cut_revision']}, "
        f"seed {meta['seed']}, {meta['per_group']} per group of {meta['populations']}, drawn {meta['drawn_at']} from "
        f"{meta['results_db']['name']} and {meta['source_db']['name']}"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "draw":
        rows = draw(
            _ro(args.results_db),
            _ro(args.source_db),
            args.summary_revision,
            args.scope_revision,
            args.cut_revision,
            args.per_group,
            args.seed,
            args.bucket_by,
        )
        meta = make_meta(
            rows,
            args.summary_revision,
            args.scope_revision,
            args.cut_revision,
            per_group=args.per_group,
            seed=args.seed,
            results_db=Path(args.results_db),
            source_db=Path(args.source_db),
            bucket_by=args.bucket_by,
            summariser_model=summariser_of(
                Path(args.results_db), args.summary_revision, args.summariser_model
            ),
        )
        write_sample(args.out, rows, meta)
        print(f"wrote {len(rows)} rows to {args.out}\n{_identity_line(args.out)}")
        for group in GROUPS:
            mine = [x for x in rows if x["group"] == group]
            print(
                f"  {group}: {len(mine)} of {sum(float(x['weight']) for x in mine):.0f} summarised items"
            )
    elif args.cmd == "judge":
        from news_nlp.eval.config import EvalSettings  # noqa: PLC0415 -- needs the eval group
        from news_nlp.eval.model import build_judge_agent, build_model  # noqa: PLC0415

        settings = EvalSettings.load()
        model = build_model(settings)
        rows = read_sample(args.sample)
        meta = read_meta(args.sample) or {}
        summariser = str(meta.get("summariser_model") or "")
        self_judge = is_self_judge(summariser, settings.llm_model)
        if self_judge and not args.allow_self_judge:
            sys.exit(
                f"refused: the judge ({settings.llm_model}) is the model that wrote the summaries ({summariser}): it would grade its own text. "
                "Set LLM_MODEL to another model, or pass --allow-self-judge (recorded in every verdict) if the owner's marks are the primary measure"
            )
        print(f"judging with {settings.llm_model}; {_identity_line(args.sample)}", file=sys.stderr)
        counts = run_judge(
            rows,
            lambda: build_judge_agent(model, PROMPT),
            args.out,
            args.workers or settings.max_workers,
            args.limit,
            judge_model=settings.llm_model,
            sample_id=_sample_id(args.sample),
            max_calls=args.max_calls,
            extra={"summariser_model": summariser, "self_judge_allowed": self_judge},
        )
        print(json.dumps(counts))
    elif args.cmd == "report":
        rows = read_sample(args.sample)
        verdicts = read_verdicts(args.verdicts)
        check_sample(verdicts, _sample_id(args.sample))
        print(_identity_line(args.sample) + "\n")
        print(render(summarise(rows, verdicts), judge_of(verdicts)))
    elif args.cmd == "tag-sheet":
        verdicts = read_verdicts(args.verdicts) if args.verdicts else None
        ids = write_tag_sheet(
            args.out,
            read_sample(args.sample),
            args.per_group,
            args.seed,
            verdicts,
            _sample_id(args.sample),
        )
        print(f"wrote {len(ids)} rows to {args.out}\n\n{TAG_RULE}")
    elif args.cmd == "agree":
        with args.tags.open(encoding="utf-8-sig", newline="") as fh:
            tags = list(csv.DictReader(fh))
        strata_path = args.strata or args.tags.with_suffix(".strata.json")
        strata = (
            json.loads(strata_path.read_text(encoding="utf-8")) if strata_path.exists() else None
        )
        print(json.dumps(agreement(tags, read_verdicts(args.verdicts), strata), indent=2))


if __name__ == "__main__":
    main()
