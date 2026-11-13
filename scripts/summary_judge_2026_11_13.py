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
revision) and ``multi`` (the sub-news of the named cut revision). Each group is cut in three buckets by the number of chunks the summariser
used (terciles of the population: a long source goes through the chunk-and-reduce path, where errors compound), and the draw takes the same
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
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
csv.field_size_limit(sys.maxsize)

from news_nlp import triage  # noqa: E402
from news_nlp.eval.judges import _REPAIR_PROMPT, JudgeAgent, _coerce  # noqa: E402

DEFAULT_DIR = ROOT / "data" / "summary_judge"
GROUPS = ("single", "multi")
BUCKETS = (0, 1, 2)
DEFAULT_SEED = 20261113
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


class ProviderError(RuntimeError):
    """The judge's provider failed (a wrong key, a rate limit, a dropped connection): not a verdict, and never written as one."""


class JudgeOutage(RuntimeError):
    """``MAX_CONSECUTIVE_ERRORS`` provider errors in a row: the cause is not one row."""


def judge_one(agent: JudgeAgent, source: str, summary: str) -> ClaimsVerdict:
    """The judge's verdict on one summary: one repair prompt for a reply that cannot be read, then ``parse_failed``. A provider error raises
    ``ProviderError``."""
    for prompt in (user_prompt(source, summary), _REPAIR_PROMPT):
        try:
            reply = str(agent(prompt))
        except Exception as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc
        parsed = _coerce(reply, ClaimsVerdict)
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
) -> dict[str, int]:
    """Judge the rows without a good verdict **of this judge** in ``out`` (the model and the prompt are compared), appending each as it arrives
    (a run can stop and go on). A reply that arrived and could not be read is stored with ``parse_failed`` and judged again by the next run; a
    provider error is not stored, is counted, and ``MAX_CONSECUTIVE_ERRORS`` in a row stop the run with ``JudgeOutage``."""
    done = {
        i
        for i, v in read_verdicts(out).items()
        if not v.get("parse_failed")
        and v.get("judge_model") == judge_model
        and v.get("prompt_sha") == PROMPT_SHA
    }
    todo = [r for r in rows if r["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    lock = threading.Lock()
    stop = threading.Event()
    counts = {"judged": 0, "parse_failed": 0, "provider_errors": 0, "longest_verdict_chars": 0}
    streak = {"n": 0, "last": ""}
    out.parent.mkdir(parents=True, exist_ok=True)

    def one(row: dict[str, Any]) -> None:
        if stop.is_set():
            return
        try:
            verdict = judge_one(agent_factory(), row["source_text"], row["summary_text"])
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
            "summary_revision": row.get("summary_revision"),
        }
        with lock:
            streak["n"] = 0
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["parse_failed" if verdict.parse_failed else "judged"] += 1
            counts["longest_verdict_chars"] = max(counts["longest_verdict_chars"], len(content))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
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


def fetch_sources(source: sqlite3.Connection, hashes: set[str]) -> dict[str, tuple[str, str]]:
    """``(title, body)`` of the first article that carries each body hash, the one step 2 read; one pass over SOURCE, stopped when all are found."""
    found: dict[str, tuple[str, str]] = {}
    for title, text in source.execute(
        "SELECT title, body_text FROM articles WHERE fetch_status IN ('ok', 'quarantined') "
        "AND body_text IS NOT NULL AND body_text != '' ORDER BY id"
    ):
        key = triage.body_hash(text)
        if key in hashes and key not in found:
            found[key] = (title or "", text)
            if len(found) == len(hashes):
                break
    return found


def draw(
    results: sqlite3.Connection,
    source: sqlite3.Connection,
    summary_revision: str,
    scope_revision: str,
    cut_revision: str,
    per_group: int,
    seed: int,
) -> list[dict[str, Any]]:
    """The sample: ``per_group`` summarised items of each group, the same number from each length bucket, with the deterministic measures."""
    import news_company_stage as ncs  # noqa: PLC0415 -- loads torch
    from news_summary_stage import input_of_body  # noqa: PLC0415 -- loads torch

    frame = eligible(results, summary_revision, scope_revision, cut_revision)
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
        edges = bucket_edges([c for _, _, c, _ in items])
        by_bucket: dict[int, list[tuple[str, str, int, str | None]]] = {b: [] for b in BUCKETS}
        for item in items:
            by_bucket[bucket_of(item[2], edges)].append(item)
        take = allocate({b: len(v) for b, v in by_bucket.items()}, per_group)
        for b in BUCKETS:
            chosen = random.Random(f"{seed}:{group}:{b}").sample(by_bucket[b], take[b])  # noqa: S311 -- a reproducible draw, not security
            picked.extend((group, b, len(by_bucket[b]), take[b], item) for item in chosen)  # type: ignore[misc]
    sources = fetch_sources(
        source,
        {item[0] for group, _, _, _, item in picked if group == "single"},  # type: ignore[index]
    )
    wanted = {item[0] for group, _, _, _, item in picked if group == "single"}  # type: ignore[index]
    if len(sources) < len(wanted):
        raise ValueError(
            f"the SOURCE copy lacks {len(wanted) - len(sources)} of {len(wanted)} bodies of the sampled single summaries: "
            "copy RESULTS and SOURCE at the same time (docstring), or the first article of those bodies is no longer ok / quarantined"
        )
    rows: list[dict[str, Any]] = []
    for group, bucket, pop_n, sample_n, item in picked:  # type: ignore[misc]
        item_id, summary, chunks, text = item
        if group == "single":
            title, body = sources[item_id]
            text = input_of_body(title, body)
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
                "compression": len(summary) / len(text) if text else 0.0,
                "source_tickers": json.dumps(src_tickers, sort_keys=True),
                "summary_tickers": json.dumps(sum_tickers, sort_keys=True),
                "summary_revision": summary_revision,
            }
        )
    return rows


def write_sample(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists: the sample is drawn once; remove it to draw another")
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "The three buckets are terciles of a discrete chunk count and can collapse (a bucket with no sampled item is not listed).",
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
                "stratum": stratum,
                "population": population,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return [r["id"] for r in chosen]


def _kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


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
    if size and flagged["rate"] is not None and clean["rate"] is not None:
        figures["sensitivity"] = _sensitivity(size, flagged["rate"], clean["rate"])
        low = _sensitivity(size, flagged["low"], clean["high"])
        high = _sensitivity(size, flagged["high"], clean["low"])
        figures["sensitivity_interval"] = (
            [low, high] if low is not None and high is not None else None
        )
    return figures


def agreement(
    tags: Sequence[dict[str, str]],
    verdicts: dict[str, dict[str, Any]],
    strata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The judge against the owner, on the owner's marks: for 'faithful' (no unsupported or contradicted claim) and 'main event'; raw agreement
    and Cohen's kappa by group. A blank mark stops the check; a summary the judge could not read or found no claim in is left out.

    With the strata of the slice (``write_tag_sheet`` with the verdicts) the 'faithful' question also gives the owner's rate of unfaithful
    summaries among the judge's flagged ones (``precision`` of the judge) and among its clean ones, and the judge's ``sensitivity`` (the share of
    the owner's unfaithful summaries it flagged), the two strata weighed by their size in the sample. Without them the slice is random and
    says so: it cannot measure the judge on a rare event."""
    yes = {"si": True, "sí": True, "no": False}
    out: dict[str, Any] = {}
    _check_marks(tags, yes)
    chosen_by = (strata or {}).get("judge")
    if chosen_by and (chosen_by["model"], chosen_by["prompt_sha"]) != judge_identity(verdicts):
        raise ValueError(
            "the verdicts were made by another judge than the one that chose the slice "
            f"({chosen_by['model']}, {chosen_by['prompt_sha']}): the precision and the sensitivity would be of nobody"
        )
    for group in GROUPS:
        mine = [
            t
            for t in tags
            if t["grupo"] == group
            and t["id"] in verdicts
            and not verdicts[t["id"]].get("parse_failed")
        ]
        faithful, event = [], []
        flagged_marks, clean_marks = [], []
        for t in mine:
            m = item_measures(verdicts[t["id"]])
            owner_unfaithful = yes[t["afirmaciones_sin_respaldo"].strip().lower()]
            if m["clean"] is not None:
                faithful.append((bool(m["clean"]), not owner_unfaithful))
                kind = (strata or {}).get("stratum", {}).get(t["id"])
                if kind == "flagged":
                    flagged_marks.append(owner_unfaithful)
                elif kind == "control":
                    clean_marks.append(owner_unfaithful)
            event.append((bool(m["covered"]), yes[t["evento_principal"].strip().lower()]))
        if not mine:
            continue
        out[group] = {
            name: {
                "n": len(pairs),
                "agree": sum(1 for a, b in pairs if a == b),
                "raw": sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else None,
                "kappa": _kappa(pairs),
            }
            for name, pairs in (("faithful", faithful), ("main_event", event))
        }
        if strata is None:
            out[group]["faithful"]["note"] = (
                "random slice: it cannot measure the judge on a rare event (see tag-sheet --verdicts)"
            )
            continue
        out[group]["faithful"].update(
            _strata_figures(strata.get("population", {}).get(group), flagged_marks, clean_marks)
        )
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
    p.add_argument("--out", type=Path, default=DEFAULT_DIR / "sample.csv")
    j = sub.add_parser("judge")
    j.add_argument("--sample", type=Path, default=DEFAULT_DIR / "sample.csv")
    j.add_argument("--out", type=Path, default=DEFAULT_DIR / "verdicts.jsonl")
    j.add_argument("--limit", type=int, default=None)
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
        )
        write_sample(args.out, rows)
        print(f"wrote {len(rows)} rows to {args.out}")
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
        print(f"judging with {settings.llm_model}", file=sys.stderr)
        counts = run_judge(
            rows,
            lambda: build_judge_agent(model, PROMPT),
            args.out,
            args.workers or settings.max_workers,
            args.limit,
            judge_model=settings.llm_model,
        )
        print(json.dumps(counts))
    elif args.cmd == "report":
        rows = read_sample(args.sample)
        verdicts = read_verdicts(args.verdicts)
        print(render(summarise(rows, verdicts), judge_of(verdicts)))
    elif args.cmd == "tag-sheet":
        verdicts = read_verdicts(args.verdicts) if args.verdicts else None
        ids = write_tag_sheet(
            args.out, read_sample(args.sample), args.per_group, args.seed, verdicts
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
