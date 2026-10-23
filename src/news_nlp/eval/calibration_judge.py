"""The calibration judge (PLAN.md Work item 15 step 4, TASKS T-141): the **truth label** of an
article, chosen from the full taxonomy (``other`` included) without seeing the classifier's
output, so any decision rule is scored against the same truth, below 0.6 as well.

"The argmax slug fits" is ``truth_slug == argmax slug``; for the short-circuit stratum, "``other``
is correct" is ``truth_slug == 'other'``. The judge reads the same capped lead of the article
(``sampling._MAX_BODY_CHARS``) the classifier saw, so "truth" means the best label for that
text. Verdicts are stored under their own ``eval_verdict.task`` (:data:`TASK`), so a stored-label
verdict (task ``category``) can never satisfy a lookup here, and the opposite. Opt-in: this module
is not in ``sampling.STAGES``, ``judges.JUDGES`` or ``verdicts.VERDICT_MODELS``.

**The wording in ``prompts/calibration.md`` is frozen**: ``test_eval_calibration_judge`` pins its
hash, and the reuse key carries the prompt hash and the judge model (:func:`truth_experiment`), so
a verdict made under other wording or another model is never reused. A change to the wording or to
the label set invalidates every verdict stored under :data:`TASK` and costs a re-judge, to be
budgeted separately.

**Spend.** The D10 cap is on judge *calls* (a reply that needs a repair is two). A run reserves
two calls before it judges an article, counts the calls it really makes, records and commits each
verdict as it arrives, and stops at ``max_calls`` rather than starting an article it might not be
able to finish. The calls used are kept in ``eval_run.metrics_json`` as the run goes; callers pass
``remaining_calls(conn)`` so the cap holds across runs.

Free of ``strands`` / ``mlflow`` imports: the agent comes from the caller (``agent_factory``).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import NamedTuple

from pydantic import BaseModel, field_validator

from news_nlp.db import NewsNlpDatabase
from news_nlp.eval.calibration_sample import HALVES, CalibrationSample, Half, run_config
from news_nlp.eval.config import JUDGE_TEMPERATURE
from news_nlp.eval.judges import JudgeAgent, _article_block, _invoke_with_repair, load_prompt
from news_nlp.eval.sampling import EvalItem
from news_nlp.eval.store import (
    create_eval_run,
    find_verdict_json,
    finish_eval_run,
    record_inference,
    record_verdict,
)
from news_nlp.taxonomy import CATEGORY_SLUGS, OTHER_LABEL

#: ``eval_run.stage`` and ``eval_inference`` / ``eval_verdict`` ``task`` of this judge.
TASK = "category_truth"
PROMPT_NAME = "calibration"
#: D10 (TASKS T-142): the approved ceiling on judge calls for T-121(b).
D10_CAP = 3800
#: Calls one article can take: the reply and one repair.
CALLS_PER_ITEM = 2
_CHOICES = (*CATEGORY_SLUGS, OTHER_LABEL)


class TruthVerdict(BaseModel):
    """The judge's best-fitting label. An out-of-taxonomy slug is *rejected* (it goes to the
    repair prompt, then to a ``parse_failed`` fallback), not coerced to ``other``: here a
    silent ``other`` would bias the very rate the calibration measures."""

    truth_slug: str
    rationale: str = ""
    parse_failed: bool = False

    @field_validator("truth_slug")
    @classmethod
    def _in_taxonomy(cls, value: str) -> str:
        if value not in _CHOICES:
            raise ValueError(f"{value!r} is not one of {_CHOICES}")
        return value


def prompt_hash() -> str:
    """SHA-256 of the frozen prompt text, stored in the run's config."""
    return hashlib.sha256(load_prompt(PROMPT_NAME).encode("utf-8")).hexdigest()


def truth_experiment(judge_model: str) -> str:
    """The ``experiment`` key of this judge's rows: it names the prompt hash and the judge
    model, so reuse is only ever between identical wording and model."""
    return f"truth-{prompt_hash()[:12]}-{judge_model}"


def judge_truth(agent: JudgeAgent, item: EvalItem) -> TruthVerdict:
    """One judge call on the article alone (the classifier's output is not shown)."""
    fallback = TruthVerdict(
        truth_slug=OTHER_LABEL, rationale="judge reply could not be parsed", parse_failed=True
    )
    prompt = f"{_article_block(item)}\n\nReturn your JSON verdict."
    return _invoke_with_repair(agent, prompt, TruthVerdict, fallback=fallback)


def find_truth_verdict(
    conn: NewsNlpDatabase, item: EvalItem, *, experiment: str
) -> TruthVerdict | None:
    """A verdict already stored under :data:`TASK` for this article, experiment and
    prediction, or ``None``. A ``parse_failed`` fallback is never reused (it is re-judged)."""
    stored = find_verdict_json(
        conn,
        article_id=item.article_id,
        task=TASK,
        experiment=experiment,
        prediction_json=json.dumps(item.prediction, default=str),
    )
    if stored is None:
        return None
    verdict = TruthVerdict.model_validate_json(stored)
    return None if verdict.parse_failed else verdict


class JudgeUsage(BaseModel):
    reused: int = 0
    judged: int = 0
    parse_failed: int = 0
    calls: int = 0  # judge calls really made, repairs included
    not_judged: int = 0  # items left unjudged because the next one could not fit under the cap

    def metrics(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.model_dump().items()}


# --- the run -----------------------------------------------------------------


def start_calibration_run(
    conn: NewsNlpDatabase,
    sample: CalibrationSample,
    *,
    judge_model: str,
    judge_url: str,
    code_version: str,
    judge_seed: int | None = None,
    date: str | None = None,
) -> int:
    """Create the ``eval_run`` of one calibration sample, with the config a rerun and a
    re-weighting need (seed, band edges, cell populations, half assignment, the dropped
    articles, judge model, temperature, judge seed, prompt hash, date) in ``strata_json``."""
    config = run_config(
        sample,
        judge_model=judge_model,
        temperature=JUDGE_TEMPERATURE,
        judge_seed=judge_seed,
        prompt_hash=prompt_hash(),
        date=date or datetime.now(UTC).date().isoformat(),
    )
    run_id = create_eval_run(
        conn,
        stage=TASK,
        sample_size=len(sample.drawn),
        low_conf_n=0,
        random_n=len(sample.drawn),
        seed=sample.seed,
        judge_model=judge_model,
        judge_url=judge_url,
        code_version=code_version,
        strata_json=json.dumps(config),
        experiment=truth_experiment(judge_model),
    )
    conn.commit()
    return run_id


def _run_row(conn: NewsNlpDatabase, run_id: int) -> tuple[str, dict[str, object]]:
    row = conn.execute(
        "SELECT stage, experiment, strata_json FROM eval_run WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None or row[0] != TASK:
        raise ValueError(f"eval_run {run_id} is not a {TASK} run")
    config = json.loads(row[2])
    if config.get("prompt_hash") != prompt_hash():
        raise ValueError(
            f"eval_run {run_id} was started with another prompt (hash "
            f"{config.get('prompt_hash')!r}); the wording is frozen"
        )
    return str(row[1]), config


def calls_spent(conn: NewsNlpDatabase) -> int:
    """Judge calls used by every :data:`TASK` run so far (from each run's ``metrics_json``)."""
    rows = conn.execute("SELECT metrics_json FROM eval_run WHERE stage = ?", (TASK,)).fetchall()
    return sum(int(json.loads(r[0] or "{}").get("calls", 0)) for r in rows)


def remaining_calls(conn: NewsNlpDatabase, cap: int = D10_CAP) -> int:
    return max(cap - calls_spent(conn), 0)


class _Budget:
    """Reserves up to :data:`CALLS_PER_ITEM` calls per article, thread-safely."""

    def __init__(self, cap: int) -> None:
        self._free = cap
        self._lock = threading.Lock()

    def reserve(self) -> bool:
        with self._lock:
            if self._free < CALLS_PER_ITEM:
                return False
            self._free -= CALLS_PER_ITEM
            return True

    def settle(self, used: int) -> None:
        with self._lock:
            self._free += CALLS_PER_ITEM - used


class _CountingAgent:
    def __init__(self, inner: JudgeAgent) -> None:
        self._inner = inner
        self.calls = 0

    def __call__(self, prompt: str) -> object:
        self.calls += 1
        return self._inner(prompt)


def judge_and_record(
    conn: NewsNlpDatabase,
    run_id: int,
    items: list[EvalItem],
    *,
    agent_factory: Callable[[str], JudgeAgent],
    max_calls: int,
    max_workers: int = 4,
) -> tuple[list[tuple[EvalItem, TruthVerdict]], JudgeUsage]:
    """Reuse or judge every item and record it under *run_id*; the result is the
    ``(item, verdict)`` pairs that have a verdict, in order (fewer than *items* only if the
    cap stopped the run, ``usage.not_judged``).

    Each verdict is recorded and committed as it arrives, so a crash loses at most the calls in
    flight. ``max_calls`` bounds the judge calls of this invocation (pass
    ``remaining_calls(conn)``); it never starts an article it might not finish. The run's
    stored prompt hash must match the frozen prompt. Duplicate articles raise ``ValueError``."""
    ids = [it.article_id for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate article ids in the items to judge")
    experiment, _ = _run_row(conn, run_id)
    usage = JudgeUsage()
    results: dict[int, TruthVerdict] = {}
    pending: list[EvalItem] = []
    for item in items:
        existing = find_truth_verdict(conn, item, experiment=experiment)
        if existing is None:
            pending.append(item)
        else:
            results[item.article_id] = existing
            usage.reused += 1
            _record(conn, run_id, item, existing, experiment)
    _save_usage(conn, run_id, usage)
    conn.commit()

    prompt = load_prompt(PROMPT_NAME)
    budget = _Budget(max_calls)

    def run_one(item: EvalItem) -> tuple[EvalItem, TruthVerdict | None, int]:
        if not budget.reserve():
            return item, None, 0
        agent = _CountingAgent(agent_factory(prompt))
        try:
            verdict = judge_truth(agent, item)
        finally:
            budget.settle(agent.calls)
        return item, verdict, agent.calls

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for future in as_completed([pool.submit(run_one, it) for it in pending]):
            item, verdict, calls = future.result()
            if verdict is None:
                usage.not_judged += 1
                continue
            results[item.article_id] = verdict
            usage.judged += 1
            usage.calls += calls
            usage.parse_failed += int(verdict.parse_failed)
            _record(conn, run_id, item, verdict, experiment)
            _save_usage(conn, run_id, usage)
            conn.commit()
    finish_eval_run(
        conn,
        run_id,
        metrics=usage.metrics(),
        mlflow_run_id=None,
        status="stopped_at_cap" if usage.not_judged else "ok",
    )
    conn.commit()
    return [(it, results[it.article_id]) for it in items if it.article_id in results], usage


def _record(
    conn: NewsNlpDatabase, run_id: int, item: EvalItem, verdict: TruthVerdict, experiment: str
) -> None:
    inference_id = record_inference(
        conn,
        run_id,
        article_id=item.article_id,
        task=TASK,
        experiment=experiment,
        bucket=item.bucket,
        prediction=item.prediction,
    )
    record_verdict(
        conn,
        inference_id,
        run_id,
        article_id=item.article_id,
        task=TASK,
        experiment=experiment,
        verdict=verdict.model_dump(),
        correct=None,
        severity=None,
        rationale=verdict.rationale,
    )


def _save_usage(conn: NewsNlpDatabase, run_id: int, usage: JudgeUsage) -> None:
    conn.execute(
        "UPDATE eval_run SET metrics_json = ? WHERE id = ?",
        (json.dumps(usage.metrics()), run_id),
    )


# --- reading the verdicts back -----------------------------------------------


class Truth(NamedTuple):
    article_id: int
    cell: str
    half: Half
    weight: float
    truth_slug: str


class TruthSet(NamedTuple):
    rows: list[Truth]
    parse_failed: int  # drawn, judged, reply unusable: excluded from ``rows``
    unjudged: int  # drawn but no verdict (a run stopped at the cap)

    @property
    def parse_failed_rate(self) -> float:
        drawn = len(self.rows) + self.parse_failed + self.unjudged
        return self.parse_failed / drawn if drawn else 0.0


def load_truths(conn: NewsNlpDatabase, run_id: int) -> TruthSet:
    """The usable truth labels of one run with their population weights. ``parse_failed``
    fallbacks are excluded and counted. A weight is the cell population over the usable
    articles of that cell and half, so each half still sums to its cell's population (the
    parse failures and any unjudged articles are assumed missing at random within it)."""
    _, config = _run_row(conn, run_id)
    populations = dict(config["cell_populations"])  # type: ignore[call-overload]
    halves = dict(config["halves"])  # type: ignore[call-overload]
    judged: dict[int, TruthVerdict] = {
        int(a): TruthVerdict.model_validate_json(v)
        for a, v in conn.execute(
            "SELECT article_id, verdict_json FROM eval_verdict WHERE run_id = ? AND task = ?",
            (run_id, TASK),
        )
    }
    parse_failed = unjudged = 0
    usable: dict[tuple[str, Half], list[tuple[int, str]]] = {}
    for cell, by_half in halves.items():
        for half in HALVES:
            for article_id in by_half[half]:
                verdict = judged.get(article_id)
                if verdict is None:
                    unjudged += 1
                elif verdict.parse_failed:
                    parse_failed += 1
                else:
                    usable.setdefault((cell, half), []).append((article_id, verdict.truth_slug))
    rows = [
        Truth(a, cell, half, populations[cell] / len(members), slug)
        for (cell, half), members in sorted(usable.items())
        for a, slug in members
    ]
    return TruthSet(rows, parse_failed, unjudged)
