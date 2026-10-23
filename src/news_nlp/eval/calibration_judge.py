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
able to finish. The calls used are kept in ``eval_run.metrics_json`` as the run goes (a resumed run
adds to them); callers pass ``remaining_calls(conn)`` so the cap holds across runs. Articles are sent
round-robin across (cell, half), seeded within each, so a stop leaves an even gap, not whole cells.
An interrupt or an outage cancels the queue (only the calls in flight finish, and their verdicts are
recorded); a provider/transport error is never recorded as a verdict, and
:data:`MAX_CONSECUTIVE_ERRORS` in a row stop the run with :class:`JudgeOutage`. A run can be resumed
by calling ``judge_and_record`` again on the same run.

Free of ``strands`` / ``mlflow`` imports: the agent comes from the caller (``agent_factory``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import random
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any, NamedTuple

from pydantic import BaseModel, field_validator

from news_nlp.db import NewsNlpDatabase
from news_nlp.eval.calibration_sample import HALVES, CalibrationSample, Half, run_config
from news_nlp.eval.config import JUDGE_TEMPERATURE
from news_nlp.eval.judges import (
    _REPAIR_PROMPT,
    JudgeAgent,
    _article_block,
    _coerce,
    load_prompt,
)
from news_nlp.eval.sampling import EvalItem
from news_nlp.eval.store import (
    create_eval_run,
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
#: Consecutive provider/transport errors after which a run stops (an outage, not bad replies).
MAX_CONSECUTIVE_ERRORS = 5
_CHOICES = (*CATEGORY_SLUGS, OTHER_LABEL)
_log = logging.getLogger(__name__)


class JudgeTransportError(RuntimeError):
    """The agent call itself failed (rate limit, 5xx, auth, network): no reply to judge. Never
    recorded as a verdict, unlike an unusable reply."""


class JudgeOutage(RuntimeError):
    """:data:`MAX_CONSECUTIVE_ERRORS` transport errors in a row: the run was stopped."""


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
    """One judge call on the article alone (the classifier's output is not shown), with one
    repair prompt for an unusable reply. A reply that stays unusable gives the ``parse_failed``
    fallback; a failure of the call itself raises :class:`JudgeTransportError`."""
    prompt = f"{_article_block(item)}\n\nReturn your JSON verdict."
    for text in (prompt, _REPAIR_PROMPT):
        try:
            reply = str(agent(text))
        except Exception as exc:
            raise JudgeTransportError(str(exc)) from exc
        parsed = _coerce(reply, TruthVerdict)
        if parsed is not None:
            return parsed
    return TruthVerdict(
        truth_slug=OTHER_LABEL, rationale="judge reply could not be parsed", parse_failed=True
    )


def find_truth_verdict(
    conn: NewsNlpDatabase, item: EvalItem, *, experiment: str
) -> TruthVerdict | None:
    """The latest *valid* verdict already stored under :data:`TASK` for this article,
    experiment and prediction, or ``None``. A ``parse_failed`` fallback is skipped, not
    reused: a newer failure never hides an older valid verdict (no second payment), and an
    article with only failures is judged again."""
    rows = conn.execute(
        "SELECT v.verdict_json FROM eval_verdict v JOIN eval_inference i ON i.id = v.inference_id "
        "WHERE v.article_id = ? AND v.task = ? AND v.experiment = ? AND i.prediction_json = ? "
        "ORDER BY v.judged_at DESC, v.id DESC",
        (item.article_id, TASK, experiment, json.dumps(item.prediction, default=str)),
    ).fetchall()
    for (stored,) in rows:
        verdict = TruthVerdict.model_validate_json(stored)
        if not verdict.parse_failed:
            return verdict
    return None


class JudgeUsage(BaseModel):
    reused: int = 0
    judged: int = 0
    parse_failed: int = 0
    calls: int = 0  # judge calls really made, repairs included
    not_judged: int = 0  # items left unjudged because the next one could not fit under the cap
    errors: int = 0  # transport errors (the call failed; nothing recorded)
    unjudged: int = 0  # items without a verdict when the invocation ended (cap, errors, outage)

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


def _read_run(conn: NewsNlpDatabase, run_id: int) -> tuple[str, dict[str, Any], JudgeUsage]:
    """``(experiment, config, usage so far)`` of a :data:`TASK` run, readable under any prompt."""
    row = conn.execute(
        "SELECT stage, experiment, strata_json, metrics_json FROM eval_run WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None or row[0] != TASK:
        raise ValueError(f"eval_run {run_id} is not a {TASK} run")
    stored = json.loads(row[3] or "{}")
    usage = JudgeUsage(**{k: int(v) for k, v in stored.items() if k in JudgeUsage.model_fields})
    return str(row[1]), json.loads(row[2]), usage


def _require_frozen(run_id: int, config: dict[str, Any]) -> None:
    """Judging needs the frozen wording; only reading results may do without."""
    if config.get("prompt_hash") != prompt_hash():
        raise ValueError(
            f"eval_run {run_id} was started with another prompt (hash "
            f"{config.get('prompt_hash')!r}); the wording is frozen"
        )


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


def _round_robin(pending: list[EvalItem], config: dict[str, Any], seed: int) -> list[EvalItem]:
    """*pending* interleaved across (cell, half), in a seeded random order inside each, so
    stopping early leaves an even gap in every group rather than whole cells."""
    group_of = {
        int(a): (cell, half)
        for cell, by_half in config.get("halves", {}).items()
        for half, ids in by_half.items()
        for a in ids
    }
    groups: dict[tuple[str, str], list[EvalItem]] = {}
    for item in pending:
        groups.setdefault(group_of.get(item.article_id, (item.bucket, "")), []).append(item)
    for key in sorted(groups):
        random.Random(f"{seed}|{key[0]}|{key[1]}").shuffle(groups[key])  # noqa: S311
    ordered: list[EvalItem] = []
    queues = [groups[key] for key in sorted(groups)]
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
    return ordered


def _forget(conn: NewsNlpDatabase, run_id: int, article_id: int) -> None:
    """Drop this run's rows of one article (a ``parse_failed`` one about to be re-judged)."""
    for table in ("eval_verdict", "eval_inference"):
        conn.execute(
            f"DELETE FROM {table} WHERE run_id = ? AND article_id = ? AND task = ?",  # noqa: S608
            (run_id, article_id, TASK),
        )


class _Judging:
    """The state of one ``judge_and_record`` invocation."""

    def __init__(
        self,
        conn: NewsNlpDatabase,
        run_id: int,
        agent_factory: Callable[[str], JudgeAgent],
        max_calls: int,
    ) -> None:
        self.conn = conn
        self.run_id = run_id
        self.agent_factory = agent_factory
        self.experiment, self.config, self.usage = _read_run(conn, run_id)
        _require_frozen(run_id, self.config)
        self.in_run: dict[int, bool] = {  # article id -> its verdict in THIS run is a parse failure
            int(a): bool(json.loads(v).get("parse_failed"))
            for a, v in conn.execute(
                "SELECT article_id, verdict_json FROM eval_verdict WHERE run_id = ? AND task = ?",
                (run_id, TASK),
            )
        }
        self.results: dict[int, TruthVerdict] = {}
        self.budget = _Budget(max_calls)
        self.max_calls = max_calls
        self.streak = 0
        self.processed: set[object] = set()
        self.usage.reused = self.usage.not_judged = self.usage.unjudged = 0  # per invocation

    def reuse(self, items: list[EvalItem]) -> list[EvalItem]:
        """Take the verdicts already stored; return the items that still need a judge call."""
        pending: list[EvalItem] = []
        for item in items:
            existing = find_truth_verdict(self.conn, item, experiment=self.experiment)
            if existing is None:
                pending.append(item)
                continue
            self.results[item.article_id] = existing
            self.usage.reused += 1
            if self.in_run.get(item.article_id) is False:  # already recorded in this run
                continue
            self.store(item, existing)
        self.save()
        self.conn.commit()
        return pending

    def store(self, item: EvalItem, verdict: TruthVerdict) -> None:
        if item.article_id in self.in_run:  # a parse failure of this run, now replaced
            _forget(self.conn, self.run_id, item.article_id)
        _record(self.conn, self.run_id, item, verdict, self.experiment)

    def save(self) -> None:
        _save_usage(self.conn, self.run_id, self.usage)

    def run_one(
        self, item: EvalItem
    ) -> tuple[EvalItem, TruthVerdict | JudgeTransportError | None, int]:
        if not self.budget.reserve():
            return item, None, 0
        agent = _CountingAgent(self.agent_factory(load_prompt(PROMPT_NAME)))
        try:
            outcome: TruthVerdict | JudgeTransportError = judge_truth(agent, item)
        except JudgeTransportError as exc:
            outcome = exc
        finally:
            self.budget.settle(agent.calls)
        return item, outcome, agent.calls

    def process(self, future: Future[Any]) -> None:
        self.processed.add(future)
        item, outcome, calls = future.result()
        self.usage.calls += calls
        if outcome is None:
            self.usage.not_judged += 1
        elif isinstance(outcome, JudgeTransportError):
            self.usage.errors += 1
            self.streak += 1
            if self.streak >= MAX_CONSECUTIVE_ERRORS:
                raise JudgeOutage(f"{self.streak} judge calls failed in a row: {outcome}")
        else:
            self.streak = 0
            self.results[item.article_id] = outcome
            self.usage.judged += 1
            self.usage.parse_failed += int(outcome.parse_failed)
            self.store(item, outcome)
            self.save()
            self.conn.commit()

    def execute(self, pending: list[EvalItem], max_workers: int) -> None:
        if self.max_calls < CALLS_PER_ITEM * len(pending):
            _log.warning(
                "%d calls left for %d articles to judge (up to %d each): the cap may stop this run",
                self.max_calls,
                len(pending),
                CALLS_PER_ITEM,
            )
        pool = ThreadPoolExecutor(max_workers=max_workers)
        ordered = _round_robin(pending, self.config, int(self.config["seed"]))
        futures = [pool.submit(self.run_one, it) for it in ordered]
        try:
            for future in as_completed(futures):
                self.process(future)
            pool.shutdown()
        except BaseException as exc:
            pool.shutdown(wait=True, cancel_futures=True)  # only the calls in flight finish
            self.salvage(futures)
            self.save()
            if isinstance(exc, JudgeOutage):
                self.finish("error", str(exc))
            self.conn.commit()
            raise

    def salvage(self, futures: list[Future[Any]]) -> None:
        """Keep what the calls still in flight at an interrupt paid for."""
        for future in futures:
            if future in self.processed or future.cancelled() or future.exception() is not None:
                continue
            with contextlib.suppress(JudgeOutage):
                self.process(future)

    def finish(self, status: str, error: str | None = None) -> None:
        finish_eval_run(
            self.conn,
            self.run_id,
            metrics=self.usage.metrics(),
            mlflow_run_id=None,
            status=status,
            error=error,
        )


def _final_status(usage: JudgeUsage) -> str:
    """``ok`` only when every item has a verdict."""
    if usage.not_judged:
        return "stopped_at_cap"
    return "incomplete" if usage.unjudged else "ok"


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
    flight; calling again on the same run resumes it (the usage so far is carried over, an
    article already judged in the run is not recorded twice, a ``parse_failed`` one is judged
    again). ``max_calls`` bounds the judge calls of this invocation (pass
    ``remaining_calls(conn)``); it never starts an article it might not finish. The run's stored
    prompt hash must match the frozen prompt. Duplicate articles raise ``ValueError``; so do
    :data:`MAX_CONSECUTIVE_ERRORS` transport errors in a row (:class:`JudgeOutage`)."""
    ids = [it.article_id for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate article ids in the items to judge")
    judging = _Judging(conn, run_id, agent_factory, max_calls)
    judging.execute(judging.reuse(items), max_workers)
    done = [
        (it, judging.results[it.article_id]) for it in items if it.article_id in judging.results
    ]
    usage = judging.usage
    usage.unjudged = len(items) - len(done)
    judging.finish(_final_status(usage))
    conn.commit()
    return done, judging.usage


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
    _, config, _ = _read_run(conn, run_id)  # no prompt check: reading is harmless
    populations: dict[str, int] = config["cell_populations"]
    halves: dict[str, dict[str, list[int]]] = config["halves"]
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
