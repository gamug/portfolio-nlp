"""The calibration judge (PLAN.md Work item 15 step 4, TASKS T-141): the **truth label** of an
article, chosen from the full taxonomy (``other`` included) without seeing the classifier's
output, so any decision rule is scored against the same truth, below 0.6 as well.

"The argmax slug fits" is ``truth_slug == argmax slug``; for the short-circuit stratum, "``other``
is correct" is ``truth_slug == 'other'``. Verdicts are stored under their own ``eval_verdict.task``
(:data:`TASK`), so a stored-label verdict (task ``category``) can never satisfy a lookup here, and
the opposite. Opt-in: this module is not in ``sampling.STAGES``, ``judges.JUDGES`` or
``verdicts.VERDICT_MODELS``, so ``--stage all`` and ``ExperimentSpec`` do not see it.

**The wording in ``prompts/calibration.md`` is frozen**: ``test_eval_calibration_judge`` pins its
hash. A change to it, or to the label set, invalidates every verdict stored under
:data:`TASK` and costs a re-judge, to be budgeted separately.

Free of ``strands`` / ``mlflow`` imports: the agent comes from the caller (``agent_factory``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, field_validator

from news_nlp.db import NewsNlpDatabase
from news_nlp.eval.judges import JudgeAgent, _article_block, _invoke_with_repair, load_prompt
from news_nlp.eval.sampling import EvalItem
from news_nlp.eval.store import find_verdict_json, record_inference, record_verdict
from news_nlp.taxonomy import CATEGORY_SLUGS, OTHER_LABEL

#: ``eval_run.stage`` and ``eval_inference`` / ``eval_verdict`` ``task`` of this judge.
TASK = "category_truth"
PROMPT_NAME = "calibration"
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


def judge_and_record(
    conn: NewsNlpDatabase,
    run_id: int,
    items: list[EvalItem],
    *,
    agent_factory: Callable[[str], JudgeAgent],
    experiment: str,
    max_workers: int = 4,
    max_calls: int | None = None,
) -> tuple[list[TruthVerdict], JudgeUsage]:
    """One verdict per item, same order: reused if stored, else judged (a worker pool) and
    recorded under *run_id*. ``max_calls`` is the spend cap (D10): if more items need a judge
    call than that, nothing is judged and ``ValueError`` is raised, so a run never half-spends."""
    usage = JudgeUsage()
    verdicts: list[TruthVerdict | None] = [None] * len(items)
    pending: list[int] = []
    for i, item in enumerate(items):
        existing = find_truth_verdict(conn, item, experiment=experiment)
        if existing is None:
            pending.append(i)
        else:
            verdicts[i] = existing
            usage.reused += 1
    if max_calls is not None and len(pending) > max_calls:
        raise ValueError(f"{len(pending)} items need a judge call, over the cap of {max_calls}")

    prompt = load_prompt(PROMPT_NAME)

    def run_one(i: int) -> tuple[int, TruthVerdict]:
        return i, judge_truth(agent_factory(prompt), items[i])

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, verdict in pool.map(run_one, pending):
            verdicts[i] = verdict
            usage.judged += 1
            usage.parse_failed += int(verdict.parse_failed)

    resolved = [v for v in verdicts if v is not None]  # every slot filled by construction
    for item, verdict in zip(items, resolved, strict=True):
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
    return resolved, usage
