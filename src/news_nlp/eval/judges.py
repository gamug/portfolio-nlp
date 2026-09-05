"""One judge call per stage: prompt the agent, parse a JSON verdict, repair once.

Mirrors ``portfolio-financial-analysis``'s ``fundamental_agent.agents._synthesize``
/ ``_coerce``: extract the first ``{...}`` from the reply, validate against the
stage's pydantic model, send one ``_REPAIR_PROMPT`` on failure, then fall back to
a deterministic ``parse_failed`` verdict so one bad reply never aborts a run.

Kept free of ``strands`` / ``mlflow`` imports (the agent is a structural
``Protocol``) so ``test_eval_judges.py`` runs without the ``eval`` deps.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ValidationError

from news_nlp.eval.sampling import EvalItem
from news_nlp.eval.verdicts import (
    CategoryVerdict,
    NerVerdict,
    SentimentVerdict,
    SummaryVerdict,
)

_PROMPT_DIR = Path(__file__).parent / "prompts"
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_REPAIR_PROMPT = (
    "That was not valid JSON. Reply again with ONLY the JSON object described above, "
    "nothing else -- no prose, no markdown fences."
)


class JudgeAgent(Protocol):
    """The subset of ``strands.Agent`` the judges use: call with a string, get
    something whose ``str()`` is the assistant's final text."""

    def __call__(self, prompt: str) -> object: ...


@cache
def load_prompt(stage: str) -> str:
    """The judge system prompt for *stage* (from ``prompts/<stage>.md``)."""
    return (_PROMPT_DIR / f"{stage}.md").read_text(encoding="utf-8")


def _coerce[V: BaseModel](text: str, model_cls: type[V]) -> V | None:
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        return model_cls.model_validate_json(match.group(0))
    except ValidationError:
        return None


def _invoke_with_repair[V: BaseModel](
    agent: JudgeAgent, user_prompt: str, model_cls: type[V], *, fallback: V
) -> V:
    for prompt in (user_prompt, _REPAIR_PROMPT):
        try:
            reply = str(agent(prompt))
        except Exception:  # provider / transport / decode error -- fall back, don't abort the run
            break
        parsed = _coerce(reply, model_cls)
        if parsed is not None:
            return parsed
    return fallback


def _article_block(item: EvalItem) -> str:
    return f"ARTICLE\nTitle: {item.title}\n\n{item.body_text}"


def _prediction_block(item: EvalItem) -> str:
    return "MODEL PREDICTION\n" + json.dumps(item.prediction, indent=2, default=str)


def _user_prompt(item: EvalItem) -> str:
    return f"{_article_block(item)}\n\n{_prediction_block(item)}\n\nReturn your JSON verdict."


def judge_sentiment(agent: JudgeAgent, item: EvalItem) -> SentimentVerdict:
    model_label = str(item.prediction.get("label", "neutral"))
    fallback = SentimentVerdict(
        agrees=False,
        ideal_label="neutral",
        severity=0,
        rationale="judge reply could not be parsed",
        parse_failed=True,
    )
    verdict = _invoke_with_repair(agent, _user_prompt(item), SentimentVerdict, fallback=fallback)
    if not verdict.parse_failed:
        # Trust the label comparison over a possibly-inconsistent `agrees` flag.
        verdict.agrees = verdict.ideal_label == model_label
    return verdict


def judge_category(agent: JudgeAgent, item: EvalItem) -> CategoryVerdict:
    model_label = str(item.prediction.get("label", "other"))
    fallback = CategoryVerdict(
        agrees=False,
        ideal_slug="other",
        severity=0,
        rationale="judge reply could not be parsed",
        parse_failed=True,
    )
    verdict = _invoke_with_repair(agent, _user_prompt(item), CategoryVerdict, fallback=fallback)
    if not verdict.parse_failed:
        verdict.agrees = verdict.ideal_slug == model_label
    return verdict


def judge_ner(agent: JudgeAgent, item: EvalItem) -> NerVerdict:
    fallback = NerVerdict(rationale="judge reply could not be parsed", parse_failed=True)
    return _invoke_with_repair(agent, _user_prompt(item), NerVerdict, fallback=fallback)


def judge_c_summary(agent: JudgeAgent, item: EvalItem) -> SummaryVerdict:
    fallback = SummaryVerdict(rationale="judge reply could not be parsed", parse_failed=True)
    return _invoke_with_repair(agent, _user_prompt(item), SummaryVerdict, fallback=fallback)


JUDGES = {
    "sentiment": judge_sentiment,
    "category": judge_category,
    "ner": judge_ner,
    "c_summary": judge_c_summary,
}
