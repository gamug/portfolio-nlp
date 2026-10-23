"""Strands ``Agent`` / ``OpenAIModel`` construction for the judge.

Same pattern as ``portfolio-financial-analysis``'s ``fundamental_agent.agents``:
Strands' OpenAI provider pointed at the configured (DeepSeek) endpoint. The judge
is a single-shot classifier with no tools, so it is a plain ``Agent`` with a
system prompt -- not the metrics-master orchestrator + specialist tools that
repo uses. ``temperature=0.0`` for run-to-run stability.
"""

from __future__ import annotations

from collections.abc import Callable

from strands import Agent
from strands.models.openai import OpenAIModel

from news_nlp.eval.config import JUDGE_TEMPERATURE, EvalSettings


def build_model(settings: EvalSettings) -> OpenAIModel:
    """Point Strands' OpenAI provider at the configured endpoint."""
    return OpenAIModel(
        client_args={"api_key": settings.llm_api_key, "base_url": settings.llm_url},
        model_id=settings.llm_model,
        params={"temperature": JUDGE_TEMPERATURE, "max_tokens": 2000},
    )


def build_judge_agent(model: OpenAIModel, system_prompt: str) -> Agent:
    """A stateless judge agent: one system prompt, no tools, no stdout streaming."""
    return Agent(model=model, system_prompt=system_prompt, callback_handler=None)


def calibration_agent_factory(settings: EvalSettings) -> Callable[[str], Agent]:
    """The ``agent_factory`` of ``calibration_judge.judge_and_record``: one shared model, a
    fresh stateless agent per call (the judge module itself stays free of ``strands``)."""
    model = build_model(settings)
    return lambda prompt: build_judge_agent(model, prompt)
