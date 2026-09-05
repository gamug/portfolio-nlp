"""Strands ``Agent`` / ``OpenAIModel`` construction for the judge.

Same pattern as ``portfolio-financial-analysis``'s ``fundamental_agent.agents``:
Strands' OpenAI provider pointed at the configured (DeepSeek) endpoint. The judge
is a single-shot classifier with no tools, so it is a plain ``Agent`` with a
system prompt -- not the metrics-master orchestrator + specialist tools that
repo uses. ``temperature=0.0`` for run-to-run stability.
"""

from __future__ import annotations

from strands import Agent
from strands.models.openai import OpenAIModel

from news_nlp.eval.config import EvalSettings


def build_model(settings: EvalSettings) -> OpenAIModel:
    """Point Strands' OpenAI provider at the configured endpoint."""
    return OpenAIModel(
        client_args={"api_key": settings.llm_api_key, "base_url": settings.llm_url},
        model_id=settings.llm_model,
        params={"temperature": 0.0, "max_tokens": 2000},
    )


def build_judge_agent(model: OpenAIModel, system_prompt: str) -> Agent:
    """A stateless judge agent: one system prompt, no tools, no stdout streaming."""
    return Agent(model=model, system_prompt=system_prompt, callback_handler=None)
