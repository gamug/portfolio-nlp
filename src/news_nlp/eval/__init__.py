"""LLM-as-judge evaluation: the store of the ``eval_run`` / ``eval_inference`` / ``eval_verdict`` / ``eval_confusion`` tables, the MLflow tracking, the judge
model and the blind step-7 category judge (``step_category``). The sampling, the runner and the metrics of the v1 stages read the v1 tables and were
removed with them (T-220); their stored runs stay in the tables and in MLflow. See ``docs/evaluation.md``."""

from __future__ import annotations

from news_nlp.eval.config import EvalSettings

__all__ = ["EvalSettings"]
