"""LLM-as-judge accuracy evaluation for the news-NLP pipeline stages.

The pipeline stores sentiment / NER / category / c_summary predictions with no
accuracy measurement of any kind. This package samples a slice of those stored
predictions, has an LLM judge (a ``strands-agents`` agent pointed at an
OpenAI-compatible endpoint) score each one against the source article text, and
writes aggregate metrics + per-row verdicts to **MLflow** and to the
``eval_run`` / ``eval_judgement`` tables in the RESULTS store.

The judge is itself a model, so "agreement" is a proxy for correctness, not
correctness -- see ``docs/evaluation.md``. Each run samples ``sample_size`` rows
per stage as a fixed **60% low-confidence + 40% uniform-random** split so both
headline and worst-case accuracy are reported.

``run_eval`` (and only it) pulls in ``mlflow`` + ``strands`` -- the ``eval``
dependency group -- so it is imported lazily here: ``news_nlp.eval.metrics`` /
``.sampling`` / ``.store`` can be imported and unit-tested without those.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from news_nlp.eval.config import EvalSettings

__all__ = ["EvalSettings", "run_eval"]

if TYPE_CHECKING:
    from news_nlp.eval.runner import run_eval


def __getattr__(name: str) -> Any:
    if name == "run_eval":
        from news_nlp.eval.runner import run_eval  # noqa: PLC0415 -- lazy: pulls in mlflow/strands

        return run_eval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
