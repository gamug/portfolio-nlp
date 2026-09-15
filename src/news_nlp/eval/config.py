"""Runtime configuration for ``news_nlp.eval``, sourced from the project ``.env``.

Mirrors ``portfolio-financial-analysis``'s ``fundamental_agent.config.Settings``:
the LLM is reached through Strands' ``OpenAIModel`` using ``LLM_API_KEY`` /
``LLM_MODEL`` / ``LLM_URL`` (DeepSeek in practice). ``MLFLOW_TRACKING_URI`` is
read by ``mlflow`` itself; it is surfaced here only so the CLI can echo it and
override it with ``--mlflow-uri``.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

_REQUIRED_LLM_VARS = ("LLM_API_KEY", "LLM_MODEL", "LLM_URL")

DEFAULT_TRACKING_URI = "./mlruns"
DEFAULT_SAMPLE_SIZE = 80
# Lowered from 0.6 (pre-stratification, when low_conf was the only mechanism
# surfacing informative rows) now that dedicated target_* strata
# (sampling.py) do the "surface likely-wrong rows" job more precisely. 60% of
# every run's budget going to the diagnostic-only low_conf bucket is no
# longer the right split; still fully available via --low-conf-frac for
# anyone who wants the old ratio. See docs/evaluation.md.
DEFAULT_LOW_CONF_FRAC = 0.2
DEFAULT_TARGET_FRAC = 0.6
DEFAULT_MAX_WORKERS = 4
DEFAULT_REGRESSION_TOLERANCE = 0.05


class EvalSettings(BaseModel):
    """Everything an eval run needs to reach the LLM and MLflow."""

    llm_api_key: str
    llm_model: str
    llm_url: str
    mlflow_tracking_uri: str = DEFAULT_TRACKING_URI
    sample_size: int = Field(default=DEFAULT_SAMPLE_SIZE, gt=0)
    low_conf_frac: float = Field(default=DEFAULT_LOW_CONF_FRAC, ge=0.0, le=1.0)
    target_frac: float = Field(default=DEFAULT_TARGET_FRAC, ge=0.0, le=1.0)
    seed: int | None = None
    max_workers: int = Field(default=DEFAULT_MAX_WORKERS, gt=0)

    @classmethod
    def load(
        cls,
        *,
        env_file: str | os.PathLike[str] | None = None,
        mlflow_tracking_uri: str | None = None,
        sample_size: int | None = None,
        low_conf_frac: float | None = None,
        target_frac: float | None = None,
        seed: int | None = None,
        max_workers: int | None = None,
    ) -> EvalSettings:
        """Populate the environment from ``.env`` first, then read it. Explicit
        keyword overrides (from the CLI) win over env / defaults. Raises
        ``RuntimeError`` listing every missing ``LLM_*`` variable."""
        load_dotenv(env_file, override=False)
        missing = [name for name in _REQUIRED_LLM_VARS if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                "news_nlp.eval needs an OpenAI-compatible LLM endpoint; missing "
                f"environment variable(s): {', '.join(missing)}. See docs/evaluation.md."
            )
        uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
        return cls(
            llm_api_key=os.environ["LLM_API_KEY"],
            llm_model=os.environ["LLM_MODEL"],
            llm_url=os.environ["LLM_URL"],
            mlflow_tracking_uri=uri,
            sample_size=sample_size if sample_size is not None else DEFAULT_SAMPLE_SIZE,
            low_conf_frac=low_conf_frac if low_conf_frac is not None else DEFAULT_LOW_CONF_FRAC,
            target_frac=target_frac if target_frac is not None else DEFAULT_TARGET_FRAC,
            seed=seed,
            max_workers=max_workers if max_workers is not None else DEFAULT_MAX_WORKERS,
        )
