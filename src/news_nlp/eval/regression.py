"""Compare a fresh eval run's headline metric against the previous run.

Used only by the scheduled / ``--check-regression`` path. "Headline metric" per
stage is defined in ``metrics.HEADLINE``. Non-determinism in the judge is real,
so the default tolerance is 0.05 -- a drop smaller than that is not a regression.
"""

from __future__ import annotations

from dataclasses import dataclass

from news_nlp.eval.metrics import HEADLINE
from news_nlp.eval.tracking import previous_headline


@dataclass(frozen=True)
class RegressionResult:
    stage: str
    metric: str
    current: float | None
    previous: float | None
    tolerance: float
    regressed: bool

    def describe(self) -> str:
        if self.current is None:
            return f"{self.stage}: {self.metric} not in this run's metrics -- skipped"
        if self.previous is None:
            return f"{self.stage}: {self.metric}={self.current:.4f} (no prior run to compare)"
        delta = self.current - self.previous
        verb = "REGRESSED" if self.regressed else "ok"
        return (
            f"{self.stage}: {self.metric} {self.previous:.4f} -> {self.current:.4f} "
            f"({delta:+.4f}, tol {self.tolerance:.2f}) [{verb}]"
        )


def check_regression(
    stage: str,
    metrics: dict[str, float],
    *,
    tolerance: float = 0.05,
    tracking_uri: str,
) -> RegressionResult:
    """A drop of more than *tolerance* in the stage's headline metric vs the
    previous MLflow run is a regression. No prior run / metric absent -> not a
    regression (``regressed=False``)."""
    metric = HEADLINE[stage]
    current = metrics.get(metric)
    previous = previous_headline(stage, metric, tracking_uri)
    regressed = current is not None and previous is not None and current < previous - tolerance
    return RegressionResult(
        stage=stage,
        metric=metric,
        current=current,
        previous=previous,
        tolerance=tolerance,
        regressed=regressed,
    )
