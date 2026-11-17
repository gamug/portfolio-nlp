"""The stage models as a service (T-219): the pinned model of a stage, loaded on demand, asked about a few texts.

For each stage that classifies, the object the step itself runs (``ScopeClassifier``, ``SectorClassifier``, ``CompanyFinder``,
``CategoryFinder``, ``SentimentFinder``) is built the way the step builds it and called with ad-hoc texts: no database row is written. The GPU is
shared with the jobs through ``GpuGate`` (a call while a job runs is refused, and the other way round) and **one stage stays loaded**: asking
for another (or the same one with other options) frees the first (a 6 GB card). What the v2 stages do not have is a ``Feature`` class: only the NER of
step 3 uses ``fti.Feature``, so the features are internal to each finder and no ``features`` route exists; the training of any stage stays in the
scripts (owner 2026-11-17).

What the CLI does not guess, the service does not guess: the scope model must be the pinned one (``main_scope`` refuses another), the companies
stage needs its ``threshold`` and ``candidates`` (part of the revision, the owner's), and a sentiment item carries the ``relevance`` it was scored with.
Each adapter takes its heavy parts as keyword arguments so a test builds the real one with stubs.
"""

from __future__ import annotations

import dataclasses
import gc
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from news_nlp.service.jobs import Busy, GpuGate

MAX_ITEMS = 32
STAGES = ("scope", "sector", "companies", "category", "sentiment")
ConnFactory = Callable[[], Any]


class StageUnavailable(RuntimeError):
    """The stage cannot be served here (no GPU, no model folder, not the pinned model, no universe in the SOURCE store)."""


class StageModel(Protocol):
    def predict(self, items: Sequence[Mapping[str, Any]]) -> list[Any]: ...


def jsonable(value: Any) -> Any:  # noqa: PLR0911 -- one return for each kind of value
    """A result of a finder as plain JSON data (named tuples, dataclasses, pydantic models, numpy numbers, nested lists)."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple) and hasattr(value, "_asdict"):  # a named tuple
        return {k: jsonable(v) for k, v in value._asdict().items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        return jsonable(value.item())
    return str(value)


def _need(item: Mapping[str, Any], *keys: str) -> list[Any]:
    missing = [k for k in keys if k not in item]
    if missing:
        raise ValueError(f"each item needs {', '.join(keys)} (missing: {', '.join(missing)})")
    return [item[k] for k in keys]


class _Scope:
    description = "items: {title, body}; the pinned model only"

    def __init__(
        self,
        conn_factory: ConnFactory,
        options: Mapping[str, Any] | None = None,
        *,
        model_dir: Path | None = None,
    ) -> None:
        from news_nlp.service.steps_runner import cli  # noqa: PLC0415
        from scope_stage import (  # noqa: PLC0415 -- the pin check does not load a model
            ScopeClassifier,
            is_pinned,
            revision_of,
        )

        folder = model_dir or cli().default_model_dir("scope")
        if not (folder / "training_record.json").exists():
            raise StageUnavailable(
                f"{folder}/training_record.json is missing: train or fetch the model first"
            )
        revision = revision_of(folder)
        if (
            is_pinned(revision) is False
        ):  # main_scope makes the same check; --allow-unpinned is not available here
            raise StageUnavailable(
                f"{revision} is not the pinned model (config/scope_model.json): the service serves the pinned scope model only"
            )
        self.model = ScopeClassifier(folder)

    def predict(self, items: Sequence[Mapping[str, Any]]) -> list[Any]:
        return list(self.model.classify([tuple(_need(i, "title", "body")) for i in items]))  # type: ignore[arg-type]


class _Sector:
    description = "items: {title, text}"

    def __init__(
        self,
        conn_factory: ConnFactory,
        options: Mapping[str, Any] | None = None,
        *,
        embed: Any | None = None,
    ) -> None:
        from cut_stage import companies_of  # noqa: PLC0415
        from sector_stage import SectorClassifier  # noqa: PLC0415

        conn = conn_factory()
        try:
            companies = companies_of(conn)
        finally:
            conn.close()
        if not companies:
            raise StageUnavailable(
                "the SOURCE store has no companies: the sector stage needs the universe"
            )
        self.model = SectorClassifier(companies, embed=embed)

    def predict(self, items: Sequence[Mapping[str, Any]]) -> list[Any]:
        return list(self.model.classify([tuple(_need(i, "title", "text")) for i in items]))  # type: ignore[arg-type]


class _Companies:
    description = "options: {threshold, candidates} (both required, as the CLI); items: {text}"

    def __init__(
        self,
        conn_factory: ConnFactory,
        options: Mapping[str, Any] | None = None,
        *,
        spans: Any | None = None,
        scorer: Any | None = None,
    ) -> None:
        from cut_stage import companies_of  # noqa: PLC0415
        from news_company_stage import CompanyFinder  # noqa: PLC0415

        opts = options or {}
        missing = [k for k in ("threshold", "candidates") if opts.get(k) is None]
        if (
            missing
        ):  # the value is the owner's and part of the revision (D39 (i), T-169): not guessed here
            raise ValueError(
                f"the companies stage needs options {', '.join(missing)} (as the CLI: --threshold, --candidates)"
            )
        conn = conn_factory()
        try:
            universe = companies_of(conn)
        finally:
            conn.close()
        if not universe:
            raise StageUnavailable(
                "the SOURCE store has no companies: the companies stage needs the universe"
            )
        self.model = CompanyFinder(
            universe, float(opts["threshold"]), str(opts["candidates"]), spans=spans, scorer=scorer
        )

    def predict(self, items: Sequence[Mapping[str, Any]]) -> list[Any]:
        return list(self.model.find([str(_need(i, "text")[0]) for i in items]))


class _Category:
    description = "items: {text, company}"

    def __init__(
        self,
        conn_factory: ConnFactory,
        options: Mapping[str, Any] | None = None,
        *,
        scorer: Any | None = None,
    ) -> None:
        from news_category_stage import CategoryFinder  # noqa: PLC0415 -- loads torch

        # the two revisions only name a run; they do not change a score
        self.model = CategoryFinder("adhoc", "adhoc", scorer=scorer)

    def predict(self, items: Sequence[Mapping[str, Any]]) -> list[Any]:
        return list(self.model.classify([tuple(_need(i, "text", "company")) for i in items]))  # type: ignore[arg-type]


class _Sentiment:
    description = "items: {text, company, relevance}; relevance (0 to 1) is required: the relevance model's score of that company in the text"

    def __init__(
        self,
        conn_factory: ConnFactory,
        options: Mapping[str, Any] | None = None,
        *,
        scorer: Any | None = None,
    ) -> None:
        from news_sentiment_stage import SentimentFinder  # noqa: PLC0415 -- loads torch

        self.model = SentimentFinder("adhoc", "adhoc", "adhoc", scorer=scorer)

    def predict(self, items: Sequence[Mapping[str, Any]]) -> list[Any]:
        triples = [
            (str(text), str(company), float(relevance))
            for text, company, relevance in (
                _need(i, "text", "company", "relevance") for i in items
            )
        ]
        return list(self.model.classify(triples))


BUILDERS: dict[str, Callable[..., StageModel]] = {
    "scope": _Scope,
    "sector": _Sector,
    "companies": _Companies,
    "category": _Category,
    "sentiment": _Sentiment,
}


class StageModels:
    """Builds a stage model when asked, keeps one, and refuses a call while a job holds the GPU."""

    def __init__(
        self,
        gate: GpuGate,
        conn_factory: ConnFactory,
        builders: Mapping[str, Callable[..., StageModel]] | None = None,
    ) -> None:
        self.gate = gate
        self.conn_factory = conn_factory
        self.builders = dict(builders or BUILDERS)
        self._loaded: tuple[tuple[str, tuple[tuple[str, str], ...]], StageModel] | None = None

    def stages(self) -> dict[str, dict[str, str]]:
        return {
            name: {
                "predict": getattr(builder, "description", ""),
                "loaded": "yes" if self._loaded and self._loaded[0][0] == name else "no",
            }
            for name, builder in self.builders.items()
        }

    def predict(
        self,
        stage: str,
        items: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        if stage not in self.builders:
            raise KeyError(stage)
        if not items or len(items) > MAX_ITEMS:
            raise ValueError(f"between 1 and {MAX_ITEMS} items per call")
        if not self.gate.acquire(f"stage model {stage}"):
            raise Busy(f"the GPU is in use by {self.gate.holder}")
        try:
            key = (stage, tuple(sorted((k, str(v)) for k, v in (options or {}).items())))
            if self._loaded is None or self._loaded[0] != key:  # another stage, or other options
                self._free()
                self._loaded = (key, self.builders[stage](self.conn_factory, options or {}))
            return [jsonable(r) for r in self._loaded[1].predict(items)]
        finally:
            self.gate.release()

    def _free(self) -> None:
        self._loaded = None
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
