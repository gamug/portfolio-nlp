"""Step 7, the category of each main company of a text (PLAN D39 (7), D40; T-168; **provisional**, T-174 not closed).

The text is the one step 2 left for the item (``news_summary``) and the companies are the main ones step 3 found (``news_company.is_main``). Each
(text, main company) pair gets one of the 14 labels of the taxonomy of D15 (``docs/category-taxonomy.md``), and the group of that label (option B,
owner 2026-11-13).

**The model is provisional**: the pinned zero-shot model of ``config/pipeline_models.json`` (``category``), untouched, read the way the baseline of
``scripts/category_baseline_2026_11_13.py`` read it: premise ``<company>: <text>`` (the first ``PREMISE_CHARS`` characters), one hypothesis per
label (``The news about <company> is about <definition>.``), the score is the probability of *entailment*, the label is the best score. On the
gold it agrees with the owner's label in 56.4% of the 225 train rows and 43.0% of the 135 single held-out rows; the group agrees in 62.7% and
55.6%; the bar of D24 is 80%, so the labels are a first version, not the accepted ones. A fine-tuned model replaces this one by changing the pin
(its revision is then another revision of the step); nothing else of the step changes.

``group_label`` is the group of the best label; ``group_score`` is the share of the entailment mass the group's labels hold (0 to 1);
``slug_scores`` is the JSON audit trail, one entailment probability per label. Models load on the first text, so ``--status`` and the tests need
none; CUDA when there is one, the CPU otherwise (constitution, Pipeline models 2).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from news_company_stage import company_name
from news_nlp.steps import CategoryScore
from news_summary_stage import normalized_code

DEFINITIONS: dict[str, str] = {
    "earnings_performance": "results, guidance, revenue or profit figures",
    "mergers_acquisitions": "deals, takeovers, divestitures",
    "leadership_governance": "executive appointments and departures, board actions, an executive's insider trade",
    "legal_regulatory": "litigation, regulatory action, sanctions, bankruptcy",
    "product_innovation": "product launches, R&D, technology",
    "capital_shareholder_returns": "dividends, buybacks, debt and credit actions of the company",
    "labor_human_capital": "layoffs, hiring, workforce and labour relations",
    "market_analyst_sentiment": "analyst ratings and price targets, stock price moves, an investor's bullish or bearish view",
    "partnerships_business_dev": "strategic alliances, joint ventures, distribution deals",
    "reputational_harm": "an event that damages or repairs the company's public standing: apologies, service failures, controversies, boycotts",
    "competitive_actions": "the company's own move or statement against or about a competitor: share fights, pricing pressure",
    "investor_activity": "a fund or famous investor buys or sells the stock, an activist position, 13F filings, block trades",
    "peer_comparison": "a comparison between companies: which is the better stock, who is ahead",
    "other": "an event that fits none of the other topics",
}
LABELS: tuple[str, ...] = tuple(DEFINITIONS)
#: The groups of the labels (option B, owner 2026-11-13).
GROUPS: dict[str, tuple[str, ...]] = {
    "results_capital": ("earnings_performance", "capital_shareholder_returns"),
    "market_view": ("market_analyst_sentiment", "investor_activity", "peer_comparison"),
    "deals_alliances": ("mergers_acquisitions", "partnerships_business_dev", "competitive_actions"),
    "product_people": ("product_innovation", "labor_human_capital"),
    "governance_legal_reputation": (
        "leadership_governance",
        "legal_regulatory",
        "reputational_harm",
    ),
    "other": ("other",),
}
GROUP_OF: dict[str, str] = {label: g for g, labels in GROUPS.items() for label in labels}
PREMISE = "{company}: {text}"
HYPOTHESIS = "The news about {company} is about {definition}."
PREMISE_CHARS = 2000  # the size of the summaries step 2 leaves
MAX_LENGTH = 512
_ROOT = Path(__file__).resolve().parent.parent
_PIPELINE_MODELS = _ROOT / "config" / "pipeline_models.json"


def pinned_category() -> dict[str, Any]:
    """The pinned category model of ``config/pipeline_models.json`` (``model``, ``revision``)."""
    config: dict[str, Any] = json.loads(_PIPELINE_MODELS.read_text(encoding="utf-8"))["category"]
    return config


def models_of_the_step() -> dict[str, str]:
    """The model the step runs, as ``name@revision``, from the pin (nothing is loaded): ``--status`` prints it."""
    pinned = pinned_category()
    return {"category": f"{pinned['model']}@{pinned['revision']}"}


def _module_source(module: Any) -> str:
    """The source of a module (a function of its own so a test can change what the revision reads)."""
    return Path(inspect.getsourcefile(module) or "").read_text(encoding="utf-8")


def revision_of(model_identity: str) -> str:
    """The revision of a run of the step: the model (``name@revision``), the labels with their definitions and groups, the premise and the
    hypothesis, the lengths, the version of ``transformers`` and the code of this module (comments and docstrings do not count)."""
    key = json.dumps(
        {
            "model": model_identity,
            "definitions": DEFINITIONS,
            "groups": {g: list(v) for g, v in GROUPS.items()},
            "premise": PREMISE,
            "hypothesis": HYPOTHESIS,
            "premise_chars": PREMISE_CHARS,
            "max_length": MAX_LENGTH,
            "transformers": transformers.__version__,
            "code": hashlib.sha256(
                normalized_code(_module_source(sys.modules[__name__])).encode()
            ).hexdigest()[:12],
            "company_name": hashlib.sha256(
                normalized_code(inspect.getsource(company_name)).encode()
            ).hexdigest()[:12],
        },
        sort_keys=True,
    )
    return "category-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def score_of(entailment: Sequence[float]) -> CategoryScore:
    """The label, the group and the audit trail of one pair from the entailment probability of each label (in ``LABELS`` order)."""
    best = max(range(len(LABELS)), key=lambda i: entailment[i])
    label = LABELS[best]
    group = GROUP_OF[label]
    total = sum(entailment) or 1.0
    mass = sum(entailment[LABELS.index(m)] for m in GROUPS[group])
    return CategoryScore(
        label=label,
        score=float(entailment[best]),
        group_label=group,
        group_score=float(mass / total),
        slug_scores={name: round(float(p), 6) for name, p in zip(LABELS, entailment, strict=True)},
    )


class CategoryScorer:
    """The pinned zero-shot model: ``score`` gives the entailment probability of every label for each (text, company name) pair."""

    def __init__(self, device: torch.device | None = None) -> None:
        pinned = pinned_category()
        self.model_name: str = pinned["model"]
        self.model_revision: str = pinned["revision"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: Any = None
        self._model: Any = None
        self._entailment = 0

    def _load(self) -> None:
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.model_revision
            )
            # float32: transformers 5 loads the checkpoint as fp16 by default
            self._model = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, revision=self.model_revision, dtype=torch.float32
                )
                .to(self.device)
                .eval()
            )
            self._entailment = int(self._model.config.label2id["entailment"])

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[list[float]]:
        """The entailment probability of each label for each ``(text, company name)`` pair, in ``LABELS`` order."""
        self._load()
        out: list[list[float]] = []
        for text, company in pairs:
            name = company_name(company)
            premise = PREMISE.format(company=name, text=text[:PREMISE_CHARS])
            hypotheses = [
                HYPOTHESIS.format(company=name, definition=DEFINITIONS[label]) for label in LABELS
            ]
            encoded = self._tokenizer(
                [premise] * len(hypotheses),
                hypotheses,
                truncation="only_first",
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**encoded).logits.float()
            out.append(torch.softmax(logits, dim=-1)[:, self._entailment].cpu().tolist())
        return out


class CategoryFinder:
    """What the category step runs: ``classify`` gives a ``CategoryScore`` for each ``(text, company name)`` pair."""

    def __init__(self, scorer: Any | None = None) -> None:
        self.scorer = scorer or CategoryScorer()
        self.model_name: str = self.scorer.model_name
        self.model_revision: str = self.scorer.model_revision
        self.revision = revision_of(f"{self.model_name}@{self.model_revision}")

    def classify(self, pairs: Sequence[tuple[str, str]]) -> list[CategoryScore]:
        return [score_of(p) for p in self.scorer.score(pairs)]
