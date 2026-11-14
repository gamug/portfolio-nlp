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
``slug_scores`` is the JSON audit trail, one entailment probability per label (and ``_truncated_chunks`` = 1 when the model read the pair cut at its window; read the
column through ``steps.class_scores``). ``group_score`` is **not** the ``group_score`` of the v1 ``article_category`` (a softmax over groups): see ``docs/category_step.md``. Models load on the first text, so ``--status`` and the tests need
none; CUDA when there is one, the CPU otherwise (constitution, Pipeline models 2).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from news_company_stage import company_name
from news_nlp.steps import TRUNCATED_KEY, CategoryScore
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


#: A revision of the step that was written before the revision followed the scoring only (the first run of the development, 2026-11-13: the code of commit
#: 82d6b36 over the revisions of step 2 and step 3 named in the key). Its rows are the ones this code would write: the same labels in 10,355 of 10,355 pairs
#: of its repeats and 1,000 stored pairs re-scored (``scripts/category_verify_revision_2026_11_13.py``, figures in ``docs/category_results_2026_11_13/``), so the
#: run continues it: the same command resumes, no row is moved. Key: the revision the scoring identity gives (``scoring_revision``); value: the revision
#: already in the store. **Retired** in the change that first alters a score (a definition, the premise, the model, the code of ``_read_pair`` or ``score_of``):
#: the test that pins it fails then, and the owner decides whether the legacy rows are still wanted.
LEGACY_REVISIONS: dict[str, str] = {"category-37c188900f38": "category-65f33b55b26b"}


def _scoring_code() -> str:
    """The code that decides a score: the label, the group and the share of the mass (``score_of``), the reading of the pair by the model (``_read_pair``)
    and the name of the company in the hypothesis (``company_name``), normalised (comments, docstrings and formatting do not count). The bookkeeping of
    ``CategoryScorer.score`` (which pairs were cut at the window) is not here on purpose: editing it must not restart a run."""
    return hashlib.sha256(
        "\n".join(
            normalized_code(inspect.getsource(f)) for f in (score_of, _read_pair, company_name)
        ).encode()
    ).hexdigest()[:12]


def scoring_revision(model_identity: str, summary_revision: str, companies_revision: str) -> str:
    """What the scores of a run depend on, as a revision: the model (``name@revision``), the revisions of the texts (step 2) and of the main companies (step 3)
    it reads, the labels with their definitions and groups, the premise and the hypothesis, the lengths and the code that scores. **Nothing else**: an edit of
    the command, of the storage, of the logging or of a document is not a new revision, so a run stopped and resumed after one resumes (the idempotency of the
    step does not depend on the code of the module as a whole). The inputs are part of it: a step 2 or step 3 run again at a new revision is read as new work."""
    key = json.dumps(
        {
            "model": model_identity,
            "summary_revision": summary_revision,
            "companies_revision": companies_revision,
            "definitions": DEFINITIONS,
            "groups": {g: list(v) for g, v in GROUPS.items()},
            "premise": PREMISE,
            "hypothesis": HYPOTHESIS,
            "premise_chars": PREMISE_CHARS,
            "max_length": MAX_LENGTH,
            "scoring_code": _scoring_code(),
        },
        sort_keys=True,
    )
    return "category-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def revision_of(model_identity: str, summary_revision: str, companies_revision: str) -> str:
    """The revision a run writes: ``scoring_revision``, or the legacy revision already in the store for that same scoring (``LEGACY_REVISIONS``)."""
    computed = scoring_revision(model_identity, summary_revision, companies_revision)
    return LEGACY_REVISIONS.get(computed, computed)


def code_revision(summary_revision: str, companies_revision: str) -> str:
    """The revision this code would write for those inputs, from the pin and the source (no model is loaded): ``--status`` prints it, so the rows of
    a store can be checked against the code that was merged."""
    pinned = pinned_category()
    return revision_of(
        f"{pinned['model']}@{pinned['revision']}", summary_revision, companies_revision
    )


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


def _read_pair(
    tokenizer: Any, model: Any, device: Any, entailment: int, text: str, company: str
) -> tuple[list[float], Any]:
    """What decides a score: the premise, one hypothesis per label, the model's entailment probability of each. Returns the probabilities and the encoded
    batch (the caller records from it whether the pair was cut at the window). **This function is part of the revision** (``_scoring_code``): an edit of its
    statements is a new revision, one of its comments, its docstring or its formatting is not; keep the bookkeeping out of it."""
    name = company_name(company)
    premise = PREMISE.format(company=name, text=text[:PREMISE_CHARS])
    hypotheses = [
        HYPOTHESIS.format(company=name, definition=DEFINITIONS[label]) for label in LABELS
    ]
    encoded = tokenizer(
        [premise] * len(hypotheses),
        hypotheses,
        truncation="only_first",
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**encoded).logits.float()
    return torch.softmax(logits, dim=-1)[:, entailment].cpu().tolist(), encoded


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
        #: For each pair of the last ``score`` call: whether a hypothesis filled the window of the model, so the pair was read cut there.
        self.last_truncated: list[bool] = []

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

    def warm_up(self) -> None:
        """Load the model now (the first ``score`` would otherwise load it, inside the progress bar of the run)."""
        self._load()

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[list[float]]:
        """The entailment probability of each label for each ``(text, company name)`` pair, in ``LABELS`` order."""
        self._load()
        out: list[list[float]] = []
        self.last_truncated = []
        for text, company in pairs:
            probabilities, encoded = _read_pair(
                self._tokenizer, self._model, self.device, self._entailment, text, company
            )
            self.last_truncated.append(  # the record of a pair cut at the window: not part of what a score is
                bool(
                    encoded["input_ids"].shape[-1] >= MAX_LENGTH
                    and encoded["attention_mask"][:, -1].any()
                )
            )
            out.append(probabilities)
        return out


class CategoryFinder:
    """What the category step runs: ``classify`` gives a ``CategoryScore`` for each ``(text, company name)`` pair."""

    def __init__(
        self, summary_revision: str, companies_revision: str, scorer: Any | None = None
    ) -> None:
        self.scorer = scorer or CategoryScorer()
        self.model_name: str = self.scorer.model_name
        self.model_revision: str = self.scorer.model_revision
        self.revision = revision_of(
            f"{self.model_name}@{self.model_revision}", summary_revision, companies_revision
        )

    def warm_up(self) -> None:
        warm = getattr(self.scorer, "warm_up", None)
        if warm is not None:
            warm()

    def classify(self, pairs: Sequence[tuple[str, str]]) -> list[CategoryScore]:
        """A pair the model read cut at its window carries ``TRUNCATED_KEY`` = 1 in ``slug_scores`` (absent means not cut); read the column
        through ``steps.class_scores`` (the reserved key is not a label), constitution Pipeline models 3."""
        results = [score_of(p) for p in self.scorer.score(pairs)]
        for result, cut in zip(results, getattr(self.scorer, "last_truncated", []), strict=False):
            if cut:
                result.slug_scores[TRUNCATED_KEY] = 1
        return results
