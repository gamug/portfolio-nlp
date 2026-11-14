"""Step 6, the sentiment of each main company of a text (PLAN D39 (6), D40; T-168; the model of ``docs/sentiment_training.md``, "The decision").

The text is the one step 2 left for the item (``news_summary``) and the companies are the main ones step 3 found (``news_company.is_main``). Each
(text, main company) pair gets a label (``positive``, ``negative`` or ``neutral``) and the three probabilities.

**The model is the stack the owner selected (2026-11-10, confirmed 2026-11-14)**: the pair model that reads ``[CLS] company [SEP] text [SEP]``
(``sentiment_pair``; the run ``final_all__e2_lr3e-05_s4``, started from ``gamug/FinBERT-financial-news``), whose three class probabilities are the
mean over the chunks of the text, combined with **the relevance score step 3 stored for that pair** (``news_company.relevance_score``: the same
relevance model, the same text, so nothing is scored twice) by the logistic regression of ``config/sentiment_stack.json``. The weights are the local
folder named in ``config/sentiment_step.json`` (git-ignored; its copy on the private Hub is ``gamug/FinBERT-company-sentiment@9a7b1eb3``): the step never
downloads a model, and it refuses a folder whose identity is not the pinned one.

What the stack's figures are, and are not: 0.814 on the nested cross-validation (weighted, D37), not held out; the 180-text gold has not been scored
with it, so the pin of ``config/pipeline_models.json`` is still the v1 model (T-148c4, T-193). The strict nesting of the relevance model was not done
(``docs/sentiment_training.md``, "Published"). Models load on the first text, so ``--status`` and the tests need none; CUDA when there is one, the CPU
otherwise (constitution, Pipeline models 2).

``label`` is the largest of the stack's probabilities, ``score`` that probability; ``positive``, ``negative`` and ``neutral`` are the stack's. The
pair model's own probabilities (before the stack) and the relevance score are not columns: they are the audit trail only through a rerun.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import chunking
import sentiment_pair
from chunking import chunk_text
from news_nlp import attribution_model as am
from news_nlp.steps import SentimentScore
from news_summary_stage import normalized_code

#: The order of the pair model's labels and of the stack's classes and features.
CLASSES: tuple[str, ...] = ("positive", "negative", "neutral")
_ROOT = Path(__file__).resolve().parent.parent
PIN_PATH = _ROOT / "config" / "sentiment_step.json"
STACK_PATH = _ROOT / "config" / "sentiment_stack.json"
_EPS = 1e-6  # the clip of ``scripts/final_models_2026_11_10.py`` ``feats``


def pin(pin_path: Path | None = None) -> dict[str, Any]:
    """The pin of the step: the folder of the pair model, its identity and the Hub copy it was published as."""
    config: dict[str, Any] = json.loads((pin_path or PIN_PATH).read_text(encoding="utf-8"))
    return config


def model_dir(pin_path: Path | None = None) -> Path:
    """The folder of the pair model: the pin's path, from the root of the repository unless absolute."""
    folder = Path(pin(pin_path)["model_dir"])
    return folder if folder.is_absolute() else _ROOT / folder


def stack_coefficients(stack_path: Path | None = None) -> dict[str, Any]:
    """The stack's coefficients (``config/sentiment_stack.json``); the classes are in the model's order."""
    coefficients: dict[str, Any] = json.loads(
        (stack_path or STACK_PATH).read_text(encoding="utf-8")
    )
    if tuple(coefficients["classes"]) != CLASSES:
        raise ValueError(f"the stack's classes are {coefficients['classes']}, not {list(CLASSES)}")
    return coefficients


def stack_identity(stack_path: Path | None = None) -> str:
    """A hash of the stack file: what the stack's probabilities depend on."""
    return hashlib.sha256((stack_path or STACK_PATH).read_bytes()).hexdigest()[:16]


def models_of_the_step(pin_path: Path | None = None) -> dict[str, str]:
    """The model the step runs, from the pin (nothing is loaded): ``--status`` prints it. Its Hub copy is a record, never read by the step."""
    config = pin(pin_path)
    published = config["published"]
    return {
        "sentiment": f"{config['model_dir']} (identity {config['identity']})",
        "published_as": f"{published['repo']}@{published['revision']}",
        "stack": f"config/sentiment_stack.json ({stack_identity()})",
        "relevance": "the score step 3 stored for the pair (news_company.relevance_score)",
    }


def stacked(
    coefficients: dict[str, Any], probabilities: Sequence[float], relevance: float
) -> dict[str, float]:
    """The stack's probabilities of one pair: the multinomial logistic regression of the log of the pair model's three probabilities and the logit of
    the relevance score (``scripts/final_models_2026_11_10.py`` ``apply_stack``, the same arithmetic). The scoring code of the revision."""
    r = min(max(relevance, _EPS), 1 - _EPS)
    features = [math.log(min(max(p, _EPS), 1.0)) for p in probabilities] + [math.log(r / (1 - r))]
    z = [
        sum(c * f for c, f in zip(row, features, strict=True)) + b
        for row, b in zip(coefficients["coef"], coefficients["intercept"], strict=True)
    ]
    top = max(z)
    e = [math.exp(v - top) for v in z]
    total = sum(e)
    return dict(zip(coefficients["classes"], (v / total for v in e), strict=True))


def score_of(
    coefficients: dict[str, Any], probabilities: Sequence[float], relevance: float
) -> SentimentScore:
    """The label, its probability and the three probabilities of one pair, from the pair model's probabilities (mean over the chunks) and the relevance."""
    p = stacked(coefficients, probabilities, relevance)
    label = max(CLASSES, key=lambda c: p[c])
    return SentimentScore(
        label=label,
        score=float(p[label]),
        positive=float(p["positive"]),
        negative=float(p["negative"]),
        neutral=float(p["neutral"]),
    )


def _scoring_code() -> str:
    """The code that decides a score: the stack's arithmetic (``stacked``), the label (``score_of``), the reading of the pair by the model
    (``_pair_probabilities``) and the cut of the text it calls (``chunking.chunk_text``, ``sentiment_pair.chunk_budget``,
    ``sentiment_pair.encode_pair``), normalised (comments, docstrings and formatting do not count). The helpers those four call
    (the sentence splitter and the like) are not hashed: a change there needs a new revision by hand (``docs/sentiment_step.md``). The
    bookkeeping (which texts were cut) is not here on purpose: editing it must not restart a run."""
    return hashlib.sha256(
        "\n".join(
            normalized_code(inspect.getsource(f))
            for f in (
                stacked,
                score_of,
                _pair_probabilities,
                chunking.chunk_text,
                sentiment_pair.chunk_budget,
                sentiment_pair.encode_pair,
            )
        ).encode()
    ).hexdigest()[:12]


def scoring_revision(
    model_identity: str, stack: str, summary_revision: str, companies_revision: str
) -> str:
    """What the scores of a run depend on, as a revision: the pair model's identity (a hash of its files), the stack file, the revisions of the texts
    (step 2) and of the main companies and their relevance scores (step 3) it reads, the window and the code that scores. **Nothing else**: an edit of the
    command, of the storage, of the logging or of a document is not a new revision, so a run stopped and resumed after one resumes. The inputs are part of
    it: a step 2 or step 3 run again at a new revision is read as new work."""
    key = json.dumps(
        {
            "model": model_identity,
            "stack": stack,
            "summary_revision": summary_revision,
            "companies_revision": companies_revision,
            "max_length": sentiment_pair.MAX_LENGTH,
            "company_chars": sentiment_pair.MAX_COMPANY_CHARS,
            "scoring_code": _scoring_code(),
        },
        sort_keys=True,
    )
    return "sentiment-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def _pair_probabilities(tokenizer: Any, model: Any, device: Any, company: str, text: str) -> Any:
    """What decides the pair model's reading: the text cut in chunks that leave room for the company, each read as ``[CLS] company [SEP] chunk [SEP]``,
    the class probabilities averaged over the chunks with equal weights (the reading of ``scripts/draw_sentiment_gold_2026_11_05.py``
    ``predict_probs``, the one the cross-validation scored). Returns ``(mean probabilities, number of chunks)``; ``None`` when the text has no chunk.
    **Part of the revision** (``_scoring_code``): an edit of its statements is a new revision."""
    budget = sentiment_pair.chunk_budget(tokenizer, company)
    chunks = chunk_text(text, tokenizer, max_tokens=budget)
    if not chunks:
        return None
    rows = []
    for chunk in chunks:
        inputs = sentiment_pair.encode_pair(tokenizer, company, chunk.text, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            rows.append(torch.softmax(model(**inputs).logits[0].float(), dim=-1).cpu().tolist())
    return [sum(column) / len(rows) for column in zip(*rows, strict=True)], len(rows)


class SentimentScorer:
    """The pair model: ``probabilities`` gives the mean class probabilities of each ``(text, company name)`` pair, before the stack."""

    def __init__(self, folder: Path | None = None, device: torch.device | None = None) -> None:
        self.folder = folder or model_dir()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: Any = None
        self._model: Any = None
        self._order: list[int] = []

    def identity(self) -> str:
        """The identity of the folder (a hash of its files): computed once per run, the weights are large."""
        return am.directory_identity(self.folder)

    def _load(self) -> None:
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.folder))
            # float32: transformers 5 loads the checkpoint as fp16 by default, and the model was trained and scored in float32
            self._model = (
                AutoModelForSequenceClassification.from_pretrained(
                    str(self.folder), dtype=torch.float32
                )
                .to(self.device)
                .eval()
            )
            labels = {int(i): str(v).lower() for i, v in self._model.config.id2label.items()}
            if sorted(labels.values()) != sorted(CLASSES):
                raise ValueError(f"the model's labels are {labels}, not {list(CLASSES)}")
            # the position of each class of CLASSES in the model's output
            self._order = [next(i for i, v in labels.items() if v == c) for c in CLASSES]

    def warm_up(self) -> None:
        """Load the model now (the first ``probabilities`` would otherwise load it, inside the progress bar of the run)."""
        self._load()

    def probabilities(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[tuple[list[float], int] | None]:
        """For each ``(text, company name)`` pair, the mean class probabilities in ``CLASSES`` order and the number of chunks; ``None`` when the text has
        no chunk (nothing to read). The name is the universe's, as the model was trained on it (``(Class A)`` and the like included)."""
        self._load()
        out: list[tuple[list[float], int] | None] = []
        for text, company in pairs:
            read = _pair_probabilities(self._tokenizer, self._model, self.device, company, text)
            if read is None:
                out.append(None)
                continue
            mean, chunks = read
            out.append(([mean[i] for i in self._order], chunks))
        return out


class SentimentFinder:
    """What the sentiment step runs: ``classify`` gives a ``SentimentScore`` for each ``(text, company name, relevance)`` triple."""

    def __init__(
        self,
        summary_revision: str,
        companies_revision: str,
        scorer: Any | None = None,
        coefficients: dict[str, Any] | None = None,
    ) -> None:
        self.scorer = scorer or SentimentScorer()
        self.coefficients = coefficients or stack_coefficients()
        published = pin()["published"]
        self.model_name: str = f"{published['repo']}@{published['revision']}"
        self.model_identity: str = self.scorer.identity()
        self.revision = scoring_revision(
            self.model_identity, stack_identity(), summary_revision, companies_revision
        )

    def warm_up(self) -> None:
        warm = getattr(self.scorer, "warm_up", None)
        if warm is not None:
            warm()

    def classify(self, triples: Sequence[tuple[str, str, float]]) -> list[SentimentScore | None]:
        """The score of each ``(text, company name, relevance score)``; ``None`` for a text the model has no chunk to read."""
        reads = self.scorer.probabilities([(text, company) for text, company, _ in triples])
        return [
            None if read is None else score_of(self.coefficients, read[0], relevance)
            for read, (_, _, relevance) in zip(reads, triples, strict=True)
        ]


def code_revision(summary_revision: str, companies_revision: str) -> str:
    """The revision this code would write for those inputs, from the pin and the source (no model is loaded, no weight is read): ``--status`` prints it.
    The identity is the pin's; the run checks that the folder has it."""
    return scoring_revision(
        pin()["identity"], stack_identity(), summary_revision, companies_revision
    )
