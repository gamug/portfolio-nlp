"""Step 3, the companies: which universe companies a text names, and which of them it is mainly about (PLAN D39 (2), (3.1), (3.2), D40; T-164, T-165).

The text is the one step 2 left for the item (``news_summary``: the summary, or the item itself when it was under the size), a single body or a
sub-news alike. Three things happen to it:

1. **NER** (``config/pipeline_models.json`` ``ner``) finds the organisation spans (3.1);
2. **the matcher** (``news_nlp.triage.CompanyResolver.ticker_of_name``) turns each span into a universe ticker; a span that matches no ticker is
   ignored (D39 (iii)), and every comparison from here on is by ticker. The number of spans that name a ticker is its ``mention_count``;
3. **the relevance model** (``gamug/deberta-v3-company-relevance``, ``config/published_models.json``, private) scores each (text, ticker) pair (3.2).
   It reads the premise ``<text>`` and the hypothesis ``This text is mainly about <company>.`` the way it was trained
   (``scripts/final_models_2026_11_10.py``, ``docs/sentiment_training.md``), and the score is the probability of *entailment*.

Every ticker whose score is at the threshold or over it is main (``steps.is_main``); the others stay as mentions. The threshold belongs to the revision of the
step (D39 (vi)): another threshold is another set of rows. Models load on the first text, so ``--status`` and the tests need none; CUDA when there is one,
the CPU otherwise (constitution, Pipeline models 2).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import chunking
import ner_stage
from news_nlp import triage
from news_nlp.steps import CompanyScore
from news_nlp.triage import Company, CompanyResolver, in_disclosure
from news_summary_stage import normalized_code

HYPOTHESIS = "This text is mainly about {}."
MAX_LENGTH = 320  # the length the relevance model was trained and scored at (RECIPE["max_length"])
RELEVANCE_BATCH = 32
ENTITY_TYPE = "ORG"
#: Where the candidate companies of a text come from: NER alone (D39 (3.1)), or NER and the matcher over the text.
CANDIDATES = ("ner", "ner+text")
_ROOT = Path(__file__).resolve().parent.parent
_PIPELINE_MODELS = _ROOT / "config" / "pipeline_models.json"
_PUBLISHED_MODELS = _ROOT / "config" / "published_models.json"


def company_name(company: str) -> str:
    """The name as the relevance model was trained on it: the universe name without its parenthesised share class."""
    return company.split(" (", maxsplit=1)[0]


#: Boilerplate words of a disclosure that name a bank or a fund without the text being about it ("An employee of Citigroup Global Markets or its
#: affiliates is a trustee of (TGT)"); looked for in a window around the match, as ``triage``'s own rules do for the phrases it knows.
_DISCLOSURE_WORDS = re.compile(
    r"\b(?:or\s+(?:its\s+)?affiliates|an\s+associate\s+of|is\s+a\s+trustee\s+of|investment\s+banking\s+client|"
    r"within\s+the\s+past\s+12\s+months|employees?\s+of\s+\w+\s+own)\b",
    re.IGNORECASE,
)
_DISCLOSURE_WINDOW = 160


def _in_disclosure(text: str, start: int, end: int) -> bool:
    """Whether the match at ``text[start:end]`` sits in a disclosure: a phrase ``triage`` knows (``long``, ``owns``, ...), a run of five or
    more tickers whatever number of them are in the universe (``triage`` asks for three; a run of ticker symbols is not prose), or one of the
    boilerplate words of ``_DISCLOSURE_WORDS``."""
    if in_disclosure(text, start, end, None):
        return True
    low = max(0, start - _DISCLOSURE_WINDOW)
    return bool(_DISCLOSURE_WORDS.search(text[low : end + _DISCLOSURE_WINDOW]))


def direct_mentions(resolver: CompanyResolver, text: str) -> dict[str, int]:
    """How many times the matcher finds each ticker in the text itself, for the candidates NER did not tag (``--candidates ner+text``).

    The mentions ``CompanyResolver.counted_matches`` counts (a word that is also a surname or an everyday word needs the company's full name
    in the text), minus two kinds that name a company without the text being about it: a match that does not start with a capital letter or
    a digit (the patterns ignore the case: "the gap", "apple season"), and one inside a disclosure (``_in_disclosure``: "Tim Seymour is long
    AMZN, AAPL, ...", a list of ticker symbols, "Citigroup Global Markets or its affiliates"). The disclosure rules are word and pattern
    matches: a disclosure written in other words passes (``docs/companies_step.md``)."""
    counts: Counter[str] = Counter()
    for match, ticker in resolver.counted_matches(text):
        first = match.group(0)[:1]
        if not (first.isupper() or first.isdigit()):
            continue
        if _in_disclosure(text, match.start(), match.end()):
            continue
        counts[ticker] += 1
    return dict(counts)


def mentions_of(
    resolver: CompanyResolver, spans: Sequence[Mapping[str, Any]], text: str | None = None
) -> dict[str, int]:
    """How many organisation spans name each universe ticker; a span with no ticker is ignored.

    With ``text`` the matcher also reads the text itself (``direct_mentions``) and a ticker it finds that NER did not tag is a candidate
    too; a ticker both find keeps the larger of the two counts (one mention is not counted twice, and ``mention_count`` is a count of
    mentions, not of the two readers')."""
    counts: Counter[str] = Counter()
    for span in spans:
        if span.get("entity_type") != ENTITY_TYPE:
            continue
        ticker = resolver.ticker_of_name(str(span.get("text", "")))
        if ticker is not None:
            counts[ticker] += 1
    if text is not None:
        for ticker, n in direct_mentions(resolver, text).items():
            counts[ticker] = max(counts[ticker], n)
    return dict(counts)


def pinned_ner() -> dict[str, Any]:
    """The pinned NER model of ``config/pipeline_models.json`` (``model``, ``revision``, ``batch_size``)."""
    config: dict[str, Any] = json.loads(_PIPELINE_MODELS.read_text(encoding="utf-8"))["ner"]
    return config


def published_relevance() -> dict[str, Any]:
    """The relevance model of ``config/published_models.json`` (``repo``, ``revision``; the repository is private: ``HF_TOKEN``)."""
    config: dict[str, Any] = json.loads(_PUBLISHED_MODELS.read_text(encoding="utf-8"))["relevance"]
    return config


def models_of_the_step() -> dict[str, Any]:
    """The two models the step runs, as ``name@revision``, from the pins (no model is loaded): ``--status`` prints them. The relevance
    repository is private (``HF_TOKEN``) and is not pinned in ``config/pipeline_models.json`` (T-148c4)."""
    ner, relevance = pinned_ner(), published_relevance()
    return {
        "ner": f"{ner['model']}@{ner['revision']}",
        "relevance": f"{relevance['repo']}@{relevance['revision']}",
        "relevance_private": bool(relevance.get("private", False)),
    }


def _module_source(module: Any) -> str:
    """The source of a module (a function of its own so a test can change what the revision reads)."""
    return Path(inspect.getsourcefile(module) or "").read_text(encoding="utf-8")


def _definitions(tree: ast.Module) -> dict[str, ast.AST]:
    """The functions, classes and assigned constants of a module's top level, by name."""
    defs: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            defs[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defs[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defs[node.target.id] = node
    return defs


def _closure(module: Any, roots: Sequence[str]) -> dict[str, str]:
    """The code a set of names in a module reaches, normalised: each root and, transitively, every other top-level function, class or constant
    of the same module that its code names (``normalized_code`` of the node's source: comments, formatting and docstrings do not count). This is
    how the revision follows the helpers and the constants a matcher rule reads, without a list kept by hand."""
    source = _module_source(module)
    defs = _definitions(ast.parse(source))
    seen: set[str] = set()
    queue = list(roots)
    while queue:
        name = queue.pop()
        if name in seen or name not in defs:
            continue
        seen.add(name)
        queue.extend(n.id for n in ast.walk(defs[name]) if isinstance(n, ast.Name) and n.id in defs)
    return {
        f"{module.__name__}.{name}": normalized_code(
            ast.get_source_segment(source, defs[name]) or ""
        )
        for name in sorted(seen)
    }


def hashed_sources() -> dict[str, str]:
    """The code the revision hashes, normalised: this module whole; the matcher and the disclosure rules it runs, with every helper and constant
    of ``triage`` they reach; what the NER stage runs on a text, with the helpers of ``chunking`` it reaches. The closures are computed
    (``_closure``), so a function or a constant added to the matcher later is hashed without anyone remembering to list it; what they do not
    reach (the scope features of ``triage``, the training of the models) does not restart the step."""
    sources = {
        "news_company_stage": normalized_code(_module_source(sys.modules[__name__])),
        "news_company_stage._DISCLOSURE_WORDS": f"{_DISCLOSURE_WORDS.pattern}|{_DISCLOSURE_WORDS.flags}",
    }
    sources.update(_closure(triage, ("CompanyResolver", "in_disclosure", "disclosure_branch")))
    sources.update(_closure(ner_stage, ("merge_bio_predictions", "NerFeature", "NerInference")))
    sources.update(_closure(chunking, ("chunk_text", "merge_char_spans", "Chunk")))
    return sources


def revision_of(
    ner_identity: str,
    relevance_identity: str,
    threshold: float,
    universe: Sequence[Company],
    candidates: str = "ner",
) -> str:
    """The revision of a run of the step: the two models (``name@revision``), the threshold, the universe the matcher reads, the hypothesis,
    the length, the version of ``transformers`` and the code (``hashed_sources``)."""
    key = json.dumps(
        {
            "ner": ner_identity,
            "relevance": relevance_identity,
            "threshold": threshold,
            "candidates": candidates,
            "universe": hashlib.sha256(
                json.dumps(sorted(tuple(c) for c in universe)).encode()
            ).hexdigest()[:12],
            "hypothesis": HYPOTHESIS,
            "max_length": MAX_LENGTH,
            "entity": ENTITY_TYPE,
            "transformers": transformers.__version__,
            "code": {
                name: hashlib.sha256(src.encode()).hexdigest()[:12]
                for name, src in hashed_sources().items()
            },
        },
        sort_keys=True,
    )
    return "companies-" + hashlib.sha256(key.encode()).hexdigest()[:12]


class RelevanceScorer:
    """The published relevance model: ``score`` gives the probability that each text is mainly about its company."""

    def __init__(
        self,
        repo: str | None = None,
        revision: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        published = published_relevance()
        self.repo: str = repo or published["repo"]
        self.revision: str = revision or published["revision"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
            # float32: transformers 5 loads the checkpoint as fp16 by default, and the model was trained and scored in float32
            self._model = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.repo, revision=self.revision, dtype=torch.float32
                )
                .to(self.device)
                .eval()
            )

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """The relevance of each ``(text, company name)`` pair, in the order given (label 0 is entailment)."""
        self._load()
        scores: list[float] = []
        for start in range(0, len(pairs), RELEVANCE_BATCH):
            batch = pairs[start : start + RELEVANCE_BATCH]
            encoded = self._tokenizer(
                [text for text, _ in batch],
                [HYPOTHESIS.format(company_name(name)) for _, name in batch],
                truncation="only_first",
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            # bf16 autocast on CUDA only, as the model was trained and scored; the CPU runs plain float32
            with (
                torch.no_grad(),
                torch.autocast(
                    self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
                ),
            ):
                logits = self._model(**encoded).logits.float()
            scores.extend(torch.softmax(logits, dim=-1)[:, 0].cpu().tolist())
        return scores


class SpanFinder:
    """The pinned NER model of the pipeline (``ner_stage``'s own feature and inference, reused as they are): organisation spans of each text."""

    def __init__(self, device: torch.device | None = None) -> None:
        pinned = pinned_ner()
        self.model_name: str = pinned["model"]
        self.model_revision: str = pinned["revision"]
        self._inference = ner_stage.NerInference(
            ner_stage.NerFeature(),
            model_name=self.model_name,
            revision=self.model_revision,
            device=device,
            batch_size=int(pinned.get("batch_size", 8)),
        )
        self._loaded = False

    def spans(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        if not self._loaded:
            self._inference.load_model()
            self._loaded = True
        rows = list(enumerate(texts))
        features = self._inference.feature.extract_batch(self._inference.tokenizer, rows)
        per_text: list[list[dict[str, Any]]] = self._inference.predict_batch(features)
        return per_text


class Spanning(Protocol):
    """What the finder needs of the NER side (``SpanFinder`` or a test stub)."""

    model_name: str
    model_revision: str

    def spans(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]: ...


class Scoring(Protocol):
    """What the finder needs of the relevance side (``RelevanceScorer`` or a test stub)."""

    repo: str
    revision: str
    device: Any

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class CompanyFinder:
    """What the companies step runs: NER, the matcher, the relevance model. ``find`` gives, for each text, a ``CompanyScore`` per ticker named in it."""

    def __init__(
        self,
        universe: Sequence[Company],
        threshold: float,
        candidates: str = "ner",
        spans: Spanning | None = None,
        scorer: Scoring | None = None,
    ) -> None:
        if candidates not in CANDIDATES:
            raise ValueError(f"candidates must be one of {CANDIDATES}, not {candidates!r}")
        self.threshold = threshold
        self.candidates = candidates
        self.resolver = CompanyResolver(list(universe))
        self.names: dict[str, str] = {c.ticker: c.name for c in universe}
        self.spans = spans or SpanFinder()
        self.scorer = scorer or RelevanceScorer()
        self.revision = revision_of(
            f"{self.spans.model_name}@{self.spans.model_revision}",
            f"{self.scorer.repo}@{self.scorer.revision}",
            threshold,
            universe,
            candidates,
        )

    def find(self, texts: Sequence[str]) -> list[list[CompanyScore]]:
        with_text = self.candidates == "ner+text"
        mentions = [
            mentions_of(self.resolver, spans, text if with_text else None)
            for text, spans in zip(texts, self.spans.spans(list(texts)), strict=True)
        ]
        pairs: list[tuple[str, str]] = []
        for text, counts in zip(texts, mentions, strict=True):
            pairs.extend((text, self.names.get(ticker, ticker)) for ticker in counts)
        scores = iter(self.scorer.score(pairs))
        return [
            [CompanyScore(ticker, count, next(scores)) for ticker, count in counts.items()]
            for counts in mentions
        ]
