"""Step 2, the summary: single bodies and sub-news over a size are summarised, the rest is used as it is (PLAN D39 (2), (3.3), D40; T-164, T-166).

The size is the owner's (2026-11-12): **2,000 characters** of the text the summariser would read. A text over it goes through the pinned
local summariser (``config/pipeline_models.json`` ``c_summary``: ``sshleifer/distilbart-cnn-12-6`` at a fixed revision) with the chunk-and-reduce
of ``summary_stage.hierarchical_summarize_batch`` (the v1 stage's own, reused as it is); a text at or under it is stored unchanged
(``text_source = original``), so every item has one text for the steps after this one to read. The measure behind the size is in
``docs/summary_step.md``.

The input of a single body is ``Title-<title>\\nBody-<body>``, as the earlier summary run wrote it (``scripts/cut_summaries_2026_11_07.py``,
``draw_sentiment_gold``); a sub-news has no title of its own (the title is the whole round-up's), so its text goes alone.

The revision of the step is ``summary-`` plus 12 hex digits of the SHA-256 of the model and its pinned revision, the ``transformers`` version, the size, the
input format, the generation limits and the code of this module and of the functions it runs in ``summary_stage.py`` and ``chunking.py`` (``hashed_sources``, as parsed
code: comments, formatting and docstrings do not count): any of them changing starts new rows and leaves the old ones.

A summariser is cheap to build: the model is loaded on the first text it is given, so ``--status`` and the tests need none. It runs on CUDA when
there is one and on the CPU otherwise (constitution, Pipeline models 2).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import chunking
import summary_stage
from summary_stage import (
    SUMMARY_MAX_INPUT_TOKENS,
    SUMMARY_MAX_OUTPUT_TOKENS,
    SUMMARY_MIN_OUTPUT_TOKENS,
    hierarchical_summarize_batch,
)

MIN_CHARS = 2000  # owner, 2026-11-12: 64% of the 93,034 single bodies, 1.6% of the sub-news of the first 500 bodies
INPUT_FORMAT = "Title-{title}\nBody-{body}"
_CONFIG = Path(__file__).resolve().parent.parent / "config" / "pipeline_models.json"


def pinned_summariser() -> dict[str, Any]:
    """The pinned summariser of ``config/pipeline_models.json`` (``model``, ``revision``, ``batch_size``)."""
    config: dict[str, Any] = json.loads(_CONFIG.read_text(encoding="utf-8"))["c_summary"]
    return config


def needs_summary(text: str, min_chars: int = MIN_CHARS) -> bool:
    """Whether a text is over the size: the size itself is used as it is."""
    return len(text) > min_chars


def input_of_body(title: str, text: str) -> str:
    """What the summariser reads of a single body: its title and its text."""
    return INPUT_FORMAT.format(title=title, body=text)


#: What the step runs of the shared v1 modules. ``summary_stage.py`` also holds the v1 stage and ``chunking.py`` the NER span merge: an edit to those
#: parts must not change the revision, or ``--step summary`` would redo hours of GPU work for nothing.
_SUMMARY_STAGE_FUNCTIONS = (
    "hierarchical_summarize_batch",
    "_leaf_summarize_batch",
    "_reduce_pass",
    "_summarize_in_batches",
    "_summarize_batch",
)
_CHUNKING_FUNCTIONS = ("chunk_text", "split_sentences")


def _read_module_source() -> str:
    return Path(__file__).read_text(encoding="utf-8")


def normalized_code(source: str) -> str:
    """The code of a source without what does not run: the parse drops comments and formatting (spaces, line breaks, the kind of quotes) and the
    docstrings are removed, so a typo fix in a docstring, a comment or a formatter upgrade does not change what is hashed; a change of a name, a
    constant or an operator does."""
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)  # without positions: the line and column of a node are not in it


def hashed_sources() -> dict[str, str]:
    """The code the revision of the step hashes, normalised (``normalized_code``): this module whole, and the functions it runs in ``summary_stage``
    and ``chunking`` (``Chunk``, the sentence regex and ``MAX_REDUCE_PASSES`` with them)."""
    sources = {"news_summary_stage": normalized_code(_read_module_source())}
    for module, names in (
        (summary_stage, _SUMMARY_STAGE_FUNCTIONS),
        (chunking, _CHUNKING_FUNCTIONS),
    ):
        for name in names:
            sources[f"{module.__name__}.{name}"] = normalized_code(
                inspect.getsource(getattr(module, name))
            )
    sources["chunking.Chunk"] = normalized_code(inspect.getsource(chunking.Chunk))
    # the one constant the sentence split depends on
    sources["chunking.sentence_regex"] = chunking._SENTENCE_SPLIT_RE.pattern
    sources["summary_stage.MAX_REDUCE_PASSES"] = str(summary_stage.MAX_REDUCE_PASSES)
    return sources


def revision_of(model_name: str, model_revision: str, min_chars: int = MIN_CHARS) -> str:
    """The revision of a summary step: the model, the size, the input, the generation limits and the code (``hashed_sources``)."""
    key = json.dumps(
        {
            "model": f"{model_name}@{model_revision}",
            # beam search lives in transformers and pyproject only requires >=4.56: a lock bump must not leave old and new summaries under one
            # revision. torch is left out: its version carries the build (+cu124, +cpu) and the revision is the same on both devices
            "transformers": transformers.__version__,
            "min_chars": min_chars,
            "input": INPUT_FORMAT,
            "max_in": SUMMARY_MAX_INPUT_TOKENS,
            "out": [SUMMARY_MIN_OUTPUT_TOKENS, SUMMARY_MAX_OUTPUT_TOKENS],
            "code": {
                name: hashlib.sha256(src.encode()).hexdigest()[:12]
                for name, src in hashed_sources().items()
            },
        },
        sort_keys=True,
    )
    return "summary-" + hashlib.sha256(key.encode()).hexdigest()[:12]


class Summarizer:
    """The pinned local summariser for the summary step: ``summarize`` returns ``(summary, chunks)`` for each text it is given."""

    def __init__(
        self,
        min_chars: int = MIN_CHARS,
        model_name: str | None = None,
        model_revision: str | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        pinned = pinned_summariser()
        self.min_chars = min_chars
        self.model_name: str = model_name or pinned["model"]
        self.model_revision: str = model_revision or pinned["revision"]
        self.batch_size: int = batch_size or int(pinned.get("batch_size", 4))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.revision = revision_of(self.model_name, self.model_revision, min_chars)
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.model_revision
            )
            self._model = (
                AutoModelForSeq2SeqLM.from_pretrained(self.model_name, revision=self.model_revision)
                .to(self.device)
                .eval()
            )

    def summarize(self, texts: Sequence[str]) -> list[tuple[str, int]]:
        """Summarise each text (chunk, summarise, reduce until one remains); ``(summary, chunks)`` in the order given."""
        self._load()
        return hierarchical_summarize_batch(
            list(texts),
            self._tokenizer,
            self._model,
            self.device,
            max_input_tokens=SUMMARY_MAX_INPUT_TOKENS,
            batch_size=self.batch_size,
        )
