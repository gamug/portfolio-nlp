"""How the sentiment model reads a text **for a company** (Work item 18, T-148c, D35 as amended).

The company-text sentiment model takes a pair: segment A is the company's name and segment B is a
chunk of the text, ``[CLS] company [SEP] chunk [SEP]``, so the label is about that company and not
about whatever the text is about. The trainer, the scorer of the gold and the stage read a text
through the helpers here, so the three cannot drift: the one place the rule is written.

The company name is cut to :data:`MAX_COMPANY_CHARS` characters (a long legal name would take the
chunk's room); the chunk budget is what is left of the model's window after the special tokens and
the company's tokens. A text with no company (a sector, a market or an economy text) is not read as
a pair: it goes to the text-only model (the routing is the stage's, PLAN D35).
"""

from __future__ import annotations

from typing import Any

#: The model's window (BERT) and what a pair spends outside the two segments: [CLS], [SEP], [SEP].
MAX_LENGTH = 512
SPECIAL_TOKENS = 3
#: The longest company name read, in characters.
MAX_COMPANY_CHARS = 80
#: A chunk is never given less than this many tokens, whatever the company name takes.
MIN_CHUNK_TOKENS = 128


def company_segment(company: str) -> str:
    """Segment A: the company's name, white space collapsed and cut to ``MAX_COMPANY_CHARS``."""
    return " ".join(company.split())[:MAX_COMPANY_CHARS].strip()


def chunk_budget(tokenizer: Any, company: str, max_length: int = MAX_LENGTH) -> int:
    """Tokens left for the chunk (segment B) once the company and the special tokens are in."""
    used = len(tokenizer.encode(company_segment(company), add_special_tokens=False))
    budget = max_length - SPECIAL_TOKENS - used
    if budget < MIN_CHUNK_TOKENS:
        raise ValueError(
            f"the company {company!r} takes {used} tokens: only {budget} are left for the text "
            f"(at least {MIN_CHUNK_TOKENS} are needed)"
        )
    return budget


def encode_pair(
    tokenizer: Any, company: str, chunk: str, max_length: int = MAX_LENGTH, **kwargs: Any
) -> Any:
    """The tokenizer's output for ``(company, chunk)``; only the chunk is ever truncated."""
    return tokenizer(
        company_segment(company),
        chunk,
        truncation="only_second",
        max_length=max_length,
        **kwargs,
    )
