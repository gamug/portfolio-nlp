"""How the sentiment model reads a text for a company (T-148c, D35 as amended)."""

from __future__ import annotations

from typing import Any

import pytest

import sentiment_pair as pair


class _Tok:
    """A whitespace tokenizer: one token per word, [CLS]/[SEP] added for a pair."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))

    def __call__(self, a: str, b: str, **kwargs: Any) -> dict[str, Any]:
        return {"a": a, "b": b, **kwargs}


def test_the_company_segment_collapses_white_space_and_is_cut() -> None:
    assert pair.company_segment("  Quanta \n Services  ") == "Quanta Services"
    assert len(pair.company_segment("x" * 500)) == pair.MAX_COMPANY_CHARS


def test_the_chunk_budget_is_the_window_less_the_special_tokens_and_the_company() -> None:
    tok = _Tok()
    assert pair.chunk_budget(tok, "Quanta Services") == 512 - 3 - 2
    assert pair.chunk_budget(tok, "Tesla, Inc.", max_length=300) == 300 - 3 - 2


def test_a_company_that_leaves_too_little_room_is_refused() -> None:
    with pytest.raises(ValueError, match="only"):
        pair.chunk_budget(_Tok(), "a b c d", max_length=pair.MIN_CHUNK_TOKENS + 3)


def test_the_pair_truncates_only_the_chunk() -> None:
    out = pair.encode_pair(_Tok(), " Tesla,  Inc. ", "the chunk", return_tensors="pt")
    assert out["a"] == "Tesla, Inc." and out["b"] == "the chunk"
    assert out["truncation"] == "only_second" and out["max_length"] == 512
    assert out["return_tensors"] == "pt"
