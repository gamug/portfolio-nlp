"""The owner's re-tag of the sentiment held-out (2026-11-20, version 2): the tracked gold, the eleven tags that changed against the
2026-11-16 version, and the scorer's pooled figures. The 2026-11-16 version stays beside it, unchanged (test_val_gold_2026_11_16)."""

from __future__ import annotations

import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "gold"
V2 = GOLD / "sentiment_val190_2026_11_20_v2.csv"
V1 = GOLD / "sentiment_val190.csv"

CHANGED = {
    "sn297601#TMUS": ("positive", "neutral"),
    "sn299893#PSX": ("negative", "neutral"),
    "sn276024#AMZN": ("positive", "neutral"),
    "sn371456#KEY": ("positive", "neutral"),
    "sn356727#WMT": ("positive", "neutral"),
    "sn299671#GS": ("positive", "neutral"),
    "sn252880#VZ": ("negative", "neutral"),
    "sn373887#CMCSA": ("neutral", "positive"),
    "sn315680#CMG": ("neutral", "positive"),
    "sn402530#AVGO": ("neutral", "negative"),
    "sn276855#AMZN": ("neutral", "positive"),
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows: list[dict[str, str]] = list(csv.DictReader(fh))
    return rows


def test_the_v2_tags_are_the_owner_s_re_tag() -> None:
    v1 = {r["key"]: r["sentiment"] for r in _read(V1)}
    v2 = _read(V2)
    assert len(v2) == 190
    assert {r["sentiment"] for r in v2} == {"positive", "negative", "neutral"}
    assert Counter(r["sentiment"] for r in v2) == {"positive": 94, "negative": 61, "neutral": 35}
    changed = {
        r["key"]: (v1[r["key"]], r["sentiment"]) for r in v2 if r["sentiment"] != v1[r["key"]]
    }
    assert changed == CHANGED


def test_the_scorer_reads_v2_and_prints_the_pooled_figures() -> None:
    done = subprocess.run(  # noqa: S603 -- runs this repo's own scorer with this interpreter
        [sys.executable, str(ROOT / "scripts" / "score_sentiment_val190_2026_11_16.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    out = done.stdout
    assert "sentiment_val190_2026_11_20_v2.csv" in out
    assert "accuracy 150/190 = 78.9%" in out
    assert "neutral: P 80.0% [62.7, 90.5] (24/30), R 68.6% [52.0, 81.4] (24/35)" in out
