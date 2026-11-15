# ruff: noqa
"""Arm D of the summary experiment: DeepSeek as the summariser, on the 300 articles of summary_lead_experiment_2026_11_13.

    uv run scripts/summary_deepseek_arm_2026_11_15.py generate   # calls the API, records the token usage, adds arm D to arms.json
    uv run scripts/summary_lead_experiment_2026_11_13.py judge   # the same judge and cache as the other arms
    uv run scripts/summary_deepseek_arm_2026_11_15.py report     # D against B (paired, 95% bootstrap) and the cost

Scratch, like the experiment it extends. Same input as the step (``Title-...\\nBody-...``), temperature 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
OUT = Path(__file__).resolve().parents[1] / "experiments" / "results" / "summary_lead_2026_11_13"
PROMPT = (
    "Summarise the news article below in about 60 words (2 to 4 sentences). Use only facts stated in the article; "
    "do not add anything from outside it. Name the companies and people the article is about, and keep who did what "
    "right. Reply with the summary only."
)
CAP = 100_000


def sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]  # noqa: S324 -- a cache key


def call(title: str, body: str) -> tuple[str, int, int]:
    url = os.environ["LLM_URL"].rstrip("/") + "/chat/completions"
    payload = {
        "model": os.environ["LLM_MODEL"],
        "temperature": 0,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Title-{title}\nBody-{body[:CAP]}"},
        ],
    }
    req = urllib.request.Request(  # noqa: S310 -- the owner's configured endpoint
        url,
        json.dumps(payload).encode(),
        {
            "Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                d = json.load(r)
            u = d["usage"]
            return (
                d["choices"][0]["message"]["content"].strip(),
                u["prompt_tokens"],
                u["completion_tokens"],
            )
        except Exception:
            if attempt == 3:
                raise
    raise RuntimeError


def generate() -> None:
    import news_nlp as db

    db.env.load_env_file()
    sample = json.loads((OUT / "sample.json").read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda r: call(r["title"], r["body"]), sample))
    (OUT / "deepseek.json").write_text(
        json.dumps(
            [
                {"article_id": r["article_id"], "summary": s, "in": i, "out": o}
                for r, (s, i, o) in zip(sample, res)
            ]
        ),
        encoding="utf-8",
    )
    arms = json.loads((OUT / "arms.json").read_text(encoding="utf-8"))
    by = {r["article_id"]: s for r, (s, _, _) in zip(sample, res)}
    for a in arms:
        a["arms"]["D"] = by[a["article_id"]]
        a["arm_chunks"]["D"] = a["arm_chunks"].get("B", 1)
    (OUT / "arms.json").write_text(json.dumps(arms, ensure_ascii=False), encoding="utf-8")
    print(
        f"{len(res)} summaries; input tokens {sum(x[1] for x in res)}, output {sum(x[2] for x in res)}"
    )


def boot(d: list[float]) -> tuple[float, float, float]:
    rng = random.Random(0)
    n = len(d)
    ms = sorted(sum(rng.choices(d, k=n)) / n for _ in range(4000))
    return sum(d) / n, ms[100], ms[3900]


def report() -> None:
    arms = json.loads((OUT / "arms.json").read_text(encoding="utf-8"))
    ds = {
        r["article_id"]: r for r in json.loads((OUT / "deepseek.json").read_text(encoding="utf-8"))
    }
    ver = {}
    for ln in (OUT / "verdicts.jsonl").read_text().splitlines():
        r = json.loads(ln)
        ver[(r["article_id"], r["sha"])] = r["verdict"]
    rows = []
    for a in arms:
        b, d = (
            ver.get((a["article_id"], sha(a["arms"]["B"]))),
            ver.get((a["article_id"], sha(a["arms"]["D"]))),
        )
        if b and d:
            rows.append((a, b, d))
    print(f"paired articles: {len(rows)}")
    for name, k in (("B", 1), ("D", 2)):
        vs = [r[k] for r in rows]
        print(
            name,
            {
                m: round(sum(v[m] for v in vs) / len(vs), 2)
                for m in ("faithfulness", "coverage", "conciseness")
            },
            f"halluc {sum(1 for v in vs if v['hallucinations']) / len(vs):.1%}",
        )
    for m in ("faithfulness", "coverage", "conciseness"):
        print("D-B", m, "%.3f [%.3f, %.3f]" % boot([r[2][m] - r[1][m] for r in rows]))
    print(
        "D-B halluc pts",
        "%.1f [%.1f, %.1f]"
        % tuple(
            100 * x
            for x in boot(
                [
                    float(bool(r[2]["hallucinations"])) - float(bool(r[1]["hallucinations"]))
                    for r in rows
                ]
            )
        ),
    )
    for t in (1, 2, 3):
        sub = [r for r in rows if r[0]["tier"] == t]
        print(
            f"tier {t}: B faith {sum(r[1]['faithfulness'] for r in sub) / len(sub):.2f} cov {sum(r[1]['coverage'] for r in sub) / len(sub):.2f} "
            f"halluc {sum(bool(r[1]['hallucinations']) for r in sub) / len(sub):.0%} | D faith {sum(r[2]['faithfulness'] for r in sub) / len(sub):.2f} "
            f"cov {sum(r[2]['coverage'] for r in sub) / len(sub):.2f} halluc {sum(bool(r[2]['hallucinations']) for r in sub) / len(sub):.0%}"
        )
    i, o = sum(r["in"] for r in ds.values()), sum(r["out"] for r in ds.values())
    n = len(ds)
    print(
        f"tokens per article: in {i / n:.0f}, out {o / n:.0f}; summary words mean {sum(len(r['summary'].split()) for r in ds.values()) / n:.0f}"
    )
    pin, pout = 0.28, 0.42
    per = (i / n * pin + o / n * pout) / 1e6
    print(
        f"cost of 64,480 items at ${pin}/${pout} per M: ${per * 64480:.2f} (this sample's mean article length, not the corpus's)"
    )


if __name__ == "__main__":
    {"generate": generate, "report": report}[sys.argv[1]]()
