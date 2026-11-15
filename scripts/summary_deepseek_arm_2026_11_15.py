"""Arm D of the summary experiment: DeepSeek as the summariser, on the 300 articles of summary_lead_experiment_2026_11_13.py (T-206).

    uv run scripts/summary_deepseek_arm_2026_11_15.py generate [--max-calls 300]  # the API; writes deepseek.json (the summaries and the tokens), nothing else
    uv run scripts/summary_deepseek_arm_2026_11_15.py attach [--force]            # adds arm D to arms.json (a backup first), so the judge can read it
    uv run scripts/summary_lead_experiment_2026_11_13.py judge                    # the same judge and cache as the other arms
    uv run scripts/summary_deepseek_report_2026_11_15.py export | tables          # the committed figures, no text

Scratch, like the experiment it extends. Same input as the step (``Title-...\\nBody-...``), temperature 0. The key and the endpoint are ``LLM_API_KEY`` and
``LLM_URL`` of ``.env``. Errors are handled in the way the judge of step 2 does: a rate limit or a server error is retried with a growing wait, a
wrong key stops at once, ``MAX_CONSECUTIVE_FAILURES`` articles in a row that could not be summarised stop the run (nothing is written for them), and
``--max-calls`` is a hard cap on the requests made, retries included.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
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
MAX_ATTEMPTS = 4
MAX_CONSECUTIVE_FAILURES = 5
RETRYABLE = {408, 429, 500, 502, 503, 504}
HTTP_UNAUTHORIZED = 401, 403


class ApiError(RuntimeError):
    """A request that failed for good: the status and the reason, never an empty summary."""


class CallBudget:
    """The requests made so far, with a hard cap (retries included)."""

    def __init__(self, max_calls: int | None) -> None:
        self.max_calls = max_calls
        self.calls = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                return False
            self.calls += 1
            return True


def call(title: str, body: str, budget: CallBudget, sleep: float = 2.0) -> tuple[str, int, int]:
    """One summary and the tokens it used. Retries a rate limit, a server error or a dropped connection with a growing wait; a wrong key or a refusal is not retried."""
    url = os.environ["LLM_URL"].rstrip("/") + "/chat/completions"
    payload = {
        "model": os.environ["LLM_MODEL"],
        "temperature": 0,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Title-{title}\nBody-{body[:CAP]}"},
        ],
    }
    last = "no attempt"
    for attempt in range(MAX_ATTEMPTS):
        if not budget.take():
            raise ApiError("the cap of calls was reached")
        req = urllib.request.Request(  # noqa: S310 -- the owner's configured endpoint
            url,
            json.dumps(payload).encode(),
            {
                "Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                d = json.load(r)
            text = d["choices"][0]["message"]["content"].strip()
            if not text:
                raise ApiError("the reply was empty")
            return text, d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code in HTTP_UNAUTHORIZED:
                raise ApiError(f"{last}: the key or the endpoint is wrong") from exc
            if exc.code not in RETRYABLE:
                raise ApiError(last) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(sleep * 2**attempt)
    raise ApiError(f"{MAX_ATTEMPTS} attempts failed (the last: {last})")


def generate(max_calls: int | None) -> None:
    import news_nlp as db  # noqa: PLC0415

    db.env.load_env_file()
    sample = json.loads((OUT / "sample.json").read_text(encoding="utf-8"))
    path = OUT / "deepseek.json"
    done = (
        {r["article_id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
        if path.exists()
        else {}
    )
    todo = [r for r in sample if r["article_id"] not in done]
    budget = CallBudget(max_calls)
    stop = threading.Event()
    streak = {"n": 0, "last": ""}
    lock = threading.Lock()

    def one(r: dict[str, object]) -> None:
        if stop.is_set():
            return
        try:
            summary, tin, tout = call(str(r["title"]), str(r["body"]), budget)
        except ApiError as exc:
            with lock:
                streak["n"] += 1
                streak["last"] = str(exc)
                if streak["n"] >= MAX_CONSECUTIVE_FAILURES or "cap of calls" in str(exc):
                    stop.set()
            return
        with lock:
            streak["n"] = 0
            done[int(str(r["article_id"]))] = {
                "article_id": r["article_id"],
                "summary": summary,
                "in": tin,
                "out": tout,
            }

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, todo))
    path.write_text(
        json.dumps(list(done.values())), encoding="utf-8"
    )  # what arrived is kept; a rerun goes on
    print(
        f"{len(done)} of {len(sample)} summaries; {budget.calls} requests; input tokens {sum(r['in'] for r in done.values())}, output {sum(r['out'] for r in done.values())}"
    )
    if len(done) < len(sample):
        sys.exit(
            f"stopped with {len(sample) - len(done)} articles left (the last error: {streak['last'] or 'none'}); nothing was written for them, rerun to go on"
        )


def attach(force: bool) -> None:
    arms_path = OUT / "arms.json"
    arms = json.loads(arms_path.read_text(encoding="utf-8"))
    ds = {
        r["article_id"]: r["summary"]
        for r in json.loads((OUT / "deepseek.json").read_text(encoding="utf-8"))
    }
    if any("D" in a["arms"] for a in arms) and not force:
        sys.exit("arms.json already has arm D; use --force to replace it")
    missing = [a["article_id"] for a in arms if a["article_id"] not in ds]
    if missing:
        sys.exit(f"{len(missing)} articles have no DeepSeek summary: run generate again")
    backup = OUT / "arms_before_D.json"
    if not backup.exists():
        backup.write_text(arms_path.read_text(encoding="utf-8"), encoding="utf-8")
    for a in arms:
        a["arms"]["D"] = ds[a["article_id"]]
        a["arm_chunks"]["D"] = a["arm_chunks"].get("B", 1)
    arms_path.write_text(json.dumps(arms, ensure_ascii=False), encoding="utf-8")
    print(f"arm D attached to {len(arms)} articles (backup: {backup})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["generate", "attach"])
    ap.add_argument(
        "--max-calls",
        type=int,
        default=400,
        help="a hard cap on the requests of generate, retries included (300 articles)",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args.max_calls)
    else:
        attach(args.force)


if __name__ == "__main__":
    main()
