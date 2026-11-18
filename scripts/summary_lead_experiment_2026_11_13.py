# ruff: noqa: C901, PLC0415, PLR0912, PLR0915, PLR2004, S311, SIM102
"""Summary experiment, 2026-11-13: lead-weighted chunks, no reduce pass, an NLI gate.

Question: does summarising the chunks of a long body once (no summary-of-summaries) and joining them beat
the production chunk-and-reduce (``summary_stage.hierarchical_summarize_batch``) on the same articles, judged by the
same LLM judge and prompt as ``eval_run`` 44 (``c_summary_production``, seed 1)?

Sample: 300 of the 1,000 articles judged in ``eval_run`` 44, 100 for each of ``num_chunks`` 1, 2 and >=3 (seeded).
Everything is read-only on the RESULTS and SOURCE databases; output goes to ``experiments/results/summary_lead_2026_11_13/``.
``summary_stage.py`` is not touched (its code is hashed into the summary step's revision).

Arms (the same 300 articles each):
  A  the stored v1 summary of run 44 (judged again now: the judge's own drift is measured against its stored verdicts)
  B  the current step as it is: ``hierarchical_summarize_batch`` on ``Title-<t>\\nBody-<b>`` (like for like with C*)
  C0 lead only: the summary of chunk 1
  C1 chunk 1 at full length (56-142 tokens) + shorter summaries of the next chunks (12-60), once, joined
  C2 C1 + the NLI gate (a sentence is kept if some window of its own chunk entails it; chunk 1 falls back to its lead sentences)
  C3 chunk 1 abstractive (gated) + the lead sentences of the next chunks, copied
In C1-C3: balanced chunks, every chunk carries ``Title-<t>`` and the join is capped at ``TOTAL_BUDGET`` tokens, near-duplicate sentences dropped.

  uv run scripts/summary_lead_experiment_2026_11_13.py sample|generate|nli|assemble|judge|report
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sqlite3
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from chunking import chunk_text, split_sentences  # noqa: E402

RESULTS_DB = "/workspaces/thesis/data/nlp.db"
SOURCE_DB = "/workspaces/thesis/data/urls.db"
BASELINE_RUN = 44
PER_TIER = 100
SEED = 1
OUT = ROOT / "experiments" / "results" / "summary_lead_2026_11_13"
ARMS = ("A", "B", "C0", "C1", "C2", "C3")

MAX_IN = 1000  # as production
LEAD_LEN = (56, 142)  # as production, for chunk 1
TAIL_LEN = (
    12,
    60,
)  # the next chunks: shorter, and a low floor (a high floor makes the model pad a short chunk)
MAX_TAILS = 8  # chunks after the first that are read: the budget below cuts the rest anyway
TOTAL_BUDGET = 450  # tokens of the joined summary
NLI_THRESHOLD = 0.5  # chosen before seeing any score
NLI_WINDOW = 360  # deberta tokens of premise per window (its input is 512 with the hypothesis)
DUP_JACCARD = 0.6
JUDGE_BODY_CAP = 100_000  # news_nlp.eval.sampling._SENTIMENT_MAX_BODY_CHARS: the judge saw the same view in run 44


def tier_of(n_chunks: int) -> int:
    return min(n_chunks, 3)


def tok_len(tok: Any, s: str) -> int:
    return len(tok.encode(s, add_special_tokens=False))


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pinned(key: str) -> dict[str, Any]:
    cfg: dict[str, Any] = json.loads((ROOT / "config" / "pipeline_models.json").read_text())
    return cfg[key]


# ---------------------------------------------------------------- sample


def cmd_sample() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = sqlite3.connect(f"file:{RESULTS_DB}?mode=ro", uri=True, timeout=10)
    src = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True, timeout=10)
    stored = {
        a: json.loads(v)
        for a, v in res.execute(
            "SELECT article_id, verdict_json FROM eval_verdict WHERE run_id=?", (BASELINE_RUN,)
        )
    }
    by_tier: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for aid, pred in res.execute(
        "SELECT article_id, prediction_json FROM eval_inference WHERE run_id=? ORDER BY article_id",
        (BASELINE_RUN,),
    ):
        p = json.loads(pred)
        by_tier[tier_of(p["num_chunks"])].append((aid, p))
    rng = random.Random(SEED)
    rows = []
    for tier in (1, 2, 3):
        for aid, p in rng.sample(by_tier[tier], PER_TIER):
            title, body, lang = src.execute(
                "SELECT title, body_text, language FROM articles WHERE id=?", (aid,)
            ).fetchone()
            rows.append(
                {
                    "article_id": aid,
                    "tier": tier,
                    "title": title or "",
                    "body": body,
                    "language": lang,
                    "stored_summary": p["summary_text"],
                    "stored_chunks": p["num_chunks"],
                    "stored_verdict": stored.get(aid),
                }
            )
    (OUT / "sample.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    langs: dict[Any, int] = defaultdict(int)
    for r in rows:
        langs[r["language"]] += 1
    print(f"sample: {len(rows)} articles, per tier {PER_TIER}; languages {dict(langs)}")


def load_sample() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads((OUT / "sample.json").read_text(encoding="utf-8"))
    return data


# ---------------------------------------------------------------- generation


def balanced_chunks(body: str, tok: Any, budget: int) -> list[str]:
    """Chunks of near-equal size on sentence boundaries: no tiny tail that the model would pad."""
    total = tok_len(tok, body)
    n = max(1, math.ceil(total / budget))
    cap = budget if n == 1 else min(budget, math.ceil(total / n) + 50)
    chunks = chunk_text(body, tok, max_tokens=cap)
    texts = [c.text for c in chunks]
    # a last piece under a quarter of the cap goes back into the one before it, when that fits
    while len(chunks) > 1 and tok_len(tok, chunks[-1].text) < cap // 4:
        merged = body[chunks[-2].start_char : chunks[-1].end_char]
        if tok_len(tok, merged) > budget:
            break
        chunks[-2:] = [type(chunks[-1])(merged, chunks[-2].start_char, chunks[-1].end_char)]
        texts = [c.text for c in chunks]
    return texts


def generate(
    texts: list[str], tok: Any, model: Any, dev: torch.device, lens: tuple[int, int], batch: int = 4
) -> list[str]:
    """The production call (``summary_stage._summarize_batch``) with the output lengths a parameter."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [""] * len(texts)
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        enc = tok(
            [texts[i] for i in idx],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to(dev)
        with torch.no_grad():
            ids = model.generate(**enc, max_length=lens[1], min_length=lens[0], num_beams=4)
        for i, t in zip(idx, tok.batch_decode(ids, skip_special_tokens=True), strict=True):
            out[i] = t.strip()
    return out


_ECHO = re.compile(r"^\s*(title\s*[-:]\s*)", re.IGNORECASE)


def strip_echo(summary: str, title: str) -> tuple[str, bool]:
    """A model that repeats the ``Title-...`` line it was given: drop it (and say so)."""
    s = _ECHO.sub("", summary)
    t = title.strip()
    echoed = s != summary
    if t and s.lower().startswith(t.lower()[: max(10, len(t) // 2)]):
        cut = s.lower().find(t.lower())
        s = s[cut + len(t) :] if cut == 0 else s
        echoed = True
    return s.lstrip(" .:-\n"), echoed


def cmd_generate() -> None:
    from summary_stage import hierarchical_summarize_batch

    sample = load_sample()
    pin = pinned("c_summary")
    tok = AutoTokenizer.from_pretrained(pin["model"], revision=pin["revision"])
    dev = device()
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(pin["model"], revision=pin["revision"]).to(dev).eval()
    )
    path = OUT / "gen.jsonl"
    done = {r["article_id"] for r in read_jsonl(path)}
    todo = [r for r in sample if r["article_id"] not in done]
    print(f"generate: {len(todo)} to do ({len(done)} done) on {dev}")
    for s in range(0, len(todo), 10):
        part = todo[s : s + 10]
        # B: production, as it is, on the input format of the current step
        b = hierarchical_summarize_batch(
            [f"Title-{r['title']}\nBody-{r['body']}" for r in part], tok, model, dev
        )
        # C: balanced chunks, each with its title line; chunk 1 long, the next ones short; once
        chunk_lists = []
        for r in part:
            head = f"Title-{r['title']}\nBody-"
            budget = MAX_IN - tok_len(tok, head) - 4
            chunk_lists.append((head, balanced_chunks(r["body"], tok, budget)))
        lead_in = [head + ch[0] for head, ch in chunk_lists]
        tail_in, owner = [], []
        for i, (head, ch) in enumerate(chunk_lists):
            for c in ch[1 : 1 + MAX_TAILS]:
                tail_in.append(head + c)
                owner.append(i)
        leads = generate(lead_in, tok, model, dev, LEAD_LEN)
        tails = generate(tail_in, tok, model, dev, TAIL_LEN) if tail_in else []
        per_tail: list[list[str]] = [[] for _ in part]
        for o, t in zip(owner, tails, strict=True):
            per_tail[o].append(t)
        for i, r in enumerate(part):
            echoes = 0
            lead, e = strip_echo(leads[i], r["title"])
            echoes += e
            tl = []
            for t in per_tail[i]:
                t2, e = strip_echo(t, r["title"])
                echoes += e
                tl.append(t2)
            append_jsonl(
                path,
                {
                    "article_id": r["article_id"],
                    "B": b[i][0],
                    "B_chunks": b[i][1],
                    "C_chunks": len(chunk_lists[i][1]),
                    "chunks": chunk_lists[i][1][: 1 + MAX_TAILS],
                    "lead": lead,
                    "tails": tl,
                    "echoes": echoes,
                },
            )
        print(f"  {min(s + 10, len(todo))}/{len(todo)}", flush=True)


# ---------------------------------------------------------------- NLI


def windows(sentences: list[str], nli_tok: Any) -> list[str]:
    """Windows of whole sentences of at most NLI_WINDOW tokens, each starting at the last sentence of the one before."""
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        n = tok_len(nli_tok, s)
        if cur and cur_len + n > NLI_WINDOW:
            out.append(" ".join(cur))
            cur = cur[-1:]
            cur_len = tok_len(nli_tok, cur[0])
            if cur_len + n > NLI_WINDOW:
                cur, cur_len = [], 0
        cur.append(s)
        cur_len += n
    if cur:
        out.append(" ".join(cur))
    return out


def cmd_nli() -> None:
    sample = {r["article_id"]: r for r in load_sample()}
    gen = read_jsonl(OUT / "gen.jsonl")
    pin = pinned(
        "category"
    )  # MoritzLaurer/deberta-v3-base-zeroshot-v2.0: entailment / not_entailment
    tok = AutoTokenizer.from_pretrained(pin["model"], revision=pin["revision"])
    dev = device()
    model = (
        AutoModelForSequenceClassification.from_pretrained(pin["model"], revision=pin["revision"])
        .to(dev)
        .eval()
    )
    ent = next(i for i, n in model.config.id2label.items() if n.lower() == "entailment")
    path = OUT / "nli.jsonl"
    done = {r["article_id"] for r in read_jsonl(path)}
    for g in gen:
        aid = g["article_id"]
        if aid in done:
            continue
        title = sample[aid]["title"]
        pairs: list[
            tuple[int, int, str, str]
        ] = []  # (block, sentence no, premise window, hypothesis)
        blocks = [g["lead"], *g["tails"]]
        for b, summ in enumerate(blocks):
            prem_sents = [s for s, _, _ in split_sentences(f"{title}. {g['chunks'][b]}")]
            wins = windows(prem_sents, tok)
            for k, (hyp, _, _) in enumerate(split_sentences(summ)):
                pairs.extend((b, k, w, hyp) for w in wins)
        probs: list[float] = []
        for s in range(0, len(pairs), 16):
            part = pairs[s : s + 16]
            enc = tok(
                [p[2] for p in part],
                [p[3] for p in part],
                return_tensors="pt",
                truncation="only_first",
                max_length=512,
                padding=True,
            ).to(dev)
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits, dim=-1)[:, ent]
            probs.extend(p.tolist())
        best: dict[tuple[int, int], tuple[str, float]] = {}
        for (b, k, _, hyp), p in zip(pairs, probs, strict=True):
            if (b, k) not in best or p > best[(b, k)][1]:
                best[(b, k)] = (hyp, p)
        append_jsonl(
            path,
            {
                "article_id": aid,
                "sentences": [
                    {"block": b, "no": k, "text": h, "entail": p}
                    for (b, k), (h, p) in sorted(best.items())
                ],
            },
        )
    print(f"nli: {len(gen)} articles scored")


# ---------------------------------------------------------------- assemble


def words(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def join_blocks(blocks: list[list[str]], tok: Any) -> str:
    """Sentences of the blocks in order, near-duplicates dropped, stopping at TOTAL_BUDGET tokens; blocks on their own lines."""
    kept_w: list[set[str]] = []
    used = 0
    lines: list[str] = []
    for blk in blocks:
        line: list[str] = []
        for s in blk:
            w = words(s)
            if not w or any(len(w & k) / len(w | k) >= DUP_JACCARD for k in kept_w):
                continue
            n = tok_len(tok, s)
            if lines or line:  # the first sentence is always kept
                if used + n > TOTAL_BUDGET:
                    break
            line.append(s)
            kept_w.append(w)
            used += n
        if line:
            lines.append(" ".join(line))
    return "\n".join(lines)


def cmd_assemble() -> None:
    sample = {r["article_id"]: r for r in load_sample()}
    pin = pinned("c_summary")
    tok = AutoTokenizer.from_pretrained(pin["model"], revision=pin["revision"])
    nli = {r["article_id"]: r["sentences"] for r in read_jsonl(OUT / "nli.jsonl")}
    out = []
    for g in read_jsonl(OUT / "gen.jsonl"):
        aid = g["article_id"]
        r = sample[aid]
        sc = nli[aid]
        by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for s in sc:
            by_block[s["block"]].append(s)
        n_blocks = 1 + len(g["tails"])
        raw = [[s["text"] for s in by_block[b]] for b in range(n_blocks)]
        gated = [
            [s["text"] for s in by_block[b] if s["entail"] >= NLI_THRESHOLD]
            for b in range(n_blocks)
        ]
        lead_sents = [s for s, _, _ in split_sentences(g["chunks"][0])][:2]
        fallback = not gated[0]
        if fallback:
            gated[0] = lead_sents
        extract = [[s for s, _, _ in split_sentences(c)][:2] for c in g["chunks"][1:]]
        arms = {
            "A": r["stored_summary"],
            "B": g["B"],
            "C0": join_blocks([raw[0]], tok),
            "C1": join_blocks(raw, tok),
            "C2": join_blocks(gated, tok),
            "C3": join_blocks([gated[0], *extract], tok),
        }
        n_sent = len(sc)
        n_drop = sum(s["entail"] < NLI_THRESHOLD for s in sc)
        out.append(
            {
                "article_id": aid,
                "tier": r["tier"],
                "arms": arms,
                "arm_chunks": {"A": r["stored_chunks"], "B": g["B_chunks"]}
                | {a: g["C_chunks"] for a in ("C0", "C1", "C2", "C3")},
                "tokens": {a: tok_len(tok, s) for a, s in arms.items()},
                "nli_sentences": n_sent,
                "nli_dropped": n_drop,
                "lead_fallback": fallback,
                "echoes": g["echoes"],
            }
        )
    (OUT / "arms.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"assemble: {len(out)} articles, {len(ARMS)} arms")


# ---------------------------------------------------------------- judge


def sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]  # noqa: S324 -- a cache key, not security


def cmd_judge() -> None:
    from news_nlp.eval.config import EvalSettings
    from news_nlp.eval.item import EvalItem
    from news_nlp.eval.judges import judge_c_summary, load_prompt
    from news_nlp.eval.model import build_judge_agent, build_model

    settings = EvalSettings.load()
    model = build_model(settings)
    prompt = load_prompt("c_summary")  # the prompt of run 44
    sample = {r["article_id"]: r for r in load_sample()}
    path = OUT / "verdicts.jsonl"
    cache = {(r["article_id"], r["sha"]) for r in read_jsonl(path)}
    jobs = []
    for a in json.loads((OUT / "arms.json").read_text(encoding="utf-8")):
        for arm, summ in a["arms"].items():
            key = (a["article_id"], sha(summ))
            if summ.strip() and key not in cache:
                cache.add(key)  # identical summaries of one article are judged once
                jobs.append((a["article_id"], sha(summ), summ, a["arm_chunks"][arm]))
    print(f"judge: {len(jobs)} calls with {settings.max_workers} workers")

    def one(job: tuple[int, str, str, int]) -> dict[str, Any]:
        aid, h, summ, n = job
        r = sample[aid]
        body = r["body"]
        if len(body) > JUDGE_BODY_CAP:
            body = body[:JUDGE_BODY_CAP] + "\n[... truncated ...]"
        item = EvalItem(
            article_id=aid,
            bucket=f"tier_{r['tier']}",
            stratum_population=1,
            title=r["title"],
            body_text=body,
            prediction={"summary_text": summ, "num_chunks": n},
        )
        v = judge_c_summary(build_judge_agent(model, prompt), item)
        return {"article_id": aid, "sha": h, "verdict": v.model_dump()}

    with ThreadPoolExecutor(max_workers=settings.max_workers) as ex:
        for i, row in enumerate(ex.map(one, jobs), 1):
            append_jsonl(path, row)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)


# ---------------------------------------------------------------- report


def boot_ci(diffs: list[float], n: int = 4000) -> tuple[float, float, float]:
    rng = random.Random(0)
    m = sum(diffs) / len(diffs)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return m, means[int(0.025 * n)], means[int(0.975 * n)]


def cmd_report() -> None:
    arms_rows = json.loads((OUT / "arms.json").read_text(encoding="utf-8"))
    sample = {r["article_id"]: r for r in load_sample()}
    ver = {(r["article_id"], r["sha"]): r["verdict"] for r in read_jsonl(OUT / "verdicts.jsonl")}

    def v(a: dict[str, Any], arm: str) -> dict[str, Any] | None:
        return ver.get((a["article_id"], sha(a["arms"][arm])))

    def stats(rows: list[dict[str, Any]], arm: str) -> str:
        vs = [(a, v(a, arm)) for a in rows]
        ok = [(a, x) for a, x in vs if x and not x["parse_failed"]]
        if not ok:
            return "n=0"
        f = sum(x["faithfulness"] for _, x in ok) / len(ok)
        c = sum(x["coverage"] for _, x in ok) / len(ok)
        k = sum(x["conciseness"] for _, x in ok) / len(ok)
        h = sum(bool(x["hallucinations"]) for _, x in ok) / len(ok)
        t = sum(a["tokens"][arm] for a, _ in ok) / len(ok)
        return f"n={len(ok):3d}  faith {f:.2f}  cov {c:.2f}  conc {k:.2f}  halluc {h:5.1%}  tokens {t:5.0f}"

    for label, rows in [("ALL", arms_rows)] + [
        (
            f"tier {t} ({'1 chunk' if t == 1 else '2 chunks' if t == 2 else '3+ chunks'})",
            [a for a in arms_rows if a["tier"] == t],
        )
        for t in (1, 2, 3)
    ]:
        print(f"\n== {label}: {len(rows)} articles")
        for arm in ARMS:
            print(f"  {arm:3s} {stats(rows, arm)}")

    print("\n== paired differences against B (same articles; mean, 95% bootstrap CI)")
    for label, rows in [("ALL", arms_rows)] + [
        (f"tier {t}", [a for a in arms_rows if a["tier"] == t]) for t in (1, 2, 3)
    ]:
        for arm in ("C0", "C1", "C2", "C3"):
            for metric in ("faithfulness", "coverage", "halluc"):
                d = []
                for a in rows:
                    x, b = v(a, arm), v(a, "B")
                    if not x or not b or x["parse_failed"] or b["parse_failed"]:
                        continue
                    if metric == "halluc":
                        d.append(
                            float(bool(x["hallucinations"])) - float(bool(b["hallucinations"]))
                        )
                    else:
                        d.append(float(x[metric] - b[metric]))
                if d:
                    m, lo, hi = boot_ci(d)
                    print(
                        f"  {label:7s} {arm:3s} {metric:12s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  n={len(d)}"
                    )

    # judge drift: arm A judged now vs the stored verdicts of run 44
    pairs = []
    for a in arms_rows:
        old, new = sample[a["article_id"]]["stored_verdict"], v(a, "A")
        if old and new and not new["parse_failed"]:
            pairs.append((old, new))
    if pairs:
        print("\n== judge drift on arm A (stored run-44 verdict vs the same summary judged now)")
        for m in ("faithfulness", "coverage", "conciseness"):
            same = sum(o[m] == n[m] for o, n in pairs) / len(pairs)
            d = sum(n[m] - o[m] for o, n in pairs) / len(pairs)
            print(f"  {m:12s} same score {same:5.1%}  mean shift {d:+.3f}")
        print(
            f"  halluc flag agrees {sum(bool(o['hallucinations']) == bool(n['hallucinations']) for o, n in pairs) / len(pairs):5.1%}"
        )

    print("\n== NLI gate and mechanics")
    for t in (1, 2, 3):
        rows = [a for a in arms_rows if a["tier"] == t]
        s = sum(a["nli_sentences"] for a in rows)
        d = sum(a["nli_dropped"] for a in rows)
        print(
            f"  tier {t}: {d}/{s} summary sentences below {NLI_THRESHOLD} ({d / s:.1%}); "
            f"chunk-1 fell back to the lead: {sum(a['lead_fallback'] for a in rows)}; title echoes stripped: {sum(a['echoes'] for a in rows)}"
        )
    by_lang: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for a in arms_rows:
        by_lang[sample[a["article_id"]]["language"]].append(a)
    print("\n== by language (the model is English-only)")
    for lang, rows in sorted(by_lang.items(), key=lambda kv: -len(kv[1])):
        print(f" {lang}: {len(rows)} articles")
        for arm in ("B", "C2"):
            print(f"    {arm:3s} {stats(rows, arm)}")

    with (OUT / "summaries.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["article_id", "tier", "arm", "tokens", "faith", "cov", "conc", "n_halluc", "summary"]
        )
        for a in arms_rows:
            for arm in ARMS:
                x = v(a, arm)
                w.writerow(
                    [a["article_id"], a["tier"], arm, a["tokens"][arm]]
                    + (
                        [
                            x["faithfulness"],
                            x["coverage"],
                            x["conciseness"],
                            len(x["hallucinations"]),
                        ]
                        if x
                        else ["", "", "", ""]
                    )
                    + [a["arms"][arm]]
                )
    print(f"\nper-summary rows: {OUT / 'summaries.csv'}")


COMMANDS = {
    "sample": cmd_sample,
    "generate": cmd_generate,
    "nli": cmd_nli,
    "assemble": cmd_assemble,
    "judge": cmd_judge,
    "report": cmd_report,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    COMMANDS[ap.parse_args().command]()


if __name__ == "__main__":
    main()
