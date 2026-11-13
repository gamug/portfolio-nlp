"""The evidence behind the rows of the table in ``docs/sector_step.md``, in one script (2026-11-13; answers the review of #179).

    uv run scripts/sector_rollup_2026_11_12.py                      # once: data/sector_rollup.csv, the 70 single-sector bodies, from gold/
    uv run scripts/sector_evidence_2026_11_13.py [descriptions] [examples] [models] [votes]   # default: all four parts

Data: the 70 single-sector tagged bodies (A, rolled up to the 2-digit sector) and the 100 real rows of the first yes/no review (B,
``gold/sector_validation.csv`` with its texts); each is split by position (even = TUNE, odd = TEST). Everything is printed for both halves:
the TEST halves were read for several variants, so these numbers are the record of how the choices were made, not a clean measure (the
clean one is ``score_sector_fresh_2026_11_13.py``). Parts:

* ``descriptions``: v2 (the current), v2 with the *not a sector* classes, v3 (from the GICS structure), v2+v3, v4 (v2 plus lines for the tune errors);
* ``examples``: the labelled bodies of the TUNE half as extra "descriptions" (leave-one-out on the TUNE half);
* ``models``: bge-small (the pinned one) against bge-base, bge-large, gte-base and e5-base (needs the models in the Hugging Face cache);
* ``votes``: the sector of the companies a body names (``triage.CompanyResolver`` over ``gold/universe.csv``) alone, as a second opinion, and the gates.

B is biased towards the small model: its rows were picked, shown and balanced with it, and a ``yes`` means "its sector is right".
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sector_embed_baseline_2026_11_12 as base
import sector_embed_tune_2026_11_12 as tune
import sector_review_eval_2026_11_13 as rev

from news_nlp.triage import Company, CompanyResolver

ROOT = Path(__file__).resolve().parent.parent
SECTOR_CODES = {
    "Energy": "10", "Materials": "15", "Industrials": "20", "Consumer Discretionary": "25", "Consumer Staples": "30",
    "Health Care": "35", "Financials": "40", "Information Technology": "45", "Communication Services": "50",
    "Utilities": "55", "Real Estate": "60",
}  # fmt: skip
MODELS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "cls", ""),
    "bge-base": ("BAAI/bge-base-en-v1.5", "cls", ""),
    "bge-large": ("BAAI/bge-large-en-v1.5", "cls", ""),
    "gte-base": ("thenlper/gte-base", "mean", ""),
    "e5-base": ("intfloat/e5-base-v2", "mean", "query: "),
}
#: v4: v2 plus a line for what the tune half got wrong (named EV makers, consumer tech, steel and chemicals, mall owners).
V4_ADD = {
    "25": ["Tesla, Nio and other electric car makers, EV sales and EV ETFs, luxury goods and watch makers, Nike and apparel brands"],
    "45": ["Apple, Samsung, iPhone and consumer electronics makers; bitcoin miners and crypto companies as technology businesses"],
    "15": ["Steel, iron ore, copper and aluminum producers, timber and lumber, chemical makers such as 3M, DuPont and Dow"],
    "60": ["Mall owners such as Simon Property, REIT landlords, distressed retail properties, housing and homebuilders' land"],
    "20": ["Caterpillar, Boeing, GE, 3M and other industrial companies, FedEx and UPS shipping"],
}  # fmt: skip
SECTORS = list(tune.DESC_V2)
MIN_VOTES = 2  # the variant of the gate that asks for at least two votes


class Data:
    """The labelled items, the split and the labels (``reject`` = not a sector body, ``?`` = unknown)."""

    def __init__(self) -> None:
        a_rows, a_texts = base.load_inputs()
        b_rows = rev.read_review()
        self.items = [a_texts[r["article_id"]] for r in a_rows] + [
            (r["title"], r["text"]) for r in b_rows
        ]
        self.labels = np.array(
            [r["sectors"] for r in a_rows] + [rev.label_of(r) or "?" for r in b_rows]
        )
        self.na = len(a_rows)
        pos = [i if i < self.na else i - self.na for i in range(len(self.items))]
        self.tune = np.array([i for i, p in enumerate(pos) if p % 2 == 0])
        self.test = np.array([i for i, p in enumerate(pos) if p % 2 == 1])

    def part(self, idx: np.ndarray, which: str) -> np.ndarray:
        return idx[idx < self.na] if which == "A" else idx[idx >= self.na]


def metrics(data: Data, scores: np.ndarray, names: list[str], idx: np.ndarray) -> str:
    """Per part: precision of the sectors assigned (a not-a-sector class winning assigns nothing), accuracy on the known labels, scope errors caught."""
    pred = np.array(names)[scores.argmax(1)]
    out = []
    for which in ("A", "B"):
        sel = data.part(idx, which)
        sector = np.array([p in tune.DESC_V2 for p in pred[sel]])
        right = pred[sel] == data.labels[sel]
        known = np.array([x not in ("?", "reject") for x in data.labels[sel]])
        rejects = data.labels[sel] == "reject"
        caught = f"{(~sector[rejects]).mean():.0%}" if rejects.any() else "-"
        prec = right[sector].mean() if sector.any() else 0.0
        out.append(
            f"{which}: precision {prec:5.1%}  accuracy(known) {right[known].mean():5.1%}  rejects caught {caught}"
        )
    return "   ".join(out)


def part_descriptions(data: Data, scorer: rev.Scorer) -> None:
    print("== descriptions (bge-small, top-3 chunks)")
    v4 = {k: [*v, *V4_ADD.get(k, [])] for k, v in tune.DESC_V2.items()}
    union = {k: [*tune.DESC_V2[k], *rev.DESC_V3[k]] for k in rev.DESC_V3}
    variants = {
        "v2, no reject classes": tune.DESC_V2,
        "v2 + reject classes (current)": {**tune.DESC_V2, **rev.REJECT},
        "v3 + reject classes": {**rev.DESC_V3, **rev.REJECT},
        "v2+v3 + reject classes": {**union, **rev.REJECT},
        "v4 + reject classes": {**v4, **rev.REJECT},
    }
    for name, desc in variants.items():
        s = scorer.scores(desc, 3)
        for label, idx in (("TUNE", data.tune), ("TEST", data.test)):
            print(f"  {name:32s} {label}  {metrics(data, s, list(desc), idx)}")


def part_examples(data: Data, scorer: rev.Scorer) -> None:
    print("== labelled bodies of the TUNE half as extra descriptions (leave-one-out)")
    desc = {**tune.DESC_V2, **rev.REJECT}
    names = list(desc)
    s = scorer.scores(desc, 3)
    body = np.stack([rev.unit(v.mean(0)) for v in scorer.vecs])
    for w in (0.0, 0.5, 1.0, 2.0):
        proto = np.zeros_like(s)
        for i in range(len(s)):
            for j, n in enumerate(names):
                pool = [t for t in data.tune if data.labels[t] == n and t != i]
                if pool:
                    proto[i, j] = np.sort(body[pool] @ body[i])[-2:].mean()
        print(f"  weight {w:3.1f}  TUNE  {metrics(data, s + w * proto, names, data.tune)}")


def votes_of(data: Data) -> tuple[np.ndarray, np.ndarray]:
    """Votes per sector for each item (each company at most 3) and the number of companies named."""
    with (ROOT / "gold" / "universe.csv").open(encoding="utf-8", newline="") as fh:
        uni = list(csv.DictReader(fh))
    resolver = CompanyResolver(
        [Company(u["ticker"], u["company"], u["gics_sector"], "") for u in uni]
    )
    sector_of = {u["ticker"]: SECTOR_CODES[u["gics_sector"]] for u in uni}
    votes = np.zeros((len(data.items), len(SECTORS)))
    named = np.zeros(len(data.items), dtype=int)
    for i, (title, text) in enumerate(data.items):
        found = resolver.resolve(f"{title}. {text}")
        named[i] = len(found)
        for ticker, n in found.items():
            if ticker in sector_of:
                votes[i, SECTORS.index(sector_of[ticker])] += min(n, 3)
    return votes, named


def gate_line(name: str, data: Data, keep: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> str:
    cells = []
    for which, sel in (("A", data.part(idx, "A")), ("B", data.part(idx, "B")), ("A+B", idx)):
        k = keep[sel]
        prec = (pred[sel] == data.labels[sel])[k].mean() if k.any() else float("nan")
        cells.append(f"{which}: kept {k.mean():4.0%} precision {prec:5.1%}")
    return f"  {name:40s} " + " | ".join(cells)


def part_votes(data: Data, scorer: rev.Scorer) -> None:
    print("== the companies a body names, and the gates (bge-small, v2 + reject classes)")
    desc = {**tune.DESC_V2, **rev.REJECT}
    names = list(desc)
    s = scorer.scores(desc, 3)
    pred = np.array(names)[s.argmax(1)]
    srt = np.sort(s, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    votes, named = votes_of(data)
    has = votes.sum(1) > 0
    vote_pred = np.array(SECTORS)[votes.argmax(1)]
    print(f"  bodies naming at least one universe company: {(named > 0).mean():.0%}")
    known = np.array([x not in ("?", "reject") for x in data.labels])
    for which in ("A", "B"):
        sel = data.part(np.arange(len(data.items)), which)
        k = sel[known[sel] & has[sel]]
        agree = pred[k] == vote_pred[k]
        print(
            f"  {which}: votes alone right {(vote_pred[k] == data.labels[k]).mean():.1%}; embeddings right {(pred[k] == data.labels[k]).mean():.1%}; "
            f"they agree on {agree.mean():.0%} of the bodies and are right then {(pred[k] == data.labels[k])[agree].mean():.1%}"
        )
    is_sector = np.array([p in tune.DESC_V2 for p in pred])
    agree_all = (pred == vote_pred) & has
    # what "agree" means when the vote has no single top sector (a tie) or little to go on (one vote)
    unique = (votes == votes.max(1)[:, None]).sum(1) == 1
    agree_unique = agree_all & unique
    agree_two = agree_unique & (votes.sum(1) >= MIN_VOTES)
    print(
        f"  {(has & ~unique).sum()} of {len(votes)} bodies have a tie at the top of the vote; "
        f"{(has & (votes.sum(1) < MIN_VOTES)).sum()} have a single vote"
    )
    thr = np.quantile(margin[data.tune], 0.4)
    print(f"  margin threshold = 40th percentile of the TUNE half = {thr:.4f}")
    gates = (
        ("embeddings, no gate", is_sector),
        ("margin >= threshold", is_sector & (margin >= thr)),
        ("votes agree (tie: lowest code)", is_sector & agree_all),
        (
            "margin OR agree (tie: lowest code) [first build]",
            is_sector & ((margin >= thr) | agree_all),
        ),
        ("margin OR agree (unique top) [current]", is_sector & ((margin >= thr) | agree_unique)),
        ("margin OR agree (unique top, >= 2 votes)", is_sector & ((margin >= thr) | agree_two)),
    )
    for label, idx in (("TUNE", data.tune), ("TEST", data.test)):
        print(label)
        for name, keep in gates:
            print(gate_line(name, data, keep, pred, idx))


def part_models(data: Data) -> None:
    print(
        "== embedders (v2 + reject classes, top-3 chunks; gate = margin >= 40th percentile of TUNE)"
    )
    desc = {**tune.DESC_V2, **rev.REJECT}
    for label, (model, pooling, prefix) in MODELS.items():
        base.MODEL, base.POOLING, base.PREFIX = model, pooling, prefix
        s = rev.Scorer(data.items).scores(desc, 3)
        names = list(desc)
        pred = np.array(names)[s.argmax(1)]
        srt = np.sort(s, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        keep = np.array([p in tune.DESC_V2 for p in pred]) & (
            margin >= np.quantile(margin[data.tune], 0.4)
        )
        print(f"  {label:10s} TEST  {metrics(data, s, names, data.test)}")
        print(gate_line(f"{label} gate", data, keep, pred, data.test))


def main() -> None:
    parts = Counter(sys.argv[1:] or ["descriptions", "examples", "models", "votes"])
    data = Data()
    print(
        f"A: {data.na} tagged bodies; B: {len(data.items) - data.na} reviewed rows; TUNE {len(data.tune)}, TEST {len(data.test)}"
    )
    if parts.keys() & {"descriptions", "examples", "votes"}:
        scorer = rev.Scorer(data.items)  # the pinned embedder
        if "descriptions" in parts:
            part_descriptions(data, scorer)
        if "examples" in parts:
            part_examples(data, scorer)
        if "votes" in parts:
            part_votes(data, scorer)
    if "models" in parts:
        part_models(data)


if __name__ == "__main__":
    main()
