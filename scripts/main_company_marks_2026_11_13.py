"""The owner's marks for step 3 (T-170): which company is main in a sub-news, and which organisation spans the matcher ignored.

    uv run scripts/main_company_marks_2026_11_13.py draw  --results-db R --source-db S --summary-revision REV --cut-revision REV --companies-revision REV
    uv run scripts/main_company_marks_2026_11_13.py spans --results-db R --source-db S --summary-revision REV --cut-revision REV
    uv run scripts/main_company_marks_2026_11_13.py score --tags data/main_company/to_tag.csv

``draw`` writes the sheet of main companies: one row per (sub-news, candidate company), the text step 3 read, **no score on the sheet**
(the owner marks blind). The sub-news are drawn **by the relevance score** of their best company (six bands and the ones with no
candidate are left out), the same number from each band, so the bands near the threshold are marked as well as the sure ones; the
population of each band is kept in the sidecar so ``score`` weighs the marks back to the corpus. ``spans`` runs the NER model on a random
draw of sub-news and writes the organisation spans the matcher turned into no ticker, for the owner to say which ones are universe companies.
``score`` reads the marked sheet and says, by threshold, how often the model's main company is the owner's, the AUC, and whether the
rule of T-170 (AUC more than 5 points under 0.90) gives sub-news a threshold of their own.

The files are read by the owner and stay in ``data/`` (git-ignored); nothing is written to the databases.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DIR = ROOT / "data" / "main_company"
DEFAULT_SEED = 20261113
#: The bands of the best relevance score of a sub-news: [lo, hi); the last one includes 1.0.
BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.1),
    (0.1, 0.3),
    (0.3, 0.5),
    (0.5, 0.7),
    (0.7, 0.9),
    (0.9, 1.0000001),
)
YES = {"si", "sí", "s", "yes", "y", "1", "x"}
NO = {"no", "n", "0"}
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
#: T-170: the relevance model's AUC on the summaries; sub-news get their own threshold when the AUC is more than this many points under it.
SUMMARY_AUC = 0.90
AUC_MARGIN = 0.05
CONTEXT = 160

TAG_COLUMNS = (
    "item_id",
    "texto",
    "ticker",
    "empresa",
    "es_principal",
    "falta_la_principal",
    "notas",
)
SPAN_COLUMNS = ("span_id", "span", "contexto", "es_del_universo", "ticker", "notas")

README = """Regla de lectura de la hoja de empresas principales (to_tag.csv): cada noticia (`item_id`) sale una vez por cada empresa candidata.
- es_principal: `si` si el texto trata principalmente de esa empresa, `no` si solo la menciona. Una noticia puede tener varias principales o ninguna.
- falta_la_principal: solo en la primera fila de cada noticia; escribe el ticker si la empresa principal del texto no está entre las filas
  (el modelo no la encontró), o déjalo vacío.
La hoja no muestra ninguna puntuación del modelo, a propósito.
"""
README_SPANS = """Regla de lectura de la hoja de menciones ignoradas (spans_to_tag.csv): cada fila es un nombre que el modelo de NER marcó como organización y el
buscador no pudo asociar a ninguna empresa del universo.
- es_del_universo: `si` si el nombre es una empresa del universo (y entonces escribe su `ticker`), `no` si no lo es (otra empresa, un fondo, un banco...).
"""


# ----------------------------------------------------------------------------------------------------------------------
# the draw


@dataclass(frozen=True)
class Candidate:
    ticker: str
    score: float
    is_main: bool
    mentions: int


@dataclass(frozen=True)
class SubNews:
    item_id: str
    text: str
    state: str
    candidates: tuple[Candidate, ...]

    @property
    def best(self) -> float | None:
        return max((c.score for c in self.candidates), default=None)


def band_of(score: float) -> int:
    """The index of the band a best score falls in."""
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= score < hi:
            return i
    raise ValueError(f"a relevance score is between 0 and 1, not {score}")


def fetch_sub_news(
    conn: sqlite3.Connection, summary_revision: str, cut_revision: str, companies_revision: str
) -> list[SubNews]:
    """Every sub-news of the cut that step 3 finished at ``companies_revision`` (``done`` or ``review``), with the text step 3 read and
    its candidates."""
    candidates: dict[str, list[Candidate]] = defaultdict(list)
    for item_id, ticker, score, is_main, mentions in conn.execute(
        "SELECT item_id, ticker, relevance_score, is_main, mention_count FROM news_company "
        "WHERE item_kind = 'sub_news' AND model_revision = ?",
        (companies_revision,),
    ):
        candidates[str(item_id)].append(
            Candidate(ticker, float(score), bool(is_main), int(mentions))
        )
    out: list[SubNews] = []
    for item_id, text, state in conn.execute(
        "SELECT s.item_id, s.summary_text, p.status FROM news_summary s "
        "JOIN sub_news n ON CAST(n.id AS TEXT) = s.item_id AND n.extractor_revision = ? "
        "JOIN pipeline_step_state p ON p.item_kind = 'sub_news' AND p.item_id = s.item_id "
        "AND p.step = 'companies' AND p.model_revision = ? AND p.status IN ('done', 'review') "
        "WHERE s.item_kind = 'sub_news' AND s.model_revision = ? ORDER BY n.id",
        (cut_revision, companies_revision, summary_revision),
    ):
        found = sorted(candidates.get(str(item_id), []), key=lambda c: (-c.score, c.ticker))
        out.append(SubNews(str(item_id), str(text), str(state), tuple(found)))
    return out


def draw_items(
    items: Sequence[SubNews], per_band: int, seed: int
) -> tuple[list[SubNews], dict[int, int]]:
    """``per_band`` sub-news from each band of the best score (all of a band when it has fewer), in a random order, and the population of
    each band. The ones with no candidate are not a band: there is no company to mark."""
    by_band: dict[int, list[SubNews]] = defaultdict(list)
    for item in items:
        if item.best is not None:
            by_band[band_of(item.best)].append(item)
    rng = random.Random(seed)  # noqa: S311 -- a reproducible draw, not security
    picked: list[SubNews] = []
    for band in sorted(by_band):
        pool = sorted(by_band[band], key=lambda i: int(i.item_id))
        picked.extend(rng.sample(pool, min(per_band, len(pool))))
    rng.shuffle(picked)  # the sheet's order says nothing about the score
    return picked, {band: len(pool) for band, pool in by_band.items()}


def sheet_rows(sample: Sequence[SubNews], names: Mapping[str, str]) -> list[dict[str, str]]:
    """One row per (sub-news, candidate); the text and the first-row column of the missing company are on the first row of each item.
    The order of the candidates inside an item is by ticker, not by score (the sheet does not show the model's order)."""
    rows: list[dict[str, str]] = []
    for item in sample:
        for k, cand in enumerate(sorted(item.candidates, key=lambda c: c.ticker)):
            rows.append(
                {
                    "item_id": item.item_id,
                    "texto": item.text,
                    "ticker": cand.ticker,
                    "empresa": names.get(cand.ticker, ""),
                    "es_principal": "",
                    "falta_la_principal": "" if k == 0 else "-",
                    "notas": "",
                }
            )
    return rows


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists: it may hold the owner's marks; move it first")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def strata_of(sample: Sequence[SubNews], population: Mapping[int, int]) -> dict[str, Any]:
    """The sidecar: the band of each item and the figures ``score`` needs (the model's scores are here, not on the sheet)."""
    drawn: dict[int, int] = defaultdict(int)
    items: dict[str, dict[str, Any]] = {}
    for item in sample:
        assert item.best is not None  # the draw takes no item without a candidate
        band = band_of(item.best)
        drawn[band] += 1
        items[item.item_id] = {
            "band": band,
            "scores": {c.ticker: c.score for c in item.candidates},
        }
    return {
        "bands": [list(b) for b in BANDS],
        "population": {str(k): v for k, v in sorted(population.items())},
        "sampled": {str(k): v for k, v in sorted(drawn.items())},
        "items": items,
    }


def names_of(source: sqlite3.Connection) -> dict[str, str]:
    """ticker -> company name, as the cut and step 3 read the universe from the SOURCE store."""
    return {
        str(t): str(n or "")
        for t, n in source.execute(
            "SELECT ticker, MIN(company) FROM articles WHERE fetch_status = 'ok' GROUP BY ticker"
        )
    }


def open_ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=120)


def cmd_draw(args: argparse.Namespace) -> None:
    results, source = open_ro(args.results_db), open_ro(args.source_db)
    items = fetch_sub_news(
        results, args.summary_revision, args.cut_revision, args.companies_revision
    )
    sample, population = draw_items(items, args.per_band, args.seed)
    rows = sheet_rows(sample, names_of(source))
    out: Path = args.out
    write_csv(out, TAG_COLUMNS, rows)
    meta = strata_of(sample, population)
    meta.update(
        seed=args.seed,
        summary_revision=args.summary_revision,
        cut_revision=args.cut_revision,
        companies_revision=args.companies_revision,
        no_candidate=sum(1 for i in items if i.best is None),
        items_total=len(items),
    )
    out.with_suffix(".strata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(
        f"{len(sample)} sub-news, {len(rows)} rows -> {out}\n"
        f"population by band {meta['population']}; with no candidate: {meta['no_candidate']} of {len(items)}\n\n{README}"
    )


# ----------------------------------------------------------------------------------------------------------------------
# the ignored spans


def ignored_spans(
    texts: Mapping[str, str],
    spans_of: Callable[[Sequence[str]], list[list[dict[str, Any]]]],
    ticker_of_name: Callable[[str], str | None],
) -> list[dict[str, str]]:
    """The organisation spans of ``texts`` that the matcher turns into no ticker (the ones step 3 ignores), with their context."""
    ids = list(texts)
    found = spans_of([texts[i] for i in ids])
    out: list[dict[str, str]] = []
    for item_id, spans in zip(ids, found, strict=True):
        text = texts[item_id]
        for span in spans:
            if span.get("entity_type") != "ORG":
                continue
            name = str(span.get("text", "")).strip()
            if not name or ticker_of_name(name) is not None:
                continue
            start, end = int(span.get("start", 0)), int(span.get("end", 0))
            out.append(
                {
                    "item_id": item_id,
                    "span": name,
                    "contexto": text[max(0, start - CONTEXT) : end + CONTEXT].replace("\n", " "),
                }
            )
    return out


def cmd_spans(args: argparse.Namespace) -> None:
    from news_company_stage import SpanFinder  # noqa: PLC0415 -- torch, only for this command
    from news_nlp.triage import Company, CompanyResolver  # noqa: PLC0415

    results, source = open_ro(args.results_db), open_ro(args.source_db)
    universe = [
        Company(str(t), str(c or ""), str(s or ""), str(g or ""))
        for t, c, s, g in source.execute(
            "SELECT DISTINCT ticker, company, gics_sector, gics_sub_industry FROM articles "
            "WHERE fetch_status = 'ok' ORDER BY ticker, company"
        )
    ]
    resolver = CompanyResolver(universe)
    items = fetch_sub_news(
        results, args.summary_revision, args.cut_revision, args.companies_revision
    )
    rng = random.Random(args.seed)  # noqa: S311 -- a reproducible draw, not security
    pool = sorted(items, key=lambda i: int(i.item_id))
    texts = {i.item_id: i.text for i in rng.sample(pool, min(args.texts, len(pool)))}
    found = ignored_spans(texts, SpanFinder().spans, resolver.ticker_of_name)
    chosen = rng.sample(found, min(args.spans, len(found)))
    rows = [
        {
            "span_id": str(k + 1),
            "span": r["span"],
            "contexto": r["contexto"],
            "es_del_universo": "",
            "ticker": "",
            "notas": "",
        }
        for k, r in enumerate(chosen)
    ]
    out: Path = args.out
    write_csv(out, SPAN_COLUMNS, rows)
    out.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "texts_read": len(texts),
                "ignored_spans_found": len(found),
                "spans_on_sheet": len(rows),
                "summary_revision": args.summary_revision,
                "cut_revision": args.cut_revision,
                "companies_revision": args.companies_revision,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"{len(texts)} texts read, {len(found)} ignored spans, {len(rows)} on the sheet -> {out}\n\n{README_SPANS}"
    )


# ----------------------------------------------------------------------------------------------------------------------
# the score


def _mark(value: str) -> bool | None:
    v = value.strip().lower()
    if v in YES:
        return True
    if v in NO:
        return False
    return None


def read_marks(
    tags: Path, meta: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str], int]:
    """The marked rows (item, ticker, owner's mark, the model's score, the band) and the ``falta_la_principal`` of each item; the rows
    not marked are counted, not guessed."""
    with tags.open(encoding="utf-8-sig", newline="") as fh:
        raw = list(csv.DictReader(fh))
    rows: list[dict[str, Any]] = []
    missing: dict[str, str] = {}
    unmarked = 0
    for r in raw:
        item = meta["items"].get(r["item_id"])
        if item is None:
            raise SystemExit(
                f"item {r['item_id']} is not in the sidecar: the sheet and the sidecar are of different draws"
            )
        gap = r.get("falta_la_principal", "").strip()
        if gap and gap != "-":
            missing[r["item_id"]] = gap
        mark = _mark(r.get("es_principal", ""))
        if mark is None:
            unmarked += 1
            continue
        rows.append(
            {
                "item": r["item_id"],
                "ticker": r["ticker"],
                "owner": mark,
                "score": float(item["scores"][r["ticker"]]),
                "band": int(item["band"]),
            }
        )
    return rows, missing, unmarked


def weights(meta: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """The weight of each row: its band's population over the items marked of that band (every item counts for its band's share)."""
    items_in_band: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        items_in_band[int(r["band"])].add(str(r["item"]))
    return [meta["population"][str(r["band"])] / len(items_in_band[int(r["band"])]) for r in rows]


def weighted_auc(
    scores: Sequence[float], labels: Sequence[bool], w: Sequence[float]
) -> float | None:
    """The probability that a weighted main row scores above a weighted not-main row (ties count half); ``None`` with one class only."""
    pos = [(s, wi) for s, y, wi in zip(scores, labels, w, strict=True) if y]
    neg = [(s, wi) for s, y, wi in zip(scores, labels, w, strict=True) if not y]
    if not pos or not neg:
        return None
    num = sum(
        wp * wn * (1.0 if sp > sn else 0.5 if sp == sn else 0.0) for sp, wp in pos for sn, wn in neg
    )
    return num / (sum(wp for _, wp in pos) * sum(wn for _, wn in neg))


def by_threshold(
    rows: Sequence[Mapping[str, Any]], w: Sequence[float], thresholds: Sequence[float] = THRESHOLDS
) -> list[dict[str, float | None]]:
    """For each threshold, over the weighted rows: precision and recall of "main" against the owner's marks, and the share of the
    sub-news with a main company (the model's) against the owner's share."""
    out: list[dict[str, float | None]] = []
    for t in thresholds:
        tp = sum(wi for r, wi in zip(rows, w, strict=True) if r["score"] >= t and r["owner"])
        fp = sum(wi for r, wi in zip(rows, w, strict=True) if r["score"] >= t and not r["owner"])
        fn = sum(wi for r, wi in zip(rows, w, strict=True) if r["score"] < t and r["owner"])
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        out.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1})
    return out


def share_with_main(
    rows: Sequence[Mapping[str, Any]], w: Sequence[float], threshold: float | None
) -> float:
    """The weighted share of the marked sub-news that have a main company: the owner's (``threshold`` ``None``) or the model's at it."""
    total = 0.0
    hit = 0.0
    seen: dict[str, tuple[float, bool]] = {}
    for r, wi in zip(rows, w, strict=True):
        item = str(r["item"])
        main = bool(r["owner"]) if threshold is None else r["score"] >= threshold
        prev = seen.get(item)
        seen[item] = (wi, main or (prev[1] if prev else False))
    for wi, main in seen.values():
        total += wi
        hit += wi if main else 0.0
    return hit / total if total else 0.0


def cmd_score(args: argparse.Namespace) -> None:
    tags: Path = args.tags
    meta = json.loads(tags.with_suffix(".strata.json").read_text(encoding="utf-8"))
    rows, missing, unmarked = read_marks(tags, meta)
    if not rows:
        sys.exit("no row is marked yet (es_principal: si / no)")
    w = weights(meta, rows)
    auc = weighted_auc([r["score"] for r in rows], [bool(r["owner"]) for r in rows], w)
    lines = [
        f"{len(rows)} rows marked ({unmarked} not marked, left out), {len({r['item'] for r in rows})} sub-news; weighted to the corpus by band.",
        f"the owner's share of sub-news with a main company: {share_with_main(rows, w, None):.1%}",
    ]
    if auc is None:
        lines.append("AUC: not defined (the owner marked only one class)")
    else:
        verdict = (
            "**sub-news get a threshold of their own** (T-170 rule)"
            if auc < SUMMARY_AUC - AUC_MARGIN
            else "the summaries' threshold is kept (T-170 rule)"
        )
        lines.append(
            f"AUC of the relevance score against the owner's marks: {auc:.3f} (summaries: {SUMMARY_AUC}) -> {verdict}"
        )
    if missing:
        lines.append(
            f"sub-news where the owner says the main company was not among the candidates: {len(missing)}"
        )
    lines += [
        "",
        "| threshold | precision | recall | F1 | model's share with a main company |",
        "|---|---|---|---|---|",
    ]
    for row in by_threshold(rows, w):
        t = float(row["threshold"] or 0)
        cells = [
            f"{row[k]:.1%}" if row[k] is not None else "-" for k in ("precision", "recall", "f1")
        ]
        lines.append(
            f"| {t:.1f} | {cells[0]} | {cells[1]} | {cells[2]} | {share_with_main(rows, w, t):.1%} |"
        )
    report = "\n".join(lines)
    print(report)
    tags.with_suffix(".report.md").write_text(report + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("draw", "spans"):
        p = sub.add_parser(name)
        p.add_argument("--results-db", required=True)
        p.add_argument("--source-db", required=True)
        p.add_argument("--summary-revision", required=True)
        p.add_argument("--cut-revision", required=True)
        p.add_argument("--companies-revision", required=True)
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        if name == "draw":
            p.add_argument("--per-band", type=int, default=30)
            p.add_argument("--out", type=Path, default=DEFAULT_DIR / "to_tag.csv")
        else:
            p.add_argument("--texts", type=int, default=150)
            p.add_argument("--spans", type=int, default=50)
            p.add_argument("--out", type=Path, default=DEFAULT_DIR / "spans_to_tag.csv")
    s = sub.add_parser("score")
    s.add_argument("--tags", type=Path, default=DEFAULT_DIR / "to_tag.csv")
    args = parser.parse_args()
    {"draw": cmd_draw, "spans": cmd_spans, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
