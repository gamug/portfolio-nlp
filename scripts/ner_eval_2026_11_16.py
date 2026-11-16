"""Score the pinned NER model on the owner's 100-text organisation set (`data/ner_val100/to_tag.csv`, git-ignored: it holds news text).

Run: ``uv run scripts/ner_eval_2026_11_16.py``. It (1) checks the integrity of the owner's file and stops if the structure is broken
(the labels that are not in the text are reported, not fatal); (2) runs the model step 3 runs (``news_company_stage.SpanFinder``) on the ``text``
column; (3) standardises both sides to one canonical company (a universe ticker through the matcher and an alias list, else the normalised
name) and compares them per text as sets; (4) writes ``docs/ner_results_2026_11_16/ner_metrics.json`` and ``ner_errors.csv`` (organisation
names and spans only, no news text). See ``docs/ner_training.md``.

Three treatments of the labels that are not in the text: **as they are**, **removed**, and **repaired** (``REPAIRS``: the developer's proposal for
the owner to confirm; the owner's file is never edited). Each is scored on every organisation and on the universe companies only. Further views:
``ner+text`` (the spans plus ``news_company_stage.direct_mentions``, what step 3 runs with ``--candidates ner+text``), no alias list, and the
cause of each missed universe company (the NER did not tag the name, or tagged it and the matcher did not map it).
"""

import csv
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

import news_company_stage as ncs  # noqa: E402
from news_nlp.triage import Company, CompanyResolver, normalize_name  # noqa: E402

SET_DIR = ROOT / "data" / "ner_val100"
SET_PATH = SET_DIR / "to_tag.csv"
KEY_PATH = SET_DIR / "key.csv"
OUT = ROOT / "docs" / "ner_results_2026_11_16"

# normalized name -> ticker: the same company under another name. **Chosen after the first run showed these as errors**, so the list is fitted to
# this set (docs/ner_training.md, section 3); the "no alias list" view shows what it moves.
ALIAS = {
    "google": "GOOGL", "alphabet": "GOOGL", "facebook": "META", "j & j": "JNJ", "johnson & johnson": "JNJ",
    "united": "UAL", "united airlines": "UAL", "continental airlines": "UAL", "amazon web services": "AMZN",
    "aws": "AMZN", "bank of america merrill lynch": "BAC", "jpmorgan": "JPM", "jp morgan": "JPM",
    "jp morgan chase": "JPM", "jpmorgan chase": "JPM", "citi": "C", "citigroup": "C",
}  # fmt: skip
SAME_TICKER = {"GOOG": "GOOGL", "FOXA": "FOX", "NWS": "NWSA"}  # share classes of one company
SAME_NAME = {"european union": "eu", "the fed": "federal reserve", "fed": "federal reserve"}
MEDIA = {
    "cnbc",
    "reuters",
    "wsj",
    "wall street journal",
    "barron",
    "barrons",
    "bloomberg",
    "financial times",
    "mad money",
}
# The developer's repair of the labels that are not in the English text as whole words (the owner confirms; none of the four labels of row 26 can be repaired:
# they are in neither the English nor the Spanish text, so they are dropped). Every repaired name is checked to be in the text.
REPAIRS: dict[tuple[int, str], list[str]] = {
    (14, "Alphabe"): ["Alphabet"],  # a typo of the label
    (17, "Casa Blanca"): ["White House"],
    (18, "Japón Aflac"): ["Aflac"],
    (20, "Unión Europea"): ["European Union"],
    (26, "Dish Network"): [],
    (26, "Verizon Communications"): [],
    (26, "Vodafone"): [],
    (26, "Allergan"): [],
    (27, "Reserva Federal"): ["Fed"],
    (33, "Douyin de ByteDance"): ["ByteDance", "Douyin"],
    (37, "NHTSA"): [
        "NHTSA"
    ],  # the label is right; the text writes "NHTSAsaid" (words stuck together), so it is not a whole word there
    (48, "Dow UnitedHealth"): ["UnitedHealth"],
    (96, "Xiaom"): ["Xiaomi"],  # a typo of the label
    (92, "Alipay.Starbucks"): ["Alipay", "Starbucks"],
}
REQUIRED = ("review_order", "key", "title", "text", "organizations")


class IntegrityError(RuntimeError):
    """The owner's file is not structurally usable: nothing is scored."""


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def split_labels(cell: str) -> list[str]:
    """The comma-separated organisations of one cell, trimmed, empty items dropped."""
    return [x.strip() for x in str(cell).split(",") if x.strip()]


def make_canon(resolver: CompanyResolver, aliases: bool = True) -> Any:
    """The function that turns a name into its canonical company: ``T:<ticker>`` for a universe company, else ``N:<normalised name>``."""

    def canon(name: str) -> str:
        if name.startswith(
            "T:"
        ):  # a ticker already (the matcher's own mentions in the ner+text view)
            return "T:" + SAME_TICKER.get(name[2:], name[2:])
        key = normalize_name(name)
        if aliases and key in ALIAS:
            return "T:" + ALIAS[key]
        ticker = resolver.ticker_of_name(name)
        if ticker:
            return "T:" + SAME_TICKER.get(ticker, ticker)
        key = SAME_NAME.get(key, key) if aliases else key
        return "N:" + (key or name.strip().lower())

    return canon


def _written(name: str, text: str) -> bool:
    """Whether ``name`` is written in ``text`` as a whole word or words (case ignored): ``Alphabe`` is not in ``Alphabet``."""
    return (
        re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, flags=re.IGNORECASE) is not None
    )


def _structure(data: pd.DataFrame, key: pd.DataFrame) -> None:
    """Raises ``IntegrityError`` for a structural failure of the owner's file."""
    missing = [c for c in REQUIRED if c not in data.columns]
    if missing:
        raise IntegrityError(f"columns missing: {missing}")
    problems: list[str] = []
    if sorted(data.review_order) != list(range(1, len(data) + 1)):
        problems.append("review_order is not 1..N without gaps")
    if not data.key.is_unique:
        problems.append("key is not unique")
    if (
        len(key) != len(data)
        or not (
            data.sort_values("review_order").key.to_numpy()
            == key.sort_values("review_order").key.to_numpy()
        ).all()
    ):
        problems.append("key does not match key.csv")
    for column in ("text", "organizations"):
        blanks = data[column].isna() | (data[column].astype(str).str.strip() == "")
        if blanks.any():
            problems.append(f"empty {column} in rows {data.review_order[blanks].tolist()}")
    if problems:
        raise IntegrityError("; ".join(problems))


def integrity(data: pd.DataFrame, key: pd.DataFrame, resolver: CompanyResolver) -> dict[str, Any]:
    """The checks on the owner's file. Raises ``IntegrityError`` for a structural failure; the labels that are not in the text are returned.

    A label is *not in the text* when it is not written in the ``text`` column as a whole word (case ignored) and does not name a universe company the text
    names (``direct_mentions``, no model: ``United Airlines`` for ``UAL`` passes, ``Casa Blanca`` does not). **Only ``text`` is searched, because it is the
    only column the model reads**: a label written from the title alone is reported, as the model could never find it."""
    _structure(data, key)
    canon = make_canon(resolver)
    not_in_text: list[tuple[int, str]] = []
    repeated = 0
    total = 0
    for row in data.itertuples():
        labels = split_labels(row.organizations)
        total += len(labels)
        repeated += len(labels) - len(set(labels))
        haystack = str(row.text)
        named = {"T:" + t for t in ncs.direct_mentions(resolver, haystack)}
        for label in labels:
            if not _written(label, haystack) and canon(label) not in named:
                not_in_text.append((int(row.review_order), label))
    return {
        "rows": len(data),
        "organizations": total,
        "repeated_in_a_row": repeated,
        "labels_not_in_the_text": [{"row": r, "label": x} for r, x in not_in_text],
        "rows_with_such_labels": len({r for r, _ in not_in_text}),
    }


def check_repairs(data: pd.DataFrame, flagged: Sequence[tuple[int, str]]) -> None:
    """Every repaired name must be in its ``text`` (the model's input), and every flagged label must have a repair entry (so a new flagged label is noticed)."""
    by_row = {int(r.review_order): str(r.text) for r in data.itertuples()}
    unknown = [f for f in flagged if f not in REPAIRS]
    if unknown:
        raise IntegrityError(f"labels not in the text that REPAIRS does not list: {unknown}")
    for (row, label), names in REPAIRS.items():
        if row not in by_row:
            continue  # a repair for a row this file does not have (another set)
        for name in names:
            if (
                name.lower() not in by_row[row].lower()
            ):  # a substring: row 37's text has the name stuck to the next word
                raise IntegrityError(
                    f"repair {name!r} of {label!r} is not in the text of row {row}"
                )


def gold_names(
    rows: Sequence[int], labels: Sequence[Sequence[str]], treatment: str
) -> list[list[str]]:
    """The owner's labels per text under one treatment: ``as_is``, ``removed`` (the labels of REPAIRS dropped) or ``repaired``."""
    out: list[list[str]] = []
    for row, names in zip(rows, labels, strict=True):
        kept: list[str] = []
        for name in names:
            if (row, name) not in REPAIRS or treatment == "as_is":
                kept.append(name)
            elif treatment == "repaired":
                kept.extend(REPAIRS[(row, name)])
        out.append(kept)
    return out


def score(
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    canon: Any,
    universe_only: bool = False,
    drop_media: bool = False,
) -> tuple[dict[str, float | int], list[tuple[set[str], set[str]]]]:
    """Micro precision, recall and F1 of the per-text canonical sets, and the (missed, extra) sets of each text."""
    tp = fp = fn = 0
    per_text: list[tuple[set[str], set[str]]] = []
    for g, p in zip(gold, pred, strict=True):
        gold_set = {canon(x) for x in g}
        pred_set = {canon(x) for x in p}
        if universe_only:
            gold_set = {x for x in gold_set if x.startswith("T:")}
            pred_set = {x for x in pred_set if x.startswith("T:")}
        if drop_media:
            gold_set = {x for x in gold_set if x[2:] not in MEDIA}
            pred_set = {x for x in pred_set if x[2:] not in MEDIA}
        tp, fp, fn = (
            tp + len(gold_set & pred_set),
            fp + len(pred_set - gold_set),
            fn + len(gold_set - pred_set),
        )
        per_text.append((gold_set - pred_set, pred_set - gold_set))
    return prf(tp, fp, fn), per_text


def near_misses(per_text: Sequence[tuple[set[str], set[str]]]) -> int:
    """Pairs, in one text, of a missed name and an extra name where one normalised name contains the other (``solaris energy`` and
    ``solaris energy infrastructure``): counted once as a miss and once as an extra by the exact match."""
    pairs = 0
    for missed, extra in per_text:
        used: set[str] = set()
        for m in sorted(x[2:] for x in missed if x.startswith("N:")):
            for e in sorted(x[2:] for x in extra if x.startswith("N:") and x not in used):
                if m and e and (m in e or e in m):
                    used.add(e)
                    pairs += 1
                    break
    return pairs


def miss_causes(
    data: pd.DataFrame,
    gold: Sequence[Sequence[str]],
    spans: Sequence[Sequence[Mapping[str, Any]]],
    canon: Any,
) -> dict[str, list[str]]:
    """For every universe company the owner names and the NER-plus-matcher misses: *ner_untagged* (no ORG span overlaps **any** occurrence of the name in the text),
    *matcher_unmapped* (a span overlaps one and the matcher gives it another company or none) or *not_located* (the label is not written in the
    text, so no place to look: a ticker written as ``MSFT`` for ``Microsoft``)."""
    causes: dict[str, list[str]] = {"ner_untagged": [], "matcher_unmapped": [], "not_located": []}
    for row, g, sp in zip(data.itertuples(), gold, spans, strict=True):
        text = str(row.text)
        found = {canon(str(s["text"])) for s in sp if s.get("entity_type") == "ORG"}
        for name in g:
            target = canon(name)
            if not target.startswith("T:") or target in found:
                continue
            places = [m.start() for m in re.finditer(re.escape(name), text, flags=re.IGNORECASE)]
            label = f"{int(row.review_order)}:{name}"
            if not places:
                causes["not_located"].append(label)
            elif any(  # a span over ANY occurrence of the name: the NER tagged it
                s.get("entity_type") == "ORG"
                and s["start_char"] < at + len(name)
                and s["end_char"] > at
                for s in sp
                for at in places
            ):
                causes["matcher_unmapped"].append(label)
            else:
                causes["ner_untagged"].append(label)
    return causes


def repairs_summary(flagged: Sequence[tuple[int, str]]) -> dict[str, int]:
    """What the repair overlay does with the labels not in the text: *replaced* by another name, *kept* (the label is right, the text is
    defective: the replacement is the same name) or *dropped* (no replacement is possible)."""
    out = {"labels_not_in_the_text": len(flagged), "replaced": 0, "kept": 0, "dropped": 0}
    for row, label in flagged:
        names = REPAIRS[(row, label)]
        out["dropped" if not names else "kept" if names == [label] else "replaced"] += 1
    return out


def run(
    data: pd.DataFrame,
    key: pd.DataFrame,
    resolver: CompanyResolver,
    spans: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Everything the page reports, from the owner's file and the NER spans (a test passes stub spans)."""
    checks = integrity(data, key, resolver)
    flagged = [(x["row"], x["label"]) for x in checks["labels_not_in_the_text"]]
    check_repairs(data, flagged)
    rows = [int(r) for r in data.review_order]
    labels = [split_labels(s) for s in data.organizations]
    canon = make_canon(resolver)
    ner = [[str(s["text"]) for s in sp if s.get("entity_type") == "ORG"] for sp in spans]
    with_text = [
        ner[i] + ["T:" + t for t in ncs.direct_mentions(resolver, str(data.text.iloc[i]))]
        for i in range(len(data))
    ]
    views: dict[str, Any] = {}
    errors: dict[str, list[tuple[set[str], set[str]]]] = {}
    for treatment in ("as_is", "removed", "repaired"):
        gold = gold_names(rows, labels, treatment)
        for scope, universe_only in (("all_organisations", False), ("universe_only", True)):
            views[f"{treatment}/{scope}"], per_text = score(gold, ner, canon, universe_only)
            errors[f"{treatment}/{scope}"] = per_text
    repaired = gold_names(rows, labels, "repaired")
    views["repaired/all_organisations_media_left_out"] = score(repaired, ner, canon, False, True)[0]
    views["repaired/universe_only_ner_plus_text"] = score(repaired, with_text, canon, True)[0]
    views["as_is/all_organisations_no_alias_list"] = score(
        gold_names(rows, labels, "as_is"), ner, make_canon(resolver, aliases=False)
    )[0]
    causes = miss_causes(data, repaired, spans, canon)
    return {
        "texts": len(data),
        "integrity": checks,
        "repairs": repairs_summary(flagged),
        "views": views,
        "near_miss_pairs_repaired_all": near_misses(errors["repaired/all_organisations"]),
        "miss_causes_repaired_universe": {k: v for k, v in causes.items()},
        "_errors": errors["repaired/all_organisations"],
        "_ner": ner,
    }


def write_outputs(
    out_dir: Path,
    result: dict[str, Any],
    errors: Sequence[tuple[set[str], set[str]]],
    ner: Sequence[Sequence[str]],
    data: pd.DataFrame,
    canon: Any,
) -> None:
    """``ner_metrics.json`` and ``ner_errors.csv`` (both LF; one row per text that has a miss or an extra, with the span text of each extra)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ner_metrics.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (out_dir / "ner_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["review_order", "missed", "extra", "extra_spans"]
        )  # T:<ticker> = a universe company, N:<name> = a normalised name
        for row, (missed, extra), names in zip(data.review_order, errors, ner, strict=True):
            if missed or extra:
                spans_of = Counter(f"{canon(n)}={n}" for n in names if canon(n) in extra)
                writer.writerow(
                    [
                        int(row),
                        "; ".join(sorted(missed)),
                        "; ".join(sorted(extra)),
                        "; ".join(sorted(spans_of)),
                    ]
                )


def main() -> None:
    universe_rows = pd.read_csv(ROOT / "gold" / "universe.csv").fillna("")
    resolver = CompanyResolver(
        [Company(r.ticker, r.company, r.gics_sector, "") for r in universe_rows.itertuples()]
    )
    data = pd.read_csv(SET_PATH, encoding="utf-8-sig")
    key = pd.read_csv(KEY_PATH, encoding="utf-8-sig")
    finder = ncs.SpanFinder()
    spans = finder.spans(data.text.tolist())
    result = run(data, key, resolver, spans)
    errors, ner = result.pop("_errors"), result.pop("_ner")
    result["model"] = f"{finder.model_name}@{finder.model_revision}"
    write_outputs(OUT, result, errors, ner, data, make_canon(resolver))
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
