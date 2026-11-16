"""Score the pinned NER model on the owner's 100-text organisation set (`data/ner_val100/to_tag.csv`, git-ignored: it holds news text).

Run: ``uv run scripts/ner_eval_2026_11_16.py``. Reads the owner's ``organizations`` column (comma-separated, names as they appear in the text),
runs ``ner_stage`` through ``news_company_stage.SpanFinder`` (the model step 3 runs) on the ``text`` column, standardises both sides to one
canonical company (universe ticker through the matcher and an alias list, else the normalised name) and compares them per text as sets.
Writes ``docs/ner_results_2026_11_16/ner_metrics.json`` and ``ner_errors.csv`` (organisation names only, no news text). See ``docs/ner_training.md``.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

import news_company_stage as ncs  # noqa: E402
from news_nlp.triage import Company, CompanyResolver, normalize_name  # noqa: E402

SET_PATH = ROOT / "data" / "ner_val100" / "to_tag.csv"
OUT = ROOT / "docs" / "ner_results_2026_11_16"

# normalized name -> ticker: the same company under another name
ALIAS = {
    "google": "GOOGL", "alphabet": "GOOGL", "facebook": "META", "j & j": "JNJ", "johnson & johnson": "JNJ",
    "united": "UAL", "united airlines": "UAL", "continental airlines": "UAL", "amazon web services": "AWS-AMZN",
    "aws": "AWS-AMZN", "bank of america merrill lynch": "BAC", "jpmorgan": "JPM", "jp morgan": "JPM",
    "jp morgan chase": "JPM", "jpmorgan chase": "JPM", "citi": "C", "citigroup": "C",
}  # fmt: skip
ALIAS["amazon web services"] = ALIAS["aws"] = "AMZN"
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
# labels the integrity check found that are not in the English text (Spanish names, merged names, labels of another article)
FLAGGED = {
    (17, "Casa Blanca"), (18, "Japón Aflac"), (20, "Unión Europea"), (26, "Dish Network"), (26, "Verizon Communications"),
    (26, "Vodafone"), (26, "Allergan"), (27, "Reserva Federal"), (33, "Douyin de ByteDance"), (48, "Dow UnitedHealth"),
    (92, "Alipay.Starbucks"),
}  # fmt: skip


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


def main() -> None:
    universe_rows = pd.read_csv(ROOT / "gold" / "universe.csv").fillna("")
    resolver = CompanyResolver(
        [Company(r.ticker, r.company, r.gics_sector, "") for r in universe_rows.itertuples()]
    )

    def canon(name: str) -> str:
        key = normalize_name(name)
        if key in ALIAS:
            return "T:" + ALIAS[key]
        ticker = resolver.ticker_of_name(name)
        if ticker:
            return "T:" + SAME_TICKER.get(ticker, ticker)
        key = SAME_NAME.get(key, key)
        return "N:" + (key or name.strip().lower())

    data = pd.read_csv(SET_PATH, encoding="utf-8-sig")
    gold = [[x.strip() for x in s.split(",") if x.strip()] for s in data.organizations]
    finder = ncs.SpanFinder()
    spans = finder.spans(data.text.tolist())
    pred = [[s["text"] for s in sp if s.get("entity_type") == "ORG"] for sp in spans]

    def view(
        drop_flagged: bool, universe_only: bool, drop_media: bool
    ) -> tuple[dict[str, float | int], list[tuple[int, list[str], list[str]]]]:
        tp = fp = fn = 0
        errors = []
        for row, g, p in zip(data.review_order, gold, pred, strict=True):
            gold_set = {canon(x) for x in g if not (drop_flagged and (int(row), x) in FLAGGED)}
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
            errors.append((int(row), sorted(gold_set - pred_set), sorted(pred_set - gold_set)))
        return prf(tp, fp, fn), errors

    views = {
        "all_labels": view(False, False, False)[0],
        "flagged_labels_removed": view(True, False, False)[0],
        "flagged_labels_and_media_removed": view(True, False, True)[0],
        "universe_only_flagged_removed": view(True, True, False)[0],
    }
    _, errors = view(True, False, False)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "model": f"{finder.model_name}@{finder.model_revision}",
        "texts": len(data),
        "gold_organizations": sum(map(len, gold)),
        "views": views,
    }
    (OUT / "ner_metrics.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (OUT / "ner_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "review_order",
                "missed (in the owner's labels, not found)",
                "extra (found, not in the owner's labels)",
            ]
        )
        for row, missed, extra in errors:
            if missed or extra:
                writer.writerow([row, "; ".join(missed), "; ".join(extra)])
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
