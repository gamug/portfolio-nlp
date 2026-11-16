"""The NER hold-out script (`scripts/ner_eval_2026_11_16.py`): the integrity check of the owner's file, the standardisation of names, the
three treatments of the labels that are not in the text, the scores and the cause of a miss. The model is not loaded: the spans are stubs."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from news_nlp.triage import Company, CompanyResolver

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "ner_eval", ROOT / "scripts" / "ner_eval_2026_11_16.py"
)
assert spec and spec.loader
ne = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ne
spec.loader.exec_module(ne)

UNIVERSE = [
    Company("AAPL", "Apple Inc.", "Information Technology", ""),
    Company("MSFT", "Microsoft Corporation", "Information Technology", ""),
    Company("GOOGL", "Alphabet Inc. (Class A)", "Communication Services", ""),
    Company("META", "Meta Platforms, Inc.", "Communication Services", ""),
]


@pytest.fixture
def resolver() -> CompanyResolver:
    return CompanyResolver(UNIVERSE)


def _frame(rows: list[tuple[str, str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.DataFrame(
        {
            "review_order": range(1, len(rows) + 1),
            "key": [f"k{i}" for i in range(1, len(rows) + 1)],
            "title": [r[0] for r in rows],
            "text": [r[1] for r in rows],
            "organizations": [r[2] for r in rows],
        }
    )
    return data, data[["review_order", "key"]].copy()


def test_prf_and_empty_cases() -> None:
    assert ne.prf(8, 2, 2) == {
        "tp": 8,
        "fp": 2,
        "fn": 2,
        "precision": 0.8,
        "recall": 0.8,
        "f1": 0.8,
    }
    assert ne.prf(0, 0, 0)["f1"] == 0.0


def test_split_labels_trims_and_drops_empty_items() -> None:
    assert ne.split_labels(" Apple, ,Microsoft ,") == ["Apple", "Microsoft"]


def test_canon_maps_aliases_tickers_and_plain_names(resolver: CompanyResolver) -> None:
    canon = ne.make_canon(resolver)
    assert canon("Facebook") == "T:META"  # the alias list
    assert canon("Google") == canon("Alphabet") == "T:GOOGL"
    assert canon("MSFT") == canon("Microsoft") == "T:MSFT"  # a ticker written in capitals
    assert canon("CNBC") == "N:cnbc"  # no ticker: the normalised name
    assert canon("Fed") == canon("Federal Reserve") == "N:federal reserve"
    assert canon("T:GOOG") == "T:GOOGL"  # a ready ticker: a share class is one company


def test_no_alias_list_leaves_the_names_apart(resolver: CompanyResolver) -> None:
    canon = ne.make_canon(resolver, aliases=False)
    assert canon("Facebook") == "N:facebook"
    assert canon("Fed") != canon("Federal Reserve")


def test_integrity_passes_and_reports_the_labels_not_in_the_text(resolver: CompanyResolver) -> None:
    data, key = _frame(
        [
            ("Tech", "Apple and Microsoft rose.", "Apple,Microsoft"),
            ("Politics", "The White House spoke.", "Casa Blanca"),
            (
                "Row",
                "MSFT gained.",
                "Microsoft",
            ),  # a variant of a name the text writes: not flagged
            ("Typo", "IBM and Alphabet rose.", "Alphabe"),  # a truncated name is not a whole word
            (
                "Apple stock",
                "Shares rose.",
                "Apple",
            ),  # written in the title only: the model reads the text
        ]
    )
    report = ne.integrity(data, key, resolver)
    assert report["rows"] == 5 and report["organizations"] == 6
    assert report["labels_not_in_the_text"] == [
        {"row": 2, "label": "Casa Blanca"},
        {"row": 4, "label": "Alphabe"},
        {"row": 5, "label": "Apple"},
    ]
    assert report["rows_with_such_labels"] == 3


@pytest.mark.parametrize(
    ("break_it", "message"),
    [
        (lambda d, k: d.__setitem__("review_order", [1, 1]), "review_order"),
        (lambda d, k: d.__setitem__("key", ["a", "a"]), "not unique"),
        (lambda d, k: d.__setitem__("text", ["x", " "]), "empty text"),
        (lambda d, k: d.__setitem__("organizations", ["x", ""]), "empty organizations"),
        (lambda d, k: k.__setitem__("key", ["other", "k2"]), "key.csv"),
    ],
)
def test_integrity_stops_on_a_structural_failure(
    resolver: CompanyResolver,
    break_it: Callable[[pd.DataFrame, pd.DataFrame], object],
    message: str,
) -> None:
    data, key = _frame([("t", "Apple rose.", "Apple"), ("t", "Microsoft rose.", "Microsoft")])
    break_it(data, key)
    with pytest.raises(ne.IntegrityError, match=message):
        ne.integrity(data, key, resolver)


def test_a_missing_column_is_an_integrity_error(resolver: CompanyResolver) -> None:
    data, key = _frame([("t", "Apple rose.", "Apple")])
    with pytest.raises(ne.IntegrityError, match="columns missing"):
        ne.integrity(data.drop(columns=["title"]), key, resolver)


def test_gold_treatments_as_is_removed_repaired() -> None:
    labels = [["Casa Blanca", "Apple"], ["Japón Aflac"]]
    rows = [17, 18]
    assert ne.gold_names(rows, labels, "as_is") == labels
    assert ne.gold_names(rows, labels, "removed") == [["Apple"], []]
    assert ne.gold_names(rows, labels, "repaired") == [["White House", "Apple"], ["Aflac"]]


def test_check_repairs_needs_every_flagged_label_listed_and_each_repair_in_the_text() -> None:
    data = pd.DataFrame({"review_order": [17], "title": ["t"], "text": ["The White House spoke."]})
    ne.check_repairs(data, [(17, "Casa Blanca")])
    with pytest.raises(ne.IntegrityError, match="does not list"):
        ne.check_repairs(data, [(17, "Another label")])
    bad = pd.DataFrame({"review_order": [17], "title": ["t"], "text": ["Nothing here."]})
    with pytest.raises(ne.IntegrityError, match="not in the text"):
        ne.check_repairs(bad, [(17, "Casa Blanca")])


def test_score_counts_sets_per_text_and_the_universe_view(resolver: CompanyResolver) -> None:
    canon = ne.make_canon(resolver)
    gold = [["Apple", "Microsoft"], ["CNBC"]]
    pred = [["Apple", "Apple", "Facebook"], ["CNBC", "Reuters"]]  # a name twice counts once
    figures, per_text = ne.score(gold, pred, canon)
    assert (figures["tp"], figures["fp"], figures["fn"]) == (2, 2, 1)
    assert per_text[0] == ({"T:MSFT"}, {"T:META"})
    universe, _ = ne.score(gold, pred, canon, universe_only=True)
    assert (universe["tp"], universe["fp"], universe["fn"]) == (1, 1, 1)
    media, _ = ne.score(gold, pred, canon, drop_media=True)
    assert (media["tp"], media["fp"], media["fn"]) == (1, 1, 1)


def test_near_misses_pair_a_name_with_the_one_that_contains_it() -> None:
    per_text = [({"N:solaris energy", "T:AAPL"}, {"N:solaris energy infrastructure", "N:cnbc"})]
    assert ne.near_misses(per_text) == 1


def test_miss_cause_separates_the_ner_from_the_matcher(resolver: CompanyResolver) -> None:
    data, _ = _frame(
        [("t", "Meta Platforms and Apple rose; MSFT too.", "Meta Platforms,Apple,MSFT")]
    )
    canon = ne.make_canon(resolver)

    def span(text: str, start: int) -> dict[str, object]:
        return {
            "entity_type": "ORG",
            "text": text,
            "start_char": start,
            "end_char": start + len(text),
        }

    # Apple tagged; "Meta Platforms" tagged but cut so that the matcher cannot map it; MSFT not tagged at all
    spans = [[span("Apple", 19), span("Meta Platfo", 0)]]
    causes = ne.miss_causes(data, [["Meta Platforms", "Apple", "MSFT"]], spans, canon)
    assert causes == {
        "ner_untagged": ["1:MSFT"],
        "matcher_unmapped": ["1:Meta Platforms"],
        "not_located": [],
    }


def test_miss_cause_looks_at_every_occurrence_of_the_name(resolver: CompanyResolver) -> None:
    """The first occurrence is untagged and the second is tagged (a cut span the matcher cannot map): the NER did tag the name."""
    data, _ = _frame([("t", "Meta Platforms rose. Later Meta Platforms fell.", "Meta Platforms")])
    canon = ne.make_canon(resolver)
    spans = [[{"entity_type": "ORG", "text": "Meta Platfo", "start_char": 27, "end_char": 38}]]
    causes = ne.miss_causes(data, [["Meta Platforms"]], spans, canon)
    assert causes["matcher_unmapped"] == ["1:Meta Platforms"] and causes["ner_untagged"] == []
    assert ne.miss_causes(data, [["Meta Platforms"]], [[]], canon)["ner_untagged"] == [
        "1:Meta Platforms"
    ]


def test_the_repairs_are_counted_as_replaced_kept_and_dropped() -> None:
    flagged = [(17, "Casa Blanca"), (37, "NHTSA"), (26, "Vodafone")]
    assert ne.repairs_summary(flagged) == {
        "labels_not_in_the_text": 3,
        "replaced": 1,
        "kept": 1,
        "dropped": 1,
    }


def test_the_outputs_are_written_with_lf_and_the_columns_the_page_names(
    tmp_path: Path, resolver: CompanyResolver
) -> None:
    data, _ = _frame([("t", "Apple rose, CNBC said.", "Apple")])
    canon = ne.make_canon(resolver)
    errors: list[tuple[set[str], set[str]]] = [(set(), {"N:cnbc"})]
    ne.write_outputs(
        tmp_path, {"views": {}, "model": "m@r"}, errors, [["Apple", "CNBC"]], data, canon
    )
    raw = (tmp_path / "ner_errors.csv").read_bytes()
    assert b"\r" not in raw
    lines = raw.decode().splitlines()
    assert lines[0] == "review_order,missed,extra,extra_spans"
    assert lines[1] == "1,,N:cnbc,N:cnbc=CNBC"
    assert set(json.loads((tmp_path / "ner_metrics.json").read_text())) == {"views", "model"}


def test_run_gives_the_views_and_stops_when_a_label_has_no_repair(
    resolver: CompanyResolver,
) -> None:
    data, key = _frame(
        [("t", "Apple rose, CNBC said.", "Apple,CNBC"), ("t", "Microsoft rose.", "Microsoft")]
    )
    spans = [
        [{"entity_type": "ORG", "text": "Apple", "start_char": 0, "end_char": 5}],
        [{"entity_type": "ORG", "text": "Microsoft", "start_char": 0, "end_char": 9}],
    ]
    result = ne.run(data, key, resolver, spans)
    view = result["views"]["as_is/all_organisations"]
    assert (view["tp"], view["fp"], view["fn"]) == (2, 0, 1)
    assert result["views"]["as_is/universe_only"]["recall"] == 1.0
    assert "repaired/universe_only_ner_plus_text" in result["views"]
    data.loc[0, "organizations"] = (
        "Apple,Casa Blanca"  # not in the text, and REPAIRS does not list it for this row
    )
    with pytest.raises(ne.IntegrityError, match="does not list"):
        ne.run(data, key, resolver, spans)


# ---- the pages quote the committed figures, so a rerun that moves them fails here ----

RESULTS = ROOT / "docs" / "ner_results_2026_11_16" / "ner_metrics.json"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _quoted(metrics: dict[str, object]) -> dict[str, list[str]]:
    views = metrics["views"]
    assert isinstance(views, dict)

    def f(view: str, key: str) -> str:
        return _pct(views[view][key])

    repaired_u, plus = "repaired/universe_only", "repaired/universe_only_ner_plus_text"
    as_is_u, repaired_a = "as_is/universe_only", "repaired/all_organisations"
    return {
        "docs/ner_training.md": [
            f(repaired_u, "precision"),
            f(repaired_u, "recall"),
            f(plus, "recall"),
            f(plus, "precision"),
            f(repaired_a, "precision"),
            f(repaired_a, "recall"),
            f(as_is_u, "precision"),
            f(as_is_u, "recall"),
            f("as_is/all_organisations", "precision"),
            f("as_is/all_organisations", "recall"),
            f("as_is/all_organisations_no_alias_list", "precision"),
            f("as_is/all_organisations_no_alias_list", "recall"),
        ],
        "docs/companies_step.md": [
            f(repaired_u, "precision"),
            f(repaired_u, "recall"),
            f(plus, "recall"),
            f(as_is_u, "precision"),
            f(as_is_u, "recall"),
        ],
        "docs/modules/news-nlp.md": [
            f(repaired_u, "precision"),
            f(repaired_u, "recall"),
            f(plus, "recall"),
            f(as_is_u, "precision"),
        ],
        "docs/evaluation.md": [
            f(repaired_u, "precision"),
            f(repaired_u, "recall"),
            f(plus, "recall"),
            f(as_is_u, "precision"),
            f(as_is_u, "recall"),
        ],
        "docs/gold_datasets.md": [
            f(repaired_u, "precision"),
            f(repaired_u, "recall"),
            f(as_is_u, "precision"),
            f(as_is_u, "recall"),
        ],
        "pipeline_metrics_report.md": [
            f(repaired_u, "precision"),
            f(repaired_u, "recall"),
            f(plus, "recall"),
            f(repaired_a, "precision"),
            f(repaired_a, "recall"),
            f(as_is_u, "precision"),
            f("as_is/all_organisations_no_alias_list", "precision"),
        ],
        "pipeline_metrics_report_es.md": [
            f(repaired_u, "precision"),
            f(repaired_u, "recall"),
            f(plus, "recall"),
            f(repaired_a, "precision"),
            f(repaired_a, "recall"),
            f(as_is_u, "precision"),
            f("as_is/all_organisations_no_alias_list", "precision"),
        ],
    }


def test_the_pages_quote_the_committed_ner_figures() -> None:
    metrics = json.loads(RESULTS.read_text(encoding="utf-8"))
    for doc, figures in _quoted(metrics).items():
        text = (ROOT / doc).read_text(encoding="utf-8").replace("\r\n", "\n")
        for figure in figures:
            assert figure in text, f"{doc}: {figure}"


def test_the_pages_say_what_was_done_with_the_labels_not_in_the_text() -> None:
    repairs = json.loads(RESULTS.read_text(encoding="utf-8"))["repairs"]
    assert (
        repairs["labels_not_in_the_text"],
        repairs["replaced"],
        repairs["kept"],
        repairs["dropped"],
    ) == (14, 9, 1, 4)
    text = (ROOT / "docs" / "ner_training.md").read_text(encoding="utf-8")
    assert "9 replaced, 1 kept, 4 dropped" in text
