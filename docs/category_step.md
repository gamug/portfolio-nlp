# The category step (`--step category`, T-168; provisional)

One of the 14 labels of D15 (`docs/category-taxonomy.md`) and its group, for each **main company** of each text. It is built
**provisionally with the pinned zero-shot model** so the next team can read a category; the trained model of T-174 replaces it
by changing the pin (it is then another revision of the step; the rows of this one stay).

## Run it

```
uv run cli/news_nlp_steps.py --step category --summary-revision summary-... --companies-revision companies-... --limit 500
uv run cli/news_nlp_steps.py --step category --status
```

`--step summary --status` and `--step companies --status` list the revisions that ran; both are named, never the latest.
`--source-db` and `--results-db` (or `SOURCE_DATABASE_URL` and `DATABASE_URL`) name the two stores, as for the other steps. A rerun
adds nothing: an item already done or skipped at the revision is not read again, a failed one is retried. The step is resumable,
so a long run can be cut with `--limit` and continued; it prints the plan, the model and the revision before it starts, and a
warning if some main-company items have no text at the summary revision named (the companies step read another one).

To run the step again at one revision, or after a re-cut of the sub-news, the SQL is in `--help` (`RECATEGORY_SQL`, `RECUT_SQL`).

## What it reads and writes

* **Reads:** the texts of step 2 (`news_summary` at `--summary-revision`) of every item that has a **main** company at
  `--companies-revision` (`news_company.is_main = 1`). An item on the review list of step 3 has no main company and gets no category.
* **Writes:** `news_category`, one row per (item, main ticker, revision) (no column was added): `label`, `score` (the entailment
  probability of the label), `group_label`, `group_score` (the share of the probability mass the group's labels hold, 0 to 1),
  `slug_scores` (JSON, the probability of each of the 14 labels), `model_name`. The state of the item is in `pipeline_step_state`
  (`step = 'category'`).

## The model (provisional)

`MoritzLaurer/deberta-v3-base-zeroshot-v2.0` at the pin of `config/pipeline_models.json` (`category`), untouched. For each pair the
premise is `<company>: <text>` (the first 2,000 characters) and there is one hypothesis per label,
`The news about <company> is about <definition>.`; the best entailment probability gives the label, and the group is the group of that
label (option B, owner 2026-11-13: `results_capital`, `market_view`, `deals_alliances`, `product_people`,
`governance_legal_reputation`, `other`).

## How good it is (read before using the labels)

Agreement with the owner's labels (`scripts/category_baseline_2026_11_13.py`; held-out scored once):

| | Train rows (225) | Held-out single rows (135) |
|---|---|---|
| label | 56.4% | 43.0% |
| group | 62.7% | 55.6% |

The bar of D24 is **80%**: the labels are a first version, not the accepted ones. They are reliable for `earnings_performance`
(recall about 95%) and weak for the rare labels (`capital_shareholder_returns`, `competitive_actions`, `peer_comparison`, `other`
are almost never found). The held-out 200 also has 44 narrative multi bodies that are not scored. The group is the better field to rely on.

A fine-tune of the same model (`scripts/category_train_cv_2026_11_13.py`, the mixed recipe) reached 62.2% out-of-fold on the train rows
and 48.9% on the held-out single rows: better than the untouched model, still under the bar; it is not published and the step does not use it.

## Edge cases

A blank text is not classified: the item is `skipped` in `pipeline_step_state` with the reason `blank text`, and is not retried at this
revision. A ticker that is not in the universe is read as its own name (counted as `name_missing` in the run output). The `batch_size`
of `config/pipeline_models.json` (`category`) belongs to the v1 stage and is not read here: the model scores one pair at a time,
14 hypotheses each, and the step stores four items per transaction.

## Metrics

D37 records Precision, Recall and ROC OVR per class. `scripts/category_roc_ovr_2026_11_13.py` gives the ROC OVR from saved scores
(the held-out scoring of the mixed fine-tune, or the out-of-fold scores of a CV run, which `category_train_cv_2026_11_13.py` now saves).
The baseline scripts of the untouched model report accuracy and per-class recall; they do not save scores.

## Not done

No threshold or review state: every pair gets a label (the owner decides what a low score means). No sector summary (it has no company).
Sentiment (step 6) is a separate step.
