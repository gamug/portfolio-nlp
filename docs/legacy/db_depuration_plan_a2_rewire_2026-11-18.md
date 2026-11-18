# docs/db_depuration_plan.md: text replaced on 2026-11-18, verbatim

The parts of the plan that the owner's decision of 2026-11-18 (delete the old eval stages; the three revision gates answered) made false. Replaced, not annotated.

## PR A paragraph (guard, golden check, eval rewire)

```
**PR A: the guard, then the eval rewire.** *(Split, 2026-11-18: **A1**, the guard and the golden check, is built (`tests/news_nlp/test_revision_guard.py`, `src/delivery_golden.py`, `tests/news_nlp/test_delivery_golden.py`); **A2** is the eval rewire. The guard found that the cut revision hashes the **bytes** of five modules (`attribution_model`, `chunking`, `company_texts`, `triage`, `attribution_units`): a comment moves it, so PR B must not touch those files at all. The snapshot revision computed from the code is `snapshot-c3a753affa5d`, the delivered one; the period revision also hashes the sector mapping, so it is not reproducible without the database (see the open point on `period-ae014b56b055`).)*

* A **revision guard test** first: it pins the revision of every step (the code hashes) and the inputs of the delivered snapshot and
  period revisions. Any later edit that changes a revision fails CI. This is the protection of "do not break what is built".
* A **golden check**: the SQL of `snapshow_delivery.md` (a copy in the tests) runs on a fixture, and the same SQL runs on the backup
  copy before and after each destructive step and must return identical rows.
* Then eval is rewired to `news_*`: sampling from `news_sentiment`, `news_category`, `news_company`, `news_summary`; scoring with
  the v2 Inference classes; the `sector_summary` stage and the v1 zero-shot category stage leave `STAGES`;
  `.github/workflows/eval.yml`, `cli/news_nlp_eval.py` and `docs/evaluation.md` follow. **Design point to settle first:** the key of
  `eval_inference` is (article, task, experiment) and the v2 key is (item_kind, item_id, ticker). A sub-news maps to its article
  through `sub_news`; the mapping must keep the stored eval history readable (additive columns only, SPEC.md:330-336).


```

## A2 section (the rewire design)

```
## A2, the eval rewire: what reading the code and the backup showed (2026-11-18)

The rewire is a redesign of the sampling, the judges and the metrics, not a swap of table names. Facts, then the defaults taken for the owner's
"go" (the owner can change any of them before A2a is built).

**Facts**

* `eval/sampling.py` is v1 throughout: the low-confidence and target strata read `article_sentiment.score`, the nine slug columns of `article_category`
  (`CATEGORY_SLUGS`, the 10-label v1 taxonomy), the entities of `article_entities`; `_text` reads `articles.body_text` by article id.
* The v2 rows are keyed by `(item_kind, item_id, ticker, model_revision)`. In the kept revisions every item has exactly one main company, so one row per
  item: `news_sentiment` and `news_category` 226,139 each (66,738 `body`, 159,401 `sub_news`), `news_company` 363,939 rows of which 226,139 main.
* **A `body` item is its body hash, not an article id** (only 34,960 of the 66,738 body ids are also an `articles.id`). A `sub_news` item is an
  integer id, and 159,401 of them come from **28,799 articles** (5.5 pairs per article). `news_snapshot_stage.articles_of` already maps an item to its
  lowest article id by hashing the corpus once; the eval must use it (for the sampled items only) rather than a second mapping.
* `judges.judge_category` and `CategoryVerdict` know the 10 v1 slugs; `news_category.label` has 12. The blind step-7 judge exists
  (`eval/step_category.py`, `config`-free prompt `category_step.md`, `STEP_LABELS`).
* `metrics.py` (category: `CATEGORY_SLUGS`, per-slug target accuracy), `calibration.py`, `calibration_sample.py`, `calibration_judge.py` and `partition.py`
  (about 2,300 lines and their tests) belong to the **v1 category model** (thresholds of the hierarchical zero-shot classifier; `calibration_sample.py:134` joins
  `article_category`). They do not measure the entailment model of step 7.
* `candidate.py` and `src/experiment.py` import the v1 FTI classes of the four v1 stages (A2b).

**Defaults (A2a), to be confirmed or changed**

1. **Stages:** `sentiment`, `category`, `c_summary`; `ner` and `sector_summary` leave `STAGES` (the history in `eval_run` stays; their prompts, verdicts and
   aggregators go in PR B with the v1 code, not before).
2. **Unit and key:** the unit is the item's main pair (the item's own row of `news_sentiment` / `news_category`; one per item). `eval_inference.article_id`
   keeps its meaning (an `articles.id`) and is the item's lowest article id through `articles_of`; when two sampled items share an article, the
   draw keeps one per article (lowest item id), so `UNIQUE (article_id, task, experiment, run_id)` holds and **no schema change is needed**. The item's
   own key (`item_kind`, `item_id`, `ticker`) and the revision go into the stored `prediction_json`, so a verdict can be traced to the row.
3. **Revision:** one revision per table is evaluated, `--revision` to name it; by default the one with the latest `processed_at`. Never a mix.
4. **Category** is judged blind with `step_category` (12 labels); `CategoryVerdict` and `judge_category` leave the `category` stage. **Sentiment** is judged
   on the main pair: the text the step read and the company, the prompt naming the company. **c_summary** reads `news_summary` (the text of the item) with
   the judge it has today.
5. **Calibration** (`calibration*.py`, `partition.py`) is **not rewired**: it is the v1 category model's. It is removed with the v1 code in PR B (the owner
   confirms), which is why `article_category` cannot be dropped before PR B.
6. **A2a** = sampling, the judges' wiring, the runner and the store, with their tests; **A2b** = `candidate.py`, `src/experiment.py`, `cli/news_nlp_eval.py`,
   `.github/workflows/eval.yml`, `docs/evaluation.md`.


```

## Phase 0 listing, item 3 closing sentence

```
The owner decides: keep `sentiment-5999fe131d8f` and `category-65f33b55b26b`, or rerun the snapshot step on
   the kept revisions (a new delivery), or accept the dangling provenance in `inputs`.
```

## Left open, items 2 and 3 (period and sentiment revision)

```
2. **The period revision.** The delivery note names `period-ae014b56b055`; the keep list says `period-3b07b83ecec3`. #204 changed
   the period revision, so the second may be the one written after it, but that is not verified: the Phase 0 listing settles it. The
   script never prunes a revision named in the delivery note unless the owner removes it from the note's list explicitly.
3. **The sentiment revision** (above): `sentiment-5999fe131d8f` documented, `sentiment-59863d15fc42` in the list.

```

## Three more sentences (Decided, the facts section, a heading)

```
* **Eval is rewired to the `news_*` tables** (not kept on `article_*`).
```

```
This is why the eval rewire (PR A) comes before the removal of 
```

```
### v1 code with no kept importer once eval is rewired
```

