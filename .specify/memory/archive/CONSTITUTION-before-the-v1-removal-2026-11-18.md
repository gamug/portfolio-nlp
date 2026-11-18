# Constitution text replaced by T-220 PR B (2026-11-18), verbatim

Version 2.13.0 text, before the v1 chain's code was removed. Replaced, not annotated (INDEX rule 3). The new text is in `constitution.md` 3.0.0.

## lines 22-26 (Principle: stack)

```
   Chosen for GPU/CPU portability (`pipeline.py`'s `DEVICE` falls back to CPU
   automatically) and because all four models it runs
   (`ProsusAI/finbert`
```

## lines 55-62 (src layout)

```
1. **`src/` is flat, plus one local package.** Pipeline stages and utilities
   are single files directly under `src/` (`pipeline.py`, `chunking.py`,
   `setup.py`, `train_ner.py`); `news_nlp/` is the one local package (the
   vendored DB layer: connection, schema, queries, corrections, taxonomy,
   `sector_summary`, plus `news_nlp/eval/` for LLM-as-judge evaluation). A
```

## lines 90-94 (.env loading)

```
   loaded by the entrypoint, never by a library module: `pipeline.py`
   calls `load_dotenv()` at import (so everything that imports it is
   covered), `news_nlp.eval.config.EvalSettings.load` loads it for the
```

## lines 100-104 (naming)

```
   `article_sentiment`, NER -> `article_entities`, category ->
   `article_category`, summarization -> `article_summary`/
   `sector_summary`) rather than inventing a new term for the same concept.
```

## AI behavior 4 and 5

```
4. **Outputs are decision-support, not authoritative claims.** Sentiment,
   entity, category, and summary outputs are model predictions over
   third-party news text — nothing in the pipeline or API should present
   them as verified fact, financial advice, or ground truth. Corrections
   (`news_nlp.corrections`) exist precisely because model output is
   expected to need human override.
5. **Accuracy claims are backed by `news_nlp.eval`, not spot-checks.** A
   change asserted to "improve" a stage must cite an LLM-as-judge eval run
   (tracked in MLflow) against the recorded baseline
   (`docs/evaluation.md`), not a handful of manually inspected examples.
```

## CI rule 2 (eval.yml)

```
2. **`eval.yml` is a separate, non-blocking workflow** for the real
   LLM-as-judge run — it is not part of the merge gate for ordinary PRs.
```

## constitution, AI behavior #1 (the eval run required for a model swap)

```
it needs an eval run (`news_nlp.eval`) comparing
   against the current baseline before it replaces the pinned checkpoint.
```

## constitution, AI behavior #12, the accuracy_ovr naming reference

```
same formula/naming as
    `news_nlp.eval.metrics.aggregate_category`'s `accuracy_ovr_<slug>`, kept
    consistent across every stage rather than invented fresh per stage).
```

## constitution, AI behavior #12, the reference implementations

```
`train_sentiment.py`'s `make_compute_metrics()` is the
    reference implementation for sentiment; `news_nlp.eval.metrics`'s
    `aggregate_category`/`aggregate_ner` already follow this shape for
    their stages.
```

## constitution, AI behavior #12, the downstream LLM-judge bullet

```
    - **Downstream, LLM-judge eval against real traffic**
      (`news_nlp.eval.metrics.aggregate_sentiment`/`aggregate_category`/
      `aggregate_ner`): per class, precision/recall/F1/`accuracy_ovr` (HT-
      reweighted — the `_naive_pooled` variants exist for diagnostics, not
      as the headline); overall, `agreement_rate` (or the stage's own
      accuracy-equivalent), `macro_f1_vs_judge`, and `mean_severity` where
      the stage's judge produces one. A run recorded before a metric
      existed (an `eval_run` row from before `accuracy_ovr` was added, say)
      can be recomputed from its own stored `eval_judgement` rows through
      the current `aggregate_*` function — no new judge calls needed — the
      same "recompute, don't leave a blank cell" rule as above, and cheaper
      here than for the offline case.
```
