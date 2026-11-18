# SPEC rows removed by T-220 PR B (2026-11-18), verbatim

The requirements of the v1 chain whose code was deleted. Each row is as it stood in SPEC.md. Replaced, not annotated (INDEX rule 3).

## FR-001

| **FR-001** | The v1 sentiment stage (`gamug/FinBERT-financial-news`, chunk-level, entity-scoped; one `article_sentiment` row per article): the full text is in `archive/PIPELINE-superseded-2026-11-10.md`. **Superseded for the new pipeline (D39, FR-020)** except for `sector` summaries (D39 (iv)): a step of `news_nlp_steps.py`, revision in `pipeline_step_state`, rows `sentiment_checked = false` until T-170 adds sector texts. | After a pipeline run, every article with SOURCE text has an `article_sentiment` row with `label ∈ {positive, negative, neutral}`, `score`/`positive`/`negative`/`neutral` ∈ `[0, 1]`, and `model_name`/`processed_at` populated. |

## FR-002

| **FR-002** | The NER stage runs `gamug/sec-bert-finer-ord-ner`, merges BIO-tagged sub-token predictions **word-boundary aware**, and writes one `article_entities` row per detected span with a char offset and confidence. | No emitted span starts or ends mid-word (regression test for the fixed "3"-as-`ORG` subword-fragmentation bug, `82c5e6a`); every row has `entity_type ∈ {PER, LOC, ORG}`, valid `start_char < end_char` into the source text, and a `score`. |

## FR-003

| **FR-003** | The category stage classifies each article via two-level hierarchical zero-shot NLI (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`) against the fixed taxonomy in `news_nlp.taxonomy` (3 groups → 9 leaf labels + `other`), on the title + lead chunk only, and writes one `article_category` row. | `label` is one of the 9 taxonomy slugs or `other`; `group_label`/`group_score` are populated even when `label = other`; all 9 raw NLI distribution columns are populated (audit trail for threshold retuning); `label = other` iff the winning slug's score is below `CATEGORY_CONFIDENCE_THRESHOLD`. |

## FR-004

| **FR-004** | The v1 `c_summary` stage (opt-in `--summarize`, `sshleifer/distilbart-cnn-12-6`, gated on a sentiment row and an entity > 0.8; writes `article_summary`): the full text is in `archive/PIPELINE-superseded-2026-11-10.md`. **Superseded for the new pipeline (D39, D40, FR-020):** a summary per item over the size, no entity gate. | A normal (non-`--summarize`) pipeline run leaves `article_summary` untouched; an article failing the gate never gets a row even under `--summarize`; `num_chunks ≥ 1`. |

## FR-005

| **FR-005** | The `sector_summary` stage (opt-in) composes one row per `(gics_sector, gics_sub_industry, week_start)` for each **closed** calendar week, deterministically concatenating member companies' `c_summary` text attributed by ticker. The stats-only intro sentence (`build_sector_intro_seed`) is itself fully deterministic as of 2026-09-14 (previously run through a model as a "paraphrase" step — dropped after that step was found to hallucinate on 42-50% of rows; see `docs/evaluation.md`'s 2026-09-14 follow-up) — the entire `sector_summary` stage is now non-generative, no model or GPU involved. | `build_sector_intro_seed`'s output never contains a ticker, company name, or summary text (unit-testable — cross-company blending is structurally impossible, not just avoided); `UNIQUE(gics_sector, gics_sub_industry, week_start)` holds; `facts_json` carries the structured (non-narrative) aggregate; `run_sector_summary_stage` never calls `AutoTokenizer`/`AutoModelForSeq2SeqLM.from_pretrained` (test-asserted). |

## FR-008

| **FR-008** | A FastAPI service (`apps/news_nlp_api.py`, `:8003`) exposes pipeline trigger/status, article/category/entity/sentiment read + stats endpoints, `sector_summary` retrieval, the latest eval summary, and human-correction `PATCH`/`DELETE` endpoints for sentiment/entities/category — all working off the RESULTS store alone. | `GET /articles/{id}`, `/stats/*`, `/sectors/summary`, `/eval/latest` all return `200` with only `$DATABASE_URL` configured (no `$SOURCE_DATABASE_URL`) once a pipeline run has already populated RESULTS. |

## FR-009

| **FR-009** | A human can correct a stage's output (`news_nlp.corrections`: `update_*`/`delete_*` for sentiment, entity, category) and that correction survives a subsequent pipeline run on the same article. | After `update_category` changes a label and a pipeline rerun completes, the row's `label` is unchanged (the article is no longer "pending" for that stage — `fetch_pending_category_articles` selects on row absence, not staleness); `delete_category` makes the article eligible for reprocessing again. |

## FR-010

| **FR-010** | `news_nlp.eval` samples processed articles per stage, gets an LLM-as-judge verdict per sampled row, and records both an `eval_run` summary (aggregate metrics, MLflow run ID) and per-row `eval_judgement` rows. | `uv run cli/news_nlp_eval.py --stage <s> --sample-size N` completes without GPU or a labelled corpus, produces exactly one new `eval_run` row with `status = ok` and `metrics_json` populated, `N` (or fewer, if population is smaller) `eval_judgement` rows, and one MLflow run under `news_nlp_eval/<stage>`. |

## FR-012

| **FR-012** | `sector_summary` (heuristic, deterministic, no model — FR-005) is fully decoupled from the FTI hierarchy in its own dedicated module, including the orchestration entrypoint itself (not just the composition/query helpers already under `news_nlp/sector_summary/` as of 2026-09-14), since it has no Feature/Train/Inference shape to share with the four ML stages. | `run_sector_summary_stage` (or its renamed equivalent) lives under `news_nlp/sector_summary/`, not `pipeline.py`; nothing in that module imports from the FR-011 FTI base classes; `pipeline.run_pipeline` calls into it the same way it calls the four FTI-based stages, so the external orchestration contract (FR-006) is unchanged. |

## FR-013

| **FR-013** | A redesigned `news_nlp.eval` module reuses the FR-011 Inference classes for its own model-scoring step (sampling a real article, running the production (or a candidate) model over it) instead of duplicating model-loading/prediction logic independently, as the current `news_nlp/eval/` package's relationship to `pipeline.py` does today. | The eval module's inference step for a given stage calls the same `Inference` subclass `pipeline.py` uses for that stage (same class, parameterizable model/revision — not a re-implementation with its own `from_pretrained`/forward-pass code); a code-search for a second, independent `AutoModelFor*.from_pretrained` call site duplicating an FTI `Inference` class's own model-loading returns none inside `news_nlp/eval/`. |

## FR-017

| **FR-017** | A single JSON file (`ExperimentSpec`, schema-validated) fully specifies one model experiment for any of the four ML stages and one command runs it end to end (train if requested, evaluate with `run_eval` unchanged, optionally publish); the config cannot describe an ambiguous or unreproducible run; `pretrain` is rejected for `c_summary` and for category (the category model of T-174 is trained by the owner's scripts, outside `ExperimentSpec`, PLAN D41). | `uv run cli/run_experiment.py --config <path>.json` reproduces a backfilled historical experiment; an invalid spec fails validation before any work, naming what is wrong; a spec exists for every real historical experiment with a runnable equivalent (two gaps named in `experiments/README.md`). Full text: `archive/archive/SPEC-TASKS-condensed-for-the-category-step-2026-11-13.md` § FR-017. |

