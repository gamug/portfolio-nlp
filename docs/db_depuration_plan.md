# Plan: depurate the NLP database (T-220), after a new FastAPI (T-219)

**Status (2026-11-17): a plan, nothing is built.** This document and the two tasks in `.specify/memory/TASKS.md` are the whole of the
planning PR: no code, no test, no schema change, no access to any database. The owner decided the points under "Decided"; the
points under "Left open" are gates the work answers before anything is deleted.

## Why

The RESULTS database still holds the tables of the **v1 chain** (`article_sentiment`, `article_entities`, `article_category`,
`article_summary`, `sector_summary`) beside the tables of the **stepped pipeline** (scope to the period snapshot) and the eval/MLflow
tables. The owner wants the database, and the code and tests that need it, reduced to:

1. the stepped NLP pipeline, from the scope step to the period snapshot;
2. the MLflow/eval feature;
3. the admin tables of both (none exist apart from the lean `articles` row, see below);
4. the tests that need the kept code (a chore: delete or adapt the tests of the deprecated code).

The owner keeps a backup of the RESULTS database in **`nlp_backup.db`** (the owner's file: it is not in the repository and no document
names it). It is the safety net of every destructive step below.

## Decided (owner, 2026-11-17)

* **No reader outside this repository uses the v1 `article_*` tables.** `docs/modules/news-nlp.md` and `docs/db-topology.md` still say
  that `portfolio_common.news_export.fetch_processed_articles` reads them; that package is not installed here, so this repository
  cannot check it. The owner's answer replaces that statement. The delivery note for the portfolio-financial-analysis team
  (`snapshow_delivery.md`, repository root) names only `news_snapshot`, `period_run`, `period_summary`, `period_item` and `sector`.
* **Eval is rewired to the `news_*` tables** (not kept on `article_*`).
* **A new FastAPI comes first** (T-219): it serves each step, the whole pipeline and the FTI services. The v1 API
  (`apps/news_nlp_api.py`) reads only v1 tables and the Docker image serves it (`Dockerfile:106-109`), so the v1 tables cannot go
  while it exists. The depuration (T-220) starts after T-219 is merged.
* **`snapshow_delivery.md` stays untracked** (the owner's local file). The golden check below uses a copy of its queries inside the tests.
* **Old revisions are pruned**, keeping exactly: `scope-nli-8ff21e50796b`, `units5-d3e2cbfa830e`, `summary-llm-70cb6bb07292`,
  `companies-f349536cf701`, `sentiment-59863d15fc42`, `category-a23569dba86f`, `snapshot-c3a753affa5d`, `period-3b07b83ecec3`.
* The last bullet of the request, "chore code-tests", is read as: tests that only exercise deprecated code or tables are deleted;
  mixed tests are adapted to the kept tables; `tests/sets.py` is kept in step.

## What the code says today (read on 2026-11-17, master after #205)

### The tables (`src/news_nlp/schema.py`; no views, no triggers, no other DDL in `src/`, `cli/`, `apps/`)

| Layer | Tables | Verdict |
|---|---|---|
| Stepped pipeline | `body_scope`, `body_sector`, `sub_news`, `news_summary`, `news_summary_call`, `news_company`, `news_sentiment`, `news_category`, `news_snapshot`, `period_run`, `period_summary`, `period_item`, `sector`, `company_sector`, `pipeline_step_state` | **keep** |
| Eval / MLflow | `eval_run`, `eval_inference`, `eval_verdict`, `eval_confusion` | **keep** |
| Lean `articles` row (RESULTS) | not created by `schema.py`: `db._ensure_article_row` (`db.py:285`) copies it from the SOURCE; `steps.record_cut` calls it (`steps.py:720`); `sub_news.article_id REFERENCES articles(id)` (`schema.py:306`) | **keep** (the foreign-key parent of `sub_news`) |
| v1 chain | `article_sentiment`, `article_entities` (with the unused `sub_news_id` and `ticker` columns), `article_category` (with `group_label`, `group_score`), `article_summary`, `sector_summary` (with `format_version`, `facts_json`, `intro_text`) | **drop** (gated) |
| Legacy eval | `eval_judgement` (superseded 2026-09-18, still named in `eval/store.py` and two scripts) | **drop if unused** (to confirm) |

There are **no admin tables**: `corrections.py` only updates and deletes in the three v1 tables, so it goes with them. The
`Allowlist` of `db.py` is a schema-name guard, not a table. The v1 migrations in `init_schema` (`schema.py:499-575`) go with the tables.

### No v2 step reads a v1 results table

The stepped pipeline never reads `article_*`, `sector_summary` or `corrections` (the only mention is a docstring,
`news_category_stage.py:16`). Its couplings to the old layer are elsewhere, and they decide how the work is ordered:

1. **Hashed shared code.** A step's revision is a hash of the source of the code it runs, and some of that code lives in modules of the
   v1 era: `news_summary_stage.py:35-42,65,113` imports and hashes `summary_stage` and `chunking`; `news_company_stage.py:36,202-213`
   imports and hashes `ner_stage`; `news_sentiment_stage.py` hashes `chunking` and `sentiment_pair`. **Editing any of these files, even
   only to delete a v1 class inside it, changes the revision of a v2 step**: the finished items become pending again (a rerun of
   hours) and the delivered revisions stop matching. **They stay byte-identical.** The v1 classes inside them stay (dead code, named in
   the docs). `src/fti.py` is also shared.
2. **Not v1 although they look like it:** `news_nlp/taxonomy.py` (imported by `retag.py`, which `cut_stage` uses, and by eval),
   `category_multilabel.py`, `sentiment_pair*.py`, `news_summary_llm.py`, `train_ner.py`, `train_sentiment.py`. They stay.
3. **The SOURCE `articles` table** is read directly by `steps.py:65`, `cut_stage.py:42`, `news_snapshot_stage.py:354`. It is not touched.

### Eval depends on the v1 tables today

`eval/sampling.py:54-60,267-358,411` samples from `article_*` and `sector_summary`; `eval/candidate.py:36-40` imports the FTI classes
of `category_stage`, `ner_stage`, `sentiment_stage`, `summary_stage`; `eval/calibration_sample.py:134-135` joins `article_category`;
`eval/__init__`, `runner`, `cli/news_nlp_eval.py`, `src/experiment.py` and `.github/workflows/eval.yml` follow.
`eval/step_category.py` is already shaped for step 7. This is why the eval rewire (PR A) comes before the removal of the v1 code.

### v1 code with no kept importer once eval is rewired

`src/pipeline.py` (importers: the API, `cli/news_nlp_cli.py`, `src/setup.py:15`, tests), `pipeline_config.py`,
`apps/news_nlp_api.py`, `cli/news_nlp_cli.py`, `news_nlp/queries.py`, `corrections.py`, `news_nlp/sector_summary/*`,
`category_stage.py`, `sentiment_stage.py`, the re-exports in `news_nlp/__init__.py`, and `config/pipeline_models.json` if step 7 does
not read it (FR-018 says the category step reads its `category` pin: to verify before deleting).

### Tests (grouped; the next PRs confirm each file by reading it)

* **Only v1, to delete:** `test_pipeline_run`, `test_pipeline_progress`, `test_pipeline_model_config`, `test_pipeline_endpoints`,
  `test_query_endpoints`, `test_correction_endpoints`, `test_sentiment_pipeline`, `test_ner_pipeline`, `test_summary_pipeline`,
  `test_category_pipeline`, `test_corrections`, `test_queries`, `test_sector_summary`, `test_two_tier`.
* **Mixed, to adapt:** `test_eval_sampling`, `test_eval_candidate`, `test_eval_runner`, `test_eval_calibration*`,
  `test_experiment_run`, `test_category_step`, `test_schema` (lists both families of tables), `test_setup`, `test_db`,
  `test_cut_summaries`, `test_company_news_counts`, and the API fixture in `tests/news_nlp/conftest.py:108-119`.
* **Kept as they are:** every test of the steps, the eval store/judges/metrics/partition/tracking, attribution, gold and labelling.
* `tests/news_nlp/test_sets.py` fails when a test file is missing from `tests/sets.py` or a name there is not a file: every deletion
  edits `tests/sets.py`.

### The rules that bind the work

* `constitution.md:147` and PLAN D44 say the v1 chain is unchanged: a **constitution amendment** (version bump) and a new PLAN
  decision are part of the work.
* SPEC.md:575: a schema change needs the owner's approval (this plan asks for it). SPEC.md:330-336: migrations are additive; dropping
  tables is a deliberate exception, recorded.
* **T-131** (open: move aside the results of quarantined pairs, copying them to `<table>_removed` first, "never rename a whole table")
  and the "old tables to v1" part of **T-150** (open, with the `portfolio_common.news_export` contract update) both concern the v1
  tables. If the v1 tables are dropped, T-131 is moot for them and that part of T-150 is superseded; the owner confirms when T-220
  closes. The **rule 16** of the constitution (an article that was fetched with a title and a body is never deleted) is not touched:
  only derived results are dropped; the SOURCE rows and the lean `articles` row stay.
* `docs/db-topology.md:183-187`: a live SQLite file is copied only with the SQLite backup API, never by copying the `.db` and its `-wal`.
* `tests/test_memory_size.py`: limits on `TASKS.md`, `PLAN.md`, `SPEC.md`; an entry of at most 4,000 characters; no checked task in
  `TASKS.md`; every file of `.specify/memory/archive/` named in `INDEX.md`. Text moves out to `CHANGELOG.md` or `archive/` verbatim.
* `CLAUDE.md` is git-ignored and must never be deleted.

## The work, in order

**Rule for all of it:** no PR touches the live database. Every destructive step is a script that defaults to a **dry run**, runs first
on a **copy of `nlp_backup.db`**, and is applied to the live file only by the owner. The PRs are separate (one development each).

### T-219 first: the new FastAPI (its own task, its own PRs)

Replaces `apps/news_nlp_api.py`. Proposed scope, for the owner to settle at its start: an endpoint to run and to read the status of each
step (scope, cut, summary, companies, category, sentiment, snapshot, period, sector: the functions `cli/news_nlp_steps.py` calls), one
to run the whole pipeline in order, and the FTI services (the feature, train and inference of the stages that have them). Revisions
are explicit and never "latest" (as the delivery note says). Long runs are background jobs with a status. The Docker image and
`docker.yml` (PR #205) are retargeted to it. Open for its design: authentication, one GPU (jobs are serialised), what the FTI services
expose, the port. Until it is merged the v1 API keeps working.

### T-220: the depuration (after T-219)

**Phase 0, gates (no code).**

1. **Backup verified.** `nlp_backup.db` is a copy made with the SQLite backup API; `PRAGMA integrity_check` returns `ok`; the row count
   of every table equals the live file's; its SHA-256 and the counts are recorded in the PR. Work on a *second copy* of it.
2. **Owner approvals** recorded: the drop of the v1 tables (SPEC.md:575), the constitution amendment and the new PLAN decision.
3. **The verified backup replaces the `<table>_removed` copies of T-131.** Those would add tables to a database that must hold only
   kept ones. The owner accepts this or says otherwise.
4. **The two keep-list gates** (see "Left open").
5. **Order:** after #199 and #204 (both merged on 2026-11-16/17): the guard below pins the revisions as they are on master.

**PR A: the guard, then the eval rewire.**

* A **revision guard test** first: it pins the revision of every step (the code hashes) and the inputs of the delivered snapshot and
  period revisions. Any later edit that changes a revision fails CI. This is the protection of "do not break what is built".
* A **golden check**: the SQL of `snapshow_delivery.md` (a copy in the tests) runs on a fixture, and the same SQL runs on the backup
  copy before and after each destructive step and must return identical rows.
* Then eval is rewired to `news_*`: sampling from `news_sentiment`, `news_category`, `news_company`, `news_summary`; scoring with
  the v2 Inference classes; the `sector_summary` stage and the v1 zero-shot category stage leave `STAGES`;
  `.github/workflows/eval.yml`, `cli/news_nlp_eval.py` and `docs/evaluation.md` follow. **Design point to settle first:** the key of
  `eval_inference` is (article, task, experiment) and the v2 key is (item_kind, item_id, ticker). A sub-news maps to its article
  through `sub_news`; the mapping must keep the stored eval history readable (additive columns only, SPEC.md:330-336).

**PR B: remove the v1 code and its tests.** Delete the files listed above, rewire `src/setup.py` (it imports constants from
`pipeline`), clean `news_nlp/__init__.py`, delete the only-v1 tests, adapt the mixed ones, update `tests/sets.py`. The hashed shared
modules stay byte-identical (the guard proves it). The Docker retarget was done in T-219.

**PR C: the schema, the drop and the prune.**

* `schema.py`: remove the v1 DDL and its migrations; a test lists exactly the tables an empty database gets (the kept ones).
* `scripts/depurate_db_<date>.py`, dry run by default:
  * `--apply` refuses without `--backup nlp_backup.db` verified (integrity and counts); one transaction; `DROP TABLE` of the v1
    tables (and `eval_judgement` if confirmed unused); then `PRAGMA foreign_key_check`, `PRAGMA integrity_check`, the row counts of
    every kept table, and the counts of the delivered revisions must equal the "before"; the golden SQL must return identical rows.
  * `--apply-prune` (its own dry run): the **keep list is a file in the repository** (the eight revisions above, plus what the gates
    add). The tables with a revision column (`body_scope`, `body_sector`, `sub_news`, `news_summary`, `news_summary_call`,
    `news_company`, `news_sentiment`, `news_category`, `news_snapshot`, `period_run`, `period_summary`, `period_item`,
    `pipeline_step_state`) lose the rows of any other revision. The dry run prints, per table and revision, the rows to delete and
    **refuses** when a kept row refers to a pruned one: a kept `news_*` row whose `sub_news` id exists only in a pruned cut revision;
    a kept period whose snapshot revision is not kept; a revision named in the `inputs` of a kept snapshot row that is not kept.
    Deletes follow the dependency order (`period_item`, `period_summary`, `period_run`, then the rest), one transaction per table, then
    the same checks, then `VACUUM` on the copy.
* Docs: `docs/db-topology.md`, `docs/modules/news-nlp.md`, `docs/legacy/README.md`, README, SPEC (FR-001 to FR-005 and the schema
  list), PLAN (a new decision), constitution (amendment, version), `CLAUDE.md` by hand (local file, never deleted); replaced text goes
  to `archive/` verbatim; both architecture artifacts are reconciled at the close of the development (not now: nothing is built).

**Acceptance of T-220:** the revision guard passes unchanged; the golden SQL returns identical rows on the backup copy before and
after; `init_schema` on an empty file creates exactly the kept tables; `--step <step> --status` on the backup copy shows the same
counts; ruff, mypy and the full suite with `--group eval` pass; the live database is touched only by the owner, after a dry run.

## Left open (gates; none blocks this planning PR)

1. **The period revision.** The delivery note names `period-ae014b56b055`; the keep list says `period-3b07b83ecec3`. PR #204 (merged
   2026-11-17) changed the period revision, so the second may be the one written after it, but that is not verified. The script never
   prunes a revision named in the delivery note unless the owner removes it from the note's list explicitly.
2. **No sector-step revision in the keep list.** The kept snapshot's `inputs` names one (the `body_sector` step). The script reads
   every revision named in `inputs` of the kept snapshot rows and in `period_run.snapshot_revision` of the kept period, and refuses to
   prune unless all of them are in the keep list.
3. **`eval_judgement`:** confirm unused (read the code in PR A/B).
4. **`config/pipeline_models.json`:** is it read by step 7 (FR-018)? Read in PR B.
5. **The eval key mapping** (PR A) and **T-219's design** (authentication, jobs, the FTI services): settled at the start of each.
6. **Open tasks that overlap:** T-131 and part of T-150 (above); T-218 (testing the Docker image) is affected by the retarget in T-219.
