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

The **admin tables** (the owner's third bullet) are `sector`, `company_sector` (the owner's sector overrides) and `pipeline_step_state` (the per-item state of every step), all kept. `corrections.py` only updates and deletes in the three v1 tables, so it goes with them. The
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
`category_stage.py`, `sentiment_stage.py` and the re-exports in `news_nlp/__init__.py`. **`config/pipeline_models.json` stays:** step 7
reads it (`news_category_stage.py:72`, validated there; `news_sentiment_stage.py:14` names it), so it is not on the delete list.
`pipeline_config.py` has no importer outside `pipeline.py`, so it does go.

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

* **What says the v1 chain stays** (read on master; these are the lines PR C rewrites): `CLAUDE.md`, "The table above is the **v1 chain**,
  still in production" (a local, git-ignored file: edited by hand, never deleted); `SPEC.md:190` ("The code of today (v1), until the
  backfill (T-150)") and `:275`; `PLAN.md:212` ("the v1 `cli/news_nlp_cli.py` is not touched until the backfill (T-150)");
  `docs/modules/news-nlp.md:10` ("Downstream repos still read its `article_*` tables": the sentence the owner's answer replaces);
  `docs/legacy/README.md:7-8`; FR-001 to FR-005 of SPEC. The constitution names the v1 stages only in passing (`constitution.md:119`,
  `:124`, and a parenthesis in `:147`, "the v1 chain is unchanged", inside the DeepSeek summary rule): it forbids nothing, but those
  three lines are reworded and its version bumped. D44 of PLAN is the DeepSeek decision, not a rule about v1: a **new PLAN decision**
  records the depuration. No document requires the owner's approval of a schema change in general (`SPEC.md:575` is about T-138);
  dropping tables is new (the existing migrations, `schema.py:499-575`, are additive), so the owner's approval is asked here and
  recorded in the PR of T-220.
* **T-131** (open: move aside the results of quarantined pairs, copying them to `<table>_removed` first, "never rename a whole table")
  and the "old tables to v1" part of **T-150** (open, with the `portfolio_common.news_export` contract update) both concern the v1
  tables. If the v1 tables are dropped, T-131 is moot for them and that part of T-150 is superseded; the owner confirms when T-220
  closes. **T-189** (open: delete `category-981dacf6f68d`, drop `LEGACY_REVISIONS`) is overtaken by the prune for the first part (the
  second part, the code, stays its own); **T-212** and **T-190** (repeat the 190-piece check and the summary tests on the DeepSeek
  revision) need the rows of the kept summary and sentiment revisions, which the prune keeps; **T-185**, **T-191**, **T-194** and
  **T-206** name revisions in their text, which the prune does not change but PR C checks (see "Documents and tasks that cite a
  revision"). The **rule 16** of the constitution (an article that was fetched with a title and a body is never deleted) is not touched:
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
expose, the port. Until it is merged the v1 API keeps working. **Acceptance (proposed):** every step and the whole pipeline run
through the API with an explicit revision and write the same rows as `cli/news_nlp_steps.py` on a copy of the results store; a job's
status is readable while it runs; the Docker image serves it and `/health` answers; the v1 API and its tests are not touched (T-220
removes them).

### T-220: the depuration (after T-219)

**Phase 0, gates (no code).**

1. **Read-only listing of the revisions, first.** From the backup copy, for every table with a revision column: each distinct revision,
   its row count and its date range (`SELECT <col>, COUNT(*), MIN(<date>), MAX(<date>) FROM <table> GROUP BY <col>`; the column per
   table is in the table of PR C), plus `period_run.snapshot_revision` and the revisions inside `news_snapshot.inputs`. It is attached
   to the PR. **The owner confirms the keep list against that listing, not from memory:** four of the eight revisions in the list
   (`sentiment-59863d15fc42`, `category-a23569dba86f`, `snapshot-c3a753affa5d`, `period-3b07b83ecec3`) and the delivery note's
   `period-ae014b56b055` appear in no tracked file, and the documents cite other ones (see below).
2. **A maintenance window, then a fresh backup.** The daily 15:00 run writes to the live file, so a backup made earlier never equals
   it. The owner stops the scheduled run and confirms no run is in progress (an exclusive transaction on the live file succeeds after
   `PRAGMA wal_checkpoint(TRUNCATE)`), a **fresh** `nlp_backup.db` is made with the SQLite backup API, `PRAGMA integrity_check` returns
   `ok`, the row count of every table equals the live file's (it can now, nothing writes), and its SHA-256 and the counts are recorded
   in the PR. Everything else is rehearsed on a *second copy*. The scheduler is restarted only after the checks of PR C pass.
3. **A restore is rehearsed, not assumed.** The backup is restored onto a scratch file; `--step <step> --status` and the golden SQL run
   on it and match the live file's "before". The backup made in step 2 is kept until the owner closes T-220. `VACUUM` of the live file
   needs about the size of the database in free disk space (and a clean WAL): the free space is checked and written in the PR.
4. **Owner approvals** recorded: the drop of the v1 tables, the rewording of the lines listed in "The rules that bind the work", the
   new PLAN decision.
5. **The verified backup replaces the `<table>_removed` copies of T-131.** Those would add tables to a database that must hold only
   kept ones. The owner accepts this or says otherwise.
6. **The sector-step revision is named** (gate 1 of "Left open"): **the prune does not run until it is.**
7. **Order:** after #199 (`7e3b6b3`) and #204 (`a435e20`), both on master: the guard below pins the revisions as they are on master.

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

* `schema.py`: remove the v1 DDL and its migrations; a test lists exactly the tables `init_schema` creates on a file that already has the
  lean `articles` table (`schema.py` does not create it: the crawler side owns it and `db._ensure_article_row` fills it; the steps CLI
  refuses a file without it, `cli/news_nlp_steps.py:591-595`). `eval_judgement` is still created at `schema.py:168`: it goes with the
  v1 DDL if it is confirmed unused.
* `scripts/depurate_db_<date>.py`, dry run by default:
  * `--apply` refuses without `--backup nlp_backup.db` verified (integrity and counts); one transaction; `DROP TABLE` of the v1
    tables (and `eval_judgement` if confirmed unused); then `PRAGMA foreign_key_check`, `PRAGMA integrity_check`, the row counts of
    every kept table, and the counts of the delivered revisions must equal the "before"; the golden SQL must return identical rows.
  * `--apply-prune` (its own dry run): the **keep list is a file in the repository**, one revision per family, and **each table is
    pruned on its own column against the revisions of its own step** (a flat list would be wrong: the column is not the same in every
    table, and a table whose step has no kept revision would lose every row):

    | Table | Revision column | Kept family |
    |---|---|---|
    | `body_scope` | `classifier_revision` | `scope-nli-8ff21e50796b` |
    | `body_sector` | `classifier_revision` | **none yet: the sector-step revision is missing from the owner's list** |
    | `sub_news` | `extractor_revision` | `units5-d3e2cbfa830e` |
    | `news_summary`, `news_summary_call` | `model_revision` | `summary-llm-70cb6bb07292` |
    | `news_company` | `model_revision` | `companies-f349536cf701` |
    | `news_sentiment` | `model_revision` | `sentiment-59863d15fc42` |
    | `news_category` | `model_revision` | `category-a23569dba86f` |
    | `news_snapshot` | `model_revision` | `snapshot-c3a753affa5d` |
    | `period_run`, `period_summary`, `period_item` | `model_revision` | one `period-...` (gate 2 of "Left open") |
    | `pipeline_step_state` | `model_revision`, with its `step` | the kept revision of that step (a row of a step with no kept revision is not deleted: the script stops and asks) |

    Not pruned: `sector`, `company_sector`, `eval_*`, the lean `articles`. **The script refuses to run for a table whose family is empty
    in the keep list** (today `body_sector`), and it refuses when a kept row refers to a pruned one: a kept `news_*` row whose
    `sub_news` id exists only in a pruned cut revision; a kept period whose snapshot revision is not kept; a revision named in the
    `inputs` of a kept snapshot row that is not kept. The dry run prints, per table and revision, the rows to delete. Deletes follow the
    dependency order (`period_item`, `period_summary`, `period_run`, then the rest), one transaction per table, then the same checks,
    then `VACUUM` on the copy.
* Docs: `docs/db-topology.md`, `docs/modules/news-nlp.md`, `docs/legacy/README.md`, README, SPEC (FR-001 to FR-005 and the schema
  list), PLAN (a new decision), constitution (amendment, version), `CLAUDE.md` by hand (local file, never deleted); replaced text goes
  to `archive/` verbatim; both architecture artifacts are reconciled at the close of the development (not now: nothing is built).

**Acceptance of T-220:** the revision guard passes unchanged; the golden SQL returns identical rows on the backup copy before and
after; the restore of the pre-apply backup onto a scratch file gives the same `--status` and golden SQL; `init_schema` creates exactly
the listed tables (above); `--step <step> --status` on the backup copy shows the same
counts; ruff, mypy and the full suite with `--group eval` pass; the live database is touched only by the owner, after a dry run.

## Documents and tasks that cite a revision

The prune removes the rows of every revision outside the keep list, and these documents cite such revisions (searched on master on
2026-11-17; the figures they report stay as history, they were scored once and are recorded in the document or its CSV):

| Revision cited (not in the keep list) | Documents |
|---|---|
| `summary-7a9bd2062bc4` (distilbart summaries) | `docs/summary_step.md`, `docs/sentiment_training.md`, `docs/ner_training.md`, `docs/companies_step.md`, `docs/category_results_2026_11_13/README.md`, `pipeline_metrics_report.md`, `pipeline_metrics_report_es.md` |
| `sentiment-5999fe131d8f` (the 190-piece check of #201 was scored on it) | `docs/sentiment_training.md`, both pipeline reports |
| `category-981dacf6f68d`, `category-65f33b55b26b` | `docs/category_step.md`, `docs/category_training.md`, `docs/category_results_2026_11_13/README.md` |
| `units-...`, `units4-...` | `docs/scope_training.md`, `docs/summary_step.md` |
| `sector-cf81d9b32565`, `sector-417a93feb789` | `docs/sector_step.md`, `docs/gold_datasets.md` |

For each one PR C does one of two things, and the owner says which in the dry run: **(a)** the figure stays as history and the document
gets one line ("revision pruned on <date>; the figure is as recorded"); or **(b)** the rows behind a figure that must stay
reproducible are exported to a CSV in `docs/` first. Note that the documented sentiment revision (`sentiment-5999fe131d8f`) is not the
one in the owner's keep list (`sentiment-59863d15fc42`): the owner says whether that is the same model after a code change or a
different run. The open tasks that name revisions (T-185, T-189, T-190, T-191, T-194, T-206, T-212) are checked the same way.

## Left open (gates; none blocks this planning PR, all block the prune)

1. **The sector-step revision.** The owner's keep list has none; `body_sector` would lose every row and the kept snapshot reads it
   (`news_snapshot_stage.py:445`). The kept snapshot's `inputs` names one; the documents mention `sector-cf81d9b32565` and
   `sector-417a93feb789`, which may be the earlier ones. The owner names the revision(s) from the Phase 0 listing.
2. **The period revision.** The delivery note names `period-ae014b56b055`; the keep list says `period-3b07b83ecec3`. #204 changed
   the period revision, so the second may be the one written after it, but that is not verified: the Phase 0 listing settles it. The
   script never prunes a revision named in the delivery note unless the owner removes it from the note's list explicitly.
3. **The sentiment revision** (above): `sentiment-5999fe131d8f` documented, `sentiment-59863d15fc42` in the list.
4. **`eval_judgement`:** confirm unused (read the code in PR A/B).
5. **The eval key mapping** (PR A) and **T-219's design** (authentication, jobs, the FTI services): settled at the start of each.
6. **Open tasks that overlap:** T-131, T-150 (part), T-189, T-212 (see "The rules that bind the work"); T-218 (testing the Docker
   image) is affected by the retarget in T-219.

(Closed here, not open: `config/pipeline_models.json` stays, read by step 7.)
