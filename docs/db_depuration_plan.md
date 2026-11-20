# Plan: depurate the NLP database (T-220), after a new FastAPI (T-219)

**Status (2026-11-17): a plan, nothing of T-220 is built.** T-219 (the new FastAPI) is done: `docs/pipeline_api.md`. This document and the two tasks in `.specify/memory/TASKS.md` are the whole of the
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
* **The old eval stages are deleted, not rewired** (owner, 2026-11-18; on 2026-11-17 the decision was to rewire them to `news_*`, which turned out to be a redesign: see "A2 was dropped"). The eval store, the MLflow tracking and the blind step-7 judge stay.
* **A new FastAPI comes first** (T-219): it serves each step, the whole pipeline and the FTI services. The v1 API
  (`apps/news_nlp_api.py`) reads only v1 tables and the Docker image serves it (`Dockerfile:106-109`), so the v1 tables cannot go
  while it exists. The depuration (T-220) starts after T-219 is merged.
* **`snapshow_delivery.md` stays untracked** (the owner's local file). The golden check below uses a copy of its queries inside the tests.
* **Old revisions are pruned**, keeping exactly: `scope-nli-8ff21e50796b`, `units5-d3e2cbfa830e`, `summary-llm-70cb6bb07292`,
  `companies-f349536cf701`, `sentiment-59863d15fc42`, `category-a23569dba86f`, `snapshot-c3a753affa5d`, `period-3b07b83ecec3`.
* **Gates answered (owner, interactive session, 2026-11-18, after the Phase 0 listing below).** (1) `sector-417a93feb789` joins the keep list
  (the only sector revision). (2) Of the periods **only `period-3b07b83ecec3` stays**; `period-ae014b56b055` (the delivery note's) and
  `period-dd098e1c5b9f` are pruned, **after a new delivery note naming `period-3b07b83ecec3` goes to the upstream team** (the plan's rule
  "the upstream team is told first"). (3) **Only `sentiment-59863d15fc42` and `category-a23569dba86f` stay**: `sentiment-5999fe131d8f` and
  `category-65f33b55b26b` are pruned. Consequence, accepted: the 159,401 rows of `snapshot-c3a753affa5d` built on them keep their own text,
  category and sentiment, but their `inputs` names revisions that no longer exist (the dry run lists them and does not refuse). A rerun of
  the snapshot on the kept revisions was offered and **not** chosen; it stays available if the owner asks. The final keep list is nine
  revisions: the eight above plus `sector-417a93feb789`.
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
`eval/step_category.py` is already shaped for step 7. This is why the old eval stages are deleted with the v1 code in PR B (the rewire was dropped) before the removal of the v1 code.

### v1 code with no kept importer once the old eval stages are deleted

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
* **T-131** (dropped by the owner on 2026-11-20: nothing is copied aside to `<table>_removed`; the verified backup replaces it. It was: move aside the results of quarantined pairs, copying them to `<table>_removed` first, "never rename a whole table")
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
   to the PR. **The keep list was checked against that listing (the owner's answers of 2026-11-18, "Gates answered"; the listing is the reference, not memory):** four of the eight revisions in the list
   (`sentiment-59863d15fc42`, `category-a23569dba86f`, `snapshot-c3a753affa5d`, `period-3b07b83ecec3`) and the delivery note's
   `period-ae014b56b055` appear in no tracked file, and the documents cite other ones (see below).
2. **A maintenance window, then a fresh backup.** The daily 15:00 run writes to the live file, so a backup made earlier never equals
   it. The owner stops the scheduled run **and the pipeline API (`docs/pipeline_api.md`: `GET /jobs` shows no running job, or the service is stopped)** and confirms no run is in progress (an exclusive transaction on the live file succeeds after
   `PRAGMA wal_checkpoint(TRUNCATE)`), a **fresh** `nlp_backup.db` is made with the SQLite backup API, `PRAGMA integrity_check` returns
   `ok`, the row count of every table equals the live file's (it can now, nothing writes), and its SHA-256 and the counts are recorded
   in the PR. Everything else is rehearsed on a *second copy*. The scheduler is restarted only after the checks of PR C pass.
3. **A restore is rehearsed, not assumed.** The backup is restored onto a scratch file; `--step <step> --status` and the golden SQL run
   on it and match the live file's "before". The backup made in step 2 is kept until the owner closes T-220. `VACUUM` of the live file
   needs about the size of the database in free disk space (and a clean WAL): the free space is checked and written in the PR.
4. **Owner approvals** recorded: the drop of the v1 tables, the rewording of the lines listed in "The rules that bind the work", the
   new PLAN decision.
5. **T-131 dropped (owner, 2026-11-20):** the verified backup replaces the `<table>_removed` copies. Nothing is copied aside, so the
   database holds only the kept tables.
6. **The sector-step revision is named** (gate 1 of "Left open"): **the prune does not run until it is.**
7. **The upstream team is aware (owner, 2026-11-20).** The owner states that portfolio-financial-analysis already knows about the change
   to `period-3b07b83ecec3`, so no new delivery note file is written. This replaces the rule that a note reaches them before the prune,
   which was not followed on 2026-11-18 (see the review note). The script still refuses to prune a revision named in the delivery note
   unless the owner removes it from the note.
8. **Order: satisfied on 2026-11-17.** #199 (`7e3b6b3`) and #204 (`a435e20`) are on master, so the guard below pins the revisions as they are now; a later change of a step's code before T-220 starts is a new revision and the guard is re-pinned then.

**PR A (A1, #211): the guard and the golden check.** A **revision guard test** pins the code each step hashes, the config files the revisions also hash, and the
delivered snapshot revision; any later edit that changes a revision fails CI. A **golden check** holds the SQL of `snapshow_delivery.md` as code
(`src/delivery_golden.py`): it runs on a fixture in the tests and, with `python -m delivery_golden`, on the backup copy before and after each destructive
step, where it must return the same digest. **A2, the eval rewire, was dropped by the owner on 2026-11-18**: the old eval stages are deleted in PR B, not
rewired; T-222 rebuilds an eval on `news_*` if wanted. The design that was dropped is in `docs/legacy/db_depuration_plan_a2_rewire_2026-11-18.md`.

**PR B: remove the v1 code and its tests.** Delete the files listed above, rewire `src/setup.py` (it imports constants from
`pipeline`), clean `news_nlp/__init__.py`, delete the only-v1 tests, adapt the mixed ones, update `tests/sets.py`. The hashed shared
modules stay byte-identical (the guard proves it). The Docker retarget was done in T-219.

**PR C: the schema, the drop and the prune.**

* `schema.py`: remove the v1 DDL and its migrations; a test lists exactly the tables `init_schema` creates on a file that already has the
  lean `articles` table. **Done (round 6 of #216):** `schema.py` creates the lean `articles` table (fixed definition, `IF NOT EXISTS`;
  the crawler still owns the SOURCE table and `db._ensure_article_row` fills the RESULTS rows). The `--step cut` check for a missing
  `articles` table was removed as unreachable. `eval_judgement` is no longer created (the v1 DDL is gone); it is in the purge's drop list.
* **The sequence of the apply (one maintenance window of Phase 0, the owner at the keyboard):** (a) the scheduled run and the pipeline API stopped, a fresh
  backup verified (Phase 0 step 2), the dry runs of `--apply` and `--apply-prune` read and accepted; (b) `--apply` (the drop), its
  checks and the golden SQL; (c) a **second fresh backup**, verified, so that each step can be restored on its own; (d) `--apply-prune`,
  its checks and the golden SQL; (e) the scheduled run and the API restarted. **A failed check stops the sequence** and the last verified backup is
  restored (rehearsed in Phase 0 step 3); nothing is retried on the live file.
* `scripts/depurate_db_<date>.py`, dry run by default:
  * `--apply` refuses without `--backup nlp_backup.db` verified (integrity and counts); one transaction; `DROP TABLE` of the v1
    tables (and `eval_judgement` if confirmed unused); then `PRAGMA foreign_key_check`, `PRAGMA integrity_check`, the row counts of
    every kept table, and the counts of the delivered revisions must equal the "before"; the golden SQL must return identical rows.
  * `--apply-prune` (its own dry run): the **keep list is a file in the repository**, one revision per family, and **each table is
    pruned on its own column against the revisions of its own step** (a flat list would be wrong: the column is not the same in every
    table, and a table whose step has no kept revision would lose every row):

    | Table | Revision column | Date column (for the Phase 0 listing) | Kept family |
    |---|---|---|---|
    | `body_scope` | `classifier_revision` | `processed_at` | `scope-nli-8ff21e50796b` |
    | `body_sector` | `classifier_revision` | `processed_at` | **none yet: the sector-step revision is missing from the owner's list** |
    | `sub_news` | `extractor_revision` | `processed_at` | `units5-d3e2cbfa830e` |
    | `news_summary`, `news_summary_call` | `model_revision` | `processed_at` (`news_summary`), `called_at` (`news_summary_call`) | `summary-llm-70cb6bb07292` |
    | `news_company` | `model_revision` | `processed_at` | `companies-f349536cf701` |
    | `news_sentiment` | `model_revision` | `processed_at` | `sentiment-59863d15fc42` |
    | `news_category` | `model_revision` | `processed_at` | `category-a23569dba86f` |
    | `news_snapshot` | `model_revision` | `processed_at` | `snapshot-c3a753affa5d` |
    | `period_run`, `period_summary`, `period_item` | `model_revision` | `period_run.created_at` | one `period-...` (gate 2 of "Left open") |
    | `pipeline_step_state` | `model_revision`, with its `step` | `started_at` / `finished_at` | the kept revision of **that step** (the family is per `step`) |

    Not pruned: `sector`, `company_sector`, `eval_*`, the lean `articles`. **The script refuses to run for a table whose family is empty
    in the keep list** (today `body_sector`; for `pipeline_step_state` it lists every `step` with no kept revision and stops: no row is
    deleted until the owner answers), and it refuses when a kept row refers to a pruned one: a kept `news_*` row whose
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

## Phase 0 listing of `nlp_backup.db` (read 2026-11-18, read-only and immutable; backup file of 2026-11-16 18:08, `integrity_check` = `ok`)

The file is a copy of `nlp.db` (same size, same timestamp). **Revisions per table** (row counts):

| Table | Revisions (rows) |
|---|---|
| `body_scope` | `scope-nli-8ff21e50796b` (309,972) |
| `body_sector` | `sector-417a93feb789` (32,887): **the only one** |
| `sub_news` | `units5-d3e2cbfa830e` (161,574), `units-5f831c298bcb` (165,192), `units4-d3e2cbfa830e` (7,311) |
| `news_summary` | `summary-llm-70cb6bb07292` (254,607), `summary-7a9bd2062bc4` (433,266) |
| `news_company` | `companies-f349536cf701` (363,939) |
| `news_category` | `category-a23569dba86f` (226,139), `category-65f33b55b26b` (159,401), `category-981dacf6f68d` (10,355) |
| `news_sentiment` | `sentiment-59863d15fc42` (226,139), `sentiment-5999fe131d8f` (159,401) |
| `news_snapshot` | `snapshot-c3a753affa5d` (226,139), `snapshot-1b8ee113b05f` (159,401) |
| `period_run` | `period-3b07b83ecec3` (85), `period-ae014b56b055` (85), `period-dd098e1c5b9f` (85) |

v1 tables: `article_entities` 16,436,161 rows, `article_sentiment` and `article_category` 459,112, `article_summary` 458,692,
`sector_summary` 3,630, `eval_judgement` 48,403 (drop candidates). `company_sector` is empty; `sector` has 11 rows.

**What the listing settles**

1. **The sector gate.** One sector revision exists, `sector-417a93feb789`: name it in the keep list. The kept snapshot rows have an empty
   `inputs.sector`, so no kept row reads another one.
2. **The period gate.** `period-ae014b56b055` (written 2026-11-14 17:07) and `period-3b07b83ecec3` (2026-11-16 18:08, after the fold of
   #204) are **both built on the delivered `snapshot-c3a753affa5d`**, with the same 85 periods ending 2026-11-02. They differ by the
   fold: the delivery note describes the pre-fold counts. Pruning `ae014` changes what the upstream team's queries return (zero rows)
   until a new delivery note names `period-3b07b83ecec3`. `period-dd098e1c5b9f` is built on `snapshot-1b8ee113b05f` (not delivered).
3. **The sentiment gate, and a new one.** `sentiment-59863d15fc42` covers 226,139 pairs (the keep list); `sentiment-5999fe131d8f` (the
   documented one) covers 159,401. **The delivered snapshot holds both:** 66,738 rows built on `sentiment-59863d15fc42` and
   `category-a23569dba86f`, and **159,401 rows built on `sentiment-5999fe131d8f` and `category-65f33b55b26b`** (the rows keep their own
   copy of the text, category and sentiment; `inputs` records the revisions). Pruning those two revisions, as the keep list says, leaves
   159,401 delivered rows whose `inputs` name revisions that no longer exist: the dry run's refusal rule (a kept row that refers to a
   pruned one) stops there. **Decided (owner, 2026-11-18): both are pruned and the dangling provenance in `inputs` is accepted; no snapshot rerun (see Decided).**
4. `snapshot-1b8ee113b05f` has 159,401 rows and is not delivered: a prune candidate, once nothing kept refers to it.

## Golden baseline on `nlp_backup.db` (run 2026-11-18, read-only and immutable)

`PYTHONPATH=src uv run python -m delivery_golden --db /workspaces/thesis/data/nlp_backup.db --immutable --snapshot snapshot-c3a753affa5d --period <revision>`. This is the "before" of PR C: after a drop or a prune on the copy, the same command must print the same digest.

**The file that produced it:** `nlp_backup.db`, 5,271,089,152 bytes, modified 2026-11-16 18:08:45 UTC, SHA-256 `4d65d5873f60d136898a556f6485a7dfa333761ed2cfabf6e68e5bccea6818f2`, read with SQLite 3.40.1. **Reproducible:** the command was run three times for each period revision and printed the same digest every time.

| Query of the note | `period-ae014b56b055` (the note) | `period-3b07b83ecec3` (kept) |
|---|---|---|
| `snapshot_date_sources` | 2 rows (14-21 s) | 2 rows (14 s) |
| `company_news_in_a_range` | 74 rows (13 s) | 74 rows (13 s) |
| `periods` | 85 | 85 |
| `counts_per_company_category_period` | 27,725 | 34,253 |
| `net_sentiment_per_company_period` | 15,099 | 17,777 |
| `counts_per_sector` | 5,259 | 5,649 |
| `news_a_period_counted` | 481 | 707 |
| **digest** | **`bd104b26acea6f3f`** | **`2832f270e9a52457`** |

**After the prune** (what a reviewer should see on the copy): for `period-3b07b83ecec3` the digest stays `2832f270e9a52457`. For `period-ae014b56b055` the five period queries return nothing and the two snapshot queries are unchanged, so the digest becomes **`9a71c0f4d9001a85`** (the command prints it as "digest if --period is pruned"). It is not the digest of seven empty results: the two snapshot queries read only the snapshot revision.

The two snapshot queries take 13 to 20 seconds: `news_snapshot` has no index on `model_revision`, so they scan the table. PR C's prune (a `DELETE ... WHERE model_revision` per table) should expect full scans and run in one window.

**Phase 0 gate (a), what is and is not done.** Done: `integrity_check` = `ok`, the SHA-256 above, the per-table revision listing. **Not done:** the per-table row counts of the backup against the live `nlp.db`; this work does not open the live file. It is the owner's step in the maintenance window (the SQLite backup API, then the counts), and PR C's script refuses to apply without it.

## What the revisions read from `config/` (found by changing every key)

Every key of every file in `config/` was changed one at a time and the revisions were recomputed (category, sentiment, summary local and API, companies, sector). Only these moved a revision, and only these are pinned in `tests/news_nlp/test_revision_guard.py`:

| File | Keys that move a revision | Revision |
|---|---|---|
| `category_step.json` | `name`, `identity` | `category-` |
| `pipeline_models.json` | `ner.model`, `ner.revision` | `companies-` |
| `pipeline_models.json` | `c_summary.model`, `c_summary.revision` | `summary-` (local summariser) |
| `published_models.json` | `relevance.repo`, `relevance.revision` | `companies-` |
| `sentiment_step.json` | `identity` | `sentiment-` |
| `sentiment_stack.json` | every key | `sentiment-` |
| `sector_step.json` | every key | `sector-` |
| `summary_prompt.md` | its bytes | `summary-llm-` |

**No revision reads:** `attribution_model.json`, `category_model.json`, `scope_model.json`, the `sentiment`, `category` and `batch_size` entries of `pipeline_models.json`, the other entries of `published_models.json`, and the evidence, run and label keys of `category_step.json`. An edit of those (a new evidence figure, a published model) must not fail the guard, and does not. (The scope and cut revisions come from the model folder, not from `config/`; their pins only feed the status lines and `--fetch`.)

## The delivered cut revision is not reproducible from the code

The backup holds `units5-d3e2cbfa830e` (161,574 sub-news, written 2026-11-12 14:37 to 17:19 UTC). Recomputed from the pin of `config/attribution_model.json`, the universe of `urls.db` and the code, the cut revision is **`units5-4f682083d961`** today, and **`units5-858ec90552aa`** at the commit before `e7a6f02`. Neither is the delivered one, so the line-ending question cannot be settled from this data. What the history shows:

* **`e7a6f02` (2026-11-12 23:10 -05:00, "step 3, the companies") edited `src/news_nlp/triage.py`, one of the five modules the cut hashes by bytes, hours after the cut ran.** That commit moved the cut revision from `units5-858ec90552aa` to `units5-4f682083d961`. A rerun of `--step cut` today would write `units5-4f682083d961` and cut the corpus again. Nothing reruns it: the later steps name the delivered cut explicitly (`--cut-revision units5-d3e2cbfa830e`).
* Even before that commit the code does not give `d3e2cbfa830e` with today's pin and universe, so an input of that day (the identity of the local model folder, the universe at that moment) is not recoverable here. The same hash suffix sits under `units4-` and `units5-` in the backup.

Consequence for the guard: its cut pin (`be358e64f554`) protects **today's** cut code, not the delivered revision. The owner decides whether that matters: the delivery note names the snapshot and the period only, and the cut revision is an input recorded in each row (`inputs`).

## PR C1: the purge script, and its rehearsal on a copy of the backup (2026-11-18)

`PYTHONPATH=src uv run python -m depurate_db --db <file> [--backup <file> --plan <digest> --apply]`. The schema (`schema.py`) is not touched here; PR C2 removes the v1 DDL after the purge.

**Gates of `--apply`:** a backup that is a readable copy of the same database (same tables, the same **count and rowid fingerprint** in each, `quick_check` ok; with `--require-identical-backup` also the same SHA-256, the strong gate, valid while the database has not been written since the copy); `--plan` equal to the digest a dry run printed (so the apply deletes what was read); no `-journal` and no non-empty `-wal` beside the file (an empty `-wal` and a `-shm` are what any reader leaves, a dry run included: the first rehearsal run was refused by them, which is now a test); the refusal checks (no kept row on a pruned cut, no kept period on a pruned snapshot, every kept revision present, no table the script does not know). Then one transaction per table, and after it `foreign_key_check`, `quick_check`, the kept row counts and the golden digest must equal the "before", then `VACUUM`.

**The rehearsal** (copies of `nlp_backup.db` in a scratch folder; the live file was not opened):

| | |
|---|---|
| dry run | plan digest `63e141924e02e7fe`, no refusal |
| dropped | 17,865,110 rows: `article_entities` 16,436,161; `article_sentiment` 459,112; `article_category` 459,112; `article_summary` 458,692; `sector_summary` 3,630; `eval_judgement` 48,403 |
| deleted (old revisions) | 2,151,279 rows (`news_summary` 433,266; `pipeline_step_state` 721,098; `sub_news` 172,503; the old periods, snapshot, category and sentiment revisions) |
| checks after | `foreign_key_check` 0, `quick_check` ok, kept counts equal, **golden digest `2832f270e9a52457` before and after** |
| size | 5,271,089,152 → 1,628,180,480 bytes |
| time | the table steps took about 85 s on local disk. The **live purge on its copy took 11 minutes** (19:14:34 to 19:25:35): it also counted every table of the 5 GB backup on the Windows drive and ran the golden queries before and after (about 15 s each for the two snapshot queries). The copy of the 5 GB file itself took about 2 minutes |
| a second dry run | nothing to delete; the `inputs` report is unchanged |
| restore | the backup copy gives the original digest (`2832f270e9a52457` for the kept period) |

**Reported, not refused:** the kept snapshot rows whose `inputs` name a pruned revision: `inputs.summary = summary-7a9bd2062bc4` 159,401; `inputs.category = category-65f33b55b26b` 159,401; `inputs.sentiment = sentiment-5999fe131d8f` 159,401 (the summary one was not in the earlier plan; the rows keep their text, category and sentiment).

**`eval_judgement`** (48,403 legacy rows, superseded 2026-09-18) is read by no code of `src/`, `apps/` or `cli/`; only a one-shot script of 2026-09-13 (`label_sentiment_sentences_2026_09_13.py`) read it. By the rule stated while the question was open ("keep it if anything reads it") it should have been kept; **the owner confirmed the drop on 2026-11-18** (asked in those words in the review of #215) **and asked for an export**: `/workspaces/thesis/data/eval_judgement_2026-11-18.csv`, 48,403 rows, 93,555,678 bytes, SHA-256 `e35d22d863a18f65afae34c6d15a334f2477d2cf016b591683e7e7be939f668e`, read from the untouched backup (read-only, immutable). The backup holds the table too.

**The live file** (`/workspaces/thesis/data/nlp.db`, which the `.env` names as `D:\thesis\data\nlp.db`) has the same size, the same modification time to the nanosecond and **the same SHA-256 as `nlp_backup.db`** (`4d65d5873f60d136898a556f6485a7dfa333761ed2cfabf6e68e5bccea6818f2`, checked 2026-11-18): the backup is a byte-identical copy and nothing wrote to the live file after it. Because SQLite on the Windows drive (DrvFs) is slow and not safe for a long write, the purge is run on a **local copy** and the result replaces the live file only after it is verified; the original backup is not touched.

## The purge was run on the live database (2026-11-18)

Run by the assistant at the owner's request ("run it yourself"), after a rehearsal, with the owner's confirmation that nothing was writing to the file.

1. **Backup:** `nlp_backup.db` is byte-identical to the live `nlp.db` (SHA-256 `4d65d5873f60d136898a556f6485a7dfa333761ed2cfabf6e68e5bccea6818f2`, same size, same modification time to the nanosecond); it was not touched and stays the rollback.
2. **Purge on a local copy** of the live file (DrvFs is slow and unsafe for a long write): dry run plan `63e141924e02e7fe` (the same as the rehearsal), no refusal; `--apply` with `--backup nlp_backup.db`: 17,865,110 rows dropped, 2,151,279 old-revision rows deleted, `foreign_key_check` 0, `quick_check` ok, kept counts equal, golden digest `2832f270e9a52457` before and after.
3. **Swap:** the purged file (SHA-256 `bcd7dfe3f82792c746d910fc7b69a1ac30c47c9d5b1ed81cfae5933f368af239`, 1,628,180,480 bytes) was copied next to the live file, its SHA-256 checked on the drive, the live file checked unchanged, then renamed over `nlp.db`.
4. **Checked on the live file after the swap:** `quick_check` ok; the tables are exactly the kept ones (`articles`, `body_scope`, `body_sector`, `company_sector`, `eval_*` x4, `news_category`, `news_company`, `news_sentiment`, `news_snapshot`, `news_summary`, `news_summary_call`, `period_item`, `period_run`, `period_summary`, `pipeline_step_state`, `sector`, `sub_news`); the golden digest of the kept period is **`2832f270e9a52457`** (unchanged), and that of the delivery note's `period-ae014b56b055` is **`9a71c0f4d9001a85`**, the digest this document predicted for a pruned period.

**Rollback:** copy `nlp_backup.db` over `nlp.db`. **Still to do:** PR C2 removes the v1 DDL and the v1 migrations from `schema.py` (a fresh database then has only the kept tables), and the documents that describe the v1 tables are updated.

**Review note (round 3 of #216, 2026-11-18): the gates above were not all met before the live prune.** The prune ran in the purge with the owner's "run it yourself". Two of the plan's own gates were still open then: **gate 1** (the sector-step revision: the owner's answer is recorded in "Gates answered", item (1), 2026-11-18, so this is not open; an earlier version of this note said the opposite, and was wrong) and **rule 7** (no new delivery note naming `period-3b07b83ecec3` was sent upstream before `period-ae014b56b055` was pruned; the note in the repository is untracked and still names `period-ae014b56b055`). Gate 2 (the dangling `inputs`) has the owner's decision above. Update (owner, 2026-11-20): the upstream team already knows about the change to `period-3b07b83ecec3`, so no new note is written; this item is closed on that point. What is left for the owner is the live state: accept the purged file as it is, or restore the rollback (`nlp_backup.db`, the verified copy of the file before the purge).

## Runbook: purge a database safely (what was done on 2026-11-18, now in the tool)

The shape that protects a production file: **purge a local copy, verify, then swap**. Never `--apply` in place on the Windows drive (the tool refuses unless `--in-place`).

1. **Stop every writer and close every tool that has the file open** (the scheduled run, the pipeline API, a DB browser, an IDE extension, a shell with a SQLite connection) and let the last connection close. The swap detects a **writer** (it takes an exclusive lock on the live file and rolls it back before the rename) but **cannot detect a reader**. A `-journal` or a **non-empty `-wal`** beside the file means a writer is attached or crashed: with a non-empty `-wal`, open the file once with the SQLite tool and let it close (that checkpoints the `-wal` into the main file) before copying; copying only `nlp.db` would lose what is in the `-wal`. An empty `-wal` and a `-shm` are left by any reader and mean nothing.
2. **Backup:** copy the file (or use the SQLite backup API) to a name that is never overwritten; record its SHA-256. With the file not written since, `sha256(live) == sha256(backup)` is the proof.
3. **Copy the live file to a local folder** and run the dry run on the copy: `PYTHONPATH=src uv run python -m depurate_db --db <copy>`. Read it; note the plan digest and the "free space" line. **Every revision shows the date of its newest row, and a revision to be DELETED that is newer than the keep decision (2026-11-18) gets a WARNING line**: the keep list is a hard-coded decision, so a revision made later (a new summary run, a retrained model) would otherwise be pruned by someone who did not read the plan.
4. **Apply on the copy:** `... --db <copy> --backup <backup> --plan <digest> --require-identical-backup --apply --report <report.json>`. It deletes only the revisions the dry run listed, table by table under an exclusive lock (the two refusal checks that protect `sub_news` and `news_snapshot` run again inside their locks), then checks `foreign_key_check`, `quick_check`, the kept counts and the golden digest, then `VACUUM`. A failure leaves the live file untouched and the report says what was done on the copy; a successful report carries the `sha256` and size of the purged file.
5. **Swap:** `... --db <purged copy> --swap-into <live> --backup <backup> --from-report <report.json of the apply>` (or `--expect-sha <the report's sha256>`; one of them is required). It checks that both files have no holder; that the purged file has the expected SHA-256, passes `quick_check`, has nothing left for a purge to do and has **the same kept rows as the live file** (so it was made from it); that the live file is byte-identical to the backup; that nobody holds a write lock on it; that the copy next to the live file has the same SHA-256; and that the live file did not change, gain a holder or get locked while copying. that both files are in the **same journal mode** (a difference is refused unless `--allow-journal-mode-change`; the modes are logged and in the report). Then one atomic rename. It logs the sizes and SHA-256 it compared. The writer probe opens the live file read-write; the empty `-wal` and the `-shm` that it creates are removed again (only those, only if they were not there before). A stale copy (a step ran on the live file after the copy was taken) is reported as that, with the first table whose kept rows differ.
6. **Check the live file:** the golden digest (`python -m delivery_golden --immutable`) and a dry run (nothing left to do).
7. **Rollback:** copy the backup over the live file (with the writers stopped).

**What the 2026-11-18 live run actually used.** The backup gate of that run was the **row-count** comparison; the backup was proven identical to the live file by SHA-256 **by hand** (`4d65d587…` for both). The `--require-identical-backup`, the fingerprint, the in-lock guards and the swap checks above came with the review of #215 and were not part of that run; the result on the live file was verified afterwards (SHA-256, `quick_check`, the tables, the golden digests).

**The 20:02 sidecars.** After the swap an empty `nlp.db-wal` and a `nlp.db-shm` appeared beside the live file (20:02); the file's bytes did not change (SHA-256 still `bcd7dfe3…`). The owner said they came from **a process inside this development environment, now ended**, and told the assistant to proceed. The two empty files were removed (no connection was open) and the SHA-256 was checked again.

## A2 was dropped (owner, 2026-11-18)

Reading the code showed that rewiring the eval was a redesign of sampling, judges and metrics (the v1 judge knows 10 labels, `news_category` has 12; a `body` item is a
hash, not an article id; 159,401 sub-news pairs come from 28,799 articles), not a change of table names. The owner chose to **delete** the old eval stages instead. PR B (#212)
does it; what stays (the eval store, the MLflow tracking, the blind step-7 judge) and why is in its description and in T-220. **T-222** records the rebuild of an eval on `news_*`.
The full analysis and the defaults proposed for the rewire are in `docs/legacy/db_depuration_plan_a2_rewire_2026-11-18.md`.

## Documents and tasks that cite a revision

The prune removes the rows of every revision outside the keep list, and these documents cite such revisions (searched on master on
2026-11-17; the figures they report stay as history, they were scored once and are recorded in the document or its CSV):

| Revision cited (not in the keep list) | Documents |
|---|---|
| `summary-7a9bd2062bc4` (distilbart summaries) | `docs/summary_step.md`, `docs/sentiment_training.md`, `docs/ner_training.md`, `docs/companies_step.md`, `docs/category_results_2026_11_13/README.md`, `pipeline_metrics_report.md`, `pipeline_metrics_report_es.md` |
| `sentiment-5999fe131d8f` (the 190-piece check of #201 was scored on it) | `docs/sentiment_training.md`, both pipeline reports |
| `category-981dacf6f68d`, `category-65f33b55b26b` | `docs/category_step.md`, `docs/category_training.md`, `docs/category_results_2026_11_13/README.md` |
| `units-...`, `units4-...` | `docs/scope_training.md`, `docs/summary_step.md` |
| `sector-cf81d9b32565` (pruned by the purge on 2026-11-18: the live `body_sector` holds only `sector-417a93feb789`, and no kept snapshot row names it), `sector-417a93feb789` (kept) | `docs/sector_step.md`, `docs/gold_datasets.md` |

For each one PR C does one of two things, and the owner says which in the dry run: **(a)** the figure stays as history and the document
gets one line ("revision pruned on <date>; the figure is as recorded"); or **(b)** the rows behind a figure that must stay
reproducible are exported to a CSV in `docs/` first. Note that the documented sentiment revision (`sentiment-5999fe131d8f`) is not the
one in the owner's keep list (`sentiment-59863d15fc42`): the owner says whether that is the same model after a code change or a
different run. The open tasks that name revisions (T-185, T-189, T-190, T-191, T-194, T-206, T-212) are checked the same way.

## Left open (gates; none blocks this planning PR, all block the prune)

1. **The sector-step revision: answered (2026-11-18, "Gates answered" item (1)).** `sector-417a93feb789`, the only one, is kept. Before that answer the keep list had none; `body_sector` would have lost every row and the kept snapshot reads it
   (`news_snapshot_stage.py:445`). The answer covers `sector-417a93feb789` only. `sector-cf81d9b32565` is not kept and is gone from the live `body_sector` (checked 2026-11-20); the docs that cite it are listed in the "Documents and tasks that cite a revision" table, and their line is PR C's to write.
2. **The period revision: answered (2026-11-18).** Only `period-3b07b83ecec3` stays; a new delivery note naming it goes to the upstream team before `period-ae014b56b055` is pruned.
3. **The sentiment revision: answered (2026-11-18).** Only `sentiment-59863d15fc42` (and `category-a23569dba86f`) stay; the older two are pruned (see the Phase 0 listing).
4. **`eval_judgement`:** confirm unused (read the code in PR A/B).
5. **A flaky test in the group PR A adapts.** `tests/news_nlp/test_eval_calibration_judge.py::test_an_interrupt_cancels_the_queue_instead_of_paying_for_it`
   (`assert len(log) <= 3`) depends on timing (`time.sleep(0.05)` against a runner's speed): it failed on one CI run of this docs-only
   PR (`assert 6 <= 3`) and passed on the others. PR A makes it deterministic (a threading event instead of a sleep).
6. **The eval key mapping** (PR A) and **T-219's design** (authentication, jobs, the FTI services): settled at the start of each.
7. **Open tasks that overlap:** T-131, T-150 (part), T-189, T-212 (see "The rules that bind the work"); T-218 (testing the Docker
   image) is affected by the retarget in T-219.

(Closed here, not open: `config/pipeline_models.json` stays, read by step 7.)

## PR C2 check on the purged database (2026-11-18)

Run on scratch copies of the live `nlp.db` after the swap (SHA-256 `bcd7dfe3f82792c7…`, the copy checked) and of `nlp_backup.db`
(the file before the purge), with the C2 code (`init_schema` without the v1 tables).

* **`--status` of the nine steps** (the kept revisions named, `--source-db` the crawl DB read-only) exits 0 on both:
  `--step scope`; `--step cut`; `--step sector`; `--step summary --cut-revision units5-d3e2cbfa830e`; `--step companies --cut-revision units5-d3e2cbfa830e --summary-revision summary-llm-70cb6bb07292`;
  `--step category` and `--step sentiment` (each `--summary-revision summary-llm-70cb6bb07292 --companies-revision companies-f349536cf701`);
  `--step snapshot` (those three plus `--category-revision category-a23569dba86f --sentiment-revision sentiment-59863d15fc42`);
  `--step period --snapshot-revision snapshot-c3a753affa5d`. **Compared by parsing the JSON of each output:** the entries and values that name a
  pruned revision (`units-5f831c298bcb`, `units4-d3e2cbfa830e`, `summary-7a9bd2062bc4`, `category-65f33b55b26b`, `category-981dacf6f68d`,
  `sentiment-5999fe131d8f`, `snapshot-1b8ee113b05f`, `period-ae014b56b055`, `period-dd098e1c5b9f`) are dropped from both, and **the rest is
  equal in all nine**. Without that filter the diff has no line only in the purged copy (the backup's extra lines are those entries
  with their counts and dates). **Not compared:** the sector part of the snapshot status (run without `--sector-revision`; the CLI
  refuses it outside `--items sector`), and the rows themselves (the golden digest covers the delivered ones).
* **Two steps run for real** on the purged copy (they need no model, GPU or API): `--step snapshot` (written 0, waiting 0, 226,139
  company rows) and `--step period --catch-up` (0 periods to do). Afterwards **no v1 table had come back**: 20 user tables
  (the 19 of `init_schema` plus the lean `articles`, which was already in the file. A fresh RESULTS file had no `articles` table and the first lean-article copy failed there; from round 6 of #216 `init_schema` creates it, and the strict `xfail` is gone). Counted without SQLite's own `sqlite_*` tables (`sqlite_sequence`, `sqlite_stat1`). **Checked on the files themselves (2026-11-20, read-only, `immutable=1`):** the live `nlp.db` (purged) lists those 20 and the two `sqlite_*` tables; `nlp_backup.db` lists the same plus the six dropped tables (`article_category`, `article_entities`, `article_sentiment`, `article_summary`, `sector_summary`, `eval_judgement`). Neither file has an `article_entities_v1` or the `article_sentiment_v2_published_snapshot_2026_09_15` / `..._v4_experiment_2026_09_15` tables that the DDL file's header and the restore script name: the archival tables it mentions are not on the working store. `foreign_key_check` empty,
  `quick_check` ok, golden digest `2832f270e9a52457` (and `9a71c0f4d9001a85` with the period pruned).
* **`init_schema` on a copy of the purged file (round 6, 2026-11-20).** The purged `nlp.db` was copied with the SQLite backup API (the source opened read-only, `immutable=1`), and `init_schema` run on the copy with the round-6 code. Before and after: the `articles` definition is identical (`sqlite_master` text), 459,844 rows in `articles` both times, and 20 user tables both times. So `IF NOT EXISTS` leaves the purged file's `articles` as it is. The copy is a 1.63 GB scratch file, not in the repo.
* **Not run:** `scope`, `cut`, `summary`, `companies`, `category`, `sentiment` as work (they need models or the API; their status
  was read, and with nothing pending they would not run either). The lean-article upsert of the cut (`_ensure_article_row`) is
  covered by three tests in `tests/news_nlp/test_db.py`.
* The snapshot status reports **159,401 rows "built from another" summary, category and sentiment revision**: the accepted
  case of "Left open" (their `inputs` name pruned revisions).

