# The pipeline API (T-219, FR-025, D45; owner, 2026-11-17)

`apps/pipeline_api.py`: a FastAPI service that runs **each step** of the stepped pipeline, runs the **whole pipeline**, and serves the **pinned
models of the stages**. It is the service the Docker image runs (`Dockerfile`, port 8003, `/health`). The v1 API (`apps/news_nlp_api.py`, v1 tables)
is untouched and still in the repository: T-220 removes it. Run it with `uv run apps/pipeline_api.py` (127.0.0.1:8003, `/docs` for the schema).

## How a step runs

The API does **not** reimplement a step. A request's options become the command line of `cli/news_nlp_steps.py`, parsed and checked by the CLI's own
`build_parser()` and `check_options()`, and the run is the CLI's own `main_<step>` function (`src/news_nlp/service/steps_runner.py`, which loads the CLI file as
a module and changes nothing in it). So a refusal of the command line is a refusal of the API (HTTP 422, the CLI's message), and a run writes the rows
the CLI writes: the T-219 acceptance. Revisions are explicit options (`summary_revision`, `companies_revision`, ...), never "latest".

**Options the API refuses** (422): `fetch`, `model_dir`, `allow_unpinned`, `source_db`, `results_db`, `reset`. They act on files, the stores or the weights, and the
service has **no authentication** (owner's choice): the stores come from the environment as for the CLI, and `reset` (it deletes rows) stays with the command line.

## Routes

| Route | What |
|---|---|
| `GET /health` | `{"status": "ok"}` (the Docker healthcheck) |
| `GET /steps` | the steps in the order of the pipeline and the refused options |
| `GET /steps/{step}/status?revision=&snapshot_revision=` | the counts of the step, read-only, no model loaded (`pipeline_step_state`; for `snapshot` and `period` their own status). Not the whole `--status` of the CLI (no model details): run the CLI for those |
| `POST /steps/{step}/run` body `{"options": {...}}` | starts a job, 202 with the job |
| `POST /pipeline/run` body `{"steps": [{"step": "scope", "options": {...}}, ...]}` | one job that runs the steps in the order of the pipeline (unique steps, that order, else 422); the first failure stops it and the rest are `skipped` |
| `GET /jobs`, `GET /jobs/{id}` | a job: status, the last progress line (the bar of the CLI), and for each step its JSON outputs and log |
| `POST /jobs/{id}/cancel` | stops a **pipeline** job between steps |
| `GET /fti/stages`, `POST /fti/{stage}/predict` body `{"items": [...]}` | the stage models (below) |

## Jobs

One worker, jobs in memory (owner's choice). There is one GPU and a step runs for hours (the corpus run takes 8 to 10 hours), so a request while a job (or a
stage-model call) holds the GPU is **409**, not queued. Jobs are lost with the process; that is safe because every step is idempotent and its state is in the
tables, so a restart resumes where it stopped. **A running step cannot be stopped in the middle**: the CLI has no hook for it, and `cancel` only stops a
pipeline job before its next step; to stop a step, restart the service. A job's stdout is parsed into the JSON documents the CLI printed (a result, then a status).
No table was added for jobs: T-220 prunes the schema.

## The stage models (the FTI services)

For `scope`, `sector`, `companies`, `category` and `sentiment`, the object the step itself runs (`ScopeClassifier`, `SectorClassifier`, `CompanyFinder`,
`CategoryFinder`, `SentimentFinder`) is built the way the step builds it and asked about 1 to 32 ad-hoc items; **no row is written**. Items: scope `{title, body}`,
sector `{title, text}`, companies `{text}`, category `{text, company}`, sentiment `{text, company, relevance}`. One stage stays loaded (a 6 GB card): asking for another frees
the first. A call while a job runs is 409 and the other way round; a stage that cannot be served here (no GPU, no model folder, no universe in the SOURCE store) is 503.

**Not exposed, and why:** a `features` route (the v2 stages have no `Feature` class; only the NER of step 3 uses `fti.Feature`, so the features are internal to each finder), **training** (hours, writes weights, can publish: it stays in
the scripts), `cut` and `summary` (the cut needs the trained attribution folder and works body by body; the summary is a paid API call: both are steps, not stage models).

## What was and was not checked

Hermetic tests (`tests/news_nlp/test_pipeline_api.py`): the command line, the CLI's refusals, the real CLI path on an empty store (connect, schema, dispatch, the CLI's own
refusal), the jobs (order, failure, cancel between steps, the GPU gate), the routes, the stage-model gate with fake models. **Not checked here:** a step on a real corpus or a GPU, a stage model on
a GPU, the Docker image (T-218 adds a call through the API to its test). The service has no authentication: the port is for a private network.
