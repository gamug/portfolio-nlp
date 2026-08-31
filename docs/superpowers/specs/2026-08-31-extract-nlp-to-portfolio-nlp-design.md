# Extract NLP slice into `portfolio-nlp`

**Date:** 2026-08-31
**Status:** Approved for planning
**Repos touched:** `gamug/portfolio-nlp` (new home), `gamug/portfolio-data-mining` (source, NLP removed)

## Goal

Move all NLP development out of `portfolio-data-mining` and into this repo
(`portfolio-nlp`), following the source repo's tooling conventions (uv, ruff,
pre-commit, mypy, commitizen). Point the new repo at a dedicated database
`D:\thesis\data\nlp.db`. As the final step, migrate the NLP-related data from
`D:\thesis\data\urls.db` into `nlp.db`.

## Decisions (locked)

| Question | Decision |
|---|---|
| Migration data scope | NLP *processing* results only, **not** the "articles mining" corpus. |
| `articles` table in `nlp.db` | Migrate a **lean subset** — every `articles` column except `body_text`, only for rows referenced by a migrated NLP result. No `discovered_urls`. |
| Strip NLP from `portfolio-data-mining`? | Yes, as its own branch/PR on that repo. |
| New repo layout | Mirror the source exactly: `src/news_nlp/`, `apps/news_nlp_api.py`, `cli/news_nlp_cli.py`, `tests/news_nlp/`. |
| Run the migration? | Yes — build the script, test on a fixture, then run it for real against `D:\thesis\data`. |
| CI | Yes — GitHub Actions (lint + test). |
| PR merges | Claude merges each `portfolio-nlp` PR after CI is green (squash), so the next branch builds on it. The `portfolio-data-mining` PR is opened, not merged. |
| `train_ner.py` | Carry it along in the port. |

## Approach

**Mirror-and-repoint.** Copy the NLP slice of `portfolio-data-mining` verbatim
into `portfolio-nlp` with identical directory layout. Change only what must
change: the database path (`urls.db` -> `nlp.db`), project metadata, and docs.
A standalone migration script copies the NLP result tables plus a lean
`articles` subset from `urls.db` into `nlp.db`. A separate PR removes the NLP
slice from the source repo.

Rejected alternatives:
- **Flatten** (rename `news_nlp` -> `nlp`, collapse `apps/` + `cli/`): churn that
  risks breaking the ported tests for cosmetic gain; contradicts the "mirror
  exactly" decision.
- **git history transplant** (`git filter-repo`): the entrypoints don't live
  under a `news_nlp/` directory, path rewriting is fiddly, and grafting onto the
  existing single-commit `master` fights the granular-PR goal.

## Source inventory (what moves)

From `D:\projects\web_scraping\portfolio-data-mining`:

| Source path | Destination path | Change on copy |
|---|---|---|
| `src/news_nlp/__init__.py` | same | none (empty) |
| `src/news_nlp/categories.py` | same | none |
| `src/news_nlp/chunking.py` | same | none |
| `src/news_nlp/corrections.py` | same | none |
| `src/news_nlp/db.py` | same | DB-path fallback + docstring (see §3) |
| `src/news_nlp/pipeline.py` | same | none |
| `src/news_nlp/setup.py` | same | none (HF `snapshot_download` helper, run as `python -m news_nlp.setup`) |
| `src/news_nlp/train_ner.py` | same | none |
| `apps/news_nlp_api.py` | same | none (keeps `sys.path` bootstrap, port 8003) |
| `cli/news_nlp_cli.py` | same | none |
| `tests/news_nlp/**` (~13 files + `conftest.py`) | same | none |
| `docs/category-taxonomy.md` | `docs/category-taxonomy.md` | none |
| `docs/modules/news-nlp.md` | `docs/modules/news-nlp.md` | update `urls.db` references |

`src/news_nlp/` imports only stdlib + third-party (`torch`, `transformers`,
`tqdm`, `datasets`, `evaluate`, `huggingface_hub`, `numpy`, `dotenv`) + its own
relative modules. **No `src/common/` dependency** — nothing else needs to come
along.

`tests/news_nlp/conftest.py` defines its own minimal `ARTICLES_SCHEMA` and
`seed_article()` helper and `monkeypatch`es `db.DB_PATH`, so the ported suite is
self-contained and needs no real database.

## Design

### 1. Repo scaffold & tooling

New / adapted files in `portfolio-nlp`:

- **`pyproject.toml`**
  - `[project] name = "portfolio-nlp"`, `version = "0.1.0"`,
    `requires-python = ">=3.11,<3.12"`.
  - Dependencies — the NLP subset only:
    `httpx[http2]`, `fastapi`, `uvicorn[standard]`, `python-dotenv`, `pydantic`,
    `numpy`, `tqdm>=4.66`, `torch>=2.4,<2.7`, `transformers>=4.40`,
    `datasets>=2.19`, `accelerate>=0.30`, `evaluate>=0.4`, `seqeval>=1.2`,
    `huggingface_hub>=0.24`.
  - `[dependency-groups] dev`: `pytest>=8.2`, `ruff==0.16.3`, `mypy>=1.13`,
    `pre-commit>=4.0`. (`httpx` already in runtime deps covers FastAPI
    `TestClient`.)
  - `[tool.uv] package = false`.
  - `[tool.uv.sources] torch = { index = "pytorch-cu124" }` +
    `[[tool.uv.index]]` for `https://download.pytorch.org/whl/cu124`
    (`explicit = true`) — carried over verbatim.
  - Fresh `uv.lock` via `uv lock`.
- **`.python-version`** -> `3.11`
- **`ruff.toml`** (root) -> `extend = ".code_quality/ruff.toml"` (pointer, verbatim).
- **`.code_quality/ruff.toml`** — copied, then per-file-ignore globs re-scoped to
  this repo: keep the `apps/*` / `cli/*` `E402` ignore and the `tests/**`
  `PLR2004` ignore; drop globs naming `news_collector` / `extractor` / `pricing`
  / `sec_edgar`. Reconcile with the already-present untracked `.code_quality/`.
- **`.code_quality/mypy.ini`** — copied; keep `mypy_path` covering `src` + repo
  root (mirrors `pythonpath = src .`); drop references to non-ported packages.
- **`.pre-commit-config.yaml`** — already present untracked; keep as-is
  (pre-commit-hooks v6, ruff-pre-commit v0.16.3, mirrors-mypy, commitizen
  commit-msg + commitizen-branch pre-push).
- **`pytest.ini`** — `pythonpath = src .`, `testpaths = tests`. (`asyncio_mode`
  line dropped — no async tests in the news_nlp suite; no `pytest-asyncio` dep.)
- **`.gitignore`** — already present, already ignores `.env`; keep.
- **`.env.example`** — NEW, no secrets:
  `DATABASE_URL=D:/thesis/data/nlp.db` plus a note on the Hugging Face cache
  (`HF_HOME`) used by `python -m news_nlp.setup`.
- **`.env`** — untracked, never committed. Currently carries a live
  `TURSO_AUTH_TOKEN`; Turso is out of scope. Locally set
  `DATABASE_URL=D:/thesis/data/nlp.db`.
- **`.devcontainer/`** — already present untracked; keep. Adjust the
  thesis-dir bind-mount comment/path if it names `urls.db`.
- **`README.md`** — replace the one-line placeholder: what the repo is, `uv sync`,
  `python -m news_nlp.setup`, run the API (`uv run apps/news_nlp_api.py` ->
  `:8003/docs`), run the CLI (`uv run cli/news_nlp_cli.py --help`), run tests
  (`uv run pytest`).
- **`CLAUDE.md`** — NEW, repo-specific (structure, commands, DB shape,
  entrypoints). Not copied from the source repo's monorepo-scoped one.

### 2. Ported code

Copy the source inventory above into the identical layout. Only `db.py` changes
(see §3). Acceptance: `uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy` (per `.code_quality/mypy.ini`), and `uv run pytest` all pass.

The `apps/news_nlp_api.py` + `cli/news_nlp_cli.py` `sys.path.insert(0, .../"src")`
bootstrap is preserved unchanged so both run standalone.

### 3. Database repointing (`urls.db` -> `nlp.db`)

Single change site: `src/news_nlp/db.py`.

- Unset-`DATABASE_URL` fallback: `_PROJECT_ROOT / "data" / "urls.db"` ->
  `_PROJECT_ROOT / "data" / "nlp.db"`.
- Update the module docstring and the `DATABASE_URL` comment block that
  currently say `data/urls.db`.
- Relative-path-resolved-against-project-root logic stays (tests depend on the
  `DB_PATH` monkeypatch; `.env` supplies the absolute `D:/thesis/data/nlp.db`).
- Repo-wide sweep for literal `urls.db` / `urls\.db` in comments, docstrings,
  docs, README — repoint or delete.

`init_schema()` creates (unchanged): `article_sentiment`, `article_entities`
(+ `idx_article_entities_article_id`), `article_summary`, `article_category`,
`sector_summary`. Each of the first four has `article_id ... REFERENCES
articles(id)`; SQLite allows creating these before `articles` exists, and the
migration (§4) populates the lean `articles` table so the FKs resolve.

### 4. Migration script + execution

**`scripts/migrate_from_urls_db.py`** — stdlib `sqlite3` + `argparse`, no
third-party deps.

- Args: `--source` (default `D:/thesis/data/urls.db`), `--dest`
  (default `D:/thesis/data/nlp.db`), `--dry-run`.
- Steps:
  1. Refuse to run if `--source` is missing. Create `--dest` if absent.
  2. Open `--dest`, call `news_nlp.db.connect()` + `init_schema()` on it, then
     create the lean `articles` table if absent. Lean `articles` DDL = the
     source `articles` schema **minus `body_text`** (columns confirmed against
     `conftest.ARTICLES_SCHEMA` + the live `urls.db` schema during planning:
     `id, ticker, company, gics_sector, gics_sub_industry, title, author,
     pub_date, fetched_at, word_count, source_domain, fetch_status,
     http_status_code`, plus any extra columns the live table has except
     `body_text` / any other large-text column).
  3. `ATTACH '<source>' AS src` (read-only via `file:...?mode=ro` URI).
  4. Copy, in FK-safe order, with `INSERT OR IGNORE ... SELECT` (idempotent):
     - `article_sentiment`, `article_entities`, `article_category`,
       `article_summary`, `sector_summary` — full copies from `src`.
     - lean `articles` — only rows whose `id` appears in any of the copied
       result tables (`SELECT ... FROM src.articles WHERE id IN (
       SELECT article_id FROM article_sentiment UNION ... )`), selecting all
       columns except `body_text`.
  5. `PRAGMA foreign_key_check` on `--dest`; abort with a non-zero exit and a
     clear message if it reports rows.
  6. Print a per-table before/after row-count table and total.
- `--dry-run` does steps 1-3 and prints the counts it *would* copy.

**Fixture test** — `tests/migration/test_migrate_from_urls_db.py`: builds a tiny
in-`tmp_path` `urls.db` (full `articles` incl. `body_text` + a few rows in each
result table + one orphan result row), runs the script, asserts: result tables
copied fully, `articles` copied without `body_text`, orphan-referenced article
absent (or its result row skipped — decide in planning; prefer copying only
referenced articles and letting `INSERT OR IGNORE` + `foreign_key_check` catch
a genuine orphan), re-running is a no-op, `foreign_key_check` clean.

**Real run** — after the script PR merges: run
`uv run python scripts/migrate_from_urls_db.py` (defaults) to create
`D:\thesis\data\nlp.db`. Capture the row-count output in the PR / final report.
`urls.db` is opened read-only; it is never modified.

### 5. Strip NLP from `portfolio-data-mining` (separate PR, not merged by Claude)

Branch `chore/extract-news-nlp-to-portfolio-nlp` on `gamug/portfolio-data-mining`:

- Delete `src/news_nlp/`, `apps/news_nlp_api.py`, `cli/news_nlp_cli.py`,
  `tests/news_nlp/`, `docs/modules/news-nlp.md`.
- `pyproject.toml`: remove NLP-only deps (`torch`, `transformers`, `datasets`,
  `accelerate`, `evaluate`, `seqeval`, `huggingface_hub`) and the
  `[tool.uv.sources] torch` + `[[tool.uv.index]] pytorch-cu124` blocks; regen
  `uv.lock`.
- `apps/gateway.py`: remove the `news_nlp` mount and its import.
- Docs: `CLAUDE.md` (pipeline diagram `... -> news_nlp`, the `src/` package
  table row, the `apps/` vs `cli/` table row, the shared-DB paragraph that names
  `article_sentiment`/`article_entities`/`article_category`, the
  `uv run pytest tests/news_nlp` command), `README.md`, `.env.example`
  (NLP-specific comments), `pytest.ini` comment listing packages.
- Open as a PR with a description linking the `portfolio-nlp` repo. Leave for the
  user to merge.

### 6. Branch / commit / PR sequence

All `portfolio-nlp` PRs target `master`; Claude squash-merges each once CI is
green, then cuts the next branch from the updated `master`.

| # | Branch | Contents | Commits (illustrative) |
|---|---|---|---|
| 1 | `chore/project-scaffold` | §1 tooling | `build: uv pyproject + lock`; `chore: ruff + .code_quality config`; `chore: pre-commit + pytest + .env.example`; `docs: README` |
| 2 | `feat/port-news-nlp-package` | `src/news_nlp/**` (db.py still pointing at `urls.db` default here, or repoint in PR 5 — repoint in PR 5) | `feat: port news_nlp package from portfolio-data-mining` |
| 3 | `feat/port-api-and-cli` | `apps/news_nlp_api.py`, `cli/news_nlp_cli.py` | `feat: port news_nlp FastAPI app`; `feat: port news_nlp CLI` |
| 4 | `test/port-news-nlp-suite` | `tests/news_nlp/**`, `pytest.ini` wiring | `test: port news_nlp test suite`; `fix: <any import/path adjustments to get green>` |
| 5 | `feat/repoint-database-to-nlp-db` | §3 | `feat: default DATABASE_URL to data/nlp.db`; `docs: repoint urls.db references` |
| 6 | `docs/claude-md-and-module-docs` | `CLAUDE.md`, `docs/category-taxonomy.md`, `docs/modules/news-nlp.md` | `docs: add CLAUDE.md`; `docs: port category taxonomy + module doc` |
| 7 | `ci/lint-and-test-workflow` | `.github/workflows/ci.yml` | `ci: lint + test on push/PR` |
| 8 | `feat/migration-script` | `scripts/migrate_from_urls_db.py` + `tests/migration/` | `feat: urls.db -> nlp.db migration script`; `test: migration fixture test` |
| 9 | *(operational, no PR)* | run the migration against `D:\thesis\data` | — |
| 10 | `chore/extract-news-nlp-to-portfolio-nlp` **on `portfolio-data-mining`** | §5 | `chore: remove news_nlp (moved to portfolio-nlp)` |

Ordering notes:
- PR 2 ports `db.py` with its default still `data/nlp.db`? No — PR 2 copies
  `db.py` **verbatim** (default `urls.db`); PR 5 is the deliberate, reviewable
  repoint. PR 4's tests pass regardless because `conftest.py` monkeypatches
  `DB_PATH`.
- PR 7 (CI) can land any time after PR 4; placed here so the workflow runs a
  green suite on arrival.
- PR 8 depends on PR 5 (imports `news_nlp.db.init_schema` and expects the
  `nlp.db` naming).

### 7. CI — `.github/workflows/ci.yml`

- Trigger: `push` to `master`, `pull_request`.
- Runner: `ubuntu-latest`, Python 3.11.
- Steps: checkout; `astral-sh/setup-uv`; `uv sync --no-sources` — CI is
  CPU-only, and `--no-sources` makes uv ignore the `[tool.uv.sources]` cu124 pin
  and resolve `torch` from PyPI's CPU wheels. Then `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run mypy`, `uv run pytest` (see model-test
  handling below).
- `tests/news_nlp/test_pipeline_run.py` and any test importing `torch` heavily /
  loading models: confirm during planning whether they run without a GPU and
  without downloading multi-GB checkpoints. If some require models, mark them
  `@pytest.mark.slow` / `skipif` on a `CI` env var and exclude them from the CI
  run (documented in `CLAUDE.md`). The non-model tests (db, corrections,
  endpoints, chunking, categories) are the CI gate.

## Testing strategy

| Layer | How |
|---|---|
| Ported unit/endpoint tests | `uv run pytest` — unchanged suite, must stay green (PR 4). |
| Lint / format / types | `uv run ruff check .`, `ruff format --check .`, `mypy` — clean (PR 1-2). |
| pre-commit | `uv run pre-commit run --all-files` — clean. |
| Migration script | Fixture test (PR 8) + `--dry-run` against real `urls.db` before the real run + `PRAGMA foreign_key_check` after. |
| End-to-end sanity | After migration: `uv run cli/news_nlp_cli.py` query subcommand (or a short `sqlite3` count) against `D:\thesis\data\nlp.db` confirms non-zero rows in each result table and that `articles` has no `body_text` column. |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| CI can't install the cu124 torch pin | `uv sync --no-sources` in CI -> PyPI CPU torch. |
| Model-loading tests need GPU / large downloads in CI | Identify in planning; `skipif(CI)` + document; CI gates on the non-model tests. |
| Live `urls.db` `articles` schema has columns beyond `conftest.ARTICLES_SCHEMA` | Migration reads the real `PRAGMA table_info(articles)` and copies all columns except `body_text` (and any other obvious large-text/BLOB column), rather than hardcoding the list. |
| Orphan NLP result rows (no matching `articles.id`) | Copy only referenced `articles`; `PRAGMA foreign_key_check` after; abort + report if any result row is a true orphan. |
| `urls.db` locked / WAL sidecar files present | Open source with `mode=ro` URI; never write to it; `busy_timeout` on the dest connection. |
| `.env` Turso token committed by accident | `.gitignore` already lists `.env`; only `.env.example` (no secrets) is tracked; verify `git status` before each commit. |
| Old-repo removal breaks `apps/gateway.py` import | PR 10 removes the mount + import together; run `portfolio-data-mining`'s remaining suite before opening the PR. |

## Out of scope

- Turso / libSQL backend (`.env` token left unused).
- Any change to `news_collector` / `extractor` / `pricing` / `sec_edgar`.
- Re-running the NLP pipeline to regenerate results — migration copies existing rows.
- Preserving per-file git history from the source repo (plain copy + fresh commits).
