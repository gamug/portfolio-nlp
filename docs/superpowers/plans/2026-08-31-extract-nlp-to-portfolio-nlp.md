# Extract NLP slice into portfolio-nlp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the NLP slice of `portfolio-data-mining` into this repo as a standalone uv/ruff/pre-commit project pointed at `D:\thesis\data\nlp.db`, then migrate the NLP data out of `urls.db`.

**Architecture:** Verbatim file-copy of `src/news_nlp/`, `apps/news_nlp_api.py`, `cli/news_nlp_cli.py`, `tests/news_nlp/` into an identical layout. Only `db.py`'s database-path default changes. A standalone stdlib-`sqlite3` script copies the five NLP result tables plus a `body_text`-free `articles` subset from `urls.db` into `nlp.db`. A separate PR strips the NLP code from `portfolio-data-mining`.

**Tech Stack:** Python 3.11, uv, ruff 0.16.3, mypy, pre-commit + commitizen, FastAPI, PyTorch (cu124 wheels locally, CPU wheels in CI), transformers, SQLite, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-31-extract-nlp-to-portfolio-nlp-design.md`

## Global Constraints

- Python: `requires-python = ">=3.11,<3.12"`; `.python-version` = `3.11`.
- ruff pinned `==0.16.3` (dep group + `.pre-commit-config.yaml` rev already at `v0.16.3`).
- Source of truth for lint config: `.code_quality/ruff.toml`; root `ruff.toml` is only `extend = ".code_quality/ruff.toml"`.
- `mypy` config: `.code_quality/mypy.ini`, must contain `mypy_path = src`.
- Import style: `src/news_nlp/` uses relative imports (`from .categories import ...`); `apps/`/`cli/` prepend `src/` to `sys.path` then import `news_nlp.*` absolutely. Do not rewrite either.
- `pythonpath = src .` in `pytest.ini` (the `.` lets `tests/news_nlp/conftest.py` import `apps.news_nlp_api`).
- DB path indirection lives at exactly one site: `src/news_nlp/db.py` `DB_PATH`. Unset-`DATABASE_URL` fallback must be `_PROJECT_ROOT / "data" / "nlp.db"`.
- `.env` is git-ignored and untracked; it currently holds a live `TURSO_AUTH_TOKEN`. NEVER `git add .env`. Only `.env.example` (no secrets) is tracked.
- FastAPI app title stays `"news-nlp"`, port stays `8003`.
- Model constants unchanged: `SENTIMENT_MODEL="ProsusAI/finbert"`, `NER_MODEL="gamug/sec-bert-finer-ord-ner"`, `CATEGORY_MODEL="MoritzLaurer/deberta-v3-base-zeroshot-v2.0"`, `SUMMARY_MODEL="sshleifer/distilbart-cnn-12-6"`.
- Commit messages: Conventional Commits (commitizen `commit-msg` hook enforces it). Branch names: commitizen `commitizen-branch` runs at `pre-push`.
- Source repo absolute path (read-only reference): `D:/projects/web_scraping/portfolio-data-mining`.
- This repo absolute path: `D:/projects/web_scraping/portfolio-nlp`.
- Live source DB: `D:/thesis/data/urls.db` (open read-only, never write). Target: `D:/thesis/data/nlp.db`.

### PR / merge workflow (applies to every task)

1. `git switch -c <branch>` from an up-to-date `master`.
2. Do the task's steps, committing as the steps say.
3. `git push -u origin <branch>`.
4. `gh pr create --fill --base master` (expand body with a short summary + `Spec:`/`Plan:` links).
5. Wait for CI (once Task 7 has landed). When green: `gh pr merge --squash --delete-branch`.
6. `git switch master && git pull` before starting the next task.

Tasks 1–3 land before CI exists; merge them after local verification (`uv run ruff check .`, `uv run mypy`, `uv run pytest`) passes. Re-run CI expectations retroactively is not required.

Task 10 targets the **other** repo and its PR is **left open** (not merged).

---

## File Structure

New/changed files in `portfolio-nlp`, by responsibility:

| Path | Responsibility | Task |
|---|---|---|
| `pyproject.toml` | project metadata + NLP dependency set + uv/torch-cu124 index config | 1 |
| `uv.lock` | pinned resolution (generated) | 1 |
| `.python-version` | `3.11` | 1 |
| `ruff.toml` | one-line pointer to `.code_quality/ruff.toml` | 1 |
| `.code_quality/ruff.toml` | lint/format rules (trim dead per-file-ignore globs) | 1 |
| `.code_quality/mypy.ini` | type-check rules + `mypy_path = src` | 1 |
| `pytest.ini` | `pythonpath = src .`, `testpaths = tests`, `asyncio_mode = auto` | 1 |
| `.pre-commit-config.yaml` | already present untracked — track as-is | 1 |
| `.devcontainer/**` | already present untracked — track, fix one stale comment | 1 |
| `.gitignore` | already present — add `data/` + `*.db*` rules | 1 |
| `.env.example` | documented env template, no secrets | 1 |
| `README.md` | real project readme | 1 |
| `src/news_nlp/__init__.py` `categories.py` `chunking.py` `corrections.py` `pipeline.py` `setup.py` `train_ner.py` | verbatim copy | 2 |
| `src/news_nlp/db.py` | verbatim copy in Task 2; DB-path default edited in Task 5 | 2 / 5 |
| `apps/news_nlp_api.py` | verbatim copy | 3 |
| `cli/news_nlp_cli.py` | verbatim copy | 3 |
| `tests/news_nlp/**` (13 files) | verbatim copy | 4 |
| `tests/news_nlp/test_db_defaults.py` | NEW — asserts the `nlp.db` fallback | 5 |
| `CLAUDE.md` | repo guide for future agents | 6 |
| `docs/category-taxonomy.md` | verbatim copy | 6 |
| `docs/modules/news-nlp.md` | copy, repoint `urls.db` → `nlp.db` mentions | 6 |
| `.github/workflows/ci.yml` | lint + type + test on push/PR | 7 |
| `scripts/migrate_from_urls_db.py` | one-shot `urls.db` → `nlp.db` copier | 8 |
| `tests/migration/__init__.py` | package marker (empty) | 8 |
| `tests/migration/test_migrate_from_urls_db.py` | fixture-DB test of the copier | 8 |

---

## Task 1: Project scaffold & tooling

**Branch:** `chore/project-scaffold`

**Files:**
- Create: `pyproject.toml`, `.python-version`, `ruff.toml`, `pytest.ini`, `.env.example`, `README.md`
- Create (generated): `uv.lock`
- Modify: `.code_quality/ruff.toml`, `.code_quality/mypy.ini`, `.gitignore`, `.devcontainer/devcontainer.json`
- Track as-is: `.pre-commit-config.yaml`, `.devcontainer/devcontainer-lock.json`

**Interfaces:**
- Consumes: nothing.
- Produces: a repo where `uv sync` succeeds and `uv run ruff check .` / `uv run mypy` / `uv run pytest` run (no files to act on yet). Dependency names later tasks rely on: `fastapi`, `uvicorn`, `httpx`, `pydantic`, `python-dotenv`, `numpy`, `tqdm`, `torch`, `transformers`, `datasets`, `accelerate`, `evaluate`, `seqeval`, `huggingface_hub`; dev: `pytest`, `pytest-asyncio`, `ruff==0.16.3`, `mypy`, `pre-commit`.

- [ ] **Step 1: Branch**

```bash
cd /d/projects/web_scraping/portfolio-nlp
git switch -c chore/project-scaffold
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "portfolio-nlp"
version = "0.1.0"
description = "Financial-news NLP pipeline (FinBERT sentiment + SEC-BERT NER + zero-shot category + summarization), extracted from portfolio-data-mining"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
    # --- shared web/runtime -----------------------------------------------
    "httpx[http2]>=0.27.0",
    "fastapi==0.141.1",
    "uvicorn[standard]==0.52.1",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0",
    "numpy",
    "tqdm>=4.66",
    # --- news_nlp (sentiment + NER + category + summarization) -----------
    # torch is pinned to what the cu124 wheel index (see [tool.uv.sources]
    # below) publishes - that index stops at 2.6.0, so an unbounded/newer
    # requirement here would make uv fall back to a plain PyPI build.
    "torch>=2.4,<2.7",
    "transformers>=4.40",
    "datasets>=2.19",
    "accelerate>=0.30",
    "evaluate>=0.4",
    "seqeval>=1.2",
    "huggingface_hub>=0.24",
]

[dependency-groups]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.24",
    "pre-commit>=4.0",
    "ruff==0.16.3",
    "mypy>=1.13",
]

[tool.uv]
package = false

# torch is a direct [project] dependency, but uv still needs to know which
# wheel to fetch. This pins it to the same cu124 build the old
# `pip install torch --index-url https://download.pytorch.org/whl/cu124`
# step used, so `uv sync`/`uv lock` reproduce it exactly. CI passes
# `uv sync --no-sources` to ignore this and take CPU wheels from PyPI.
[tool.uv.sources]
torch = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

- [ ] **Step 3: Write `.python-version`**

```
3.11
```

- [ ] **Step 4: Write `ruff.toml`**

```toml
# Root pointer so bare `ruff`/`uvx ruff` invocations and editor integrations
# resolve settings without needing `--config` on the command line every time.
# .code_quality/ruff.toml is the actual source of truth (also what
# .pre-commit-config.yaml points ruff-pre-commit at explicitly).
extend = ".code_quality/ruff.toml"
```

- [ ] **Step 5: Trim dead per-file-ignore globs in `.code_quality/ruff.toml`**

The copied file (identical to source) has two globs that match nothing in this repo. Delete these two lines from the `[lint.per-file-ignores]` block:

```toml
"../apps/gateway.py" = ["B008"]
"../src/*/api/**/*.py" = ["B008"]
```

Keep `"../apps/*_api.py" = ["B008"]`, `"../apps/*.py" = ["E402"]`, `"../cli/*.py" = ["E402"]`, `"../tests/**/*.py" = ["PLR2004"]` unchanged.

- [ ] **Step 6: Add `mypy_path` to `.code_quality/mypy.ini`**

The copied file stops at `ignore_missing_imports = True`. Append:

```ini
# apps/*.py and cli/*.py bootstrap `src/` onto sys.path at runtime, so
# without this mypy can't resolve `news_nlp.*` from those entrypoints and
# silently treats every such import as Any. Mirrors pytest.ini's
# `pythonpath = src .` for the same reason.
mypy_path = src
```

- [ ] **Step 7: Add DB-file rules to `.gitignore`**

Under the `# Environments` section (after the `.env` / `.envrc` lines), add:

```
# local SQLite databases (the real DB lives at D:\thesis\data\nlp.db)
data/
*.db
*.db-shm
*.db-wal
```

- [ ] **Step 8: Fix the stale workspace name in `.devcontainer/devcontainer.json`**

In the `mounts` comment, change `/workspaces/portfolio-data-mining` to `/workspaces/portfolio-nlp`. Leave everything else (image, GPU `runArgs`, `D:\thesis` mount) unchanged.

- [ ] **Step 9: Write `.env.example`**

```
# Copy this file to .env and adjust. .env is git-ignored.
# news_nlp reads DATABASE_URL at import time (src/news_nlp/db.py). A relative
# value is resolved against the repo root; an absolute value is used as-is.
# Unset -> defaults to <repo>/data/nlp.db.
DATABASE_URL=D:/thesis/data/nlp.db

# Optional: where `python -m news_nlp.setup` caches the Hugging Face models
# (FinBERT, SEC-BERT NER, DeBERTa-v3 zero-shot, distilbart-cnn). Defaults to
# the platform HF cache if unset.
# HF_HOME=D:/thesis/models/hf
```

- [ ] **Step 10: Write `README.md`**

```markdown
# portfolio-nlp

Financial-news NLP pipeline, extracted from
[`portfolio-data-mining`](https://github.com/gamug/portfolio-data-mining):

- **Sentiment** — FinBERT (`ProsusAI/finbert`)
- **NER** — SEC-BERT (`gamug/sec-bert-finer-ord-ner`)
- **Category** — zero-shot DeBERTa-v3 against a fixed 10-category taxonomy
  (`docs/category-taxonomy.md`)
- **Summarization** (opt-in) — distilbart per-article + per-sector-week roll-ups

Reads an `articles` table and writes `article_sentiment`, `article_entities`,
`article_category`, `article_summary`, `sector_summary`, keyed by `article_id`.

## Setup

```bash
uv sync
cp .env.example .env          # then set DATABASE_URL
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
uv run python -m news_nlp.setup   # download the HF models
```

## Run

```bash
uv run apps/news_nlp_api.py            # FastAPI -> http://127.0.0.1:8003/docs
uv run cli/news_nlp_cli.py --help      # batch pipeline, no server
uv run cli/news_nlp_cli.py --limit 50  # process 50 pending articles
```

## Test / lint

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy
```

## Database

`DATABASE_URL` (a filesystem path today) selects the SQLite file; unset it
defaults to `data/nlp.db` under the repo root. The working database is
`D:\thesis\data\nlp.db`. `scripts/migrate_from_urls_db.py` populates it from
the legacy shared `urls.db`.
```

- [ ] **Step 11: Generate the lockfile and sync**

Run: `uv lock`
Then: `uv sync`
Expected: both succeed; `.venv/` created; `torch` resolves from the `pytorch-cu124` index.

- [ ] **Step 12: Smoke-check the toolchain**

Run: `uv run ruff check .`
Expected: `All checks passed!` (no Python files yet).
Run: `uv run python -c "import fastapi, torch, transformers, huggingface_hub; print('ok')"`
Expected: `ok`.

- [ ] **Step 13: Commit (split into logical commits)**

```bash
git add pyproject.toml uv.lock .python-version
git commit -m "build: uv project with the news_nlp dependency subset"

git add ruff.toml .code_quality/ruff.toml .code_quality/mypy.ini
git commit -m "chore: ruff pointer + .code_quality config for this repo"

git add pytest.ini .pre-commit-config.yaml .devcontainer .gitignore .env.example
git commit -m "chore: pytest, pre-commit, devcontainer, gitignore, env template"

git add README.md
git commit -m "docs: project README"
```

- [ ] **Step 14: Push, PR, merge after local checks**

```bash
git push -u origin chore/project-scaffold
gh pr create --fill --base master
```
After `uv run ruff check .` passes: `gh pr merge --squash --delete-branch`, then `git switch master && git pull`.

---

## Task 2: Port the `news_nlp` package

**Branch:** `feat/port-news-nlp-package`

**Files:**
- Create: `src/news_nlp/__init__.py`, `categories.py`, `chunking.py`, `corrections.py`, `db.py`, `pipeline.py`, `setup.py`, `train_ner.py` (all copied verbatim from `D:/projects/web_scraping/portfolio-data-mining/src/news_nlp/`)

**Interfaces:**
- Consumes: the dependency set from Task 1.
- Produces (names later tasks rely on):
  - `news_nlp.db.connect(db_path: str | os.PathLike) -> sqlite3.Connection`
  - `news_nlp.db.init_schema(conn: sqlite3.Connection) -> None` (creates `article_sentiment`, `article_entities` + `idx_article_entities_article_id`, `article_summary`, `article_category`, `sector_summary`)
  - `news_nlp.db.DB_PATH: pathlib.Path` (module-level; monkeypatched by tests)
  - `news_nlp.pipeline.run_pipeline(...)`, stage fns `run_sentiment_stage` / `run_ner_stage` / `run_category_stage` / company+sector summary stages, module constants `SENTIMENT_MODEL` / `NER_MODEL` / `CATEGORY_MODEL` / `SUMMARY_MODEL` / `DEVICE`
  - `news_nlp.setup.snapshot_download`, `news_nlp.setup.AutoConfig` (import surface tests patch)
  - `news_nlp.corrections.*`, `news_nlp.categories.CATEGORY_LABELS` / `CATEGORY_SLUGS` / `OTHER_LABEL`, `news_nlp.chunking.chunk_text` / `merge_char_spans`

- [ ] **Step 1: Branch**

```bash
cd /d/projects/web_scraping/portfolio-nlp
git switch -c feat/port-news-nlp-package
```

- [ ] **Step 2: Copy the package verbatim**

```bash
mkdir -p src/news_nlp
cp "/d/projects/web_scraping/portfolio-data-mining/src/news_nlp/__init__.py" \
   "/d/projects/web_scraping/portfolio-data-mining/src/news_nlp/categories.py" \
   "/d/projects/web_scraping/portfolio-data-mining/src/news_nlp/chunking.py" \
   "/d/projects/web_scraping/portfolio-data-mining/src/news_nlp/corrections.py" \
   "/d/projects/web_scraping/portfolio-data-mining/src/news_nlp/db.py" \
   "/d/projects/web_scraping/portfolio-data-mining/src/news_nlp/pipeline.py" \
   "/d/projects/web_scraping/portfolio-data-mining/src/news_nlp/setup.py" \
   "/d/projects/web_scraping/portfolio-data-mining/src/news_nlp/train_ner.py" \
   src/news_nlp/
```

- [ ] **Step 3: Verify every module imports**

Run:
```bash
uv run python -c "import news_nlp.categories, news_nlp.chunking, news_nlp.corrections, news_nlp.db, news_nlp.pipeline, news_nlp.setup, news_nlp.train_ner; print('all import ok')"
```
Expected: `all import ok`. (This confirms `torch`/`transformers`/`datasets`/`evaluate`/`huggingface_hub` are all satisfied and the relative imports resolve. It also creates a `data/nlp.db`-less `DB_PATH` pointing at `data/urls.db` still — that's fine; Task 5 repoints it.)

- [ ] **Step 4: Lint + type-check the copied code**

Run: `uv run ruff check .`
Expected: `All checks passed!` (the source repo already keeps this code ruff-clean).
Run: `uv run ruff format --check .`
Expected: no reformat needed.
Run: `uv run mypy src/news_nlp`
Expected: `Success: no issues found`. If mypy reports missing-import noise for third-party libs, confirm `.code_quality/mypy.ini` has `ignore_missing_imports = True` (it does) — do not add per-module overrides.

- [ ] **Step 5: Commit**

```bash
git add src/news_nlp
git commit -m "feat: port news_nlp package from portfolio-data-mining"
```

- [ ] **Step 6: Push, PR, merge after local checks**

```bash
git push -u origin feat/port-news-nlp-package
gh pr create --fill --base master
```
After ruff + mypy pass: squash-merge, delete branch, update `master`.

---

## Task 3: Port the API and CLI entrypoints

**Branch:** `feat/port-api-and-cli`

**Files:**
- Create: `apps/news_nlp_api.py`, `cli/news_nlp_cli.py` (verbatim copies)

**Interfaces:**
- Consumes: `news_nlp.*` from Task 2.
- Produces:
  - `apps.news_nlp_api.app` (FastAPI instance, title `"news-nlp"`)
  - `apps.news_nlp_api.pipeline_status` (has `.reset()`) — used by `tests/news_nlp/conftest.py` `client` fixture
  - `apps.news_nlp_api` re-exports `db`, `pipeline`, `corrections` as module attributes (tests do `monkeypatch.setattr(app_module.pipeline, "run_pipeline", ...)`)
  - `cli/news_nlp_cli.py` runnable as `uv run cli/news_nlp_cli.py --help` with `--limit` / `--summarize` flags

- [ ] **Step 1: Branch**

```bash
git switch -c feat/port-api-and-cli
```

- [ ] **Step 2: Copy both entrypoints verbatim**

```bash
mkdir -p apps cli
cp "/d/projects/web_scraping/portfolio-data-mining/apps/news_nlp_api.py" apps/
cp "/d/projects/web_scraping/portfolio-data-mining/cli/news_nlp_cli.py" cli/
```

Do NOT copy `apps/__init__.py` from the source (its docstring is monorepo-scoped). Create a minimal one so `tests/news_nlp/conftest.py`'s `from apps.news_nlp_api import ...` resolves as a package:

```bash
printf '' > apps/__init__.py
```

- [ ] **Step 3: Verify the API imports and the app builds**

Run:
```bash
uv run python -c "import sys; sys.path.insert(0, 'src'); from apps.news_nlp_api import app, pipeline_status; print(app.title); pipeline_status.reset(); print('api ok')"
```
Expected: prints `news-nlp` then `api ok`.

- [ ] **Step 4: Verify the CLI runs**

Run: `uv run cli/news_nlp_cli.py --help`
Expected: usage text listing `--limit` and `--summarize`. Exit 0.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check .`
Expected: pass. The `E402` after `sys.path.insert(...)` in both files is covered by the `../apps/*.py` / `../cli/*.py` per-file-ignore.
Run: `uv run mypy apps cli`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add apps cli
git commit -m "feat: port news_nlp FastAPI app and CLI entrypoint"
```

- [ ] **Step 7: Push, PR, merge after local checks**

---

## Task 4: Port the test suite

**Branch:** `test/port-news-nlp-suite`

**Files:**
- Create: `tests/news_nlp/` — `conftest.py` + these 12 test modules, verbatim:
  `test_category_pipeline.py`, `test_correction_endpoints.py`, `test_corrections.py`,
  `test_db_module.py`, `test_db_queries.py`, `test_pipeline_endpoints.py`,
  `test_pipeline_progress.py`, `test_pipeline_run.py`, `test_query_endpoints.py`,
  `test_setup.py`, `test_summary_db.py`, `test_summary_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 2–3. `conftest.py` builds its own minimal `articles` table (`ARTICLES_SCHEMA`) and `seed_article()` helper; it monkeypatches `news_nlp.db.DB_PATH`. No real database is touched.
- Produces: the green regression suite every later task re-runs.

- [ ] **Step 1: Branch**

```bash
git switch -c test/port-news-nlp-suite
```

- [ ] **Step 2: Copy the suite verbatim**

```bash
mkdir -p tests/news_nlp
cp "/d/projects/web_scraping/portfolio-data-mining/tests/news_nlp/"*.py tests/news_nlp/
```

Confirm 13 files landed (`conftest.py` + 12 `test_*.py`):
```bash
ls tests/news_nlp/*.py | wc -l   # -> 13
```

If the source has `tests/__init__.py` or `tests/news_nlp/__init__.py`, do NOT copy them — `pytest.ini` sets `pythonpath = src .` and rootdir insertion handles `from conftest import seed_article`. (If a collection error about `conftest` import appears in Step 3, add an empty `tests/news_nlp/__init__.py` and an empty `tests/__init__.py` and re-run.)

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass (the source repo keeps this suite green; every model load is monkeypatched, so no network / GPU needed).

Known-hazard checklist if anything fails:
- `ModuleNotFoundError: apps` → ensure `apps/__init__.py` exists (Task 3 Step 2) and you're invoking from the repo root.
- `ModuleNotFoundError: conftest` → add empty `tests/__init__.py` + `tests/news_nlp/__init__.py`.
- `sqlite3.OperationalError: no such table: articles` in a query test → that test relies on `conftest.ARTICLES_SCHEMA`; confirm `conftest.py` copied intact.
- A test importing `torch` failing on load → confirm `uv sync` installed torch (Task 1 Step 12).

- [ ] **Step 4: Lint the tests**

Run: `uv run ruff check tests`
Expected: pass (`PLR2004` in tests is ignored via per-file-ignore).

- [ ] **Step 5: Commit**

```bash
git add tests/news_nlp
git commit -m "test: port news_nlp test suite (13 files, fully hermetic)"
```

- [ ] **Step 6: Push, PR, merge after `uv run pytest -q` is green**

---

## Task 5: Repoint the database default to `nlp.db`

**Branch:** `feat/repoint-database-to-nlp-db`

**Files:**
- Modify: `src/news_nlp/db.py` (the `DB_PATH` fallback + the two comment blocks that name `urls.db`)
- Create: `tests/news_nlp/test_db_defaults.py`

**Interfaces:**
- Consumes: `news_nlp.db` from Task 2.
- Produces: unset-`DATABASE_URL` → `DB_PATH == <repo>/data/nlp.db`. No signature changes.

- [ ] **Step 1: Branch**

```bash
git switch -c feat/repoint-database-to-nlp-db
```

- [ ] **Step 2: Write the failing test**

Create `tests/news_nlp/test_db_defaults.py`:

```python
"""The unset-DATABASE_URL fallback must point at data/nlp.db (this repo's
dedicated database), not the legacy shared data/urls.db."""

import importlib
from pathlib import Path

import pytest


def _reload_db_with_env(monkeypatch: pytest.MonkeyPatch, value: str | None):
    if value is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", value)
    import news_nlp.db as db_module

    return importlib.reload(db_module)


def test_default_db_path_is_nlp_db(monkeypatch: pytest.MonkeyPatch) -> None:
    db_module = _reload_db_with_env(monkeypatch, None)
    assert db_module.DB_PATH.name == "nlp.db"
    assert db_module.DB_PATH.parent.name == "data"
    assert "urls.db" not in str(db_module.DB_PATH)


def test_relative_database_url_resolved_against_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_module = _reload_db_with_env(monkeypatch, "data/custom.db")
    assert db_module.DB_PATH.is_absolute()
    assert db_module.DB_PATH.parts[-2:] == ("data", "custom.db")


def test_absolute_database_url_used_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    abs_path = str(Path("D:/thesis/data/nlp.db")) if Path("D:/").exists() else "/tmp/x/nlp.db"
    db_module = _reload_db_with_env(monkeypatch, abs_path)
    assert str(db_module.DB_PATH) == str(Path(abs_path))
    # restore module state for other tests
    _reload_db_with_env(monkeypatch, None)
```

- [ ] **Step 3: Run it — expect failure**

Run: `uv run pytest tests/news_nlp/test_db_defaults.py -q`
Expected: `test_default_db_path_is_nlp_db` FAILS (`DB_PATH.name` is `urls.db`).

- [ ] **Step 4: Make the change in `src/news_nlp/db.py`**

Change the fallback line inside the `DB_PATH = (...)` expression:

```python
# before
    else _PROJECT_ROOT / "data" / "urls.db"
# after
    else _PROJECT_ROOT / "data" / "nlp.db"
```

Update the module docstring's first paragraph — replace:

> Reads from the existing `articles` table and writes to three results tables --
> `article_sentiment`, `article_entities`, and `article_category` -- each keyed
> by article_id.

with:

> Reads from an `articles` table and writes results tables --
> `article_sentiment`, `article_entities`, `article_category`, `article_summary`,
> `sector_summary` -- each keyed by `article_id`. The database is selected by
> `$DATABASE_URL` (unset -> `<repo>/data/nlp.db`).

In the `$DATABASE_URL` comment block just above `_PROJECT_ROOT = ...`, change the parenthetical `(the .env.example default, "data/urls.db")` to `(e.g. "data/nlp.db")`.

- [ ] **Step 5: Sweep for other `urls.db` mentions in ported code**

Run: `uv run python -c "import subprocess,sys; sys.exit(subprocess.call(['git','grep','-n','urls.db','--','src','apps','cli']))"`
Or simply: `git grep -n "urls\.db" -- src apps cli`
Expected after the edit: no matches. Fix any stragglers (comments/docstrings only — there should be none outside `db.py`).

- [ ] **Step 6: Run the new test + full suite**

Run: `uv run pytest tests/news_nlp/test_db_defaults.py -q`
Expected: PASS (3 tests).
Run: `uv run pytest -q`
Expected: whole suite still green.

- [ ] **Step 7: Commit**

```bash
git add src/news_nlp/db.py tests/news_nlp/test_db_defaults.py
git commit -m "feat: default DATABASE_URL to data/nlp.db"
```

- [ ] **Step 8: Push, PR, merge when green**

---

## Task 6: Documentation

**Branch:** `docs/claude-md-and-module-docs`

**Files:**
- Create: `CLAUDE.md`, `docs/category-taxonomy.md` (verbatim copy), `docs/modules/news-nlp.md` (copy + repoint)

- [ ] **Step 1: Branch**

```bash
git switch -c docs/claude-md-and-module-docs
```

- [ ] **Step 2: Copy the taxonomy doc verbatim**

```bash
mkdir -p docs/modules
cp "/d/projects/web_scraping/portfolio-data-mining/docs/category-taxonomy.md" docs/
```

- [ ] **Step 3: Copy the module doc and repoint DB references**

```bash
cp "/d/projects/web_scraping/portfolio-data-mining/docs/modules/news-nlp.md" docs/modules/
```
Then in `docs/modules/news-nlp.md`: replace every `urls.db` with `nlp.db`, and drop any sentence describing the DB as *shared* with `news_collector`/`extractor` (in this repo it is not). Keep everything else (endpoint list, stage descriptions, model names) as-is.

- [ ] **Step 4: Write `CLAUDE.md`**

```markdown
# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`portfolio-nlp` — the NLP stage extracted from `portfolio-data-mining`
(https://github.com/gamug/portfolio-data-mining). It runs four transformer
stages over financial-news article text and stores the results in SQLite:

| Stage | Model | Table |
|---|---|---|
| Sentiment | `ProsusAI/finbert` | `article_sentiment` |
| NER | `gamug/sec-bert-finer-ord-ner` | `article_entities` |
| Category (zero-shot, 10-label taxonomy) | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | `article_category` |
| Summarization (opt-in) | `sshleifer/distilbart-cnn-12-6` | `article_summary`, `sector_summary` |

Category taxonomy: `docs/category-taxonomy.md`. Per-module detail:
`docs/modules/news-nlp.md`.

## Layout

- `src/news_nlp/` — the package. Relative imports internally
  (`from .categories import ...`).
  - `db.py` — SQLite access layer: `connect()`, `init_schema()`, read/write
    helpers, and the read-only query helpers behind the FastAPI query
    endpoints. `DB_PATH` is computed at import from `$DATABASE_URL`
    (unset -> `<repo>/data/nlp.db`; relative values resolve against the repo
    root, not CWD).
  - `pipeline.py` — the batch pipeline (`run_pipeline`) and per-stage
    functions. `DEVICE` is CUDA if available else CPU.
  - `categories.py` / `chunking.py` / `corrections.py` — taxonomy constants,
    text chunking + char-span merging, and the correction (manual-edit)
    helpers.
  - `setup.py` — `python -m news_nlp.setup` pre-downloads the four HF models.
  - `train_ner.py` — standalone SEC-BERT NER fine-tuning script (not imported
    by the pipeline).
- `apps/news_nlp_api.py` — FastAPI service (title `news-nlp`, port 8003).
  Prepends `src/` to `sys.path` so it runs standalone.
- `cli/news_nlp_cli.py` — argparse batch driver (`--limit`, `--summarize`),
  same `sys.path` bootstrap.
- `tests/news_nlp/` — hermetic: `conftest.py` builds a minimal `articles`
  table and every model load is monkeypatched. No network or GPU.
- `tests/migration/` — covers `scripts/migrate_from_urls_db.py`.
- `scripts/migrate_from_urls_db.py` — one-shot copy of the NLP tables (+ a
  `body_text`-free `articles` subset) out of the legacy shared `urls.db`.

## Commands

```bash
uv sync
uv run python -m news_nlp.setup           # download models
uv run apps/news_nlp_api.py               # -> :8003/docs
uv run cli/news_nlp_cli.py --limit 50
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pre-commit run --all-files
```

## Config

- Lint/format: `.code_quality/ruff.toml` (root `ruff.toml` only `extend`s it).
- Types: `.code_quality/mypy.ini` (`mypy_path = src`).
- `pytest.ini`: `pythonpath = src .` (the `.` lets `tests/news_nlp/conftest.py`
  import `apps.news_nlp_api`).
- `.env` (git-ignored) holds `DATABASE_URL`; `.env.example` is the template.

## Database

Standalone SQLite. Working file: `D:\thesis\data\nlp.db`. Schema created by
`news_nlp.db.init_schema()`. Not shared with any other service.
```

- [ ] **Step 5: Sanity-check links/paths**

Run: `uv run ruff check .` (docs don't affect it, but confirms nothing broke) and eyeball that `docs/modules/news-nlp.md` has no remaining `urls.db`.
Run: `git grep -n "urls\.db" -- docs` → expect no matches.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md docs
git commit -m "docs: add CLAUDE.md and port category taxonomy + module doc"
```

- [ ] **Step 7: Push, PR, merge**

---

## Task 7: CI workflow

**Branch:** `ci/lint-and-test-workflow`

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Branch**

```bash
git switch -c ci/lint-and-test-workflow
```

- [ ] **Step 2: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.11"

      - name: Sync (CPU torch — ignore the cu124 source pin)
        run: uv sync --no-sources --group dev

      - name: Ruff lint
        run: uv run --no-sync ruff check .

      - name: Ruff format check
        run: uv run --no-sync ruff format --check .

      - name: mypy
        run: uv run --no-sync mypy

      - name: pytest
        run: uv run --no-sync pytest -q
```

- [ ] **Step 3: Validate locally as far as possible**

Run: `uv run python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text()); print('yaml ok')"`
Expected: `yaml ok`. (`pyyaml` may not be installed — if it errors with `ModuleNotFoundError`, skip; the push will validate.)

Reproduce the CI resolution locally in a scratch dir to catch a `--no-sources` break early:
```bash
uv export --no-sources --group dev --format requirements-txt > /tmp/ci-reqs.txt && head -5 /tmp/ci-reqs.txt
```
Expected: succeeds and shows `torch` without the `download.pytorch.org` URL.

- [ ] **Step 4: Commit, push, open PR**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: lint, format, mypy and pytest on push/PR"
git push -u origin ci/lint-and-test-workflow
gh pr create --fill --base master
```

- [ ] **Step 5: Watch the first CI run**

Run: `gh pr checks --watch`
Expected: all four steps green. If `mypy` or `ruff` fail only in CI (not locally), reconcile versions — CI uses the locked `ruff==0.16.3` and `mypy>=1.13` same as local, so a diff means an OS-specific path issue; fix in this PR.

- [ ] **Step 6: Merge**

`gh pr merge --squash --delete-branch`, update `master`. From here on, every PR waits for green CI before merge.

---

## Task 8: Migration script

**Branch:** `feat/migration-script`

**Files:**
- Create: `scripts/migrate_from_urls_db.py`, `tests/migration/__init__.py` (empty), `tests/migration/test_migrate_from_urls_db.py`

**Interfaces:**
- Consumes: `news_nlp.db.connect` / `news_nlp.db.init_schema` (for the result-table + index DDL), stdlib `sqlite3`, `argparse`.
- Produces: CLI `uv run python scripts/migrate_from_urls_db.py [--source PATH] [--dest PATH] [--dry-run]`.
  - `migrate(source: Path, dest: Path, *, dry_run: bool = False) -> dict[str, int]` — returns `{table_name: rows_copied}` (or rows that *would* copy under `--dry-run`).

**Migration rules (locked):**
- Result tables copied: `article_sentiment`, `article_category`, `article_summary`, `article_entities`, `sector_summary`.
- Every result table except `sector_summary` is filtered to `article_id IN (SELECT id FROM src.articles)` — orphaned result rows are skipped, keeping `PRAGMA foreign_key_check` clean.
- `articles` copied as a **lean subset**: all live columns **except `body_text`**, only rows referenced by a copied result row.
  - Live `articles` columns (from `PRAGMA table_info`): `id, ticker, company, gics_sector, gics_sub_industry, title, author, pub_date, body_text, word_count, language, source_domain, extraction_method, fetch_status, http_status_code, fetched_at`. The script derives the column list dynamically (`PRAGMA table_info(articles)` minus `body_text`) rather than hardcoding, so an extra column in the live DB still comes across.
- `discovered_urls`, `discovery_progress` are NOT copied.
- Idempotent: `INSERT OR IGNORE` on every copy; re-running is a no-op.
- Performance: on `dest`, `PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF` for the load; drop `idx_article_entities_article_id` before the 17.6M-row `article_entities` copy and recreate it after.
- `dest` opened read-write; `src` **attached read-only** via `file:...?mode=ro`.
- Ends with `PRAGMA foreign_key_check` on `dest`; if it returns any row, print them and exit 2.

- [ ] **Step 1: Branch + scaffold**

```bash
git switch -c feat/migration-script
mkdir -p scripts tests/migration
printf '' > tests/migration/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/migration/test_migrate_from_urls_db.py`:

```python
"""Fixture-DB coverage for scripts/migrate_from_urls_db.py."""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "migrate_from_urls_db",
    Path(__file__).resolve().parents[2] / "scripts" / "migrate_from_urls_db.py",
)
migrate_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migrate_mod)

FULL_ARTICLES_DDL = """
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    ticker TEXT, company TEXT, gics_sector TEXT, gics_sub_industry TEXT,
    title TEXT, author TEXT, pub_date TEXT, body_text TEXT, word_count INTEGER,
    language TEXT, source_domain TEXT, extraction_method TEXT,
    fetch_status TEXT, http_status_code INTEGER, fetched_at TEXT,
    FOREIGN KEY (id) REFERENCES discovered_urls (id)
)
"""

RESULT_DDL = {
    "article_sentiment": """CREATE TABLE article_sentiment (
        article_id INTEGER PRIMARY KEY REFERENCES articles(id), label TEXT NOT NULL,
        score REAL NOT NULL, positive REAL NOT NULL, negative REAL NOT NULL,
        neutral REAL NOT NULL, model_name TEXT NOT NULL, processed_at TEXT NOT NULL)""",
    "article_entities": """CREATE TABLE article_entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER NOT NULL REFERENCES articles(id),
        entity_type TEXT NOT NULL, text TEXT NOT NULL, start_char INTEGER NOT NULL,
        end_char INTEGER NOT NULL, score REAL, model_name TEXT NOT NULL,
        processed_at TEXT NOT NULL)""",
    "article_category": """CREATE TABLE article_category (
        article_id INTEGER PRIMARY KEY REFERENCES articles(id), label TEXT NOT NULL,
        score REAL NOT NULL, earnings_performance REAL NOT NULL,
        mergers_acquisitions REAL NOT NULL, leadership_governance REAL NOT NULL,
        legal_regulatory REAL NOT NULL, product_innovation REAL NOT NULL,
        capital_shareholder_returns REAL NOT NULL, labor_human_capital REAL NOT NULL,
        market_analyst_sentiment REAL NOT NULL, partnerships_business_dev REAL NOT NULL,
        model_name TEXT NOT NULL, processed_at TEXT NOT NULL)""",
    "article_summary": """CREATE TABLE article_summary (
        article_id INTEGER PRIMARY KEY REFERENCES articles(id),
        summary_text TEXT NOT NULL, num_chunks INTEGER NOT NULL,
        model_name TEXT NOT NULL, processed_at TEXT NOT NULL)""",
    "sector_summary": """CREATE TABLE sector_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT, gics_sector TEXT NOT NULL,
        gics_sub_industry TEXT NOT NULL, week_start TEXT NOT NULL, week_end TEXT NOT NULL,
        summary_text TEXT NOT NULL, num_articles INTEGER NOT NULL,
        num_companies INTEGER NOT NULL, model_name TEXT NOT NULL,
        processed_at TEXT NOT NULL, format_version INTEGER NOT NULL DEFAULT 0,
        facts_json TEXT NOT NULL DEFAULT '{}', intro_text TEXT NOT NULL DEFAULT '')""",
}


def _build_source(path: Path) -> None:
    c = sqlite3.connect(path)
    c.executescript("CREATE TABLE discovered_urls (id INTEGER PRIMARY KEY, url TEXT);")
    c.executescript(FULL_ARTICLES_DDL)
    for ddl in RESULT_DDL.values():
        c.executescript(ddl)
    c.executemany("INSERT INTO discovered_urls VALUES (?, ?)",
                  [(i, f"http://x/{i}") for i in (1, 2, 3)])
    # articles 1,2,3 exist; result rows will also reference a missing id 99
    for i in (1, 2, 3):
        c.execute(
            "INSERT INTO articles (id, ticker, title, body_text, http_status_code) "
            "VALUES (?, ?, ?, ?, 200)", (i, "MMM", f"t{i}", "X" * 5000))
    for i in (1, 2):
        c.execute("INSERT INTO article_sentiment VALUES (?,?,?,?,?,?,?,?)",
                  (i, "positive", 0.9, 0.9, 0.05, 0.05, "finbert", "2024-01-01"))
        c.execute("INSERT INTO article_category VALUES "
                  "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (i, "earnings_performance", 0.8, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, "deberta", "2024-01-01"))
        c.execute("INSERT INTO article_summary VALUES (?,?,?,?,?)",
                  (i, "sum", 1, "distilbart", "2024-01-01"))
    c.executemany(
        "INSERT INTO article_entities "
        "(article_id, entity_type, text, start_char, end_char, score, model_name, processed_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [(1, "ORG", "3M", 0, 2, 0.99, "sec-bert", "2024-01-01"),
         (2, "ORG", "3M", 0, 2, 0.99, "sec-bert", "2024-01-01"),
         (99, "ORG", "Ghost", 0, 5, 0.99, "sec-bert", "2024-01-01")],  # orphan
    )
    c.execute("INSERT INTO article_sentiment VALUES (?,?,?,?,?,?,?,?)",
              (99, "neutral", 0.5, 0.3, 0.2, 0.5, "finbert", "2024-01-01"))  # orphan
    c.execute("INSERT INTO sector_summary "
              "(gics_sector, gics_sub_industry, week_start, week_end, summary_text, "
              "num_articles, num_companies, model_name, processed_at) "
              "VALUES ('Industrials','X','2024-01-01','2024-01-07','s',2,1,'m','2024-01-01')")
    c.commit()
    c.close()


def test_migrate_copies_results_and_lean_articles(tmp_path: Path) -> None:
    src, dest = tmp_path / "urls.db", tmp_path / "nlp.db"
    _build_source(src)

    counts = migrate_mod.migrate(src, dest)

    d = sqlite3.connect(dest)
    cols = {r[1] for r in d.execute("PRAGMA table_info(articles)")}
    assert "body_text" not in cols
    assert cols == {
        "id", "ticker", "company", "gics_sector", "gics_sub_industry", "title",
        "author", "pub_date", "word_count", "language", "source_domain",
        "extraction_method", "fetch_status", "http_status_code", "fetched_at",
    }
    # only articles referenced by a *kept* result row: ids 1 and 2
    assert {r[0] for r in d.execute("SELECT id FROM articles")} == {1, 2}
    # orphan (id 99) result rows skipped
    assert d.execute("SELECT COUNT(*) FROM article_sentiment").fetchone()[0] == 2
    assert d.execute("SELECT COUNT(*) FROM article_entities").fetchone()[0] == 2
    assert d.execute("SELECT COUNT(*) FROM sector_summary").fetchone()[0] == 1
    assert d.execute("PRAGMA foreign_key_check").fetchall() == []
    # the entities index is back
    idx = {r[1] for r in d.execute("PRAGMA index_list(article_entities)")}
    assert "idx_article_entities_article_id" in idx
    d.close()

    assert counts["article_sentiment"] == 2
    assert counts["article_entities"] == 2
    assert counts["articles"] == 2


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    src, dest = tmp_path / "urls.db", tmp_path / "nlp.db"
    _build_source(src)
    migrate_mod.migrate(src, dest)
    second = migrate_mod.migrate(src, dest)
    assert all(v == 0 for v in second.values())
    d = sqlite3.connect(dest)
    assert d.execute("SELECT COUNT(*) FROM article_sentiment").fetchone()[0] == 2
    d.close()


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    src, dest = tmp_path / "urls.db", tmp_path / "nlp.db"
    _build_source(src)
    counts = migrate_mod.migrate(src, dest, dry_run=True)
    assert counts["article_sentiment"] == 2
    assert not dest.exists() or sqlite3.connect(dest).execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
    ).fetchone()[0] == 0


def test_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        migrate_mod.migrate(tmp_path / "nope.db", tmp_path / "nlp.db")
```

- [ ] **Step 3: Run — expect failure**

Run: `uv run pytest tests/migration -q`
Expected: FAIL — `scripts/migrate_from_urls_db.py` does not exist yet (`exec_module` raises `FileNotFoundError` at import).

- [ ] **Step 4: Write `scripts/migrate_from_urls_db.py`**

```python
#!/usr/bin/env python
"""One-shot copy of the NLP data out of the legacy shared `urls.db` into this
repo's dedicated `nlp.db`.

Copies the five result tables (`article_sentiment`, `article_category`,
`article_summary`, `article_entities`, `sector_summary`) and a lean `articles`
subset -- every column except `body_text`, only for rows a copied result row
references. `discovered_urls` / `discovery_progress` are left behind.

Idempotent (INSERT OR IGNORE). The source is attached read-only and never
written. Usage:

    uv run python scripts/migrate_from_urls_db.py \
        --source D:/thesis/data/urls.db --dest D:/thesis/data/nlp.db
    uv run python scripts/migrate_from_urls_db.py --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from news_nlp import db as nlp_db  # noqa: E402

DEFAULT_SOURCE = Path("D:/thesis/data/urls.db")
DEFAULT_DEST = Path("D:/thesis/data/nlp.db")

# copied in this order; every one except sector_summary is filtered to
# article_id present in src.articles
RESULT_TABLES = (
    "article_sentiment",
    "article_category",
    "article_summary",
    "article_entities",
    "sector_summary",
)
ENTITIES_INDEX = "idx_article_entities_article_id"


def _lean_article_columns(src: sqlite3.Connection) -> list[str]:
    cols = [row[1] for row in src.execute("PRAGMA table_info(articles)")]
    return [c for c in cols if c != "body_text"]


def _referenced_article_ids_sql() -> str:
    parts = [
        f"SELECT article_id FROM src.{t}"
        for t in RESULT_TABLES
        if t != "sector_summary"
    ]
    return " UNION ".join(parts)


def migrate(source: Path, dest: Path, *, dry_run: bool = False) -> dict[str, int]:
    source, dest = Path(source), Path(dest)
    if not source.exists():
        raise FileNotFoundError(f"source database not found: {source}")

    src_ro = f"file:{source.as_posix()}?mode=ro"

    if dry_run:
        probe = sqlite3.connect(":memory:")
        probe.execute("ATTACH DATABASE ? AS src", (src_ro,))
        counts: dict[str, int] = {}
        for t in RESULT_TABLES:
            if t == "sector_summary":
                n = probe.execute("SELECT COUNT(*) FROM src.sector_summary").fetchone()[0]
            else:
                n = probe.execute(
                    f"SELECT COUNT(*) FROM src.{t} "
                    f"WHERE article_id IN (SELECT id FROM src.articles)"
                ).fetchone()[0]
            counts[t] = n
        counts["articles"] = probe.execute(
            f"SELECT COUNT(*) FROM src.articles WHERE id IN ({_referenced_article_ids_sql()})"
        ).fetchone()[0]
        probe.close()
        _print_report(counts, dry_run=True)
        return counts

    dest.parent.mkdir(parents=True, exist_ok=True)
    conn = nlp_db.connect(dest)          # sets busy_timeout + foreign_keys pragma
    nlp_db.init_schema(conn)             # creates result tables + entities index
    _ensure_lean_articles_table(conn, source, src_ro)

    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("ATTACH DATABASE ? AS src", (src_ro,))

    counts = {}
    art_cols = _lean_article_columns(sqlite3.connect(src_ro, uri=True))
    col_list = ", ".join(art_cols)
    before = _table_count(conn, "articles")
    conn.execute(
        f"INSERT OR IGNORE INTO articles ({col_list}) "
        f"SELECT {col_list} FROM src.articles "
        f"WHERE id IN ({_referenced_article_ids_sql()})"
    )
    counts["articles"] = _table_count(conn, "articles") - before

    conn.execute(f"DROP INDEX IF EXISTS {ENTITIES_INDEX}")
    for t in RESULT_TABLES:
        before = _table_count(conn, t)
        if t == "sector_summary":
            conn.execute(f"INSERT OR IGNORE INTO {t} SELECT * FROM src.{t}")
        else:
            conn.execute(
                f"INSERT OR IGNORE INTO {t} SELECT * FROM src.{t} "
                f"WHERE article_id IN (SELECT id FROM src.articles)"
            )
        counts[t] = _table_count(conn, t) - before
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {ENTITIES_INDEX} "
        f"ON article_entities(article_id)"
    )

    conn.commit()
    conn.execute("DETACH DATABASE src")

    fk_problems = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk_problems:
        for row in fk_problems:
            print(f"  FK violation: {row}", file=sys.stderr)
        conn.close()
        print("foreign_key_check failed -- aborting", file=sys.stderr)
        sys.exit(2)

    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("ANALYZE")
    conn.close()
    _print_report(counts, dry_run=False)
    return counts


def _ensure_lean_articles_table(
    conn: sqlite3.Connection, source: Path, src_ro: str
) -> None:
    if conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
    ).fetchone():
        return
    src = sqlite3.connect(src_ro, uri=True)
    cols = [
        (row[1], row[2]) for row in src.execute("PRAGMA table_info(articles)")
        if row[1] != "body_text"
    ]
    src.close()
    defs = ", ".join(
        f"{name} {ctype or 'TEXT'}{' PRIMARY KEY' if name == 'id' else ''}"
        for name, ctype in cols
    )
    conn.execute(f"CREATE TABLE articles ({defs})")


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _print_report(counts: dict[str, int], *, dry_run: bool) -> None:
    verb = "would copy" if dry_run else "copied"
    width = max(len(k) for k in counts)
    print(f"\n{'table':<{width}}  rows {verb}")
    print("-" * (width + 6 + len(verb)))
    for name, n in counts.items():
        print(f"{name:<{width}}  {n:>12,}")
    print(f"{'TOTAL':<{width}}  {sum(counts.values()):>12,}\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    migrate(args.source, args.dest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the migration tests**

Run: `uv run pytest tests/migration -q`
Expected: 4 passed.

Fix loop if a test fails:
- `articles` column set mismatch → check `_lean_article_columns` filtering.
- orphan rows present in dest → the `WHERE article_id IN (SELECT id FROM src.articles)` filter is missing on a table.
- `foreign_key_check` non-empty → `articles` subset query (`_referenced_article_ids_sql`) is not covering that table's ids.
- idempotency test sees non-zero on 2nd run → a copy statement is missing `OR IGNORE`.

- [ ] **Step 6: Lint + type-check the script**

Run: `uv run ruff check scripts tests/migration`
Expected: pass. `E402` on the post-`sys.path.insert` import is covered only for `apps/`/`cli/` — add a targeted `# noqa: E402` (already in the code above) rather than widening the per-file-ignore.
Run: `uv run mypy scripts`
Expected: `Success`. If mypy flags `nlp_db` attributes, it's the same `ignore_missing_imports` path as the package — should be fine.

- [ ] **Step 7: Full suite + commit**

Run: `uv run pytest -q`
Expected: whole suite (news_nlp + migration) green.

```bash
git add scripts/migrate_from_urls_db.py tests/migration
git commit -m "feat: urls.db -> nlp.db migration script with fixture tests"
```

- [ ] **Step 8: Push, PR, wait for green CI, merge**

---

## Task 9: Run the migration (operational — no PR)

**Precondition:** Tasks 1–8 merged to `master`; `git switch master && git pull`.

- [ ] **Step 1: Confirm inputs**

```bash
ls -la /d/thesis/data/urls.db          # exists, ~4.7 GB
ls -la /d/thesis/data/nlp.db 2>&1      # expected: does not exist yet
```

- [ ] **Step 2: Dry run**

Run: `uv run python scripts/migrate_from_urls_db.py --dry-run`
Expected report (order-of-magnitude, from the live DB at plan time):

```
table              rows would copy
article_sentiment          ~459,112
article_category           ~459,112
article_summary            ~458,641
article_entities        ~17,666,722
sector_summary               ~3,628
articles                   ~459,xxx
```

Sanity: `articles` count should be `>= max(sentiment, category)` and `<= 481,332`.

- [ ] **Step 3: Real run**

Run: `uv run python scripts/migrate_from_urls_db.py`
Expected: same table, `rows copied` populated, ends without the `foreign_key_check failed` message, exit 0. Wall time: minutes (dominated by the 17.6M-row entities copy). Capture the printed report.

- [ ] **Step 4: Verify the result**

```bash
uv run python -c "
import sqlite3; d = sqlite3.connect('D:/thesis/data/nlp.db')
for t in ['articles','article_sentiment','article_category','article_summary','article_entities','sector_summary']:
    print(f'{t:20}', d.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0])
print('body_text present:', any(r[1]=='body_text' for r in d.execute('PRAGMA table_info(articles)')))
print('fk_check rows:', len(d.execute('PRAGMA foreign_key_check').fetchall()))
print('entities index:', [r[1] for r in d.execute('PRAGMA index_list(article_entities)')])
"
```
Expected: counts match the Step 3 report; `body_text present: False`; `fk_check rows: 0`; `idx_article_entities_article_id` listed.

- [ ] **Step 5: Point local `.env` at it (if not already)**

Ensure `D:/projects/web_scraping/portfolio-nlp/.env` has `DATABASE_URL=D:/thesis/data/nlp.db`. (`.env` is git-ignored — no commit.)

- [ ] **Step 6: End-to-end sanity against the real DB**

Run: `uv run python -c "import news_nlp.db as db; c = db.connect(db.DB_PATH); print(db.DB_PATH); print('sentiment rows:', c.execute('SELECT COUNT(*) FROM article_sentiment').fetchone()[0])"`
Expected: prints `...\thesis\data\nlp.db` and a ~459K count.

- [ ] **Step 7: Record the outcome**

Add the captured row-count report to the Task 8 PR thread (comment) or a short `docs/migration-2026-08-31.md` note committed straight to `master` with `docs:` prefix. (Small enough to skip the branch/PR ceremony — a plain doc note.)

---

## Task 10: Strip NLP from `portfolio-data-mining` (PR left OPEN)

**Repo:** `D:/projects/web_scraping/portfolio-data-mining`
**Branch:** `chore/extract-news-nlp-to-portfolio-nlp`

**Files:**
- Delete: `src/news_nlp/`, `apps/news_nlp_api.py`, `cli/news_nlp_cli.py`, `tests/news_nlp/`, `docs/modules/news-nlp.md`
- Modify: `pyproject.toml`, `uv.lock` (regenerated), `apps/gateway.py`, `CLAUDE.md`, `README.md`, `.env.example`, `pytest.ini`

- [ ] **Step 1: Branch in the other repo**

```bash
cd /d/projects/web_scraping/portfolio-data-mining
git switch -c chore/extract-news-nlp-to-portfolio-nlp
```

- [ ] **Step 2: Delete the NLP files**

```bash
git rm -r src/news_nlp apps/news_nlp_api.py cli/news_nlp_cli.py tests/news_nlp docs/modules/news-nlp.md
```

- [ ] **Step 3: Prune `pyproject.toml`**

Remove these lines from `[project].dependencies` (the `# --- news_nlp ...` block):
`torch>=2.4,<2.7`, `transformers>=4.40`, `datasets>=2.19`, `accelerate>=0.30`, `evaluate>=0.4`, `seqeval>=1.2`, `huggingface_hub>=0.24` (and the explanatory torch comment).
Remove the whole `[tool.uv.sources]` block (`torch = { index = "pytorch-cu124" }`) and the `[[tool.uv.index]]` block for `pytorch-cu124`.
Leave `pytest-asyncio` (still used by other suites) and everything else.

- [ ] **Step 4: Regenerate the lock**

Run: `uv lock`
Then: `uv sync`
Expected: resolves without the pytorch index; `torch` and friends gone from `uv.lock`.

- [ ] **Step 5: Unmount news_nlp from the gateway**

In `apps/gateway.py`:
- Remove the tuple `("/nlp", "news_nlp_api", "News NLP (sentiment + NER)"),` from the mounts list (~line 51).
- In the docstring (~line 18) drop the `or torch not installed for news_nlp` clause.
- In the closing help string (~line 106) change `apps/{news_collector,news_crawler,news_nlp,pricing,sec_edgar}_api.py` to `apps/{news_collector,news_crawler,pricing,sec_edgar}_api.py`.

- [ ] **Step 6: Update docs**

- `CLAUDE.md`: in the pipeline diagram drop `-> news_nlp (sentiment + NER)`; delete the `news_nlp/` row from the `src/` package table and the `news_nlp` row from the `apps/` vs `cli/` table; in the "Shared SQLite database" section remove the clause naming `news_nlp reads articles and writes article_sentiment/article_entities/article_category`, and change "read/written by three modules" to "two modules" (`news_collector`, `extractor`); drop the `uv run pytest tests/news_nlp` command line; update the "What this is" paragraph's "four previously separate projects" note only if it now reads wrong (leave the historical sentence, it's accurate history).
- `README.md`: same pipeline-diagram / module-list edits.
- `.env.example`: remove NLP-specific comments if any single line is solely about news_nlp; keep `DATABASE_URL`.
- `pytest.ini`: the comment lists packages `(news_collector, extractor, news_nlp, common, pricing, sec_edgar)` — drop `news_nlp`.

- [ ] **Step 7: Verify the remaining suite**

Run: `uv run pytest -q`
Expected: passes with the same baseline as before extraction (per that repo's `CLAUDE.md`, one pre-existing Windows-only flake in `tests/news_collector` is acceptable; nothing new should break).
Run: `uv run ruff check . && uv run mypy`
Expected: clean.
Run: `uv run python apps/gateway.py &` then `curl -s localhost:8000/ | head` (or import-check `python -c "import apps.gateway"`) — expected: starts / imports with four sub-apps, no `news_nlp` reference.

- [ ] **Step 8: Commit + push + open PR (do NOT merge)**

```bash
git add -A
git commit -m "chore: remove news_nlp (extracted to gamug/portfolio-nlp)"
git push -u origin chore/extract-news-nlp-to-portfolio-nlp
gh pr create --base master --title "chore: remove news_nlp (extracted to portfolio-nlp)" \
  --body "The NLP stage now lives in https://github.com/gamug/portfolio-nlp. This removes src/news_nlp/, apps/news_nlp_api.py, cli/news_nlp_cli.py, tests/news_nlp/, the NLP dependency block + pytorch-cu124 index, and the gateway mount. Leaving open for review."
```

Report the PR URL back. Leave it for the user to merge.

---

## Self-Review

**1. Spec coverage**

| Spec section | Task(s) |
|---|---|
| §1 Repo scaffold & tooling | Task 1 (every listed file: pyproject, .python-version, ruff pointer + `.code_quality/*`, pre-commit, pytest.ini, .gitignore, .env.example, .devcontainer, README; `CLAUDE.md` moved to Task 6) |
| §2 Ported code (`src/news_nlp`, `apps`, `cli`, `tests`, docs) | Tasks 2, 3, 4, 6 |
| §3 DB repointing to `nlp.db` | Task 5 (+ `.env.example` already `nlp.db` in Task 1) |
| §4 Migration script + execution | Tasks 8 (script + fixture test) and 9 (real run + verify) |
| §5 Strip NLP from source repo | Task 10 |
| §6 Branch/commit/PR sequence | "PR / merge workflow" section + one branch per task |
| §7 CI | Task 7 (`uv sync --no-sources`, ruff + format + mypy + pytest) |
| Testing strategy | Tasks 2/3 (ruff+mypy), 4 (`pytest`), 8 (fixture + `--dry-run`), 9 (`foreign_key_check`, end-to-end count) |
| Risk: CI cu124 pin | Task 7 Step 2/3 (`uv sync --no-sources`, `uv export` pre-check) |
| Risk: model tests need GPU | Resolved during planning — all model loads monkeypatched; no skips needed (noted in Task 4 Step 3) |
| Risk: live `articles` has extra columns | Task 8 `_lean_article_columns` reads `PRAGMA table_info` dynamically |
| Risk: orphan NLP rows | Task 8 migration rule — result copies filtered to existing `article_id`; `foreign_key_check` gate |
| Risk: `urls.db` locked / WAL | Task 8 — `?mode=ro` attach, never writes source |
| Risk: `.env` Turso token committed | Global Constraints + Task 1 (never `git add .env`; `.gitignore` covers it) |
| Risk: gateway import breaks | Task 10 Steps 5, 7 |

No gaps.

**2. Placeholder scan** — no "TBD"/"handle edge cases"/"similar to Task N"/"write tests for the above". Every code step has literal content; every run step has an exact command + expected output.

**3. Type consistency** — `migrate(source, dest, *, dry_run=False) -> dict[str, int]` used identically in the script and all four tests. `news_nlp.db.connect` / `init_schema` / `DB_PATH` names match across Tasks 2, 5, 8, 9 and the spec's Interfaces blocks. `RESULT_TABLES` / `ENTITIES_INDEX` defined once in the script and referenced by name. FastAPI attrs (`app`, `pipeline_status.reset()`, `app_module.pipeline`) match what `conftest.py` and `test_pipeline_endpoints.py` actually use (verified against source).
