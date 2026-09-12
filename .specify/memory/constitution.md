# Project Constitution

Governing principles for `portfolio-nlp` under a spec-driven ("spec coding")
workflow: specs and plans are written before implementation, and this
document is the fixed reference they must not contradict. A spec or plan
that conflicts with a rule below must change the rule here first (see
Governance) rather than override it silently.

## Technological stock

`portfolio-nlp` is a headless Python service — there is no R or Streamlit
component; every rule below assumes the stack actually pinned in
`pyproject.toml`.

1. **Runtime**: Python `>=3.12,<3.13`, dependency-managed with `uv` (lockfile
   `uv.lock`). Do not add a second package manager (pip/poetry/conda) — all
   installs go through `uv sync` / `uv add`.
2. **ML/NLP core**: PyTorch (`torch>=2.4,<2.7`, pinned to the `cu124` wheel
   index so `uv sync` reproduces the same GPU build) + Hugging Face
   `transformers`/`datasets`/`accelerate`/`evaluate`/`seqeval`/`huggingface_hub`.
   Chosen for GPU/CPU portability (`pipeline.py`'s `DEVICE` falls back to CPU
   automatically) and because all four models it runs
   (`ProsusAI/finbert`, `gamug/sec-bert-finer-ord-ner`,
   `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`,
   `sshleifer/distilbart-cnn-12-6`) are published as `transformers`
   checkpoints — swapping frameworks would mean re-hosting every model.
3. **Web/service layer**: FastAPI (`==0.141.1`) + `uvicorn[standard]` for
   `apps/news_nlp_api.py`; `httpx[http2]` for outbound calls; `pydantic>=2.0`
   for request/response models. Pin exact versions for FastAPI/uvicorn/ruff
   (reproducible CI); range-pin libraries that are additive/stable
   (`pydantic`, `numpy`, `tqdm`).
4. **Storage**: SQLite, accessed two-tier (SOURCE ATTACHed read-only, RESULTS
   read/write) through `portfolio_common.db.Database` /
   `Allowlist` / `in_clause`, git-tag-pinned (`portfolio-common @ tag
   v1.2.0`) rather than a floating version, so a DB-engine-contract change is
   an explicit, reviewed re-pin (`docs/engine-agnostic-rollout.md`). Chosen
   for zero-ops single-file deployment at this corpus scale; moving off
   SQLite is a `portfolio-common`-level decision, not a change made here.
5. **Optional/heavy group**: `strands-agents[openai]` + `mlflow`, isolated in
   the `eval` dependency group (`[dependency-groups].eval`) because they pull
   a heavy tree (boto3, opentelemetry, mcp) that the pipeline and API must
   never require just to import `news_nlp`.
6. **Licensing**: all pinned dependencies and the four HF model checkpoints
   must carry a license compatible with this repo's own (check the model
   card before adding a fifth model or a new library) — flag anything
   copyleft or usage-restricted in the PR that introduces it.
7. **Adopting a new library/framework is a constitution-level change**: add
   it to `pyproject.toml` with a rationale in the PR, and if it changes a
   rule above, amend this section (see Governance).

## Project structure

1. **`src/` is flat, plus one local package.** Pipeline stages and utilities
   are single files directly under `src/` (`pipeline.py`, `chunking.py`,
   `setup.py`, `train_ner.py`); `news_nlp/` is the one local package (the
   vendored DB layer: connection, schema, queries, corrections, taxonomy,
   `sector_summary`, plus `news_nlp/eval/` for LLM-as-judge evaluation). A
   new package needs the same justification `news_nlp/` has — a genuinely
   self-contained subsystem with its own submodules — not just "this file is
   getting long."
2. **Entrypoints live by kind, each with the same bootstrap.** `apps/*.py`
   (FastAPI services), `cli/*.py` (argparse batch drivers), `scripts/*.py`
   (one-shot operational scripts) each prepend `src/` to `sys.path` before
   importing the flat modules — copy that pattern, don't invent a second one
   (e.g. an installed console-script entry point).
3. **Tests mirror `src/` under `tests/news_nlp/`**, driven by
   `tests/news_nlp/conftest.py`'s hermetic fixtures (a minimal `articles`
   table, two-tier SOURCE/RESULTS fixtures, every model load monkeypatched).
   `pytest.ini`'s `pythonpath = src .` is what makes both `import news_nlp`
   and `import apps.news_nlp_api` work in tests — don't add `sys.path` hacks
   inside test files to route around it.
4. **Docs live under `docs/`**, one topic per file (`db-topology.md`,
   `category-taxonomy.md`, `modules/news-nlp.md`, migration notes named by
   date/topic). A new cross-cutting doc goes in `docs/`; a spec-kit artifact
   (this constitution, future specs/plans) goes under `.specify/`.
5. **Config lives where its tool expects it, not duplicated.** Ruff:
   `.code_quality/ruff.toml` (root `ruff.toml` only `extend`s it so plain
   `ruff check .` from the repo root resolves the same config `pre-commit`
   uses). Mypy: `.code_quality/mypy.ini`. Pytest: root `pytest.ini`. Don't
   fork a second config file for a tool that already has one.
6. **Environment**: `.env` (git-ignored) holds `DATABASE_URL` /
   `SOURCE_DATABASE_URL` and, for `news_nlp.eval` only, `LLM_API_KEY` /
   `LLM_MODEL` / `LLM_URL` / `MLFLOW_TRACKING_URI`; `.env.example` is the
   committed template — keep it in sync with every env var a new feature
   reads. `pipeline.py` calling `load_dotenv()` at import is the single
   place `.env` gets loaded; don't add a second `load_dotenv()` call
   elsewhere.
7. **Naming**: modules and functions describe the pipeline stage or DB
   concern they implement — match the stage/table pairing (sentiment ->
   `article_sentiment`, NER -> `article_entities`, category ->
   `article_category`, summarization -> `article_summary`/
   `sector_summary`) rather than inventing a new term for the same concept.

## AI behavior

*Pipeline models (the system's own AI components):*

1. **Model selection is pinned, not dynamic.** The four stages
   (sentiment/`ProsusAI/finbert`, NER/`gamug/sec-bert-finer-ord-ner`,
   category/`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`,
   summarization/`sshleifer/distilbart-cnn-12-6`, opt-in) each use exactly
   the checkpoint listed above — no dynamic model selection. Swapping a
   model is a spec-level change — it needs an eval run (`news_nlp.eval`) comparing
   against the current baseline before it replaces the pinned checkpoint.
2. **Inference is deterministic and CPU-capable.** `DEVICE` selects CUDA
   when available, else CPU — every stage must run correctly (not just fast)
   on CPU, since hermetic tests and some deployments have no GPU. Don't add
   a code path that hard-requires CUDA.
3. **Fallback is fail-fast, not silent degradation.** If SOURCE text is
   unavailable, `require_source_text` must stop the run rather than have a
   stage silently emit empty/default predictions. A stage encountering
   malformed input (e.g. text too long for the model's context) chunks via
   `chunking.py` and merges spans — it does not truncate silently without
   recording that it did.
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

*Claude Code / coding-agent conduct on this repo:*

6. **Match existing structure before introducing new structure** — check
   where a file's siblings live and follow that placement, naming, and
   import style (`import news_nlp as db`, the `sys.path` bootstrap) rather
   than a generic layout.
7. **This constitution and `.specify/memory/SPEC.md` are the binding
   reference for planning and review** — read both before drafting a
   spec/plan, and resolve any conflict between a request and a stated
   principle or requirement by surfacing it or proposing an amendment, not
   by quietly overriding either. A local, untracked `CLAUDE.md` may carry
   situational/session notes, but it is never authoritative and must not be
   treated as a source of fact for anything either document already states.
8. **Prefer the smallest change consistent with the existing pattern**; no
   opportunistic refactors, renames, or new abstractions outside what the
   spec/task calls for.
9. **`CLAUDE.md` must always exist on disk and must never be deleted**,
   even though it is intentionally untracked, and it must always carry a
   reference to both this constitution (`.specify/memory/constitution.md`)
   and `.specify/memory/SPEC.md`. If `CLAUDE.md` is missing at the start of
   a session, run `/init` to regenerate it before doing anything else; if
   it exists but is missing either reference (freshly `/init`-generated or
   otherwise edited), add it before proceeding — don't treat the reference
   as a one-time regeneration step. Never `git checkout` /
   `git reset --hard` onto a commit older than PR #12 (`53522e8`) — that
   predates the file being untracked, and such a reset has previously wiped
   it from disk.
10. **Ask before expanding scope this constitution doesn't cover** — a new
    external service, a new heavy dependency, a schema change to
    RESULTS/SOURCE, or anything touching the two-tier DB contract.

## Executable cmds

Canonical commands — a spec/plan should reference these, not invent new
ad-hoc invocations:

```bash
uv sync                                     # install deps (dev group)
uv sync --group eval                        # + strands-agents/mlflow for news_nlp.eval
uv run python -m setup                      # pre-download the four HF models

uv run apps/news_nlp_api.py                 # FastAPI service -> :8003/docs
uv run cli/news_nlp_cli.py --limit 50       # batch pipeline; needs SOURCE_DATABASE_URL
uv run cli/news_nlp_eval.py --stage all --sample-size 80   # needs LLM_API_KEY/LLM_MODEL/LLM_URL
uv run mlflow ui                            # browse eval runs

uv run pytest                               # full hermetic suite (no network/GPU)
uv run pytest -q                            # as run in CI

uv run ruff check .                         # lint (config: .code_quality/ruff.toml via root pointer)
uv run ruff format --check .                # format check
uv run mypy --config-file=.code_quality/mypy.ini   # types

uv run pre-commit run --all-files           # all of the above hooks, plus hygiene checks
```

1. **CI (`ci.yml`) is the source of truth for the required gate order**:
   `uv sync --group dev --group eval` → ruff check → ruff format --check →
   mypy → `pytest -q`. Run the same four checks locally before opening a
   PR; don't rely on CI to catch a lint/type/test failure first.
2. **`eval.yml` is a separate, non-blocking workflow** for the real
   LLM-as-judge run — it is not part of the merge gate for ordinary PRs.
3. **Don't hardcode a different Python/uv invocation** (bare `python`,
   `pip install`, `pytest` without `uv run`) in scripts, docs, or CI — every
   command goes through `uv run` so it resolves the locked environment.

## Code & Git

1. **Formatting/linting/types are enforced, not advisory**: `ruff-check
   --fix` + `ruff-format` + `mypy` (project venv, whole-graph) all run via
   `pre-commit` and again in CI. A `# noqa` / `# type: ignore` needs a
   comment saying why the finding is wrong for this code, not just silence.
2. **Commit messages are Conventional Commits**, enforced by the
   `commitizen` pre-commit/pre-push hook — `type(scope): summary`, matching
   the existing history (`fix(ner): ...`, `feat(eval): ...`, `docs(eval):
   ...`, `chore: ...`, `refactor: ...`). Reference the PR number in the
   subject once it exists, as the existing log does (`(#29)`).
3. **Branch off `master`, never commit to it directly.** `master` is the
   integration branch (`origin/HEAD -> origin/master`); feature/fix/docs
   work happens on a descriptively-named branch (`fix/...`, `feat/...`,
   `docs/...`, `chore/...`, `refactor/...`) opened as a PR.
4. **Never `git checkout` / `git reset --hard` onto a commit older than PR
   #12 (`53522e8`)** — see the `CLAUDE.md`-protection rule under AI
   behavior. Keep local `master` fast-forwarded from `origin/master`
   instead of rewriting it.
5. **Pre-commit hooks are mandatory, not optional**: `check-yaml`,
   `check-case-conflict`, `debug-statements`, `detect-private-key`,
   `check-merge-conflict`, `check-added-large-files` run alongside
   ruff/mypy/commitizen — install them (`uv run pre-commit install`) rather
   than relying on remembering to run checks manually.
6. **CI must be green before merge**: the `lint-and-test` job in
   `.github/workflows/ci.yml` (ruff check, ruff format --check, mypy,
   `pytest -q`) gates every push to `master` and every PR. A PR that turns
   this red does not merge until it's fixed, not suppressed.
7. **No secrets committed.** `.env` stays git-ignored; `.env.example` holds
   placeholder values only; `detect-private-key` is a backstop, not the
   first line of defense — never paste a real key into a commit, issue, or
   PR description to "show" a config.

## Governance

This constitution supersedes ad-hoc convention when the two conflict. A
spec or plan may not silently contradict a rule above; instead:

1. Propose the amendment as its own change (state which section, what
   changes, and why).
2. Get it reviewed the same way a code PR would be (this repo's normal
   review path) before relying on it.
3. Bump the version below per semver: **MAJOR** for a removed/redefined
   principle, **MINOR** for a new principle or materially expanded
   guidance, **PATCH** for wording/typo fixes.
4. Record the change under "Last Amended" with the date.

Compliance is expected to be checked the same way lint/type/test gates
are — a reviewer (human or agent) rejecting a PR that violates a principle
above should cite the section by name.

**Version**: 2.1.0 | **Ratified**: 2026-09-11 | **Last Amended**: 2026-09-12
