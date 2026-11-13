# portfolio-nlp

The financial-news NLP layer of the Portfolio Thesis, extracted from
[`portfolio-data-mining`](https://github.com/gamug/portfolio-data-mining). It reads the article text that repo
crawls (the SOURCE store, read-only) and writes its results to its own RESULTS store. There is no scraper here.

**Two pipelines live here, one replacing the other** (full detail: `docs/modules/news-nlp.md`):

- **The stepped pipeline (Work item 18; `.specify/memory/PLAN.md` D39, D40), the current approach.** It works on
  unique article bodies, decides what each body is about before anything reads it for a company, and runs one
  tracked, idempotent command per step (`cli/news_nlp_steps.py`).

  State on 2026-11-12:

  | Step | What it does | Model | Table | State |
  |---|---|---|---|---|
  | 1.1 scope | `single`, `multi`, `sector` or `out_of_scope` from the title and the first 800 characters | `gamug/deberta-v3-news-scope-nli` (pinned, public) | `body_scope` | built; ran on the corpus |
  | 1.2 cut | cuts a `multi` round-up into sub-news | the `units` extractor, reading `units5` (attribution checkpoint local, not pinned) | `sub_news` | built and run: `units5-d3e2cbfa830e`, 161,574 sub-news of 30,305 bodies; `units5` is final (owner 2026-11-12); faithful 84.6% on all 143 marked pieces, below the 80% bar by its interval, 89.1% without the 15 pieces scoped wrongly (`docs/scope_training.md` §8.3) |
  | 2 summary | summarises a text over 2,000 characters, keeps a shorter one as it is | `sshleifer/distilbart-cnn-12-6` (pinned) | `news_summary` | built; corpus run pending |
  | 3 companies | NER, universe tickers, the main company of each text | `gamug/sec-bert-finer-ord-ner` + `gamug/deberta-v3-company-relevance` (private, not pinned) | `news_company` | pending |
  | 4 sector | the GICS sector of a `sector` body, then its summary | to be chosen | | pending |
  | 6 sentiment | per text and main company | `gamug/FinBERT-company-sentiment` + relevance stack (private, not pinned) | `news_sentiment` | pending |
  | 7 category | per text and main company, the D15 taxonomy (`docs/category-taxonomy.md`) | trained by the owner (T-174) | `news_category` | pending |

- **The v1 chain, until the backfill (T-150).** Five stages over (URL, ticker) pairs, run by
  `cli/news_nlp_cli.py` and the FastAPI service: sentiment (`gamug/FinBERT-financial-news`), NER
  (`gamug/sec-bert-finer-ord-ner`), category (zero-shot `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`), and, with
  `--summarize`, `c_summary` (distilbart) and the deterministic `sector_summary`. It writes `article_sentiment`,
  `article_entities`, `article_category`, `article_summary` and `sector_summary`, which downstream repos still read.
  **Its sentiment, zero-shot category and `sector_summary` stages are deprecated** (owner, 2026-11-12;
  `docs/legacy/`).

The models of the stepped pipeline are measured on gold sets the owner tagged (`docs/gold_datasets.md`); LLMs only
label and evaluate, they never run inside the pipeline.

## Setup

```bash
uv sync
cp .env.example .env          # then set DATABASE_URL (+ SOURCE_DATABASE_URL to run the pipeline)
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
uv run python -m setup            # download the HF models
```

Opening the repo in the provided **Dev Container** (`.devcontainer/`) also bind-mounts a shared
`thesis` data directory at `/workspaces/thesis`. Two pieces of configuration:

- `.env` (repo root, git-ignored; `cp .env.example .env`) is passed to the container with
  `docker run --env-file` (`runArgs` in `devcontainer.json`), so its variables are set in the dev shell
  without `export`. The file must exist before the container is created, and its values must be plain
  `KEY=value` (no quotes, no `export`). Paths in it use the in-container layout
  (`/workspaces/thesis/...`). Optionally set `HF_TOKEN` there: a read token avoids anonymous rate limits
  in `python -m setup`; a write-scope token is required by `scripts/publish_*.py`. After editing it, run
  **Dev Containers: Rebuild Container**.
- **`THESIS_HOST_DIR`** — host path of the shared `thesis` data directory, bind-mounted at
  `/workspaces/thesis`. The `mounts` entry is resolved on the host before the container exists
  (`${localEnv:THESIS_HOST_DIR}`), so it must be set in your **host environment** (shell profile or OS
  environment variables, read before VS Code launches) — `.env` cannot provide it. E.g.
  `export THESIS_HOST_DIR=/Users/you/thesis` or, on Windows, `setx THESIS_HOST_DIR D:\thesis`. This
  variable name is shared across the sibling `portfolio-*` repos that mount the same directory.

## Run

```bash
# the stepped pipeline, one step per command (each takes --limit N and --status)
uv run cli/news_nlp_steps.py --step scope --limit 500
uv run cli/news_nlp_steps.py --step cut --limit 500                       # prints its reading and revision first
uv run cli/news_nlp_steps.py --step summary --items single                # single bodies
uv run cli/news_nlp_steps.py --step summary --cut-revision <cut revision> # sub-news of a named cut
uv run cli/news_nlp_steps.py --step cut --status

# the v1 chain
uv run apps/news_nlp_api.py            # FastAPI -> http://127.0.0.1:8003/docs
uv run cli/news_nlp_cli.py --limit 50  # 50 pending articles
```

Every command that reads article text needs `SOURCE_DATABASE_URL` set to a database with `articles.body_text`
(`docs/db-topology.md`). The steps' run rules: `docs/scope_training.md` §8 (scope and cut), `docs/summary_step.md`, `docs/companies_step.md`.

## Test / lint

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy --config-file=.code_quality/mypy.ini
uv run pre-commit run --all-files
```

## Database

Two-tier (see `docs/db-topology.md`):

- **`DATABASE_URL`** — the RESULTS store: the v1 result tables, the stepped pipeline's tables (`body_scope`,
  `sub_news`, `news_summary`, `pipeline_step_state`, and the ones its pending steps will write) and a
  `body_text`-free `articles` subset. A filesystem path; unset defaults to `data/nlp.db` under the repo root.
  Working file: `D:\thesis\data\nlp.db`.
- **`SOURCE_DATABASE_URL`** — the read-only SOURCE store, which must have `articles.body_text` (the crawl DB
  `urls.db`). **Required** by everything that reads text: the steps, and the v1 sentiment / NER / category /
  `c_summary` stages; it is ATTACHed read-only and never written. Not needed for the serving and query endpoints or
  the v1 `sector_summary` stage.

A run reads text from SOURCE and writes results, plus a lean `articles` row per article it used, into RESULTS, so the
RESULTS store stays self-consistent with no separate migration step. The steps read every properly fetched article,
`ok` and `quarantined` (a quarantined pair is a verdict on the company tag, not on the article; PLAN D36); the v1
chain reads `ok` only. (The first results were backfilled once from `urls.db`; see
`docs/legacy/migration-2026-09-01.md`.)

**Upgrading:** the old single-knob recipe (`DATABASE_URL=urls.db` to run,
`DATABASE_URL=nlp.db` to serve) now errors on a run — set
`SOURCE_DATABASE_URL=…/urls.db` **and** `DATABASE_URL=…/nlp.db` instead. Serving
is unchanged. Full table in `docs/db-topology.md` ("Upgrading from the
single-`DATABASE_URL` setup").

## Documentation

`docs/modules/news-nlp.md` (the module), `docs/db-topology.md` (databases), `docs/scope_training.md`,
`docs/extractor_training.md`, `docs/summary_step.md`, `docs/companies_step.md`, `docs/sentiment_training.md` (the models of the steps),
`docs/gold_datasets.md` and `docs/labelling-spec-v2.md` (the owner's gold sets and tagging rules),
`docs/category-taxonomy.md` (the D15 labels), `docs/evaluation.md` (the LLM-as-judge evaluation of the v1 stages).
What is out of scope or deprecated is in `docs/legacy/`. The project's rules and plan are in `.specify/memory/`.
