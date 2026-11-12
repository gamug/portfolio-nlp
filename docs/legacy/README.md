# docs/legacy

What is here is **out of the current scope or deprecated**, kept as the record (owner, 2026-11-12). Nothing in it
describes the current approach; the current docs are one folder up. Files moved whole keep their text as it was;
sections cut from a current doc are copied verbatim and say where they came from (file, lines, commit `09d90ae`).

| File | What it is | Why it is here | What replaces it |
|---|---|---|---|
| `category-taxonomy-v1.md` (was `docs/category-taxonomy.md`) | the 10-slug taxonomy of the v1 zero-shot category stage: sources, hierarchy, thresholds | the v1 zero-shot category stage is deprecated (owner, 2026-11-12); its code still runs until the backfill (`news_nlp.taxonomy` holds the slugs) | `docs/category-taxonomy.md` (the D15 taxonomy) |
| `news-nlp-v1-stages.md` | the v1 sentiment, zero-shot category and `sector_summary` stages, and the old Evaluation section of `docs/modules/news-nlp.md` | those stages are deprecated (owner, 2026-11-12) | `docs/modules/news-nlp.md`, the stepped pipeline |
| `evaluation-v1-stages.md` | the LLM-as-judge follow-ups of the v1 sentiment, category and `sector_summary` stages, and of Work item 15 | deprecated stages; Work item 15 superseded by Work item 18 (2026-10-28) | `docs/evaluation.md` keeps the machinery, NER, `c_summary` and Work item 18; the new models are measured on the owner's gold (`docs/gold_datasets.md`) |
| `gold_datasets_roadmap_2026-11-06.md` | the order to tag the gold sets against the dates of D34 | D34 (the backtest cut for the 2026-11-20 delivery) is out of the MVP; every set is tagged | `docs/gold_datasets.md` |
| `migration-2026-09-01.md` | the one-time backfill of `nlp.db` from `urls.db` | done; the script was removed | `docs/db-topology.md` |
| `portfolio-common-v1-migration-plan.md` | the plan to vendor `news_nlp` back from `portfolio-common` v1.0.0 | executed in #21 | `docs/db-topology.md` |
| `portfolio-common-v1.1-news-export.md` | why `fetch_processed_articles` delegates to `portfolio_common.news_export` | executed (v1.1.0); the fact is in `docs/db-topology.md`, "External consumers" | `docs/db-topology.md` |
| `portfolio-common-v1.2-engine-agnostic.md` | the move of `src/news_nlp/` onto the `Dialect` seam | executed in #24/#25 | `docs/db-topology.md` |
| `engine-agnostic-rollout.md` | the cross-repository plan for the seam | done in every repository (its status table is of its own day); a pointer stays at `docs/engine-agnostic-rollout.md` because the constitution cites that path | `docs/db-topology.md` |

`.specify/memory/CHANGELOG.md` and `.specify/memory/archive/` quote the old paths as they were; read them with this table.
