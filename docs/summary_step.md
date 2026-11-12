# Step 2: the summary (T-164, T-166; PLAN D39 (2), (3.3), D40)

`uv run cli/news_nlp_steps.py --step summary [--items single|sub_news] [--cut-revision REV] [--limit N] [--scope-revision REV] [--status]`

## What it does

It gives every `single` body of step 1 and every sub-news of the cut **one text for the steps after it to read**:

- a text **over 2,000 characters** is summarised by the pinned local summariser (`config/pipeline_models.json` `c_summary`:
  `sshleifer/distilbart-cnn-12-6` at revision `a4f8f3ea…`), with the chunk-and-reduce of `summary_stage.hierarchical_summarize_batch`
  (the v1 stage's own function, reused as it is: 1,000-token chunks on sentence boundaries, summaries of 56 to 142 tokens, reduced until one is left);
  the row says `text_source = summary` and the number of chunks;
- a text **at or under 2,000 characters** is stored unchanged, `text_source = original`, no chunks.

The size is the owner's (2026-11-12). It is measured on the text the summariser would read: for a single body that is
`Title-<title>\nBody-<body>` (the input of the earlier summary run), for a sub-news its own text (the title belongs to the whole round-up).
`sector` bodies are not here: step 4 identifies the sector first (T-167).

Rows go to `news_summary` (`item_kind` `body` with the body hash, or `sub_news` with the `sub_news.id`; no schema change), state to
`pipeline_step_state` (step `summary`). The step is idempotent and resumable like the scope and the cut: a rerun at the same revision adds
nothing, a failed item is retried and keeps its reason (`Type: message`), a batch that fails is redone item by item, an empty summary is a
failure (never stored), `--limit N` touches at most N pending items (single bodies first), and three batches in a row that fail completely stop the run.
Options the step does not read are refused, never ignored: `--model-dir`, `--reading`, the scope-only flags, `--cut-revision` with `--items single`, and `--scope-revision` with `--items sub_news`
(it names the single bodies to read; the JSON a run prints lists only the revisions it read).

**The revision** is `summary-` plus 12 hex digits of the SHA-256 of the model and its pinned revision, the size, the input format, the generation limits and
the code that summarises: the source of `news_summary_stage.py` whole, and of the functions this step runs in `summary_stage.py` (`hierarchical_summarize_batch` and
its helpers, `MAX_REDUCE_PASSES`) and `chunking.py` (`chunk_text`, `split_sentences`, `Chunk`, the sentence regex): `news_summary_stage.hashed_sources()` lists them. A new size, model or edit
of that code starts new rows and leaves the old ones. **An edit to the rest of those two files does not**: `summary_stage.py` also holds the v1 stage and `chunking.py` the NER span merge, and
shared code that this step does not run must not make it redo about 14 GPU hours (a test edits both and checks the revision stays). On CPU or GPU alike (CUDA when there is one).

**A cut is named, never "the latest".** The sub-news are read from the cut revision given with `--cut-revision`; without it the command stops and lists the
cuts that ran (`--items single` needs none). The same rule as for every reader of `sub_news` (`docs/scope_training.md` §8.1). **The steps after this one read
`news_summary` at the revision they name, never "the latest".**

## Running it, and running it again

Run it in slices (`--limit 2000`, repeated): a run holds the text of every pending item in memory, about 280 MB for all the 93,034 single bodies, and writes
one transaction per item. `--status` prints the counts by state, by source (`summary`, `original`) and by kind, and the revisions.

Summarise again at one revision of the summary step (`<summary revision>`: what `--status` prints; the state is keyed by revision and `news_summary` is written with `INSERT OR IGNORE`, so deleting one table is not enough):

```sql
DELETE FROM pipeline_step_state WHERE step = 'summary' AND model_revision = '<summary revision>';
DELETE FROM news_summary WHERE model_revision = '<summary revision>';
```

**Before cutting again a revision of the cut** (`<cut revision>`: its `extractor_revision`) (`docs/scope_training.md` §8.1), delete its summaries: they are keyed by `sub_news.id`, and new ids would leave them
pointing at nothing:

```sql
DELETE FROM news_summary WHERE item_kind = 'sub_news' AND item_id IN (SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');
DELETE FROM pipeline_step_state WHERE step = 'summary' AND item_kind = 'sub_news' AND item_id IN (SELECT CAST(id AS TEXT) FROM sub_news WHERE extractor_revision = '<cut revision>');
```

The same statements are in `--help` (one source in `cli/news_nlp_steps.py`, tested against a database).

## The measure behind 2,000 (2026-11-12)

`uv run scripts/summary_size_measure_2026_11_12.py --results-db <nlp.db copy> --source-db <urls.db copy> --cut-revision units4-e1dff4f90b24` (read-only)
gave, over the 93,034 `single` bodies of the corpus run (input `Title-…\nBody-…`) and the 3,134 sub-news of the first 500 `multi` bodies (`units4`):

| over (characters) | single bodies | sub-news | time for the singles at 0.84 s a body (an extrapolation) |
|---|---|---|---|
| 1,500 | 75,368 (81.0%) | 89 (2.8%) | 17.6 h |
| **2,000** | **59,432 (63.9%)** | **50 (1.6%)** | **13.9 h** |
| 3,000 | 31,992 (34.4%) | 14 (0.4%) | 7.5 h |
| 4,000 | 17,621 (18.9%) | 4 (0.1%) | 4.1 h |

Single bodies: median 2,436 characters, p90 5,216, p99 11,715. Sub-news: median 268, p90 672, p99 2,309. The owner chose 2,000. The reason offered with it,
not checked against each downstream model: a 512-token transformer input is about 2,000 to 2,400 characters, so a text under 2,000 reaches NER, the
relevance model and sentiment whole, and a longer one is the kind that would be truncated. The 0.84 s a body is the earlier summary run's figure (all sizes of
bodies, on the GPU); it is **not** a measure of this model on this machine, and long texts take more chunks, so read the hours as an order of magnitude until the first GPU run.

## First run and determinism (2026-11-12, scratch copies of the databases, CPU)

12 single bodies, all over the size, at revision `summary-be70abc39af7`: 12 summaries, no failure, `--status` adds up, and the printed JSON lists only the revisions the run read. The first
run (an earlier revision, before the hash covered the stage) gave summaries of 287 to 561 characters, 11 of one chunk and one of 6. A body took 13 to 32 s on the CPU depending on the load of the
machine; the GPU time is **not measured yet**.

`scripts/summary_determinism_check_2026_11_12.py -n 4` (the constitution asks for deterministic inference) on those rows, **like for like** (the stored rows and the code at the same revision):
the four stored summaries were summarised again in a new process, twice: **0 differ from the stored ones, 0 differ between the two runs** (CPU). The check refuses to compare rows of another
revision: it prints both revisions and **exits 2**, which is not a pass (`--allow-other-revision` goes on and says so in the result). The first determinism run of this PR (2 items, 0 and 0) had the
rows at an older revision and exited 0 with a warning; it is replaced by this one. **Not checked: the GPU**, which is the run that matters (a GPU can flip a beam near a tie): run it on the GPU
after the cut and record the result here.

## Limits

A text is chunked on sentence boundaries and each pass is cut to the model's input (`truncation=True`); when `MAX_REDUCE_PASSES` (6) is exhausted before the summaries
collapse to one, a last pass drops the tail without a record (`summary_stage.hierarchical_summarize_batch`, inherited from the v1 stage, very rare); `news_summary`
stores the number of chunks, not whether this happened. The summaries are model output, not facts (constitution, Pipeline models 4); their faithfulness is **not measured**.

## Not done

The run over the corpus (hours of GPU: it starts after the cut, and the sub-news wait for the cut revision to be chosen, T-177); the faithfulness of the summaries;
`sector` bodies (T-167); the steps after this one (NER on the text, relevance, sentiment, category).
