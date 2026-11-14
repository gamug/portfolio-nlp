# Step 2: the summary (T-164, T-166, T-184; PLAN D39 (2), (3.3), D40)

`uv run cli/news_nlp_steps.py --step summary (--cut-revision REV | --items single | --items sector --sector-revision REV) [--items sub_news] [--limit N] [--scope-revision REV]` (and `--step summary --status`)

## What it does

It gives every `single` body of step 1 and every sub-news of the cut **one text for the steps after it to read**:

- a text **over 2,000 characters** is summarised by the pinned local summariser (`config/pipeline_models.json` `c_summary`:
  `sshleifer/distilbart-cnn-12-6` at revision `a4f8f3ea…`), with the chunk-and-reduce of `summary_stage.hierarchical_summarize_batch`
  (the v1 stage's own function, reused as it is: 1,000-token chunks on sentence boundaries, summaries of 56 to 142 tokens, reduced until one is left);
  the row says `text_source = summary` and the number of chunks;
- a text **at or under 2,000 characters** is stored unchanged, `text_source = original`, no chunks.

The size is the owner's (2026-11-12). It is measured on the text the summariser would read: for a single body that is
`Title-<title>\nBody-<body>` (the input of the earlier summary run), for a sub-news its own text (the title belongs to the whole round-up).
**Sector bodies** (`--items sector --sector-revision REV`, T-167): the bodies the sector step assigned a sector (`body_sector.status = 'assigned'`) at the
revision named, with the same size rule and the same input as a single body (`item_kind = 'body'`, the same summary revision, so nothing already summarised is redone;
a body summarised as a single one under another scope revision keeps its summary). The sector revision is **named, never the latest** (the ones the sector step has are
listed when it is missing or wrong) and the sector step must have run first: a `sector` body it has not reached, and one it sent `out_of_scope`, are not read. It is not
part of the default (`single` and `sub_news`); `--cut-revision` and `--scope-revision` do not apply to it.

Rows go to `news_summary` (`item_kind` `body` with the body hash, or `sub_news` with the `sub_news.id`; no schema change), state to
`pipeline_step_state` (step `summary`). The step is idempotent and resumable like the scope and the cut: a rerun at the same revision adds
nothing, a failed item is retried and keeps its reason (`Type: message`), a batch that fails is redone item by item, an empty summary is a
failure (never stored), `--limit N` touches at most N pending items (single bodies first), and three batches in a row that fail completely stop the run.
Options the step does not read are refused, never ignored: `--model-dir`, `--reading`, `--titles`, `--fetch` and `--allow-unpinned` (the scope's and the cut's), `--cut-revision` with `--items single`, and `--scope-revision` with `--items sub_news`
(it names the single bodies to read; the JSON a run prints lists only the revisions it read).

**The revision** is `summary-` plus 12 hex digits of the SHA-256 of the model and its pinned revision, **the `transformers` version** (beam search lives there and `pyproject.toml` only requires `>=4.56`; `uv.lock` pins 5.16.1: a lock bump starts a new revision instead of mixing old and new summaries under one; `torch` is left out because its version carries the build, `+cu124` or `+cpu`, and the revision is the same on both devices), the size, the input format, the generation limits and
the code that summarises: `news_summary_stage.py` whole, and the functions this step runs in `summary_stage.py` (`hierarchical_summarize_batch` and
its helpers, `MAX_REDUCE_PASSES`) and `chunking.py` (`chunk_text`, `split_sentences`, `Chunk`, the sentence regex): `news_summary_stage.hashed_sources()` lists them. **The code is hashed as parsed
(`normalized_code`): comments, formatting, the kind of quotes and docstrings do not change the revision, so a typo fix or a formatter upgrade does not make the step redo hours of GPU work; a change of
a name, a constant or an operator does.** A new size, model or such a change starts new rows and leaves the old ones. **An edit to the rest of those two files does not either**: `summary_stage.py` also holds the v1 stage and `chunking.py` the NER span merge, and
shared code that this step does not run must not make it redo about 15 GPU hours (a test edits both and checks the revision stays). On CPU or GPU alike (CUDA when there is one).

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

## What a run prints, and what the bars count

A run first says what it is about to do, on stderr, and then works in two phases with a bar each:

```
pending 254,608 items (93,034 single bodies, 161,574 sub-news): 64,480 to summarise with the model (over 2,000 characters), 190,128 stored as they are
original: 100%|...| 190128/190128     <- the items under the size: a write each, no model
summary:   12%|...| 7738/64480        <- the items over the size: the model; its rate and time left are the real ones
```

The numbers above are computed for the owner's corpus, not printed by a full run: the single bodies of scope revision `scope-nli-8ff21e50796b` (93,034, of which 59,432 are over the size) and the sub-news of the reference cut `units5-d3e2cbfa830e` (161,574, of which 5,048 are over it: 3.12%, measured 2026-11-13 on a consistent copy of the owner's database with `--cut-revision units5-d3e2cbfa830e`, below). 59,432 + 5,048 = 64,480 items go to the model and 93,034 + 161,574 - 64,480 = 190,128 are stored as they are. (The first version of this plan was computed on the older `units` cut, `units-5f831c298bcb`: 165,192 sub-news, 5,046 over the size, 64,478 to summarise; the reading barely changes the share, 3.05% against 3.12%, and the GPU estimate does not move.) The size table below says why the whole cut and not a sample: on the first 500 `multi` bodies by id the `units5` cut gives 50 over of 3,233 (1.5%), so the sample under-reads by half. **The 254,608 is not a count of articles**: it is single bodies plus sub-news (a body of another scope, `multi`, `sector` or
`out_of_scope`, is not in it), and about three quarters of it is stored as it is. The first version of the step put all of them on one bar, which mixed instant writes (286 a second measured) with seconds per model item, so
neither its rate nor its time left meant anything (reported by the owner, 2026-11-12). With `--limit N` the plan counts what this run will do, after the limit; with nothing to do it says `nothing pending`.

The `original` phase writes one transaction per item: 286 items a second on a local disk (a temporary database, 3,000 items), about 11 minutes for 190,128 items; over a mounted disk each commit can cost more, and a commit every
few hundred items would cut it to seconds (not done: the model phase is the cost).

## The measure behind 2,000 (2026-11-12)

`uv run scripts/summary_size_measure_2026_11_12.py --results-db <nlp.db copy> --source-db <urls.db copy> --cut-revision units5-d3e2cbfa830e --first-bodies 500` (read-only)
gave, over the 93,034 `single` bodies of the corpus run (input `Title-…\nBody-…`) and the 161,574 sub-news of the reference cut (**the whole cut; its first 500 bodies give 1.5% over 2,000, a sample that under-reads**):

| over (characters) | single bodies | sub-news (whole `units5` cut) | time for the singles at 0.84 s a body (an extrapolation) |
|---|---|---|---|
| 1,500 | 75,368 (81.0%) | 9,249 (5.7%) | 17.6 h |
| **2,000** | **59,432 (63.9%)** | **5,048 (3.1%)** | **13.9 h** |
| 3,000 | 31,992 (34.4%) | 1,731 (1.1%) | 7.5 h |
| 4,000 | 17,621 (18.9%) | 708 (0.4%) | 4.1 h |

`--first-bodies 500` prints the sub-news of the cut twice, whole and for its first 500 bodies, which is how the 1.5% and the 3.1% were compared.

Single bodies: median 2,436 characters, p90 5,216, p99 11,715. Sub-news, the whole `units5` cut (161,574): median 280, p90 1,077, p99 3,065 (the first-500 sample: median 262, p90 647, p99 2,303: its tail is shorter than the corpus's). The owner chose 2,000. The reason offered with it,
not checked against each downstream model: a 512-token transformer input is about 2,000 to 2,400 characters, so a text under 2,000 reaches NER, the
relevance model and sentiment whole, and a longer one is the kind that would be truncated. The 0.84 s a body is the earlier summary run's figure (all sizes of
bodies, on the GPU); it is **not** a measure of this model on this machine, and long texts take more chunks, which makes the figure more likely low than high (a reason, not a measure), so read the hours as an order of magnitude until the first GPU run. For the whole step: 59,432 + 5,048 = 64,480 items × 0.84 s = 54,163 s, about 15 h.

## First run, determinism and running next to the cut (2026-11-12, scratch copies of the databases, CPU)

12 single bodies, all over the size, at revision `summary-7a9bd2062bc4` (the revision of the code of this PR): 12 summaries, no failure, `--status` adds up, and the printed JSON lists only the
revisions the run read. A body took 13 to 32 s on the CPU depending on the load of the machine; the GPU time is **not measured yet**.

**Determinism.** `scripts/summary_determinism_check_2026_11_12.py -n 4` (the constitution asks for deterministic inference) on those rows, **like for like** (stored rows and code at the same revision): the
four stored summaries were summarised again in a new process, twice: **4 of 4 compared, 0 differ from the stored ones, 0 differ between the two runs** (CPU; the result line also prints the versions that produced them: torch 2.6.0+cu124, transformers 5.16.1). The check is **not a pass unless it compared what it was
asked to**: rows of another revision, a source without the bodies (`found 0 of 4`, run for real on an empty source) or with only some of them (`found 1 of 2`) end with **exit 2** and the reason; `--allow-other-revision` goes on
and says so. Exit codes: 0 clean, 1 a summary differs, 2 not comparable. `--kind sub_news` reads the sub-news path (tested). **Not checked: the GPU**, which is the run that matters (a GPU can flip a beam near a
tie): run it on the GPU after the cut and record the result here.

**Next to the cut.** Both steps write the same RESULTS file, and SQLite takes one writer at a time (`portfolio_common` waits 30 s for the lock). On the scratch copy the cut of 15 bodies (99 sub-news, 64 s) and the
summary of 6 items (134 s) ran **at the same time**: both exited 0, no lock error, no traceback. The safe way is still one writer at a time; if both run and a write waits more than 30 s, `database is locked` ends the
run with a traceback (it is not caught inside `record_summary`): the batch in hand is lost and every item already committed stays, so the next run continues from there.

## Limits

A text is chunked on sentence boundaries and each pass is cut to the model's input (`truncation=True`); when `MAX_REDUCE_PASSES` (6) is exhausted before the summaries
collapse to one, a last pass drops the tail without a record (`summary_stage.hierarchical_summarize_batch`, inherited from the v1 stage, very rare); `news_summary`
stores the number of chunks, not whether this happened. The summaries are model output, not facts (constitution, Pipeline models 4); their faithfulness is **not measured**.

## Not done

All in T-184: the run over the corpus (hours of GPU; the single bodies do not depend on the cut and can run now; the sub-news are read from the reference cut `units5-d3e2cbfa830e`, whose corpus run has finished: 161,574 sub-news of 30,305 bodies, 5,048 over the size, 64,480 items in all to summarise); the GPU time per item and the determinism check on the GPU; the faithfulness of the summaries.
`sector` bodies (T-167); the steps after this one (NER on the text, relevance, sentiment, category).
