# Step 2: the summary (T-164, T-166; PLAN D39 (2), (3.3), D40)

`uv run cli/news_nlp_steps.py --step summary [--items single|sub_news] [--limit N] [--scope-revision REV] [--cut-revision REV] [--status]`

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

**The revision** is `summary-` plus 12 hex digits of the SHA-256 of the model and its pinned revision, the size, the input format, the generation
limits and the source of `news_summary_stage.py`, `summary_stage.py` and `chunking.py`: a new size or model starts new rows and leaves the old ones.
`--cut-revision` names the cut whose sub-news are read (default the latest that has state); sub-news ids are unique, so two cuts never collide.
**The steps after this one read `news_summary` at the revision they name, never "the latest".** On CPU or GPU alike (CUDA when there is one).

## The measure behind 2,000 (2026-11-12)

Over the 93,034 `single` bodies of the corpus run (input `Title-…\nBody-…`) and the 3,134 sub-news of the first 500 `multi` bodies (`units4`):

| over (characters) | single bodies | sub-news | time for the singles at 0.84 s a body (a floor: long texts take more) |
|---|---|---|---|
| 1,500 | 75,368 (81.0%) | 89 (2.8%) | 17.5 h |
| **2,000** | **59,432 (63.9%)** | **50 (1.6%)** | **13.9 h** |
| 3,000 | 31,992 (34.4%) | 14 (0.4%) | 7.5 h |
| 4,000 | 17,621 (18.9%) | 4 (0.1%) | 4.1 h |

Single bodies: median 2,436 characters, p90 5,216, p99 11,715. Sub-news: median 268, p90 672, p99 2,309. The owner chose 2,000. The reason offered with it,
not checked against each downstream model: a 512-token transformer input is about 2,000 to 2,400 characters, so a text under 2,000 reaches NER, the
relevance model and sentiment whole, and a longer one is the kind that would be truncated.

## First run (2026-11-12, scratch copies of the databases, CPU)

12 single bodies, all over the size: 12 summaries of 287 to 561 characters (11 of one chunk, one of 6), no failure, `--status` adds up. On the CPU a body took about 32 s;
the GPU time of this model on this machine is **not measured yet** (the GPU was running the cut): the 0.84 s figure is the earlier run's.

## Not done

The run over the corpus (hours of GPU: it starts after the cut, and the sub-news wait for the cut's revision to be chosen, T-177); faithfulness of the summaries is not
measured; `sector` bodies (T-167); the steps after this one (NER on the text, relevance, sentiment, category).
