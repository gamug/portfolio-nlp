# TASKS.md — `portfolio-nlp`

Discrete, checkable task breakdown for `.specify/memory/PLAN.md`. Each
task references the plan work item and the `SPEC.md` section it closes.
Check a box only when its acceptance criterion (in `PLAN.md`) is actually
met — not when the code is merely written.

Task IDs are stable, same rule as `SPEC.md`'s `FR-0xx`/`NR-0xx`: don't
renumber; mark a cancelled/superseded task in place instead.

## Work item 1 — Pin HF model checkpoints (code, no blockers)

- [ ] **T-001** Look up and record the current commit SHA for each of the
      four HF model repos (`ProsusAI/finbert`, `gamug/sec-bert-finer-ord-ner`,
      `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`,
      `sshleifer/distilbart-cnn-12-6`) — from the HF Hub API/UI, not
      guessed. → `PLAN.md` Work item 1, step 1.
- [ ] **T-002** Add a revision constant per model in `src/pipeline.py`
      (paired with the existing `SENTIMENT_MODEL`/`NER_MODEL`/
      `CATEGORY_MODEL`/`SUMMARY_MODEL` name constants). → step 1.
- [ ] **T-003** Pass `revision=` at all 9 `from_pretrained` call sites in
      `src/pipeline.py` (lines 150–151, 283–284, 506–507, 747–748,
      785–786). → step 2.
- [ ] **T-004** Pass `revision=` in `src/setup.py`'s `download_models()`
      (`snapshot_download` + `AutoConfig.from_pretrained`, both calls, for
      all four models). → step 2.
- [ ] **T-005** Document the four pins (comment or a table in
      `docs/modules/news-nlp.md`) so a future bump is a reviewed, visible
      diff. → step 3.
- [ ] **T-006** Verify: `grep -rn "from_pretrained\|snapshot_download" src/`
      shows `revision=` at every call site; `uv run python -m setup`
      succeeds; `uv run pytest` stays green. → `PLAN.md` acceptance
      criteria.
- [ ] **T-007** Update `SPEC.md` §13 item 4 to note the reproducibility
      half resolved (annotate in place, keep the item number). → `PLAN.md`
      Work item 1, last acceptance criterion. Also update the two
      architecture artifacts per constitution AI behavior #11 (Portfolio
      Thesis + Portfolio NLP) — reconcile, never rename.

## Work item 2 — Make the regression gate runnable (ops, blocked on maintainer)

- [ ] **T-010** *(maintainer)* Designate/stand up a machine with durable
      access to `$SOURCE_DATABASE_URL`/`$DATABASE_URL` and a persistent
      `mlruns` dir or MLflow server. → `PLAN.md` Work item 2, step 1.
- [ ] **T-011** *(maintainer)* Register it as a GitHub self-hosted runner
      labeled `thesis-data`. → step 2.
- [ ] **T-012** *(maintainer)* Add repo secrets: `LLM_API_KEY`,
      `LLM_MODEL`, `LLM_URL`, `MLFLOW_TRACKING_URI`,
      `SOURCE_DATABASE_URL`, `DATABASE_URL`. → step 3.
- [ ] **T-013** Edit `.github/workflows/eval.yml`: uncomment
      `runs-on: [self-hosted, thesis-data]`, remove the
      `runs-on: ubuntu-latest` placeholder. → step 4. (Code change, but
      meaningless without T-010–T-012 done first — don't land this alone
      expecting it to work.)
- [ ] **T-014** *(maintainer)* Trigger `workflow_dispatch` once; confirm a
      real MLflow run + `eval_run` row is produced. → step 5 / first
      acceptance criterion.
- [ ] **T-015** *(maintainer)* Confirm the gate actually gates: induce a
      regression (temporarily lowered `--regression-tolerance`, or a
      second run compared against a known-worse one) and confirm the
      workflow fails. → second acceptance criterion.
- [ ] **T-016** Update `SPEC.md` §13 item 8 to reflect the resolved (or
      deliberately re-scoped) state. → third acceptance criterion. Also
      update the two architecture artifacts per constitution AI behavior
      #11 — reconcile, never rename.

## Status

Nothing above is started. T-001–T-007 have no blockers and can begin
immediately; T-010–T-016 are blocked on the maintainer's infrastructure
decision (see `PLAN.md` Work item 2).
