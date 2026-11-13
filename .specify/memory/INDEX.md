# INDEX.md — how to read `.specify/memory/` without loading all of it

The planning files are large. **Read this page, then only the section the task needs** (`grep -n '^## ' <file>`
lists a file's sections; read that range). Do not load whole files by default.

| File | What it is | Read it when | Size limit (kept by `tests/test_memory_size.py`, which holds the numbers) |
|---|---|---|---|
| `constitution.md` | principles and AI behaviour rules | always, once per session | none |
| `SPEC.md` | requirements (§2 FR/NR), architecture, risks (§13) | the task touches a stated requirement; §2 and the section of the stage | capped |
| `PLAN.md` | the plan: one `## Work item N` per item, decisions D-numbers inside Work item 18 | the task belongs to an open work item; today **Work item 18** (and 2, 17) | capped, and each `## Work item` section |
| `TASKS.md` | open work only: one entry per open task | looking up a task by ID: `grep -n 'T-148b10' TASKS.md`; **if the ID is not there it was moved: search `CHANGELOG.md`, then `archive/`** | capped, and each entry |
| `CHANGELOG.md` | done work: closed work items, and "Tasks closed while their work item is still open" | a closed task's history is needed | none (read by search) |
| `archive/` | verbatim text moved out of the files above (closed plans, long task entries, resolved risks) | the history of one item is needed | none (read by search) |

## Where things are

- **Current focus:** Work item 18, triage-first pipeline (`PLAN.md`, `TASKS.md`). Decisions D11–D39 are all in `PLAN.md` Work item 18 ("Decisions" for D11–D24, then D25–D39; D25 and D34 are pointers); T-143 (the original record of D11–D24) and T-151 moved to `CHANGELOG.md`.
- **The corpus is preserved (D36, 2026-11-08):** a quarantined pair is a verdict on the company tag, not on the article; a properly fetched article (title and body) is never deleted or left out because of it; every size says which corpus it is of (`PLAN.md` D36, `TASKS.md` T-160, constitution #16). D36 (h1) to (h3) were decided by the owner on 2026-11-09.
- **Step 4 (company texts):** the extractor `units` was selected on 2026-11-08 (`TASKS.md` T-148b10 (9), `PLAN.md` D34 (e)); datasets, experiments, metrics and results are in `docs/extractor_training.md`; wiring it into the cut is T-148b12 (proposed).
- **Closed work items:** 1, 5, 6, 8–14, 16 (and 15, superseded by 18). Their plan is in `archive/PLAN-closed-work-items-2026-11-08.md` (grep `## Work item N`), their tasks in `CHANGELOG.md`. **A reference such as "PLAN.md Work item 10 step 3" in a doc or a code comment resolves in that archive file.**
- **Work items 3, 4, 7:** resolved except T-022, T-032, T-062 (in `TASKS.md`); their plan and **acceptance criteria are only in the archive**.
- **Resolved risks:** `SPEC.md` §13 keeps one short paragraph per item; the long text is in `archive/SPEC-open-questions-2026-11-08.md`.
- **D35 "Order of the end" (long form):** `archive/PLAN-d35-order-of-the-end-2026-11-10.md` (`PLAN.md` keeps a short form).
- **The pipeline (D39, owner 2026-11-10):** `PLAN.md` D39 and D40 (the steps, the build order, the scope classifier, the three conditions, the owner's answers of 2026-11-10), `SPEC.md` FR-020/FR-021, `TASKS.md` T-147 (scope classifier) and T-162 to T-185. **Step 2, the summary (size 2,000 characters, owner 2026-11-12):** `docs/summary_step.md` (the measure, the rules of a run), `TASKS.md` T-166. **Step 3, the companies (built and decided 2026-11-13: threshold 0.5, candidates `ner+text`):** `docs/companies_step.md` (the rules, the dry run, the limits), `TASKS.md` T-164, T-165, T-185. The replaced text (D25, D34, parts of D27, D32 and D39, T-148, T-149, the v1 diagram and workflow, two FR sentences): `archive/PIPELINE-superseded-2026-11-10.md`.
- **Long open-task histories:** `archive/TASKS-open-entries-2026-11-08.md`, one section per task ID.
- **Legacy docs (2026-11-12):** `docs/legacy/` holds what is out of scope or deprecated (the v1 sentiment, zero-shot category and `sector_summary` stages, Work item 15, finished migrations); its `README.md` maps each old path to the new one.
- **Evidence, figures, review history:** `docs/evaluation.md` (dated follow-ups), `docs/extractor_training.md`, `docs/gold_datasets.md`, and the PR bodies. A task entry points there; it does not repeat it.

## Rules for keeping the files small (constitution AI behavior #15)

1. A task entry is a status, the decision, what is open and a pointer. Figures, tables and review rounds go to `docs/` or the PR.
2. When a task is checked, move it to `CHANGELOG.md` ("Tasks closed while their work item is still open") in the same change; when a whole work item closes, move its section and its `PLAN.md` section to `CHANGELOG.md` and `archive/`, leaving the heading and one line.
3. A superseded rule is replaced, not annotated: the old text goes to `archive/`.
4. Moves are verbatim and task IDs never change. The archive files record the commit they were cut from.
