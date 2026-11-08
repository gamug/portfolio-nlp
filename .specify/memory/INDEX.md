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

- **Current focus:** Work item 18, triage-first pipeline (`PLAN.md`, `TASKS.md`). Decisions D11–D36 are all in `PLAN.md` Work item 18 ("Decisions" for D11–D24, then D25–D36); T-143 (the original record of D11–D24) and T-151 moved to `CHANGELOG.md`.
- **The corpus is preserved (D36, 2026-11-08):** a quarantined pair is a verdict on the company tag, not on the article; a properly fetched article (title and body) is never deleted or left out because of it; every size says which corpus it is of (`PLAN.md` D36, `TASKS.md` T-159, constitution #16). **D36 (h1) to (h3) are proposals awaiting the owner; remove this note when answered.**
- **Step 4 (company texts):** the extractor `units` was selected on 2026-11-08 (`TASKS.md` T-148b10 (9), `PLAN.md` D34 (e)); datasets, experiments, metrics and results are in `docs/extractor_training.md`; wiring it into the cut is T-148b12 (proposed).
- **Closed work items:** 1, 5, 6, 8–14, 16 (and 15, superseded by 18). Their plan is in `archive/PLAN-closed-work-items-2026-11-08.md` (grep `## Work item N`), their tasks in `CHANGELOG.md`. **A reference such as "PLAN.md Work item 10 step 3" in a doc or a code comment resolves in that archive file.**
- **Work items 3, 4, 7:** resolved except T-022, T-032, T-062 (in `TASKS.md`); their plan and **acceptance criteria are only in the archive**.
- **Resolved risks:** `SPEC.md` §13 keeps one short paragraph per item; the long text is in `archive/SPEC-open-questions-2026-11-08.md`.
- **Long open-task histories:** `archive/TASKS-open-entries-2026-11-08.md`, one section per task ID.
- **Evidence, figures, review history:** `docs/evaluation.md` (dated follow-ups), `docs/extractor_training.md`, `docs/gold_datasets.md`, and the PR bodies. A task entry points there; it does not repeat it.

## Rules for keeping the files small (constitution AI behavior #15)

1. A task entry is a status, the decision, what is open and a pointer. Figures, tables and review rounds go to `docs/` or the PR.
2. When a task is checked, move it to `CHANGELOG.md` ("Tasks closed while their work item is still open") in the same change; when a whole work item closes, move its section and its `PLAN.md` section to `CHANGELOG.md` and `archive/`, leaving the heading and one line.
3. A superseded rule is replaced, not annotated: the old text goes to `archive/`.
4. Moves are verbatim and task IDs never change. The archive files record the commit they were cut from.
