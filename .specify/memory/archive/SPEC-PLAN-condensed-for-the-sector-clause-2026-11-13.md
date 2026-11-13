# Sentences condensed to make room for the requirement of the sector step, 2026-11-13

Verbatim wording moved out of `SPEC.md` (FR-018, FR-021) and `PLAN.md` (D39 (4), the numbered list) when the fifth review of #179 asked for a requirement and an
acceptance for the sector step in FR-020 and `SPEC.md` had 8 characters of room (INDEX rule 3: a superseded rule is replaced, the old text goes to `archive/`). The meaning
is the same except for the sector step's own item (the PLAN list said "the sector (GICS) is identified, then the text is summarised", before the owner's decisions of 2026-11-13).
Cut from commit `02cef7be41cc7ca9a54beffd34f55a3d746a74ef`.

## FR-018

- **Was:** This is a format change only, not a relaxation of constitution AI-behavior #1 ("model selection is pinned, not dynamic"): changing a model still requires editing this file and going through normal PR review, same as editing a `.py` constant did before.
- **Now:** A format change only, not a relaxation of constitution AI #1 ("model selection is pinned, not dynamic"): changing a model still means editing this file in a reviewed PR.

## FR-021

- **Was:** the cut reads a body as `units5` unless `--reading units4` or `--reading units` is given (T-177, #176; `units4` was the default of 2026-11-11, T-176), prints its reading and revision before it starts, and the reading is the start of the cut's revision, so a reader names the revision (`units5-…`) it wants (a test asserts the default, the start line and the distinct revisions).
- **Now:** the cut reads a body as `units5` unless `--reading units4` or `--reading units` is given (T-177), prints its reading and revision before it starts, and the reading is the start of its revision (a test asserts the default, the start line and the distinct revisions).

## FR-018

- **Was:** (repository, commit, step revision; `scripts/publish_scope_model_2026_11_11.py --check` verifies it): no stage of the v1 pipeline uses it, and its pin carries a step revision the four stages do not have.
- **Now:** (repository, commit, step revision; `scripts/publish_scope_model_2026_11_11.py --check` verifies it); no v1 stage uses it. The sector embedder is pinned in `config/sector_step.json`.

## FR-021

- **Was:** (`review` is neither done nor failed: the run is not complete while items wait, T-172)
- **Now:** (`review` is neither done nor failed, T-172)

## FR-018

- **Was:** `sentiment` has no `batch_size` field (nothing in `sentiment_stage.py` reads one — its batch size is hardcoded to 1).
- **Now:** `sentiment` has no `batch_size` (its batch size is hardcoded to 1).

## FR-021

- **Was:** no command runs all steps (a test asserts the step list is run one at a time);
- **Now:** no command runs all steps (tested);

## PLAN D39 (4), the numbered list

- **Was:** (4) **sector**:
  the sector (GICS) is identified, then the text is summarised.
- **Now:** (4) **sector**:
  a 2-digit GICS sector or `out_of_scope` is given (embedder, company votes and a gate: `docs/sector_step.md`); its bodies are summarised in step 2 (`--items sector`).

