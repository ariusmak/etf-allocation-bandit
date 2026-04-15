# CLAUDE.md

## Purpose of this file

This file gives project context, objectives, constraints, and working norms for an agent operating inside this repository.

This is **not** a full specification of the exact target architecture. You must inspect the repo first, infer the current structure and workflow, and then make conservative, well-justified changes.

The priority is to **improve structure, reproducibility, and portfolio quality without changing the core modeling behavior**.

---

## Project objective

This repository implements an ETF allocation research pipeline centered on Bayesian / bandit-style portfolio construction.

At a high level, the repo appears to involve some combination of:
- building a monthly ETF dataset from market and macro inputs
- constructing ETF metadata / tags
- running non-contextual model variants or baselines
- running a final contextual model
- producing walk-forward backtests, metrics, and artifacts

Your job is to reorganize the repository from a **notebook-first research repo** into a **clean hybrid research/software repo** that:
- remains easy to understand for a reviewer
- remains faithful to the current methodology
- is easier to run and maintain
- presents better as a portfolio project for ML / quant / applied research roles

This is a **refactor and packaging task**, not a methodological redesign.

---

## Core principles

1. **Preserve behavior over elegance**
   - Do not rewrite logic just because a different architecture looks cleaner.
   - Favor conservative extraction and cleanup.

2. **Inspect before changing**
   - Before any major refactor, inspect the repo and summarize:
     - current directory structure
     - current pipeline stages
     - where reusable logic seems to live
     - which notebooks are exploratory versus operational
     - which code paths are timing-sensitive or high-risk

3. **Prefer faithful extraction**
   - Move clearly reusable logic into Python modules.
   - Keep exploratory, ambiguous, or highly presentation-oriented material in notebooks unless there is a strong reason to extract it.

4. **Do not invent methodology**
   - If something is unclear, preserve the current behavior and document uncertainty.
   - Do not infer a “better” model definition without evidence from the existing code.

5. **Keep the notebook story**
   - This should remain a hybrid repo, not a pure package.
   - Notebooks can remain part of the workflow, but should become thinner and more readable where possible.

---

## Non-negotiable constraints

You must preserve the following unless you find a clear bug and explicitly call it out.

### 1. No-look-ahead integrity
- Do not change the timing logic of the strategy.
- Decisions made at time `t` must not accidentally use information from `t+1` or later.
- Any refactor touching lagging, alignment, merging, state updates, or backtest loops is high risk.

### 2. Modeling semantics
- Do not silently change:
  - formulas
  - priors
  - score definitions
  - thresholds
  - weighting semantics
  - cold-start behavior
  - hyperparameter meanings
  - portfolio construction rules

### 3. Economic interpretation
- Do not “clean up” code in a way that changes the effective economic strategy.
- The appearance of cleaner abstractions is less important than preserving what the current repo actually does.

### 4. Existing outputs / claims
- Do not make README or documentation claims that are not supported by the code.
- Do not oversell the project as production trading infrastructure if it is fundamentally a research/backtest repo.

### 5. Notebook preservation
- Do not delete notebooks.
- If code is extracted from notebooks, leave the notebooks usable as analysis / demo layers.

---

## Desired end state

The repo should move toward the following qualities, but you should decide the exact implementation after inspection.

### The final repo should:
- have a clearer top-level structure
- separate reusable code from exploratory notebook content
- make major pipeline stages easy to identify
- have at least one documented run path
- avoid hardcoded paths where practical
- have lightweight validation/tests for the most fragile logic
- have a recruiter- and reviewer-friendly README
- look like a serious, well-organized research/software project rather than a loose notebook dump

### Likely improvements may include:
- a `src/` package or equivalent module area
- a `scripts/` directory or thin CLI entrypoints
- config files for paths and key options
- an `outputs/` location for generated artifacts
- a `tests/` directory with lightweight tests
- notebook cleanup / thinning
- improved `.gitignore`
- better documentation of repo structure and run flow

These are **directions**, not strict requirements. Choose the minimum structure that materially improves the repo.

---

## How to approach the work

Proceed in phases.

### Phase 1: Inspect and summarize
Before making major edits:
- inspect the repo structure
- identify the current workflow
- identify which files are central
- identify reusable logic candidates
- identify risky logic boundaries
- propose a conservative refactor plan tailored to the actual repo

Do not jump straight into a large rewrite.

### Phase 2: Build scaffolding only
If structural cleanup is warranted:
- create missing directories only as needed
- add minimal scaffolding for code organization
- improve `.gitignore`
- avoid moving high-risk logic yet

### Phase 3: Extract clearly reusable logic
Only after inspection:
- move stable, repeated, clearly reusable code out of notebooks
- keep extracted code close to current behavior
- avoid mixing extraction with methodological changes

### Phase 4: Thin notebooks
Once extraction is stable:
- reduce duplicated operational logic in notebooks
- preserve markdown explanation, orchestration, plots, and inspection
- keep notebooks readable for humans

### Phase 5: Add validation
Add lightweight checks/tests around the most fragile areas:
- no-look-ahead timing
- date alignment / lagging
- tagging or metadata rules if applicable
- weight normalization / cash logic if applicable
- smoke tests for a minimal run path where practical

### Phase 6: Improve documentation
Rewrite or refine the README to reflect the actual repo after refactor.

---

## What to decide yourself after inspection

You have more direct repo context than this file does. Use your judgment on:
- exact module boundaries
- exact filenames
- whether a helper belongs in `data`, `evaluation`, `utils`, etc.
- which code should remain notebook-only
- whether packaging via `pyproject.toml` is worth it
- how much of the pipeline should be runnable from scripts
- whether configs should be YAML, TOML, JSON, or minimal Python config

Make these decisions based on the actual repository, not on a generic preferred architecture.

---

## What to surface explicitly before changing

Pause and summarize before proceeding if you encounter any of the following:
- ambiguity about current strategy timing
- uncertainty about whether a variable is lagged or contemporaneous by design
- uncertainty about whether a notebook cell is exploratory or functionally required
- conflicting implementations across notebooks
- suspicious formulas that might be bugs but may also be intentional
- unclear cold-start / prior initialization behavior
- unclear weighting or cash-allocation semantics

When in doubt, preserve the current implementation and document the uncertainty.

---

## Code extraction guidance

When extracting code from notebooks:
- prefer pure functions for deterministic transformations
- isolate side effects like file I/O, plotting, and notebook display logic
- preserve variable meaning unless the rename is obviously safe
- consolidate duplicated logic only when the equivalence is clear
- avoid broad renames that make diffs hard to review
- avoid changing outputs and interfaces unless necessary

Good candidates for extraction usually include:
- data loading and saving
- merge/alignment helpers
- feature engineering utilities
- tagging rule helpers
- model state update functions
- scoring and weighting helpers
- backtest loop helpers
- evaluation metric functions

Poor candidates for forced extraction usually include:
- one-off exploration cells
- presentation-heavy notebook sections
- ambiguous scratch work
- logic whose role is not yet clear

---

## Validation expectations

You should validate consequential changes.

At minimum, add or improve lightweight checks around the highest-risk logic, where relevant:
- time alignment
- lag correctness
- merge key uniqueness
- required columns
- state update consistency
- nonnegative weights / proper normalization
- decisions at `t` applied to returns at `t+1`
- smoke-run path that does not crash

Be honest about what has and has not been validated.

Do not pretend to prove correctness if only a smoke test exists.

---

## Documentation expectations

The README should become easier for an external reviewer to understand quickly.

A good README will usually cover:
- what the project does
- why it matters
- high-level pipeline
- repository structure
- how to run key parts
- what artifacts/results it produces
- technical highlights
- limitations / current status

Keep claims supported by the codebase.

---

## Data handling expectations

Be conservative with data handling.
- Do not commit large datasets.
- Avoid hardcoded personal/local paths where practical.
- Add clear instructions for expected input locations if needed.
- Use placeholders or sample data only if appropriate and safe.
- Update `.gitignore` accordingly.

---

## Output style while working

When making substantial changes:
1. summarize what you found
2. state what you plan to change
3. make the changes
4. report what changed
5. report any risks, assumptions, or unresolved ambiguity
6. report validation performed

Prefer incremental, reviewable progress over giant rewrites.

---

## Success criteria

The work is successful if, after your refactor:
- the repo is materially easier to understand
- reusable logic is better organized
- notebook sprawl is reduced without losing the research workflow
- core modeling behavior is preserved
- no-look-ahead integrity is maintained
- the project presents more strongly to recruiters, collaborators, and reviewers
- the structure reflects the actual repo rather than a generic template

---

## Final reminder

You are expected to use the repository itself as the primary source of truth.

Do not force the repo into a preconceived architecture unless the repo clearly supports it.

When choosing between:
- a cleaner abstraction that might change behavior, and
- a slightly messier structure that preserves behavior,

choose the structure that preserves behavior.