# Repository Guidelines

## Project Structure & Module Organization
This repository contains computational teaching material for an MSc-level elective in Macroeconomics and Inequality taught at the University of Vienna. The course website source lives under `src/`, where topic folders such as `src/preliminaries/`, `src/lifecycle/`, `src/housing/`, `src/continuous-time/`, `src/long-run/`, and `src/tutorials/` hold Pluto notebooks saved as `.jl` files. Top-level pages like `src/index.md`, `src/installation.md`, `src/syllabus.md`, and `src/cheatsheets.md` provide site content. Shared templates live in `src/_includes/`, while site CSS and JavaScript live in `src/assets/`; public data and logos used by notebooks live in `src/assets_pub/`. Treat `src/unfinished/` as draft material.

The `SimpleOLG/` directory is a standalone Julia package that backs the finite-horizon OLG exercises. Its source lives in `SimpleOLG/src/`, tests in `SimpleOLG/test/`, and Weave.jl `.texw` exercise sources in `SimpleOLG/exercises/`. Run the package test suite with `julia --project=SimpleOLG -e 'using Pkg; Pkg.test()'` and regenerate exercises with `julia SimpleOLG/exercises/build_exercises.jl`. See `SimpleOLG/AGENTS.md` for package-specific guidelines and `SimpleOLG/exercises/AGENTS.md` for exercise authoring patterns.

The `exercises/` directory contains the master exercise collection (`collection.tex`), an exam template (`exam.tex`), a bash build script (`build.sh`), and hand-authored question files in `exercises/basics/` and `exercises/comparisons/`. The build script compiles four PDFs (collection and exam, each with and without solutions). See `exercises/AGENTS.md` for directory-level instructions and `exercises/index.md` for the question catalogue.

## Build, Test, and Development Commands
Use Julia and Pluto for notebook work:

- `julia` starts the Julia REPL.
- `using Pluto; Pluto.run()` launches the notebook server locally.
- `include("src/lifecycle/dynamic-programming.jl")` runs a notebook as a Julia script when you want a quick non-interactive check.

There is no central build script in this directory. When editing site pages or templates, keep changes limited to the relevant Markdown, `src/_includes/`, or `src/assets/` files.

## Coding Style & Naming Conventions
Follow existing notebook style: 4-space indentation in Julia code, descriptive lowercase file names with hyphens (for example `simple-income-risk.jl`), and section-oriented notebook prose.

**Pluto file format.** Each cell is delimited by `# ╔═╡ <uuid>` on its own line, followed by the cell body. Markdown cells use `md"""..."""`. The cell-order block at the end of the file (`# ╔═╡ Cell order:`) lists every cell UUID; add new cell UUIDs there in the intended visual position. Generate a fresh UUID with `import UUIDs; string(UUIDs.uuid4())`. Do not remove or reorder existing `# ╔═╡` markers without updating the cell-order block. Do not strip `@bind` or `PlutoUI` expressions.

The forbidden environment block (`PLUTO_PROJECT_TOML_CONTENTS = """` through the closing `PLUTO_MANIFEST_TOML_CONTENTS = """`) is always at the bottom of the file; skip past it entirely unless the task explicitly changes dependencies. Some notebooks intentionally use `#! format: off` around fragile blocks; do not reformat those sections casually.

## Testing Guidelines
Testing is notebook-local rather than centralized. Several notebooks use `PlutoTest` and inline `@test` assertions, especially in `src/lifecycle/`, `src/continuous-time/`, and `src/unfinished/`. When changing model logic, rerun the affected notebook in Pluto or execute it with `include(...)` and confirm the embedded tests still pass. Add new tests close to the functions or exercises they validate.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `add simple-entrepreneurs.jl (rough draft)` and `revamp assignment 3 (incl EGM)`. Keep commit messages concise, lowercase when appropriate, and focused on one change. Pull requests should state which notebook or page changed, summarize any modeling or content impact, and include screenshots only for visible site or asset updates.

When an AI agent authors or co-authors a commit, add a `Co-Authored-By` trailer with the agent's identity. For Claude, include the model family and version if known, for example:

```
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

## Contributor Notes
Prefer small, targeted edits. This material is student-facing, so changes should prioritize clarity, reproducibility, and stable teaching flow over clever abstractions. Avoid moving files unless the sidebar or internal links are updated accordingly, and keep external data fetches reproducible when touching notebooks that download source data.
