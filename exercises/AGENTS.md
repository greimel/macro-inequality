# Exercises directory — agent instructions

## Directory purpose

`exercises/` holds the master exercise collection.

- `collection.tex` — the master document. Inputs hand-authored questions from `basics/` and `comparisons/` plus generated questions from `../SimpleOLG/exercises/generated/`.
- `build.sh` — produces `collection-exercises.pdf` and `collection-solutions.pdf` (cleans aux files on success). Exam PDFs are built separately in the private `macro-inequality-private/exams` repo.
- `basics/`, `comparisons/` — hand-authored `.tex` questions that live in this directory.
- `index.md` — catalogue of every active question (one row per `\titledquestion`). See "Maintaining `index.md`" below.

SimpleOLG-package questions live in `../SimpleOLG/exercises/*.texw` and are weaved to `../SimpleOLG/exercises/generated/*.tex` by `julia SimpleOLG/exercises/build_exercises.jl`. `collection.tex` inputs the generated `.tex`, but the editable source is the `.texw`.

## Pointers

- `.texw` authoring conventions (Weave.jl format, Julia preamble, `j fmt` interpolation, anti-patterns) → `../SimpleOLG/exercises/AGENTS.md`.
- Hand-authored `.tex` style and how to register a new question in `collection.tex` → TODO (to be drafted in a follow-up).

## Assembling an exam

Exam assembly — `exam.tex`, the `final-exam-collection`, answer-box sizing, and the
`build.sh` that produces the exam PDFs — lives in the **private** repo
`macro-inequality-private/exams/`, which reuses these exercise sources by `\input`
over a sibling path (`../../macro-inequality/exercises/...`). See that repo's
`AGENTS.md` for the box-sizing rule and how to put a question on an exam.

The only obligation on exercises in *this* repo is the `\begin{solution}[Xcm]`
height convention (see Style conventions below): the height is ignored in the
collection (`\cancelspacetrue`) but becomes the student answer-box height in the
exam.

## Style conventions

- **`\titledquestion{Title}`** — do NOT add `[N]`. Points are auto-computed from `\part[N]` via the `addpoints` class option and `\totalpoints` in the qformat. Exception: questions without any `\part` (e.g. `J-period-bc.tex`) may carry `[N]` on the title.
- **`\begin{solution}[Xcm]`** — include the height argument. In exam mode it sizes the student answer box; in collection mode `\printanswers` shows the full solution and `\cancelspacetrue` ignores the height.
- **Lagrangian approach.** Use a single multiplier $\lambda$ on the lifetime budget constraint. Per-period multipliers ($\lambda_j$, $\mu$) only when the return is nonlinear or occupation-dependent (e.g. entrepreneur).
- **`\uplevel{...}`** for scaffolding text between parts ("Take the following results as given ...").
- **Numerical result tables.** When presenting $\geq 3$ values for interpretation, use a `booktabs` table instead of inline prose. `\usepackage{booktabs}` is loaded by `collection.tex`. See `entrepreneur-1period.texw` part (e) for the pattern.

## Maintaining `index.md`

### Purpose

`index.md` catalogues all 16 questions (active and deferred). The catalogue makes it easy to evaluate topic coverage and to assemble exams. Exercises that are temporarily commented out in `collection.tex` are flagged in the "Deferred exercises" section at the top of `index.md`.

### When to update

- A question is added, removed, or moved in `collection.tex` → sync the inventory and the table here in the same commit.
- A question's `\part[N]` points change → update the `Pts` cell.
- A question's `.texw` preamble changes such that `Numbers` flips between `Pkg-interpolated` / `none` / `hardcoded` → update the `Numbers` cell.
- A new vocabulary tag is needed (e.g. a new `Features` tag) → extend the relevant list in this file *first*, then use it in `index.md`.

### Auto-extraction snippets

Run from the repo root.

```bash
# Pts = sum of \part[N] for one file
grep -oE '\\part\[[0-9]+\]' <file> | grep -oE '[0-9]+' | awk '{s+=$1} END {print s+0}'
# Numbers indicator: count Pkg interpolations (>0 → Pkg-interpolated)
grep -c '`j fmt' <file>
# All 16 catalogued questions at once: see helper at the bottom of this file.
```

If a hand-authored `.tex` has no `\part[N]` (single-derivation question), check whether `\titledquestion{...}[N]` carries the points (e.g. `J-period-bc.tex`). Add that to the `Pts` total for that question.

### Column reference (canonical schema)

| Column | Source | Filled how |
|---|---|---|
| **#** | counter from `collection.tex` order | auto |
| **Title** | `\titledquestion{...}` | auto |
| **Section** | preceding `\section{...}` in `collection.tex` | auto |
| **Pts** | sum of `\part[N]` (plus `\titledquestion[N]` if no parts) | auto |
| **Parts** | count + per-part type tags, e.g. `4 (D D I C)` | manual |
| **Diff** | 1 (warm-up) / 2 (standard) / 3 (challenging) | manual; mark ambiguous as `2?` |
| **Model** | structural setting of the economy | manual |
| **Features** | comma-separated economic ingredients | manual |
| **Method** | tools the student must use | manual |
| **Numbers** | source of any numerical values | auto from `grep '`j fmt'` |
| **Standalone** | exam-readiness flag | manual |
| **Reduce to** | smaller variant if a foundation result is given | manual |
| **Notes** | short free text | manual |

### Vocabularies

**Type** (used in `Parts`, single letter per part):

- `D` derivation (FOCs, Euler, BC manipulations, lifecycle path)
- `P` proof (sign, identity, monotonicity)
- `I` interpretation (verbal economic reasoning, role of a parameter)
- `C` computation (plug numbers in, evaluate)
- `A` algorithm (describe a procedure)

**Model**: `2-period HH` · `J-period HH` · `2-period OLG` · `general OLG` · `multi-type OLG` · `1-period entrepreneur` · `2-period entrepreneur`.

**Features**: `mortality`, `annuities`, `accidental-bequests`, `warm-glow`, `bequest floor`, `inheritance redistribution`, `multi-type`, `entrepreneur`, `collateral constraint`, `occupation choice`, `general equilibrium`, `lifetime BC`, `Bellman/recursive`, `EGM`, `VFI`, `CRRA`. Multiple tags per cell.

**Method**: `Lagrangian (sequence)` · `Lagrangian (per-period)` · `Bellman` · `closed-form algebra` · `matrix algebra` · `fixed-point iteration` · `verbal only`.

**Numbers**: `none` · `Pkg-interpolated` · `hardcoded` · `mix`. `Pkg-interpolated` flags any `` `j fmt(...)` `` use; the question needs `julia SimpleOLG/exercises/build_exercises.jl` before reuse.

**Standalone**: `yes` (self-contained) · `cites Q<N>` (quotes a result from another question) · `Pkg-numbers` (works alone but contains `j fmt` values).

### Filling subjective columns

- Read the `.texw` body, **not** the `generated/.tex`, so you see Julia parameters and `j fmt(...)` calls.
- Pick the most prominent intent for each part; if two type tags fit equally well, write the one closer to the student's main task (e.g. `D` over `C` if substitution is the point, not the arithmetic).
- Mark genuinely ambiguous calls with a trailing `?` (e.g. `D?`) and add a `TODO:` line at the bottom of `index.md` referencing the question number.

### One-shot Pts/Numbers extraction for the full set

```bash
python3 - <<'PY'
import re
from pathlib import Path
order = [
    "exercises/basics/EIS-proof.tex",
    "exercises/basics/EIS-compute.tex",
    "SimpleOLG/exercises/mortality.texw",
    "SimpleOLG/exercises/annuities.texw",
    "SimpleOLG/exercises/bequests-warmglow-focs.texw",
    "SimpleOLG/exercises/bequests-warmglow-solve.texw",
    "SimpleOLG/exercises/entrepreneur-1period.texw",
    "SimpleOLG/exercises/entrepreneur-2period.texw",
    "SimpleOLG/exercises/bequests-accidental.texw",
    "SimpleOLG/exercises/bequests-aggregation.texw",
    "SimpleOLG/exercises/bequests-equilibrium.texw",
    "SimpleOLG/exercises/bequests-types.texw",
    "SimpleOLG/exercises/olg-algorithm.texw",
    "exercises/basics/J-period-bc.tex",
    "exercises/basics/J-period-model.tex",
    "SimpleOLG/exercises/dynamic-programming.texw",
]
part_re  = re.compile(r"\\part\[(\d+)\]")
title_re = re.compile(r"\\titledquestion\{([^}]+)\}(?:\[(\d+)\])?")
for i, f in enumerate(order, 1):
    s = Path(f).read_text()
    parts = [int(x) for x in part_re.findall(s)]
    m = title_re.search(s)
    title_pts = int(m.group(2)) if (m and m.group(2)) else 0
    pts = sum(parts) + (title_pts if not parts else 0)
    n_jfmt = s.count('`j fmt')
    print(f"Q{i:>2}  pts={pts:>3}  parts={len(parts)}  jfmt={n_jfmt:>2}  {f}")
PY
```
