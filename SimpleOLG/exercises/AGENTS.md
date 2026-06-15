# Exercise Authoring Guidelines

## Overview

This directory contains `.texw` (Weave.jl Noweb format) exercise files for the SimpleOLG package. Each file is a single source that produces both executable Julia and a LaTeX fragment for the `exam` document class. Follow these patterns exactly.

## Exercise Structure (Three Layers)

Every exercise must follow this structure:

1. **Model specification (given).** State the optimization problem formally. The student does not construct this — it is provided. Ask the student to *explain* parts of it (e.g., "What is the role of θ?").

2. **Derivation (the task).** FOCs, Euler equations, closed-form solutions — this is the intellectual work. Never reveal the approach or the answer in the problem statement.

3. **Interpretation with numerical results (given).** Provide precomputed values and ask the student to interpret them economically. The numbers come from the package via `j` interpolation.

### What counts as a task vs. what is given

- **Task:** "Derive the profit-maximizing capital stock k*", "Show that mortality cancels in the Euler equation"
- **Given:** "For the parameters above, k* = 4 and π = 2. Interpret these results."
- **Never a task:** "Plug in α = 0.5 and show that k* = 4" — this reveals the formula (the derivation) and reduces the exercise to arithmetic.

## LaTeX Patterns

### Exam class structure

Use `\titledquestion{Title}` for each exercise — do NOT include `[total points]` (the `addpoints` class option auto-computes the total from `\part[N]` calls via `\totalpoints`). Use `\part[pts]` for sub-questions, `\begin{solution}[Xcm]...\end{solution}` for model answers. Include point breakdowns at the end of each solution.

The `[Xcm]` optional argument on `\begin{solution}` is the natural answer-box height when the question is rendered as an exam (see `../../exercises/AGENTS.md` → *Assembling an exam*). `collection.tex` ignores it; `exam.tex` uses it as a `solutionorbox` argument that stretches proportionally to fill page slack. Proxy sizing: `max(3, ceil(words/8) + display_eqs)` cm. Omit to get a 5 cm default.

### Optimization problems

Always use `align*` with nested `aligned` for constraints:

```latex
\begin{align*}
    &\max_{c, k} u(c) \\
    &\begin{aligned}
        \text{s.t. } & c = (1+r)a + zk^\alpha - (r+\delta)k - f, \\
        & k \leq \theta a, \\
        & k \geq 0.
    \end{aligned}
\end{align*}
```

Do **not** use `\begin{cases}`, plain `align*` without nesting, or any other format for constrained optimization.

### Do not redefine exam class commands

Never redefine `\titledquestion`, `\part`, or other `exam` class internals. The `collection.tex` wrapper uses the class as-is.

## Weave.jl Patterns

### File format

Each `.texw` file has:
1. A hidden Julia preamble that loads the package and computes values
2. A LaTeX body that is **native LaTeX** — not wrapped in any Julia string
3. Inline `j expr` interpolation for computed numerical values

### Hidden Julia chunks

```
<<echo=false; results="hidden">>=
using SimpleOLG

fmt(x; digits = 2) = round(x; digits)

# ... compute values here ...
@
```

### Inline interpolation

Use `` `j fmt(value)` `` to insert computed values into LaTeX. Example:

```latex
the optimal capital stock is $k^* = `j fmt(k_star)`$ and profits are $\pi = `j fmt(pi_star)`$.
```

### Anti-patterns — do NOT do any of these

- **`print(raw"""...""")`**: Never put the exercise body inside a Julia print statement. The LaTeX body is native in `.texw` files.
- **`print(repr(...))`**: Never use repr-based preprocessing to escape LaTeX. If Weave chokes on a specific command, fix that command.
- **`\newcommand` for values**: Never define `\newcommand{\kstar}{4}` and use `\kstar` in the text. Use `j` interpolation directly — that is the whole point of Weave.
- **`preprocess_source`**: Never add build-script preprocessing that wraps LaTeX in Julia code chunks. The build script should be a simple loop calling `weave()`.

## Exercise Content Rules

### Derivation is the task, not arithmetic

The student's job is to derive formulas from first principles (objective → FOCs → solve). Numerical values are scaffolding for the interpretation part only.

- Good: "Derive the first-order condition for capital."
- Bad: "Show that k* = 4." (reveals the formula, reduces task to plugging in)

### Separate derivation from numerical results

Never mix a general derivation with a specific numerical plug-in in the same `\part`. Structure as:
- Part (a)–(c): derive general formulas
- Part (d): "Take the following results as given: ... Interpret."

### Scaffolding text

Use `\uplevel{...}` for text between parts that provides numerical values or sets up the next sub-question (e.g. "Take the following results as given..."). Do not use bare text outside a `\part` without `\uplevel`.

### Numerical result tables

When presenting three or more numerical values for interpretation, use a `booktabs` table (`\usepackage{booktabs}` is loaded by `collection.tex`) instead of inline prose. See `entrepreneur-1period.texw` part (e) for the pattern.

### Lagrangian approach

Use a single multiplier $\lambda$ on the lifetime budget constraint wherever possible. Per-period multipliers ($\lambda_j$, $\mu$) are only appropriate when the return is nonlinear or occupation-dependent and a standard lifetime BC cannot be written (e.g. entrepreneur exercises).

### Specify given vs. chosen

The problem statement should make clear what is endogenous (chosen) and what is exogenous (given/parameters). When part (a) asks the student to *explain* the mathematical formulation (e.g. "What are the choices? What is θ?"), do **not** spell out the answer in the problem statement — let the student identify endogenous vs. exogenous objects themselves (see `entrepreneur-1period.texw`). When part (a) is a derivation rather than an explanation, state the distinction up front: "The household chooses c and k. The endowments a and z, as well as the interest rate r, are taken as given."

### No code language in exam questions

Never write "the simulated policy chooses E" or "the constraint flag is true." Use economic language: "the household becomes an entrepreneur" or "the collateral constraint binds."

### Point breakdowns

Every `\begin{solution}` block must end with a parenthetical point breakdown, e.g.:

```latex
(1 point for the FOC, 1 point for solving for k*, 1 point for interpreting the result.)
```

### Cross-validation with SimpleOLG

Every `.texw` file should `using SimpleOLG` and compute its numerical values from the package. This ensures the exercise stays consistent with the solver. If a value can be computed by the package, compute it — do not hardcode.

## Gold Standard

See `entrepreneur-1period.texw` as the reference implementation. It demonstrates all patterns above.
