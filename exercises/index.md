# Exercise index

See [`AGENTS.md`](AGENTS.md) for column definitions, vocabularies, and how to maintain this file. Snapshot of `collection.tex` on 2026-04-17: 16 catalogued questions across 4 sections, 215 total points; 5 of them are temporarily commented out in `collection.tex` for the initial student release (see "Deferred exercises" below).

## Deferred exercises

The following five questions are sound in principle but need polish and consistency checks against the lecture notes before the final exam or next year's version. They are commented out in `collection.tex` and will not appear in the PDF the students receive, but the catalogue and tables below are unchanged so the work can be re-enabled with a one-line uncomment.

- Q8 — Two-Period Entrepreneur (`entrepreneur-2period.texw`)
- Q9 — Accidental Bequests (`bequests-accidental.texw`)
- Q10 — OLG Aggregation: Bequests and Inheritances (`bequests-aggregation.texw`)
- Q11 — Stationary Equilibrium with Warm-Glow Bequests: Single Type (`bequests-equilibrium.texw`)
- Q12 — Stationary Equilibrium with Two Permanent Types (`bequests-types.texw`)

With these deferred, the active student release contains 11 questions / 140 points: §1 (26 pts) + §2 (67 pts, Q3–Q7) + §3 (13 pts, Q13 only) + §4 (34 pts).

## Inventory

| # | Section | File |
|---|---|---|
| 1 | Elasticity of Intertemporal Substitution | [`exercises/basics/EIS-proof.tex`](basics/EIS-proof.tex) |
| 2 | Elasticity of Intertemporal Substitution | [`exercises/basics/EIS-compute.tex`](basics/EIS-compute.tex) |
| 3 | Household Decision Problems | [`SimpleOLG/exercises/mortality.texw`](../SimpleOLG/exercises/mortality.texw) |
| 4 | Household Decision Problems | [`SimpleOLG/exercises/annuities.texw`](../SimpleOLG/exercises/annuities.texw) |
| 5 | Household Decision Problems | [`SimpleOLG/exercises/bequests-warmglow-focs.texw`](../SimpleOLG/exercises/bequests-warmglow-focs.texw) |
| 6 | Household Decision Problems | [`SimpleOLG/exercises/bequests-warmglow-solve.texw`](../SimpleOLG/exercises/bequests-warmglow-solve.texw) |
| 7 | Household Decision Problems | [`SimpleOLG/exercises/entrepreneur-1period.texw`](../SimpleOLG/exercises/entrepreneur-1period.texw) |
| 8 | Household Decision Problems | [`SimpleOLG/exercises/entrepreneur-2period.texw`](../SimpleOLG/exercises/entrepreneur-2period.texw) |
| 9 | OLG Aggregation and Stationary Equilibrium | [`SimpleOLG/exercises/bequests-accidental.texw`](../SimpleOLG/exercises/bequests-accidental.texw) |
| 10 | OLG Aggregation and Stationary Equilibrium | [`SimpleOLG/exercises/bequests-aggregation.texw`](../SimpleOLG/exercises/bequests-aggregation.texw) |
| 11 | OLG Aggregation and Stationary Equilibrium | [`SimpleOLG/exercises/bequests-equilibrium.texw`](../SimpleOLG/exercises/bequests-equilibrium.texw) |
| 12 | OLG Aggregation and Stationary Equilibrium | [`SimpleOLG/exercises/bequests-types.texw`](../SimpleOLG/exercises/bequests-types.texw) |
| 13 | OLG Aggregation and Stationary Equilibrium | [`SimpleOLG/exercises/olg-algorithm.texw`](../SimpleOLG/exercises/olg-algorithm.texw) |
| 14 | $J$-Period Lifecycle Problem | [`exercises/basics/J-period-bc.tex`](basics/J-period-bc.tex) |
| 15 | $J$-Period Lifecycle Problem | [`exercises/basics/J-period-model.tex`](basics/J-period-model.tex) |
| 16 | $J$-Period Lifecycle Problem | [`SimpleOLG/exercises/dynamic-programming.texw`](../SimpleOLG/exercises/dynamic-programming.texw) |

## Index

### §1 Elasticity of Intertemporal Substitution (2 questions, 26 pts)

| # | Title | Pts | Parts | Diff | Model | Features | Method | Numbers | Standalone | Reduce to | Notes |
|---|---|---:|---|:-:|---|---|---|---|---|---|---|
| 1 | EIS: Derivation | 18 | 6 (D C D D I P) | 2 | J-period HH | lifetime BC, CRRA-free | closed-form algebra | none | yes | n/a (chained derivation) | Linear approx of $\ln u'$ → EIS expression; the only place a `P` part appears in the section. |
| 2 | EIS: Computation | 8 | 3 (I D I) | 1 | none (utility only) | CRRA | closed-form algebra | none | yes | n/a | Companion to Q1; assumes Euler eq is known. |

### §2 Household Decision Problems (6 questions, 83 pts)

| # | Title | Pts | Parts | Diff | Model | Features | Method | Numbers | Standalone | Reduce to | Notes |
|---|---|---:|---|:-:|---|---|---|---|---|---|---|
| 3 | Mortality Risk without Annuities | 10 | 3 (I D I) | 2 | 2-period HH | mortality, CRRA | Lagrangian (lifetime BC) | Pkg-interpolated | Pkg-numbers | ~6 pts: give the Euler, ask only role of $m$ + numerical comparison | Companion to Q4. Each question states the other's allocation in part (c); must not appear together. |
| 4 | Actuarially Fair Annuities | 12 | 3 (I D I) | 2 | 2-period HH | mortality, annuities, CRRA | Lagrangian (lifetime BC) | Pkg-interpolated | Pkg-numbers | ~7 pts: give the Euler, ask only pooling argument + numerical comparison | Companion to Q3. The annuity-return derivation in (a) carries 4 pts. |
| 5 | Warm-Glow Bequests: FOCs | 12 | 5 (I D D D I) | 2 | 2-period HH | warm-glow, bequest floor, lifetime BC, CRRA | Lagrangian (lifetime BC) | none | yes | ~5 pts: give the FOCs, ask only $c_1^*(c_0^*), a_2^*(c_0^*)$ + role of $\nu$ | Foundation for Q6. |
| 6 | Warm-Glow Bequests: Solve | 20 | 7 (I D I D P D I) | 3 | 2-period HH | warm-glow, bequest floor, lifetime BC, CRRA | closed-form algebra | Pkg-interpolated | Pkg-numbers | n/a (already starts from FOCs given) | Heaviest algebra in the collection; homotheticity discussion + special case. Self-contained (states FOCs inline). |
| 7 | One-Period Entrepreneur | 13 | 5 (I D I D I) | 2 | 1-period entrepreneur | entrepreneur, collateral constraint, occupation choice | closed-form algebra | Pkg-interpolated | Pkg-numbers | ~6 pts: give $k^*, \pi(z,k^*)$, ask only occupation choice + collateral effect | Profit $\pi(z,k)$ bundles the fixed cost $f$. Utility functional form left unspecified — only monotonicity is used. |
| 8 | Two-Period Entrepreneur | 16 | 6 (I D D D I I) | 2 | 2-period entrepreneur | entrepreneur, collateral constraint, occupation choice, mortality, CRRA | Lagrangian (per-period) | Pkg-interpolated | Pkg-numbers | ~8 pts: give $R_E$ and Euler, ask only constrained Euler + interpretation | Adds savings dimension on top of Q7. |

### §3 OLG Aggregation and Stationary Equilibrium (5 questions, 72 pts)

| # | Title | Pts | Parts | Diff | Model | Features | Method | Numbers | Standalone | Reduce to | Notes |
|---|---|---:|---|:-:|---|---|---|---|---|---|---|
| 9 | Accidental Bequests | 12 | 5 (I D D D I) | 2 | 2-period OLG | mortality, accidental-bequests, inheritance redistribution, CRRA | Lagrangian (per-period) + algebra | Pkg-interpolated | Pkg-numbers | ~7 pts: give $c_0, c_1, a_1$, ask only $\pi$ + $B$ + $F$ interpretation | No annuities; mortality stays in the Euler. Shares the no-annuity Euler with Q3. |
| 10 | OLG Aggregation: Bequests and Inheritances | 15 | 5 (D D D D I) | 2 | multi-type OLG | warm-glow, multi-type, inheritance redistribution | matrix algebra | Pkg-interpolated | Pkg-numbers | ~8 pts: skip multi-type extension (d), keep single-type aggregation | Gateway to multi-type machinery. Parts (a)–(c) single-type; multi-type setup introduced just before (d). |
| 11 | Stationary Equilibrium with Warm-Glow Bequests: Single Type | 14 | 4 (I D I I) | 2 | 2-period OLG | warm-glow, inheritance redistribution, general equilibrium, CRRA | closed-form algebra | Pkg-interpolated | Pkg-numbers | ~7 pts: skip (a), (b); start from $a_2 = ((1+r)y_0+y_1)/D$ | Self-contained equilibrium statement: household optimum given inline at fixed prices, no FOCs to rederive. |
| 12 | Stationary Equilibrium with Two Permanent Types | 18 | 5 (I D D D I) | 3 | multi-type OLG | warm-glow, multi-type, inheritance redistribution, general equilibrium, CRRA | matrix algebra (2×2 coupled) | Pkg-interpolated | Pkg-numbers | ~10 pts: skip (a), (b); start from coupled-system result | Self-contained: per-type optimum and multi-type inheritance aggregation given inline. |
| 13 | The OLG Steady-State Algorithm | 13 | 6 (D C I C C A) | 1 | 2-period OLG, general equilibrium | general equilibrium, fixed-point iteration | fixed-point iteration walkthrough | Pkg-interpolated | Pkg-numbers | ~5 pts: skip (a), (c); just (d), (e), (f) on aggregation + update | Pedagogical walk-through of one iteration of the GE algorithm. |

### §4 $J$-Period Lifecycle Problem (3 questions, 34 pts)

| # | Title | Pts | Parts | Diff | Model | Features | Method | Numbers | Standalone | Reduce to | Notes |
|---|---|---:|---|:-:|---|---|---|---|---|---|---|
| 14 | Lifetime budget constraint | 7 | single derivation (D) | 2 | J-period HH | lifetime BC | closed-form algebra (telescoping) | none | yes | n/a (already minimal) | Hand-authored. Points carried by `\titledquestion{...}[7]`, no `\part`. Uses concrete $J = 4$ example. |
| 15 | $J$-period problem: Derivation | 14 | 6 (D D D D D I) | 2 | J-period HH | lifetime BC, CRRA | Lagrangian (sequence) | none | yes | ~6 pts: skip (a)–(c), start from "the FOCs imply" | Companion to Q14; sequence approach. |
| 16 | The $J$-Period Problem in Recursive Form | 13 | 6 (I D D D A A) | 2 | J-period HH | Bellman/recursive | Bellman | none | yes | ~5 pts: skip (a)–(d); just the algorithm | Recursive counterpart to Q15. |

## Coverage notes

**Section weights.** EIS 26 pts, HH Decisions 83 pts, OLG Aggregation 72 pts, J-Period 34 pts. Total 215. The two big middle sections dominate; the EIS and J-period framing sections are lighter.

**Type distribution (across the 88 part-tags above).** Derivation `D` ≈ 39, Interpretation `I` ≈ 28, Computation `C` ≈ 7, Algorithm `A` ≈ 4, Proof `P` ≈ 2 (only in Q1 and Q6). The collection is heavily derivation- and interpretation-driven; proofs and pure-algorithm parts are rare.

**Models covered.** 2-period HH (Q3, Q4, Q5, Q6), J-period HH (Q1, Q14, Q15, Q16), 1-period entrepreneur (Q7), 2-period entrepreneur (Q8), 2-period OLG (Q9, Q11, Q13), multi-type OLG (Q10, Q12). General equilibrium appears in Q11, Q12, Q13. EIS as an abstract concept (Q2) sits outside any model.

**Features that recur.** `CRRA` (10/16), `warm-glow` (5/16), `lifetime BC` (4/16), `inheritance redistribution` (5/16), `general equilibrium` (3/16), `mortality` (4/16), `entrepreneur` (2/16). `EGM`, `VFI`, `housing`, `social comparisons` are absent — the EGM section and Social Comparisons section are commented out in `collection.tex`, and the relevant questions live there.

**Exam compilability.** All 16 questions are self-contained (no `cites Q<N>`): every question states any borrowed upstream result inline rather than referring to another exercise. 6 questions are fully standalone with no `j fmt` numbers (Q1, Q2, Q5, Q14, Q15, Q16 — 72 pts). The remaining 10 contain `j fmt` (Pkg-numbers — need a package re-weave before reuse: Q3, Q4, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13).

**Reducibility.** 10 questions can be meaningfully shrunk by giving an upstream result and asking only the downstream parts (see `Reduce to`). This gives flexibility when sizing an exam.

**Gaps to consider.**
- No question on dynamic income risk / income shocks.
- No question on housing or borrowing constraints in a richer-than-CRRA setting.
- Only one part anywhere asks for a `P` proof of a sign-style claim (Q6 (e)).
- The J-Period section ends at the recursive formulation; no question yet uses the recursive setup numerically (would normally be where EGM enters).

## Exam conflicts

Every exercise is self-contained: when an exercise needs a result from another, it states that result inline rather than citing. As a side effect, pairs of exercises can reveal each other's answers, so they must not be combined on a single exam. Each entry below lists the conflicting pair and what the conflict is.

- **Q3 (Mortality no-annuity) ↔ Q4 (Annuities)** — part (c) of each states both consumption allocations inline, and each Euler/role-of-$m$ answer is half of the other question's pedagogical punchline (mortality stays in vs. cancels out).
- **Q3 (Mortality no-annuity) ↔ Q9 (Accidental Bequests)** — Q9 derives the same no-annuity Euler $u'(c_0) = \beta(1-m)(1+r)u'(c_1)$ that Q3 part (b) asks for.
- **Q5 (Warm-Glow FOCs) ↔ Q6 (Warm-Glow Solve)** — Q6 states the FOCs inline that Q5 part (b)–(d) asks the student to derive.
- **Q10 (Aggregation) ↔ Q11 (Single-Type Equilibrium)** — Q11's problem statement states $\pi_0 = \pi_1 = 1/2$, $b = a_2/2$, and $\mathrm{inh}_j = b F_j/\pi_j$ inline; Q10 parts (a)–(c) ask the student to derive all three.
- **Q10 (Aggregation) ↔ Q12 (Two-Type Equilibrium)** — Q12's problem statement states the multi-type inheritance formula $\mathrm{inh}_{z_L} = p b_L + (1-p) b_H$ inline; Q10 part (d) asks the student to derive the general $n$-type version.
- **Q11 (Single-Type Equilibrium) ↔ Q12 (Two-Type Equilibrium)** — Q12 states the per-type optimum $A x_i = w\theta_i \mathcal{Y}_w + K\,\mathrm{inh}_{z_i}$ inline, and part (d) verifies that the single-type specialization collapses to Q11's $a_2 = ((1+r)y_0+y_1)/D$.
- **Q1 (EIS Derivation) ↔ Q2 (EIS Computation)** — Q2 part (a) asks for the EIS definition that Q1 derives; Q2 part (c) uses the closed-form Euler that Q1 closes out.
- **Q14 (Lifetime BC) ↔ Q15 (J-period Derivation)** — Q15 part (a) asks for the lifetime budget constraint that Q14 derives.

## TODOs

- Q2 part (a) "State the definition of EIS" — tagged `I` (interpretation/recall) but a defensible `D` (the student usually derives the closed-form from the limit definition). If the intended interpretation is "state the closed form $-u'/(u'' c)$", `D` is more accurate. Resolve when the question is next reviewed.
- Q6 part (b) "Solve for $c_0^*$" — tagged `D` (algebraic substitution is the work). If the intent is more arithmetic than derivation, `C` would fit. Currently `D`.
- Q10 part (c) "Explain why $F_j$ must be divided by $\pi_j$, and evaluate $\mathrm{inh}_0, \mathrm{inh}_1$" — mixes `D` (derivation of the per-capita normalization) and `C` (numerical evaluation). Tagged `D` because the per-capita normalization is the conceptual point.
- Reduce-to estimates are eyeballed; actual point savings depend on what wording the exam uses to "give" the upstream result. Treat as ±2 pts.
