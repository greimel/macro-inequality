# SimpleOLG Package Notes

## Overview

`SimpleOLG` is a small Julia package for finite-horizon OLG teaching models. The baseline package solves household savings problems with VFI or EGM and aggregates them in partial or general equilibrium. The entrepreneur extension adds occupation choice, collateral-constrained capital, and tax-regime comparisons in partial equilibrium.

## Source Files

- `src/types.jl`: solution-method dispatch types (`VFI`, `EGM`)
- `src/demographics.jl`: mortality profiles and stationary cohort weights
- `src/parameters.jl`: parameter construction and utility closure
- `src/budget.jl`: household budget identities and backward-EGM helpers
- `src/production.jl`: aggregate production, factor prices, and inverse demand for capital
- `src/vfi.jl`: baseline backward induction and forward simulation for household savings
- `src/egm.jl`: endogenous-grid household solver
- `src/aggregation.jl`: aggregation of simulated lifecycle outcomes
- `src/equilibrium.jl`: partial- and general-equilibrium wrappers
- `src/entrepreneur_types.jl`: entrepreneur/tax type hierarchy
- `src/entrepreneur_profit.jl`: entrepreneur capital choice, profits, and after-tax wealth
- `src/entrepreneur_vfi.jl`: lifecycle VFI with worker-versus-entrepreneur occupation choice
- `src/entrepreneur_egm.jl`: endogenous-grid entrepreneur solver with branchwise interpolation and occupation-choice upper envelope
- `src/entrepreneur_simulation.jl`: multi-type simulation and tax-regime comparisons
- `src/entrepreneur_bequests.jl`: bequest accounting, inheritance redistribution, and fixed-point wrappers with intergenerational persistence

## Dependencies

The package relies on `DataFrames`, `DimensionalData`, `Interpolations`, `Roots`, `LinearAlgebra`, `Chain`, and `DataFrameMacros`. Entrepreneur code follows the same stack and does not introduce extra package dependencies.

## Running Tests

Run the package test suite from the repository root with:

```bash
julia --project=SimpleOLG -e 'using Pkg; Pkg.test()'
```

## Exercise Authoring

See `exercises/AGENTS.md` for detailed guidelines on writing `.texw` exercise files (structure, LaTeX patterns, Weave.jl patterns, anti-patterns).

## Design Patterns

- Use `DimArray`/`DimVector` objects for lifecycle state dimensions such as `a`, `j`, and `t`.
- Return compact `NamedTuple`s from solvers instead of custom mutable structs.
- Dispatch on `SolutionMethod` for baseline solver variants and on `TaxRegime` for entrepreneur tax logic.
- Keep forward-simulation outputs as `DataFrame`s so downstream teaching notebooks can inspect or plot them directly.

## API Overview

Baseline public functions include `get_par`, `solve_backward_forward`, `partial_equilibrium`, and `general_equilibrium`.

Entrepreneur public functions include `entrepreneur_capital`, `entrepreneur_profit`, `after_tax_wealth`, `disposable_resources`, `solve_entrepreneur_lifecycle`, `simulate_entrepreneur_types`, and `compare_tax_regimes`.

Bequest public functions include `BequestParams`, `compute_bequests`, `distribute_inheritances`, `inheritance_income`, `solve_entrepreneur_with_bequests`, and `compare_tax_regimes_with_bequests`.
