module SimpleOLG

using Chain
using DataFrameMacros
using DataFrames
using DimensionalData
using Interpolations: linear_interpolation, Line
using LinearAlgebra: dot, norm
using Roots: find_zero

const DD = DimensionalData

include("types.jl")
include("demographics.jl")
include("parameters.jl")
include("entrepreneur_types.jl")
include("entrepreneur_profit.jl")
include("budget.jl")
include("production.jl")
include("vfi.jl")
include("entrepreneur_vfi.jl")
include("egm.jl")
include("entrepreneur_egm.jl")
include("aggregation.jl")
include("entrepreneur_simulation.jl")
include("entrepreneur_bequests.jl")
include("lifecycle_bequests.jl")
include("worker_bequests.jl")
include("equilibrium.jl")

export SolutionMethod, VFI, EGM
export PermanentType, LaborProductivity, EntrepreneurSkill, CombinedProductivity
export TaxRegime, NoTax, CapitalIncomeTax, WealthTax, EntrepreneurParams
export BequestParams
export p_surv₀, PPT, pmf, mortality, get_par, income_profile, simple_income_profile
export c, a_next, c_curr, c_prev, a_prev, a_prev_c_prev
export interest_rate, wage, output, inverse_interest_rate
export iterate_back!, solve_backward, solve_forward, solve_backward_forward, _solve_backward_forward_
export efficient_entrepreneur_capital, entrepreneur_capital, entrepreneur_profit, after_tax_wealth, disposable_resources
export solve_entrepreneur_lifecycle, simulate_entrepreneur_types, compare_tax_regimes
export compute_bequests, distribute_inheritances, inheritance_income
export solve_entrepreneur_with_bequests, compare_tax_regimes_with_bequests
export solve_worker_with_bequests, solve_lifecycle_with_bequests, aggregate_wealth
export aggregate, criterion, partial_equilibrium, general_equilibrium

end
