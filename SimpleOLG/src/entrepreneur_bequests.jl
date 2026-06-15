Base.@kwdef struct BequestParams{TP <: AbstractMatrix{<:Real}, TV <: AbstractVector{<:Real}, TF <: AbstractVector{<:Real}}
    P_z::TP
    π_z::TV
    F::TF
end

function _bequest_setup(bpar::BequestParams, par, n_types)
    P_z = Float64.(collect(bpar.P_z))
    π_z = Float64.(collect(bpar.π_z))
    F = Float64.(collect(bpar.F))

    size(P_z) == (n_types, n_types) || throw(DimensionMismatch("P_z must be $n_types × $n_types"))
    length(π_z) == n_types || throw(DimensionMismatch("π_z must have length $n_types"))
    length(F) == par.J + 1 || throw(DimensionMismatch("F must have length $(par.J + 1)"))

    all(P_z .>= -1e-12) || throw(ArgumentError("P_z must have nonnegative entries"))
    all(π_z .> 0) || throw(ArgumentError("π_z must have strictly positive entries"))
    all(F .>= -1e-12) || throw(ArgumentError("F must have nonnegative entries"))

    π_z ./= sum(π_z)

    row_sums = vec(sum(P_z, dims = 2))
    maximum(abs.(row_sums .- 1.0)) <= 1e-10 || throw(ArgumentError("Each row of P_z must sum to one"))

    stationary = vec(transpose(π_z) * P_z)
    maximum(abs.(stationary .- π_z)) <= 1e-8 || throw(ArgumentError("π_z must be a stationary distribution of P_z"))

    F_mass = sum(F)
    if !(isapprox(F_mass, 0.0; atol = 1e-12, rtol = 0.0) || isapprox(F_mass, 1.0; atol = 1e-8, rtol = 1e-8))
        throw(ArgumentError("F must sum to either zero (disable inheritances) or one"))
    end

    if F_mass > 0
        F ./= F_mass
    end

    return (;
        P_z,
        π_z,
        F,
        F_mass,
        π_j = Float64.(collect(pmf(par.m))),
        type_dim = Dim{:type}(1:n_types),
        j_dim = DD.dims(par.m, :j),
    )
end

_has_inheritances(setup) = setup.F_mass > 1e-12

function _inheritances_by_type(bequests_z, setup)
    !_has_inheritances(setup) && return zeros(Float64, length(bequests_z))
    return vec(transpose(setup.P_z) * (bequests_z .* setup.π_z)) ./ setup.π_z
end

function _inheritance_schedule(inheritances_z, setup)
    age_profile = _has_inheritances(setup) ? setup.F ./ setup.π_j : zeros(Float64, length(setup.j_dim))
    schedule = reshape(Float64.(collect(inheritances_z)), :, 1) .* reshape(age_profile, 1, :)
    return DimArray(schedule, (setup.type_dim, setup.j_dim), name = :inheritance)
end

function compute_bequests(sim_df::DataFrame, par)
    expected_j = collect(0:par.J)
    # Average accidental bequests per capita using the stationary age distribution.
    bequest_weights = Float64.(collect(pmf(par.m) .* par.m))
    has_types = hasproperty(sim_df, :type)
    type_ids = has_types ? sort(unique(sim_df.type)) : [1]
    bequests_z = zeros(Float64, length(type_ids))

    for (i, type) in enumerate(type_ids)
        sim_df_type = has_types ? sim_df[sim_df.type .== type, :] : sim_df
        j_order = sortperm(sim_df_type.j)
        j_sorted = sim_df_type.j[j_order]

        (length(j_sorted) == length(expected_j) && all(j_sorted .== expected_j)) ||
            throw(ArgumentError("sim_df must contain exactly one observation per age and type"))

        bequests_z[i] = dot(bequest_weights, Float64.(sim_df_type.a_next[j_order]))
    end

    return bequests_z
end

function distribute_inheritances(B_z, bpar::BequestParams, par)
    bequests_z = Float64.(collect(B_z))
    setup = _bequest_setup(bpar, par, length(bequests_z))
    inheritances_z = _inheritances_by_type(bequests_z, setup)
    return _inheritance_schedule(inheritances_z, setup)
end

inheritance_income(z_index, j, inh_zj) = Float64(inh_zj[type = At(z_index), j = At(j)])

"""
    solve_entrepreneur_with_bequests(solver, par, epar, bpar, a_grid; z_grid, r, w, tax, ...)

Back-compat wrapper over `solve_lifecycle_with_bequests`. Each `z ∈ z_grid`
maps to `EntrepreneurSkill(z)` (i.e. `z_labor = 1.0, z_entrepreneur = z`),
preserving the pre-refactor convention that the scalar `z` scales only
entrepreneurial productivity. Returns a `Dict{z => sol}` of per-type inner
solutions alongside the aggregated fixed point.
"""
function solve_entrepreneur_with_bequests(
    solver::SolutionMethod,
    par,
    epar::EntrepreneurParams,
    bpar::BequestParams,
    a_grid;
    z_grid,
    r,
    w,
    tax = NoTax(),
    a_init = first(a_grid),
    maxiter = 200,
    tol = 1e-10,
    λ_inherit = 1.0,
)
    type_grid = PermanentType[EntrepreneurSkill(z) for z in z_grid]
    out = solve_lifecycle_with_bequests(
        solver, par, bpar, a_grid;
        type_grid, epar, r, w, tax, a_init,
        maxiter, tol, λ_inherit,
    )

    zs = collect(z_grid)
    solutions = Dict{Any, Any}(zs[i] => out.solutions[i] for i in eachindex(zs))
    summary_df = _weighted_entrepreneur_summary(out.sim_df)
    type_df = DataFrame(
        type = 1:length(zs),
        z = zs,
        weight = out.type_df.weight,
        bequests = out.bequests_z,
        inheritances = out.inheritances_z,
    )

    return (;
        solutions,
        inheritances_z = out.inheritances_z,
        bequests_z = out.bequests_z,
        inh_zj = out.inh_zj,
        iterations = out.iterations,
        sim_df = out.sim_df,
        summary_df,
        type_df,
    )
end

function compare_tax_regimes_with_bequests(
    solver::SolutionMethod,
    par,
    epar::EntrepreneurParams,
    bpar::BequestParams,
    a_grid;
    z_grid,
    r,
    w,
    taxes,
    a_init = first(a_grid),
    maxiter = 200,
    tol = 1e-10,
    λ_inherit = 1.0,
)
    results = NamedTuple[]
    sim_dfs = DataFrame[]
    summary_dfs = DataFrame[]

    for tax in taxes
        out = solve_entrepreneur_with_bequests(
            solver,
            par,
            epar,
            bpar,
            a_grid;
            z_grid,
            r,
            w,
            tax,
            a_init,
            maxiter,
            tol,
            λ_inherit,
        )
        tax_regime = _tax_regime_label(tax)
        tax_rate = _tax_regime_rate(tax)

        sim_df_tax = copy(out.sim_df)
        sim_df_tax.tax_regime = fill(tax_regime, nrow(sim_df_tax))
        sim_df_tax.tax_rate = fill(tax_rate, nrow(sim_df_tax))

        summary_df_tax = copy(out.summary_df)
        summary_df_tax.tax_regime = fill(tax_regime, nrow(summary_df_tax))
        summary_df_tax.tax_rate = fill(tax_rate, nrow(summary_df_tax))

        push!(results, (; tax, tax_regime, tax_rate, out))
        push!(sim_dfs, sim_df_tax)
        push!(summary_dfs, summary_df_tax)
    end

    return (; results, sim_df = vcat(sim_dfs...), summary_df = vcat(summary_dfs...))
end
