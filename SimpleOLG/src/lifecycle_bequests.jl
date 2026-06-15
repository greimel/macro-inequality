"""
    _lifecycle_type_sim_df(sol, type_i, pt, π_z_i, par)

Stamp a per-type simulation DataFrame (worker or entrepreneur) with the
canonical weight columns (`weight_type`, `weight_age`, `weight`) plus the
permanent-type productivity fields (`z_labor`, `z_entrepreneur`). Leaves all
other columns produced by the inner solver (e.g. `O`, `k`, `pi` for the
entrepreneur path) untouched.
"""
function _lifecycle_type_sim_df(sol, type_i, pt::PermanentType, π_z_i, par)
    sim_df = copy(sol.sim_df)
    n = nrow(sim_df)
    π_j = pmf(par.m)
    weight_j = [Float64(π_j[j = At(sim_df.j[k])]) for k in 1:n]
    sim_df.type           = fill(type_i, n)
    sim_df.z_labor        = fill(pt.z_labor, n)
    sim_df.z_entrepreneur = fill(pt.z_entrepreneur, n)
    sim_df.weight_type    = fill(Float64(π_z_i), n)
    sim_df.weight_age     = weight_j
    sim_df.weight         = weight_j .* Float64(π_z_i)
    return sim_df
end

"""
    aggregate_wealth(sim_df)

Population-weighted mean, Gini, p50, p90, p90/p50 ratio, and top-10% wealth
share on the pooled age × type distribution. Uses the canonical `:weight`
column stamped by `_lifecycle_type_sim_df`.
"""
function aggregate_wealth(sim_df::DataFrame)
    hasproperty(sim_df, :weight) || throw(ArgumentError("sim_df needs a :weight column"))
    a = Float64.(sim_df.a)
    w = Float64.(sim_df.weight)
    total_w = sum(w)
    total_w > 0 || throw(ArgumentError("total weight is zero"))
    w_norm = w ./ total_w

    mean_a = dot(w_norm, a)

    order = sortperm(a)
    a_sorted = a[order]
    w_sorted = w_norm[order]
    cum_w = cumsum(w_sorted)
    cum_wa = cumsum(w_sorted .* a_sorted)
    total_wa = last(cum_wa)

    gini = if mean_a > 0
        lorenz = cum_wa ./ total_wa
        prev_w = 0.0
        prev_l = 0.0
        area = 0.0
        for k in eachindex(cum_w)
            Δ = cum_w[k] - prev_w
            area += 0.5 * Δ * (lorenz[k] + prev_l)
            prev_w = cum_w[k]
            prev_l = lorenz[k]
        end
        1 - 2 * area
    else
        0.0
    end

    p(q) = a_sorted[searchsortedfirst(cum_w, q)]
    p50 = p(0.5)
    p90 = p(0.9)
    p90_p50 = p50 > 0 ? p90 / p50 : NaN

    # Top-10% wealth share: fraction of total weighted wealth held by the
    # households strictly above the 90th percentile of the wealth distribution.
    top10_share = if total_wa > 0
        idx_below = searchsortedlast(cum_w, 0.9)
        wealth_below = idx_below > 0 ? cum_wa[idx_below] : 0.0
        1 - wealth_below / total_wa
    else
        0.0
    end

    return (; mean = mean_a, gini, p50, p90, p90_p50, top10_share)
end

"""
    solve_lifecycle_with_bequests(solver, par, bpar, a_grid; type_grid,
                                  epar = nothing, r, w, tax = NoTax(),
                                  a_init, maxiter, tol, λ_inherit,
                                  inh_init_scale)

Unified fixed-point solver for the OLG lifecycle with intergenerational
inheritance transmission. Dispatches on `epar`:

- `epar === nothing`: pure worker problem; each type solves
  `solve_backward_forward(solver, par, a_grid; z = pt.z_labor, inh_j)`.
- `epar::EntrepreneurParams`: occupation-choice problem; each type solves
  `solve_entrepreneur_lifecycle(solver, par, epar, a_grid;
   z_labor = pt.z_labor, z_entrepreneur = pt.z_entrepreneur, tax, inh_j)`.

Returns `(; sim_df, type_df, bequests_z, inheritances_z, inh_zj,
           iterations, gap, solutions)`. `solutions` is a `Vector` of the
per-type inner-solver outputs, indexed by the position in `type_grid`.
"""
function solve_lifecycle_with_bequests(
    solver::SolutionMethod,
    par,
    bpar::BequestParams,
    a_grid;
    type_grid::AbstractVector{PermanentType},
    epar::Union{EntrepreneurParams, Nothing} = nothing,
    r,
    w,
    tax::TaxRegime = NoTax(),
    a_init = first(a_grid),
    maxiter = 200,
    tol = 1e-10,
    λ_inherit = 0.5,
    inh_init_scale = 1e-2,
)
    0 < λ_inherit <= 1 || throw(ArgumentError("λ_inherit must lie in (0, 1]"))

    _check_terminal_safe(par, bpar, a_grid, a_init)

    n_types = length(type_grid)
    setup = _bequest_setup(bpar, par, n_types)

    z_labors = [t.z_labor for t in type_grid]
    z_entrepreneurs = [t.z_entrepreneur for t in type_grid]
    inheritances_z = inh_init_scale .* w .* z_labors
    inh_zj = _inheritance_schedule(inheritances_z, setup)

    sim_df = DataFrame()
    type_df = DataFrame()
    solutions = Vector{Any}(undef, n_types)
    bequests_z = zeros(Float64, n_types)
    new_inh_z = copy(inheritances_z)
    gap = Inf

    for it in 1:maxiter
        sim_dfs = DataFrame[]
        for type_i in 1:n_types
            pt = type_grid[type_i]
            inh_j = collect(inh_zj[type = At(type_i)])
            sol = if isnothing(epar)
                solve_backward_forward(solver, par, a_grid;
                    r, w, z = pt.z_labor, inh_j, a_init)
            else
                solve_entrepreneur_lifecycle(solver, par, epar, a_grid;
                    z_labor = pt.z_labor, z_entrepreneur = pt.z_entrepreneur,
                    r, w, tax, a_init, inh_j)
            end
            solutions[type_i] = sol
            push!(sim_dfs, _lifecycle_type_sim_df(sol, type_i, pt, setup.π_z[type_i], par))
        end

        sim_df = vcat(sim_dfs...)
        bequests_z = compute_bequests(sim_df, par)
        new_inh_z = _inheritances_by_type(bequests_z, setup)
        gap = norm(new_inh_z .- inheritances_z)

        if gap < tol
            inh_zj = _inheritance_schedule(new_inh_z, setup)
            type_df = DataFrame(
                type = 1:n_types,
                z_labor = z_labors,
                z_entrepreneur = z_entrepreneurs,
                weight = setup.π_z,
                bequests = bequests_z,
                inheritances = new_inh_z,
            )
            return (; sim_df, type_df, bequests_z, inheritances_z = new_inh_z,
                    inh_zj, iterations = it, gap, solutions)
        end

        inheritances_z .= λ_inherit .* new_inh_z .+ (1 - λ_inherit) .* inheritances_z
        inh_zj = _inheritance_schedule(inheritances_z, setup)
    end

    @warn "Inheritance fixed-point did not converge" maxiter tol gap
    inh_zj = _inheritance_schedule(new_inh_z, setup)
    type_df = DataFrame(
        type = 1:n_types,
        z_labor = z_labors,
        z_entrepreneur = z_entrepreneurs,
        weight = setup.π_z,
        bequests = bequests_z,
        inheritances = new_inh_z,
    )
    return (; sim_df, type_df, bequests_z, inheritances_z = new_inh_z,
            inh_zj, iterations = maxiter, gap, solutions)
end
