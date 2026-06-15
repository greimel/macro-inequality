_tax_regime_label(::NoTax) = :no_tax
_tax_regime_label(::CapitalIncomeTax) = :capital_income_tax
_tax_regime_label(::WealthTax) = :wealth_tax

_tax_regime_rate(::NoTax) = 0.0
_tax_regime_rate(tax::CapitalIncomeTax) = tax.tau_k
_tax_regime_rate(tax::WealthTax) = tax.tau_a

function _weighted_entrepreneur_summary(sim_df)
    rows = NamedTuple[]

    for j in sort(unique(sim_df.j))
        sim_df_j = sim_df[sim_df.j .== j, :]
        weights = sim_df_j.weight
        weight_total = sum(weights)

        push!(
            rows,
            (;
                j,
                weight_total,
                a = dot(sim_df_j.a, weights) / weight_total,
                a_next = dot(sim_df_j.a_next, weights) / weight_total,
                c = dot(sim_df_j.c, weights) / weight_total,
                k = dot(sim_df_j.k, weights) / weight_total,
                pi = dot(sim_df_j.pi, weights) / weight_total,
                W_after_tax = dot(sim_df_j.W_after_tax, weights) / weight_total,
                entrepreneur_share = dot(Float64.(sim_df_j.O .== :E), weights) / weight_total,
                constrained_share = dot(Float64.(sim_df_j.constrained), weights) / weight_total,
            ),
        )
    end

    return DataFrame(rows)
end

function simulate_entrepreneur_types(solver::SolutionMethod, par, epar::EntrepreneurParams, a_grid; z_grid, weights, r, w, tax = NoTax(), a_init = first(a_grid))
    @assert length(z_grid) == length(weights)

    type_rows = NamedTuple[]
    sim_dfs = DataFrame[]

    for type in eachindex(z_grid, weights)
        z = z_grid[type]
        weight = weights[type]

        sol = solve_entrepreneur_lifecycle(solver, par, epar, a_grid; z, r, w, tax, a_init)
        sim_df_type = copy(sol.sim_df)
        sim_df_type.type = fill(type, nrow(sim_df_type))
        sim_df_type.z = fill(z, nrow(sim_df_type))
        sim_df_type.weight = fill(weight, nrow(sim_df_type))

        push!(type_rows, (; type, z, weight, sol))
        push!(sim_dfs, sim_df_type)
    end

    sim_df = vcat(sim_dfs...)
    summary_df = _weighted_entrepreneur_summary(sim_df)
    type_df = DataFrame(type = eachindex(z_grid), z = collect(z_grid), weight = collect(weights))

    return (; type_results = type_rows, type_df, sim_df, summary_df)
end

function compare_tax_regimes(solver::SolutionMethod, par, epar::EntrepreneurParams, a_grid; z_grid, weights, r, w, taxes, a_init = first(a_grid))
    results = NamedTuple[]
    sim_dfs = DataFrame[]
    summary_dfs = DataFrame[]

    for tax in taxes
        out = simulate_entrepreneur_types(solver, par, epar, a_grid; z_grid, weights, r, w, tax, a_init)
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
