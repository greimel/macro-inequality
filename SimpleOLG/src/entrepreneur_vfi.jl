_closest_grid_index(a_grid, a) = findmin(abs.(a_grid .- a))[2]

function _validate_entrepreneur_par(par)
    par.annuities && throw(ArgumentError("Entrepreneur solvers require annuities = false"))
    return nothing
end

function _inheritance_path(inh_j, J)
    isnothing(inh_j) && return zeros(Float64, J + 1)

    length(inh_j) == J + 1 || throw(DimensionMismatch("inh_j must have length $(J + 1)"))
    return Float64.(collect(inh_j))
end

function _best_savings_choice(resources, continuation, a_grid, par; m, survival)
    if survival == 0
        terminal = terminal_allocation(resources, par)
        return (; value = terminal.value, a_i_next = _closest_grid_index(a_grid, terminal.a_next), a_next = terminal.a_next)
    end

    values = similar(collect(a_grid), Float64)

    for (a_i_next, a_next) in enumerate(a_grid)
        values[a_i_next] = par.u(resources - saving_weight(m, par) * a_next) + par.β * survival * continuation[a_i_next]
    end

    value, a_i_next = findmax(values)
    return (; value, a_i_next, a_next = a_grid[a_i_next])
end

function _clean_interp(a_grid, values; floor = -1e16)
    clean_values = map(value -> isfinite(value) ? value : floor, values)
    return linear_interpolation(a_grid, clean_values, extrapolation_bc = Line())
end

function _evaluate_branch(a_grid, branch_values, branch_policy, a)
    a_i = _closest_grid_index(a_grid, a)

    if isapprox(a_grid[a_i], a; atol = 1e-12, rtol = 1e-12)
        return branch_values[a_i], branch_policy[a_i]
    end

    value_itp = _clean_interp(a_grid, branch_values)
    policy_itp = linear_interpolation(a_grid, branch_policy, extrapolation_bc = Line())

    return value_itp(a), policy_itp(a)
end

function _simulate_entrepreneur_path(
    par,
    epar,
    a_grid,
    value_W,
    value_E,
    policy_a_W,
    policy_a_E;
    z = nothing,
    z_labor = 1.0,
    z_entrepreneur = nothing,
    r,
    w,
    tax,
    a_init,
    inh_j,
)
    isnothing(z) || isnothing(z_entrepreneur) ||
        throw(ArgumentError("Pass either `z` or `z_entrepreneur`, not both"))
    z_ent = isnothing(z_entrepreneur) ? (isnothing(z) ? 1.0 : z) : z_entrepreneur

    rows = NamedTuple[]
    a = float(a_init)
    k_star = _efficient_entrepreneur_capital(z_ent, epar; r)
    inheritance_path = _inheritance_path(inh_j, par.J)

    for j in 0:par.J
        m = par.m[j = At(j)]
        y = par.y[j = At(j)]
        inheritance = inheritance_path[j + 1]

        value_W_j = collect(value_W[j = At(j)])
        value_E_j = collect(value_E[j = At(j)])
        policy_a_W_j = collect(policy_a_W[j = At(j)])
        policy_a_E_j = collect(policy_a_E[j = At(j)])

        v_W, a_next_W = _evaluate_branch(a_grid, value_W_j, policy_a_W_j, a)
        v_E, a_next_E = _evaluate_branch(a_grid, value_E_j, policy_a_E_j, a)

        if v_E > v_W
            O = :E
            k = entrepreneur_capital(a, z_ent, epar; r)
            π = entrepreneur_profit(a, z_ent, epar; r)
            W_after_tax = after_tax_wealth(a, z_ent, epar, tax; r)
            resources = W_after_tax + w * y * z_labor - epar.f + inheritance
            a_next = j == par.J ? max(a_next_E, 0.0) : clamp(a_next_E, first(a_grid), last(a_grid))
            constrained = k < k_star - 1e-10
        else
            O = :W
            k = 0.0
            π = 0.0
            W_after_tax = after_tax_wealth(a, 0.0, epar, tax; r)
            resources = W_after_tax + w * y * z_labor + inheritance
            a_next = j == par.J ? max(a_next_W, 0.0) : clamp(a_next_W, first(a_grid), last(a_grid))
            constrained = false
        end

        c = resources - saving_weight(m, par) * a_next

        push!(rows, (; j, m, y, r, w, a, c, a_next, k, O, constrained, pi = π, W_after_tax, inheritance))
        a = a_next
    end

    return DataFrame(rows)
end

function solve_entrepreneur_lifecycle(::VFI, par, epar::EntrepreneurParams, a_grid;
    z = nothing, z_labor = 1.0, z_entrepreneur = nothing,
    r, w, tax = NoTax(), a_init = first(a_grid), inh_j = nothing,
)
    _validate_entrepreneur_par(par)

    isnothing(z) || isnothing(z_entrepreneur) ||
        throw(ArgumentError("Pass either `z` or `z_entrepreneur`, not both"))
    z_ent = isnothing(z_entrepreneur) ? (isnothing(z) ? 1.0 : z) : z_entrepreneur

    j_dim = DD.dims(par.m, :j)
    @assert collect(j_dim) == 0:par.J

    a_dim = Dim{:a}(a_grid)
    dims = (a_dim, j_dim)
    size_out = (length(a_grid), length(j_dim))
    zero_index = _closest_grid_index(a_grid, 0.0)
    inheritance_path = _inheritance_path(inh_j, par.J)

    value_W = DimArray(fill(-Inf, size_out...), dims, name = :value_W)
    value_E = DimArray(fill(-Inf, size_out...), dims, name = :value_E)
    value = DimArray(fill(-Inf, size_out...), dims, name = :value)

    c_W = DimArray(fill(0.0, size_out...), dims, name = :c_W)
    c_E = DimArray(fill(0.0, size_out...), dims, name = :c_E)
    c = DimArray(fill(0.0, size_out...), dims, name = :c)

    policy_a_W = DimArray(fill(0.0, size_out...), dims, name = :policy_a_W)
    policy_a_E = DimArray(fill(0.0, size_out...), dims, name = :policy_a_E)
    policy_a = DimArray(fill(zero_index, size_out...), dims, name = :policy_a)
    policy_k = DimArray(fill(0.0, size_out...), dims, name = :policy_k)
    policy_O = DimArray(fill(:W, size_out...), dims, name = :policy_O)

    for j in reverse(collect(j_dim))
        continuation = j == par.J ? zeros(length(a_grid)) : collect(value[j = At(j + 1)])
        m = par.m[j = At(j)]
        y = par.y[j = At(j)]
        inheritance = inheritance_path[j + 1]

        for (a_i, a) in enumerate(a_grid)
            k = entrepreneur_capital(a, z_ent, epar; r)

            resources_W = after_tax_wealth(a, 0.0, epar, tax; r) + w * y * z_labor + inheritance
            resources_E = after_tax_wealth(a, z_ent, epar, tax; r) + w * y * z_labor - epar.f + inheritance

            choice_W = _best_savings_choice(resources_W, continuation, a_grid, par; m, survival = 1 - m)
            choice_E = _best_savings_choice(resources_E, continuation, a_grid, par; m, survival = 1 - m)

            value_W[a = a_i, j = At(j)] = choice_W.value
            value_E[a = a_i, j = At(j)] = choice_E.value
            c_W[a = a_i, j = At(j)] = resources_W - saving_weight(m, par) * choice_W.a_next
            c_E[a = a_i, j = At(j)] = resources_E - saving_weight(m, par) * choice_E.a_next
            policy_a_W[a = a_i, j = At(j)] = choice_W.a_next
            policy_a_E[a = a_i, j = At(j)] = choice_E.a_next

            if choice_E.value > choice_W.value
                value[a = a_i, j = At(j)] = choice_E.value
                c[a = a_i, j = At(j)] = c_E[a = a_i, j = At(j)]
                policy_a[a = a_i, j = At(j)] = choice_E.a_i_next
                policy_k[a = a_i, j = At(j)] = k
                policy_O[a = a_i, j = At(j)] = :E
            else
                value[a = a_i, j = At(j)] = choice_W.value
                c[a = a_i, j = At(j)] = c_W[a = a_i, j = At(j)]
                policy_a[a = a_i, j = At(j)] = choice_W.a_i_next
                policy_k[a = a_i, j = At(j)] = 0.0
                policy_O[a = a_i, j = At(j)] = :W
            end
        end
    end

    sim_df = _simulate_entrepreneur_path(
        par,
        epar,
        a_grid,
        value_W,
        value_E,
        policy_a_W,
        policy_a_E;
        z_labor,
        z_entrepreneur = z_ent,
        r,
        w,
        tax,
        a_init,
        inh_j = inheritance_path,
    )

    return (; value_W, value_E, value, c_W, c_E, c, policy_a_W, policy_a_E, policy_a, policy_k, policy_O, sim_df)
end
