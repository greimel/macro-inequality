effective_return_worker(::NoTax; r) = 1 + r

effective_return_worker(tax::CapitalIncomeTax; r) = 1 + r * (1 - tax.tau_k)

effective_return_worker(tax::WealthTax; r) = 1 - tax.tau_a + r

function _constrained_profit_derivative(a, z, epar::EntrepreneurParams; r)
    z <= 0 && return 0.0

    collateral_capital = max(epar.theta * max(a, 0.0), eps(Float64))
    return epar.theta * (epar.alpha * z * collateral_capital^(epar.alpha - 1) - (r + epar.delta))
end

function effective_return_entrepreneur(a, z, epar::EntrepreneurParams, tax::NoTax; r)
    k = entrepreneur_capital(a, z, epar; r)
    k_star = _efficient_entrepreneur_capital(z, epar; r)
    k < k_star - 1e-10 || return effective_return_worker(tax; r)

    return 1 + r + _constrained_profit_derivative(a, z, epar; r)
end

function effective_return_entrepreneur(a, z, epar::EntrepreneurParams, tax::CapitalIncomeTax; r)
    k = entrepreneur_capital(a, z, epar; r)
    k_star = _efficient_entrepreneur_capital(z, epar; r)
    k < k_star - 1e-10 || return effective_return_worker(tax; r)

    return 1 + (r + _constrained_profit_derivative(a, z, epar; r)) * (1 - tax.tau_k)
end

function effective_return_entrepreneur(a, z, epar::EntrepreneurParams, tax::WealthTax; r)
    k = entrepreneur_capital(a, z, epar; r)
    k_star = _efficient_entrepreneur_capital(z, epar; r)
    k < k_star - 1e-10 || return effective_return_worker(tax; r)

    return 1 - tax.tau_a + r + _constrained_profit_derivative(a, z, epar; r)
end

function _entrepreneur_threshold(z, epar::EntrepreneurParams; r)
    z <= 0 && return 0.0

    k_star = _efficient_entrepreneur_capital(z, epar; r)
    if !isfinite(k_star)
        return Inf
    end

    epar.theta > 0 || return Inf
    return k_star / epar.theta
end

function _invert_worker_asset(a_next, c_curr, inheritance, par, tax::TaxRegime; r, w, y, m)
    slope = effective_return_worker(tax; r)
    return (c_curr + saving_weight(m, par) * a_next - w * y - inheritance) / slope
end

function _invert_entrepreneur_unconstrained_asset(required_wealth, z, epar::EntrepreneurParams, tax::TaxRegime; r)
    threshold = _entrepreneur_threshold(z, epar; r)
    isfinite(threshold) || return nothing

    slope = effective_return_worker(tax; r)
    intercept = after_tax_wealth(threshold, z, epar, tax; r) - slope * threshold
    a = (required_wealth - intercept) / slope

    return a >= threshold - 1e-10 ? a : nothing
end

function _invert_entrepreneur_asset(required_wealth, z, epar::EntrepreneurParams, tax::TaxRegime; r)
    required_wealth <= 0 && return 0.0

    unconstrained = _invert_entrepreneur_unconstrained_asset(required_wealth, z, epar, tax; r)
    !isnothing(unconstrained) && return unconstrained

    objective(a) = after_tax_wealth(a, z, epar, tax; r) - required_wealth

    objective(0.0) >= 0 && return 0.0

    upper = _entrepreneur_threshold(z, epar; r)
    upper = isfinite(upper) ? max(upper, 1.0) : 1.0

    while objective(upper) < 0
        upper *= 2
        upper > 1e10 && throw(ArgumentError("unable to bracket entrepreneur asset inversion"))
    end

    return find_zero(objective, (0.0, upper))
end

function _prepare_branch_points(a_vals, c_vals, a_next_vals, value_vals)
    keep = findall(i ->
        isfinite(a_vals[i]) && isfinite(c_vals[i]) && isfinite(a_next_vals[i]) && isfinite(value_vals[i]),
        eachindex(a_vals),
    )

    xs = Float64.(a_vals[keep])
    cs = Float64.(c_vals[keep])
    a_nexts = Float64.(a_next_vals[keep])
    vals = Float64.(value_vals[keep])

    order = sortperm(xs)
    xs = xs[order]
    cs = cs[order]
    a_nexts = a_nexts[order]
    vals = vals[order]

    xs_out = Float64[]
    cs_out = Float64[]
    a_nexts_out = Float64[]
    vals_out = Float64[]

    i = 1
    while i <= length(xs)
        j = i
        best = i

        while j < length(xs) && isapprox(xs[j + 1], xs[i]; atol = 1e-10, rtol = 1e-10)
            j += 1
            if vals[j] > vals[best]
                best = j
            end
        end

        push!(xs_out, xs[best])
        push!(cs_out, cs[best])
        push!(a_nexts_out, a_nexts[best])
        push!(vals_out, vals[best])

        i = j + 1
    end

    if length(xs_out) == 1
        push!(xs_out, xs_out[1] + 1e-8)
        push!(cs_out, cs_out[1])
        push!(a_nexts_out, a_nexts_out[1])
        push!(vals_out, vals_out[1])
    end

    return (; xs = xs_out, cs = cs_out, a_nexts = a_nexts_out, vals = vals_out)
end

function _branch_policy_on_grid(a_vals, c_vals, a_next_vals, value_vals, a_grid, par, continuation_itp, resource_fn; m, survival)
    prepared = _prepare_branch_points(a_vals, c_vals, a_next_vals, value_vals)

    c_itp = linear_interpolation(prepared.xs, prepared.cs, extrapolation_bc = Line())
    a_next_itp = linear_interpolation(prepared.xs, prepared.a_nexts, extrapolation_bc = Line())
    value_itp = linear_interpolation(prepared.xs, prepared.vals, extrapolation_bc = Line())

    c_grid = similar(collect(a_grid), Float64)
    a_next_grid = similar(collect(a_grid), Float64)
    value_grid = similar(collect(a_grid), Float64)

    a_upper = last(a_grid)

    for (a_i, a) in enumerate(a_grid)
        c_candidate = c_itp(a)
        a_next_candidate = a_next_itp(a)

        if a_next_candidate < par.a̲ || a_next_candidate > a_upper
            a_next_candidate = clamp(a_next_candidate, par.a̲, a_upper)
            c_candidate = resource_fn(a) - saving_weight(m, par) * a_next_candidate
            value_candidate = par.u(c_candidate) + par.β * survival * continuation_itp(a_next_candidate)
        else
            value_candidate = value_itp(a)
        end

        c_grid[a_i] = c_candidate
        a_next_grid[a_i] = a_next_candidate
        value_grid[a_i] = value_candidate
    end

    return (; c = c_grid, a_next = a_next_grid, value = value_grid)
end

function solve_entrepreneur_lifecycle(::EGM, par, epar::EntrepreneurParams, a_grid;
    z = nothing, z_labor = 1.0, z_entrepreneur = nothing,
    r, w, tax = NoTax(), a_init = first(a_grid), inh_j = nothing,
)
    _validate_entrepreneur_par(par)

    isnothing(z) || isnothing(z_entrepreneur) ||
        throw(ArgumentError("Pass either `z` or `z_entrepreneur`, not both"))
    z_ent = isnothing(z_entrepreneur) ? (isnothing(z) ? 1.0 : z) : z_entrepreneur

    first(a_grid) < 0.0 && throw(ArgumentError(
        "Entrepreneur EGM does not support negative asset grids; got lower bound $(first(a_grid))",
    ))

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
    marginal_value = DimArray(fill(Inf, size_out...), dims, name = :marginal_value)

    for j in reverse(collect(j_dim))
        m = par.m[j = At(j)]
        survival = 1 - m
        y = par.y[j = At(j)]
        inheritance = inheritance_path[j + 1]

        if survival == 0 || j == par.J
            for (a_i, a) in enumerate(a_grid)
                resources_W = after_tax_wealth(a, 0.0, epar, tax; r) + w * y * z_labor + inheritance
                terminal_W = terminal_allocation(resources_W, par)
                value_W[a = a_i, j = At(j)] = terminal_W.value
                c_W[a = a_i, j = At(j)] = terminal_W.c
                policy_a_W[a = a_i, j = At(j)] = terminal_W.a_next

                resources_E = after_tax_wealth(a, z_ent, epar, tax; r) + w * y * z_labor - epar.f + inheritance
                terminal_E = terminal_allocation(resources_E, par)
                value_E[a = a_i, j = At(j)] = terminal_E.value
                c_E[a = a_i, j = At(j)] = terminal_E.c
                policy_a_E[a = a_i, j = At(j)] = terminal_E.a_next

                if terminal_E.value > terminal_W.value
                    value[a = a_i, j = At(j)] = terminal_E.value
                    c[a = a_i, j = At(j)] = terminal_E.c
                    policy_a[a = a_i, j = At(j)] = _closest_grid_index(a_grid, terminal_E.a_next)
                    policy_k[a = a_i, j = At(j)] = entrepreneur_capital(a, z_ent, epar; r)
                    policy_O[a = a_i, j = At(j)] = :E
                    marginal_value[a = a_i, j = At(j)] = u_prime(terminal_E.c, par) * effective_return_entrepreneur(a, z_ent, epar, tax; r)
                else
                    value[a = a_i, j = At(j)] = terminal_W.value
                    c[a = a_i, j = At(j)] = terminal_W.c
                    policy_a[a = a_i, j = At(j)] = _closest_grid_index(a_grid, terminal_W.a_next)
                    policy_k[a = a_i, j = At(j)] = 0.0
                    policy_O[a = a_i, j = At(j)] = :W
                    marginal_value[a = a_i, j = At(j)] = u_prime(terminal_W.c, par) * effective_return_worker(tax; r)
                end
            end

            continue
        end

        value_next = collect(value[j = At(j + 1)])
        marginal_next = collect(marginal_value[j = At(j + 1)])
        continuation_itp = _clean_interp(a_grid, value_next)

        c_common = inv_u_prime.(par.β .* survival .* marginal_next, Ref(par))

        a_W = similar(c_common)
        a_E = similar(c_common)
        value_pairs_W = similar(c_common)
        value_pairs_E = similar(c_common)

        for (a_i_next, a_next_choice) in enumerate(a_grid)
            c_curr = c_common[a_i_next]
            continuation_value = par.u(c_curr) + par.β * survival * value_next[a_i_next]

            a_W[a_i_next] = _invert_worker_asset(a_next_choice, c_curr, inheritance, par, tax; r, w, y = y * z_labor, m)
            value_pairs_W[a_i_next] = continuation_value

            required_wealth = c_curr + saving_weight(m, par) * a_next_choice + epar.f - w * y * z_labor - inheritance
            a_E[a_i_next] = _invert_entrepreneur_asset(required_wealth, z_ent, epar, tax; r)
            value_pairs_E[a_i_next] = continuation_value
        end

        resources_W_fn(a) = after_tax_wealth(a, 0.0, epar, tax; r) + w * y * z_labor + inheritance
        resources_E_fn(a) = after_tax_wealth(a, z_ent, epar, tax; r) + w * y * z_labor - epar.f + inheritance

        branch_W = _branch_policy_on_grid(a_W, c_common, collect(a_grid), value_pairs_W, a_grid, par, continuation_itp, resources_W_fn; m, survival)
        branch_E = _branch_policy_on_grid(a_E, c_common, collect(a_grid), value_pairs_E, a_grid, par, continuation_itp, resources_E_fn; m, survival)

        value_W[j = At(j)] .= branch_W.value
        value_E[j = At(j)] .= branch_E.value
        c_W[j = At(j)] .= branch_W.c
        c_E[j = At(j)] .= branch_E.c
        policy_a_W[j = At(j)] .= branch_W.a_next
        policy_a_E[j = At(j)] .= branch_E.a_next

        for (a_i, a) in enumerate(a_grid)
            if branch_E.value[a_i] > branch_W.value[a_i]
                value[a = a_i, j = At(j)] = branch_E.value[a_i]
                c[a = a_i, j = At(j)] = branch_E.c[a_i]
                policy_a[a = a_i, j = At(j)] = _closest_grid_index(a_grid, branch_E.a_next[a_i])
                policy_k[a = a_i, j = At(j)] = entrepreneur_capital(a, z_ent, epar; r)
                policy_O[a = a_i, j = At(j)] = :E
                marginal_value[a = a_i, j = At(j)] = u_prime(branch_E.c[a_i], par) * effective_return_entrepreneur(a, z_ent, epar, tax; r)
            else
                value[a = a_i, j = At(j)] = branch_W.value[a_i]
                c[a = a_i, j = At(j)] = branch_W.c[a_i]
                policy_a[a = a_i, j = At(j)] = _closest_grid_index(a_grid, branch_W.a_next[a_i])
                policy_k[a = a_i, j = At(j)] = 0.0
                policy_O[a = a_i, j = At(j)] = :W
                marginal_value[a = a_i, j = At(j)] = u_prime(branch_W.c[a_i], par) * effective_return_worker(tax; r)
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
