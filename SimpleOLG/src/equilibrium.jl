criterion(a, b) = (a - b) / (1 + max(abs(a), abs(b)))

function partial_equilibrium(
    par,
    a_grid,
    (; K_guess, r, w);
    pmf = pmf(par.m),
    details = true,
    return_df = false,
    a_init = a_grid[1],
    solution_method = VFI(),
)
    (; bonds2GDP) = par

    sol = solve_backward_forward(solution_method, par, a_grid; r, w, a_init)
    (; sim_df) = sol

    agg_nt = aggregate(sim_df, pmf)

    GDP = output(K_guess, par)
    B₀ = bonds2GDP * GDP

    K_hh = agg_nt.a
    K_supply = K_hh - B₀
    ζ = K_supply - K_guess

    if details
        if return_df
            return (; ζ, K_guess, r, w, K_hh, K_supply, B₀, par, sol, pmf_df = DataFrame(pmf))
        else
            return (; ζ, K_guess, r, w, K_hh, K_supply, B₀, par, sol)
        end
    end

    return ζ
end

function general_equilibrium(
    par,
    a_grid;
    pmf = pmf(par.m),
    K_bracket = (1e-1, 10 * inverse_interest_rate(par.r, par)),
    return_df = true,
    kwargs...,
)
    function objective(K_guess; details = false, return_df = false)
        r = interest_rate(K_guess, par)
        w = wage(K_guess, par)

        partial_equilibrium(par, a_grid, (; K_guess, r, w); details, pmf, return_df, kwargs...)
    end

    K_opt = find_zero(objective, K_bracket)
    return objective(K_opt; details = true, return_df)
end
