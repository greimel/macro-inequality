@testset "Micro-test: 1-period unconstrained entrepreneur" begin
    #=
    One-Period Unconstrained Entrepreneur

    Setup:
    A terminal-period problem with ``m_0 = 1`` and therefore ``a'_0 = 0``.
    Parameters are ``\alpha = 0.5``, ``\delta = 0``, ``\theta = 10``,
    ``z = 2``, ``r = 0.5``, ``w = 0``, ``a_0 = 1``, ``\gamma = 2``,
    and ``J = 0``.

    Key equations:
    Efficient capital satisfies
    ``k^* = \left(\frac{\alpha z}{r + \delta}\right)^{\frac{1}{1-\alpha}}``.
    Entrepreneurial profits are
    ``\pi(a) = z k^\alpha - (r + \delta)k``.
    Final-period occupation choice compares
    ``R_E - R_W = \pi(a) - f``
    because both occupations share the same asset return and there is no reason to save.

    Derivation:
    We obtain
    ``k^* = (0.5 \cdot 2 / 0.5)^2 = 4``.
    Since ``\theta a_0 = 10 > 4``, the collateral constraint does not bind, so
    ``k = k^* = 4``.
    Hence
    ``\pi = 2 \cdot 4^{0.5} - 0.5 \cdot 4 = 2``.
    Resources are therefore
    ``R_E = (1+r)a_0 + \pi - f = 1.5 + 2 - f = 3.5 - f``
    and
    ``R_W = (1+r)a_0 = 1.5``.

    Verbal argument:
    Because ``m_0 = 1``, utility is strictly increasing in current consumption and
    the agent consumes all available resources immediately. The occupation decision
    therefore collapses to a static comparison of entrepreneurial profits net of the
    fixed cost ``f`` against the worker benchmark.

    Expected outcomes:
    For ``f = 1.0``, entrepreneurship dominates since ``\pi - f = 1 > 0``:
    ``O = :E``, ``k = 4.0``, ``\pi = 2.0``, ``c = 2.5``, ``a' = 0``.
    For ``f = 3.0``, working dominates since ``\pi - f = -1 < 0``:
    ``O = :W``, ``k = 0.0``, ``\pi = 0.0``, ``c = 1.5``, ``a' = 0``.
    =#
    y = simple_income_profile(1, 1; y = 0.0, yR = 0.0)
    par = get_par(; demo = :perpetual_youth, annuities = false, y, β = 1.0, γ = 2.0, a̲ = 0.0)
    a_grid = [0.0, 1.0]
    z = 2.0
    r = 0.5
    w = 0.0

    epar_enter = EntrepreneurParams(alpha = 0.5, delta = 0.0, theta = 10.0, f = 1.0)
    epar_work = EntrepreneurParams(alpha = 0.5, delta = 0.0, theta = 10.0, f = 3.0)

    k_star = SimpleOLG._efficient_entrepreneur_capital(z, epar_enter; r)
    @test k_star ≈ 4.0
    @test entrepreneur_capital(1.0, z, epar_enter; r) ≈ 4.0

    sol_enter = solve_entrepreneur_lifecycle(VFI(), par, epar_enter, a_grid; z, r, w, tax = NoTax(), a_init = 1.0)
    row_enter = only(eachrow(sol_enter.sim_df))
    @test sol_enter.policy_O[a = At(1.0), j = At(0)] == :E
    @test sol_enter.policy_k[a = At(1.0), j = At(0)] ≈ 4.0
    @test row_enter.k ≈ 4.0
    @test row_enter.pi ≈ 2.0
    @test row_enter.W_after_tax ≈ 3.5
    @test row_enter.c ≈ 2.5
    @test row_enter.a_next == 0.0
    @test !row_enter.constrained

    sol_work = solve_entrepreneur_lifecycle(VFI(), par, epar_work, a_grid; z, r, w, tax = NoTax(), a_init = 1.0)
    row_work = only(eachrow(sol_work.sim_df))
    @test sol_work.policy_O[a = At(1.0), j = At(0)] == :W
    @test sol_work.policy_k[a = At(1.0), j = At(0)] == 0.0
    @test row_work.k == 0.0
    @test row_work.pi == 0.0
    @test row_work.W_after_tax ≈ 1.5
    @test row_work.c ≈ 1.5
    @test row_work.a_next == 0.0
    @test !row_work.constrained
end

@testset "Warm-glow terminal entrepreneur leaves bequest" begin
    y = simple_income_profile(1, 1; y = 0.0, yR = 0.0)
    par = get_par(; demo = :perpetual_youth, annuities = false, y, β = 1.0, γ = 2.0, phi = 1.0, b_floor = 0.0, a̲ = 0.0)
    a_grid = [0.0, 1.0, 2.0]
    z = 2.0
    r = 0.5
    w = 0.0
    epar = EntrepreneurParams(alpha = 0.5, delta = 0.0, theta = 10.0, f = 1.0)

    sol = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid; z, r, w, tax = NoTax(), a_init = 1.0)
    row = only(eachrow(sol.sim_df))

    @test row.O == :E
    @test row.c ≈ 1.25
    @test row.a_next ≈ 1.25
    @test row.c + row.a_next ≈ row.W_after_tax - epar.f
end

@testset "Micro-test: 1-period constrained entrepreneur" begin
    #=
    One-Period Constrained Entrepreneur

    Setup:
    The environment is identical to the previous test except that
    ``\theta = 1``. Thus the entrepreneur can pledge at most current wealth,
    so the collateral limit becomes potentially binding already at ``a_0 = 1``.

    Key equations:
    Capital is chosen according to
    ``k(a) = \min\{k^*, \theta a\}``.
    Final-period occupation still depends on the sign of
    ``R_E - R_W = \pi(a) - f``.

    Derivation:
    The efficient scale remains
    ``k^* = (0.5 \cdot 2 / 0.5)^2 = 4``.
    But now ``\theta a_0 = 1 < 4``, so the collateral constraint binds and
    ``k = \theta a_0 = 1``.
    Profits fall to
    ``\pi = 2 \cdot 1^{0.5} - 0.5 \cdot 1 = 1.5``.
    Consequently,
    ``R_E = (1+r)a_0 + \pi - f = 1.5 + 1.5 - f = 3.0 - f``
    while the worker benchmark is still
    ``R_W = 1.5``.

    Verbal argument:
    Relative to the unconstrained case, the only change is that entrepreneurial
    scale is capped below the privately optimal level. That lowers profits and makes
    the fixed cost more likely to overturn the entrepreneurial option, which is why
    this test directly isolates the constrained branch and the ``constrained`` flag.

    Expected outcomes:
    For ``f = 1.0``, entrepreneurship still dominates because ``\pi - f = 0.5 > 0``:
    ``O = :E``, ``k = 1.0``, ``\pi = 1.5``, ``c = 2.0``, ``a' = 0``,
    ``\texttt{constrained} = \texttt{true}``.
    For ``f = 2.0``, working dominates because ``\pi - f = -0.5 < 0``:
    ``O = :W``, ``k = 0.0``, ``\pi = 0.0``, ``c = 1.5``, ``a' = 0``.
    =#
    y = simple_income_profile(1, 1; y = 0.0, yR = 0.0)
    par = get_par(; demo = :perpetual_youth, annuities = false, y, β = 1.0, γ = 2.0, a̲ = 0.0)
    a_grid = [0.0, 1.0]
    z = 2.0
    r = 0.5
    w = 0.0

    epar_enter = EntrepreneurParams(alpha = 0.5, delta = 0.0, theta = 1.0, f = 1.0)
    epar_work = EntrepreneurParams(alpha = 0.5, delta = 0.0, theta = 1.0, f = 2.0)

    k_star = SimpleOLG._efficient_entrepreneur_capital(z, epar_enter; r)
    @test k_star ≈ 4.0
    @test entrepreneur_capital(1.0, z, epar_enter; r) ≈ 1.0

    sol_enter = solve_entrepreneur_lifecycle(VFI(), par, epar_enter, a_grid; z, r, w, tax = NoTax(), a_init = 1.0)
    row_enter = only(eachrow(sol_enter.sim_df))
    @test sol_enter.policy_O[a = At(1.0), j = At(0)] == :E
    @test sol_enter.policy_k[a = At(1.0), j = At(0)] ≈ 1.0
    @test row_enter.k ≈ 1.0
    @test row_enter.pi ≈ 1.5
    @test row_enter.W_after_tax ≈ 3.0
    @test row_enter.c ≈ 2.0
    @test row_enter.a_next == 0.0
    @test row_enter.constrained

    sol_work = solve_entrepreneur_lifecycle(VFI(), par, epar_work, a_grid; z, r, w, tax = NoTax(), a_init = 1.0)
    row_work = only(eachrow(sol_work.sim_df))
    @test sol_work.policy_O[a = At(1.0), j = At(0)] == :W
    @test sol_work.policy_k[a = At(1.0), j = At(0)] == 0.0
    @test row_work.k == 0.0
    @test row_work.pi == 0.0
    @test row_work.W_after_tax ≈ 1.5
    @test row_work.c ≈ 1.5
    @test row_work.a_next == 0.0
    @test !row_work.constrained
end

@testset "Micro-test: 2-period unconstrained entrepreneur" begin
    #=
    Two-Period Unconstrained Entrepreneur

    Setup:
    A two-period lifecycle with ``m = [0.5, 1.0]``, ``\beta = 1``,
    ``\gamma = 2``, no labor income, and asset grid ``a \in 0:0.5:4``.
    Parameters are ``\alpha = 0.5``, ``\delta = 1``, ``\theta = 10``,
    ``z = 2``, ``r = 0``, ``w = 0``, ``a_0 = 3``, and ``\texttt{annuities} = \texttt{false}``.

    Key equations:
    Entrepreneurial technology implies
    ``k^* = \left(\frac{\alpha z}{r + \delta}\right)^{\frac{1}{1-\alpha}}``
    and
    ``\pi = z(k^*)^\alpha - (r+\delta)k^*``.
    The implemented savings problem in period 0 is now
    ``\max_{a'} u(c_0) + \beta(1-m_0)u(c_1)``
    subject to
    ``c_0 = R_0 - a'``.
    Survival still discounts continuation utility, but it no longer scales the
    period-0 savings cost.

    Derivation:
    We obtain
    ``k^* = (0.5 \cdot 2 / 1)^2 = 1``
    and therefore
    ``\pi = 2 \cdot 1 - 1 \cdot 1 = 1``.
    Since ``\theta a`` is always well above ``1`` along the relevant path,
    the entrepreneur is unconstrained at both ages.
    Entrepreneurial resources are
    ``R_E(a) = a + 1 - f``
    and worker resources are
    ``R_W(a) = a``.

    For ``f = 0.5``, entrepreneurship dominates at both ages because
    ``R_E(a) - R_W(a) = 1 - 0.5 = 0.5 > 0``.
    The two budget equations are
    ``c_0 = 3.5 - a'`` and ``c_1 = a' + 0.5``.
    On the grid, the discrete objective is maximized at ``a' = 1``.

    For ``f = 1.5``, entrepreneurship is never chosen because
    ``1 - 1.5 = -0.5 < 0``.
    Then ``c_0 = 3 - a'`` and ``c_1 = a'``, and the grid optimum is again
    ``a' = 1``.

    Verbal argument:
    This test isolates the dynamic entrepreneur problem without binding collateral.
    Because profits are constant in wealth in the unconstrained region, the fixed
    cost determines occupation, while mortality alone pushes period-0 consumption
    above period-1 consumption under ``annuities = false``.

    Grid verification:
    The relevant discrete optima lie on the grid. The test checks that
    ``a' = 1`` dominates neighboring points in both occupation regimes.
    =#
    y = simple_income_profile(2, 2; y = 0.0, yR = 0.0)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.5, y, β = 1.0, γ = 2.0, a̲ = 0.0)
    a_grid = 0.0:0.5:4.0
    z = 2.0
    r = 0.0
    w = 0.0

    objective_E(a_next) = par.u(3.5 - a_next) + 0.5 * par.u(a_next + 0.5)
    objective_W(a_next) = par.u(3.0 - a_next) + 0.5 * par.u(a_next)
    @test objective_E(1.0) > objective_E(1.5)
    @test objective_E(1.0) > objective_E(2.0)
    @test objective_W(1.0) >= objective_W(1.5)
    @test objective_W(1.0) > objective_W(2.0)

    epar_entrepreneur = EntrepreneurParams(alpha = 0.5, delta = 1.0, theta = 10.0, f = 0.5)
    sol_entrepreneur = solve_entrepreneur_lifecycle(VFI(), par, epar_entrepreneur, a_grid; z, r, w, tax = NoTax(), a_init = 3.0)
    @test collect(sol_entrepreneur.sim_df.O) == [:E, :E]
    @test sol_entrepreneur.sim_df.a ≈ [3.0, 1.0]
    @test sol_entrepreneur.sim_df.k ≈ [1.0, 1.0]
    @test sol_entrepreneur.sim_df.pi ≈ [1.0, 1.0]
    @test sol_entrepreneur.sim_df.c ≈ [2.5, 1.5]
    @test sol_entrepreneur.sim_df.a_next ≈ [1.0, 0.0]
    @test all(!constrained for constrained in sol_entrepreneur.sim_df.constrained)
    @test sol_entrepreneur.policy_k[a = At(3.0), j = At(0)] ≈ 1.0

    epar_worker = EntrepreneurParams(alpha = 0.5, delta = 1.0, theta = 10.0, f = 1.5)
    sol_worker = solve_entrepreneur_lifecycle(VFI(), par, epar_worker, a_grid; z, r, w, tax = NoTax(), a_init = 3.0)
    @test collect(sol_worker.sim_df.O) == [:W, :W]
    @test sol_worker.sim_df.a ≈ [3.0, 1.0]
    @test sol_worker.sim_df.k == [0.0, 0.0]
    @test sol_worker.sim_df.pi == [0.0, 0.0]
    @test sol_worker.sim_df.c ≈ [2.0, 1.0]
    @test sol_worker.sim_df.a_next ≈ [1.0, 0.0]
    @test all(!constrained for constrained in sol_worker.sim_df.constrained)
    @test sol_worker.policy_k[a = At(3.0), j = At(0)] == 0.0
end

@testset "Micro-test: 2-period constrained entrepreneur" begin
    #=
    Two-Period Constrained Entrepreneur

    Setup:
    A two-period lifecycle with ``m = [0.5, 1.0]``, ``\beta = 1``,
    ``\gamma = 2``, zero labor income, and grid ``a \in 0:0.5:4``.
    Parameters are ``\alpha = 1``, ``\delta = 1``, ``\theta = 1``,
    ``z = 4``, ``r = 0``, ``w = 0``, ``a_0 = 1``, and ``\texttt{annuities} = \texttt{false}``.

    Key equations:
    We choose ``\alpha = 1`` so that profits are linear in wealth under the
    collateral constraint. With constant returns to scale,
    ``k^* = \left(\frac{\alpha z}{r + \delta}\right)^{\frac{1}{1-\alpha}} = \infty``,
    so
    ``k(a) = \min\{k^*, \theta a\} = a``.
    Profits become
    ``\pi(a) = zk - (r+\delta)k = 3a``.

    Derivation:
    Entrepreneurial resources are
    ``R_E(a) = (1+r)a + \pi(a) - f = a + 3a - 0.5 = 4a - 0.5``.
    Hence the effective return on saved wealth is
    ``R'_E(a') = 4``.
    The Euler equation in this constrained branch is therefore
    ``u'(c_0) = \beta(1-m_0)R'_E(a')u'(c_1) = 2u'(c_1)``.
    With CRRA utility and ``\gamma = 2``,
    ``u'(c) = c^{-2}``, so
    ``c_0^{-2} = 2c_1^{-2}``, which implies
    ``c_1 = \sqrt{2}c_0``.

    Using the two budget equations,
    ``c_0 = 3.5 - a'``
    and
    ``c_1 = 4a' - 0.5``,
    and the grid optimum is ``a' = 1``.

    Verbal argument:
    This is the dynamic counterpart to the one-period constrained case. The key
    mechanism is that constrained entrepreneurial scale rises one-for-one with wealth,
    so saving today also relaxes tomorrow's production constraint. That makes the
    effective return exceed ``1+r`` and rationalizes increasing consumption over age.

    Grid verification:
    The discrete objective is maximized at ``a' = 1``, and the test confirms that
    this point dominates the neighboring grid values.
    =#
    y = simple_income_profile(2, 2; y = 0.0, yR = 0.0)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.5, y, β = 1.0, γ = 2.0, a̲ = 0.0)
    a_grid = 0.0:0.5:4.0
    z = 4.0
    r = 0.0
    w = 0.0
    epar = EntrepreneurParams(alpha = 1.0, delta = 1.0, theta = 1.0, f = 0.5)

    @test isinf(SimpleOLG._efficient_entrepreneur_capital(z, epar; r))
    objective(a_next) = par.u(3.5 - a_next) + 0.5 * par.u(4.0 * a_next - 0.5)
    @test objective(1.0) > objective(1.5)
    @test objective(1.0) > objective(2.0)

    sol = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid; z, r, w, tax = NoTax(), a_init = 1.0)
    @test collect(sol.sim_df.O) == [:E, :E]
    @test sol.sim_df.a ≈ [1.0, 1.0]
    @test sol.sim_df.k ≈ [1.0, 1.0]
    @test sol.sim_df.pi ≈ [3.0, 3.0]
    @test sol.sim_df.c ≈ [2.5, 3.5]
    @test sol.sim_df.a_next ≈ [1.0, 0.0]
    @test all(sol.sim_df.constrained)
    @test sol.policy_k[a = At(1.0), j = At(0)] ≈ 1.0
    @test sol.sim_df.c[2] / sol.sim_df.c[1] ≈ 1.4
end

@testset "Entrepreneur profit and tax functions" begin
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 2.0, f = 0.1)
    r = 0.04
    a = 0.75
    z = 1.5

    k_star = (epar.alpha * z / (r + epar.delta))^(1 / (1 - epar.alpha))
    @test entrepreneur_capital(3.0, z, epar; r) ≈ k_star
    @test entrepreneur_capital(0.1, z, epar; r) ≈ epar.theta * 0.1

    k = entrepreneur_capital(a, z, epar; r)
    @test entrepreneur_profit(a, z, epar; r) ≈ z * k^epar.alpha - (r + epar.delta) * k

    @test after_tax_wealth(a, z, epar, NoTax(); r) ≈ after_tax_wealth(a, z, epar, CapitalIncomeTax(0.0); r)
    @test after_tax_wealth(a, z, epar, NoTax(); r) ≈ after_tax_wealth(a, z, epar, WealthTax(0.0); r)

    τ = 0.15
    wealth_gap = after_tax_wealth(a, z, epar, WealthTax(τ); r) - after_tax_wealth(a, 0.0, epar, WealthTax(τ); r)
    capital_gap = after_tax_wealth(a, z, epar, CapitalIncomeTax(τ); r) - after_tax_wealth(a, 0.0, epar, CapitalIncomeTax(τ); r)
    @test wealth_gap ≈ entrepreneur_profit(a, z, epar; r)
    @test capital_gap ≈ (1 - τ) * entrepreneur_profit(a, z, epar; r)
    @test wealth_gap > capital_gap
end

@testset "Entrepreneur lifecycle VFI" begin
    y = simple_income_profile(6, 6; y = 0.35)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.0, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 4.0, length = 401)
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 2.0, f = 0.08)
    r = 0.04
    w = 1.0

    baseline = solve_backward_forward(VFI(), par, a_grid; r, w, a_init = 0.0)
    worker_only = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid; z = 0.0, r, w, tax = NoTax(), a_init = 0.0)

    @test all(worker_only.policy_O .== :W)
    @test worker_only.sim_df.c ≈ baseline.sim_df.c
    @test worker_only.sim_df.a_next ≈ baseline.sim_df.a_next

    z = 1.5
    sol = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid; z, r, w, tax = NoTax(), a_init = 0.0)
    k_star = (epar.alpha * z / (r + epar.delta))^(1 / (1 - epar.alpha))

    high_a_i = findfirst(>=(1.5), a_grid)
    low_a_i = findfirst(>=(0.1), a_grid)
    @test sol.policy_k[a = high_a_i, j = At(0)] ≈ k_star atol = 1e-10
    @test sol.policy_k[a = low_a_i, j = At(0)] ≈ epar.theta * a_grid[low_a_i] atol = 1e-10

    occ_j0 = Int.(collect(sol.policy_O[j = At(0)]) .== :E)
    @test all(diff(occ_j0) .>= 0)

    last = only(sol.sim_df[sol.sim_df.j .== par.J, :])
    @test last.a_next == 0.0
    @test last.c ≈ last.W_after_tax + w * par.y[j = At(par.J)] - (last.O == :E ? epar.f : 0.0)

    sol_cap0 = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid; z, r, w, tax = CapitalIncomeTax(0.0), a_init = 0.0)
    sol_wealth0 = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid; z, r, w, tax = WealthTax(0.0), a_init = 0.0)
    @test sol.value ≈ sol_cap0.value
    @test sol.value ≈ sol_wealth0.value
    @test sol.policy_k ≈ sol_cap0.policy_k
    @test sol.policy_k ≈ sol_wealth0.policy_k
    @test sol.sim_df.a_next ≈ sol_cap0.sim_df.a_next
    @test sol.sim_df.a_next ≈ sol_wealth0.sim_df.a_next
    @test sol.sim_df.c ≈ sol_cap0.sim_df.c
    @test sol.sim_df.c ≈ sol_wealth0.sim_df.c
end

@testset "Entrepreneur multi-type simulation" begin
    y = simple_income_profile(5, 5; y = 0.35)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.0, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 4.0, length = 301)
    z_grid = [0.6, 1.1, 1.6]
    weights = [0.25, 0.5, 0.25]
    r = 0.04
    w = 1.0

    base_epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 1.5, f = 0.08)
    sim = simulate_entrepreneur_types(VFI(), par, base_epar, a_grid; z_grid, weights, r, w, tax = NoTax(), a_init = 0.0)
    @test sim.type_df.weight == weights
    @test all(sim.summary_df.weight_total .≈ sum(weights))

    loose_collateral = simulate_entrepreneur_types(
        VFI(),
        par,
        EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 3.0, f = 0.08),
        a_grid;
        z_grid,
        weights,
        r,
        w,
        tax = NoTax(),
        a_init = 0.0,
    )
    tight_collateral = simulate_entrepreneur_types(
        VFI(),
        par,
        EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 1.0, f = 0.08),
        a_grid;
        z_grid,
        weights,
        r,
        w,
        tax = NoTax(),
        a_init = 0.0,
    )
    @test loose_collateral.summary_df.entrepreneur_share[1] >= tight_collateral.summary_df.entrepreneur_share[1]

    low_cost = simulate_entrepreneur_types(
        VFI(),
        par,
        EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 1.5, f = 0.02),
        a_grid;
        z_grid,
        weights,
        r,
        w,
        tax = NoTax(),
        a_init = 0.0,
    )
    high_cost = simulate_entrepreneur_types(
        VFI(),
        par,
        EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 1.5, f = 0.15),
        a_grid;
        z_grid,
        weights,
        r,
        w,
        tax = NoTax(),
        a_init = 0.0,
    )
    @test low_cost.summary_df.entrepreneur_share[1] >= high_cost.summary_df.entrepreneur_share[1]

    taxes = [CapitalIncomeTax(0.15), WealthTax(0.15)]
    comparison = compare_tax_regimes(VFI(), par, base_epar, a_grid; z_grid, weights, r, w, taxes, a_init = 0.0)

    @test Set(comparison.summary_df.tax_regime) == Set((:capital_income_tax, :wealth_tax))

    summary_entry = comparison.summary_df[comparison.summary_df.j .== 0, :]
    share_capital = only(summary_entry.entrepreneur_share[summary_entry.tax_regime .== :capital_income_tax])
    share_wealth = only(summary_entry.entrepreneur_share[summary_entry.tax_regime .== :wealth_tax])
    @test share_wealth >= share_capital
end
