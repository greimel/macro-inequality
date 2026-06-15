@testset "Warm-glow bequests: 1/3 split" begin
    y = simple_income_profile(2, 1; y = 4.5, yR = 0.0)
    par = get_par(; demo = :perpetual_youth, mm = 0.0, y, β = 1.0, γ = 2.0, phi = 1.0, b_floor = 0.0, a̲ = 0.0)
    a_grid = 0.0:0.5:5.0

    objective(a_next) = par.u(4.5 - a_next) + 2 * par.u(a_next / 2)
    @test objective(3.0) > objective(2.5)
    @test objective(3.0) > objective(3.5)

    sol_vfi = solve_backward_forward(VFI(), par, a_grid; r = 0.0, w = 1.0, a_init = 0.0)
    sol_egm = solve_backward_forward(EGM(), par, a_grid; r = 0.0, w = 1.0, a_init = 0.0)

    @test sol_vfi.sim_df.c ≈ [1.5, 1.5]
    @test sol_vfi.sim_df.a_next ≈ [3.0, 1.5]
    @test sol_egm.sim_df.c ≈ [1.5, 1.5]
    @test sol_egm.sim_df.a_next ≈ [3.0, 1.5]

    π_j = pmf(par.m)
    @test collect(π_j) ≈ [0.5, 0.5]
    @test only(compute_bequests(sol_vfi.sim_df, par)) ≈ 0.75

    F_match = DimVector([0.5, 0.5], Dim{:j}(0:1), name = :F)
    F_birth = DimVector([1.0, 0.0], Dim{:j}(0:1), name = :F)

    inh_match = distribute_inheritances([0.75], BequestParams(P_z = [1.0;;], π_z = [1.0], F = F_match), par)
    @test inheritance_income(1, 0, inh_match) ≈ 0.75
    @test inheritance_income(1, 1, inh_match) ≈ 0.75

    inh_birth = distribute_inheritances([0.75], BequestParams(P_z = [1.0;;], π_z = [1.0], F = F_birth), par)
    @test inheritance_income(1, 0, inh_birth) ≈ 1.5
    @test inheritance_income(1, 1, inh_birth) ≈ 0.0
end

@testset "Warm-glow bequests: positive bequest floor" begin
    y = simple_income_profile(2, 1; y = 0.0, yR = 0.0)
    par = get_par(; demo = :perpetual_youth, mm = 0.0, y, β = 1.0, γ = 2.0, phi = 1.0, b_floor = 1.0, a̲ = 0.0)

    allocation = SimpleOLG.terminal_allocation(4.5, par)

    @test allocation.c ≈ 2.75
    @test allocation.a_next ≈ 1.75
    @test allocation.c ≈ allocation.a_next + par.b_floor
    @test allocation.a_next < allocation.c
end

@testset "Micro-test: 2-period bequest pipeline" begin
    #=
    Two-Period Bequest Pipeline

    Setup:
    A two-period accidental-bequest environment with one type, no entrepreneurial
    activity (``z = 0``), and worker income profile ``y = [4.5, 0]``.
    Parameters are ``\gamma = 2``, ``\beta = 1``, ``r = 0``, ``w = 1``,
    ``m = [0.5, 1.0]``, ``a_0 = 0``, and ``a \in 0:0.5:5``.
    This test jointly covers the lifecycle solver, ``compute_bequests``, and
    ``distribute_inheritances``.

    Key equations:
    The worker problem satisfies
    ``c_0 = 4.5 - 0.5a'``
    and
    ``c_1 = a'``.
    The period-0 objective is
    ``u(c_0) + \beta(1-m_0)u(c_1) = u(c_0) + 0.5u(c_1)``.
    In the stationary cross-section, age masses are
    ``\pi_j = [2/3, 1/3]``.
    Accidental bequests are averaged with weights
    ``\pi_j m_j``.

    Derivation:
    The Euler condition again implies equal consumption across the two ages in this
    setup, so
    ``4.5 - 0.5a' = a'``.
    Hence ``a' = 3`` and
    ``c_0 = c_1 = 3``.

    The stationary age distribution is
    ``\pi_j = [2/3, 1/3]``.
    Therefore accidental-bequest weights are
    ``\pi_j m_j = [2/3 \cdot 0.5, 1/3 \cdot 1] = [1/3, 1/3]``.
    Aggregate per-capita bequests are
    ``B = \sum_j \pi_j m_j a'_{j} = (1/3)\cdot 3 + (1/3)\cdot 0 = 1``.

    Verbal argument:
    ``compute_bequests`` is computing the bequest flow seen by a stationary
    cross-section, not by a single birth cohort. That is why the relevant weights are
    the stationary age masses multiplied by mortality. Once aggregate bequests are
    known, ``distribute_inheritances`` simply redistributes them across ages according
    to the exogenous inheritance profile ``F``.

    Expected outcomes:
    If ``F = [2/3, 1/3]``, inheritances are uniform per capita:
    ``\text{inh} = [1, 1]``.
    If ``F = [1, 0]``, all inheritances arrive at birth:
    ``\text{inh} = [1.5, 0]``.
    If ``F = [0, 1]``, all inheritances arrive in the second period:
    ``\text{inh} = [0, 3]``.

    Grid verification:
    The discrete objective
    ``u(4.5 - 0.5a') + 0.5u(a')``
    is strictly higher at ``a' = 3`` than at the neighboring grid points
    ``a' = 2.5`` and ``a' = 3.5``.

    Note:
    There is no warm-glow bequest motive in the current model, so the bequest flow is
    purely accidental and cannot be targeted independently of optimal saving.
    =#
    y = simple_income_profile(2, 1; y = 4.5, yR = 0.0)
    par = get_par(; demo = :perpetual_youth, mm = 0.5, annuities = true, y, β = 1.0, γ = 2.0, a̲ = 0.0)
    a_grid = 0.0:0.5:5.0

    objective(a_next) = par.u(4.5 - 0.5 * a_next) + 0.5 * par.u(a_next)
    @test objective(3.0) > objective(2.5)
    @test objective(3.0) > objective(3.5)

    sol = solve_backward_forward(VFI(), par, a_grid; r = 0.0, w = 1.0, a_init = 0.0)
    @test sol.sim_df.c ≈ [3.0, 3.0]
    @test sol.sim_df.a_next ≈ [3.0, 0.0]

    π_j = pmf(par.m)
    @test collect(π_j) ≈ [2 / 3, 1 / 3]
    @test only(compute_bequests(sol.sim_df, par)) ≈ 1.0

    F_match = DimVector([2 / 3, 1 / 3], Dim{:j}(0:1), name = :F)
    F_birth = DimVector([1.0, 0.0], Dim{:j}(0:1), name = :F)
    F_old = DimVector([0.0, 1.0], Dim{:j}(0:1), name = :F)

    inh_match = distribute_inheritances([1.0], BequestParams(P_z = [1.0;;], π_z = [1.0], F = F_match), par)
    @test inheritance_income(1, 0, inh_match) ≈ 1.0
    @test inheritance_income(1, 1, inh_match) ≈ 1.0

    inh_birth = distribute_inheritances([1.0], BequestParams(P_z = [1.0;;], π_z = [1.0], F = F_birth), par)
    @test inheritance_income(1, 0, inh_birth) ≈ 1.5
    @test inheritance_income(1, 1, inh_birth) ≈ 0.0

    inh_old = distribute_inheritances([1.0], BequestParams(P_z = [1.0;;], π_z = [1.0], F = F_old), par)
    @test inheritance_income(1, 0, inh_old) ≈ 0.0
    @test inheritance_income(1, 1, inh_old) ≈ 3.0
end

@testset "Bequest accounting" begin
    y = simple_income_profile(6, 6; y = 0.35)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.15, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 5.0, length = 401)
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 2.0, f = 0.05)
    r = 0.04
    w = 1.0

    sol = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid; z = 1.5, r, w, tax = NoTax(), a_init = 0.0)
    bequests_z = compute_bequests(sol.sim_df, par)
    expected = sum(Float64.(collect(pmf(par.m) .* par.m)) .* sol.sim_df.a_next)
    @test only(bequests_z) ≈ expected

    F_birth = DimVector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], Dim{:j}(0:par.J), name = :F)
    F_age2 = DimVector([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], Dim{:j}(0:par.J), name = :F)
    π_z = [0.4, 0.6]
    B_z = [1.0, 2.0]
    π_j = pmf(par.m)

    birth_only = distribute_inheritances(B_z, BequestParams(P_z = [1.0 0.0; 0.0 1.0], π_z = π_z, F = F_birth), par)
    @test inheritance_income(1, 0, birth_only) ≈ B_z[1] / π_j[j = At(0)]
    @test inheritance_income(2, 0, birth_only) ≈ B_z[2] / π_j[j = At(0)]
    @test all(isapprox.(collect(birth_only[type = At(1), j = At(1:par.J)]), 0.0; atol = 1e-12, rtol = 0.0))

    age2_only = distribute_inheritances(B_z, BequestParams(P_z = [1.0 0.0; 0.0 1.0], π_z = π_z, F = F_age2), par)
    @test inheritance_income(1, 2, age2_only) ≈ B_z[1] / π_j[j = At(2)]
    @test inheritance_income(2, 2, age2_only) ≈ B_z[2] / π_j[j = At(2)]
    @test all(isapprox.(collect(age2_only[type = At(2), j = At((0:par.J)[[1, 2, 4, 5, 6]])]), 0.0; atol = 1e-12, rtol = 0.0))

    pooled = sum(B_z .* π_z)
    mixed = distribute_inheritances(B_z, BequestParams(P_z = [0.4 0.6; 0.4 0.6], π_z = π_z, F = F_birth), par)
    @test inheritance_income(1, 0, mixed) ≈ pooled / π_j[j = At(0)]
    @test inheritance_income(2, 0, mixed) ≈ pooled / π_j[j = At(0)]
end

@testset "Entrepreneur fixed point with bequests" begin
    y = simple_income_profile(7, 7; y = 0.35)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.12, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 6.0, length = 401)
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 2.0, f = 0.05)
    z_grid = [0.8, 1.6]
    π_z = [0.5, 0.5]
    r = 0.04
    w = 1.0

    F_zero = DimVector(zeros(par.J + 1), Dim{:j}(0:par.J), name = :F)
    zero_flow = solve_entrepreneur_with_bequests(
        VFI(),
        par,
        epar,
        BequestParams(P_z = [1.0 0.0; 0.0 1.0], π_z = π_z, F = F_zero),
        a_grid;
        z_grid,
        r,
        w,
        tax = NoTax(),
        a_init = 0.0,
    )

    @test zero_flow.iterations <= 2
    @test zero_flow.inheritances_z ≈ zeros(length(z_grid))

    for z in z_grid
        baseline = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid; z, r, w, tax = NoTax(), a_init = 0.0)
        with_bequests = zero_flow.solutions[z]

        @test with_bequests.sim_df.c ≈ baseline.sim_df.c
        @test with_bequests.sim_df.a_next ≈ baseline.sim_df.a_next
        @test with_bequests.sim_df.inheritance ≈ zeros(nrow(with_bequests.sim_df))
    end

    F_birth = DimVector([1.0; zeros(par.J)], Dim{:j}(0:par.J), name = :F)
    persistent = solve_entrepreneur_with_bequests(
        VFI(),
        par,
        epar,
        BequestParams(P_z = [1.0 0.0; 0.0 1.0], π_z = π_z, F = F_birth),
        a_grid;
        z_grid,
        r,
        w,
        tax = NoTax(),
        a_init = 0.0,
        tol = 1e-9,
    )
    mixing = solve_entrepreneur_with_bequests(
        VFI(),
        par,
        epar,
        BequestParams(P_z = [0.5 0.5; 0.5 0.5], π_z = π_z, F = F_birth),
        a_grid;
        z_grid,
        r,
        w,
        tax = NoTax(),
        a_init = 0.0,
        tol = 1e-9,
    )

    @test persistent.iterations < 100
    @test mixing.iterations < 100
    @test persistent.inheritances_z ≈ persistent.bequests_z
    @test mixing.inheritances_z[1] ≈ mixing.inheritances_z[2]

    taxes = [CapitalIncomeTax(0.15), WealthTax(0.15)]
    comparison = compare_tax_regimes_with_bequests(
        VFI(),
        par,
        epar,
        BequestParams(P_z = [1.0 0.0; 0.0 1.0], π_z = π_z, F = F_birth),
        a_grid;
        z_grid,
        r,
        w,
        taxes,
        a_init = 0.0,
        tol = 1e-9,
    )

    @test Set(comparison.summary_df.tax_regime) == Set((:capital_income_tax, :wealth_tax))

    capital = only(filter(result -> result.tax_regime == :capital_income_tax, comparison.results))
    wealth = only(filter(result -> result.tax_regime == :wealth_tax, comparison.results))
    @test all(result.out.iterations < 100 for result in comparison.results)
    @test maximum(abs.(wealth.out.inheritances_z .- capital.out.inheritances_z)) > 1e-8
end
