@testset "V1: degenerate single type reproduces baseline (EGM)" begin
    y = simple_income_profile(6, 4; y = 0.35, yR = 0.05)
    par = get_par(; demo = :perpetual_youth, mm = 0.12, annuities = true, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 5.0, length = 201)
    r = 0.03
    w = 1.0

    F_zero = DimVector(zeros(par.J + 1), Dim{:j}(0:par.J), name = :F)
    bpar = BequestParams(P_z = [1.0;;], π_z = [1.0], F = F_zero)

    result = solve_worker_with_bequests(EGM(), par, bpar, a_grid; type_grid = [PermanentType()], r, w)
    baseline = solve_backward_forward(EGM(), par, a_grid; r, w, a_init = first(a_grid))

    @test result.sim_df.c ≈ baseline.sim_df.c
    @test result.sim_df.a_next ≈ baseline.sim_df.a_next
    @test result.inheritances_z ≈ zeros(1)
end

@testset "V1: degenerate single type reproduces baseline (VFI)" begin
    y = simple_income_profile(4, 2; y = 1.0, yR = 0.01)
    par = get_par(; demo = :perpetual_youth, mm = 0.2, annuities = true, y, β = 0.98, γ = 1.5, a̲ = 0.0)
    a_grid = 0.0:0.25:4.0
    r = 0.02
    w = 1.0

    F_zero = DimVector(zeros(par.J + 1), Dim{:j}(0:par.J), name = :F)
    bpar = BequestParams(P_z = [1.0;;], π_z = [1.0], F = F_zero)

    result = solve_worker_with_bequests(VFI(), par, bpar, a_grid; type_grid = [PermanentType()], r, w)
    baseline = solve_backward_forward(VFI(), par, a_grid; r, w, a_init = first(a_grid))

    @test result.sim_df.c ≈ baseline.sim_df.c
    @test result.sim_df.a_next ≈ baseline.sim_df.a_next
end

@testset "V3: two-period closed-form with warm-glow bequests" begin
    # J = 1 (two ages). β(1+r) = 1, phi = 1, γ = 2.
    # Closed form: a_2(z) = w z [(1+r) y_0 + y_1] / D(F, r),
    # with D = (3+r) − (1+r)² F_0 − (1+r) F_1.
    r = 0.03
    w = 1.0
    y = DimArray([1.0, 1.0], Dim{:j}(0:1), name = :y)
    par = get_par(; demo = :perpetual_youth, mm = 0.0, annuities = true,
                   y, β = 1 / (1 + r), γ = 2.0, phi = 1.0, a̲ = 0.0)
    a_grid = range(0.0, 2.0, length = 4001)

    F = DimVector([0.8, 0.2], Dim{:j}(0:1), name = :F)

    D = (3 + r) - (1 + r)^2 * 0.8 - (1 + r) * 0.2
    a2_1 = w * ((1 + r) * 1.0 + 1.0) / D

    # Single type: fixed point reproduces the closed form.
    bpar_single = BequestParams(P_z = [1.0;;], π_z = [1.0], F = F)
    single = solve_worker_with_bequests(EGM(), par, bpar_single, a_grid;
        type_grid = [LaborProductivity(1.0)], r, w, tol = 1e-12, maxiter = 1000, λ_inherit = 0.5)
    @test single.sim_df.a_next[end] ≈ a2_1 atol = 1e-6

    # Two absorbing types: per-type closed form, bequest ratio = z_H / z_L = 3.
    z_L, z_H = 0.5, 1.5
    bpar_abs = BequestParams(P_z = [1.0 0.0; 0.0 1.0], π_z = [0.5, 0.5], F = F)
    abs_res = solve_worker_with_bequests(EGM(), par, bpar_abs, a_grid;
        type_grid = [LaborProductivity(z_L), LaborProductivity(z_H)],
        r, w, tol = 1e-12, maxiter = 1000, λ_inherit = 0.5)

    a_next_L = abs_res.sim_df[(abs_res.sim_df.type .== 1) .& (abs_res.sim_df.j .== 1), :a_next][1]
    a_next_H = abs_res.sim_df[(abs_res.sim_df.type .== 2) .& (abs_res.sim_df.j .== 1), :a_next][1]
    @test a_next_L ≈ z_L * a2_1 atol = 1e-6
    @test a_next_H ≈ z_H * a2_1 atol = 1e-6
    @test abs_res.bequests_z[2] / abs_res.bequests_z[1] ≈ 3.0 atol = 1e-6
end

@testset "V4: absorbing types yield higher wealth dispersion than uniform mixing" begin
    y = simple_income_profile(6, 4; y = 0.35, yR = 0.05)
    par = get_par(; demo = :perpetual_youth, mm = 0.12, annuities = true, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 8.0, length = 201)
    type_grid = [LaborProductivity(0.5), LaborProductivity(1.5)]
    π_z = [0.5, 0.5]
    r = 0.03
    w = 1.0

    F_birth = DimVector([1.0; zeros(par.J)], Dim{:j}(0:par.J), name = :F)

    absorbing = solve_worker_with_bequests(
        EGM(), par,
        BequestParams(P_z = [1.0 0.0; 0.0 1.0], π_z = π_z, F = F_birth),
        a_grid; type_grid, r, w, tol = 1e-9,
    )
    mixing = solve_worker_with_bequests(
        EGM(), par,
        BequestParams(P_z = [0.5 0.5; 0.5 0.5], π_z = π_z, F = F_birth),
        a_grid; type_grid, r, w, tol = 1e-9,
    )

    @test absorbing.iterations < 200
    @test mixing.iterations < 200

    # Absorbing types: each type inherits its own bequests.
    @test absorbing.inheritances_z ≈ absorbing.bequests_z atol = 1e-8
    # Uniform mixing: both types receive the same inheritance.
    @test mixing.inheritances_z[1] ≈ mixing.inheritances_z[2] atol = 1e-8

    absorbing_spread = abs(absorbing.inheritances_z[2] - absorbing.inheritances_z[1])
    mixing_spread = abs(mixing.inheritances_z[2] - mixing.inheritances_z[1])
    @test absorbing_spread > mixing_spread

    # aggregate_wealth returns sensible numbers on pooled sim_df.
    agg_abs = aggregate_wealth(absorbing.sim_df)
    agg_mix = aggregate_wealth(mixing.sim_df)
    @test agg_abs.mean > 0
    @test 0 ≤ agg_abs.gini ≤ 1
    @test agg_abs.gini > agg_mix.gini
end

@testset "Worker fixed-point: z-scaling shifts savings" begin
    y = simple_income_profile(4, 3; y = 0.5, yR = 0.01)
    par = get_par(; demo = :perpetual_youth, mm = 0.15, annuities = true, y, β = 0.95, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 6.0, length = 201)
    r = 0.03
    w = 1.0

    F_birth = DimVector([1.0; zeros(par.J)], Dim{:j}(0:par.J), name = :F)
    bpar = BequestParams(P_z = [1.0 0.0; 0.0 1.0], π_z = [0.5, 0.5], F = F_birth)

    type_grid = [LaborProductivity(0.8), LaborProductivity(1.2)]
    result = solve_worker_with_bequests(EGM(), par, bpar, a_grid; type_grid, r, w, tol = 1e-9)

    high_z_a_next = result.sim_df[result.sim_df.type .== 2, :a_next]
    low_z_a_next  = result.sim_df[result.sim_df.type .== 1, :a_next]
    @test sum(high_z_a_next) > sum(low_z_a_next)
    @test result.bequests_z[2] > result.bequests_z[1]
end

@testset "V5: terminal CRRA knife-edge is caught pre-flight" begin
    y = simple_income_profile(4, 2; y = 1.0, yR = 0.0)
    par = get_par(; demo = :perpetual_youth, mm = 0.2, annuities = true, y, β = 0.98, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 4.0, length = 101)
    F_zero = DimVector(zeros(par.J + 1), Dim{:j}(0:par.J), name = :F)
    bpar = BequestParams(P_z = [1.0;;], π_z = [1.0], F = F_zero)

    # y[J] = 0, F[J] = 0, amin = 0 → throws.
    @test_throws ArgumentError solve_worker_with_bequests(
        EGM(), par, bpar, a_grid; type_grid = [PermanentType()], r = 0.02, w = 1.0)

    # Positive case: nudge amin > 0 lets the solver proceed.
    a_grid_safe = range(1e-3, 4.0, length = 101)
    out = solve_worker_with_bequests(
        EGM(), par, bpar, a_grid_safe; type_grid = [PermanentType()], r = 0.02, w = 1.0,
        a_init = first(a_grid_safe), tol = 1e-9)
    @test out.iterations ≥ 1
end
