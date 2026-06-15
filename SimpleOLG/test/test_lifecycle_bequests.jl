@testset "Unified driver: worker delegation matches solve_worker_with_bequests" begin
    y = simple_income_profile(6, 4; y = 0.35, yR = 0.05)
    par = get_par(; demo = :perpetual_youth, mm = 0.12, annuities = true, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 5.0, length = 201)
    r, w = 0.03, 1.0

    F_birth = DimVector([1.0; zeros(par.J)], Dim{:j}(0:par.J), name = :F)
    bpar = BequestParams(P_z = [1.0 0.0; 0.0 1.0], π_z = [0.5, 0.5], F = F_birth)
    type_grid = [LaborProductivity(0.8), LaborProductivity(1.2)]

    via_worker = solve_worker_with_bequests(EGM(), par, bpar, a_grid;
        type_grid, r, w, tol = 1e-10)
    via_unified = solve_lifecycle_with_bequests(EGM(), par, bpar, a_grid;
        type_grid, epar = nothing, r, w, tol = 1e-10)

    @test via_worker.sim_df.a_next ≈ via_unified.sim_df.a_next
    @test via_worker.sim_df.c ≈ via_unified.sim_df.c
    @test via_worker.inheritances_z ≈ via_unified.inheritances_z
    @test via_worker.iterations == via_unified.iterations
end

@testset "Unified driver: entrepreneur delegation matches solve_entrepreneur_with_bequests" begin
    y = simple_income_profile(7, 7; y = 0.35)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.12, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 6.0, length = 201)
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 2.0, f = 0.05)
    r, w = 0.04, 1.0

    F_birth = DimVector([1.0; zeros(par.J)], Dim{:j}(0:par.J), name = :F)
    bpar = BequestParams(P_z = [1.0 0.0; 0.0 1.0], π_z = [0.5, 0.5], F = F_birth)
    z_grid = [0.8, 1.6]

    via_wrapper = solve_entrepreneur_with_bequests(VFI(), par, epar, bpar, a_grid;
        z_grid, r, w, tax = NoTax(), a_init = 0.0, tol = 1e-9)
    type_grid = [EntrepreneurSkill(z) for z in z_grid]
    via_unified = solve_lifecycle_with_bequests(VFI(), par, bpar, a_grid;
        type_grid, epar, r, w, tax = NoTax(), a_init = 0.0, tol = 1e-9, λ_inherit = 1.0)

    @test via_wrapper.inheritances_z ≈ via_unified.inheritances_z
    @test via_wrapper.bequests_z ≈ via_unified.bequests_z
    @test via_wrapper.sim_df.a_next ≈ via_unified.sim_df.a_next
    @test via_wrapper.sim_df.c ≈ via_unified.sim_df.c
end

@testset "Entrepreneur solver: new z_labor / z_entrepreneur API matches scalar z" begin
    y = simple_income_profile(5, 5; y = 0.4)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.15, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 4.0, length = 201)
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 2.0, f = 0.05)

    z = 1.4
    via_scalar = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid;
        z, r = 0.04, w = 1.0, a_init = 0.0)
    via_split = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid;
        z_labor = 1.0, z_entrepreneur = z, r = 0.04, w = 1.0, a_init = 0.0)

    @test via_scalar.sim_df.c ≈ via_split.sim_df.c
    @test via_scalar.sim_df.a_next ≈ via_split.sim_df.a_next
    @test via_scalar.sim_df.O == via_split.sim_df.O
end

@testset "aggregate_wealth: top-10% share on a synthetic two-point distribution" begin
    sim_df = DataFrame(a = [1.0, 10.0], weight = [0.9, 0.1])
    agg = aggregate_wealth(sim_df)
    @test agg.top10_share ≈ (10.0 * 0.1) / (1.0 * 0.9 + 10.0 * 0.1)
    @test 0 ≤ agg.gini ≤ 1
    @test agg.mean ≈ 1.0 * 0.9 + 10.0 * 0.1
end

@testset "aggregate_wealth: equal wealth → top10_share = 0.1" begin
    sim_df = DataFrame(a = fill(1.0, 10), weight = fill(0.1, 10))
    agg = aggregate_wealth(sim_df)
    @test agg.top10_share ≈ 0.1 atol = 1e-10
    @test agg.gini ≈ 0.0 atol = 1e-10
end
