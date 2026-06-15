@testset "Entrepreneur EGM return formulas" begin
    r = 0.04

    @test SimpleOLG.effective_return_worker(NoTax(); r) ≈ 1.04
    @test SimpleOLG.effective_return_worker(CapitalIncomeTax(0.15); r) ≈ 1.034
    @test SimpleOLG.effective_return_worker(WealthTax(0.10); r) ≈ 0.94

    constrained = EntrepreneurParams(alpha = 1.0, delta = 1.0, theta = 1.0, f = 0.0)
    @test SimpleOLG.effective_return_entrepreneur(1.0, 4.0, constrained, NoTax(); r = 0.0) ≈ 4.0
    @test SimpleOLG.effective_return_entrepreneur(1.0, 4.0, constrained, CapitalIncomeTax(0.25); r = 0.0) ≈ 3.25
    @test SimpleOLG.effective_return_entrepreneur(1.0, 4.0, constrained, WealthTax(0.10); r = 0.0) ≈ 3.9

    unconstrained = EntrepreneurParams(alpha = 0.5, delta = 0.0, theta = 10.0, f = 0.0)
    @test SimpleOLG.effective_return_entrepreneur(1.0, 2.0, unconstrained, NoTax(); r = 0.5) ≈ SimpleOLG.effective_return_worker(NoTax(); r = 0.5)
end

@testset "Entrepreneur EGM rejects negative grids" begin
    y = simple_income_profile(4, 4; y = 0.35)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.0, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 2.0, f = 0.08)
    a_grid = range(-1.0, 4.0, length = 41)

    @test_throws ArgumentError solve_entrepreneur_lifecycle(EGM(), par, epar, a_grid; z = 1.5, r = 0.04, w = 1.0, tax = NoTax(), a_init = 0.0)
end

@testset "Entrepreneur EGM matches worker EGM when z = 0" begin
    y = simple_income_profile(6, 6; y = 0.35)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.0, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 4.0, length = 201)
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 2.0, f = 0.08)
    r = 0.04
    w = 1.0

    baseline = solve_backward_forward(EGM(), par, a_grid; r, w, a_init = 0.0)
    entrepreneur = solve_entrepreneur_lifecycle(EGM(), par, epar, a_grid; z = 0.0, r, w, tax = NoTax(), a_init = 0.0)

    @test all(entrepreneur.policy_O .== :W)
    @test entrepreneur.sim_df.c ≈ baseline.sim_df.c
    @test entrepreneur.sim_df.a_next ≈ baseline.sim_df.a_next
end

@testset "Entrepreneur EGM simulation wrappers" begin
    y = simple_income_profile(5, 5; y = 0.35)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.0, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 4.0, length = 201)
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 1.5, f = 0.08)
    z_grid = [0.6, 1.1, 1.6]
    weights = [0.25, 0.5, 0.25]
    r = 0.04
    w = 1.0

    vfi = simulate_entrepreneur_types(VFI(), par, epar, a_grid; z_grid, weights, r, w, tax = NoTax(), a_init = 0.0)
    egm = simulate_entrepreneur_types(EGM(), par, epar, a_grid; z_grid, weights, r, w, tax = NoTax(), a_init = 0.0)

    @test propertynames(egm) == propertynames(vfi)
    @test propertynames(first(egm.type_results).sol) == propertynames(first(vfi.type_results).sol)
    @test names(egm.sim_df) == names(vfi.sim_df)
    @test names(egm.summary_df) == names(vfi.summary_df)
    @test egm.type_df.weight == weights
    @test all(egm.summary_df.weight_total .≈ sum(weights))

    taxes = [CapitalIncomeTax(0.15), WealthTax(0.15)]
    vfi_comparison = compare_tax_regimes(VFI(), par, epar, a_grid; z_grid, weights, r, w, taxes, a_init = 0.0)
    egm_comparison = compare_tax_regimes(EGM(), par, epar, a_grid; z_grid, weights, r, w, taxes, a_init = 0.0)

    @test propertynames(egm_comparison) == propertynames(vfi_comparison)
    @test names(egm_comparison.sim_df) == names(vfi_comparison.sim_df)
    @test names(egm_comparison.summary_df) == names(vfi_comparison.summary_df)
    @test Set(egm_comparison.summary_df.tax_regime) == Set((:capital_income_tax, :wealth_tax))
end

@testset "Entrepreneur EGM cross-validation" begin
    y = simple_income_profile(6, 6; y = 0.35)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.0, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 4.0, length = 401)
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 2.0, f = 0.08)
    z = 1.5
    r = 0.04
    w = 1.0

    threshold_index(policy_O_j) = something(findfirst(==(:E), collect(policy_O_j)), length(policy_O_j) + 1)

    for tax in (NoTax(), CapitalIncomeTax(0.15), WealthTax(0.15))
        vfi = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid; z, r, w, tax, a_init = 1.0)
        egm = solve_entrepreneur_lifecycle(EGM(), par, epar, a_grid; z, r, w, tax, a_init = 1.0)

        @test all(vfi.sim_df.O .== egm.sim_df.O)
        @test maximum(abs.(vfi.sim_df.c .- egm.sim_df.c)) < 6e-3
        @test maximum(abs.(vfi.sim_df.a_next .- egm.sim_df.a_next)) < 7e-3
        @test all(egm.sim_df.a_next .>= par.a̲ - 1e-12)
        @test egm.sim_df.a_next[end] >= 0.0
        @test abs(threshold_index(vfi.policy_O[j = At(0)]) - threshold_index(egm.policy_O[j = At(0)])) <= 2

        lhs = egm.sim_df.c .+ (1 .- egm.sim_df.m) .* egm.sim_df.a_next
        rhs = egm.sim_df.W_after_tax .+ egm.sim_df.w .* egm.sim_df.y .+ egm.sim_df.inheritance .- ifelse.(egm.sim_df.O .== :E, epar.f, 0.0)
        @test all(isapprox.(lhs, rhs; atol = 1e-10, rtol = 1e-10))
    end
end

@testset "Entrepreneur EGM cross-validation with mortality" begin
    y = simple_income_profile(6, 6; y = 0.35)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.1, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 4.0, length = 401)
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 2.0, f = 0.08)
    z = 1.5
    r = 0.04
    w = 1.0

    for tax in (NoTax(), CapitalIncomeTax(0.15), WealthTax(0.15))
        vfi = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid; z, r, w, tax, a_init = 1.0)
        egm = solve_entrepreneur_lifecycle(EGM(), par, epar, a_grid; z, r, w, tax, a_init = 1.0)

        @test all(vfi.sim_df.O .== egm.sim_df.O)
        @test maximum(abs.(vfi.sim_df.c .- egm.sim_df.c)) < 7e-3
        @test maximum(abs.(vfi.sim_df.a_next .- egm.sim_df.a_next)) < 7e-3
    end
end

@testset "Entrepreneur EGM with inheritances" begin
    y = simple_income_profile(5, 5; y = 0.35)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.0, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 4.0, length = 301)
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 2.0, f = 0.08)
    inh_j = [0.2, 0.1, 0.05, 0.0, 0.0]
    z = 1.5
    r = 0.04
    w = 1.0

    vfi = solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid; z, r, w, tax = NoTax(), a_init = 0.5, inh_j)
    egm = solve_entrepreneur_lifecycle(EGM(), par, epar, a_grid; z, r, w, tax = NoTax(), a_init = 0.5, inh_j)

    @test egm.sim_df.inheritance ≈ inh_j
    @test all(vfi.sim_df.O .== egm.sim_df.O)
    @test maximum(abs.(vfi.sim_df.c .- egm.sim_df.c)) < 1e-2
    @test maximum(abs.(vfi.sim_df.a_next .- egm.sim_df.a_next)) < 1e-2
end

@testset "EGM bequest fixed-point" begin
    y = simple_income_profile(5, 5; y = 0.35)
    par = get_par(; demo = :perpetual_youth, annuities = false, mm = 0.12, y, β = 0.96, γ = 2.0, a̲ = 0.0)
    a_grid = range(0.0, 4.0, length = 201)
    epar = EntrepreneurParams(alpha = 0.3, delta = 0.4, theta = 2.0, f = 0.05)
    z_grid = [0.8, 1.6]
    π_z = [0.5, 0.5]
    r = 0.04
    w = 1.0

    F_zero = DimVector(zeros(par.J + 1), Dim{:j}(0:par.J), name = :F)
    zero_flow = solve_entrepreneur_with_bequests(
        EGM(),
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
        baseline = solve_entrepreneur_lifecycle(EGM(), par, epar, a_grid; z, r, w, tax = NoTax(), a_init = 0.0)
        with_bequests = zero_flow.solutions[z]

        @test with_bequests.sim_df.c ≈ baseline.sim_df.c
        @test with_bequests.sim_df.a_next ≈ baseline.sim_df.a_next
        @test with_bequests.sim_df.inheritance ≈ zeros(nrow(with_bequests.sim_df))
    end

    F_birth = DimVector(vcat(1.0, zeros(par.J)), Dim{:j}(0:par.J), name = :F)
    bpar = BequestParams(P_z = [1.0 0.0; 0.0 1.0], π_z = π_z, F = F_birth)

    vfi = solve_entrepreneur_with_bequests(VFI(), par, epar, bpar, a_grid; z_grid, r, w, tax = NoTax(), a_init = 0.0, tol = 1e-9)
    egm = solve_entrepreneur_with_bequests(EGM(), par, epar, bpar, a_grid; z_grid, r, w, tax = NoTax(), a_init = 0.0, tol = 1e-9)

    @test vfi.iterations < 100
    @test egm.iterations < 100
    @test maximum(abs.(vfi.inheritances_z .- egm.inheritances_z)) < 1e-2
    @test maximum(abs.(vfi.bequests_z .- egm.bequests_z)) < 1e-2
end
