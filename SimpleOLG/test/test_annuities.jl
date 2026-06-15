@testset "Annuity mode changes lifecycle allocations" begin
    y = simple_income_profile(2, 1; y = 0.0, yR = 0.0)
    a_grid = range(0.0, 1.0, length = 2001)
    expected_annuity = 2 / 3
    expected_no_annuity_c0 = 1 / (1 + sqrt(0.5))
    expected_no_annuity_c1 = sqrt(0.5) / (1 + sqrt(0.5))

    par_annuity = get_par(; demo = :perpetual_youth, mm = 0.5, annuities = true, y, β = 1.0, γ = 2.0, a̲ = 0.0)
    par_no_annuity = get_par(; demo = :perpetual_youth, mm = 0.5, annuities = false, y, β = 1.0, γ = 2.0, a̲ = 0.0)

    for solver in (VFI(), EGM())
        sol_annuity = solve_backward_forward(solver, par_annuity, a_grid; r = 0.0, w = 1.0, a_init = 1.0)
        sol_no_annuity = solve_backward_forward(solver, par_no_annuity, a_grid; r = 0.0, w = 1.0, a_init = 1.0)

        @test sol_annuity.sim_df.c ≈ [expected_annuity, expected_annuity] atol = 2e-3
        @test sol_annuity.sim_df.a_next ≈ [expected_annuity, 0.0] atol = 2e-3

        @test sol_no_annuity.sim_df.c[1] ≈ expected_no_annuity_c0 atol = 2e-3
        @test sol_no_annuity.sim_df.c[2] ≈ expected_no_annuity_c1 atol = 2e-3
        @test sol_no_annuity.sim_df.a_next ≈ [expected_no_annuity_c1, 0.0] atol = 2e-3
    end
end

@testset "Entrepreneur solvers reject annuity mode" begin
    y = simple_income_profile(2, 1; y = 0.0, yR = 0.0)
    par = get_par(; demo = :perpetual_youth, mm = 0.5, annuities = true, y, β = 1.0, γ = 2.0, a̲ = 0.0)
    epar = EntrepreneurParams(alpha = 0.5, delta = 0.0, theta = 1.0, f = 0.0)
    a_grid = 0.0:0.1:1.0

    @test_throws ArgumentError solve_entrepreneur_lifecycle(VFI(), par, epar, a_grid; z = 1.0, r = 0.0, w = 0.0, tax = NoTax(), a_init = 0.0)
    @test_throws ArgumentError solve_entrepreneur_lifecycle(EGM(), par, epar, a_grid; z = 1.0, r = 0.0, w = 0.0, tax = NoTax(), a_init = 0.0)
end
