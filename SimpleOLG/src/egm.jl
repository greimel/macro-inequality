function _solve_backward_forward_(::EGM, par, grid; price_paths, init_state, j_init = 0, t_born = 0, z = 1.0, inh_j = nothing)
    (; J, y, m, a̲) = par
    (; rs, ws) = price_paths
    a_init = init_state

    a_dim = Dim{:a}(grid)
    j_dim = Dim{:j}(0:J)

    inh = isnothing(inh_j) ? DimArray(zeros(Float64, J + 1), j_dim) :
          DimArray(Float64.(collect(inh_j)), j_dim)

    c = zeros(a_dim, j_dim, name = :c)

    t_J = t_born + J
    resources_J = (1 + rs[t = At(t_J)]) .* (grid .+ inh[j = At(J)]) .+ ws[t = At(t_J)] * y[j = At(J)] * z
    c[j = At(J)] .= map(resources -> terminal_allocation(resources, par).c, resources_J)

    for j in J:-1:1
        t = t_born + j
        cⱼ = c[j = At(j)]

        cⱼ₋₁_aⱼ₋₁_df = a_prev_c_prev.(
            cⱼ,
            grid,
            Ref(par);
            r = rs[t = At(t)],
            r_prev = rs[t = At(t - 1)],
            w_prev = ws[t = At(t - 1)],
            y_prev = y[j = At(j - 1)] * z,
            m_prev = m[j = At(j - 1)],
            inh_prev = inh[j = At(j - 1)],
        ) |> DataFrame

        (; cⱼ₋₁, aⱼ₋₁) = cⱼ₋₁_aⱼ₋₁_df

        cⱼ₋₁_itp = linear_interpolation(aⱼ₋₁, cⱼ₋₁, extrapolation_bc = Line())
        c[j = At(j - 1)] .= cⱼ₋₁_itp.(grid)
    end

    a_sim = zeros(j_dim, name = :a)
    a_next_sim = zeros(j_dim, name = :a_next)
    c_sim = zeros(j_dim, name = :c)

    a_sim[j = At(j_init)] = a_init

    for j in j_init:J
        t = t_born + j
        aⱼ = a_sim[j = At(j)]

        cⱼ_itp = linear_interpolation(grid, c[j = At(j)], extrapolation_bc = Line())
        cⱼ = cⱼ_itp(aⱼ)

        aⱼ₊₁ = a_next(cⱼ; a = aⱼ, w = ws[t = At(t)], y = y[j = At(j)] * z,
                       r = rs[t = At(t)], m = m[j = At(j)], inh = inh[j = At(j)], par)

        a_lower = j == J ? 0.0 : a̲
        if aⱼ₊₁ < a_lower
            aⱼ₊₁ = a_lower
            cⱼ = c_curr(aⱼ, aⱼ₊₁, par; w = ws[t = At(t)], y = y[j = At(j)] * z,
                        r = rs[t = At(t)], m = m[j = At(j)], inh = inh[j = At(j)])
        end

        c_sim[j = At(j)] = cⱼ
        a_next_sim[j = At(j)] = aⱼ₊₁

        if j < J
            a_sim[j = At(j + 1)] = aⱼ₊₁
        end
    end

    sim_df = DataFrame(DimStack(a_sim, c_sim, a_next_sim))

    return (; sim_df, path_state = a_sim, c)
end

function solve_backward_forward(M::EGM, par, a_grid; r, w, a_init, z = 1.0, inh_j = nothing)
    (; J) = par

    rs = DimArray(fill(r, J + 1), Dim{:t}(0:J))
    ws = DimArray(fill(w, J + 1), Dim{:t}(0:J))

    price_paths = (; rs, ws)
    return _solve_backward_forward_(M, par, a_grid; price_paths, init_state = a_init, z, inh_j)
end
