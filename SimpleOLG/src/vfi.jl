function iterate_back!(v_curr, policy_curr, policy_plus_curr, a_grid, par, v_next; m, y, r, w, j, t, z = 1.0, inh = 0.0)
    (; β, u) = par
    yz = y * z

    for (i, a) in enumerate(a_grid)
        if m == 1
            resources = w * yz + (1 + r) * (a + inh)
            terminal = terminal_allocation(resources, par)
            v_opt = terminal.value
            a_next = terminal.a_next
            a_i_opt = findmin(abs.(collect(a_grid) .- a_next))[2]
            c_opt = terminal.c
        else
            values = u.(c.(a, a_grid, Ref(par); m, y = yz, r, w, inh)) .+ β * (1 - m) .* v_next
            (v_opt, a_i_opt) = findmax(values)
            a_next = a_grid[a_i_opt]
            c_opt = c(a, a_next, par; m, y = yz, r, w, inh)
        end

        v_curr[a = i] = v_opt
        policy_curr[a = i] = a_i_opt

        if !isnothing(policy_plus_curr)
            policy_plus_curr[a = i] = (;
                a_i = i,
                a_i_next = a_i_opt,
                c = c_opt,
                a,
                a_next,
                m,
                y = yz,
                r,
                w,
                j,
                t,
            )
        end
    end
end

function solve_backward(par, a_grid; r, w, t_born = 0, minimal = false, z = 1.0, inh_j = nothing)
    (; y, m) = par

    j_dim = DD.dims(m, :j)
    J = maximum(j_dim)
    @assert collect(j_dim) == 0:J

    let
        t_dim = DD.dims(r, :t)
        @assert t_dim == DD.dims(w, :t)
        T₀, T₁ = extrema(t_dim)
        @assert t_born >= T₀
        @assert t_born + J <= T₁
    end

    inh_arr = isnothing(inh_j) ? DimArray(zeros(Float64, J + 1), j_dim) :
              DimArray(Float64.(collect(inh_j)), j_dim)

    dim_a = Dim{:a}(a_grid)
    dims = (dim_a, j_dim)

    value = zeros(dims, name = :value)
    policy = zeros(Int, dims, name = :policy)

    if !minimal
        policy_plus = DimArray(Matrix{PPT()}(undef, size(policy)), dims, name = :policy_plus)
    else
        policy_plus = nothing
    end

    grid = DimArray(a_grid, dim_a)
    value[j = At(J)] .= 0.0

    for j in reverse(j_dim)
        t = t_born + j

        v_curr = @view value[j = At(j)]
        v_next = j == J ? 0.0 * v_curr : @view value[j = At(j + 1)]

        policy_curr = @view policy[j = At(j)]
        policy_plus_curr = minimal ? nothing : @view(policy_plus[j = At(j)])

        iterate_back!(
            v_curr,
            policy_curr,
            policy_plus_curr,
            grid,
            par,
            v_next;
            m = m[j = At(j)],
            y = y[j = At(j)],
            w = w[t = At(t)],
            r = r[t = At(t)],
            j,
            t,
            z,
            inh = Float64(inh_arr[j = At(j)]),
        )
    end

    return (; value, policy, policy_plus)
end

function solve_forward(out, par, a_grid; a_i_init, j_init)
    (; policy, policy_plus) = out

    dim_j = DD.dims(policy, :j)

    T = typeof((; j = 0, a_i = 1, a_i_next = 1))

    path_state = zeros(Int, dim_j)
    path_choice = zeros(Int, dim_j)
    path_choice_nt = T[]

    for j in j_init:par.J
        curr_a = j == 0 ? a_i_init : path_choice[j = At(j - 1)]
        path_state[j = At(j)] = curr_a

        a_i_next = policy[a = curr_a, j = At(j)]
        path_choice[j = At(j)] = a_i_next
        push!(path_choice_nt, (; j, a_i = curr_a, a_i_next))
    end

    pp_df = @chain policy_plus begin
        DataFrame
        select!(:policy_plus => AsTable, Not(:policy_plus))
    end

    df0 = DataFrame(path_choice_nt)
    sim_df = leftjoin!(df0, pp_df, on = [:a_i, :a_i_next, :j])

    return (; sim_df, path_state)
end

function solve_backward_forward(::VFI, par, a_grid; r, w, a_init, z = 1.0, inh_j = nothing)
    (; J) = par
    rs = DimArray(fill(r, J + 1), Dim{:t}(0:J))
    ws = DimArray(fill(w, J + 1), Dim{:t}(0:J))

    out_t = solve_backward(par, a_grid; r = rs, w = ws, z, inh_j)
    a_i_init = only(findall(a_grid .== a_init))

    (; path_state, sim_df) = solve_forward(out_t, par, a_grid; a_i_init, j_init = 0)

    return (; sim_df, path_state, out_t, a_init)
end
