"type of policy_plus named tuple"
PPT() = typeof((
    a_i = 1,
    a_i_next = 1,
    c = 1.0,
    a = 1.0,
    a_next = 1.0,
    m = 1.0,
    y = 1.0,
    r = 1.0,
    w = 1.0,
    j = 1,
    t = 1,
))

function get_par(;
    demo = :perpetual_youth,
    mm = 1 / 45,
    annuities = true,
    y,
    F = nothing,
    γ = 2.0,
    phi = 0.0,
    b_floor = 0.0,
    J = length(y) - 1,
    β = 0.995,
    ρ = 1 / β - 1,
    r = ρ,
    α = 0.33,
    δ = 0.1,
    a̲ = -Inf,
    bonds2GDP = 1.0,
)
    m = mortality(demo; J, m = mm)
    J = maximum(DD.dims(m, :j))
    j_dim = DD.dims(m, :j)

    u(c) = c > 0 ? c^(1 - γ) / (1 - γ) : -Inf

    F_out = if isnothing(F)
        DimVector(fill(0.0, length(j_dim)), j_dim, name = :F)
    else
        length(F) == length(j_dim) || throw(DimensionMismatch("F must have length $(length(j_dim))"))
        DimVector(Float64.(collect(F)), j_dim, name = :F)
    end

    return (;
        δ,
        α,
        Θ = 1,
        L = 1,
        β,
        ρ,
        r,
        bonds2GDP,
        m,
        J,
        γ,
        phi,
        b_floor,
        a̲,
        annuities,
        y,
        F = F_out,
        u,
        w = 1.0,
    )
end
