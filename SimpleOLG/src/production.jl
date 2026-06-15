interest_rate(K, (; L, α, Θ, δ)) = K <= 0.0 ? NaN : α * Θ * (K / L)^(α - 1) - δ

wage(K, (; L, α, Θ, δ)) = K <= 0.0 ? NaN : (1 - α) * Θ * (K / L)^α

output(K, (; L, α, Θ, δ)) = K <= 0.0 ? NaN : Θ * K^α * L^(1 - α)

inverse_interest_rate(r, (; L, α, Θ, δ)) = ((r + δ) / (α * Θ))^(1 / (α - 1)) * L
