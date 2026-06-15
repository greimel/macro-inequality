_default_budget_par() = (; annuities = true, phi = 0.0)

saving_weight(m, (; annuities, phi)) = annuities ? (m == 1 && phi > 0 ? 1.0 : 1 - m) : 1.0

c(a, a_next, par; m, y, r, w, inh = 0.0) = w * y + (1 + r) * (a + inh) - saving_weight(m, par) * a_next

c(a, a_next; m, y, r, w, inh = 0.0) = c(a, a_next, _default_budget_par(); m, y, r, w, inh)

u_prime(c, (; γ)) = c > 0 ? c^(-γ) : Inf

inv_u_prime(mu, (; γ)) = mu > 0 ? mu^(-1 / γ) : Inf

function terminal_allocation(resources, par)
    (; phi, b_floor, u, γ) = par

    if !(phi > 0)
        return (; c = resources, a_next = 0.0, value = u(resources))
    end

    bequest_ratio = phi^(1 / γ)
    c_star = (resources + b_floor) / (1 + bequest_ratio)
    a_next_star = resources - c_star

    if !(a_next_star > 0)
        c_star = resources
        a_next_star = 0.0
    end

    value = u(c_star) + phi * u(a_next_star + b_floor)
    return (; c = c_star, a_next = a_next_star, value)
end

function a_next(c; a, w, y, r, m, inh = 0.0, par = nothing)
    effective_par = isnothing(par) ? _default_budget_par() : par
    resources = (1 + r) * (a + inh) + y * w

    if m == 1
        if effective_par.phi > 0
            return max(resources - c, 0.0)
        end
        return 0.0
    end

    return (resources - c) / saving_weight(m, effective_par)
end

c_curr(a, a_next, par; w, y, r, m, inh = 0.0) = (1 + r) * (a + inh) + y * w - saving_weight(m, par) * a_next

c_curr(a, a_next; w, y, r, m, inh = 0.0) = c_curr(a, a_next, _default_budget_par(); w, y, r, m, inh)

function c_prev(c, r, par; m_prev = 0.0)
    effective_discount = par.β * (1 + r) * (par.annuities ? 1.0 : 1 - m_prev)
    return c / effective_discount^(1 / par.γ)
end

function a_prev(c_prev, a, par; w_prev, y_prev, r_prev, m_prev, inh_prev = 0.0)
    return (saving_weight(m_prev, par) * a + c_prev - w_prev * y_prev) / (1 + r_prev) - inh_prev
end

a_prev(c_prev, a; w_prev, y_prev, r_prev, m_prev, inh_prev = 0.0) =
    a_prev(c_prev, a, _default_budget_par(); w_prev, y_prev, r_prev, m_prev, inh_prev)

function a_prev_c_prev(cⱼ, aⱼ, par; r, r_prev, y_prev, w_prev, m_prev, inh_prev = 0.0)
    cⱼ₋₁ = c_prev(cⱼ, r, par; m_prev)
    aⱼ₋₁ = a_prev(cⱼ₋₁, aⱼ, par; y_prev, w_prev, r_prev, m_prev, inh_prev)

    return (; cⱼ₋₁, aⱼ₋₁)
end
