function _efficient_entrepreneur_capital(z, epar::EntrepreneurParams; r)
    z <= 0 && return 0.0

    gross_user_cost = r + epar.delta
    gross_user_cost <= 0 && throw(DomainError(gross_user_cost, "r + delta must be positive"))

    return (epar.alpha * z / gross_user_cost)^(1 / (1 - epar.alpha))
end

efficient_entrepreneur_capital(z, epar::EntrepreneurParams; r) = _efficient_entrepreneur_capital(z, epar; r)

_collateral_entrepreneur_capital(a, epar::EntrepreneurParams) = max(epar.theta * a, 0.0)

function entrepreneur_capital(a, z, epar::EntrepreneurParams; r)
    k_star = _efficient_entrepreneur_capital(z, epar; r)
    return min(k_star, _collateral_entrepreneur_capital(a, epar))
end

function entrepreneur_profit(a, z, epar::EntrepreneurParams; r)
    k = entrepreneur_capital(a, z, epar; r)
    return z * k^epar.alpha - (r + epar.delta) * k
end

function after_tax_wealth(a, z, epar::EntrepreneurParams, ::NoTax; r)
    π = entrepreneur_profit(a, z, epar; r)
    return (1 + r) * a + π
end

function after_tax_wealth(a, z, epar::EntrepreneurParams, tax::CapitalIncomeTax; r)
    π = entrepreneur_profit(a, z, epar; r)
    return a + (π + r * a) * (1 - tax.tau_k)
end

function after_tax_wealth(a, z, epar::EntrepreneurParams, tax::WealthTax; r)
    π = entrepreneur_profit(a, z, epar; r)
    return a * (1 - tax.tau_a) + π + r * a
end

function disposable_resources(a, z, epar::EntrepreneurParams, tax::TaxRegime; r, w, y)
    fixed_cost = z > 0 ? epar.f : 0.0
    return after_tax_wealth(a, z, epar, tax; r) + w * y - fixed_cost
end
