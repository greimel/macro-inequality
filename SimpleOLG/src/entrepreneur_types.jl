abstract type TaxRegime end

struct NoTax <: TaxRegime end

struct CapitalIncomeTax <: TaxRegime
    tau_k::Float64
end

struct WealthTax <: TaxRegime
    tau_a::Float64
end

Base.@kwdef struct EntrepreneurParams
    alpha::Float64 = 0.3
    delta::Float64 = 0.0
    theta::Float64 = 2.0
    f::Float64 = 0.0
end
