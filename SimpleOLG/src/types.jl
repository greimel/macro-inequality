abstract type SolutionMethod end

struct VFI <: SolutionMethod end

struct EGM <: SolutionMethod end

Base.@kwdef struct PermanentType
    z_labor::Float64 = 1.0
    z_entrepreneur::Float64 = 1.0
end

LaborProductivity(z)    = PermanentType(; z_labor = z)
EntrepreneurSkill(z)    = PermanentType(; z_entrepreneur = z)
CombinedProductivity(z) = PermanentType(; z_labor = z, z_entrepreneur = z)
