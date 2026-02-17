### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 5a6d468f-327e-4c06-97a3-bcee431a3518
using PlutoLinks: ingredients

# ╔═╡ e51238ae-eea3-4956-b22a-59fb42f4a9e7
using PlutoUI

# ╔═╡ 5c94956c-0675-423f-bf96-30f70bd81009
using Chain, DataFrames, DataFrameMacros

# ╔═╡ 3365570c-2cbe-4caf-b56f-80dd9f3b6529
using CairoMakie, AlgebraOfGraphics

# ╔═╡ dfc5566b-187c-40f3-9349-99584efe1b64
using PlutoTest

# ╔═╡ 86a050f8-cc13-443c-881f-58f3660ec607
using StatsBase

# ╔═╡ a4a7158e-1455-478d-b9a8-cdc6f2ddc715
using DimensionalData

# ╔═╡ 9ac247d6-1b9d-4bdd-bdb4-0d62ecba61d7
using LinearAlgebra: dot, norm

# ╔═╡ 15efda29-e6d4-4096-9f5b-861eaf81e482
using Roots: find_zero

# ╔═╡ a9af97d9-8905-4550-afac-458ffe737a14
using Interpolations

# ╔═╡ 2d4581c2-6801-4754-a6b1-b91ae131e9f8
using QuantEcon: QuantEcon, rouwenhorst, MarkovChain, stationary_distributions

# ╔═╡ bacf8ffa-cb2b-4e0e-989d-c4c84b4d8d2c
using Statistics: mean

# ╔═╡ 1a4a13be-2913-4710-bae4-3899fa415198
import SparseArrays

# ╔═╡ 9c88ad1f-8a68-4260-8416-da6745623dc1
Demographics = ingredients("./demographics.jl")

# ╔═╡ 5d8c511b-d97f-4479-873d-4810e0c67a0d
(; get_π_j, get_π_jt, get_π_t) = Demographics

# ╔═╡ adb79a2d-cf1e-42a4-a821-d9afd37b73bf
md"""
## To do

* assets of the dead (case distinction)
* initial assets during the transition
* interpolate using pchip
* compute value ex-post at different ages
* specify multiple transitions (as in the slides)
"""

# ╔═╡ d271fb80-8c93-11f0-3406-ab394e785ceb
md"""
* steady state
  * `stationary_GE`
  * `stationary_PE` calls `simulate_cohort.(θs)`\
     ($\to$ `initialize_cohort(θ)` and `solve_backward_forward!(θ)`)
* transition
  * `transition_GE`
  * `transition_PE` calls `simulate_cohorts.(θs)`
    ($\to$ `initialize_cohorts(θ)` and `solve_backward_forward!.(θ, cohorts)`)
"""

# ╔═╡ 91fcc5f6-0789-4fb4-9984-105922332395
md"""
# Transition
"""

# ╔═╡ a44e51f9-3f90-4ec8-b890-e53e7b9549f6
function aggregate_paths(sim_df; normalize_population = false)
	
	df₀ = @chain sim_df begin
		@transform(:t = :j + :born) 
		
	end

	T̃ = @chain df₀ begin
		groupby(:t)
		@combine(:n_cohorts = length(:t))
		@subset(@bycol :n_cohorts .== maximum(:n_cohorts))
		maximum(_.t)
	end

	t_dim = Dim{:t}(0:T̃)

	@chain df₀ begin
		@subset!(0 ≤ :t ≤ T̃, :π > 0)
	end

	if normalize_population
		@chain df₀ begin
			@groupby(:t)
			@transform!(:π = @bycol :π ./ sum(:π))
		end
	end

	pop = @chain df₀ begin
		@groupby(:t)
		@combine(:pop = sum(:π))
		DimVector(_.pop, t_dim, name = :population)
	end

	variables = setdiff(names(df₀), string.([:j, :t, :born, :π, :permanent]))
	
	@chain df₀ begin
		@groupby(:t)
		@combine(variables = sum({variables}, weights(:π)), :population = sum(:π) )
		stack([variables; "population"])
		@groupby(:variable)
		@combine(:da = [DimVector(:value, t_dim, name = only(unique(:variable)))])
		DimStack(_.da...)
	end
end

# ╔═╡ 35347fb7-2085-4d06-9d17-60757bd331c3
function dimstack_from_nt(nt, dim)
	guessed_path = map(zip(keys(nt), values(nt))) do (name, val)
		fill(val, dim; name)
	end |> DimStack
end

# ╔═╡ c176b6b1-75aa-479d-a34c-22ecc4823e27
md"""
# Bequests
"""

# ╔═╡ c94759d1-b5b0-4320-8f42-c57d3bf6ca2a
function get_bequests_θ(sim_df, statespace)
	(; perm_dim) = statespace
	
	@chain sim_df begin
		@transform(:π = @bycol :π ./ sum(:π))
		@groupby(:permanent)
		@combine(
			:π = sum(:π),
			:bequests = mean(:bequests, weights(:π))
		)
		DimVector(_.bequests, perm_dim, name = :bequests)
	end
end

# ╔═╡ b6976a71-69c5-4016-9bf1-859b3208485f
function get_bequests_θt((; sim_df, T̃, GE₀), statespace)
	(; perm_dim) = statespace
	t_dim = Dim{:t}(-1:T̃)

	bequests₋₁_df = @chain get_bequests_θ(GE₀.sim_df, statespace) begin
		DataFrame
		@select(:t = -1, :permanent = (; θ = :θ), :bequests)
	end

	out = @chain sim_df begin
		@transform(:t = :born + :j)
		@subset(0 ≤ :t ≤ T̃)
		#@select(:j, :t, :born, :state, :ε, :π)		
		@subset(:π > 0)
		@groupby(:t, :permanent)
		@combine(:bequests = mean(:bequests, weights(:π)))
		[_; bequests₋₁_df]
		sort([:permanent, :t])
		@groupby(:permanent)
		@combine(
			:bequests = Ref(DimVector(:bequests, t_dim, name = :bequests))
		)
		stack(_.bequests)
		DimArray((t_dim, only(perm_dim)), name = :bequests)
	end
	
end

# ╔═╡ a17f0114-55f6-4d43-9b05-6bcd9601b98b
function get_inheritances_θj(inheritances_θ, par, π_j = get_π_j(par.m))
	
	inheritances_θj = 
		DimArray(@d(inheritances_θ .* par.F ./ π_j), name = :inheritances)
end

# ╔═╡ 815c5b9f-e329-4a2e-b0fe-667d2052d980
# ╠═╡ disabled = true
#=╠═╡
function get_inheritances_θtj(inheritances_θt, (; par, sim_df))
	π_age = get_age_distribution((; sim_df))
	
	inheritances_θj = 
		DimArray(@d(inheritances_θ .* F ./ π_age), name = :inheritances)
end
  ╠═╡ =#

# ╔═╡ 133e1e40-4b84-4c9d-803f-22c2884e81f5
md"""
# General functionality
"""

# ╔═╡ 16ed251b-5ae9-4520-88fa-1e67987615db
md"""
### `aggregate`, and `*_equilibrium`
"""

# ╔═╡ 917206f6-abd6-4ffb-a738-6c39024cbaaa
function aggregate(sim_df; total_mass)
	agg_nt = @chain sim_df begin
		@transform(:π = @bycol :π ./ sum(:π) .* total_mass)
		stack(Not(:j, :π, :permanent))
		@groupby(:variable)
		@combine(:value = sum(:value, weights(:π)))
		(; (Symbol.(_.variable) .=> _.value)...)
	end
end

# ╔═╡ 00c2612f-8f1c-4412-a159-e4325af0c62f
drop_m_h(; h, m, ρ_SS, kwargs...) = (; kwargs...)

# ╔═╡ 3de09148-706d-40ed-90b2-39a71e9d25ed
md"""
### `solve_backward_forward`
"""

# ╔═╡ b35bfc6e-6f39-461a-b5b4-ca2901c83da4
function constant_price_paths((; J), prices)
	
	price_paths₀ = map(zip(keys(prices), values(prices))) do (name, value)
		fill(value, Dim{:t}(-1:J+2); name)
	end
	
	price_paths = NamedTuple{keys(prices)}(price_paths₀)
end

# ╔═╡ adfca787-4bc4-419f-9a0f-12e051c36134
md"""
### `solve_forward`
"""

# ╔═╡ 501b1a2d-3b7c-4bd9-a119-3e2b829b7a56
get_states(dim::Dim) = get_states((dim,))

# ╔═╡ 0c59fd3d-263d-43ba-8ba1-5138d767cf48
get_states(dims::Tuple) = NamedTuple{name.(dims)}.(DimPoints(dims))

# ╔═╡ 6e45e994-8a1f-4b79-85bd-fb999bc5493d
function weighted_neighbours(x, x_grid)
	n = length(x_grid)
	
	if x < minimum(x_grid) || x > maximum(x_grid)
		if (minimum(x) == 0.0 && x < -1e-10) || (x ≉ minimum(x_grid)) || (x ≠ maximum(x_grid))
			#"Optimal choice outside the grid $x ∉ $(extrema(x_grid))"
		end
		low_idx = 1
		high_idx = n
	else
		low_idx = clamp(searchsortedlast(x_grid, x), 1, n)
	    high_idx = clamp(low_idx + 1, 1, n)
	end
	
	x_low  = x_grid[low_idx]
	x_high = x_grid[high_idx]
	
	if x < minimum(x_grid)
		weight_high = 0.0
	elseif x > maximum(x_grid)
		weight_high = 1.0
	else
		span = x_high - x_low
		weight_high = span == 0.0 ? 1.0 : (x - x_low) / span
	end
		
	weight_low = 1.0 - weight_high
	
	high = (; i = high_idx, x = x_high, weight = weight_high)
	low = (; i = low_idx,  x = x_low,   weight = weight_low)

	if weight_high < 0.0 || weight_high > 1.0
		@info (; x, high, low)
	end
	
	(; high, low)
	
end

# ╔═╡ 24c68069-e2d6-428e-9b08-014263b8f07b
weighted_neighbours(5.5, -10:5)

# ╔═╡ 685427d8-9f41-482e-bb08-7fbe7aff32ef
md"""
## Solving backward
"""

# ╔═╡ 9e05ea7c-bc7a-4dbd-8199-298c53447d7c
global COUNTER = 0

# ╔═╡ cfb2d2e9-f101-464f-915f-b0f6a38744e5
function dimarray_of_nts_to_nt_of_dimarrays(da)
	names = keys(first(da))

	NamedTuple{names}(tuple((DimArray(getproperty.(da, name); name) for name ∈ names)...))
end

# ╔═╡ 07a443b3-7233-43a5-aacd-7fb79bec0edf
md"""
# Model specific
"""

# ╔═╡ 6569e751-ce87-4063-a142-ad62d088e70d
begin
	Base.@kwdef struct HousingModel
		statename = :ω
		agg_statename = :H
	end
end

# ╔═╡ ae5bd8c1-dca5-495d-a5bb-a2d271262645
function get_prices(::HousingModel, price_paths, par_all, par_cohort, j; t_born, inherit)
	(; J) = par_all
	(; h, ρ_SS, m) = par_cohort
	
	(; r, p, w) = price_paths

	t = t_born + j

	prices_etcⱼ₋₁ = (; 
					  r = r[t = At(t-1)], 
					  p = p[t = At(t-1)], 
					   w = w[t = At(t-1)],
					   m = m[j = At(max(0,j-1))],
				       h = h[j = At(max(0,j-1))],
					 ρ_SS = ρ_SS[j = At(max(0,j-1))],
					 inherit = inherit[j = At(max(0,j-1))]
					  )
	
	prices_etcⱼ = (; 
					  r = r[t = At(t)], 
				     p = p[t = At(t)], 
					   w = w[t = At(t)],
					   m = m[j = At(j)],
				       h = h[j = At(j)],
					ρ_SS = ρ_SS[j = At(j)],
				    inherit = inherit[j = At(j)]
					  )

	
	if j < J ## this should be about T not J
		
		prices_etcⱼ₊₁ = (; 
						  r = r[t = At(t+1)],
						   p = p[t = At(t+1)], 
						   w = w[t = At(t+1)],
						   m = m[j = At(j+1)],
					       h = h[j = At(j+1)],
						 ρ_SS = ρ_SS[j = At(j+1)],
						  inherit = inherit[j = At(j+1)]
						  )
	else
		prices_etcⱼ₊₁ = prices_etcⱼ
	end
	
	(; prices_etcⱼ₋₁, prices_etcⱼ, prices_etcⱼ₊₁)
end

# ╔═╡ 34285cb0-4c79-43bc-a299-d2571b77e9fa
function income_plus_inheritances(ε, (; τ, d̄), (; θ), (; prices_etcⱼ₋₁))
	ⱼ₋₁ = prices_etcⱼ₋₁
	ρⱼ₋₁ = ⱼ₋₁.ρ_SS
	hⱼ₋₁ = ⱼ₋₁.h
	inhⱼ₋₁ = ⱼ₋₁.inherit
	ℓ_effⱼ₋₁ = θ * ε * hⱼ₋₁
	yⱼ₋₁   = @. (1-ρⱼ₋₁) * (1-τ) * ℓ_effⱼ₋₁ + ρⱼ₋₁ * d̄ * θ
	incⱼ₋₁ = yⱼ₋₁ .* ⱼ₋₁.w
	inc_plus_inhⱼ₋₁ = @d incⱼ₋₁ .+ (inhⱼ₋₁ .* (1 + ⱼ₋₁.r))

	(; inhⱼ₋₁, incⱼ₋₁, inc_plus_inhⱼ₋₁, ℓ_effⱼ₋₁)
end

# ╔═╡ f209f007-46a5-4100-94f9-db3bda5254db
function constraint_is_violated((; aⱼ, hⱼ₋₁), par, priceses)
	(; θ, a̲, ξ) = par
	ⱼ₋₁ = priceses.prices_etcⱼ₋₁
	if all(≠(0), ξ)
		return aⱼ < - θ * hⱼ₋₁ * ⱼ₋₁.p - a̲
	else
		return false
	end
end

# ╔═╡ 5bb0b7e1-7476-4266-9a02-511a6e24f165
function constraint_binds((; zⱼ, 𝔼u_cⱼ, inc_plus_inhⱼ₋₁), par, priceses, j)
	(; θ, a̲, σ, δ, β) = par
	ξ = par.ξ[j = At(j-1)]

	if length(par.β) == 1
		βⱼ₋₁ = par.β^(j-1)
		βⱼ   = par.β^j
	else
		βⱼ₋₁, βⱼ = par.β[j = At(j-1:j)]
	end
	
	ⱼ₋₁ = priceses.prices_etcⱼ₋₁
	ⱼ   = priceses.prices_etcⱼ

	# compute hⱼ₋₁ implied by binding collateral constraint
	X = (1-δ)/(1 + ⱼ₋₁.r) * ⱼ.p / ⱼ₋₁.p
	hⱼ₋₁ = (zⱼ + (1 + ⱼ₋₁.r) * a̲) / (X - θ) / ⱼ₋₁.p

	if hⱼ₋₁ < 0
		@info (; zⱼ, 𝔼u_cⱼ, X, hⱼ₋₁, j)
	end
	
	# compute implied aⱼ for later
	#aⱼ   = zⱼ - (1 - δ)/(1 + ⱼ.r) * hⱼ₋₁ * ⱼ.p
	aⱼ   = - hⱼ₋₁ * ⱼ₋₁.p * θ - a̲
	
	function loss(c)
		XXX = (c^(1-ξ) * hⱼ₋₁^ξ)^(1-σ)
		u_cⱼ₋₁ = (1-ξ) * XXX / c
		u_hⱼ₋₁ = ξ     * XXX / hⱼ₋₁

		LHS = u_hⱼ₋₁/u_cⱼ₋₁ 
		RHS = (1-θ) * ⱼ₋₁.p -
			βⱼ/βⱼ₋₁ * (1 - ⱼ₋₁.m) * 𝔼u_cⱼ/u_cⱼ₋₁ *
				(ⱼ.p * (1-δ) + θ * ⱼ₋₁.p * (1 + ⱼ.r))

		LHS - RHS
	end

	c_bracket = (eps(), 10_000.0)

	if !(loss(c_bracket[1]) * loss(c_bracket[2]) < 0)
		@info (; loss_at_bracket = loss.(c_bracket), j, zⱼ, 𝔼u_cⱼ, inc_plus_inhⱼ₋₁)
	end
	#@info loss(√eps()), loss(100.0)
	cⱼ₋₁_bind = max(find_zero(loss, c_bracket), eps())
	
	zⱼ₋₁_bind = cⱼ₋₁_bind + ⱼ₋₁.p * hⱼ₋₁ + aⱼ - inc_plus_inhⱼ₋₁ # from budget constraint
	
	(; cⱼ₋₁_bind, zⱼ₋₁_bind)
end

# ╔═╡ 8f9c937b-5a00-40ce-93ef-cbfc8058dacf
function aⱼ_from_hⱼ₋₁(hⱼ₋₁, zⱼ, (; δ), (; prices_etcⱼ))
	(; p, r) = prices_etcⱼ

	ph = hⱼ₋₁ == 0.0 ? hⱼ₋₁ : p * hⱼ₋₁

	ωⱼ = zⱼ * (1+r)
	aⱼ = (ωⱼ - ph * (1-δ)) / (1 + r)
end

# ╔═╡ 277e1e63-4df5-42a0-a5f3-ad0b73eab310
function zⱼ₋₁_from_ahc(aⱼ, hⱼ₋₁, cⱼ₋₁, inc_plus_inhⱼ₋₁, (; prices_etcⱼ₋₁), (; annuities))
	(; p, r, m) = prices_etcⱼ₋₁

	ph = hⱼ₋₁ == 0.0 ? hⱼ₋₁ : p * hⱼ₋₁
	
	ωⱼ₋₁ = aⱼ + ph + cⱼ₋₁ - inc_plus_inhⱼ₋₁
	if annuities
		ωⱼ₋₁ = ωⱼ₋₁ * (1 + m)
	end
	zⱼ₋₁ = ωⱼ₋₁ / (1+r)
end

# ╔═╡ 1f07f950-c69e-4a12-b404-6af89796c58e
function get_κⱼ(pₜ₍ⱼ₎, pₜ₍ⱼ₊₁₎, rₜ₍ⱼ₊₁₎, (; ξ, δ), j)
	ξ = ξ[j = At(j)]
	pₜ₍ⱼ₎ * (1-ξ)/ξ * (
		1 - (1-δ)/(1+rₜ₍ⱼ₊₁₎) * pₜ₍ⱼ₊₁₎/pₜ₍ⱼ₎) # c_by_h
	
end

# ╔═╡ 26bce906-16cf-4dd8-a9e8-a09d5270a6fd
function hⱼ₋₁_from_cⱼ₋₁(cⱼ₋₁, par, (; prices_etcⱼ₋₁, prices_etcⱼ); j)
	ⱼ₋₁ = prices_etcⱼ₋₁
	ⱼ   = prices_etcⱼ
	ξ = par.ξ[j = At(j)]
	if ξ > 0
		hⱼ₋₁ = cⱼ₋₁ ./ get_κⱼ(ⱼ₋₁.p, ⱼ.p, ⱼ.r, par, j-1)
	else
		hⱼ₋₁ = 0 .* cⱼ₋₁
	end
end

# ╔═╡ 5b07590c-b6f9-489e-8c8f-1757ede48cbc
function uu(c, h, (; ξ, σ))
	if ξ ≠ 0
		c = c^(1-ξ) * h^ξ
	end

	return c^(1-σ)/(1-σ)
end

# ╔═╡ 6b991515-29e8-4fba-a473-6dd1f8cb9d03
function choicesⱼ₋₁(::HousingModel, stateⱼ₋₁, stateⱼ, 𝔼vⱼ, constrainedⱼ₋₁, εⱼ₋₁,
	par_all, par_cohort, permanent, (; prices_etcⱼ₋₁, prices_etcⱼ), statespace, j)
	par = (; par_all..., par_cohort...)
	(; δ, θ, a̲, annuities) = par
	#(; w, p, r, m) = prices_etcⱼ₋₁
	ⱼ₋₁ = prices_etcⱼ₋₁
	ⱼ   = prices_etcⱼ
	
	zⱼ₋₁ = stateⱼ₋₁
	zⱼ   = stateⱼ

	ωⱼ   = zⱼ * (1 + ⱼ.r)
	ωⱼ₋₁ = zⱼ₋₁ * (1 + ⱼ₋₁.r)

	(; inhⱼ₋₁, incⱼ₋₁, inc_plus_inhⱼ₋₁, ℓ_effⱼ₋₁) = income_plus_inheritances(εⱼ₋₁, par_all, permanent, (; prices_etcⱼ₋₁))

	ξ = par.ξ[j = At(j)]

	if length(par.β) == 1
		βⱼ_over_βⱼ₋₁ = par.β
	else
		βⱼ₋₁, βⱼ = par.β[j = At(j-1:j)]
		βⱼ_over_βⱼ₋₁ = βⱼ / βⱼ₋₁ 
	end
	 
	if ξ ≠ 0
		κⱼ₋₁ = get_κⱼ(ⱼ₋₁.p, ⱼ.p, ⱼ.r, par, j-1)
	end
	
	choices_NC = let # not constrained
		XX = annuities ? (1 - ⱼ₋₁.m) : 1.0
		if ξ ≠ 0
			hⱼ₋₁ = (inc_plus_inhⱼ₋₁ + ωⱼ₋₁ - XX * ωⱼ / (1 + ⱼ.r)) / (κⱼ₋₁ + ⱼ₋₁.p - (1-δ)/(1 + ⱼ.r)* ⱼ.p)
			cⱼ₋₁ = κⱼ₋₁ * hⱼ₋₁
			aⱼ   = (ωⱼ - (1-δ) * ⱼ.p * hⱼ₋₁) / (1 + ⱼ.r)
		else
			aⱼ   = zⱼ
			aⱼ₋₁ = zⱼ₋₁
			hⱼ₋₁ = 0.0
			# cⱼ₋₁ = inc_plus_inhⱼ₋₁ + aⱼ₋₁ * (1 + ⱼ₋₁.r) - XX * aⱼ
			cⱼ₋₁ = inc_plus_inhⱼ₋₁ + ωⱼ₋₁ - XX * ωⱼ / (1 + ⱼ.r)
		end
			
		(; cⱼ₋₁, aⱼ, hⱼ₋₁)
	end

	if ξ > 0.0
		choices_C = let # constrained
			## might be wrong with annuities
			# compute hⱼ₋₁ implied by binding collateral constraint
			X = (1-δ)/(1 + ⱼ₋₁.r) * ⱼ.p / ⱼ₋₁.p
			hⱼ₋₁ = (zⱼ + (1 + ⱼ₋₁.r) * a̲) / (X - θ) / ⱼ₋₁.p
			aⱼ   = - θ * ⱼ₋₁.p * hⱼ₋₁ - a̲
			cⱼ₋₁ = (inc_plus_inhⱼ₋₁ + ωⱼ₋₁) - aⱼ - ⱼ₋₁.p * hⱼ₋₁
	
			(; cⱼ₋₁, aⱼ, hⱼ₋₁)
		end
		constr2 = constrainedⱼ₋₁ 
	
		cⱼ₋₁ = (1 - constrainedⱼ₋₁) * choices_NC.cⱼ₋₁ + constrainedⱼ₋₁ * choices_C.cⱼ₋₁
		hⱼ₋₁ = (1 - constrainedⱼ₋₁) * choices_NC.hⱼ₋₁ + constrainedⱼ₋₁ * choices_C.hⱼ₋₁
		aⱼ   = (1 - constrainedⱼ₋₁) * choices_NC.aⱼ   + constrainedⱼ₋₁ * choices_C.aⱼ
	else
		(; cⱼ₋₁, aⱼ, hⱼ₋₁) = choices_NC
		constr2 = 0.0
	end

	uⱼ₋₁ = uu(cⱼ₋₁, hⱼ₋₁, (; ξ, par.σ))
	vⱼ₋₁ = uⱼ₋₁ + βⱼ_over_βⱼ₋₁ * (1 - ⱼ₋₁.m) * 𝔼vⱼ
	a_next = aⱼ
	m = ⱼ₋₁.m
	stuff = (; co=cⱼ₋₁, ho=hⱼ₋₁, 
			 a_next, 
			 a_next_surv = a_next * (1 - m), 
			 bequests = zⱼ * m, ### changed !!!
			 z_next=zⱼ, ℓ_eff = ℓ_effⱼ₋₁, inheritance = inhⱼ₋₁, income = incⱼ₋₁, z = zⱼ₋₁, ⱼ₋₁.m, #=h_lc = ⱼ₋₁.h,=# value = vⱼ₋₁, next_value =  𝔼vⱼ,  utility = uⱼ₋₁, β_over_β = βⱼ_over_βⱼ₋₁, constrained = constr2)

	(; c = stuff.co, v=vⱼ₋₁, stuff)
end

# ╔═╡ 97fc998b-83cc-4d35-b78a-0c9b9bc3cfad
function last_choices(::HousingModel, zⱼ, ℓ_effⱼ, yⱼ, (; prices_etcⱼ, prices_etcⱼ₊₁), par_all, J)
	(; u′, u, v′, v, ξ, σ, δ) = par_all

	ξ = ξ[j = At(J)]
	(; w, r, p, m) = prices_etcⱼ
	rₜ₊₁ = prices_etcⱼ₊₁.r
	pₜ₊₁ = prices_etcⱼ₊₁.p

	inc_J = w * yⱼ
	inheritance_J = prices_etcⱼ.inherit
	ωⱼ = zⱼ * (1+r)
	wealth_J = (ωⱼ + inc_J + inheritance_J)

	if ξ ≠ 0
		κⱼ = get_κⱼ(p, pₜ₊₁, rₜ₊₁, par_all, J)
	end
	
	_a_next_(c) = wealth_J - c - (ξ > 0 ? p * c/κⱼ : 0.0)
	u_c(c) = if ξ ≠ 0
		(1-ξ)*κⱼ^(-ξ * (1-σ))*c^(-σ)
	else
		c^(-σ)
	end
	
	if par_all.ν₀ == 0.0 # FIX ???
		c = (1-ξ) * wealth_J
		h = ξ == 0 ? 0.0 : (ξ * wealth_J) / (p * (1 - (1-δ)/(1+rₜ₊₁) * pₜ₊₁/p))
	else
		#if
		#
		obj(c) = u_c(c) - v′(_a_next_(c))
	
		cₘᵢₙ = 0.000001
		cₘₐₓ = 0.999 * wealth_J
		if ξ > 0
			cₘₐₓ /= (1 + p/κⱼ)
		end
		
		c = find_zero(obj, [cₘᵢₙ, cₘₐₓ])
		h = ξ > 0 ? c/κⱼ : 0.0
	end

	ph    = h == 0.0 ? 0 : p    * h
	phₜ₊₁ = h == 0.0 ? 0 : pₜ₊₁ * h
	
	a_next = wealth_J - c - ph
	ω_next = a_next * (1+rₜ₊₁) + (1-δ) * phₜ₊₁
	z_next = ω_next / (1+rₜ₊₁)

	vⱼ = uu(c, h, (; ξ, σ))
	
	return (; c, v=vⱼ, next_state = ω_next, stuff = (; co=c, ho=h, a_next, a_next_surv = a_next * (1 - m), 
													 bequests = z_next * m, # changed !!!
													 z_next, ℓ_eff = ℓ_effⱼ, inheritance = inheritance_J, income = inc_J, z = zⱼ, m, #= h_lc = 99.0,=# value = vⱼ, next_value = 0.0, utility = vⱼ, β_over_β = 0.0, constrained = float(false)))
end

# ╔═╡ 9276b022-d863-473e-978a-67a614d7ee31
function get_type_of_stuff(Mo, par_all, par_cohort, permanent, statespace; price_paths, inherit)
	(; J) = par_all
	(; h) = par_cohort
	(; ε_grid, grid) = statespace

	prices_J = get_prices(Mo, price_paths, par_all, par_cohort, J; t_born=0, inherit)
	inc_J   = (ε_grid .* h[j = At(J)]) .+ 0.01
	
	out = @d last_choices.(Ref(Mo), grid, inc_J, inc_J, Ref(prices_J), Ref(par_all), Ref(J))

	stuff_J = getproperty.(out, :stuff)
	T = eltype(stuff_J)
end

# ╔═╡ fa757572-54a1-4a4a-975f-72c0bccc67aa
function initialize_cohort(Mo, par_all, par_cohort, permanent, statespace; price_paths, j_init = 0, t_born = 0, inherit)
	(; dims) = statespace
	(; J) = par_all
	
	j_dim = Dim{:j}(j_init:J)
	dims_j = (dims..., j_dim)
	
	c           = zeros(dims_j, name = :c)
	next_state  = zeros(dims_j, name = :next_state)
	value       = zeros(dims_j, name = :value)
	next_value       = zeros(dims_j, name = :next_value)
	constrained = zeros(dims_j, name = :constrained)
	
	T = get_type_of_stuff(Mo, par_all, par_cohort, permanent, statespace; price_paths, inherit)
	
	stuff = DimArray(Array{T}(undef, size(dims_j)), dims_j, name = :stuff)
	# initialize distribution and fill initial distribution
	π = zeros(dims_j, name = :π)

	(; c, next_state, value, next_value, constrained, stuff, π)
end

# ╔═╡ 4a7ee470-4496-471c-86a0-c36d3b6c5a7f
function initialize_cohorts(Mo, par_all, par_cohort, permanent, statespace, t_borns; price_paths, j_init = 0, t_born = 0, inherit)
	(; dims) = statespace
	(; J) = par_all
	
	j_dim = Dim{:j}(j_init:J)
	born_dim = Dim{:born}(t_borns)
	
	dims_X = (j_dim, born_dim, dims...)
	
	c           = zeros(dims_X, name = :c)
	next_state  = zeros(dims_X, name = :next_state)
	value       = zeros(dims_X, name = :value)
	next_value  = zeros(dims_X, name = :next_value)
	constrained = zeros(dims_X, name = :constrained)

	T = get_type_of_stuff(Mo, par_all, par_cohort, permanent, statespace; price_paths, inherit)
	
	stuff = DimArray(Array{T}(undef, size(dims_X)), dims_X, name = :stuff)
	
	# initialize distribution and fill initial distribution
	π = zeros(dims_X, name = :π)

	(; c, next_state, value, next_value, constrained, stuff, π)
end

# ╔═╡ ce1c7b3f-cc7d-4aca-afd1-cee741df6f2a
function get_aⱼ₋₁(cⱼ₋₁, aⱼ, y_prev, prices_etcⱼ₋₁)
	(; w, r, m) = prices_etcⱼ₋₁
	aⱼ₋₁ = (aⱼ * (1-m) + cⱼ₋₁ - w * y_prev)/(1+r)
end

# ╔═╡ 9b244937-36f3-4d44-b509-6f7d5bef1bca
md"""
Why not add the bequests to the age dependent income?
"""

# ╔═╡ b04b20f2-8397-415d-bd7f-567f5f175e83
housing_supply(p, (; L̄, α̃)) = (α̃ * p)^(α̃/(1-α̃)) * L̄

# ╔═╡ ff35e19b-8556-4a2d-9ab8-96e305b04ced
house_price(X, (; L̄, α̃)) = (X / L̄)^((1-α̃)/α̃) / α̃

# ╔═╡ 01b044f2-fb06-497e-9595-3be278a1c57e
wage(K, L, (; α, Θ, δ))          = K ≤ 0.0 ? NaN : (1-α) * Θ * (K/L)^(α)

# ╔═╡ f5847dfa-d4ae-4d1a-b517-f166465e320a
interest_rate(K, L, (; α, Θ, δ)) = K ≤ 0.0 ? NaN : α * Θ * (K/L)^(α-1) - δ

# ╔═╡ 314231da-5294-4a12-a4fb-5023dbbd31cf
inverse_interest_rate(r, L, (; α, Θ, δ)) = ((r + δ)/(α * Θ))^(1/(α - 1)) * L

# ╔═╡ 08dd6081-9eb4-41ec-b0e1-bea6a4a585ea
function guess(::HousingModel, par) 
	K_guess = 1 * inverse_interest_rate(par.r , par.L, par)
	H_guess = 1 * housing_supply(par.p, par) / par.δ

	[K_guess, H_guess, par.L]
end

# ╔═╡ 3b65a9ac-9675-46cf-a545-ae3dc3f1daa7
output(K, L, (; α, Θ, δ)) = K ≤ 0.0 ? NaN : Θ * K^α * L^(1-α)

# ╔═╡ e11ae85f-c6f6-4381-b5d3-d756124f1c74
function prices_from_guesses_nt(::HousingModel, guesses, par)
	(; δ) = par

	(; K_supply, H_hh, L_eff) = guesses

	r = interest_rate(K_supply, L_eff, par)
	w = wage(K_supply, L_eff, par)
	p = house_price(δ * H_hh, par)
	
	prices = (; p, r, w)

	(; prices, guesses)
end

# ╔═╡ 61ff86de-de16-49af-a9a5-6c7c2b23420f
function prices_from_guesses(HM::HousingModel, guesses, par)
	(; δ) = par

	K_guess, H_guess, L_guess = guesses
	guesses_nt = (; K_supply = K_guess, H_hh = H_guess, L_eff = L_guess)

	prices_from_guesses_nt(HM, guesses_nt, par)
end

# ╔═╡ 8ae3665e-bba7-4032-93a7-9c546d347f60
function get_π_initXX(::HousingModel, sol₀, price_paths, j_init, (; δ))
	if j_init == 0
		 return sol₀.π[j = At(j_init)]
	else
		
		(; other) = sol₀.sim[j = At(j_init-1)]
		(; hⱼ, aⱼ₊₁) = other
		(; p, r) = price_paths[t = At(0)]
		ωⱼᵢₙᵢₜ = p * hⱼ * (1-δ) + (1+r) * aⱼ₊₁

		return state_init = ωⱼᵢₙᵢₜ
	end
end

# ╔═╡ 36bde726-ac0f-4e38-baec-b507ecb0b9d1
md"""
# Helpers
"""

# ╔═╡ 9ae9deb1-b9ee-4c8b-b1a4-8c75323a56cb
md"""
### Adjusting the period ``P``

* depreciation rate ``\delta_P = 1 - (1-\delta)^P``
* discount factor ``\beta``
* ``h``
* ``J``
"""

# ╔═╡ 9ffedac3-0ef1-42f6-932e-01f74b46531c
function show_vector(x)
	@info join(repr.(x), ", ") |> Base.Text
end

# ╔═╡ 93b9f4a0-ec6c-481d-9b8a-5f04401f11cf
md"""
## Demographics
"""

# ╔═╡ d637ff0e-cd32-42c7-a9f2-8e2e4e3b3bbd


# ╔═╡ 742da011-57db-41b0-8f64-9fcecd4c0324
p_surv₀ = DimVector(
	[0.9945385, 0.9995935, 0.9997525, 0.999799, 0.999836, 0.9998605, 0.9998755, 0.999885, 0.99989, 0.99989, 0.999884, 0.9998745, 0.9998525, 0.9998115, 0.99975, 0.9996605, 0.999536, 0.9993885, 0.999241, 0.9991345, 0.99906, 0.998978, 0.9988925, 0.99881, 0.9987215, 0.998631, 0.9985435, 0.9984545, 0.998359, 0.998259, 0.998161, 0.9980625, 0.997962, 0.997868, 0.9977805, 0.997691, 0.9975995, 0.997493, 0.997366, 0.997226, 0.997077, 0.99692, 0.9967525, 0.9965905, 0.996419, 0.9962185, 0.995971, 0.995691, 0.9953685, 0.9950285, 0.994659, 0.9942635, 0.993815, 0.993334, 0.9927975, 0.9921995, 0.9915535, 0.9908825, 0.990155, 0.9893715, 0.988523, 0.987618, 0.9866885, 0.985767, 0.9848455, 0.983935, 0.982972, 0.9818665, 0.980645, 0.9793075, 0.9778105, 0.9760855, 0.9741015, 0.971813, 0.9691475, 0.9657655, 0.9623835000000001, 0.958681, 0.9545755, 0.9497485, 0.9445295, 0.9388595, 0.9326274999999999, 0.9255175, 0.9172435, 0.907863, 0.8973555, 0.885727, 0.873394, 0.859642, 0.844195, 0.827184, 0.8087880000000001, 0.78986, 0.7707634999999999, 0.7515835, 0.732595, 0.7140934999999999, 0.6963895, 0.6798, 0.662296, 0.6438275, 0.6243405, 0.6037785, 0.5820815, 0.559187, 0.5350275, 0.509533, 0.4826284999999999, 0.454237, 0.42427349999999997, 0.39265149999999993, 0.35927850000000006, 0.32405700000000004, 0.28970799999999997, 0.25419400000000003, 0.21690299999999996, 0.17774900000000005, 0.13663599999999998, 0.093468, 1.0],
	Dim{:j}(0:120)
)

# ╔═╡ 3768ac04-c428-4f45-80b9-8cb6966c72aa
p_surv(age) = p_surv₀[j = At(age)]

# ╔═╡ a09536e2-5095-42e8-ab26-3b00e4b4cdce
dp_marcelo2 = [
	0.000894302696081951, 0.000954208212234048, 0.000989840925560537, 0.000996522526309545, 0.00098215260061939,
	0.000959551106572388, 0.000942388041116207, 0.000935533446389084, 0.000946822022702617, 0.00097378267030598,
	0.00100754405484986,  0.0010463061900096,   0.00109701785072833, 0.00116237295935761,  0.00124365648706804,
	0.00133574435463189,  0.0014410461391004,   0.0015673411143621, 0.00171380631074604,  0.0018736380419753,
	0.00203766165711833,  0.00220659167333691,  0.00238942699716915, 0.00259301587170481,  0.00281861738406178,
	0.00306417992710891,  0.00332180268908611,  0.00358900693685323, 0.00386267209667191,  0.00414777667611931,
	0.00445827861595176,  0.00479990363846949,  0.00516531829562337, 0.00555390618653441,  0.00597132583819979,
	0.00642322495833418,  0.00692461135042076,  0.00749557575640038, 0.0081595130519956,   0.00892672789984719,
	0.00982654537395458,  0.010830689769232,    0.0118723751877809, 0.0128914065482476,   0.0139080330996353,
	0.0150030256703387,   0.0162668251372316,   0.0176990779563976, 0.0193202301703282,   0.0211079685238627,
	0.0229501723647085,   0.0249040093508705,   0.0271512342884117,   0.0297841240612845,   0.0327533107326732,
	0.0358306701555879,   0.0389873634123265,   0.0425026123367764,   0.0465565209898809,   0.0511997331749049,
	0.0563354044485466,   0.0618372727625817,   0.0678564046096954,   0.0745037414774353,   0.0819753395107449,
	0.0896822973078052,   0.0980311248111166,   0.107059411952568,    0.116803935241159,    0.127299983985204,
	0.138580592383723]

# ╔═╡ aabbd49a-a1cb-4054-8beb-4b542957dbfa
function mortality(model; m = 1/45, age_min = 0, age_max = 100)
	J = age_max - age_min
	j_dim = Dim{:j}(0:J)
	
	if model == :perpetual_youth
		return DimVector([fill(m, J); 1.0], j_dim, name = :m)
	elseif model == :lifecycle
		ages = Dim{:age}(0:119)
		p₀ = p_surv.(ages)
		p₀ = p₀[age = At(age_min:age_max-1)]
		J = length(p₀)
		j_dim = Dim{:j}(0:J)
		
		return DimVector([1 .- p₀; 1.0], j_dim, name = :m)
	elseif model == :marcelo
		J = length(dp_marcelo2)
		return DimVector([dp_marcelo2; 1.0], Dim{:j}(0:J), name = :m)
	else
		@error "model ∉ [:perpetual_youth, :lifecycle]. Please fix!"
	end
	
end

# ╔═╡ 31ed0294-5b87-4697-b1d2-a675b93dadb2
md"""
## Income profile
"""

# ╔═╡ ada6fdbc-a135-4814-8b44-2b89a24c1699
"""
	`J` ... age of death
    `JR` ... retirement age
"""
function income_profile(J, JR)
	@assert JR < J

	y = [3e-06 * j^3 - 0.0012 * j^2 + 0.0589 * j + 0.9503 for j ∈ 1:JR]

	y = [y; fill(y[end], J - JR)]

	DimArray(y, Dim{:j}(0:J-1), name = :y)
end

# ╔═╡ eb0848c3-d4d1-4088-a4d5-2809dac09a4a
function simple_income_profile(J, JR; y=1.0, yR = 0.0)
	y = [fill(y, JR); fill(yR, J - JR)]

	DimArray(y, Dim{:j}(0:J-1), name = :y)
end

# ╔═╡ 306cc046-cd55-435a-9a8f-7ca7d0c158c8
md"""
## Statespace
"""

# ╔═╡ 4107c1bd-44f3-42f8-b496-d0eff27b40fd
function simple_initial_distribution(statespace)
	(; dims, ε_grid, grid) = statespace

	n = length(grid)
	n_half = n ÷ 2
	
	#a₀ = a_grid[1]
	
	π₀ = zeros(dims, name = :π₀)
	π₀[state = 1:n_half] .= (1/n_half) ./ length(ε_grid)
	π₀
end

# ╔═╡ 43e3c9c8-8728-44c4-b8b1-fdc1220c4d17
function trivial_initial_distribution(statespace; init_state)
	(; dims, ε_grid, grid) = statespace
	
	π₀ = zeros(dims, name = :π₀)
	π₀[state = Near(init_state)] .= 1.0 ./ length(ε_grid)
	π₀
end

# ╔═╡ bcd59e5d-801f-4465-832f-b880fb0b73e8
function exponential_grid(amin,amax,na)
	return exp.(range(0.0, stop=log(amax-amin+1.0), length = na)) .+ amin .- 1.0
end

# ╔═╡ 4fd9f46e-8eb8-42da-be0f-eb06930e494a
function no_income_risk()
	MarkovChain(ones(1,1), [1.0])
end

# ╔═╡ f1e66bbb-3b6d-4aa2-895e-6604dd5bb0ae
function no_income_risk2()
	MarkovChain([0.9 0.1; 0.2 0.8], [0.9999, 1.0001])
end

# ╔═╡ 810a4a12-0797-4e4b-bd0a-134873430ca3
function simple_income_risk()
	MarkovChain([0.9 0.1; 0.2 0.8], [0.85, 1.15])
end

# ╔═╡ d9161c2d-e491-4c25-aea9-088661009ff2
function default_income_process(;
	ρ = 0.966, 	          #persistence of HH income process
    σ = (1.0-ρ^2)^0.5 * 0.5, #variance for household income process
	n = 7,
	normalize_mean = true
)
	if n > 1 && σ > 0
		log_rouwenhorst(n, ρ, σ; normalize_mean)
	else
		no_income_risk()
	end
end

# ╔═╡ b274b8b0-f343-49d7-8d0b-e8802c9d1500
function log_discr_AR1(args...; normalize_mean = true, method = QuantEcon.tauchen, period = 1)
	mc₀ = method(args...)
	#@info mc₀.state_values
	state_values 	= exp.(mc₀.state_values)

	π∞ = QuantEcon.stationary_distributions(mc₀) |> only

	if normalize_mean
		𝔼y = mean(state_values, weights(π∞))
		state_values ./= 𝔼y
	end
	
	return MarkovChain(mc₀.p ^ period, state_values)
end

# ╔═╡ 16a46962-f6be-4536-a2f2-f52fb796d2da
function β_AMMR(; age_min = 20, age_max = 96, β̄ = 0.9655, ξ = 0.00071)

	J = age_max - age_min
	 
	β = DimVector(
		[exp(j * log(β̄) + ξ * (j - (40 - age_min))^2) for j ∈ 0:J],
		Dim{:j}((0:J))
	)

	β[j = At(0:J-1)]

end

# ╔═╡ 86a97101-02b6-4b36-b458-ceae742d5c65
"See Calibration table and E.4 Calibration details"
function ε_chain_AMMR(σ_scale = 1.0; n = 11, n_std = 3.0, period = 1)
	ρ = 0.91
	σ_y = 0.91
	σ = σ_scale * √(1-ρ^2) * σ_y
	mc = log_discr_AR1(n, ρ, σ, 0.0, n_std; period)
end

# ╔═╡ ff84590d-8e81-47b4-bdd0-8271489be19c
"See Calibration table and E.4 Calibration details"
function θ_chain_AMMR(σ_scale = 1.0; n = 3, n_std = 3.0)
	ρ = 0.677
	σ_y = 0.61
	σ = σ_scale * √(1-ρ^2) * σ_y
	mc = log_discr_AR1(n, ρ, σ, 0.0, n_std)
end

# ╔═╡ 2a2742c1-a812-4a7b-b8fd-bfc195ac2c99
function get_permanent_states(mc, dimname = :θ)
	permanent_dim = Dim{dimname}(mc.state_values)
	π = only(stationary_distributions(mc))
	π_permanent = DimVector(π, permanent_dim)

	(; mc_permanent=mc, π_permanent)
end

# ╔═╡ 33f2a5c0-5bd5-4d7f-95ba-03035206fde6
function no_permanent_states2()
	mc = MarkovChain([0.9 0.1; 0.2 0.8], [0.9999, 1.0001])
	get_permanent_states(mc)
end

# ╔═╡ 081d029a-d67d-46a4-bf87-8a55f516fa30
no_permanent_states() = get_permanent_states(no_income_risk(), :θ)

# ╔═╡ 3ecb48cc-32c3-49f5-96bf-aad651d81e30
function permanent_states_AMMR(args...; kwargs...)
	mc_permanent = θ_chain_AMMR(args...; kwargs...)
	get_permanent_states(mc_permanent)
end

# ╔═╡ 47232117-6797-4188-a0e6-8a2e2a245e66
no_inheritances((; j_dim), (; perm_dim)) = zeros((perm_dim..., j_dim), name = :inheritances)

# ╔═╡ a49de89e-ff8d-4447-9dcd-491bb0126db6
no_permanent_states()

# ╔═╡ d0420ab9-27c8-4686-9f59-f643be9f6410
md"""
## After tax income

* ``\theta``: persistent state (related to intergenerational transfers)
* ``\varepsilon``: transitory component of productivity
* ``\bar h``: lifecycle component of productivity (_human capital_)
* ``\tau`` and ``\bar d`` are parameters

``(1-\tau)(1-\rho) \theta \varepsilon \bar h + \underbrace{\rho \theta \bar d}_{\text{tr}}``

where
```math
1 - \rho = \begin{cases} 1 & \text{if } j \leq J_R, \\
\frac{\bar h}{\bar h_R} & \text{ if } j > J_R.
\end{cases}
```
"""

# ╔═╡ d645fe54-fed9-49c6-b20a-94239ca7eb7e
md"""
Effective labor supply ``\iff`` productivity

``\ell_{\text{eff}} := \theta \varepsilon \bar h``
"""

# ╔═╡ a8da0aea-9b8d-455a-a87e-afa846fa771d
function sprint_dimstack(ds)
	string = ""
	for (key, val) ∈ pairs(ds)
		string *= "  DimVector([\n    " * join(repr.(val), ", ") * "\n  ], Dim{:t}(0:$(length(val)-1)), name = :$key),\n"
	end
	"DimStack(\n" * string * ")" |> Base.Text
end

# ╔═╡ 32faac51-9b53-49a8-87da-61860d3f227a
md"""
# Tests
"""

# ╔═╡ 33d17a66-cb65-4b8c-a269-f7a3f82ac32e
md"""
## Stationary equilibrium – no risk
"""

# ╔═╡ adce8682-ef5a-42ea-a59b-15df0cf1685f
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
out_test = let
	# equivalent to BaselineModel() in models_reduced.jl
	(; par, statespace, π_init) = get_cali_test(amax = 15, na = 100)

	Mo = HousingModel()

	K_guess = 3.5832334751343167
	guesses = (; K_supply = K_guess, H_hh = 8.55e-8, L_eff = 1.0)

	prices = let
		r = interest_rate(K_guess, 1.0, par) 
		w = wage(K_guess, 1.0, par)
		(; r, w, p = 1.2)
	end

	#######################
	# Partial equilibrium #
	#######################
	
	out = stationary_PE(Mo, (; prices.r, par...), statespace, 
						guesses, prices; 
						π_init) # j_last XXX

	(; aggregates, prices) = out
	(; K_supply) = aggregates.updated
	(; K_hh, ζ) = aggregates.aggregates
	(; state, c, ℓ_eff, a_next) = out.raw_aggregates
	(; r) = prices  

	@info @test state  ≈ 5.10668547068924
	@info @test a_next ≈ 5.168198036558206
	@info @test c      ≈ 1.721938385022462
	@info @test ℓ_eff  ≈ 1.5451965545805213

	#######################
	# General equilibrium #
	#######################
	
	out = stationary_GE(Mo, (; prices.r, par...), statespace, 
						#=guesses, prices=#; 
						π_init,
					    tol = 1e-8, λ = 0.25, details = 10) # j_last XXX

	(; aggregates, prices) = out
	(; K_supply) = aggregates.updated
	(; K_hh, ζ) = aggregates.aggregates
	(; state, c, ℓ_eff, a_next) = out.raw_aggregates
	(; r) = prices  

	@info @test state  ≈ 7.274069538405238
	@info @test a_next ≈ 7.375175510982203
	@info @test c      ≈ 1.7859332121683014
	@info @test ℓ_eff  ≈ 1.5451965545805213

	out
end
  ╠═╡ =#

# ╔═╡ ce458e11-8612-4c39-9f16-0413edc16586
#=╠═╡
visualize_stationary(out_test)
  ╠═╡ =#

# ╔═╡ 3edf748d-e2fc-4148-82f4-dfcd541d5901
md"""
## Transition path – no risk (reduce mortality)
"""

# ╔═╡ 0d4205f1-e578-4d9b-9260-0a174d89fc45
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
out_auclert_trans0 = let
	(; par, statespace, prices, π_init) = get_cali_auclert(; ξ = 0.0, risk = false)

	model = HousingModel()

	guesses = (; K_supply = 20.75270574911025, H_hh = 0.0, L_eff = 2.2591478197110324)
	(; prices) = prices_from_guesses_nt(model, guesses, par)

	#par = (; prices.r, par...)
	GE₀ = stationary_PE(model, par, statespace, guesses, prices; π_init) 
#	GE₀ = stationary_GE(model, par, statespace; guesses, #=prices,=# π_init) 
	
	
	T̃ = 300

	###########################
	## TEST 1: CONSTANT PATH ##
	GE₁ = GE₀
	GEs = (; GE₀, GE₁, statespace, j_last = par.J)
	
	guess = GE₀.aggregates.updated
	guessed_path = dimstack_from_nt(guess, Dim{:t}(0:T̃))

	demographics = let
		m₀ = par.m
		j_dim = DD.dims(m₀, :j)
		J = maximum(j_dim)
	
		borns = -J:1:T̃
		born_dim = Dim{:born}(borns)
		ms = cat(fill(m₀, born_dim)..., dims = born_dim)

		demo = DimStack(ms, )
	end
	###########################
	
#	setup = (; guessed_path, T̃, demographics, GEs, statespace, model)

	inheritances = no_inheritances(par, statespace)

#	price_paths = get_price_paths(model, paths_in, par; GE₀)
#	(; π_permanent, state_dim, perm_dim) = statespace

	out = transition_GE(model, T̃, par, statespace, demographics, GE₀, guessed_path;
						normalize_population = false, inheritances, details = 1, maxiter = 3)
	

end
  ╠═╡ =#

# ╔═╡ 139b6f9a-4073-40a3-b587-6af681e47069
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
out_auclert_trans2 = let
	#(; par, statespace, π_init) = get_cali_test()
	(; par, statespace, prices, π_init) = get_cali_auclert(ξ = 0.15, risk = false)

	model = HousingModel()

	guesses = (; K_supply = 19.8058, H_hh = 2.7714, L_eff = 2.2591478197110324)
	prices = let
		r = interest_rate(guesses.K_supply, guesses.L_eff, par) 
		w = wage(guesses.K_supply, guesses.L_eff, par)
		p = house_price(par.δ * guesses.H_hh, par)
		(; r, w, p)
	end

	GE₀ = stationary_GE(model, par, statespace#=, guesses, prices=#; guesses, π_init, tol = 1e-4) 
	
	#@info GE₀.aggregates
	#@info GE₀.prices
	
	T̃ = 100

	#####################################
	## TEST 2: REDUCE MORTALITY BY 10% ##
	
	

	#guessed_path = guess_auclert_trans
	
	demographics = let
		m₀ = par.m
		m₁ = 0.9 * par.m
		
		j_dim = DD.dims(m₀, :j)
		J = maximum(j_dim)
	
		borns = -J:1:T̃
		born_dim = Dim{:born}(borns)
		ms = DimArray(cat([m₁ for born ∈ born_dim]..., dims = born_dim), name = :m)

		
		demo = DimStack(ms, )
	end
	###########################
	
	inheritances = no_inheritances(par, statespace)

	guess = GE₀.aggregates.updated
	guessed_path₀ = dimstack_from_nt(guess, Dim{:t}(0:T̃))
	
	if false
		guessed_path = guessed_path₀
	elseif false
		#guess = GE₀.aggregates.updated
		#guessed_path₀ = dimstack_from_nt(guess, Dim{:t}(0:T̃))
		
		# initial run for guess
		out = transition_PE(model, T̃, par, statespace, demographics, GE₀, guessed_path₀;
							normalize_population = false, inheritances,
							#details = 1, λ = 0.001, maxiter = 200
							)
		mass_terminal = out.raw_aggregate_paths.population[end]
		
		par₁ = deepcopy(par)
		par₁.m .*= 0.9
	
		GE₁ = stationary_GE(model, par₁, statespace#=, guesses, prices=#; guesses, π_init, tol = 1e-4, total_mass = mass_terminal) 
	
		init = GE₀.aggregates.updated
		term = GE₁.aggregates.updated
	
		@info (; init, term)
		
		L_guess = copy(out.aggregate_paths.updated.L_eff)
	
		pattern = L_guess .- L_guess[begin]
		pattern = pattern ./ pattern[end]
	
		#@info lines(pattern)
	
		K_guess = DimVector(
			pattern .* (term.K_supply - init.K_supply) .+ init.K_supply,
			name = :K_supply
		)
		H_guess = DimVector(
			pattern .* (term.H_hh - init.H_hh) .+ init.H_hh,
			name = :H_hh
		)
		
		guessed_path = DimStack(K_guess, H_guess, L_guess)
	else
		guessed_path = updated_paths_auclert_trans2 # ξ = 0.15, risk = false
#		guessed_path = current_paths_auclert_trans2#[t = At(0:T̃)]
	end

	out = transition_GE(model, T̃, par, statespace, demographics, GE₀, guessed_path;
						normalize_population = false, inheritances,
						details = 1, λ = 0.001, maxiter = 3, tol = 5e-3
						)
end
  ╠═╡ =#

# ╔═╡ 37c98a61-3350-4eb4-a3fe-8cd8c2b5eab5
# ╠═╡ disabled = true
#=╠═╡
@chain out_auclert_trans2.price_paths begin
	DataFrame
	stack(Not(:t))
	data(_) * mapping(:t, :value, layout = :variable) * visual(ScatterLines)
	draw(; facet = (; linkyaxes = false ))
end
  ╠═╡ =#

# ╔═╡ 52df027a-b50a-4a3a-99dc-80bce86c77cc
h̄_marcelo = [ 
	1.000, 1.111, 1.222, 1.333, 1.444, 
	1.556, 1.711, 1.867, 2.022, 2.178, 
	2.333, 2.489, 2.645, 2.800, 2.956,
	3.111, 3.189, 3.267, 3.345, 3.422, 
	3.500, 3.578, 3.656, 3.733, 3.811,
	3.889, 3.837, 3.785, 3.733, 3.682, 
	3.630, 3.578, 3.526, 3.474, 3.422,
	3.370, 3.189, 3.008, 2.826, 2.645, 
	2.463, 2.282, 2.100, 1.919, 1.737, 
	1.556, 1.452, 1.348, 1.244, 1.141, 
	1.037, 0.933, 0.830, 0.726, 0.622,
	0.519, 0.484, 0.449, 0.415, 0.380, 
	0.346, 0.311, 0.277, 0.242, 0.207,
	0.173, 0.138, 0.104, 0.069, 0.035, 
	0.000, 0.000, 0.000, 0.000, 0.000,
	0.000
]

# ╔═╡ 31352b9e-69cc-4467-aa89-34249666e407
dp_marcelo = [ 
0.000940,
0.001022,
0.001108,
0.001190,
0.001278, 
0.001369, 
0.001456,
0.001545,
0.001641,
0.001741,

0.001839,
0.001938,
0.002038,
0.002132,
0.002220,
0.002309,
0.002401,
0.002507,
0.002634,
0.002774,

0.002923,
0.003080,
0.003247,
0.003409,
0.003581,
0.003781,
0.004029,
0.004309,
0.004632,
0.004972,
	  
0.005341,
0.005737,
0.006185,
0.006666,
0.007203,
0.007800,
0.008447,
0.009118,
0.009845,
0.010629,

0.011477,
0.012382,
0.013312,
0.014233,
0.015155,
0.016065,
0.017028,
0.018134,
0.019355,
0.020693,
	  
0.022190,
0.023914,
0.025899,
0.028187,
0.030852,
0.034234,
0.037617,
0.041319,
0.045425,
0.050252,
	  
0.055471,
0.061141,
0.067373,
0.074483,
0.082757,
0.092137,
0.102645, 
0.114273,
0.126606,
0.140358,

0.155805,
0.172816,
0.191212,
0.210140,
0.229237,
0.248417,
1.0 ]

# ╔═╡ 626e38d7-e337-4853-9046-3d9b1485bb24
# ξ = 0.15, risk = false
updated_paths_auclert_trans2 = DimStack(
  DimVector([
    5.254423423609724, 5.25781152913868, 5.261607656635533, 5.2657036806748145, 5.270006538620189, 5.2744373270669405, 5.278930022618655, 5.283430076965028, 5.2878930175061685, 5.292283118496953, 5.296572176912558, 5.300738403628457, 5.3047654324791, 5.308641448521755, 5.312358428769305, 5.315911486752455, 5.319298310281628, 5.322518681007375, 5.325574063035442, 5.3284672522901255, 5.33120208089429, 5.3337831702288865, 5.336215724254311, 5.338505356134709, 5.3406579451870195, 5.342679522853061, 5.344576180958305, 5.3463540007607895, 5.3480189968895235, 5.349577073461437, 5.35103398999088, 5.352395335483391, 5.3536665096361595, 5.35485270997504, 5.355958923302791, 5.356989920503832, 5.357950254163309, 5.358844260797606, 5.359676064889732, 5.360449584452234, 5.361168537615245, 5.361836449868009, 5.362456661491827, 5.363032334814312, 5.363566461135109, 5.364061867388541, 5.364521222396373, 5.364947041535385, 5.3653416913111975, 5.365707393740698, 5.366046230612542, 5.366360147795805, 5.366650959797099, 5.3669203545598005, 5.367169898532958, 5.3674010419371605, 5.367615124203408, 5.36781337880098, 5.3679969387046595, 5.36816684244993, 5.3683240407340405, 5.3684694034377385, 5.368603726979822, 5.368727741828836, 5.368842120197541, 5.368947484109788, 5.369044413770345, 5.3691334558340404, 5.369215130801847, 5.369289938695186, 5.369358362600641, 5.369420870126166, 5.369477913069169, 5.369529926006037, 5.369577324433374, 5.369620502977008, 5.369659833966862, 5.369695662502517, 5.369728301800882, 5.3697580359354395, 5.36978512233031, 5.3698097940192495, 5.369832261683302, 5.369852715483769, 5.369871326706411, 5.369888249226506, 5.369903620807919, 5.3699175642495725, 5.369930188380058, 5.369941588910338, 5.369951849156137, 5.369961040621022, 5.369969223448348, 5.36997644674458, 5.369982748760719, 5.369988156940571, 5.369992687814965, 5.369996346744273, 5.369999127477922, 5.3700010115282835, 5.370001967309377
  ], Dim{:t}(0:100), name = :H_hh),
  DimVector([
    18.496529729010636, 18.576182054129657, 18.647872150371, 18.712253304662344, 18.769986782248967, 18.821714479348493, 18.868042816246117, 18.909532142993612, 18.946693764644742, 18.979989934340374, 19.00983697558253, 19.036610483164917, 19.06064618679088, 19.08224307610347, 19.101666694101866, 19.119152035676276, 19.134907550017704, 19.149118729416227, 19.16195072537656, 19.173550399269935, 19.184046912947164, 19.193554731383216, 19.202175385972616, 19.20999984833724, 19.217107054002252, 19.22356660995707, 19.229440093789577, 19.234782036602883, 19.239641063098333, 19.244060743384072, 19.248080248916725, 19.25173478898406, 19.255055923836363, 19.258072214874566, 19.26080970609978, 19.263292138822337, 19.265541228753346, 19.267576713342553, 19.269416704504668, 19.271077915590958, 19.27257586171734, 19.273925010800014, 19.275138996463735, 19.27623068677544, 19.277212241228312, 19.278095107679935, 19.278890073101145, 19.27960719725985, 19.280255778867037, 19.280844316107444, 19.281380520270982, 19.281871221359427, 19.282322396661776, 19.28273918791341, 19.28312596707576, 19.28348638603343, 19.28382343571143, 19.28413948370151, 19.2844362406661, 19.284714849861825, 19.28497598470027, 19.285219939859303, 19.28544673581622, 19.28565622386327, 19.285848047488617, 19.28602163274078, 19.286176347575523, 19.286311672714607, 19.286427585673835, 19.286524701010485, 19.286604337453376, 19.28666833065972, 19.28671889208899, 19.2867583412542, 19.286788907622732, 19.28681258128577, 19.2868310833141, 19.28684886138449, 19.286865985085456, 19.28688251778648, 19.286898516993954, 19.286914034699432, 19.286929117753218, 19.286943808290943, 19.286958144228638, 19.286972159835123, 19.286985886381277, 19.28699935286232, 19.287012586800227, 19.28702561512369, 19.287038465124457, 19.287051165500117, 19.287063747468242, 19.28707624597848, 19.287088701081696, 19.287101159260697, 19.287113674881848, 19.28712631164716, 19.287139143927554, 19.28715225762803, 19.287165749593793
  ], Dim{:t}(0:100), name = :K_supply),
  DimVector([
    2.2591478197110297, 2.2606422150542365, 2.2620314018975436, 2.26332367582268, 2.264526558571391, 2.2656468765399898, 2.266690826464481, 2.2676639670918664, 2.26857136034112, 2.269417629382797, 2.2702070042728755, 2.270943358772978, 2.2716302367914003, 2.2722708777187344, 2.2728682385566636, 2.2734250196369645, 2.2739436922938796, 2.2744266441982637, 2.27487606043097, 2.2752939504107066, 2.2756821660737647, 2.2760424131762984, 2.276376264935031, 2.2766851800415933, 2.2769705178605757, 2.277233546064232, 2.2774754516995763, 2.277697555964909, 2.277901095516954, 2.2780872289620127, 2.2782570401606934, 2.2784115486428873, 2.2785517147450305, 2.2786784481119087, 2.2787926125164812, 2.2788950277054028, 2.278986471975148, 2.279067877501116, 2.2791401205156308, 2.2792040256696455, 2.2792603666148974, 2.2793098731225214, 2.279353231054132, 2.279391086240672, 2.279424042961282, 2.279452667897886, 2.2794774875487236, 2.279498888944551, 2.279517235772413, 2.279532868989991, 2.2795461064080453, 2.279557246209925, 2.2795665662128006, 2.2795743228162726, 2.279580754106806, 2.279586079338586, 2.2795904979995685, 2.279594119851546, 2.279597047651818, 2.2795993761257636, 2.2796011939803433, 2.2796025817629424, 2.2796036136951368, 2.279604355918271, 2.2796048682349714, 2.279605203372202, 2.279605406535169, 2.2796055168702933, 2.2796055663511954, 2.2796055811632083, 2.2796055811632114, 2.2796055811632066, 2.279605581163207, 2.279605581163212, 2.279605581163204, 2.2796055811632088, 2.2796055811632057, 2.2796055811632097, 2.2796055811632057, 2.2796055811632057, 2.2796055811632034, 2.279605581163211, 2.279605581163204, 2.279605581163208, 2.2796055811632114, 2.2796055811632043, 2.279605581163205, 2.279605581163209, 2.27960558116321, 2.279605581163206, 2.279605581163207, 2.27960558116321, 2.279605581163206, 2.2796055811632034, 2.2796055811632048, 2.2796055811632097, 2.279605581163213, 2.279605581163203, 2.279605581163205, 2.2796055811632105, 2.2796055811632074
  ], Dim{:t}(0:100), name = :L_eff),
)

# ╔═╡ f103ea34-ad3c-49e1-9e6f-d94d4b3574e6
md"""
## Stationary equilibrium – with risk
"""

# ╔═╡ 3303493e-6502-479f-b383-82ffd1ab17bf
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
out_auclert_GE_risk = let
	#(; par, statespace, π_init) = get_cali_test()
	(; par, statespace, prices, π_init) = get_cali_auclert(ξ = 0.15, risk = true, na = 100)

	model = HousingModel()

	#guesses = (; K_supply = 26.9834, H_hh = 6.09351, L_eff = 2.54539)
	guesses = (; K_supply = 5.9834, H_hh = 6.09351, L_eff = 2.54539)
	(; prices) = prices_from_guesses_nt(model, guesses, par)

	GE₀ = stationary_GE(model, par, statespace#=, guesses, prices=#; guesses, π_init, tol = 1e-4) 
	
	@info GE₀.aggregates
	@info GE₀.prices

	GE₀
end
  ╠═╡ =#

# ╔═╡ 765cfe47-b191-4228-b419-ffb6cd2bd9f4


# ╔═╡ 9c050921-f2cc-4e94-a366-6a519d7f4116
15.2 / 68.6 - 1

# ╔═╡ 208efe14-68dc-4389-bb9b-8bd996bf4749
#=╠═╡
visualize_stationary(out_auclert_GE_risk, [0.5, 0.9, 0.99, 0.999, 1.0])
  ╠═╡ =#

# ╔═╡ 32dd3921-c58a-41cb-9f11-8d4efa9f7641
md"""
## Transition path – with risk
"""

# ╔═╡ 32b95f4a-180c-4a61-b899-6b604d911926
# ╠═╡ disabled = true
#=╠═╡
@chain out_auclert_trans3.GE₀.sim_df begin
	#@subset(:born ∈ [-20, -10, 0, 10])
	@groupby(:j)
	@combine(:c = mean(:c, weights(:π)))
	data(_) * mapping(:j, :c
					  #, color = :born => nonnumeric
					  ) * visual(Lines)
	draw
end
  ╠═╡ =#

# ╔═╡ 9b706ccd-bc47-4cc8-b315-fc17375cbd93
let
	P = [0.25 0.75;
		0.5 0.5]

	i_z = [
		1 1
		2 2
		3 3
		4 4
	]

	i_ε = [
		1 2
		1 2
		1 2
		1 2
	]
	
	z_next = [
		0.1 0.2;
		0.2 0.3
		0.5 0.6
		0.2 0.7
	] # z x ε
	
	π = [
	 0.9  0.1
	 0.8  0.2
	 0.7  0.3
	 0.5  0.5
	]

	π_next = π * P

	df_check = DataFrame(; i_z = vec(i_z), i_ε = vec(i_ε), z_next = vec(z_next), π = vec(π), π_next = vec(π_next))

	df = @chain df_check begin
		#select(Not(:π_next))
		@groupby(:i_z)
		@transform(
			:π_next_2 =  @bycol P' * :π
		)
	end

end

# ╔═╡ 48443da6-0ffc-4ae2-9ebe-c4d0461ecfca
# 100: 0.01063
# 200: 

# ╔═╡ 998cb794-525c-4b2d-88c7-80d711e06d3b
md"""
let
	out = out_auclert_trans3
	
	(; GE₀, price_paths, statespace) = out
	prc_GE    = GE₀.prices
	prc_shock = price_paths[t = At(0)]

	next_state(h, a_next, (; p, r), (; δ)) = p * h * (1-δ)/(1+r) + a_next
		# p ... p_next; r ... r_next

	statespace
	
	tmp = @chain GE₀.sim_df begin
		#@transform(
			#:z_next_test = :ho * (1-GE₀.par.δ)/(1+prc_GE.r) * prc_GE.p + :a_next,
			#:next_state_test  = next_state(:ho, :a_next, prc_GE, GE₀.par),
			#:next_state_shock = next_state(:ho, :a_next, prc_shock, GE₀.par)
		#)

		# s

	 # slight deviations
		@groupby(:state, :j, :permanent)
		@transform(:π_new = @bycol statespace.P' * :π)
		@groupby(:j)#, :permanent)
		@combine(
			:state = mean(:state, weights(:π)),
			:next_state = mean(:next_state, weights(:π_new)),
		)
		maximum(abs.(_.state[2:end] ./ _.next_state[1:end-1] .- 1))
	# =#

	#=
		@groupby(:j, :ε, :permanent)
		#@transform(:π_new = @bycol statespace.P' * :π)
		#@groupby(:j)#, :permanent)
		@combine(
			:state = mean(:state, weights(:π)),
			:next_state = mean(:next_state, weights(:π)),
		)
		
		#@groupby(:j, :permanent)
		#@transform(:π_new = @bycol statespace.P' * :π)
		
		
		@groupby(:j, :permanent)
		@combine(
			:state,
			:next_state,
			#:π = :π / sum(:π),
			:next_state_2 = vec(:next_state' * statespace.P ), #.* (:π ./ sum(:π))),
			#:next_state_3 = vec(:next_state' * statespace.P')
		)
		# =#
		
		#=
		@aside @chain _ begin
			@transform(:next_state_shifted = @bycol [missing; :next_state_test[begin:end-1]])
			@subset(!ismissing(:next_state_shifted))
			#@info @test _.next_state_shifted ≈ _.state
		end

		@select(:j, :next_state_shock, :permanent, :ε, :π)
		@groupby(:j, :permanent)
		#@combine(
		#	:next_state_shock = Ref(DimVector(:next_state_shock, Dim{Symbol("ε")}(:ε))),
		#	:ε = Ref(DimVector(:ε, Dim{Symbol("ε")}(:ε))),
		#	#:π = Ref(DimVector(:π, Dim{Symbol("ε")}(:ε)))
		
	end
end
"""

# ╔═╡ b3219d9d-2bfe-445f-8f3a-4ccfa95673ed
function test_initial_state_of_transition(out)

	# check initial state during transition
	initial_df_transition = @chain out.sim_df begin
		@subset(:j + :born == 0)
		@groupby(:j, :permanent, :ε)
		@combine(
			:state = mean(:state, weights(:π)),
		) 
	end

	@chain initial_df_transition begin
		@subset(:j == 1)
		@info _
	end
	
	(; GE₀, price_paths, statespace) = out
	prc_GE    = GE₀.prices
	prc_shock = price_paths[t = At(0)]

	next_state(h, a_next, (; p, r), (; δ)) = p * h * (1-δ)/(1+r) + a_next
		# p ... p_next; r ... r_next

	statespace
	
	tmp = @chain GE₀.sim_df begin
		@transform(
			#:z_next_test = :ho * (1-GE₀.par.δ)/(1+prc_GE.r) * prc_GE.p + :a_next,
			:next_state_test  = next_state(:ho, :a_next, prc_GE, GE₀.par),
			:next_state_shock = next_state(:ho, :a_next, prc_shock, GE₀.par)
			
		)
		
		@aside @chain _ begin
			@subset(@bycol :j .< maximum(:j))
			@info @test _.next_state_test ≈ _.next_state
		end
		
		@groupby(:j, :permanent, :ε)
		@combine(
			:m = only(unique(:m)),
			:state = mean(:state, weights(:π)),
			:next_state = mean(:next_state, weights(:π)),
			:next_state_test = mean(:next_state_test, weights(:π)),
			:next_state_shock = mean(:next_state_shock, weights(:π)),
			:π = sum(:π)
				)
		

		@aside @chain _ begin
			@transform(:next_state_shifted = @bycol [missing; :next_state_test[begin:end-1]])
			@subset(!ismissing(:next_state_shifted))
			#@info @test _.next_state_shifted ≈ _.state
		end

		@select(:j, :next_state_shock, :permanent, :ε, :π)
		@groupby(:j, :permanent)
		@combine(
			:next_state_shock = Ref(DimVector(:next_state_shock, Dim{Symbol("ε")}(:ε))),
			:ε = Ref(DimVector(:ε, Dim{Symbol("ε")}(:ε))),
			#:π = Ref(DimVector(:π, Dim{Symbol("ε")}(:ε)))
		)
#		@transform(:next_state_shock = :next_state_shock' * parent(statespace.P))
#		flatten([:next_state_shock, :ε])
#		#unstack(:ε, :next_state_shock)
#		@select(:j = :j + 1, :state_shock = :next_state_shock, :permanent, :ε)
#		innerjoin(_, initial_df_transition, on = [:j, :permanent, :ε])
		#@transform(:test = :state / :state_shock .- 1)
		@subset(:j == 0)
		first
		#@
		#_.next_state_shock
	end

	choice = tmp.next_state_shock
	curr_ε = tmp.ε

	statespace.P[1,:]

	curr_i_ε = 3
	choice, mean(choice, weights(statespace.P[curr_i_ε,:]))
	# =#
end

# ╔═╡ ed3bf636-176a-422a-812d-07bc3be8fef4
#=╠═╡
test_initial_state_of_transition(out_auclert_trans3)
  ╠═╡ =#

# ╔═╡ d05e5a31-5a8b-4cad-85c5-5d6c89c67ad2
# ╠═╡ disabled = true
#=╠═╡
initial_df_transition = @chain out_auclert_trans3.sim_df begin
	@subset(:j + :born == 0)
	@groupby(:j)
	@combine(
		:state = mean(:state, weights(:π)),
	) 
end
  ╠═╡ =#

# ╔═╡ 4df66cbd-3c0d-40db-8b5b-65b26ffc4c79
#=╠═╡
(; r, p) = out_auclert_trans3.price_paths[t = At(0)]

  ╠═╡ =#

# ╔═╡ 31a39f30-740f-4a30-b3f5-e1fef2502138
# ╠═╡ disabled = true
#=╠═╡
@chain out_auclert_trans3.sim_df begin
	@subset(:born ∈ [-20, -10, 0, 5, 1, 60, 65])
	@groupby(:j, :born)
	@combine(:c = mean(:c, weights(:π)))
	unstack(:born, :c)
	@transform(
		:test1 = {"0"} / {"5"} - 1,
		:test2 = {"60"} / {"65"} - 1
	)
	
	#data(_) * mapping(:j, :c
	#				  , color = :born => nonnumeric
	#				  ) * visual(Lines)
	#draw
end
  ╠═╡ =#

# ╔═╡ 360d9593-4c34-4bbe-bde6-a554803861f1
#=╠═╡
out_auclert_trans3.price_paths.p |> lines
  ╠═╡ =#

# ╔═╡ 77d631d8-6da6-4ec3-a8d2-ebb71f9c474c
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
out_auclert_trans3 = let
	#(; par, statespace, π_init) = get_cali_test()
	(; par, statespace, prices, π_init) = get_cali_auclert(ξ = 0.15, risk = true, na = 200)

	model = HousingModel()

	guesses = (; K_supply = 26.9834, H_hh = 6.09351, L_eff = 2.54539)
	(; prices) = prices_from_guesses_nt(model, guesses, par)

	GE₀ = stationary_GE(model, par, statespace#=, guesses, prices=#; guesses, π_init, tol = 1e-4)
	
	@info GE₀.aggregates
	@info GE₀.prices
	
	T̃ = 100

	#####################################
	## TEST 2: REDUCE MORTALITY BY 10% ##
	
	

	#guessed_path = guess_auclert_trans
	
	demographics = let
		m₀ = par.m
		m₁ = 0.9 * par.m
		
		j_dim = DD.dims(m₀, :j)
		J = maximum(j_dim)
	
		borns = -J:1:T̃
		born_dim = Dim{:born}(borns)
		ms = DimArray(cat([m₁ for born ∈ born_dim]..., dims = born_dim), name = :m)

		
		demo = DimStack(ms, )
	end
	###########################
	
	inheritances = no_inheritances(par, statespace)

	#if true
		guess = GE₀.aggregates.updated
		guessed_path₀ = dimstack_from_nt(guess, Dim{:t}(0:T̃))
		
		# initial run for guess
		out = transition_PE(model, T̃, par, statespace, demographics, GE₀, guessed_path₀;
							normalize_population = false, inheritances,
							#details = 1, λ = 0.001, maxiter = 200
							)
		mass_terminal = out.raw_aggregate_paths.population[end]

		out

		
	#	par₁ = deepcopy(par)
	#	par₁.m .*= 0.9
	
	#	GE₁ = stationary_GE(model, par₁, statespace#=, guesses, prices=#; guesses, π_init, tol = 1e-4, total_mass = mass_terminal) 
	
	#	init = GE₀.aggregates.updated
	#	term = GE₁.aggregates.updated
	
	#	@info (; init, term)
		
	#	L_guess = copy(out.aggregate_paths.updated.L_eff)
	
	#	pattern = L_guess .- L_guess[begin]
	#	pattern = pattern ./ pattern[end]
	
		#@info lines(pattern)
	
	#	K_guess = DimVector(
	#		pattern .* (term.K_supply - init.K_supply) .+ init.K_supply,
	#		name = :K_supply
	#	)
	#	H_guess = DimVector(
	#		pattern .* (term.H_hh - init.H_hh) .+ init.H_hh,
	#		name = :H_hh
	#	)
		
	#	guessed_path = DimStack(K_guess, H_guess, L_guess)
	#else
	#	guessed_path = guessed_paths_auclert_trans2#[t = At(0:T̃)]
	#end
#
	#out = transition_GE(model, T̃, par, statespace, demographics, GE₀, guessed_path;
	#					normalize_population = false, inheritances,
	#					details = 1, λ = 0.001, maxiter = 1
	#					)
	#
		
end
  ╠═╡ =#

# ╔═╡ e443249e-dc9f-49b6-bd56-693fa1e49242
md"""
## Path with varying prices
"""

# ╔═╡ 066cd6cb-4bc2-418e-a04e-f9a83808de56
"from models-reduced.jl"
compare_path_no_annuities = [
	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3021001461299553, 0.6272372178419099, 0.9739164697929956, 1.340576799268494, 1.725590393516579, 2.1272617108683245, 2.5438224276573935, 2.973430821859388, 3.414174976604256, 3.8640708476262104, 4.321056057397279, 4.782981264165983, 5.247607733221523, 5.712611089567156, 6.175584361621404, 6.634037429016843, 7.085393997214947, 7.526983607713062, 7.956037183282303, 8.369681683194719, 8.764941264388371, 9.138746832813275, 9.487939942242765, 9.80926556784403, 10.09937020812086, 10.354805826287885, 10.57203451168989, 10.747443049832942, 10.877364571218873, 10.958106311948828, 10.985967179829842, 10.982402742894676, 10.94659649992422, 10.877790509939233, 10.775224378859846, 10.638155865708418, 10.465952227368366, 10.25820274648867, 10.014737288608618, 9.735668765362384, 9.42138663943396, 9.072439215979404, 8.68960692631083, 8.274132628216499, 7.827858548355433, 7.3531929973795735, 6.85284518713398, 6.329785527185949, 5.78758563208933, 5.230646670329047, 4.664265041726784, 4.094501956020336, 3.5279979567516886, 2.972110755757167, 2.434987816434748, 1.9257163512464759, 1.453521579409156, 1.0281208424031578, 0.6596191336504229, 0.3583793486847322, 0.13486764148453556],
	[0.0, 0.011664413010649466, 0.05796690636836921, 0.1379440193243593, 0.2505717382468018, 0.39475942358340066, 0.569350052550982, 0.7731236573188245, 1.0048021577413595, 1.2630501269441545, 1.5464776967059244, 1.8536378753353566, 2.1830196741882752, 2.5330448745343914, 2.902070225181003, 3.2883874451732025, 3.690222519208197, 4.105730708094011, 4.5329963142227765, 4.970036507787553, 5.414799680156648, 5.8651592840145295, 6.318905154287725, 6.773741454231985, 7.227291170479233, 7.677099831840108, 8.12063541477858, 8.555285649322494, 8.97835012590874, 9.387036179034691, 9.778453871244599, 10.149617842660874, 10.497457944941793, 10.818823888989387, 11.110478231976668, 11.369094986679386, 11.591264341472847, 11.773498104678982, 11.912245437657605, 12.003915849633575, 12.044908390567308, 12.031630059362534, 11.98583916882707, 11.906829103500733, 11.793959172463515, 11.646591207625368, 11.46411093988764, 11.246022250476829, 10.99206261625153, 10.702220944233861, 10.37677988311301, 10.016307498516356, 9.621534760293713, 9.193429861629248, 8.733432997939541, 8.2435939135015, 7.726534315797944, 7.185173479108396, 6.622685966408892, 6.042847023692834, 5.450261809223909, 4.850427403184362, 4.249594879435472, 3.6545763260997823, 3.0728814107309135, 2.512786812677585, 1.9834836109616178, 1.4942600800733783, 1.0548590417889527, 0.6753694070233838, 0.3660928972814077, 0.1373865258570226],
	[0.0, 0.0, 0.02878263121256408, 0.09124727346139583, 0.18631826891729042, 0.3128542438713473, 0.46964868068832977, 0.6554337425004271, 0.8688854905128032, 1.1086249276701947, 1.3732212089571565, 1.6611891575975999, 1.9709825490478454, 2.3009911302562083, 2.649543155869919, 3.0149056881123384, 3.3952841807063328, 3.7888177432718253, 4.193579232028452, 4.607579429160268, 5.028765688781634, 5.455016058037586, 5.8841309113150855, 6.313831288058482, 6.7417638047088335, 7.165504757057488, 7.5825603826951085, 7.99036452639402, 8.386271135014933, 8.767550571210782, 9.131385058581255, 9.474870991461286, 9.79502996292854, 10.08881377495744, 10.353097848354805, 10.584680252264878, 10.780286793145645, 10.93657678810305, 11.050158968857733, 11.117614449084874, 11.135525650837884, 11.100494409390143, 11.034658947169167, 10.937496680747705, 10.808555538486724, 10.647391454265717, 10.453589420495593, 10.226855799723925, 9.967130764501773, 9.674604537495117, 9.349757432202296, 8.993350203021434, 8.606303114954427, 8.189767554899502, 7.745353328595465, 7.275259907387698, 6.782236674130687, 6.269313598035609, 5.739758727471693, 5.197411478926785, 4.646901308231814, 4.093703253258148, 3.54399995953557, 3.0044914569593684, 2.4825238911532983, 1.9861527308172855, 1.5242805640914048, 1.105866534165128, 0.7402691562714776, 0.4371398601242913, 0.2062934054808705, 0.05755589915170001],
]

# ╔═╡ 4491634a-deff-4805-a7e5-9e03e9198d53
md"""
# Appendix
"""

# ╔═╡ 5cf0f0a8-4fa9-452d-9d58-4bccc1e42e5d
TableOfContents()

# ╔═╡ f49d0c50-2cfa-4cfb-9224-6607fb7f99eb
fonts = (; regular = Makie.MathTeXEngine.texfont(:regular), bold = Makie.MathTeXEngine.texfont(:regular))

# ╔═╡ e59b1142-1725-429c-892d-dbc70d8b035e
figure(size = (350, 250); figure_padding = 2, kwargs...) = (; size, fonts, figure_padding, kwargs...)

# ╔═╡ b8e2f4d8-4a8e-469f-8b15-0533f01246a6
let
	fig = Figure(; figure((400, 150))...)

	lines(fig[1,1], simple_income_profile(100, 50), axis = (; title = "Simplest income profile"))
	lines(fig[1,2], income_profile(120, 41), axis = (; title = "Simple income profile"))
	fig
end

# ╔═╡ a2926b44-77ec-42fb-979e-b6d23a8e56b8
function visualize_stationary((; sim_df), quantiles = [0.2, 0.5, 0.8])
	vars = [:z_next, :a_next, :ho, :c, :income, :m]

	df = select(sim_df, vars..., :π, :j, :ε)
	
	@chain df begin
		stack(vars, [:π, :j])
		@groupby(:variable, :j)
		@combine(
			:q = quantiles,
			:value = quantile(:value, weights(:π), quantiles))
		data(_) * mapping(:j, :value, 
						  color = :q => nonnumeric,
						  layout = :variable
						 ) * visual(Lines)
		draw(; facet = (; linkyaxes = false), figure = figure((500, 250)))
	end
end

# ╔═╡ ee7e5f71-5182-4b8d-9e6f-aa651cb9c1ca
criterion(a, b) = (a - b)/(1 + max(abs(a), abs(b)))

# ╔═╡ c500744d-c0cd-4dc1-a04a-7dd3f1ce860c
function _aggregates_(K_guess, par, K_hh, L_eff)
	(; bonds2GDP, NFA2GDP) = par
	
	K_in = only(K_guess)
	GDP = output(K_in, L_eff, par)
	B₀ = bonds2GDP * GDP
	NFA = NFA2GDP * GDP

	K_supply = K_hh - B₀ - NFA
	ζ = K_supply - K_in
	ℓ = criterion(K_supply, K_in)

	aggregates = (; ℓ, ζ, 
				  GDP, B₀, NFA,
				  wealth2GDP = K_hh/GDP, capital2GDP = K_supply/GDP,
				  K_hh, K_supply, K_in)
end

# ╔═╡ 9e6a5abd-fad3-43e7-b4ba-2d8adaf19a9b
function loss_and_aggregates(::HousingModel, par, guesses, raw_aggregates)
	H_guess = guesses.H_hh
	K_guess = guesses.K_supply

	H_hh = raw_aggregates.ho
	#K_hh = raw_aggregates.a_next_surv # this is correct if assets of dead are thrown away!
	K_hh = raw_aggregates.a_next # this is correct if assets of dead are thrown away!
	
	L_eff = raw_aggregates.ℓ_eff
	𝕀 = 0.0 #raw_aggregates.inherit 
	
	K_aggregates = _aggregates_(K_guess, par, K_hh, L_eff)

	ζ_H = H_hh - H_guess 
	ℓ_H = criterion(H_hh, H_guess)

	loss = [K_aggregates.ℓ ℓ_H]
	updated = (; K_aggregates.K_supply, H_hh, L_eff)

	additional_aggregates = _aggregates_(K_aggregates.K_supply, par, K_hh, L_eff)
	aggregates = (; ζ_H, ℓ_H, H_hh, L_eff, K_aggregates..., additional_aggregates)

	
	(; loss, updated, aggregates)
end

# ╔═╡ 4659e2ae-38ca-411f-b180-9589c70c3afb
const DD = DimensionalData

# ╔═╡ 976144e3-de4b-471a-868b-96abc821ca84
function iterate_cross_sectional_shock!(π₀, π₋₁, sol_backward, statespace, (; m), prices₀, par₀; π_init = π₋₁[j = At(0)])
	# assume the economy was in steady state up until period -1
	# in period 0 there is a probability-zero-shock
	# πₜ are interpreted as cross-sectional distributions in periods t ∈ {-1, 0}

	# the code is very similar to "solve_forward"
	(; p, r) = prices₀
	(; δ) = par₀

	policy = DimStack(sol_backward...)
	j₀, J = extrema(DD.dims(policy, :j))

	(; states) = statespace
	P = statespace.P_from

	π₀[j = At(0)] .= π_init
	
	for j ∈ 0:(J-1)
		# find all states with positive mass
		positive_mass = findall(@view(π₋₁[j = At(j)]) .> 0)
	
		for ind ∈ positive_mass
			# for each such state ...
			
			## 1. find optimal policy
			(; state, ε) = states[ind]

			#### ADAPTED ####
			(; ho, a_next) = policy[j = At(j), state = At(state), ε = At(ε)]
			state_n = a_next + (1-δ)/(1+r) * p * ho
			#################
			
			## 2. find closest points on grid
			(; high, low) = weighted_neighbours(state_n, statespace.grid)

			## 3. compute probability mass for states in j + 1
			π_base = π₋₁[j = At(j)][ind] * (1 - m[j = At(j)])
			π₀[j = At(j+1), state = At(high.x)] .+= π_base .* high.weight .* P[from = At(ε)]
			π₀[j = At(j+1), state = At(low.x)]  .+= π_base .* low.weight .* P[from = At(ε)]
		end
	end

	π₀
end

# ╔═╡ 0c481dec-3a04-4a51-9c11-ef03dbab3683
function get_π_init_all(GE_sol_perm, price_paths, par, statespace)
	(; sol_backward, sol_forward) = GE_sol_perm.sol
	(; π) = sol_forward
	(; m) = par

	π₋₁ = copy(π, name = :π₋₁)
	π₀  = zeros(DD.dims(π₋₁), name = :π₀)

	prices₀ = price_paths[t = At(0)]

	iterate_cross_sectional_shock!(π₀, π₋₁, sol_backward, statespace, (; m), prices₀, par)

	π₀
end

# ╔═╡ 2b26bf96-95e8-4e6e-8ba4-f1a4a7857464
function get_π_init(_, GE_sol_perm, price_paths, j_init, par, statespace)
	π₀ = get_π_init_all(GE_sol_perm, price_paths, par, statespace)

	π₀[j = At(j_init)]
end

# ╔═╡ cda15204-1cd1-4bd1-b5b6-ac72ed154309
function extend_path(path, (; J); X₋₁=first(path))
	T̃ = maximum(DD.dims(path, :t))
	#name = DD.name(path)
	
	# extend 
	path_pre2  = dimstack_from_nt(X₋₁, Dim{:t}(-2:-2))
	path_pre1  = dimstack_from_nt(X₋₁, Dim{:t}(-1:-1))
	path_post = dimstack_from_nt(last(path), Dim{:t}((T̃+1:T̃+J+2)))

	map(keys(path)) do key
		pathX = [path_pre2[key]; path_pre1[key]; path[key]; path_post[key]]
	end |> DimStack
end

# ╔═╡ 6859961c-f24f-4ae3-bb5c-1bc639ec00a1
function get_inheritances_θ(bequests_θ, statespace)
	(; perm_dim, mc_permanent, π_permanent) = statespace
	
	θ_dim = DD.dims(perm_dim, :θ)
	
	P_θ_DD = DimArray(mc_permanent.p, (only(perm_dim), only(perm_dim)))

	_inheritances_θ_ = P_θ_DD' * (@d bequests_θ .* π_permanent)

	# take into account mass of types	
	DimVector(
		(@d _inheritances_θ_ ./ π_permanent),
		name = :inheritances
	)
end

# ╔═╡ 9291d566-bf4f-49ac-aba3-277170e5daed
function inheritances_stationary((; sim_df), statespace)
	#(; F) = par

	# STEP 1: Computing bequests by type
	bequests_θ = get_bequests_θ(sim_df, statespace)
	# STEP 2: Computing inheritances by type
	inheritances_θ = get_inheritances_θ(bequests_θ, statespace)
	# STEP 3: Compute inheritances by type and cohort
	#inheritances_θj = get_inheritances_θj(inheritances_θ, par)

	(; bequests_θ, inheritances_θ #=, inheritances_θj=#)
end

# ╔═╡ f8919942-9ef9-476a-94eb-c46ec88fde94
function get_inheritances_θt(bequests_θt, statespace, π_t)
	(; perm_dim, mc_permanent, π_permanent) = statespace

	t_dim = DD.dims(bequests_θt, :t)
	T̃ = maximum(t_dim)
	θ_dim = only(perm_dim)

	P_θ_DD = DimArray(mc_permanent.p, (θ_dim, θ_dim))

	π_θt = @d π_t .* π_permanent
	
	# available bequests: -1:T-1
	# normalize available bequests by mass π_θt
	inheritances_θt = DimArray(
		((@d bequests_θt .* π_θt) * P_θ_DD)[t = At(-1:T̃-1)],
		(Dim{:t}(0:T̃), θ_dim)
	)
	
	# inheritances == bequests from last period
	# divide across all agents of given type in a given time using π_θt
	DimArray(
		(@d inheritances_θt ./ π_θt[t = At(0:T̃)]),
		name = :inheritances
	)
end

# ╔═╡ 7d9ea25a-6f76-43a2-83ab-d81e6210bbc6
function inheritances_transition(out, statespace, π_t)
		
	bequests_θt = get_bequests_θt(out, statespace)
	inheritances_θt = get_inheritances_θt(bequests_θt, statespace, π_t)

	(; bequests_θt, inheritances_θt)
end

# ╔═╡ f66ab91b-14d6-4981-b78e-ab6f55129220
function solve_forward!(π, sol_backward, statespace, (; m); π_init, j_init)
	
	(; next_state) = sol_backward
	
	#dims_j = DD.dims(next_state)
	j₀, J = extrema(DD.dims(next_state, :j))

	π[j = At(j_init)] .= π_init

	(; states) = statespace
	P = statespace.P_from

	for j ∈ j_init:(J-1)
		# find all states with positive mass
		positive_mass = findall(@view(π[j = At(j)]) .> 0)
	
		for ind ∈ positive_mass
			# for each such state ...
			
			## 1. find optimal policy
			(; state, ε) = states[ind]
			state_n = next_state[j = At(j), state = At(state), ε = At(ε)]

			## 2. find closest points on grid
			(; high, low) = weighted_neighbours(state_n, statespace.grid)

			## 3. compute probability mass for states in j + 1
			π_base = π[j = At(j)][ind] * (1 - m[j = At(j)])
			π[j = At(j+1), state = At(high.x)] .+= π_base .* high.weight .* P[from = At(ε)]
			π[j = At(j+1), state = At(low.x)]  .+= π_base .* low.weight .* P[from = At(ε)]
		end
	end

	nothing
end

# ╔═╡ 59909171-3b97-4f96-993f-db0aa09e5e00
get_states(da) = get_states(DD.dims(da))

# ╔═╡ 843d8157-85c4-4ef9-8fc4-7b4a83b01d17
function get_cⱼ₋₁_housing(cⱼ, par, (; prices_etcⱼ₋₁, prices_etcⱼ, prices_etcⱼ₊₁), statespace; j)
	(; u′, u′⁻¹, v′, β, annuities) = par
	(; ξ, σ, β) = par
	ξ = ξ[j = At(j)]
	if length(par.β) == 1
		βⱼ₋₁ = par.β^(j-1)
		βⱼ   = par.β^j
	else
		βⱼ₋₁, βⱼ = par.β[j = At(j-1:j)]
	end
	ⱼ₋₁ = prices_etcⱼ₋₁
	ⱼ   = prices_etcⱼ
	ⱼ₊₁ = prices_etcⱼ₊₁

	if ξ > 0
		κⱼ₋₁ = get_κⱼ(ⱼ₋₁.p, ⱼ.p, ⱼ.r,   par, j-1)
		κⱼ   = get_κⱼ(ⱼ.p, ⱼ₊₁.p, ⱼ₊₁.r, par, j)
	end

	if annuities
		# with annuities
		XR = (βⱼ/βⱼ₋₁ * (1 + ⱼ.r))
	else
		# without annuities
		XR = (βⱼ/βⱼ₋₁ * (1 - ⱼ₋₁.m) * (1 + ⱼ.r))
	end
	
	(; grid, ε_grid, P) = statespace

	𝔼u_cⱼ = zeros(DD.dims(cⱼ), name = :cⱼ₋₁)
	cⱼ₋₁  = zeros(DD.dims(cⱼ), name = :cⱼ₋₁)
	
	for (aⱼ_i, aⱼ) ∈ enumerate(grid)
		c_curr = cⱼ[state = aⱼ_i]
		
		a_next = aⱼ
		
		for (εⱼ₋₁_i, _) ∈ enumerate(ε_grid)

			_𝔼u_cⱼ_ = dot(c_curr .^ (-σ), P[εⱼ₋₁_i, :])
			if ξ > 0 # with housing
				_𝔼u_cⱼ_ = @. (1-ξ) * _𝔼u_cⱼ_ * κⱼ^(-ξ * (1-σ))
			end
			
			_cⱼ₋₁_ = (_𝔼u_cⱼ_ * XR + ⱼ₋₁.m * v′(a_next))^(1/-σ)

			if ξ > 0 # with housing
				_cⱼ₋₁_ = _cⱼ₋₁_ / (1-ξ)^(1/-σ) / κⱼ₋₁^(ξ * (1-σ) / σ)
			end
			
			cⱼ₋₁[state = aⱼ_i, ε = εⱼ₋₁_i] = _cⱼ₋₁_
		end

	end
	
	return (; cⱼ₋₁, 𝔼u_cⱼ)
end

# ╔═╡ c28cec8b-2ea5-4bf8-a935-7eb00af0b2a5
function iterate_backward(::HousingModel, cⱼ, par_all, par_cohort, permanent, priceses, statespace, j)
	par = (; par_all..., par_cohort...)

	# 1. Assume constraint doesn't hold
	(; cⱼ₋₁, 𝔼u_cⱼ) = get_cⱼ₋₁_housing(cⱼ, par, priceses, statespace; j)
	hⱼ₋₁ = hⱼ₋₁_from_cⱼ₋₁.(cⱼ₋₁, Ref(par), Ref(priceses); j) 
	
	zⱼ = statespace.grid
	aⱼ   =   aⱼ_from_hⱼ₋₁.(hⱼ₋₁, zⱼ, Ref(par), Ref(priceses))

	inc_plus_inhⱼ₋₁ = getproperty.(
		income_plus_inheritances.(statespace.ε_grid, Ref(par_all), Ref(permanent), Ref(priceses)),
		:inc_plus_inhⱼ₋₁)
	zⱼ₋₁ = @d zⱼ₋₁_from_ahc.(aⱼ, hⱼ₋₁, cⱼ₋₁, inc_plus_inhⱼ₋₁, Ref(priceses), Ref(par))

	DA = DimStack(
		DimArray(aⱼ, name = :aⱼ),
		DimArray(hⱼ₋₁, name = :hⱼ₋₁),
		DimArray(zⱼ, name = :zⱼ),
		DimArray(𝔼u_cⱼ, name = :𝔼u_cⱼ),
		DimArray(inc_plus_inhⱼ₋₁, name = :inc_plus_inhⱼ₋₁)
	)

	constrainedⱼ₋₁ = falses(DD.dims(cⱼ))
	
	no_violated = 0

	@assert eachindex(DA) == eachindex(cⱼ₋₁) == eachindex(zⱼ₋₁)
	for ind ∈ eachindex(DA)
		vars = DA[ind]
		
		# 2. Check if constraint binds
		if all(≠(0), par.ξ) && constraint_is_violated(vars, par, priceses)

			no_violated = no_violated + 1
			
			# 3. If so, overwrite
			(; cⱼ₋₁_bind, zⱼ₋₁_bind) = constraint_binds(vars, par, priceses, j)

			cⱼ₋₁[ind] = cⱼ₋₁_bind
			zⱼ₋₁[ind] = zⱼ₋₁_bind
			constrainedⱼ₋₁[ind] = true
			
		end
	end

	#if no_violated > 0
	#	@info "constraint is violated $no_violated times at age $j"
	#end
	
	stateⱼ₋₁ = zⱼ₋₁

	
	(; cⱼ₋₁, stateⱼ₋₁, constrainedⱼ₋₁)
end

# ╔═╡ 0bfd4678-c05c-4e62-9f11-6a1d80f4f58a
function solve_backward!(c, next_state, value, next_value, constrained, stuff, Mo, par_all, par_cohort, permanent, statespace; price_paths, t_born, j_init, 
						 inherit # for a given permanent type θ
						)

	(; h, ρ_SS) = par_cohort
	(; J, a̲, τ, d̄, Z̲) = par_all
	(; θ) = permanent
	
	(; ε_grid, grid, dims, P) = statespace
	
	
	## SOLVE BACKWARDS ("solve policy functions")
	t_J = t_born + J
	priceses_J = get_prices(Mo, price_paths, par_all, par_cohort, J; t_born, inherit)

	ρ_J = @view ρ_SS[j=At(J)]
	h_J = @view h[j = At(J)]
	ℓ_eff_J = @. ε_grid * θ * h_J # effective labor supply
	inc_J   = @. (1-ρ_J)*(1-τ) * ℓ_eff_J + ρ_J * d̄ * θ
	#inh_J = inherit[j = At(J)]

	out = @d last_choices.(Ref(Mo), grid, ℓ_eff_J, inc_J, Ref(priceses_J), Ref(par_all), Ref(J))

	         c[j = At(J)] .= getproperty.(out, :c)
	     value[j = At(J)] .= getproperty.(out, :v)
	next_state[j = At(J)] .= getproperty.(out, :next_state)
		 stuff[j = At(J)] .= getproperty.(out, :stuff)

	# TODO: SOLVE THIS MORE ELEGANTLY
	has_nans = false

	for j ∈ J:-1:(j_init + 1)
		t = t_born + j
		cⱼ     = @view     c[j = At(j)]
		valueⱼ = @view value[j = At(j)]
		
		priceses = get_prices(Mo, price_paths, par_all, par_cohort, j; t_born, inherit)
		
		(; cⱼ₋₁, stateⱼ₋₁, constrainedⱼ₋₁) = iterate_backward(Mo, cⱼ, par_all, par_cohort, permanent, priceses, statespace, j)

		if any(isnan, stateⱼ₋₁)
			has_nans = true
			throw(DomainError(stateⱼ₋₁, "Found NaNs in  stateⱼ₋₁"))
			#@info "Broke because of NaNs. (model age j = $j)"
			break
		end
		
		for ε ∈ DD.dims(dims, :ε)

			knots = stateⱼ₋₁[ε = At(ε)]
			vals = parent(grid)
			ids = sortperm(knots)		
			
			stateⱼ_itp = LinearInterpolation(
				knots[ids],
				vals[ids],
				extrapolation_bc = Line()
			)

			constrainedⱼ₋₁_itp = LinearInterpolation(
				knots[ids],
				constrainedⱼ₋₁[ε = At(ε)][ids],
				extrapolation_bc = Line(),
			)

			valueⱼ_itp = LinearInterpolation(
				parent(grid),
				valueⱼ[ε = At(ε)],
				extrapolation_bc = Line(),
			)

			_next_state_ε_ = stateⱼ_itp.(grid)
			#@info size(_next_state_ε_)
			#@info size(valueⱼ_itp.(_next_state_ε_))
			#@info size(next_value[j = At(j-1),  ε = At(ε)])
			
			  next_state[j = At(j-1), ε = At(ε)] .= #max.(
				  _next_state_ε_#, Z̲)
			 constrained[j = At(j-1), ε = At(ε)] .= constrainedⱼ₋₁_itp.(grid)

			 next_value[j = At(j-1),  ε = At(ε)] .= valueⱼ_itp.(_next_state_ε_)
		end

		let
			stateⱼ₋₁       = grid
			# state(j) == next_state(j-1) (that is, state today was chosen yesterday)
			stateⱼ         = next_state[j = At(j-1)]
			valueⱼ         = next_value[j = At(j-1)]
			constrainedⱼ₋₁ = constrained[j = At(j-1)]
			
			#if size(P, 1) > 1
			#	@info "XXXXX"
			#	@info DD.dims(valueⱼ), DD.dims(P)
			#	@info size(valueⱼ), size(P)
			#end
			
			𝔼vⱼ = DimArray(parent(valueⱼ) * parent(P)', DD.dims(valueⱼ))
			#𝔼vⱼ = valueⱼ * P'

			out = @d choicesⱼ₋₁.(Ref(Mo), stateⱼ₋₁, stateⱼ, 𝔼vⱼ, constrainedⱼ₋₁, statespace.ε_grid, Ref(par_all), Ref(par_cohort), Ref(permanent), Ref(priceses), Ref(statespace), Ref(j))
		
		    	c[j = At(j-1)] = getproperty.(out, :c)
			value[j = At(j-1)] = getproperty.(out, :v)
			stuff[j = At(j-1)] = getproperty.(out, :stuff)
		end
	end

	if has_nans
		next_state .= NaN
	end

	nothing
end

# ╔═╡ 26bc0666-0405-466b-8dea-356d9d7c4e19
function solve_backward_forward!(c, next_state, value, next_value, constrained, stuff, π, Mo, par_all, par_cohort, permanent, statespace; price_paths, π_init, j_init = 0, t_born = 0,
								 inherit_j # inheritances for each age j of a given permanent type θ
								)
	
	solve_backward!(c, next_state, value, next_value, constrained, stuff, Mo, par_all, par_cohort, permanent, statespace; price_paths, j_init, t_born, inherit = inherit_j)

	## SOLVE FORWARD
	solve_forward!(π, (; next_state), statespace, par_cohort; π_init, j_init)

	return nothing
end

# ╔═╡ 3c2f3f62-f7b9-4d66-8598-be8bb6bd6356
function simulate_cohorts(Mo, par, permanent, statespace, demographics, GE_sol_perm; price_paths, j_last = par.J, T̃, inheritances_tj)

	j_dim = Dim{:j}(0:j_last)
	t_borns = (-j_last):1:T̃

	par_all = drop_m_h(; par...)
	(; c, next_state, value, next_value, constrained, stuff, π) = let
		par_0 = let
			m = demographics.m[born = At(0)]
			par_cohort = (; par.h, par.ρ_SS, m)
		end

		inherit_j = map(j_dim) do j
			inheritances_tj[j = At(j), t = At(j)]
		end
		
	 	initialize_cohorts(Mo, par_all, par_0, permanent, statespace, t_borns;
						   price_paths, j_init = 0, t_born = 0, 
						   inherit=inherit_j)
	end

	π_init_all = get_π_init_all(GE_sol_perm, price_paths, par, statespace)
	
	for t_born ∈ t_borns
		m = demographics.m[born = At(t_born)]
		par_cohort = (; par.h, par.ρ_SS, m)

		cₜ          = @view 		  c[born = At(t_born)]
		next_stateₜ = @view  next_state[born = At(t_born)]
		valueₜ      = @view       value[born = At(t_born)]
		next_valueₜ = @view  next_value[born = At(t_born)]
		constrainedₜ= @view constrained[born = At(t_born)]
		stuffₜ      = @view 	  stuff[born = At(t_born)]
		πₜ          = @view 		  π[born = At(t_born)]

		cₜ, next_stateₜ, stuffₜ, πₜ
		j_init = max(0, -t_born)
		π_init = π_init_all[j = At(j_init)]

		inherit_j = map(j_dim) do j
			t = clamp(j + t_born, 0, T̃)
			inheritances_tj[j = At(j), t = At(t)]
		end

		solve_backward_forward!(cₜ, next_stateₜ, valueₜ, next_valueₜ, 
								constrainedₜ, stuffₜ, πₜ,
								Mo, par_all, par_cohort, permanent, statespace; 
								price_paths, π_init, j_init, t_born, inherit_j)
	end

	sol = (c, next_state, stuff, π)

	sim_ds = DimStack(
			c, value, next_state, dimarray_of_nts_to_nt_of_dimarrays(stuff)..., π
		)
	
	sim_df = DataFrame(sim_ds)

	# TODO: FIX for multiple types
	factor = @chain sim_df begin
		@transform(:t = :j + :born)
		@subset( :t == 0)
		@combine(:π = sum(:π))
		only(_.π)
	end

	@transform!(sim_df, :π = :π / factor)
	
	(; sol, sim_df, sim_ds)
end

# ╔═╡ 0ae74c43-bae4-406b-9e31-fd76a0a4a398
function simulate_cohort(Mo, par_all, par_cohort, permanent, statespace; price_paths, π_init, j_init = 0, t_born = 0,
								inherit_j # inheritances for each age of a given permanent type θ
							   )
	
	(; c, next_state, value, next_value, constrained, stuff, π) = initialize_cohort(Mo, par_all, par_cohort, permanent, statespace; price_paths, j_init, t_born, inherit = inherit_j)

	solve_backward_forward!(c, next_state, value, next_value, constrained, stuff, π, Mo, par_all, par_cohort, permanent, statespace; price_paths, π_init, j_init, t_born, inherit_j)

	sol_backward = (; c, next_state, value, dimarray_of_nts_to_nt_of_dimarrays(stuff)...)

	sol_forward = (; π)

	sim_df = DataFrame(DimStack(sol_backward..., sol_forward...))
	
	return (; sim_df, sol_backward, sol_forward)
end

# ╔═╡ acb38371-e502-4b9c-80f4-314e9458edcc
function stationary_PE(Mo, par, statespace, guesses, prices;
								  details = true,
								  π_init = simple_initial_distribution(statespace),
							 	  inheritances_θ = zeros(only(statespace.perm_dim)),
					   			total_mass = 1.0,
								  #solution_method = EGM()
								 )

	(; π_permanent, state_dim, perm_dim) = statespace

	π_j = get_π_j(par.m)
	
	inheritances_θj = 
		DimArray(@d(inheritances_θ .* par.F ./ π_j), name = :inheritances)
		
	par_cohort = (; par.h, par.ρ_SS, par.m)
	par_x = drop_m_h(; par...)
	
	price_paths = constant_price_paths(par, prices)

	sols = map(enumerate(zip(get_states(π_permanent), π_permanent))) do (i_perm, (permanent, π_perm))
		inherit_j = @view inheritances_θj[θ = i_perm]
		
		sol = simulate_cohort(Mo, par_x, par_cohort, permanent, statespace; price_paths, π_init, j_init = 0, t_born = 0, inherit_j)

		sim_df = @transform(sol.sim_df, :π = :π * π_perm)
		(; sol, sim_df, permanent)
	end

	sim_df = vcat(
		getproperty.(sols, :sim_df)...,
		source = :permanent => parent(getproperty.(sols, :permanent))
	)

	raw_aggregates = aggregate(sim_df; total_mass)
	
	aggregates = loss_and_aggregates(Mo, par_x, guesses, raw_aggregates)

	if details
		out_PE = (; aggregates, raw_aggregates, guesses, prices, par, sols, sim_df, inheritances_θj)
		inheritances_etc = inheritances_stationary(out_PE, statespace)
		return (; out_PE, inheritances_etc)
	else
		return aggregates
	end

end

# ╔═╡ 1ba9f113-c2d5-4c34-ba9f-01d39ffc6f35
function stationary_GE(Mo, par, statespace;
			guesses = guess(Mo, par), inheritances_θ_guess = nothing,
					   total_mass = 1.0,
			details = 0,
			maxiter = 600, λ = 0.1, tol = 1e-14, λ_inherit = 1.0,
			kwargs...
		)
	
	_guesses_ = collect(guesses)

	perm_dim = only(statespace.perm_dim)
	
	if isnothing(inheritances_θ_guess)
		inheritances_θ = zeros(perm_dim)
	else
		inheritances_θ = DimVector(inheritances_θ_guess, statespace.perm_dim)
	end
	
	for it ∈ 1:maxiter
		(; prices, guesses) = prices_from_guesses(Mo, _guesses_, par)

		(; out_PE) = stationary_PE(Mo, par, statespace, guesses, prices; total_mass, 
								  details=true, inheritances_θ, kwargs...)
		out₀ = out_PE.aggregates
		# update guesses
		_guesses_ = (1-λ) * _guesses_ + λ * collect(out₀.updated)
		##########
		
		# update inheritances
		inheritances_etc = inheritances_stationary(out_PE, statespace)
		inheritances_θ_new = inheritances_etc.inheritances_θ

		crit_inh = norm(inheritances_θ - inheritances_θ_new)
		inheritances_θ .= 
			λ_inherit .* inheritances_θ_new .+ (1-λ_inherit) .* inheritances_θ
		#########
		
		converged = maximum(abs.(collect(out₀.loss))) < tol && abs(crit_inh) < tol
		# print infos
		if (details > 0 && it % details == 0) || converged		
			@info """
			iteration $it, loss: $(out₀.loss)
			updated: $(out₀.updated)
			bequests_θ: $(inheritances_etc.bequests_θ)
			inheritances_θ: $(inheritances_etc.inheritances_θ)
			
			prices: $prices
			Δ_inh: $crit_inh
			$((; out_PE.raw_aggregates.a_next, out_PE.raw_aggregates.bequests, out_PE.raw_aggregates.inheritance,))
			"""
		end
		if converged
			return (; out_PE, inheritances_etc)
		end	
		if it == maxiter
			@warn "Did not converge: loss: $(out₀.loss)"

			return (; out_PE, inheritances_etc)
		end
	end
end

# ╔═╡ 1b1e4b39-a845-4e77-b05c-fc6e173b27f0
function loss_and_aggregates_t(M::HousingModel, par, paths_in, paths_out, GE₀)
	#(; H_guess) = guesses

	t_dim = DD.dims(paths_out, :t)
	T̲, T̄ = extrema(t_dim)
	@assert T̲ == 0
	
	H_out = paths_out.ho

	K_hh₀ = GE₀.raw_aggregates.a_next
	K_hhₓ = paths_out.a_next[t = At(0:(T̄-1))] |> parent
	K_hh = DimVector([K_hh₀; K_hhₓ], t_dim, name = :K_hh)
	
	L_out = paths_out.ℓ_eff
	
	
	H_in = copy(paths_in.H_hh)
	K_in = copy(paths_in.K_supply)
	L_in = copy(paths_in.L_eff)

	(; bonds2GDP, NFA2GDP) = par
	
	GDP = DimVector(output.(K_in, L_in, Ref(par)), name = :GDP)
	B₀ = DimVector(bonds2GDP .* GDP, name = :B₀)
	NFA = DimVector(NFA2GDP .* GDP, name = :NFA)
	
	K_supply = K_hh .- B₀ .- NFA
	ζ_K = DimVector(K_supply .- K_in, name = :ζ_K)
	ζ_H = DimVector(H_in - H_out, name = :ζ_H)
	ζ_L = DimVector(L_in - L_out, name = :ζ_L)
	
	ℓ_K = DimVector(criterion.(K_supply, K_in), name = :ℓ_K)
	ℓ_H = DimVector(criterion.(H_out, H_in), name = :ℓ_H)
	ℓ_L = DimVector(criterion.(L_out, L_in), name = :ℓ_L)
		
	loss = DimStack(ℓ_H, ℓ_K, ℓ_L)

	aggregates = DataFrame(DimStack(H_in, H_out, K_in, ζ_H, ζ_K, GDP, B₀, NFA, K_supply))
	
	updated = DimStack(
		DimVector(H_out, name = :H_hh),
		DimVector(K_supply, name = :K_supply),
		DimVector(L_out, name = :L_eff)
	)
	
	(; loss, updated, aggregates, H_hh = H_out, K_supply)
	
end

# ╔═╡ af69b84f-e4f5-4f90-85a3-ef00d2288b83
function get_price_paths(::HousingModel, path, par; GE₀)
	(; δ) = par
	
	X₋₁ = (; GE₀.aggregates.updated.K_supply, GE₀.aggregates.updated.H_hh, GE₀.aggregates.updated.L_eff)

	pathX = extend_path(path, par; X₋₁)
	
	H = pathX.H_hh
	
	ts = collect(collect(DD.dims(H, :t)))
	tsₓ = (minimum(ts)+1):maximum(ts)
	tₓ_dim = Dim{:t}(tsₓ)
	
	X = zeros(tₓ_dim)
	for t ∈ tsₓ
		X[t = At(t)] = H[t = At(t)] - (1-δ) * H[t = At(t-1)]
	end
	ps = DimArray(house_price.(X, Ref(par)), name = :p)
	
	K = pathX.K_supply[t = At(tsₓ)]
	L = pathX.L_eff[t = At(tsₓ)]
	
	rs = DimArray(interest_rate.(K, L, Ref(par)), name = :r)
	ws = DimArray(wage.(K, L, Ref(par)), name = :w)

	DimStack(ps, rs, ws)
end

# ╔═╡ ca363b65-dd42-475c-9114-a259691c7913
function transition_PE(model, T̃, par, statespace, demographics, GE₀, paths_in;
					   j_last=par.J, normalize_population = false,
					   inheritances_tθ,
					   π_jt = get_π_jt(
						   (; demographics, GE₀, T̃), par, statespace
					   )
					  )
	price_paths = get_price_paths(model, paths_in, par; GE₀)

	# XXX FIXME - I think ./ π_jt must be replaced by something else
	# pi_jt should sum to one in each t!!!
	inheritances_θtj = DimArray(@d(inheritances_tθ .* par.F ./ π_jt), name = :inheritance)
	
	(; π_permanent, state_dim, perm_dim) = statespace

	
	sols = map(enumerate(zip(get_states(π_permanent), π_permanent, GE₀.sols))) do (i_perm, (permanent, π_perm, GE_sol_perm))
		inheritances_θ = @view inheritances_θtj[θ = i_perm]
		
		sol = simulate_cohorts(model, par, permanent, statespace, demographics, GE_sol_perm; price_paths, j_last, T̃, inheritances_tj = inheritances_θ)

		sim_df = @transform(sol.sim_df, :π = :π * π_perm)
		(; sim_df, permanent)
	end

	sim_df = vcat(
		getproperty.(sols, :sim_df)...,
		source = :permanent => parent(getproperty.(sols, :permanent))
	)
	
	raw_aggregate_paths = aggregate_paths(sim_df; normalize_population)
	_aggregate_paths_ = loss_and_aggregates_t(model, par, paths_in, raw_aggregate_paths, GE₀)

	out_PE = (; aggregate_paths=_aggregate_paths_, price_paths, guessed_paths=deepcopy(paths_in), raw_aggregate_paths, sim_df, demographics, statespace, GE₀, T̃, inheritances_θt=inheritances_tθ, inheritances_θtj, π_jt)
end

# ╔═╡ 880637f3-81f0-48f5-8918-c03dac35e6fc
function transition_GE(model, T̃, par, statespace, demographics, GE₀, guessed_path;
					   j_last = par.J, normalize_population = false,   	   
					   inheritances_θt_guess = nothing,
					   maxiter = 100, λ = 0.05, tol = 1e-4, λ_inh = 1.0, details=1)

	path_in = deepcopy(guessed_path)
	
	if λ isa Number
		λ = fill(λ, maxiter)
	end
	
	π_jt = get_π_jt((; demographics, GE₀, T̃), (; par.J))
	π_t = get_π_t((; demographics, GE₀, T̃))
	
	perm_dim = only(statespace.perm_dim)
	t_dim = Dim{:t}(0:T̃)
	
	if isnothing(inheritances_θt_guess)
		inheritances_tθ = zeros((t_dim, perm_dim))
	else
		inheritances_tθ = DimArray(inheritances_θt_guess, (t_dim, perm_dim))
	end
	
	for it ∈ 1:maxiter
		
		out_PE = transition_PE(model, T̃, par, statespace, demographics, GE₀, path_in;
							   j_last, normalize_population,
							   inheritances_tθ, π_jt)

		crit₀ = maximum(abs, out_PE.aggregate_paths.loss)
		crit = maximum(crit₀)

		inh_tθ_etc_new = inheritances_transition(out_PE, statespace, π_t)
		inheritances_tθ_new = inh_tθ_etc_new.inheritances_θt
			
		#inheritances_tθ_new = compute_inheritance_θt(out_PE, statespace)
		
		crit_inh = norm(@d inheritances_tθ_new .- inheritances_tθ)
		
		converged = abs(crit) < tol && abs(crit_inh) < tol
		
		if converged || (details > 0 && it % details == 0)
			@info (; it, crit₀, crit, crit_inh)
			fig = Figure(size = (650, 250))
			ax = Axis(fig[1,1], title = "K_supply")
			ax2 = Axis(fig[1,2], title = "L_eff")
			ax3 = Axis(fig[1,3], title = "H_hhf")
			ax4 = Axis(fig[1,4], title = L"beq$_t$, inh$_{t-1}$")

			lines!(ax, path_in.K_supply)
			lines!(ax2, path_in.L_eff)
			lines!(ax3, path_in.H_hh)

			lines!(ax, out_PE.aggregate_paths.updated.K_supply, linestyle = :dash)
			lines!(ax2, out_PE.aggregate_paths.updated.L_eff, linestyle = :dash)
			lines!(ax3, out_PE.aggregate_paths.updated.H_hh, linestyle = :dash)
			lines!(ax4, out_PE.raw_aggregate_paths.bequests, label = "beq")
			lines!(ax4, 0:T̃-1, parent(out_PE.raw_aggregate_paths.inheritance[t = At(1:T̃)]), label = "L(inh)", linestyle = :dash)
			#axislegend(ax4)
			@info fig
		end
			
		path_out = out_PE.aggregate_paths.updated

		if converged || it == maxiter
			if converged
				@info "Transition converged after $it iterations. crit = $crit, crit_inh = $crit_inh"
			else
				@warn "Did not converge after $it iterations. rit = $crit, crit_inh = $crit_inh"
			end
			
			return (; out_PE, inheritances_tθ, inh_tθ_etc_new)
		end
		
		for key ∈ keys(path_in)
			@d path_in[key] .= 
				(1-λ[it]) * path_in[key] .+ λ[it] * path_out[key]
		end

		@d inheritances_tθ .= (1-λ[it]) * inheritances_tθ + λ[it] * inheritances_tθ_new
	end
	
end

# ╔═╡ d2262948-178c-4e42-a7e7-0f38aa76e7db
function get_par₀(; m,
		h,	# deterministic component of income (human capital)
		γ = 2.0,
		σ = γ,
		β = 0.995,
        α = 0.33,
		δ = 0.1,
 		Z̲ = -Inf,
		a̲ = 0.0,
		HS_ela = 1.5,
		L̄ = 1.0,
		p = 1.0,
		ξ = 0.15,
		F = nothing,
		θ = 0.0,
		ν₀ = 78.5, # scaling factor Y
		ν₁ = 1.92, # CRRA exponent ν
		bonds2GDP = 1.0,
		NFA2GDP = 0.0,
		check_j_dim = true,
		JR = 43,
		J = maximum(DD.dims(m, :j)),
		age_min = 20,
		age_max = age_min + J,
		d̄ = 0.92, τ = 0.32,
				  annuities = false
	)

	j_dim = DD.dims(m, :j)
	j_dim_h = DD.dims(h, :j)

	@assert J == age_max - age_min

	if isnothing(F)
		F = zeros(j_dim, name = :F)
	end
	
	if check_j_dim
		@assert j_dim == j_dim_h
		if !(β isa Real)
			j_dim_β = DD.dims(m, :j)
			@assert j_dim_β == j_dim
		end
	end
	
	α̃ = HS_ela / (1 + HS_ela)
	@assert α̃ / (1-α̃) ≈ HS_ela

	if ξ isa Real
		ξ = fill(ξ, j_dim)
	end

	# Retirement policy
	ρ_SS = let
		if JR < J + 1
			ρ_SS = 1 .- h ./ h[j = At(JR-1)]
			ρ_SS[j = At(0:JR-1)] .= 0.0
		else
			ρ_SS = zeros(DD.dims(h))
		end
		
		ρ_SS
	end
	
	u(c) = c > 0 ? c^(1-γ)/(1-γ) : -Inf
	u′(c) = c^(-γ)
	u′⁻¹(x) = x^(-1/γ)
	v(a)  = ν₀ == 0.0 ? 0.0 : a ≥ 0 ? ν₀ * a^(1-ν₁)/(1-ν₁) : -Inf
	v′(a) = ν₀ == 0.0 ? 0.0 : ν₀ * a^(-ν₁)
	
	(; δ, α, Θ = 1, L = 1, β, #= ρ, r,=# bonds2GDP, NFA2GDP,
		m, γ, σ, a̲, Z̲, θ,
		h, u, u′, u′⁻¹, v, v′, ν₀, ν₁, F,
		w = 1.0, ρ_SS, d̄, τ,
		ξ, α̃, L̄, p, j_dim, J, age_min, age_max, annuities)
	
end

# ╔═╡ 03bc50bc-5c74-4afe-ae3f-9fed0510817e
let
	J = 5
	j_dim = Dim{:j}(0:J)
		
	h = zeros(j_dim)
	m = zeros(j_dim)
		
	get_par₀(; m, h, JR = 2, ξ = 0.15)
end

# ╔═╡ fbc03e03-c9f7-4edc-85a4-2d4ae3268edf
function get_par(; 
		demo = :perpetual_youth,
		mm = 1/45,
		age_min = demo == :perpetual_youth ? 0   : 25,
		age_max = demo == :perpetual_youth ? 100 : 120,
		inc_profile = true,
		h̄ = nothing,
		h = nothing,
		kwargs...
	)
	
	m = mortality(demo; age_min, age_max, m=mm)
	j_dim = DD.dims(m, :j)
	J = maximum(j_dim)

	@assert J == age_max - age_min

	
	if isnothing(h)
		if !isnothing(h̄)
			h = fill(h̄, j_dim)
		else
			h = income_profile(J+1, 41) .^ inc_profile
		end
	end # else take provided y verbatim

	get_par₀(; m, h, kwargs...)	
end

# ╔═╡ 7f76c5ff-b8d2-466b-b064-3dceb1071062
function Statespace(; amin, amax, na, ε_chain = no_income_risk(), exponential = true, permanent = no_permanent_states())

	if exponential
		grid = exponential_grid(amin, amax, na)
	else
		grid = range(amin, amax, length=na)
	end
	ε_grid = ε_chain.state_values
	
	state_dim = Dim{:state}(grid)
	ε_dim = Dim{:ε}(ε_grid)
	dims = (state_dim, ε_dim)
	
	(; mc_permanent, π_permanent) = permanent
	perm_dim = DD.dims(π_permanent)
	
	P = let
		P_dims = (DD.dims(dims, :ε), DD.dims(dims, :ε))
		DimArray(ε_chain.p, P_dims)
	end

	P_from = DimArray(ε_chain.p, (Dim{:from}(ε_grid), Dim{:ε}(ε_grid)))

	states = get_states(dims)

	(; grid = DimVector(grid, state_dim, name = :state), state_dim,
	   ε_grid = DimVector(ε_grid, ε_dim, name = :ε), ε_chain, dims, P, P_from, states, π_permanent, mc_permanent, perm_dim)
end

# ╔═╡ b94a5e80-7c96-4269-bc85-d66850b1b926
function get_cali_auclert(; ξ = 0.0, risk = false, na = 400)
	
	#J = length(dp_marcelo) - 1
	j_dim = Dim{:j}(0:length(dp_marcelo)-2)
	m = DimVector(dp_marcelo[2:end], j_dim, name = :m)
	
	h = DimVector(h̄_marcelo, j_dim, name = :h)

	#amin = 0.0
	Z̲ = 0.0#01
	if risk
		ε_chain = ε_chain_AMMR(1.0; n = 3, n_std = 2.0)
		permanent = permanent_states_AMMR(n = 3, n_std = 2.0)
	else
		ε_chain = no_income_risk()
		permanent = no_permanent_states()
	end
	
	statespace = Statespace(; 
							amin = Z̲, amax = 1_000.0, na, 
							ε_chain,
							permanent
						   )

	r = 0.025
	δ = 0.057
	
	prices = (; r, p = 1.2, w = 1.4)

	
	par = get_par₀(; h, m, ξ, δ, β = β_AMMR(), bonds2GDP = 1.07 - 0.36, NFA2GDP = 0.0, α = 0.35, θ = 1.0 #=(1-δ)/(1+r)=#, Z̲ = -Inf, a̲ = 0.0) #-Z̲/(1+r) )

	π_init = trivial_initial_distribution(statespace, init_state = 0.0)
	
	(; par, statespace, prices, π_init)
end

# ╔═╡ a19c845c-c6a5-46c7-8703-a9fd973933ee
let
	(; par, statespace, prices, π_init) = get_cali_auclert(ξ = 0.15, risk = true)
end

# ╔═╡ 60cb0cba-cf3d-4a67-9b76-8476e4b7e630
get_cali_test(; amax = 12.0, na = 500, exponential = false) = let
	
	m = mortality(:marcelo, age_min = 20, age_max = 91)

	j_dim = DD.dims(m, :j)
	J = maximum(j_dim)
	h = income_profile(J+1, 41)

	ε_chain = no_income_risk2()
	permanent = no_permanent_states2()
	
	statespace = Statespace(; 
							amin = 0.0, amax, na, 
							ε_chain, permanent, exponential)
	
	par = get_par₀(; h, m, ξ = 0.0, δ = 0.1, β = 0.995,
				   bonds2GDP = 1.0, NFA2GDP = 0.0, τ = 0.0,
				   ν₀ = 0.0, annuities = false,
				   α = 0.33, θ = 1.0, Z̲ = 0.0, a̲ = 0.0)

	π_init = trivial_initial_distribution(statespace, init_state = 0.0)

	inheritances = no_inheritances(par, statespace)
	(; par, statespace, π_init, inheritances)
end

# ╔═╡ 33587eca-6212-4ace-972c-90c119ddc001
# ╠═╡ skip_as_script = true
#=╠═╡
let
	(; par, statespace, inheritances, π_init) = get_cali_test(; amax = 12.0, na = 500, exponential = true)

	(; prices, guesses) = let
		K_guess = 3.5832334751342727
		L_guess = 1.0
		r = interest_rate(K_guess, L_guess, par) 
		w = wage(K_guess, L_guess, par)

		guesses = (; K_supply = K_guess, L_eff = L_guess, H_hh = 1e-8)
		prices = (; r, w, p = 1.2)

		(; prices, guesses)
	end

	GE = stationary_PE(HousingModel(), par, statespace, guesses, prices)

	guessed_path = let
		t_dim = Dim{:t}(0:200)
		K₀ = GE.guesses.K_supply
		DimStack(
			DimVector(
				[range(0.9 * K₀, K₀, length = 101); fill(K₀, 100)],
				t_dim, name = :K_supply),
			fill(GE.guesses.H_hh, t_dim, name = :H_hh),
			fill(GE.guesses.L_eff, t_dim, name = :L_eff)
		)
	end

	par_cohort = (; par.h, par.m, par.ρ_SS)
	par_x = drop_m_h(; par...)

	price_paths = get_price_paths(HousingModel(), guessed_path, par; GE₀=GE)

	(; π_permanent) = statespace
	i_perm, permanent = first(enumerate(get_states(π_permanent)))
	inherit_j = @view inheritances[θ = i_perm]
		
	fig = Figure(size = (400, 200))
	ax = Axis(fig[1,1])
	
	for (i, t_born) ∈ enumerate(-10:20:30)
		j_init = max(0, -t_born)
		label = "born: $t_born"

		out = simulate_cohort(HousingModel(), par_x, par_cohort, permanent, statespace; price_paths, π_init, inherit_j, j_init, t_born)

		#path = parent(out.state_path)
		
		@chain out.sim_df begin
			@groupby(:j)
			@combine(:state = mean(:state, weights(:π)))
			lines!(ax, _.j, _.state; label)
		end
		
		#@info abs.(path - compare_paths[i]) ./ (1 .+ abs.(path))
		
		cf = compare_path_no_annuities[i]
		lines!(ax, 0:length(cf)-1, cf, linestyle = (:dash), color = :lightgray)
	end

	Legend(fig[1,2], ax, position = :cb)
	fig
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
DataFrameMacros = "75880514-38bc-4a95-a458-c2aea5a3a702"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuantEcon = "fcd29c91-0bd7-5a09-975d-7ac3f643a60c"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
AlgebraOfGraphics = "~0.11.7"
CairoMakie = "~0.15.6"
Chain = "~1.0.0"
DataFrameMacros = "~0.4.1"
DataFrames = "~1.8.0"
DimensionalData = "~0.29.23"
Interpolations = "~0.16.2"
PlutoLinks = "~0.1.7"
PlutoTest = "~0.2.2"
PlutoUI = "~0.7.71"
QuantEcon = "~0.16.8"
Roots = "~2.2.10"
StatsBase = "~0.34.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.10"
manifest_format = "2.0"
project_hash = "5cd8db8bfbe8df3144190fb8aa3bc43d477846ac"

[[deps.ADTypes]]
git-tree-sha1 = "f7304359109c768cf32dc5fa2d371565bb63b68a"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.21.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "856ecd7cebb68e5fc87abecd2326ad59f0f911f3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.43"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AlgebraOfGraphics]]
deps = ["Accessors", "Colors", "DataAPI", "Dates", "Dictionaries", "FileIO", "GLM", "GeoInterface", "GeometryBasics", "GridLayoutBase", "Isoband", "KernelDensity", "Loess", "Makie", "NaturalSort", "PlotUtils", "PolygonOps", "PooledArrays", "PrecompileTools", "RelocatableFolders", "StatsBase", "StructArrays", "Tables"]
git-tree-sha1 = "41b570924747cf4b9b463d4ae1af9129948b06ca"
uuid = "cbdf2221-f076-402e-a563-3d30da359d67"
version = "0.11.10"

    [deps.AlgebraOfGraphics.extensions]
    AlgebraOfGraphicsDynamicQuantitiesExt = "DynamicQuantities"
    AlgebraOfGraphicsUnitfulExt = "Unitful"

    [deps.AlgebraOfGraphics.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "d81ae5489e13bc03567d4fbbb06c546a5e53c857"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.22.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BaseDirs]]
git-tree-sha1 = "bca794632b8a9bbe159d56bf9e31c422671b35e0"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.3.2"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "5017d6849aff775febd36049f7d926a5fb6677ec"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.8"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.Chain]]
git-tree-sha1 = "765487f32aeece2cf28aa7038e29c31060cb5a69"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "1.0.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "b7231a755812695b8046e8471ddc34c8268cbad5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.0"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSolve]]
git-tree-sha1 = "78ea4ddbcf9c241827e7035c3a03e2e456711470"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.6"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "76dab592fa553e378f9dd8adea16fe2591aa3daa"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.6"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5989debfc3b38f736e69724818210c67ffee4352"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.4"
weakdeps = ["OffsetArrays"]

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrameMacros]]
deps = ["DataFrames", "MacroTools"]
git-tree-sha1 = "5275530d05af21f7778e3ef8f167fb493999eea1"
uuid = "75880514-38bc-4a95-a458-c2aea5a3a702"
version = "0.4.1"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "c55f5a9fd67bdbc8e089b5a3111fe4292986a8e8"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.6"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "a55766a9c8f66cf19ffcdbdb1444e249bb4ace33"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.6"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "7ae99144ea44715402c6c882bfef2adbeadbc4ce"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.16"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.DimensionalData]]
deps = ["ConstructionBase", "DataAPI", "Dates", "Extents", "Interfaces", "IntervalSets", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "PrecompileTools", "Random", "Statistics", "TableTraits", "Tables"]
git-tree-sha1 = "f2173d2303d4ef81c33ce9a2b82fba38b94060d7"
uuid = "0703355e-b756-11e9-17c0-8b28908087d0"
version = "0.29.26"

    [deps.DimensionalData.extensions]
    DimensionalDataAbstractFFTsExt = "AbstractFFTs"
    DimensionalDataAdaptExt = "Adapt"
    DimensionalDataAlgebraOfGraphicsExt = "AlgebraOfGraphics"
    DimensionalDataArrayInterfaceExt = "ArrayInterface"
    DimensionalDataCategoricalArraysExt = "CategoricalArrays"
    DimensionalDataChainRulesCoreExt = "ChainRulesCore"
    DimensionalDataDiskArraysExt = "DiskArrays"
    DimensionalDataMakieExt = "Makie"
    DimensionalDataNearestNeighborsExt = "NearestNeighbors"
    DimensionalDataPythonCallExt = "PythonCall"
    DimensionalDataRecipesBaseExt = "RecipesBase"
    DimensionalDataSparseArraysExt = "SparseArrays"
    DimensionalDataStatsBaseExt = "StatsBase"

    [deps.DimensionalData.weakdeps]
    AbstractFFTs = "621f4979-c628-5d54-868e-fcf4e3e8185c"
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
    ArrayInterface = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
    CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiskArrays = "3c3547ce-8d99-4f5e-a174-61eb10b00ae3"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "fbcc7610f6d8348428f722ecbe0e6cfe22e672c6"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.123"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "7bebc8aad6ee6217c78c5ddcf7ed289d65d0263e"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.6"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "01ba9d15e9eae375dc1eb9589df76b3572acd3f2"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.1+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6522cfb3b8fe97bec632252263057996cbd3de20"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.18.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "9340ca07ca27093ff68418b7558ca37b05f8aeb1"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.29.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "eef4c86803f47dcb61e9b8790ecaa96956fdd8ae"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.2"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "3bcb30438ee1655e3b9c42d97544de7addc9c589"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.3"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "7528a7956248c723d01a0a9b0447bf254bf4da52"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.5"

[[deps.GeoInterface]]
deps = ["DataAPI", "Extents", "GeoFormatTypes"]
git-tree-sha1 = "b7c5cdf45298877bb683bdda3f871ff7070985c4"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.6.0"
weakdeps = ["GeometryBasics", "Makie", "RecipesBase"]

    [deps.GeoInterface.extensions]
    GeoInterfaceMakieExt = ["Makie", "GeometryBasics"]
    GeoInterfaceRecipesBaseExt = "RecipesBase"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "1f5a80f4ed9f5a4aada88fc2db456e637676414b"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.10"
weakdeps = ["GeoInterface"]

    [deps.GeometryBasics.extensions]
    GeometryBasicsGeoInterfaceExt = "GeoInterface"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Inflate", "LinearAlgebra", "Random", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "031d63d09bd3e6e319df66bb466f5c3e8d147bee"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.4"
weakdeps = ["Distributed", "SharedArrays"]

    [deps.Graphs.extensions]
    GraphsSharedArraysExt = "SharedArrays"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "93d5c27c8de51687a2c70ec0716e6e76f298416f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.2"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "4c1acff2dc6b6967e7e750633c50bc3b8d83e617"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.3"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interfaces]]
git-tree-sha1 = "331ff37738aea1a3cf841ddf085442f31b84324f"
uuid = "85a1e053-f937-4924-92a5-1367d23b7b87"
version = "0.3.2"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "65d505fa4c0d7072990d659ef3fc086eb6da8208"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.2"
weakdeps = ["ForwardDiff", "Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "02b61501dbe6da3b927cc25dacd7ce32390ee970"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.2"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "d966f85b3b7a8e49d034d27a189e9a4874b4391a"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.13"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "80580012d4ed5a3e8b18c7cd86cebe4b816d17a6"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.9"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "4260cfc991b8885bf747801fb60dd4503250e478"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.11"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "97bbca976196f2a1eb9607131cb108c69ec3f8a6"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d0205286d9eceadc518742860bf23f703779a3d6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.3+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Printf"]
git-tree-sha1 = "9ea3422d03222c6de679934d1c08f0a99405aa03"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.5.1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics", "StatsAPI", "StatsFuns"]
git-tree-sha1 = "b1ad83b367b915e2dc485dee3d62a6a6317d7ad4"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.6.5"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "65ae3db6ab0e5b1b5f217043c558d9d1d33cc88d"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.5.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "d1b974f376c24dad02c873e951c5cd4e351cd7c2"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.8"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "7eb8cdaa6f0e8081616367c10b31b9d9b34bb02a"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "25a6638571a902ecfb1ae2a18fc1575f86b1d4df"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.10.0"

[[deps.NLopt]]
deps = ["CEnum", "NLopt_jll"]
git-tree-sha1 = "624785b15005a0e0f4e462b27ee745dbe5941863"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "1.2.1"

    [deps.NLopt.extensions]
    NLoptMathOptInterfaceExt = ["MathOptInterface"]

    [deps.NLopt.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b0154a615d5b2b6cf7a2501123b793577d0b9950"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.10.0+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "567515ca155d0020a45b05175449b499c63e7015"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.29+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "df9b7c88c2e7a2e77146223c526bf9e236d5f450"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.4.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c9cbeda6aceffc52d8a0017e71db27c7a7c0beaf"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.5+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "EnumX", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "48968edaf014f67e58fe4c8a4ce72d392aed3294"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.13.3"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "cf181f0b1e6a18dfeb0ee8acc4a9d1672499626c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.4"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0662b083e11420952f2e62e17eddae7fc07d5997"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "844a829c8dc9fd0fe62eced22bc2d0dfd66a3f51"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.1.0"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "10c258e189b8d097c1404ed59f6c171281a39b85"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.7"

[[deps.PlutoTest]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "Test"]
git-tree-sha1 = "09d065f418b85a6ea4be2236d2f33dd720fb1d2a"
uuid = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
version = "0.2.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3ac7038a98ef6977d44adeadc73cc6f596c08109"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.79"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "972089912ba299fba87671b025cd0da74f5f54f7"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.0"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "211530a7dc76ab59087f4d4d1fc3f086fbe87594"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.2.3"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.QuantEcon]]
deps = ["DSP", "Distributions", "FFTW", "Graphs", "LinearAlgebra", "Markdown", "NLopt", "Optim", "Primes", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase"]
git-tree-sha1 = "441453af42d42c42beeadf6cab81e313c38c493f"
uuid = "fcd29c91-0bd7-5a09-975d-7ac3f643a60c"
version = "0.16.8"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "InteractiveUtils", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "14d1bfb0a30317edc77e11094607ace3c800f193"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.13.2"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "8a433b1ede5e9be9a7ba5b1cc6698daa8d718f1d"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.10"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ebe7e59b37c400f694f52b58c93d26201387da70"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.9"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5acc6a41b3082920f79ca3c759acbcecf18a8d78"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eee1b9ad8b29ef0d936e3ec9838c7ec089620308"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.16"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "08786db4a1346d17d0a8d952d2e66fd00fa18192"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.9"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a3c1536470bf8c5e02096ad4853606d7c8f62721"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.2"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "a2c37d815bf00575332b7bd0389f771cb7987214"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.2"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "28145feabf717c5d65c1d5e09747ee7b1ff3ed13"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.3"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "98b9352a24cb6a2066f9ababcc6802de9aed8ad8"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.6"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "57e1b2c9de4bd6f40ecb9de4ac1797b81970d008"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.28.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9cce64c0fdd1960b597ba7ecda2950b5ed957438"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.2+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e015f211ebb898c8180887012b938f3851e719ac"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.55+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"
"""

# ╔═╡ Cell order:
# ╠═5a6d468f-327e-4c06-97a3-bcee431a3518
# ╠═1a4a13be-2913-4710-bae4-3899fa415198
# ╠═9c88ad1f-8a68-4260-8416-da6745623dc1
# ╠═5d8c511b-d97f-4479-873d-4810e0c67a0d
# ╠═adb79a2d-cf1e-42a4-a821-d9afd37b73bf
# ╠═d271fb80-8c93-11f0-3406-ab394e785ceb
# ╟─91fcc5f6-0789-4fb4-9984-105922332395
# ╠═880637f3-81f0-48f5-8918-c03dac35e6fc
# ╠═ca363b65-dd42-475c-9114-a259691c7913
# ╠═0c481dec-3a04-4a51-9c11-ef03dbab3683
# ╠═976144e3-de4b-471a-868b-96abc821ca84
# ╠═2b26bf96-95e8-4e6e-8ba4-f1a4a7857464
# ╠═3c2f3f62-f7b9-4d66-8598-be8bb6bd6356
# ╠═a44e51f9-3f90-4ec8-b890-e53e7b9549f6
# ╠═cda15204-1cd1-4bd1-b5b6-ac72ed154309
# ╠═35347fb7-2085-4d06-9d17-60757bd331c3
# ╠═c176b6b1-75aa-479d-a34c-22ecc4823e27
# ╠═9291d566-bf4f-49ac-aba3-277170e5daed
# ╠═7d9ea25a-6f76-43a2-83ab-d81e6210bbc6
# ╠═c94759d1-b5b0-4320-8f42-c57d3bf6ca2a
# ╠═b6976a71-69c5-4016-9bf1-859b3208485f
# ╠═6859961c-f24f-4ae3-bb5c-1bc639ec00a1
# ╠═f8919942-9ef9-476a-94eb-c46ec88fde94
# ╠═a17f0114-55f6-4d43-9b05-6bcd9601b98b
# ╠═815c5b9f-e329-4a2e-b0fe-667d2052d980
# ╠═133e1e40-4b84-4c9d-803f-22c2884e81f5
# ╠═16ed251b-5ae9-4520-88fa-1e67987615db
# ╠═917206f6-abd6-4ffb-a738-6c39024cbaaa
# ╠═00c2612f-8f1c-4412-a159-e4325af0c62f
# ╠═acb38371-e502-4b9c-80f4-314e9458edcc
# ╠═1ba9f113-c2d5-4c34-ba9f-01d39ffc6f35
# ╠═3de09148-706d-40ed-90b2-39a71e9d25ed
# ╠═b35bfc6e-6f39-461a-b5b4-ca2901c83da4
# ╠═9276b022-d863-473e-978a-67a614d7ee31
# ╠═fa757572-54a1-4a4a-975f-72c0bccc67aa
# ╠═4a7ee470-4496-471c-86a0-c36d3b6c5a7f
# ╠═0ae74c43-bae4-406b-9e31-fd76a0a4a398
# ╠═26bc0666-0405-466b-8dea-356d9d7c4e19
# ╠═adfca787-4bc4-419f-9a0f-12e051c36134
# ╠═f66ab91b-14d6-4981-b78e-ab6f55129220
# ╠═59909171-3b97-4f96-993f-db0aa09e5e00
# ╠═501b1a2d-3b7c-4bd9-a119-3e2b829b7a56
# ╠═0c59fd3d-263d-43ba-8ba1-5138d767cf48
# ╠═24c68069-e2d6-428e-9b08-014263b8f07b
# ╠═6e45e994-8a1f-4b79-85bd-fb999bc5493d
# ╠═685427d8-9f41-482e-bb08-7fbe7aff32ef
# ╠═ae5bd8c1-dca5-495d-a5bb-a2d271262645
# ╠═0bfd4678-c05c-4e62-9f11-6a1d80f4f58a
# ╠═9e05ea7c-bc7a-4dbd-8199-298c53447d7c
# ╠═cfb2d2e9-f101-464f-915f-b0f6a38744e5
# ╠═07a443b3-7233-43a5-aacd-7fb79bec0edf
# ╠═6569e751-ce87-4063-a142-ad62d088e70d
# ╠═6b991515-29e8-4fba-a473-6dd1f8cb9d03
# ╠═34285cb0-4c79-43bc-a299-d2571b77e9fa
# ╠═c28cec8b-2ea5-4bf8-a935-7eb00af0b2a5
# ╠═f209f007-46a5-4100-94f9-db3bda5254db
# ╠═5bb0b7e1-7476-4266-9a02-511a6e24f165
# ╠═843d8157-85c4-4ef9-8fc4-7b4a83b01d17
# ╠═26bce906-16cf-4dd8-a9e8-a09d5270a6fd
# ╠═8f9c937b-5a00-40ce-93ef-cbfc8058dacf
# ╠═277e1e63-4df5-42a0-a5f3-ad0b73eab310
# ╠═1f07f950-c69e-4a12-b404-6af89796c58e
# ╠═5b07590c-b6f9-489e-8c8f-1757ede48cbc
# ╠═97fc998b-83cc-4d35-b78a-0c9b9bc3cfad
# ╠═ce1c7b3f-cc7d-4aca-afd1-cee741df6f2a
# ╠═9b244937-36f3-4d44-b509-6f7d5bef1bca
# ╠═9e6a5abd-fad3-43e7-b4ba-2d8adaf19a9b
# ╠═1b1e4b39-a845-4e77-b05c-fc6e173b27f0
# ╠═c500744d-c0cd-4dc1-a04a-7dd3f1ce860c
# ╠═08dd6081-9eb4-41ec-b0e1-bea6a4a585ea
# ╠═b04b20f2-8397-415d-bd7f-567f5f175e83
# ╠═ff35e19b-8556-4a2d-9ab8-96e305b04ced
# ╠═01b044f2-fb06-497e-9595-3be278a1c57e
# ╠═f5847dfa-d4ae-4d1a-b517-f166465e320a
# ╠═314231da-5294-4a12-a4fb-5023dbbd31cf
# ╠═3b65a9ac-9675-46cf-a545-ae3dc3f1daa7
# ╠═61ff86de-de16-49af-a9a5-6c7c2b23420f
# ╠═e11ae85f-c6f6-4381-b5d3-d756124f1c74
# ╠═8ae3665e-bba7-4032-93a7-9c546d347f60
# ╠═af69b84f-e4f5-4f90-85a3-ef00d2288b83
# ╠═36bde726-ac0f-4e38-baec-b507ecb0b9d1
# ╠═9ae9deb1-b9ee-4c8b-b1a4-8c75323a56cb
# ╠═9ffedac3-0ef1-42f6-932e-01f74b46531c
# ╠═d2262948-178c-4e42-a7e7-0f38aa76e7db
# ╠═03bc50bc-5c74-4afe-ae3f-9fed0510817e
# ╠═fbc03e03-c9f7-4edc-85a4-2d4ae3268edf
# ╠═93b9f4a0-ec6c-481d-9b8a-5f04401f11cf
# ╠═d637ff0e-cd32-42c7-a9f2-8e2e4e3b3bbd
# ╠═aabbd49a-a1cb-4054-8beb-4b542957dbfa
# ╠═3768ac04-c428-4f45-80b9-8cb6966c72aa
# ╠═742da011-57db-41b0-8f64-9fcecd4c0324
# ╠═a09536e2-5095-42e8-ab26-3b00e4b4cdce
# ╠═31ed0294-5b87-4697-b1d2-a675b93dadb2
# ╠═ada6fdbc-a135-4814-8b44-2b89a24c1699
# ╠═eb0848c3-d4d1-4088-a4d5-2809dac09a4a
# ╠═b8e2f4d8-4a8e-469f-8b15-0533f01246a6
# ╠═306cc046-cd55-435a-9a8f-7ca7d0c158c8
# ╠═4107c1bd-44f3-42f8-b496-d0eff27b40fd
# ╠═43e3c9c8-8728-44c4-b8b1-fdc1220c4d17
# ╠═7f76c5ff-b8d2-466b-b064-3dceb1071062
# ╠═bcd59e5d-801f-4465-832f-b880fb0b73e8
# ╠═4fd9f46e-8eb8-42da-be0f-eb06930e494a
# ╠═f1e66bbb-3b6d-4aa2-895e-6604dd5bb0ae
# ╠═810a4a12-0797-4e4b-bd0a-134873430ca3
# ╠═d9161c2d-e491-4c25-aea9-088661009ff2
# ╠═b274b8b0-f343-49d7-8d0b-e8802c9d1500
# ╠═16a46962-f6be-4536-a2f2-f52fb796d2da
# ╠═86a97101-02b6-4b36-b458-ceae742d5c65
# ╠═ff84590d-8e81-47b4-bdd0-8271489be19c
# ╠═2a2742c1-a812-4a7b-b8fd-bfc195ac2c99
# ╠═33f2a5c0-5bd5-4d7f-95ba-03035206fde6
# ╠═081d029a-d67d-46a4-bf87-8a55f516fa30
# ╠═3ecb48cc-32c3-49f5-96bf-aad651d81e30
# ╠═47232117-6797-4188-a0e6-8a2e2a245e66
# ╠═a49de89e-ff8d-4447-9dcd-491bb0126db6
# ╠═d0420ab9-27c8-4686-9f59-f643be9f6410
# ╠═d645fe54-fed9-49c6-b20a-94239ca7eb7e
# ╠═a8da0aea-9b8d-455a-a87e-afa846fa771d
# ╠═32faac51-9b53-49a8-87da-61860d3f227a
# ╟─33d17a66-cb65-4b8c-a269-f7a3f82ac32e
# ╠═adce8682-ef5a-42ea-a59b-15df0cf1685f
# ╠═a2926b44-77ec-42fb-979e-b6d23a8e56b8
# ╠═ce458e11-8612-4c39-9f16-0413edc16586
# ╟─3edf748d-e2fc-4148-82f4-dfcd541d5901
# ╠═0d4205f1-e578-4d9b-9260-0a174d89fc45
# ╠═139b6f9a-4073-40a3-b587-6af681e47069
# ╠═37c98a61-3350-4eb4-a3fe-8cd8c2b5eab5
# ╠═b94a5e80-7c96-4269-bc85-d66850b1b926
# ╠═52df027a-b50a-4a3a-99dc-80bce86c77cc
# ╠═31352b9e-69cc-4467-aa89-34249666e407
# ╠═626e38d7-e337-4853-9046-3d9b1485bb24
# ╟─f103ea34-ad3c-49e1-9e6f-d94d4b3574e6
# ╠═3303493e-6502-479f-b383-82ffd1ab17bf
# ╠═765cfe47-b191-4228-b419-ffb6cd2bd9f4
# ╠═9c050921-f2cc-4e94-a366-6a519d7f4116
# ╠═a19c845c-c6a5-46c7-8703-a9fd973933ee
# ╠═208efe14-68dc-4389-bb9b-8bd996bf4749
# ╟─32dd3921-c58a-41cb-9f11-8d4efa9f7641
# ╠═32b95f4a-180c-4a61-b899-6b604d911926
# ╠═9b706ccd-bc47-4cc8-b315-fc17375cbd93
# ╠═48443da6-0ffc-4ae2-9ebe-c4d0461ecfca
# ╠═998cb794-525c-4b2d-88c7-80d711e06d3b
# ╠═b3219d9d-2bfe-445f-8f3a-4ccfa95673ed
# ╠═ed3bf636-176a-422a-812d-07bc3be8fef4
# ╠═d05e5a31-5a8b-4cad-85c5-5d6c89c67ad2
# ╠═4df66cbd-3c0d-40db-8b5b-65b26ffc4c79
# ╠═31a39f30-740f-4a30-b3f5-e1fef2502138
# ╠═360d9593-4c34-4bbe-bde6-a554803861f1
# ╠═77d631d8-6da6-4ec3-a8d2-ebb71f9c474c
# ╟─e443249e-dc9f-49b6-bd56-693fa1e49242
# ╠═33587eca-6212-4ace-972c-90c119ddc001
# ╠═066cd6cb-4bc2-418e-a04e-f9a83808de56
# ╠═60cb0cba-cf3d-4a67-9b76-8476e4b7e630
# ╠═4491634a-deff-4805-a7e5-9e03e9198d53
# ╠═e51238ae-eea3-4956-b22a-59fb42f4a9e7
# ╠═5cf0f0a8-4fa9-452d-9d58-4bccc1e42e5d
# ╠═e59b1142-1725-429c-892d-dbc70d8b035e
# ╠═f49d0c50-2cfa-4cfb-9224-6607fb7f99eb
# ╠═ee7e5f71-5182-4b8d-9e6f-aa651cb9c1ca
# ╠═5c94956c-0675-423f-bf96-30f70bd81009
# ╠═3365570c-2cbe-4caf-b56f-80dd9f3b6529
# ╠═dfc5566b-187c-40f3-9349-99584efe1b64
# ╠═86a050f8-cc13-443c-881f-58f3660ec607
# ╠═a4a7158e-1455-478d-b9a8-cdc6f2ddc715
# ╠═9ac247d6-1b9d-4bdd-bdb4-0d62ecba61d7
# ╠═15efda29-e6d4-4096-9f5b-861eaf81e482
# ╠═a9af97d9-8905-4550-afac-458ffe737a14
# ╠═4659e2ae-38ca-411f-b180-9589c70c3afb
# ╠═2d4581c2-6801-4754-a6b1-b91ae131e9f8
# ╠═bacf8ffa-cb2b-4e0e-989d-c4c84b4d8d2c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
