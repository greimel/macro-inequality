### A Pluto.jl notebook ###
# v0.20.16

using Markdown
using InteractiveUtils

# ╔═╡ b51f0bc6-417d-42c7-97a1-e15d001ca5cb
using GLM

# ╔═╡ 3b616808-0e94-4f34-81f2-7835775e1fdb
using LaTeXStrings

# ╔═╡ e5b26e79-d213-4f9e-a87b-e737cd12bd51
using Sobol

# ╔═╡ e8c5c3aa-cda6-4e04-a678-378c394f728a
using LinearAlgebra: norm

# ╔═╡ 33144d79-63e0-4a0e-a88d-18592f385ab2
using Polynomials

# ╔═╡ 3bcc33e3-9947-4676-9a09-edec34bd5df4
using PlutoLinks: ingredients

# ╔═╡ 1538744e-d666-414f-afeb-d92f09de5594
using PlutoUI

# ╔═╡ 074b8836-fd0e-4920-8a4e-44cac30c0446
using Chain, DataFrames, DataFrameMacros

# ╔═╡ a994c130-e869-4393-bc45-fff782a4a04f
using CairoMakie, AlgebraOfGraphics

# ╔═╡ c49c5989-6855-4b2b-a5a6-c31f3c92224d
using PlutoTest

# ╔═╡ 2378c45e-2f5a-4046-8c2d-096cdee9542f
using StatsBase

# ╔═╡ 1733f2bf-4a4e-49f8-8f89-9f74103759a5
using DimensionalData

# ╔═╡ 9d9c3fb1-a6a8-4f12-bf83-53033e9a33e4
using LinearAlgebra: dot

# ╔═╡ 03d376b5-a7e5-4641-893a-b5e6670aa201
using Roots: find_zero

# ╔═╡ 0dabf828-554c-49de-b6d8-ff0e5561b3c2
using Interpolations

# ╔═╡ f957f853-b32b-4a67-a7d0-3cc62cda70da
using QuantEcon: QuantEcon, rouwenhorst, MarkovChain, stationary_distributions

# ╔═╡ 6320fc56-94af-444f-807d-5a1b86113e16
using Statistics: mean

# ╔═╡ 6389c2d9-fc3b-4a1e-a771-4735b4ff6039
EGMHousingRisk = ingredients("./egm-housing-risk.jl")

# ╔═╡ 3aa144a9-c675-43c7-ac55-a74307e48088
(; HousingModel, stationary_GE, stationary_PE, transition_PE, transition_GE) = EGMHousingRisk

# ╔═╡ 3e877d30-74a1-4abe-9d38-83ecf820860f
(; get_states, prices_from_guesses_nt, constant_price_paths) = EGMHousingRisk

# ╔═╡ 6bf61e31-e899-47d7-a770-d649ff3198da
md"""
# To do

* move over transition code from `models.jl`
* check why there is a jump in in the last period
"""

# ╔═╡ 309a7d48-9693-4924-9748-73cae12a8739
md"""
# to do
* reduce death probabilities by 0.9
"""

# ╔═╡ e9e1887c-ced6-4af7-b5fb-e938836e5713
md"""
# Test
"""

# ╔═╡ f2d0e8ec-4be5-4a55-8539-b65eadc61304
md"""
## Bequests

In Auclert et al the bequests are distributed over the life-time, the distribution is taken from data. What could be other (simpler) options to pay out bequests?

* distribute uniformly across all ages
* distribute only to newborns
* distribute according to ``F`` (as in Auclert et al)
"""

# ╔═╡ f91ce131-6387-4646-ae91-c2ffa7320496
md"""
### Computing bequests

1. Start with distribution of wealth over ``j``: ``(\omega_j(\theta), \pi_j(\theta))_j`` and mortality rates ``(m_j(\theta))_j``, all depend on the permanent state ``\theta``.
2. for a given permanent type ``\theta`` the total bequests are given by
   
```math
𝕓_i = \sum_j \omega_j(\theta_i) m_j \pi_j(\theta_i).
```
"""

# ╔═╡ c27a5a24-6173-4d1c-baca-41efbdd2b07c
md"""
### Computing inheritances (from ``P_\theta``)

There are ``N`` permanent types ``(\theta_1, \ldots, \theta_N)``. These types are persistant across generations. The evolution of permanent types across generations is governed by a Markov Chain ``P_\theta``.

A given type ``\theta_i`` will (potentially) have descendents of each other type. The distribution of descendants' types is given by ``{P_\theta}_{i,:}``.

Say that the total bequests of type ``i`` are given by ``𝕓_i``, so the vector of bequests is ``𝕓``. So the vector of total inheritances ``𝕚`` is given by

```math
\pmatrix{\mathbb{i}_1 \\ \mathbb{i}_2 \\ \mathbb{i}_3} = \underbrace{\begin{pmatrix}
P_\theta(1,1) & P_\theta(2,1) & P_\theta(3,1) \\
P_\theta(1,2) & P_\theta(2,2) & P_\theta(3,2) \\
P_\theta(1,3) & P_\theta(2,3) & P_\theta(3,2)
\end{pmatrix}}_{P_\theta^T} \cdot \pmatrix{\mathbb{b}_1 \\ \mathbb{b}_2 \\ \mathbb{b}_3}
```

Take, for example, the first row. What matters for the inheritances of type 1 is how many type-1-descendents each type has: Type 2 has ``P_\theta(2,1)`` descendants of type 1.
"""

# ╔═╡ a8d2e703-d6ae-459e-9892-2086d2ba38a6
md"""
### Distributing inheritances across cohorts

Now that we have the total inheritances ``(\mathbb{i}_1, \mathbb{i}_2, \mathbb{i}_3)`` we can divide the inheritances across cohorts of each age.
"""

# ╔═╡ 49c2d27d-862b-4e4e-b150-fbaad14dc476
function mean_inheritances(inheritances, π_age)
	@chain begin
		DimStack(inheritances, π_age)
		DataFrame
		@transform(:mass_by_age = @bycol :mass_by_age / sum(:mass_by_age))
		@combine(:out = sum(:inheritances, weights(:mass_by_age)))
		(; mean_inh = only(_.out))
	end

end

# ╔═╡ ab139efa-c2fe-4e73-a663-3b7b6ff07cc5
md"""
p = 0.575
r = 0.012
"""

# ╔═╡ d393045c-e3be-447a-a172-340f29878d9b
#=╠═╡
let
	out = out_bequests
	
	
	vars = [:z_next, :z, :a_next, :c, :income, :ho, :inheritance, :constrained]

	df = @chain out.sim_df begin
		select(vars..., :π, :j, :permanent => AsTable)
		@transform!(:θ = round(:θ, digits = 2))
	end
	
	qs = 0.1:0.1:0.9
	@chain df begin
		stack(vars, [:π, :j, :θ])
	    @groupby(:variable, :j, :θ)
		@combine(
			:grp = [fill("Q", length(qs)); "mean"],
			:q = [string.(qs); "mean"],
			:value = [quantile(:value, weights(:π), qs); mean(:value, weights(:π))]
			
					  )
		@subset(_, :grp == "mean")
		#@subset(:θ == 0.05)

		data(_) * mapping(:j => L"model age $j$", :value,# => log => L"\log(\cdot)", 
						  group = :q => nonnumeric,
						  color = :grp,
						  #group = :θ => nonnumeric,
						  linestyle = :θ => nonnumeric,
						  layout = :variable
						 ) * visual(Lines)
		draw(; facet = (; linkyaxes = false), figure = figure((450, 250))) # 
	end
end
  ╠═╡ =#

# ╔═╡ ed3baf6e-8d45-46b7-a873-3f24d351d4ca
no_inheritances((; j_dim), (; perm_dim)) = zeros((perm_dim..., j_dim), name = :inheritances)

# ╔═╡ cf22e4a9-c04c-4747-bf8b-61df14444e9c
#=╠═╡
let
	bequests_by_type = get_bequests_by_type(out_auclert, cali_auclert.statespace)
	inheritances_by_type = get_inheritances_by_type(bequests_by_type, cali_auclert.statespace)

	@info bequests_by_type |> mean 
	#inheritances_by_type |> sum

	
	(; par) = cali_auclert
	(; j_dim) = par

	F = DimVector(F_marcelo ./ sum(F_marcelo), j_dim, name = :F)

	π_age = get_age_distribution(out_auclert)
	
	inheritances = get_inheritances(inheritances_by_type, F, π_age)

	@chain begin
		DimStack(inheritances, π_age)
		DataFrame
		@transform(:mass_by_age = @bycol :mass_by_age / sum(:mass_by_age))
		@combine(sum(:inheritances, weights(:mass_by_age)))
	end
	#=
	@chain inheritances begin
		DataFrame
		data(_) * mapping(:j, :layer1, color = :θ => nonnumeric) * visual(Lines)
		draw
	end
	=#
end	
  ╠═╡ =#

# ╔═╡ edab0641-596e-4e0f-a90c-0b1a75db3264
md"""
# Comparing standard test case with Marcelo
"""

# ╔═╡ c6eaee11-dca8-4b7a-a4e6-2fc7fced8df8
#=╠═╡
out_auclert.aggregates.upd
  ╠═╡ =#

# ╔═╡ b9df5ec6-e00a-4b21-85be-24304a10f47c
@info @test out_auclert.prices.r ≈ 0.02580117167515541

# ╔═╡ 9510b505-6504-40fd-948d-d130c0f47de5
md"""
Marcelo:

* with bequests: r = 2.08% vs 0.019836768116479102 (F)
* without bequests: r = 2.71% vs 0.025844305341543382 (F)
"""

# ╔═╡ dcbeee05-1abb-4657-80ac-67b1dc8478e8
md"""
# Replicating Auclert et al

* permanent types + bequests
* growth
"""

# ╔═╡ 111639a7-125f-4908-841c-b3e045a2b719
md"""
### Note

* aggregate `ℓ_eff` is exactly the same
* slight difference in aggregate `a_next`
* use `ℓ_eff`
"""

# ╔═╡ 47f76c94-8667-4cfb-ba20-63ce7f34b420
md"""
* ``θ = (1-δ)/(1+r)``: constrained = 0.0, ho = 8.22614, a_next = 63.7929
* z > 0:  ho = 8.22616, 63.7929


* `Z̲ = 2.5`: constrained = 0.0151896, ho = 8.378, a_next = 64.999

* ``amin = 2.5``: ho = 8.378, a_next = 74.2509
"""

# ╔═╡ a9504b67-8109-4f0c-94cc-d362b2ddc09e
# ╠═╡ disabled = true
#=╠═╡
out_auclert = let
	(; par, statespace, prices, π_init) = cali_auclert

	Mo = HousingModel()

	guesses = (; K_supply = 23.07818, H_hh = 8.55e-8, L_eff = 5.0)
	#guesses = (; K_supply = 5.0, H_hh = 1e-14)
	
	out = stationary_PE(Mo, par, statespace, guesses, prices; π_init) 

	#out = partial
	
	out = stationary_GE(Mo, par, statespace; guesses, tol = 1e-4, maxiter = 100, λ = 0.45, details = true, π_init)
	
end
  ╠═╡ =#

# ╔═╡ b42e1037-84b5-48d6-8211-3fadd8372f07
md"""
## Calibration
"""

# ╔═╡ debd6eea-7f66-4f16-b3d8-4a60835d33a8
function β_AMMR(; age_min = 20, age_max = 96, β̄ = 0.9655, ξ = 0.00071)

	J = age_max - age_min
	 
	β = DimVector(
		[exp(j * log(β̄) + ξ * (j - (40 - age_min))^2) for j ∈ 0:J],
		Dim{:j}((0:J))
	)

	β[j = At(0:J-1)]

end

# ╔═╡ aa6cc2fb-89ba-4d29-8db8-2a38761397fb
function get_permanent_states(mc, dimname = :θ)
	permanent_dim = Dim{dimname}(mc.state_values)
	π = only(stationary_distributions(mc))
	π_permanent = DimVector(π, permanent_dim)

	(; mc_permanent=mc, π_permanent)
end

# ╔═╡ 239a48fe-5004-45f8-9b15-e691941ce26e
function no_permanent_states2()
	mc = MarkovChain([0.9 0.1; 0.2 0.8], [0.9999, 1.0001])
	get_permanent_states(mc)
end

# ╔═╡ 96a63de4-87da-4aa2-bbb4-c7f5de256977
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

# ╔═╡ 69ca7b92-9b5a-46b5-8343-e9c59d445f17
md"""
Effective labor supply ``\iff`` productivity

``\ell_{\text{eff}} := \theta \varepsilon \bar h``
"""

# ╔═╡ 25add3f6-e250-4729-8380-1dd80afa1074
md"""
## Parameters from Marcelo
"""

# ╔═╡ a5014d64-97d7-4309-bced-97911603ca91
F_marcelo = [ 
	0.0000134, 0.0000199, 0.0000292, 0.0000425, 0.0000612, 
	0.0000873, 0.0001233, 0.0001723, 0.0002385, 0.0003268,
	0.0004433, 0.0005954, 0.0007917, 0.0010423, 0.0013586,
	0.0017532, 0.0022399, 0.0028333, 0.0035482, 0.0043993,          
	0.0054003, 0.0065630, 0.0078967, 0.0094070, 0.0110945,
	0.0129546, 0.0149760, 0.0171406, 0.0194228, 0.0217900,
    0.0242023, 0.0266143, 0.0289754, 0.0312322, 0.0333297,
	0.0352142, 0.0368350, 0.0381471, 0.0391128, 0.0397039,
    0.0399029, 0.0397039, 0.0391128, 0.0381471, 0.0368350,
	0.0352142, 0.0333297, 0.0312322, 0.0289754, 0.0266143,
    0.0242023, 0.0217900, 0.0194228, 0.0171406, 0.0149760,
	0.0129546, 0.0110945, 0.0094070, 0.0078967, 0.0065630,
    0.0054003, 0.0043993, 0.0035482, 0.0028333, 0.0022399,
	0.0017532, 0.0013586, 0.0010423, 0.0007917, 0.0005954,
	0.0004433, 0.0003268, 0.0002385, 0.0001723, 0.0001233,
	0.0000873
]

# ╔═╡ 55d6648b-0534-4d2a-b904-8708bc8346f9
md"""
**Note** ``F`` sums to 1.0. But shouldn't ``\sum_j \pi_j F_j = 1.0``? (So that ``F`` is the expected bequest per person, not the total bequests per cohort.)
"""

# ╔═╡ b0e87883-bc55-45b3-a290-220b24ae2c16
sum(F_marcelo)

# ╔═╡ d92a8170-1505-4d02-bdb5-77e0f53fdda9
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

# ╔═╡ c32bad5e-3b7d-41a4-a876-c6d25024b346
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

# ╔═╡ 2ee9ab01-5f41-40a9-a8ef-f24ca0deacb0
md"""
# Calibrating lifecycle profiles
"""

# ╔═╡ 7013ac3f-d193-405e-b941-d9c62156f16e
md"""
## Fit a polynomial
"""

# ╔═╡ b42f3a3f-ee8a-4adb-8e02-f5b019501470
md"""
## 2. Sobol numbers
"""

# ╔═╡ 9fbade6d-505c-4511-b929-b2bf1e5ee0b6
md"""
1_000: 2.269 (norm loss; 1.10, 1.98)
10_000: 2.269 (norm loss)
"""

# ╔═╡ 8957a247-8e3c-404d-aa77-8d338f852baf
# 0.343
# i = 4290

# ╔═╡ 22bc4986-1285-448b-90b9-58ba971c653a
function check_βs(β)

	if 0.0 ∈ β
		return false
	end
	if Inf ∈ β
		return false
	end
	if any(<(0.0), β)
		return false
	end
	if any(>(30.0), β)
		return false
	end
	return true
end

# ╔═╡ a8309fe0-a12c-456e-a525-c327fbe7fec8
function check_ξs(ξ)

	if any(<(0.01), ξ)
		return false
	end
	if any(>(0.99), ξ)
		return false
	end
	return true
end

# ╔═╡ 253a308f-9dfe-4820-afb5-3a3d5d3a3d05
md"""
## 3. Loss function etc
"""

# ╔═╡ ef84e72b-aca2-49fc-af78-904545e7ebee
w(x; α = 0.5, a, b) = 1 + α * (2 * (x - a)/(b - a) - 1)^2

# ╔═╡ d030efff-5f4b-4540-9e12-183d8809f1b4
# ╠═╡ disabled = true
#=╠═╡
function XXXcali_out(ξ_pars, β_pars; kwargs...)
	par = age_dependent_par(20, 87, ξ_pars, β_pars)

	cali_out(par; kwargs...)
end
  ╠═╡ =#

# ╔═╡ b26db514-377c-4425-982c-7e5b8ee742bb
function cali_out(model, par, statespace, prices)

	guesses = (; H_hh = 1.0, K_supply = 1.0)
	try 
		out = stationary_PE(model, par, statespace, guesses, prices) #, j_last = par.J)
		return out
	catch e
		if e isa DomainError
			return NaN
		else
			rethrow(e)
		end
	end	

end

# ╔═╡ 5fd01ce3-6bad-452b-bdf5-dbf0826beb79
function coefs_scale_30(coefs)
	coefs
	#sign.(coefs) * (-1) .* log.(coefs .* sign.(coefs))
end

# ╔═╡ 20b876fc-7f11-4b86-9098-cccc177a1e54
function coefs_scale_1(coefs)
	coefs
	#@. sign(coefs) * exp(-abs(coefs))
end

# ╔═╡ 0eafc438-9f61-4f82-a231-89ca63d212f4
function age_dependent_ξ_β(; ξ_coefs, lβ_coefs, J)
	js = 0:J
	j_dim = Dim{:j}(js)

	js_trans = js ./ (J/2) .- 1.0

	ξ_coefs_1 = coefs_scale_1.(ξ_coefs)
	p_ξ = ChebyshevT(ξ_coefs_1)
	lβ_coefs_1 = coefs_scale_1.(lβ_coefs)
	p_lβ = ChebyshevT(lβ_coefs_1)
			
	ξ  = DimVector(p_ξ.(js_trans),   j_dim, name = :ξ)
	log_β = DimVector(p_lβ.(js_trans), j_dim, name = :log_β)
	
	β = DimVector(exp.(log_β), name = :β)

	(; ξ, log_β, β, p_ξ, p_lβ, ξ_coefs_1, lβ_coefs_1, ξ_coefs_30 = ξ_coefs, lβ_coefs_30 = lβ_coefs, js_trans)
end

# ╔═╡ 54332ff2-a35a-4ad1-9248-89c912033eb2
function check_sobol(s; order_ξ, order_lβ, J)
	ξ_coefs = first(s, order_ξ + 1)
	lβ_coefs = last(s,  order_lβ + 1)

	(; β, ξ) = age_dependent_ξ_β(; ξ_coefs, lβ_coefs, J)

	return check_ξs(ξ) && check_βs(β)
end

# ╔═╡ 8c2762d3-b315-424d-9d74-84ae89875e65
function checked_next!(ss; kwargs...)
	s = next!(ss)

	while check_sobol(s; kwargs...) == false
		s = next!(ss)
	end

	return s
end

# ╔═╡ d6fdb670-c877-48d6-b76f-89d5b49db6bb
md"""
# Targets
"""

# ╔═╡ 2817bb63-b9af-485a-85f7-62056c843cbc
datadir(args...) = joinpath(expanduser("~"), "Desktop", "Research", "demographics-overleaf", "julia", args...)

# ╔═╡ 2cd527b2-cf2a-48ce-98cf-40f5c8d5d6c7
scf_df₀ = let
	M = ingredients(datadir("scf_lifecycle_summary.jl"))
	(; df) = M
	@transform(df, :income = :annual_income / 1000)
end

# ╔═╡ 661934d6-6895-45d4-95f2-6a539211e348
cex_df₀ = let
	M = ingredients(datadir("cex_lifecycle_summary.jl"))
	(; df) = M
	@transform(df, :income = :monthly_income / 1000)
end

# ╔═╡ e51d1336-5d0a-465e-91fc-83181511ba3f
targets_df = let
	scf = @chain scf_df₀ begin
		@transform(:age_bin = startswith(:age_bin, "[85") ? "85+" : :age_bin)
		@transform(
			:income = :annual_income / 1000,
			:networth = :networth / 1000
		)
		select(Not(:annual_income))
		
	end

	cex = @chain cex_df₀ begin
		@transform(
					 :age_bin = startswith(:age_bin, "[85") ? "85+" : :age_bin,
					 :age = startswith(:age_bin, "[85") ? 87.0 : :age
					)
		select(Not(:income, :age))
	end
	
	leftjoin(scf, cex, on = [:age_bin])
	
end

# ╔═╡ eb350120-a8e1-433b-a123-b9d33ce4da9f
cex_targets(min_age, max_age) = let
	#variable = :income

	function interpolated(variable)
		itp_line = linear_interpolation(targets_df.age, targets_df[!, variable], extrapolation_bc = Line()
		)
	
		itp_flat = linear_interpolation(targets_df.age, targets_df[!, variable], extrapolation_bc = Flat()
		)

		
		ages  = min_age:max_age
		js = ages .- min_age
		
		A = 50
		ages1 = ages[1:A]
		ages2 = ages[A+1:end]
	
		DimVector([itp_line(ages1); itp_flat(ages2)], Dim{:j}(js), name = string(variable))
	end

	DimStack(interpolated.([:income, :hx_share, :rent_eqv, :n2y, :networth]))
end

# ╔═╡ d93e0243-acfc-4dc9-8ae3-598042c7281b
function loss_cali(cali_out, details = false)
	if cali_out isa Number && isnan(cali_out)
		#@info "Got NaN"
		return loss = Inf
	end
	
	(; sim_df, par, prices) = cali_out

	#@info DataFrame(pmf)
	(; h, δ, age_min, age_max, J) = par
	(; p, r) = prices
	
	#pmf_df = DataFrame(pmf)
	#inc_df = DataFrame(y)
	#rename!(inc_df, :income => :yⱼ)

	cex = @chain cex_targets(age_min, age_max) begin
		DataFrame
		rename(:hx_share => "rent/income", :n2y => "networth/income")
		stack(["income", "rent/income", "networth/income"], :j)
		end


	means_by_age = @chain sim_df begin
		stack(Not([:j, :π, :permanent]))
		@groupby(:j, :variable)
		@combine(:value = mean(:value, weights(:π)))
		unstack(:variable, :value)
	end
	
	tmp = @chain means_by_age begin
		@select(:j, :ho, :z, :y = :income)
		#leftjoin(_, pmf_df, on = :j)
		#leftjoin(_, inc_df, on = :j)
		#rename(:income => :y)
		@transform(
			:rent_equivalent = (δ + r) * p * :ho,
		)
		@transform(
			"rent/income"      = :rent_equivalent / :y,
			"networth/income" = :z / :y # * (1+r) XXX TODO
		)
		rename(:y => "income", :z => "net worth")
		stack(["income", "rent/income", "networth/income"], :j)
		vcat(_, cex, source = :source => ["model", "data"])
		@subset(:variable ∈ ["rent/income", "networth/income"])
	end

	loss = @chain tmp begin
		unstack(:source, :value)
		@transform(:weights = w(:j, α = 0.0, a = 0, b = J))
		@groupby(:variable)
		@transform(@bycol :weights ./ sum(:weights) .* J)
		@groupby(:variable)
		@combine(:loss = √mean(:weights .* ((:model .- :data) ./ :data) .^ 2))
	end

	loss_nt = NamedTuple{Symbol.(Tuple(loss.variable))}(Tuple(loss.loss))
	mean_loss = mean(loss.loss)
	norm_loss = norm(loss.loss)
	
	if details
		return (; loss_nt, mean_loss, norm_loss, tmp, means_by_age)
	else
		return mean_loss
	end

end

# ╔═╡ 84a3e143-847e-47ff-9dfd-628064e848b2
md"""
# Execute transition
"""

# ╔═╡ 8499faf3-9ff9-4c75-abc3-deb89be6dea9
md"""
# Functionality: Transition
"""

# ╔═╡ 1afb5a95-ec6d-45f2-b7ac-4f92cab1a5fb


# ╔═╡ dc5e2ce1-6547-4e03-a45e-5b2c0ad0a04f
# ╠═╡ disabled = true
#=╠═╡
test_out = let
	(; par, prices, statespace) = cali_auclert
	setup = setup_transition(HousingModel(), par, statespace; scenario=:baby_boom, GE_kwargs = (; tol = 1e-4))

	transition_GE(setup, maxiter = 10, λ = 0.005)
	
end
  ╠═╡ =#

# ╔═╡ 43864981-5b6d-47b7-b995-36923c77f0ed
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

# ╔═╡ 841f0c36-9686-44a7-85b2-d655034e4d1d
function dimstack_from_nt(nt, dim)
	guessed_path = map(zip(keys(nt), values(nt))) do (name, val)
		fill(val, dim; name)
	end |> DimStack
end

# ╔═╡ d3e791a3-0995-41a2-9102-1158fd88f36e
function prepend_one(p_surv)

	cat(
		DimVector(ones(1), Dim{:j}(0:0)),
		p_surv, 
		dims = :j
	)
end

# ╔═╡ 62cf81a9-d6cc-4b83-ad00-c6485ccd0616
function change_births(p_surv, births; name = :pmf_births)
	p_surv = copy(p_surv)
	p_surv[j = At(0)] = births

	DimVector(cumprod(p_surv); name)
end

# ╔═╡ 860ce00d-66af-4e64-940b-0c73de646c75
md"""
# Helpers
"""

# ╔═╡ 6411c948-d120-4671-8eec-9327e1b2b746
md"""
### Adjusting the period ``P``

* depreciation rate ``\delta_P = 1 - (1-\delta)^P``
* discount factor ``\beta``
* ``h``
* ``J``
"""

# ╔═╡ 786dd9e4-91f9-4d1b-94e4-27522bb6fcba


# ╔═╡ 5187e754-d917-4659-a0c0-c60fead97491
function show_vector(x)
	@info join(repr.(x), ", ") |> Base.Text
end

# ╔═╡ a86b4795-0925-4651-9495-6f5c2c2cc60a


# ╔═╡ b0a30542-ea8e-498b-9691-169646ca272d
md"""
## Demographics
"""

# ╔═╡ 0cd9df59-0085-434b-9e1c-a27ac6e40f0e
function pmf(m; births = nothing)
	
	j_dim =	DimensionalData.dims(m, :j)

	_births_ = isnothing(births) ? 1.0 : births

	pmf = cumprod([_births_; (1 .- m)])[begin:end-1]
	pmf = DimVector(pmf, j_dim, name = :pmf) 
	
	if isnothing(births)
		pmf .= pmf ./ sum(pmf)
	end

	return pmf

end

# ╔═╡ 3a6bd594-5eca-41e6-9fa7-593d92f214e6
p_surv₀ = DimVector(
	[0.9945385, 0.9995935, 0.9997525, 0.999799, 0.999836, 0.9998605, 0.9998755, 0.999885, 0.99989, 0.99989, 0.999884, 0.9998745, 0.9998525, 0.9998115, 0.99975, 0.9996605, 0.999536, 0.9993885, 0.999241, 0.9991345, 0.99906, 0.998978, 0.9988925, 0.99881, 0.9987215, 0.998631, 0.9985435, 0.9984545, 0.998359, 0.998259, 0.998161, 0.9980625, 0.997962, 0.997868, 0.9977805, 0.997691, 0.9975995, 0.997493, 0.997366, 0.997226, 0.997077, 0.99692, 0.9967525, 0.9965905, 0.996419, 0.9962185, 0.995971, 0.995691, 0.9953685, 0.9950285, 0.994659, 0.9942635, 0.993815, 0.993334, 0.9927975, 0.9921995, 0.9915535, 0.9908825, 0.990155, 0.9893715, 0.988523, 0.987618, 0.9866885, 0.985767, 0.9848455, 0.983935, 0.982972, 0.9818665, 0.980645, 0.9793075, 0.9778105, 0.9760855, 0.9741015, 0.971813, 0.9691475, 0.9657655, 0.9623835000000001, 0.958681, 0.9545755, 0.9497485, 0.9445295, 0.9388595, 0.9326274999999999, 0.9255175, 0.9172435, 0.907863, 0.8973555, 0.885727, 0.873394, 0.859642, 0.844195, 0.827184, 0.8087880000000001, 0.78986, 0.7707634999999999, 0.7515835, 0.732595, 0.7140934999999999, 0.6963895, 0.6798, 0.662296, 0.6438275, 0.6243405, 0.6037785, 0.5820815, 0.559187, 0.5350275, 0.509533, 0.4826284999999999, 0.454237, 0.42427349999999997, 0.39265149999999993, 0.35927850000000006, 0.32405700000000004, 0.28970799999999997, 0.25419400000000003, 0.21690299999999996, 0.17774900000000005, 0.13663599999999998, 0.093468, 1.0],
	Dim{:j}(0:120)
)

# ╔═╡ f953815f-bbff-450a-83da-5b3584163c33
p_surv(age) = p_surv₀[j = At(age)]

# ╔═╡ 303dda81-267e-4f99-be90-0a6df21a0add
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

# ╔═╡ 4be7b686-18dc-4143-b176-024f34d3fd9a
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

# ╔═╡ 2c27773a-9fdc-4021-a26a-d5407a216342
md"""
## Income profile
"""

# ╔═╡ ce5ad54a-e77a-4faa-9897-c580e810e894
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

# ╔═╡ 0663df9f-81c6-4901-8d6d-568616b21dc7
function simple_income_profile(J, JR; y=1.0, yR = 0.0)
	y = [fill(y, JR); fill(yR, J - JR)]

	DimArray(y, Dim{:j}(0:J-1), name = :y)
end

# ╔═╡ 336795f7-8db3-4f3a-b312-4f7081e9a81b
md"""
## Statespace
"""

# ╔═╡ 24ff640f-10a1-47cd-9dd3-c41eec6bd55c
function simple_initial_distribution(statespace)
	(; dims, ε_grid, grid) = statespace

	n = length(grid)
	n_half = n ÷ 2
	
	#a₀ = a_grid[1]
	
	π₀ = zeros(dims, name = :π₀)
	π₀[state = 1:n_half] .= (1/n_half) ./ length(ε_grid)
	π₀
end

# ╔═╡ 09271053-7b1d-4cad-bc1a-efc1c3f352c5
function trivial_initial_distribution(statespace; init_state)
	(; dims, ε_grid, grid) = statespace
	
	π₀ = zeros(dims, name = :π₀)
	π₀[state = Near(init_state)] .= 1.0 ./ length(ε_grid)
	π₀
end

# ╔═╡ a38c8b5a-cd56-47de-92cd-b9dc86912ae5
function exponential_grid(amin,amax,na)
	return exp.(range(0.0, stop=log(amax-amin+1.0), length = na)) .+ amin .- 1.0
end

# ╔═╡ cc32766e-93c0-4967-929c-22f5e77fe29a
function no_income_risk()
	MarkovChain(ones(1,1), [1.0])
end

# ╔═╡ b876f4a9-424b-495e-80be-9330ad7d960d
no_permanent_states() = get_permanent_states(no_income_risk(), :θ)

# ╔═╡ 0a3d9b84-59f8-426c-a848-a1a9a645cb51
no_permanent_states()

# ╔═╡ aa0e739d-6e1d-4448-a9ff-3967891d10c8
function no_income_risk2()
	MarkovChain([0.9 0.1; 0.2 0.8], [0.9999, 1.0001])
end

# ╔═╡ 934b2b49-8d6b-43af-a1c2-38673cd4eaaa
function simple_income_risk()
	MarkovChain([0.9 0.1; 0.2 0.8], [0.85, 1.15])
end

# ╔═╡ 27e12095-6f63-4ab1-b58d-387c5865b652
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

# ╔═╡ b617ccb3-e30f-4dcc-9994-bad7014c6f0b
function log_discr_AR1(args...; normalize_mean = true, method = QuantEcon.tauchen)
	mc₀ = method(args...)
	@info mc₀.state_values
	state_values 	= exp.(mc₀.state_values)

	π∞ = QuantEcon.stationary_distributions(mc₀) |> only

	if normalize_mean
		𝔼y = mean(state_values, weights(π∞))
		state_values ./= 𝔼y
	end
	
	return MarkovChain(mc₀.p, state_values)
end

# ╔═╡ 1453a3be-eb38-4e4f-a10a-d0122b75260a
"See Calibration table and E.4 Calibration details"
function ε_chain_AMMR(σ_scale = 1.0; n = 11, n_std = 3.0)
	ρ = 0.91
	σ_y = 0.91
	σ = σ_scale * √(1-ρ^2) * σ_y
	mc = log_discr_AR1(n, ρ, σ, 0.0, n_std)
end

# ╔═╡ ee67f8fe-afa0-4b29-8d73-af93071b379e
"See Calibration table and E.4 Calibration details"
function θ_chain_AMMR(σ_scale = 1.0; n = 3, n_std = 3.0)
	ρ = 0.677
	σ_y = 0.61
	σ = σ_scale * √(1-ρ^2) * σ_y
	mc = log_discr_AR1(n, ρ, σ, 0.0, n_std)
end

# ╔═╡ 2059113e-6e49-487a-a920-2e7d17fc8257
function permanent_states_AMMR(args...; kwargs...)
	mc_permanent = θ_chain_AMMR(args...; kwargs...)
	get_permanent_states(mc, permanent)
end

# ╔═╡ 15ad0df7-a1ce-43c5-91cd-c8355a6d0361
md"""
# Tests
"""

# ╔═╡ ee0b2ae8-72e3-4ec5-be71-addd268bb720
income_profile(120, 41)

# ╔═╡ a05f69cc-1aca-4aea-adda-06f0d340e4b6
md"""
# Test: transition
"""

# ╔═╡ fab8ab09-6814-4c3d-bf89-8158937a1a43
#=╠═╡
out_auclert_trans.GE₀.aggregates.updated
  ╠═╡ =#

# ╔═╡ 4b00a4c5-9fea-4373-836f-56e6722c367f
#=╠═╡
out_auclert_trans.aggregate_paths.loss |> mean
  ╠═╡ =#

# ╔═╡ 5dba456e-052d-4d31-9bf6-a10ddb7b4191
let
	@chain out_auclert_trans.sim_df begin
		@subset(:π > 0)
		@transform(:t = :j + :born)
		@subset(:t ≤ 300)
		@groupby(:t)
		@combine(:pop = sum(:π))
		data(_) * mapping(:t, :pop) * visual(Lines)
		draw
	end
	
end

# ╔═╡ 348f4e16-2cc3-46be-a975-0e93abfff8b6
#=╠═╡
out_auclert_trans.aggregate_paths.updated |> sprint_dimstack
  ╠═╡ =#

# ╔═╡ 9b9626b2-03f4-4399-9b75-0549c91c99c9
guess_auclert_trans = DimStack(
  DimVector([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  ], Dim{:t}(0:300), name = :H_hh),
  DimVector([
    20.752612437792582, 20.752390454519098, 20.752054056518467, 20.751612439589437, 20.751071172775266, 20.75043211313686, 20.749692159724646, 20.748848123693502, 20.747900746871423, 20.74685824370707, 20.74573987742925, 20.74457806783884, 20.74341843750368, 20.742316677752022, 20.741333799357182, 20.740531294163738, 20.739967001469143, 20.739691452279942, 20.739751024073772, 20.74018845216095, 20.741043507538144, 20.742353850409835, 20.74415577312298, 20.746484587326485, 20.749375444675085, 20.752864528565365, 20.756988518338883, 20.7617739516358, 20.767248676520346, 20.773441429714467, 20.780382371883185, 20.788102883595815, 20.796636564967166, 20.806018583903633, 20.81628642821324, 20.827479933182072, 20.839637403068885, 20.85277289776522, 20.866897134403395, 20.88201636679837, 20.898131889495506, 20.915238090289613, 20.933320135744168, 20.952337992728467, 20.972245511094265, 20.99299063714711, 21.014516255826113, 21.03678643110852, 21.05976168102218, 21.08339839397012, 21.107650270446666, 21.1324685631362, 21.157801754428654, 21.183596083774603, 21.20979432655698, 21.2363394401189, 21.263123484953958, 21.290057708777937, 21.317038365388534, 21.34394952196205, 21.370656190498707, 21.397008258187633, 21.422839757331342, 21.447972903446, 21.472216006973497, 21.495365110300344, 21.517210818986246, 21.53754267176697, 21.556166135371964, 21.572904432579804, 21.58760988156418, 21.600148632709573, 21.610465264038787, 21.618579894617977, 21.62459359171178, 21.628682390933832, 21.62261469787502, 21.617236789291294, 21.61250018912104, 21.608356947574276, 21.604760357523507, 21.60166536492502, 21.599028799715217, 21.596809497506452, 21.59496835429199, 21.59346833914853, 21.59227448164059, 21.59135384398619, 21.590675485735716, 21.590210422925235, 21.589931584909696, 21.58981376978122, 21.589833598839864, 21.589969470741075, 21.590201514109957, 21.590511539140287, 21.590882987615135, 21.591300880957398, 21.591751765813413, 21.592223657408862, 21.592705980045952, 21.593189504727064, 21.593666283731178, 21.594129582645735, 21.594573809769983, 21.59499444273522, 21.59538795273138, 21.59575172813159, 21.596083997242395, 21.596383748764747, 21.596650652128346, 21.5968849763127, 21.597087512010287, 21.597259495515498, 21.597402534066266, 21.597518534060345, 21.597609632780937, 21.5976781335156, 21.59772644140489, 21.597757008445694, 21.597772283205003, 21.597774665292665, 21.59776646544213, 21.597749871283508, 21.597726918697894, 21.597699469216394, 21.597669193400097, 21.59763755990102, 21.59760582622538, 21.597575040028506, 21.597546045062288, 21.597519490420012, 21.597495843072004, 21.597475403752533, 21.59745832441093, 21.597444627101307, 21.597434224041933, 21.597426938300394, 21.59742252119053, 21.59742067238328, 21.597421059783894, 21.59742333564764, 21.59742715273886, 21.597432175655154, 21.5974380913489, 21.597444615441177, 21.597451496058902, 21.597458515925705, 21.597465492405707, 21.5974722760143, 21.597478748191623, 21.597484818566155, 21.59749042196049, 21.597495515308626, 21.597500074921577, 21.597504093435962, 21.597507577007875, 21.597510542749394, 21.59751301640889, 21.59751503028976, 21.59751662140266, 21.597517829842566, 21.597518697381325, 21.597519266265518, 21.597519578206985, 21.59751967355262, 21.5975195906212, 21.597519365191577, 21.597519030129135, 21.597518615134536, 21.597518146601793, 21.597517647570687, 21.59751713775934, 21.597516633665233, 21.597516148720537, 21.59751569349109, 21.59751527590774, 21.597514901520192, 21.597514573764364, 21.597514294235964, 21.597514062961835, 21.597513878665627, 21.59751373901984, 21.59751364088378, 21.597513580520893, 21.59751355379654, 21.597513556352265, 21.59751358375768, 21.597513631638993, 21.597513695784702, 21.5975137722303, 21.597513857321808, 21.597513947761502, 21.597514040636263, 21.597514133431215, 21.59751422403076, 21.597514310708647, 21.59751439210948, 21.597514467223334, 21.59751453535564, 21.597514596093394, 21.59751464926954, 21.59751469492716, 21.597514733283496, 21.597514764696005, 21.59751478962971, 21.597514808628087, 21.59751482228602, 21.597514831226583, 21.59751483608058, 21.59751483746985, 21.597514835993092, 21.597514832215385, 21.59751482665954, 21.597514819800764, 21.597514812063064, 21.59751480381758, 21.597514795382416, 21.597514787024163, 21.597514778960104, 21.597514771361435, 21.59751476435683, 21.597514758036912, 21.59751475245838, 21.597514747648614, 21.59751474360989, 21.597514740323756, 21.597514737755404, 21.597514735856745, 21.59751473457052, 21.597514733832927, 21.59751473357662, 21.597514733732947, 21.5975147342339, 21.597514735013824, 21.597514736010602, 21.597514737166716, 21.597514738429926, 21.59751473975377, 21.597514741097715, 21.597514742427148, 21.597514743713592, 21.597514744933875, 21.59751474607037, 21.59751474711023, 21.597514748045068, 21.597514748870264, 21.597514749584644, 21.597514750189998, 21.597514750690145, 21.59751475109114, 21.597514751400205, 21.597514751625628, 21.597514751776384, 21.597514751861716, 21.597514751890827, 21.59751475187296, 21.597514751816647, 21.59751475173023, 21.597514751621063, 21.59751475149608, 21.597514751361263, 21.597514751221812, 21.5975147510823, 21.59751475094633, 21.5975147508169, 21.597514750696348, 21.59751475058627, 21.597514750487893, 21.597514750401768, 21.59751475032816, 21.597514750266914, 21.5975147502175, 21.597514750179432, 21.597514750151692, 21.597514750133406, 21.597514750123544, 21.597514750120947, 21.597514750124585, 21.597514750133364, 21.597514750146228, 21.597514750162283, 21.597514750180594, 21.597514750200457, 21.59751475022102, 21.597514750241825, 21.597514750262278, 21.59751475028193, 21.597514750300526, 21.5975147503178, 21.59751475033345, 21.597514750347457, 21.597514750359778, 21.597514750370415, 21.597514750379318, 21.597514750386594, 21.59751475039239, 21.597514750396755, 21.59751475039985, 21.59751475040181, 21.597514750402766
  ], Dim{:t}(0:300), name = :K_supply),
  DimVector([
    2.25914781971103, 2.2591498317844376, 2.2591544395542233, 2.259162265470508, 2.2591739926807346, 2.259190371366545, 2.2592127463141445, 2.2592422010557516, 2.2592798813955906, 2.2593270537471337, 2.2593850089926946, 2.259455159014873, 2.2595389683969804, 2.2596379067938033, 2.2597535437759353, 2.25988743695139, 2.260037625871169, 2.2602052647681354, 2.260391654498964, 2.260598144015353, 2.2608262870309104, 2.2610776662914263, 2.261353962312143, 2.2616567825672127, 2.2619879873976063, 2.2623495712694797, 2.2627309495260692, 2.263132938633497, 2.2635565056859344, 2.2640027178486033, 2.264472444577783, 2.2649666783385185, 2.2654865386562264, 2.266033144507581, 2.266607710683457, 2.2672115344847588, 2.2678212488368987, 2.2684347114497765, 2.269049245230084, 2.269662235997457, 2.270270244805118, 2.270869875474666, 2.2714566778302085, 2.272026143862376, 2.2725726774955435, 2.273090819681513, 2.273601603061282, 2.2741017682113682, 2.2745879024132956, 2.275056829022154, 2.2755044212200484, 2.2759268232861225, 2.2763205101085697, 2.2766808682635147, 2.2770035751309123, 2.277284774996765, 2.277558151699457, 2.277821959473373, 2.278074945591597, 2.2783146100284637, 2.278539587830546, 2.2787471978173124, 2.2789361187391917, 2.2791038146178764, 2.279248632474914, 2.2793699197737585, 2.2794660350542095, 2.2795372717796325, 2.279583219453939, 2.27960558116321, 2.279605581163205, 2.279605581163209, 2.2796055811632097, 2.2796055811632088, 2.2796055811632048, 2.279605581163209, 2.279605581163205, 2.2796055811632074, 2.2796055811632048, 2.2796055811632097, 2.27960558116321, 2.2796055811632088, 2.279605581163209, 2.2796055811632088, 2.279605581163208, 2.279605581163208, 2.2796055811632097, 2.2796055811632097, 2.279605581163212, 2.2796055811632105, 2.2796055811632083, 2.279605581163207, 2.2796055811632074, 2.279605581163202, 2.2796055811632074, 2.2796055811632066, 2.279605581163206, 2.27960558116321, 2.2796055811632034, 2.279605581163212, 2.279605581163209, 2.2796055811632048, 2.279605581163211, 2.2796055811632105, 2.27960558116321, 2.279605581163207, 2.2796055811632128, 2.279605581163205, 2.279605581163211, 2.279605581163212, 2.2796055811632097, 2.2796055811632088, 2.2796055811632097, 2.2796055811632083, 2.279605581163205, 2.2796055811632128, 2.2796055811632114, 2.279605581163206, 2.2796055811632137, 2.2796055811632088, 2.2796055811632074, 2.2796055811632088, 2.2796055811632137, 2.2796055811632145, 2.2796055811632066, 2.279605581163207, 2.2796055811632114, 2.279605581163209, 2.2796055811632114, 2.279605581163211, 2.279605581163211, 2.279605581163206, 2.279605581163213, 2.2796055811632114, 2.279605581163204, 2.279605581163207, 2.2796055811632048, 2.279605581163206, 2.279605581163214, 2.2796055811632043, 2.279605581163211, 2.2796055811632123, 2.27960558116321, 2.2796055811632123, 2.2796055811632083, 2.279605581163214, 2.279605581163211, 2.27960558116321, 2.279605581163205, 2.2796055811632083, 2.2796055811632083, 2.2796055811632105, 2.279605581163213, 2.2796055811632154, 2.2796055811632123, 2.2796055811632088, 2.279605581163207, 2.279605581163206, 2.2796055811632097, 2.2796055811632128, 2.27960558116321, 2.279605581163207, 2.279605581163207, 2.2796055811632048, 2.2796055811632074, 2.2796055811632066, 2.2796055811632123, 2.2796055811632097, 2.279605581163212, 2.2796055811632074, 2.2796055811632105, 2.279605581163208, 2.2796055811632088, 2.2796055811632088, 2.2796055811632066, 2.279605581163209, 2.27960558116321, 2.279605581163207, 2.279605581163211, 2.2796055811632097, 2.2796055811632105, 2.2796055811632083, 2.279605581163203, 2.279605581163209, 2.2796055811632083, 2.279605581163206, 2.2796055811632097, 2.279605581163204, 2.2796055811632137, 2.279605581163204, 2.279605581163212, 2.279605581163211, 2.279605581163211, 2.2796055811632097, 2.2796055811632083, 2.2796055811632114, 2.2796055811632097, 2.2796055811632128, 2.2796055811632074, 2.279605581163216, 2.2796055811632105, 2.2796055811632128, 2.279605581163207, 2.2796055811632123, 2.2796055811632145, 2.2796055811632114, 2.2796055811632074, 2.279605581163208, 2.2796055811632088, 2.2796055811632066, 2.2796055811632043, 2.2796055811632057, 2.279605581163208, 2.279605581163208, 2.2796055811632114, 2.2796055811632066, 2.2796055811632074, 2.2796055811632083, 2.2796055811632097, 2.279605581163211, 2.279605581163215, 2.279605581163209, 2.2796055811632066, 2.279605581163207, 2.2796055811632114, 2.2796055811632097, 2.279605581163211, 2.2796055811632074, 2.2796055811632066, 2.279605581163212, 2.2796055811632123, 2.279605581163209, 2.2796055811632097, 2.2796055811632097, 2.2796055811632105, 2.2796055811632123, 2.27960558116321, 2.2796055811632066, 2.27960558116321, 2.2796055811632043, 2.27960558116321, 2.2796055811632066, 2.279605581163211, 2.279605581163209, 2.2796055811632034, 2.2796055811632114, 2.2796055811632123, 2.2796055811632097, 2.279605581163207, 2.279605581163208, 2.279605581163209, 2.279605581163215, 2.27960558116321, 2.2796055811632074, 2.279605581163208, 2.2796055811632057, 2.2796055811632114, 2.2796055811632097, 2.2796055811632123, 2.279605581163208, 2.2796055811632074, 2.27960558116321, 2.27960558116321, 2.2796055811632074, 2.2796055811632074, 2.2796055811632105, 2.2796055811632066, 2.2796055811632097, 2.279605581163208, 2.279605581163208, 2.2796055811632083, 2.2796055811632097, 2.279605581163209, 2.27960558116321, 2.279605581163209, 2.279605581163207, 2.2796055811632088, 2.2796055811632074, 2.279605581163211, 2.2796055811632114, 2.2796055811632154, 2.279605581163212, 2.2796055811632114, 2.279605581163209, 2.27960558116321, 2.2796055811632123, 2.2796055811632105, 2.279605581163212, 2.27960558116321, 2.2796055811632105, 2.279605581163216, 2.2796055811632128, 2.279605581163208, 2.279605581163213, 2.2796055811632105, 2.2796055811632105, 2.279605581163208, 2.279605581163209, 2.279605581163206, 2.2796055811632097, 2.2796055811632043
  ], Dim{:t}(0:300), name = :L_eff),
)

# ╔═╡ 37201f37-e890-409a-bd72-008a71d3b518


# ╔═╡ 9d1092f7-cd5a-4efa-bbb4-004e78e1ff1a
#=╠═╡
let 
	path = out_auclert_trans2.aggregate_paths.updated.H_hh |> lines

	hlines!([2.77141, 2.79926])

	current_figure()
#	path = out_auclert_trans2.price_paths.r
#	pc = path[end] / path[1] - 1
#	pp = path[end] - path[1]
#	(; pc, pp)
end
  ╠═╡ =#

# ╔═╡ 93b12ad3-6107-403f-a721-b0ff78b0e872
#=╠═╡
sprint_dimstack(out_auclert_trans2.aggregate_paths.guessed_paths)
  ╠═╡ =#

# ╔═╡ 1ee457b2-02ea-4965-978a-967530047c3c
current_paths_auclert_trans2 = DimStack(
  DimVector([
    2.775364973676277, 2.7795535853806035, 2.783855876364276, 2.788163996389052, 2.79238051169363, 2.7964205130949433, 2.8002131413104934, 2.8037025733033354, 2.806848477597795, 2.8096259400082237, 2.8120248736511684, 2.8140489402854403, 2.8157140352221752, 2.817046410831028, 2.8180805281697587, 2.818856744911806, 2.819418954601275, 2.81981229403057, 2.8200810281895947, 2.820266709643315, 2.8204066883374943, 2.8205330207135844, 2.8206717958537597, 2.820842867616924, 2.821059956730439, 2.821331066735766, 2.821659140732182, 2.8220428816280108, 2.8224776569796517, 2.822956417723045, 2.8234705714526602, 2.824010765245802, 2.8245675481849024, 2.8251318979921005, 2.825695608498557, 2.826251544987737, 2.8267937818924644, 2.8273176428786315, 2.8278196631249193, 2.8282974948309465, 2.8287497753056763, 2.8291759745329563, 2.8295762360175036, 2.829951221577193, 2.8303019677859216, 2.8306297591315164, 2.8309360205371252, 2.8312222295592235, 2.8314898484634115, 2.8317402746312355, 2.831974807202582, 2.832194627572218, 2.8324007912497295, 2.832594228570791, 2.832775751953942, 2.8329460675783076, 2.8331057896520595, 2.833255455412927, 2.8333955402581052, 2.8335264721032574, 2.8336486443375026, 2.833762426941352, 2.8338681755774107, 2.8339662385812194, 2.8340569620207448, 2.834140693127658, 2.8342177823513, 2.8342885841870715, 2.8343534567883673, 2.8344127604003972, 2.8344668547813097, 2.8345160959466735, 2.8345608325950304, 2.83460140266546, 2.8346381303726376, 2.83467132395313, 2.834701274211148, 2.8347282522213857, 2.83475250724245, 2.8347742684829216, 2.8347937469743782, 2.8348111374355405, 2.834826620037807, 2.834840361993999, 2.8348525189194382, 2.8348632359305483, 2.8348726484794433, 2.8348808829308227, 2.834888056911916, 2.834894279470343, 2.8348996510796445, 2.8349042635382524, 2.8349081997910344, 2.834911533711187, 2.834914329856386, 2.834916643212201, 2.8349185189238355, 2.834919992008316, 2.834921087032528, 2.834921817740075, 2.8349221866025034
  ], Dim{:t}(0:100), name = :H_hh),
  DimVector([
    19.817263279317295, 19.89243648120649, 19.960499733593334, 20.0219888829081, 20.07746957784781, 20.127504679294375, 20.17263437289818, 20.213362791954005, 20.25015251141478, 20.28342249120812, 20.31354944085027, 20.34087152342101, 20.365688723111123, 20.388265573725516, 20.40883431408296, 20.42759791558613, 20.444734199969247, 20.460399725521185, 20.474732737412317, 20.487855625167544, 20.49987605188713, 20.510890040054377, 20.520983831732725, 20.530236122179705, 20.53871685943229, 20.54648952468274, 20.553612096369427, 20.56013775669674, 20.566115512387704, 20.571590636237538, 20.576604927808294, 20.581196801475997, 20.58540131420626, 20.589250552291876, 20.59277391586989, 20.595998209626458, 20.598947850599302, 20.601645006319316, 20.604109880457635, 20.606360920261274, 20.608415076396017, 20.61028798541708, 20.61199422892681, 20.61354742260043, 20.61496034347036, 20.61624496783058, 20.61741255087198, 20.618473549877425, 20.61943765396898, 20.620313779707676, 20.621110105231434, 20.621834007187253, 20.622492113886395, 20.62309031152814, 20.623633812871095, 20.624127211828554, 20.62457454348781, 20.624979295336928, 20.625344407687358, 20.625672356133126, 20.625965239600752, 20.6262248569383, 20.626452792456195, 20.626650491918113, 20.62681921304317, 20.626960079022894, 20.627074165859668, 20.627162730154435, 20.62722741905603, 20.62727037047604, 20.627294183452385, 20.627301720443523, 20.6272959741904, 20.627279850531963, 20.627256015726324, 20.62722679453131, 20.627194163874925, 20.627162327195293, 20.627131598779073, 20.627102234715768, 20.627074438789958, 20.62704836775078, 20.627024136346076, 20.627001821882974, 20.626981468413284, 20.62696309072525, 20.626946678087247, 20.626932197874613, 20.62691959889616, 20.62690881454883, 20.62689976579898, 20.626892363978442, 20.62688651341458, 20.62688211388435, 20.626879062913083, 20.626877257926665, 20.62687659826795, 20.626876987089318, 20.626878333129905, 20.626880552367552, 20.626883569481734
  ], Dim{:t}(0:100), name = :K_supply),
  DimVector([
    2.2591478197110315, 2.260642215054237, 2.2620314018975476, 2.2633236758226807, 2.2645265585713927, 2.265646876539988, 2.2666908264644805, 2.26766396709186, 2.2685713603411144, 2.269417629382801, 2.2702070042728772, 2.270943358772974, 2.271630236791407, 2.272270877718734, 2.272868238556665, 2.273425019636961, 2.273943692293883, 2.2744266441982672, 2.2748760604309717, 2.2752939504107013, 2.275682166073767, 2.2760424131762913, 2.276376264935033, 2.2766851800415955, 2.276970517860581, 2.277233546064231, 2.2774754516995737, 2.2776975559649095, 2.277901095516951, 2.2780872289620056, 2.278257040160696, 2.2784115486428833, 2.2785517147450323, 2.278678448111909, 2.2787926125164835, 2.278895027705402, 2.2789864719751467, 2.2790678775011104, 2.2791401205156276, 2.2792040256696455, 2.279260366614897, 2.2793098731225214, 2.279353231054132, 2.279391086240666, 2.279424042961283, 2.2794526678978846, 2.2794774875487267, 2.2794988889445498, 2.279517235772409, 2.279532868989994, 2.2795461064080453, 2.279557246209922, 2.2795665662127993, 2.279574322816276, 2.279580754106802, 2.2795860793385834, 2.279590497999569, 2.2795941198515486, 2.279597047651821, 2.2795993761257725, 2.2796011939803407, 2.2796025817629486, 2.279603613695142, 2.2796043559182735, 2.2796048682349768, 2.279605203372211, 2.2796054065351683, 2.279605516870296, 2.2796055663512, 2.2796055811632105, 2.279605581163208, 2.279605581163206, 2.2796055811632137, 2.2796055811632145, 2.2796055811632043, 2.2796055811632088, 2.27960558116321, 2.2796055811632057, 2.2796055811632097, 2.2796055811632066, 2.279605581163205, 2.279605581163208, 2.279605581163207, 2.279605581163212, 2.279605581163212, 2.2796055811632043, 2.279605581163206, 2.279605581163204, 2.279605581163208, 2.2796055811632074, 2.279605581163214, 2.2796055811632123, 2.2796055811632105, 2.2796055811632088, 2.2796055811632066, 2.2796055811632114, 2.2796055811632088, 2.2796055811632123, 2.2796055811632097, 2.279605581163209, 2.2796055811632066
  ], Dim{:t}(0:100), name = :L_eff),
)

# ╔═╡ f5a2d497-ffdf-4817-b089-286af625d827
current_paths_auclert_trans2#[t = At(0:100)]

# ╔═╡ 2ffc1a78-1c5d-4f31-8bda-6c7ebad94961
# ╠═╡ disabled = true
#=╠═╡
out_auclert_trans = let
	#(; par, statespace, π_init) = get_cali_test()
	(; par, statespace, prices, π_init) = cali_auclert

	model = HousingModel()

	guesses = (; K_supply = 20.75270574911025, H_hh = 0.0, L_eff = 2.2591478197110324)
	prices = let
		r = interest_rate(guesses.K_supply, guesses.L_eff, par) 
		w = wage(guesses.K_supply, guesses.L_eff, par)
		p = house_price(par.δ * guesses.H_hh, par)
		(; r, w, p)
	end

	GE₀ = stationary_GE(model, par, statespace#=, guesses, prices=#; guesses, π_init) 
	
	T̃ = 300

	#####################################
	## TEST 2: REDUCE MORTALITY BY 10% ##
	
	guess = GE₀.aggregates.updated
	guessed_path = dimstack_from_nt(guess, Dim{:t}(0:T̃))

	guessed_path = guess_auclert_trans
	
	demographics = let
		m₀ = par.m
		m₁ = 0.9 * par.m
		
		j_dim = DD.dims(m₀, :j)
		J = maximum(j_dim)
	
		borns = -J:1:T̃
		born_dim = Dim{:born}(borns)
		ms = cat([born < 0 ? m₀ : m₁ for born ∈ born_dim]..., dims = born_dim)

		
		demo = DimStack(ms, )
	end
	###########################
	
	inheritances = no_inheritances(par, statespace)

#	price_paths = get_price_paths(model, paths_in, par; GE₀)
#	(; π_permanent, state_dim, perm_dim) = statespace
	
	out = transition_GE(model, T̃, par, statespace, demographics, GE₀, guessed_path;
						normalize_population = false, inheritances,
						details = 1, λ = 0.15
						)

	#	=#
end
  ╠═╡ =#

# ╔═╡ 608dc256-3db7-4dfc-94e2-4c1f34ff7a86
#=╠═╡
out_auclert_trans0.price_paths.r |> lines
  ╠═╡ =#

# ╔═╡ 6c20f2dc-f919-4b55-9cf1-b183f53ae269
#=╠═╡
out_auclert_trans.aggregate_paths.updated.L_eff |> lines
  ╠═╡ =#

# ╔═╡ d6fb71b2-7481-4405-affd-7a8fc0ed7cd8
#=╠═╡
let
	lines(out_auclert_trans.aggregate_paths.updated.K_supply)
	scatter!(-1,out_auclert_trans.GE₀.aggregates.updated.K_supply)

	current_figure()
end
  ╠═╡ =#

# ╔═╡ 94f097a4-beb5-4ce5-bdc9-19c7369333a7
#=╠═╡
let
	
	out = out_auclert_trans0
	layer = @chain out.GE₀.sim_df begin
		@transform(:t = 0 + :j)
		@groupby(:j, :t)
		@combine(:ho = mean(:ho, weights(:π)))
		data(_) * mapping(:j, :ho => :c, color = direct("SS")) * visual(Lines)
	end

#	@info testxxx.out.GE₀.prices

#	@info testxxx.out.price_paths.r
	
	
	fig = @chain out.sim_df begin
		#@subset(:born ∈ 0:10)
		@transform(:t = :born + :j)
		@groupby(:j, :t, :born)
		@combine(:ho = mean(:ho, weights(:π)))
		data(_) * mapping(:j, :ho, group = :born, color = direct("trans")) * visual(Lines) + layer
		draw
	end # =#


end
  ╠═╡ =#

# ╔═╡ b1426980-0fe7-4d3a-9ed2-fb39e477c2f1
#=╠═╡
out_auclert_trans0.price_paths.p |> unique |> only
  ╠═╡ =#

# ╔═╡ 088841e8-8112-49ff-b056-ce8d5ebae311
#=╠═╡
out_auclert_trans0.GE₀.prices
  ╠═╡ =#

# ╔═╡ 151df92a-cdff-4286-8aab-ed5fa5dd429a
function sprint_dimstack(ds)
	string = ""
	for (key, val) ∈ pairs(ds)
		string *= "  DimVector([\n    " * join(repr.(val), ", ") * "\n  ], Dim{:t}(0:$(length(val)-1)), name = :$key),\n"
	end
	"DimStack(\n" * string * ")" |> Base.Text
end

# ╔═╡ 8bc72a0f-77c1-468a-aff4-d5ba0e80a27a
#=╠═╡
sprint_dimstack(testxxx.out.updated)
  ╠═╡ =#

# ╔═╡ a163eb1d-07aa-43ea-b067-d6e5c4db2131
@chain testxxx.out.aggregate_paths.updated begin
	DataFrame
	stack(Not(:t))
	data(_) * mapping(:t, :value, layout = :variable) * visual(Lines)
	draw(facet = (; linkyaxes = false))
end

# ╔═╡ e1601860-602e-4218-a51c-c905bcc8e228
@chain testxxx.aggregate_paths.loss begin
	DataFrame
	stack(Not(:t))
	data(_) * mapping(:t, :value, layout = :variable) * visual(Lines)
	draw(facet = (; linkyaxes = false))
end

# ╔═╡ a044470b-02db-4fa5-bbfc-510422e989cd
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

# ╔═╡ 22786293-06b8-4701-bbf6-a4a7be0f5a13
current_paths = DimStack(
  DimVector([
    9.09361089173105e-6, 9.233481654425982e-6, 9.339672499249906e-6, 9.419005700933099e-6, 9.476977115382687e-6, 9.518030547525087e-6, 9.545766958391477e-6, 9.563107068787572e-6, 9.572419120419783e-6, 9.575621087712577e-6, 9.574262330499209e-6, 9.56959051804587e-6, 9.562605301531214e-6, 9.55410336551482e-6, 9.544714279078068e-6, 9.534931367693185e-6, 9.525135860895143e-6, 9.515618244095138e-6, 9.50659458472134e-6, 9.498221325961633e-6, 9.490606335365531e-6, 9.483819109275139e-6, 9.477898238885428e-6, 9.472858383332577e-6, 9.46869529199171e-6, 9.465390506039118e-6, 9.462914707131955e-6, 9.461230833054527e-6, 9.460296277736486e-6, 9.46006491056321e-6, 9.460488488702268e-6, 9.461517927309725e-6, 9.46310417425062e-6, 9.46519897506726e-6, 9.46775538409845e-6, 9.470728193630164e-6, 9.474074202160793e-6, 9.477752425153104e-6, 9.481724205722903e-6, 9.485953284015497e-6, 9.49040580459133e-6, 9.49505029655083e-6, 9.49985757682272e-6, 9.50480064767528e-6, 9.509854583165251e-6, 9.514996415005173e-6, 9.520205015006477e-6, 9.52546097986196e-6, 9.530746516823928e-6, 9.536045333401861e-6, 9.541342530339203e-6, 9.546624499499969e-6, 9.551878826243236e-6, 9.557094197041757e-6, 9.56226031203685e-6, 9.567367802803498e-6, 9.572408155045099e-6, 9.577373636241987e-6, 9.582257227985794e-6, 9.58705256291032e-6, 9.591753865969055e-6, 9.596355899936784e-6, 9.600853903123362e-6, 9.605243591092713e-6, 9.609521155133352e-6, 9.613683257729874e-6, 9.617727025245915e-6, 9.621650038122098e-6, 9.625450318825628e-6, 9.629126317857631e-6, 9.632676898044813e-6, 9.636101317373395e-6, 9.639399210581224e-6, 9.642570569695498e-6, 9.645615723696201e-6, 9.648535317446215e-6, 9.65133029000337e-6, 9.654001852425002e-6, 9.656551465122643e-6, 9.658980814838113e-6, 9.661291791263948e-6, 9.663486463339277e-6, 9.66556705523052e-6, 9.667535921989289e-6, 9.66939552490708e-6, 9.671148406540801e-6, 9.672797165448349e-6, 9.674344430636006e-6, 9.675792835785615e-6, 9.677144993352213e-6, 9.678403468663526e-6, 9.679570754257604e-6, 9.680649244749144e-6, 9.681641212666476e-6, 9.682548785848309e-6, 9.683373891973348e-6, 9.684118177973718e-6, 9.684784146514816e-6, 9.685375183408127e-6, 9.685895470799914e-6, 9.686349937819495e-6, 9.68674424474635e-6, 9.687086807714602e-6, 9.687384886559705e-6, 9.687644737668053e-6, 9.687871746891127e-6, 9.688070545177933e-6, 9.688245109229921e-6, 9.688398849152547e-6, 9.688534684792933e-6, 9.688655112242673e-6, 9.688762261763664e-6, 9.688857948237452e-6, 9.688943715091158e-6, 9.689020872522953e-6, 9.68909053075215e-6, 9.689153628908814e-6, 9.689210960125208e-6, 9.689263193289902e-6, 9.689310891892154e-6, 9.689354530318209e-6, 9.68939450791946e-6, 9.689431161129418e-6, 9.689464773886085e-6, 9.689495586559444e-6, 9.68952380358933e-6, 9.689549599992668e-6, 9.689573126880781e-6, 9.689594516135794e-6, 9.689613884329792e-6, 9.689631336014763e-6, 9.689646966450998e-6, 9.689660863858056e-6, 9.68967311125857e-6, 9.689683787963087e-6, 9.689692970759524e-6, 9.689700734843196e-6, 9.689707154530135e-6, 9.689712303781188e-6, 9.689716256571159e-6, 9.689719087111896e-6, 9.689720869966443e-6, 9.689721680056154e-6, 9.689721592593598e-6, 9.689720682937726e-6, 9.689719026397902e-6, 9.689716697992008e-6, 9.68971377216795e-6, 9.689710322497543e-6, 9.689706421355485e-6, 9.689702139583634e-6, 9.68969754615152e-6, 9.689692707820143e-6, 9.689687688809252e-6, 9.689682550482175e-6, 9.689677351042653e-6, 9.68967214525909e-6, 9.689666984214634e-6, 9.689661915087303e-6, 9.689656980968424e-6, 9.689652220716412e-6, 9.689647668847515e-6, 9.689643355474038e-6, 9.689639306273808e-6, 9.689635542511997e-6, 9.689632081094655e-6, 9.689628934667646e-6, 9.689626111748534e-6, 9.689623616894144e-6, 9.689621450900014e-6, 9.68961961102538e-6, 9.6896180912474e-6, 9.689616882531632e-6, 9.689615973125208e-6, 9.689615348857686e-6, 9.689614993459315e-6, 9.689614888875122e-6, 9.68961501559417e-6, 9.689615352961245e-6, 9.689615879497822e-6, 9.689616573203228e-6, 9.689617411850905e-6, 9.689618373261123e-6, 9.689619435568948e-6, 9.689620577457022e-6, 9.689621778380534e-6, 9.689623018764525e-6, 9.689624280181043e-6, 9.689625545506255e-6, 9.689626799062758e-6, 9.689628026745063e-6, 9.689629216140157e-6, 9.689630356633041e-6, 9.689631439464694e-6, 9.689632457750966e-6, 9.689633406423228e-6, 9.689634282092723e-6, 9.689635082773988e-6, 9.689635807655263e-6, 9.689636456902909e-6, 9.689637031502616e-6, 9.689637533116563e-6, 9.689637963972087e-6, 9.689638326755518e-6, 9.689638624530177e-6, 9.689638860657422e-6, 9.689639038734362e-6, 9.689639162532413e-6, 9.68963923594957e-6, 9.689639262961466e-6, 9.689639247584938e-6, 9.689639193836988e-6, 9.689639105708874e-6, 9.68963898712853e-6, 9.689638841945239e-6, 9.689638673896387e-6, 9.689638486596454e-6, 9.68963828351212e-6, 9.689638067951992e-6, 9.689637843051747e-6, 9.689637611761522e-6, 9.689637376842582e-6, 9.689637140852288e-6, 9.689636906148406e-6, 9.68963667487734e-6, 9.689636448978638e-6, 9.689636230180822e-6, 9.689636020006215e-6, 9.689635819769235e-6, 9.689635630585198e-6, 9.689635453368443e-6, 9.689635288846397e-6, 9.68963513755699e-6, 9.689634999862127e-6, 9.689634875951615e-6, 9.689634765851349e-6, 9.689634669430646e-6, 9.689634586410514e-6, 9.689634516366886e-6, 9.689634458741475e-6, 9.689634412839626e-6, 9.689634377842215e-6, 9.689634352795945e-6, 9.68963433662324e-6, 9.689634328107442e-6, 9.689634325892248e-6, 9.689634328463388e-6, 9.68963433413259e-6, 9.689634341010419e-6, 9.689634346971264e-6, 9.689634349608943e-6
  ], Dim{:t}(0:250), name = :H_hh),
  DimVector([
    8.00353259114178, 7.85495557753502, 7.72226497330628, 7.6037068139908826, 7.497861567247896, 7.403460897137982, 7.3193702497535025, 7.244573621808627, 7.178160226422777, 7.119311742420036, 7.067291679262171, 7.021436485435439, 6.9811477154793, 6.945885120079439, 6.915160541749817, 6.888532514318727, 6.8656014780131525, 6.84600553367804, 6.829416669812795, 6.8155374048668635, 6.804097794799617, 6.794852762435577, 6.787579710781743, 6.782076387337905, 6.778158970635132, 6.7756603538712685, 6.774428603658213, 6.774325574619592, 6.7752256629424865, 6.777014684039511, 6.779588861265608, 6.782853914191838, 6.7867242362998255, 6.791122153153192, 6.795977253148658, 6.8012257838716685, 6.806810123294197, 6.812678291121295, 6.818783500186933, 6.825083753397594, 6.831541483850738, 6.83812323548504, 6.844798623436774, 6.851540036847328, 6.858322387299849, 6.865122879561499, 6.871920801977129, 6.878697334422454, 6.885435372167776, 6.892119364255603, 6.898735165088735, 6.905269897959823, 6.911711829300088, 6.918050252497302, 6.924275380211895, 6.930378244191852, 6.93635060164841, 6.942184847304, 6.947873930269097, 6.953411274939082, 6.9587907051283695, 6.964006370676163, 6.969053368259413, 6.973927712671353, 6.978626303860103, 6.983146890568395, 6.98748803130852, 6.99164905327475, 6.995630009677117, 6.9994316358705735, 7.003055304556562, 7.006502980270976, 7.009777173270322, 7.0128808928522695, 7.015817600075157, 7.018591159774054, 7.02120579170687, 7.023666020600965, 7.025976624809386, 7.028142583223413, 7.030169020024725, 7.0320611467944785, 7.033824201427468, 7.035463383225166, 7.03698378346195, 7.038390310631675, 7.039687609486759, 7.040879972876791, 7.041971245277225, 7.042964716769231, 7.043863006087088, 7.0446679311878215, 7.045380365616448, 7.04600007873592, 7.046525557662156, 7.046955861514684, 7.047302520147521, 7.047578620544449, 7.0477974943281225, 7.0479729183580035, 7.048119321134743, 7.048251997400841, 7.048372593291147, 7.048483054724969, 7.048584986414258, 7.048679700727482, 7.048768260844239, 7.048851518755369, 7.048930148620616, 7.0490046759546034, 7.049075503074092, 7.049142931201221, 7.049207179584321, 7.049268401964437, 7.049326700686241, 7.049382138723474, 7.0494347498636065, 7.04948454727172, 7.049531530632805, 7.049575692050279, 7.0496170208606355, 7.049655507507855, 7.049691146604325, 7.04972393929292, 7.049753895010664, 7.049781032744216, 7.049805381856027, 7.049826982551914, 7.049845886051053, 7.049862154512754, 7.049875860766719, 7.049887087887439, 7.049895928646927, 7.049902484875309, 7.04990686675336, 7.049909192057357, 7.049909585371294, 7.04990817723008, 7.049905103208587, 7.049900502970239, 7.049894519286798, 7.049887297040201, 7.0498789822158505, 7.049869720895934, 7.049859658261378, 7.049848937609175, 7.0498376993933025, 7.049826080295473, 7.049814212333246, 7.049802222012578, 7.0497902295320385, 7.049778348046929, 7.0497666829999766, 7.0497553315288615, 7.049744381959214, 7.0497339133933785, 7.049723995406167, 7.049714687845527, 7.049706040736999, 7.049698094291742, 7.049690879014601, 7.0496844159115675, 7.049678716793978, 7.049673784678178, 7.049669614258443, 7.049666192446284, 7.049663498966843, 7.049661507004663, 7.049660183889144, 7.049659491811353, 7.049659388562359, 7.049659828284183, 7.049660762223847, 7.0496621394817165, 7.04966390774451, 7.0496660139952585, 7.049668405191412, 7.049671028903967, 7.049673833910908, 7.0496767707395875, 7.0496797921535785, 7.049682853581944, 7.049685913489874, 7.049688933693062, 7.049691879620273, 7.049694720532313, 7.049697429709887, 7.049699984627039, 7.049702367132896, 7.0497045636708, 7.049706565529488, 7.04970836887408, 7.0497099744973735, 7.049711387251727, 7.049712615115739, 7.049713667841996, 7.049714555123371, 7.049715286616959, 7.049715871954596, 7.049716320742322, 7.049716642550714, 7.049716846898158, 7.049716943228321, 7.049716940883604, 7.049716849075569, 7.049716676853716, 7.049716433073401, 7.049716126363748, 7.0497157650963205, 7.04971535735529, 7.049714910908992, 7.0497144331844455, 7.049713931243638, 7.049713411763252, 7.049712881017025, 7.049712344861385, 7.049711808724097, 7.049711277596302, 7.049710756027583, 7.049710248124293, 7.049709757550808, 7.049709287534034, 7.0497088408703314, 7.049708419935493, 7.049708026697096, 7.049707662729158, 7.049707329228991, 7.049707027036156, 7.049706756652889, 7.0497065182664524, 7.049706311772588, 7.049706136800179, 7.049705992736997, 7.049705878755929, 7.049705793841959, 7.049705736819384, 7.049705706379094, 7.049705701106008, 7.049705719506152, 7.049705760033359, 7.049705821115667, 7.049705901180866, 7.0497059986814685, 7.049706112118922, 7.049706240066834, 7.049706381193498, 7.049706534283438, 7.049706698258223, 7.049706872196328, 7.049707055352616, 7.049707247176902
  ], Dim{:t}(0:250), name = :K_supply),
  DimVector([
    1.5351838287304242, 1.5390260228203858, 1.5428316306904606, 1.5465987393628284, 1.550325572202254, 1.554010484857343, 1.5576519612908126, 1.5612486098969747, 1.564799159704863, 1.5683024566654495, 1.5717574600212998, 1.5751632387572465, 1.5785189681304208, 1.5818239262782723, 1.5850774909030696, 1.5882791360314616, 1.5914284288476004, 1.59452502659859, 1.5975686735707486, 1.6005591981354135, 1.6034965098629614, 1.606380596703693, 1.6092115222344028, 1.611989422969274, 1.6147145057339485, 1.6173870451014731, 1.620007380889011, 1.6225759157141226, 1.6250931126094075, 1.6275594926944512, 1.6299756329039492, 1.6323421637708393, 1.634659767263467, 1.636929174675688, 1.6391511645688408, 1.6413265607646004, 1.6434562303877047, 1.6455410819575602, 1.6475820635277845, 1.6495801608726621, 1.6515363957196891, 1.6534518240271845, 1.6553268000777814, 1.6571616744751252, 1.6589567941803278, 1.6607125025480365, 1.6624291393621542, 1.6641070408711969, 1.6657465398233262, 1.6673479655009629, 1.668911643755114, 1.6704378970393452, 1.6719270444433774, 1.673379401726407, 1.674795281350027, 1.6761749925108673, 1.6775188411728856, 1.6788271300993287, 1.6801001588843842, 1.6813382239845334, 1.6825416187495283, 1.6837106334531542, 1.6848455553235882, 1.685946668573488, 1.6870142544298194, 1.688048591163333, 1.689049954117737, 1.6900186157386574, 1.6909548456021921, 1.6918589104432784, 1.6927310741837374, 1.6935715979600068, 1.694380740150643, 1.6951587564035409, 1.6959058996628387, 1.6966224201956213, 1.6973085656182703, 1.6979645809226458, 1.6985907085019132, 1.6991871881761809, 1.6997542572178423, 1.700292150376676, 1.7008010999046967, 1.7012813355807392, 1.7017330847348417, 1.7021565722722998, 1.70255202069756, 1.7029196501378778, 1.7032596783666347, 1.7035723208265237, 1.7038577906525005, 1.7041162986944123, 1.704348053539513, 1.7045532615346561, 1.7047321268083446, 1.704884851292495, 1.7050116347440225, 1.7051126747662018, 1.7051881668297866, 1.705238304293969, 1.7052632784270676, 1.705263278427065, 1.7052632784270703, 1.7052632784270685, 1.7052632784270632, 1.7052632784270654, 1.7052632784270672, 1.7052632784270685, 1.7052632784270683, 1.7052632784270667, 1.705263278427066, 1.7052632784270698, 1.7052632784270643, 1.7052632784270665, 1.7052632784270676, 1.7052632784270674, 1.7052632784270623, 1.7052632784270656, 1.7052632784270676, 1.7052632784270674, 1.705263278427069, 1.7052632784270716, 1.705263278427071, 1.705263278427067, 1.705263278427066, 1.7052632784270667, 1.7052632784270694, 1.705263278427061, 1.705263278427069, 1.7052632784270676, 1.7052632784270667, 1.7052632784270663, 1.705263278427062, 1.7052632784270692, 1.7052632784270654, 1.7052632784270647, 1.7052632784270711, 1.7052632784270658, 1.7052632784270672, 1.7052632784270632, 1.7052632784270632, 1.7052632784270703, 1.7052632784270625, 1.7052632784270652, 1.7052632784270672, 1.7052632784270632, 1.7052632784270674, 1.7052632784270705, 1.7052632784270698, 1.7052632784270676, 1.7052632784270665, 1.7052632784270645, 1.7052632784270672, 1.705263278427068, 1.7052632784270663, 1.7052632784270674, 1.7052632784270654, 1.7052632784270694, 1.7052632784270658, 1.7052632784270685, 1.705263278427065, 1.7052632784270652, 1.705263278427066, 1.705263278427063, 1.7052632784270654, 1.705263278427065, 1.705263278427067, 1.7052632784270643, 1.7052632784270645, 1.7052632784270698, 1.7052632784270647, 1.7052632784270718, 1.7052632784270674, 1.7052632784270665, 1.7052632784270632, 1.7052632784270674, 1.705263278427064, 1.7052632784270683, 1.7052632784270645, 1.7052632784270638, 1.7052632784270676, 1.7052632784270676, 1.7052632784270672, 1.7052632784270663, 1.7052632784270634, 1.7052632784270667, 1.7052632784270618, 1.7052632784270678, 1.7052632784270616, 1.7052632784270623, 1.705263278427066, 1.7052632784270623, 1.7052632784270658, 1.7052632784270672, 1.7052632784270678, 1.7052632784270645, 1.7052632784270647, 1.7052632784270698, 1.7052632784270614, 1.7052632784270647, 1.7052632784270694, 1.7052632784270618, 1.7052632784270683, 1.7052632784270614, 1.7052632784270618, 1.7052632784270614, 1.7052632784270667, 1.7052632784270647, 1.7052632784270652, 1.7052632784270625, 1.7052632784270647, 1.7052632784270636, 1.7052632784270658, 1.7052632784270618, 1.705263278427061, 1.7052632784270643, 1.7052632784270627, 1.7052632784270647, 1.7052632784270667, 1.7052632784270627, 1.7052632784270665, 1.7052632784270683, 1.7052632784270632, 1.7052632784270652, 1.705263278427067, 1.7052632784270674, 1.7052632784270692, 1.7052632784270676, 1.705263278427067, 1.705263278427066, 1.7052632784270696, 1.7052632784270654, 1.7052632784270643, 1.7052632784270663, 1.7052632784270652, 1.7052632784270656, 1.7052632784270643, 1.7052632784270696, 1.705263278427068, 1.7052632784270623, 1.7052632784270643, 1.705263278427065, 1.7052632784270643, 1.7052632784270683, 1.7052632784270614, 1.7052632784270674, 1.7052632784270658, 1.7052632784270645, 1.705263278427067, 1.7052632784270665, 1.7052632784270676
  ], Dim{:t}(0:250), name = :L_eff),
)

# ╔═╡ 842119b1-0044-4034-a2c5-118c351aa8b8
let
	layer = @chain testxxx.out.GE₀.sim_df begin
		@transform(:t = 0 + :j)
		@groupby(:j, :t)
		@combine(:state = mean(:state, weights(:π)))
		data(_) * mapping(:j, :state, color = direct("SS")) * visual(Lines)
	end

#	@info testxxx.out.GE₀.prices

#	@info testxxx.out.price_paths.r
	
	
	fig = @chain testxxx.out.sim_df begin
		#@subset(:born ∈ 0:10)
		@transform(:t = :born + :j)
		@groupby(:j, :t, :born)
		@combine(:state = mean(:state, weights(:π)))
		data(_) * mapping(:j, :state, group = :born, color = direct("trans")) * visual(Lines) + layer
		draw
	end


end

# ╔═╡ 4fbca4ab-9e43-4b5d-94b0-e1765ab11d82
#=╠═╡
let
	
	@info testxxx.out.GE₀.aggregates.aggregates

	testxxx.out.aggregate_paths
end
  ╠═╡ =#

# ╔═╡ de8617b6-ff90-4de3-b332-beaeb5a5921d
# ╠═╡ disabled = true
#=╠═╡
out_test_trans = let
	(; par, statespace, π_init) = get_cali_basic() # perpetual youth

	model = HousingModel()
	
	guesses = (; K_supply = 5.817027126878351, H_hh = 0.0, L_eff = 
1.5351838287304251)
	prices = let
		r = interest_rate(guesses.K_supply, guesses.L_eff, par) 
		w = wage(guesses.K_supply, guesses.L_eff, par)
		p = house_price(par.δ * guesses.H_hh, par)
		(; r, w, p=nothing)
	end

	GE₀ = stationary_GE(model, par, statespace#=, guesses, prices=#; guesses, π_init, details = 10, tol = 1e-4)

	@info GE₀.aggregates.updated.K_supply
	@info GE₀.aggregates.updated.L_eff


	T̃ = 300

	#####################################
	## TEST 2: REDUCE MORTALITY BY 10% ##
	
	guess = GE₀.aggregates.updated
	guessed_path = dimstack_from_nt(guess, Dim{:t}(0:T̃))

	#guessed_path = guess_auclert_trans
	
	demographics = let
		m₀ = par.m
		m₁ = par.m ./ 0.0125 .* 0.01
		
		j_dim = DD.dims(m₀, :j)
		J = maximum(j_dim)
	
		borns = -J:1:T̃
		born_dim = Dim{:born}(borns)
		ms = cat([born < 0 ? m₀ : m₁ for born ∈ born_dim]..., dims = born_dim)

		
		demo = DimStack(ms, )
	end
	###########################
	
	inheritances = no_inheritances(par, statespace)
	
	out = transition_GE(model, T̃, par, statespace, demographics, GE₀, guessed_path;
						normalize_population = false, inheritances,
						details = 1, λ = 0.15
						)

	#	=#
end
  ╠═╡ =#

# ╔═╡ c77d4e45-53e1-403e-8eb6-6850a1c42c43
# ╠═╡ disabled = true
#=╠═╡
testxxx = let
	#K_path = K_guess_test2
	T̃ = 250 #length(K_path) - 1

	Me = :EGM
	Mo = HousingModel()

	length = 100
	inc_profile = true
	m₀ = 0.0125
	m₁ = 0.01
	
	constant_births = :even_simpler
	#GEs = get_GEs(Mo; length, inc_profile, #=constant_births,=# m₀, m₁)
	
	setup = setup_simple_transition(Mo; 
		m₀, m₁ = m₀, 
		length, #=constant_births,=# inc_profile,
		T̃, tol = 1e-6,
	)
	#setup.guessed_path.K_supply .= 4.718256725614218
	#setup.guessed_path.L_eff    .= 1.0

	(; demographics, guessed_path, statespace, GEs) = setup

	demographics = let
		m₀ = GEs.GE₀.par.m
		m₁ = GEs.GE₀.par.m ./ 0.0125 .* 0.01
		
		j_dim = DD.dims(m₀, :j)
		J = maximum(j_dim)
	
		borns = -J:1:T̃
		born_dim = Dim{:born}(borns)
		ms = cat([born < 0 ? m₀ : m₁ for born ∈ born_dim]..., dims = born_dim)

		
		demo = DimStack(ms, )
	end
	
	let
		fig = Figure()
		lines(fig[1,1], guessed_path.K_supply)
		lines(fig[1,2], guessed_path.L_eff)
		#@info fig
	end
	
	#setup.guessed_path
	
	#paths_in = current_paths
	#paths_in.H_hh .= 0.0
	
	paths_in = guessed_path
	
	(; GE₀) = GEs
	(; par) = GEs.GE₁
	inheritances = no_inheritances(par, statespace)

	model = Mo
	price_paths = get_price_paths(model, paths_in, par; GE₀)
	


	out = transition_GE(model, T̃, par, statespace, demographics, GE₀, paths_in;
						normalize_population = false, inheritances,
						λ = 0.1
					   )
		
	(; out, setup)
	
	#regression_test_transition(K_path, trans)
	# =#
end
  ╠═╡ =#

# ╔═╡ dc95d4a1-fd37-4c53-a6ab-96ef0c6f1bd2
function update_inheritances_transition!(inheritances_old, out_PE, statespace, F)

end

# ╔═╡ 0cc7187e-bf5c-4609-baeb-d986e5da7c52
#=╠═╡
let
	(; out, setup) = testxxx
	(; sim_df) = out
	(; statespace) = setup
	(; perm_dim) = statespace
	
	t_dim = Dim{:t}(0:10)
	@chain sim_df begin
		@transform(
			:t = :j + :born,
			:left_behind = :z_next * :m
		)
		@subset(0 ≤ :t ≤ 10)
		@groupby(:permanent, :t)
		@combine(
			:mass = sum(:π),
			:avg_left_behind = mean(:left_behind, weights(:π))
		)
		@groupby(:permanent)
		@combine(
			:lb = Ref(DimVector(:avg_left_behind, t_dim))
		)
		#stack(_.lb, (perm_dim, t_dim))
		
	end
end
  ╠═╡ =#

# ╔═╡ f063455e-694d-42d3-8b1a-b154a9fc8b7f
#=╠═╡
testxxx.out.aggregate_paths.loss.ζ_K |> lines
  ╠═╡ =#

# ╔═╡ 384cc606-fdb4-4fe5-8721-89acf4fbe832
#=╠═╡
let
	(; out, setup) = testxxx
	(; sim_df, GE₀) = out
	(; statespace) = setup
	(; perm_dim) = statespace

	factor = @chain sim_df begin
		@transform(:t = :j + :born)
		@subset( :t == 0)
		@combine(:π = sum(:π))
		only(_.π)
	end

	@chain DataFrame(first(GE₀.sols).sol.sol_forward.π) begin
		@groupby(:j)
		@combine(:val = mean(:state, weights(:π)))
		data(_) * mapping(:j, :val) * visual(Lines)
		draw
	end

	@chain sim_df begin
		@transform(:t = :j + :born)
		#@subset(:t == 3)
		#@subset(0 ≤ :t ≤ 100, :born == 0)
		@transform(:π = :π ./ factor)
		
		@groupby(:j, :born, :t)
		@combine(
			:π = sum(:π),
			:state = mean(:state, weights(:π))
		)
		@subset(:born ∈ -50:5:0)
		data(_) * mapping(:t, :state, group = :born => nonnumeric) * visual(Lines)
		draw
		
#		@combine(:state = mean(:state, weights(:π)))
#		lines(_.t, _.state)
	end
#	=#
end
  ╠═╡ =#

# ╔═╡ 873a8e67-f903-4c5d-b3a8-16a34ee5a2b2
md"""
# Appendix
"""

# ╔═╡ bd9cb9ce-e97e-4d6e-a2fa-ebaa1ce9fa41
TableOfContents()

# ╔═╡ 2b1fee28-1ddf-454a-81c6-f652c41a4cdd
fonts = (; regular = Makie.MathTeXEngine.texfont(:regular), bold = Makie.MathTeXEngine.texfont(:regular))

# ╔═╡ 37bf32d9-a199-469e-8141-fa7409c22b38
figure(size = (350, 250); figure_padding = 2, kwargs...) = (; size, fonts, figure_padding, kwargs...)

# ╔═╡ 34141abb-2723-44e8-8ba9-9d2345921212
let
	
	
	vars = [:z_next, :z, :a_next, :c, :income]

	df = @chain out_auclert.sim_df begin
		select(vars..., :π, :j, :permanent => AsTable)
		@transform!(:θ = round(:θ, digits = 2))
	end
	
	qs = 0.1:0.1:0.9
	@chain df begin
		stack(vars, [:π, :j])
	    @groupby(:variable, :j)
		@combine(
			:grp = [fill("Q", length(qs)); "mean"],
			:q = [string.(qs); "mean"],
			:value = [quantile(:value, weights(:π), qs); mean(:value, weights(:π))]
			
					  )
		@aside @show @subset(_, :grp == "mean")
		#@subset(:grp == "mean")

		data(_) * mapping(:j => L"model age $j$", :value,# => log => L"\log(\cdot)", 
						  group = :q => nonnumeric,
						  color = :grp,
						 # group = :θ => nonnumeric,
						#  linestyle = :θ => nonnumeric,
						  layout = :variable
						 ) * visual(Lines)
		draw(; facet = (; linkyaxes = false), figure = figure((450, 250))) # 
	end
	# =#
end

# ╔═╡ 40817425-b996-4528-8935-5ddaf73dedf1
let
	fig = Figure(; figure((400, 150))...)

	lines(fig[1,1], simple_income_profile(100, 50), axis = (; title = "Simplest income profile"))
	lines(fig[1,2], income_profile(120, 41), axis = (; title = "Simple income profile"))
	fig
end

# ╔═╡ d757776b-6651-404e-81d6-d5d91cd88ac9
criterion(a, b) = (a - b)/(1 + max(abs(a), abs(b)))

# ╔═╡ a7a4bc12-e8c6-4f7e-bd20-5fb97219f7ef
const DD = DimensionalData

# ╔═╡ b4bb0db5-33a9-4da9-ab3c-a2e99b79115d
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

# ╔═╡ 62192287-bcb7-4bf6-a4d0-fec028379cb3
function get_demographics(scenario, borns; age_min, age_max)

	if scenario ∉ [:baby_boom, :births_down, :mortality_down]
		throw(ArgumentError("scenario must be in [:babyboom, :births_down, :mortality_down]"))
	end
	
	ages = Dim{:age}(0:119)
	p₀ = p_surv.(ages)
	p₀ = p₀[age = At(age_min:age_max-1)]
	J = length(p₀)

	j_dim = Dim{:j}(0:J)
	m_baseline = DimVector([1 .- p₀; 1.0], j_dim, name = :m) # correct indexing for mortality
	pₓ = DimVector(p₀, Dim{:j}(1:J)) # shifted indexing for pmf
	p₊ = prepend_one(pₓ)
		

	pmf_baseline = cumprod(p₊)
	pmf_baseline = pmf_baseline ./ sum(pmf_baseline)
	@info @test sum(pmf_baseline) ≈ 1.0
	births_baseline = pmf_baseline[j = At(0)]
	p₊[j = At(0)] = births_baseline
	j₊_dim = DD.dims(p₊, :j)
	j₊s = collect(collect(j₊_dim))
	
	pmf_baseline  = change_births(p₊, 1.0 * births_baseline, name = :pmf)
	
	
	born_dim = Dim{:born}(borns)
	
	if scenario == :baby_boom
		pmf_scenario  = change_births(p₊, 1.25 * births_baseline, name = :pmf)
		
		pmfs = cat(
			(10 < born ≤ 30 ? pmf_scenario : pmf_baseline for born ∈ borns)...,
			dims = born_dim
		)

		ms = cat(fill(m_baseline, born_dim)..., dims = :born)
		
	elseif scenario == :births_down
		pmf_scenario = change_births(p₊, 0.75 * births_baseline, name = :pmf)
		
		pmfs = cat(
			(born > 0 ? pmf_scenario : pmf_baseline for born ∈ borns)...,
			dims = born_dim
		)
		
		ms = cat(fill(m_baseline,   born_dim)..., dims = :born)
		
	elseif scenario == :mortality_down
		stretch_factor = 0.8
		stretched = stretch_mortality(
			pₓ, stretch_factor; births = births_baseline
		)
		pmf_scenario = stretched.pmf
		p_scenario   = stretched.m

		pmfs = cat(
			(born > 0 ? pmf_scenario : pmf_baseline for born ∈ borns)...,
			dims = born_dim
		)

		ms = cat(
			(born > 0 ? p_scenario : m_baseline for born ∈ borns)...,
			dims = born_dim
		)
	end

	demo = DimStack(pmfs, ms)
end

# ╔═╡ 7ad1c088-ab3d-4363-912b-2cb8344f2478
function setup_transition(Mo, par, statespace; T̃ = 150,
						  scenario = :baby_boom, 
						  GE_kwargs = (;)
						 )
	(; age_min, age_max) = par
	Me = :EGM
	
	GEs = let
			
		par₀ = par
		par₁ = par

		guesses₀ = (; K_supply = 19.2435, H_hh = 5.25371, L_eff = 2.25915)
		guesses₁ = (; K_supply = 19.2435, H_hh = 5.25371, L_eff = 2.25915)

		GE₀ = stationary_GE(Mo, par₀, statespace; guesses=guesses₀, GE_kwargs...)
		GE₁ = GE₀
		#GE₁ = stationary_GE(Mo, par₁, statespace; guesses=guesses₁, GE_kwargs...)
		
		init = nothing #grid[1]
	
		pmf₀ = pmf(par₀.m)
		pmf₁ = pmf(par₁.m)
		
		j_last = par₀.J
		
		(; GE₀, GE₁, j_last)
	end

	pre_shock_periods = 10

	(; GE₀, j_last) = GEs
	demographics = get_demographics(scenario, -(j_last+pre_shock_periods):T̃; age_min, age_max)
	
	guess = GE₀.aggregates.updated
	guessed_path = dimstack_from_nt(guess, Dim{:t}(0:T̃))

	(; solution_method=Me, model=Mo, GEs, statespace, demographics, guessed_path, T̃)
end

# ╔═╡ d28aab10-5650-41eb-bce5-85be9481d067
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
		h, u, u′, u′⁻¹, v, v′, ν₀, ν₁,
		w = 1.0, ρ_SS, d̄, τ,
		ξ, α̃, L̄, p, j_dim, J, age_min, age_max, annuities)
	
end

# ╔═╡ 324cb71a-eb73-45bd-8aad-431ebdbb4ea1
let
	J = 5
	j_dim = Dim{:j}(0:J)
		
	h = zeros(j_dim)
	m = zeros(j_dim)
		
	get_par₀(; m, h, JR = 2, ξ = 0.15)
end

# ╔═╡ 98a4e14e-29e0-4007-862c-2dbeef19ae8a
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

# ╔═╡ 8e7bd67c-10b2-4f1e-a2a4-26ec41a65e0f
function age_dependent_par_ξ_β(age_min, age_max; ξ, β, kwargs...)
	J = age_max - age_min
	j_dim = Dim{:j}(0:J)

	@assert DD.dims(ξ, :j) == j_dim
	@assert DD.dims(β, :j) == j_dim
	
	par = get_par(; 
		inc_profile = true, demo=:lifecycle, β, ξ, age_min, age_max,
		h = cex_targets(age_min, age_max).income, a̲ = 0.0, τ = 0.0, JR = J, kwargs...
	)
end

# ╔═╡ 7df0f1a9-1461-4cee-9692-7941d67790d4
par_cali_by_hand = let
	age_min = 20
	age_max = 87
	J = age_max - age_min
	js = 0:J
	j_dim = Dim{:j}(js)

	j_ξ = [0,    10,   25,   37,  50,   55,   57,   60,  65,   J]
	ξ_j = [0.17, 0.18, 0.18, 0.24, 0.32, 0.4, 0.44, 0.5, 0.7, 0.8]
	
	j_β = [0,    2,    5,    10,   25,   50,   55,   60,   65,   J] 
	β_j = [0.90, 0.90, 0.90, 0.90, 0.90, 0.80, 0.80, 0.70, 0.60, 0.60]

	ξ_itp = linear_interpolation(j_ξ, ξ_j)
	ξ = DimVector(ξ_itp[js], j_dim, name = "ξ")
	
	β_itp = linear_interpolation(j_β, β_j)
	β = DimVector(
		[1.0; cumprod(β_itp[js])[1:end-1]],
		j_dim, name = "β")
	
	par = age_dependent_par_ξ_β(age_min, age_max; ξ, β)
end

# ╔═╡ 0d13aba8-24a1-4d18-9ab2-ac09e8c21bd5
function age_dependent_par(age_min, age_max, ξ_coefs, lβ_coefs; kwargs...)
	
	J = age_max - age_min

	(; ξ, β) = age_dependent_ξ_β(; ξ_coefs, lβ_coefs, J)
	
	age_dependent_par_ξ_β(age_min, age_max; ξ, β, kwargs...)
end

# ╔═╡ ee5344fc-9efb-4de0-809a-fded19a4f0aa
function cali_out(ξ_pars, β_pars, model, statespace, prices)
	par = age_dependent_par(20, 87, ξ_pars, β_pars)

	cali_out(model, par, statespace, prices)
end

# ╔═╡ a308aeb5-f284-4bfb-86fb-e993070ec4a7
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

# ╔═╡ c6c967db-2d63-4178-ad5b-d6ff4d454300
par_no_risk = let
	#y = income_profile(120, 41)
	
	par = get_par(; demo = :lifecycle, a̲ = 0.0, age_min = 0, age_max = 100, ξ = 0.0000001, JR = 100)

	statespace = Statespace(; amin = 0.0, amax = 25.0, na = 500, ε_chain = no_income_risk(), exponential = false)

	tmp = let
		L_eff = 1.0
		H_hh = 1.0
		K_supply = 4.522301994771901
		
		guesses_nt = (; K_supply, L_eff, H_hh)
		(; prices, guesses) = prices_from_guesses_nt(HousingModel(), guesses_nt, par)
		
		price_paths = constant_price_paths(par, prices)

		(; prices, guesses, price_paths)
	end
	
	(; par, statespace, tmp..., π_init = trivial_initial_distribution(statespace, init_state = 0.0))

end

# ╔═╡ 23aca92f-9ae2-4c3e-9762-613f7ab6d147
out_safe = let
	(; statespace, prices, price_paths, guesses, par, π_init) = par_no_risk

	permanent = (; θ = 1)
	Mo = HousingModel()
	out = stationary_PE(Mo, par, statespace, guesses, prices; π_init) 

	#out = stationary_GE(Mo, par, statespace; π_init, tol = 1e-4, details = true) 
	
	#@info (; out.K_supply, out.ζ, out.r, out.K_hh)

	
	#(; sim_df) = ou

	#agg_nt = aggregate(sim_df)

	out
end

# ╔═╡ c40c0622-e228-4f70-bbf7-2ea0e8289f6d
out_safe.sim_df

# ╔═╡ 299ae2ba-91f9-4e58-89a3-188c7cc7926a
par_with_risk = let
	#y = income_profile(120, 41)

	a̲ = -0.0 # -0.35
	par = get_par(; demo = :lifecycle, a̲, ξ = 1e-10, JR = 96)

	statespace = Statespace(; amin = a̲, amax = 30.0, na = 600, ε_chain = simple_income_risk(), exponential = true)

	out = let
		L_eff = 1.0
		H_hh = 1.0
		K_supply = 4.522301994771901
		
		guesses_nt = (; K_supply, L_eff, H_hh)
		(; prices, guesses) = prices_from_guesses_nt(HousingModel(), guesses_nt, par)

		prices = (; prices.r, prices.w, p = 1.0)
		price_paths = constant_price_paths(par, prices)

		(; prices, guesses, price_paths)
	end
			
	(; par, statespace, out..., π_init = trivial_initial_distribution(statespace, init_state = 0.0))

end

# ╔═╡ a0cc6059-1f5c-44d7-9592-321b9f526736
out_risky = let
	(; par, statespace, prices, guesses, π_init) = par_with_risk
	
	Mo = HousingModel()

	permanent = (; θ = 1.0)
	guesses = (; K_supply = 3.37966, H_hh = 2.65662, L_eff = 1.0)
	guesses = (; K_supply = 5.64885, H_hh = 2.98439, L_eff = 1.0)

	out = stationary_PE(Mo, par, statespace, guesses, prices; π_init) 
	#out = partial
	#out = stationary_GE(Mo, par, statespace; guesses, tol = 1e-4, maxiter = 100, λ = 0.01, details = true)
	
end

# ╔═╡ 3ae08b70-3b24-4f6e-8d0d-16642e3a8904
let
	vars = [:z_next, :a_next, :ho, :c]

	#df = select(out_safe.sim_df, vars..., :π, :j, :y)
	df = select(out_risky.sim_df, vars..., :π, :j, :ε)
	
	@chain df begin
		@transform!(:method = "EGM")
		stack(vars, [:π, :j, :method])
		@groupby(:variable, :j, :method)
		@combine(
			:q = [0.2, 0.5, 0.8],
			:value = quantile(:value, weights(:π), [0.2, 0.5, 0.8]))
		#@subset(:variable == "a")
		data(_) * mapping(:j, :value, 
						  group = :q => nonnumeric, color = :method,
						  linestyle = :method,
						  layout = :variable
						 ) * visual(Lines)
		draw(; facet = (; linkyaxes = false), figure = figure((500, 250)))
	end
end

# ╔═╡ 10c45b23-e437-421f-baa0-97e91597f125
function get_cali_auclert(; ξ = 0.0, risk = false)
	
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
							amin = Z̲, amax = 1_000.0, na = 400, 
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

# ╔═╡ 753ce801-bec3-4b1e-8184-5a00bc5d41b2
cali_auclert = get_cali_auclert(; ξ = 0.0) 

# ╔═╡ b298eafd-a826-4570-ac92-e4f9270dc9b9
cali_auclert.prices

# ╔═╡ 21c8e2d0-fe6a-4e7b-b07f-d1c85a861a00
cali_auclert.par.h

# ╔═╡ 76dbf7ae-910a-4649-ad21-7d3814236f36
cali_auclert.par.m

# ╔═╡ 290b76eb-e8f5-45ee-9f20-4f4c34787256
cali_auclert.statespace.mc_permanent.p

# ╔═╡ 708bd44f-bd01-4cfa-8f9c-b6f80515764a
# ╠═╡ disabled = true
#=╠═╡
out_bequests = let
	(; par, statespace, prices, π_init) = cali_auclert

	prices = (; prices.p, prices.w, r = -0.05)
	
	Mo = HousingModel()

	guesses = (; K_supply = 22.7283, H_hh = 0.0)
	# HELPERS
	(; par) = cali_auclert
	par = (; par..., r = 0.01)
	(; j_dim) = par
	F = DimVector(F_marcelo ./ sum(F_marcelo), j_dim, name = :F)
	#π_age = get_age_distribution(out_auclert)

	
	out = stationary_GE(Mo, par, statespace; F, π_init, details = 1, λ = 0.25, maxiter = 500, tol = 1e-4)
end
  ╠═╡ =#

# ╔═╡ a3951f0c-20d0-4abb-90fb-2f61a0b50247
#=╠═╡
@info @test out_bequests.prices.r ≈ 0.01980695128459809
  ╠═╡ =#

# ╔═╡ 84b89600-4713-4f51-8d7c-161545994b1a
let
	(; par, statespace) = cali_auclert

	no_inheritances(par, statespace)

end

# ╔═╡ a8718f51-c43d-4832-8b73-7ba7a15dae10
cali_auclert.par

# ╔═╡ 7152e4f2-7600-4e83-8dbd-fb3e88f33e90
cali_by_hand = let
	age_min = 20
	age_max = 87
	J = age_max - age_min
	js = 0:J
	j_dim = Dim{:j}(js)

	j_ξ = [0,    10,   25,   37,  50,   55,   57,   60,  65,   J]
	ξ_j = [0.17, 0.18, 0.18, 0.24, 0.32, 0.4, 0.44, 0.5, 0.7, 0.8]
	
	j_β = [0,    2,    5,    10,   25,   50,   55,   60,   65,   J] 
	β_j = [0.90, 0.90, 0.90, 0.90, 0.90, 0.80, 0.80, 0.70, 0.60, 0.60]

	ξ_itp = linear_interpolation(j_ξ, ξ_j)
	ξ = DimVector(ξ_itp[js], j_dim, name = "ξ")
	
	β_itp = linear_interpolation(j_β, β_j)
	β = DimVector(
		[1.0; cumprod(β_itp[js])[1:end-1]],
		j_dim, name = "β")
		
	model = HousingModel()
	par = age_dependent_par_ξ_β(age_min, age_max; ξ, β)
	statespace = Statespace(; amin=0.0, amax=450, na=1000, exponential = true)
	prices = (r = 0.002, p = 1.0, w = 1.0)
	
	out = cali_out(model, par, statespace, prices)
	
	(; loss_nt, tmp, means_by_age) = loss_cali(out, true)
	
	means_long = @chain means_by_age begin
		stack([:z_next], :j)
		@transform(:source = "model")
#		data(_) * mapping(:j, :ω_next) * visual(Lines)
#		draw
	end
	
	#@info loss_nt
	fig = @chain tmp begin
		[_; means_long]
		data(_) * mapping(:j => L"model age $j$", :value => "", col = :variable, linestyle = :source) * visual(Lines)
		draw(facet = (; linkyaxes = false), figure = figure((600, 150)))
	end

	(; fig, β, ξ, loss_nt, J, model, statespace, prices)
	# =#
end

# ╔═╡ facbc90c-b8e2-4204-9cbf-547ce4372d35
cali_by_hand.fig

# ╔═╡ 05993d47-932b-46a7-a39e-beeca0c872f2
cali_fitted = let
	(; ξ, β, J, model, statespace, prices) = cali_by_hand

	df = @chain DimStack(ξ, β) begin
		DataFrame
		@transform(:log_β = log(:β), :source = "hand")
		@transform(:j_t = :j ./ (J/2) .- 1.0)
	end

	reg_lβ = lm(@formula(log_β ~ j_t + j_t^2 + j_t^3), df)
	reg_ξ  = lm(@formula(ξ     ~ j_t + j_t^2 + j_t^3), df)

	coef_lβ = coef(reg_lβ)
	coef_ξ  = coef(reg_ξ)
	
	p_lβ = Polynomial(coef_lβ)
	p_ξ  = Polynomial(coef_ξ)

	df_fitted = DataFrame(; df.j_t, df.j)
	@transform!(df_fitted, :ξ = p_ξ(:j_t), :log_β = p_lβ(:j_t), :source = "fitted")
	
	fig = @chain begin
		[select(df, Not(:β)); df_fitted]
		rename(:log_β => L"\log(β)")
		stack([L"\log(β)", "ξ"])
		@transform(:variable = latexstring(:variable))
		data(_) * mapping(
			:j, :value => "", 
			row = :variable, linestyle = :source
		) * visual(Lines)
		draw(facet = (; linkyaxes = false), figure = figure(; title = "age dependent parameters") )
	end
	
	@transform!(df_fitted, :β = exp(:log_β))
	
	(; coef_lβ, coef_ξ, fig, js = df.j, js_trans = df.j_t, df_fitted, model, statespace, prices)
	#lines(df.j, j -> p_lβ(j))
end

# ╔═╡ 5623a579-d066-4743-adc1-4bd73ce47588
cali_fitted.fig

# ╔═╡ 558a92af-b868-4e3a-8c00-a5db713bc467
let
	age_min = 20
	age_max = 87
	J = age_max - age_min
	
	(; coef_lβ, coef_ξ, js, js_trans, df_fitted, model, statespace, prices) = cali_fitted

	@info (; js_trans)
	#@info df_fitted.log_β
	
	p_lβ = ChebyshevT(Polynomial(coef_lβ))
	p_ξ  = ChebyshevT(Polynomial(coef_ξ))

	lβ_coefs_30 = coefs_scale_30(coeffs(p_lβ))
	ξ_coefs_30 = coefs_scale_30(coeffs(p_ξ))

	@info (; lβ_coefs_30, ξ_coefs_30)
	par = age_dependent_par(age_min, age_max, ξ_coefs_30, lβ_coefs_30)
	
	out = cali_out(model, par, statespace, prices)
		
	(; loss_nt, mean_loss, tmp) = loss_cali(out, true)

	@info (; mean_loss, loss_nt)
	
	fig = @chain tmp begin
		data(_) * mapping(:j => L"model age $j$", :value => "", row = :variable, linestyle = :source) * visual(Lines)
		draw(facet = (; linkyaxes = false), figure = figure(title = L"Model fit with polynomial approx $\beta$ and $\xi$" ))
	end

	fig
	# =#
end

# ╔═╡ c9c5d463-c557-45a7-ada4-a3dbbef06556
out_sobol = let
	J = 87 - 20
	order_ξ = 3
	order_lβ = 3

	statespace = Statespace(; amin = 0.0, amax = 450, na = 1000)
	prices = (; r = 0.038, p = 1.0, w = 1.0)
	model = HousingModel()
	
	tries = let
		N_sobol = 5000
		lb = [fill(0.0, order_ξ + 1); fill(-7.0, order_lβ + 1)]
		ub = [fill(0.5, order_ξ + 1); fill(0.0, order_lβ + 1)]
		#lb[order_ξ + 2] = 0
		
		ss = SobolSeq(lb, ub)
		
		[checked_next!(ss; order_ξ, order_lβ, J) for _ ∈ 1:N_sobol]
	end

	tries = tries[4290:4290]
	
	ξ_tries = first.(tries, order_ξ + 1)
	β_tries = last.(tries, order_lβ + 1)

	tmp = loss_cali.(cali_out.(ξ_tries, β_tries, Ref(model), Ref(statespace), Ref(prices)))

	@info mean(isfinite.(tmp)), sum(isfinite.(tmp))

	keep = isfinite.(tmp)
	
	(min, i) = findmin(tmp[keep])
	
	ξ_pars = ξ_tries[keep][i]
	β_pars = β_tries[keep][i]

	par = age_dependent_par(20, 87, ξ_pars, β_pars)
	
	out = loss_cali(cali_out(model, par, statespace, prices), true)
	(; ξ_pars, β_pars, min, out, i, model, statespace, prices, par)
	
end

# ╔═╡ 13bf8c89-c974-4211-a4fc-812caed060f0
let
	@chain out_sobol[4].tmp begin
		unstack(:source, :value)
		@transform(:Δ = abs(:model - :data) / (:data) )
		@groupby(:variable)
		@combine(√mean(:Δ .^ 2))
	end
end
	

# ╔═╡ 5bc11374-27c6-4b94-a1e6-0825d68ae8cc
out_sobol.ξ_pars

# ╔═╡ 1bd5dba4-18ac-436a-b553-9b1dc6a46b1b
out_sobol.β_pars

# ╔═╡ e4b287cb-525f-42be-97db-32c66bd00916
let
	(; out, par) = out_sobol

	@info lines(par.β, figure = (; size = (250, 250)))
	
	(; loss_nt, tmp) = out # loss_cali(out, true)

	@info loss_nt
	@chain tmp begin
		data(_) * mapping(:j, :value, row = :variable, linestyle = :source) * visual(Lines)
		draw(facet = (; linkyaxes = false), figure=figure())
	end

end

# ╔═╡ 7fd6c219-35e9-4fff-9270-6b67010f6d88
function get_cali_basic(; amax = 20.0, na = 500, exponential = false)

	par_kwargs = (; ξ = 0.0, δ = 0.1, β = 0.995,
				   bonds2GDP = 1.0, NFA2GDP = 0.0, τ = 0.0,
				   ν₀ = 0.0, annuities = false,
				   α = 0.33, θ = 1.0, Z̲ = 0.0, a̲ = 0.0)

	
	par = get_par(; demo=:perpetual_youth, inc_profile=true, mm=0.0125, par_kwargs...)

	statespace = let
		ε_chain = no_income_risk()
		permanent = no_permanent_states()
	
		Statespace(; amin = 0.0, amax, na, 
					 ε_chain, permanent, exponential = true)
	end

	π_init = trivial_initial_distribution(statespace, init_state = 0.0)
	
	inheritances = no_inheritances(par, statespace)
	
	(; par, statespace, π_init, inheritances)
end

# ╔═╡ 93a1ce7b-d275-4f97-b28c-4e86683b4930
function get_GEs(Mo; length, demo = :perpetual_youth,
						m₀=1/45, m₁=m₀,
 						inc_profile = true,
					# constant_birhts = even_simpler	
						 tol=1e-6,
				 )

	par_kwargs = (; ξ = 0.0, δ = 0.1, β = 0.995,
				   bonds2GDP = 1.0, NFA2GDP = 0.0, τ = 0.0,
				   ν₀ = 0.0, annuities = false,
				   α = 0.33, θ = 1.0, Z̲ = 0.0, a̲ = 0.0)

	
	par₀ = get_par(; demo, inc_profile, mm=m₀, par_kwargs...)
	par₁ = get_par(; demo, inc_profile, mm=m₁, par_kwargs...)

	statespace = let
		ε_chain = no_income_risk2()
		permanent = no_permanent_states2()
	
		Statespace(; amin = 0.0, amax = 20.0, na = length, 
					 ε_chain, permanent, exponential = true)
	end

	π_init = trivial_initial_distribution(statespace, init_state = 0.0)
	
	guess₀ = (; K_supply = 5.742671821160339, H_hh =8.745507756539205e-6, L_eff = 1.5351838287304207)
	guess₁ = (; K_supply = 5.980849860878322, H_hh =8.944627928083163e-6, L_eff = 1.538941534193358)

	GE₀ = stationary_GE(Mo, par₀, statespace; π_init, guesses=guess₀, tol)
	GE₁ = stationary_GE(Mo, par₁, statespace; π_init, guesses=guess₀, tol)

	#@info GE₀.aggregates.updated.H_hh
	#@info GE₀.aggregates.updated.L_eff
	
	#@info GE₁.aggregates.updated.H_hh
	#@info GE₁.aggregates.updated.L_eff
	
	j_last = par₀.J
	
	(; GE₀, GE₁, statespace, j_last)
end

# ╔═╡ 97beacd0-7a2e-4754-8c3d-017901019e16
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

# ╔═╡ c70b7252-06ab-4fc6-97f0-43e28d277d98
cali_test = get_cali_test()

# ╔═╡ 8ddec5f5-2107-469e-8bba-923b77153a8e
out_auclert_trans2 = let
	#(; par, statespace, π_init) = get_cali_test()
	(; par, statespace, prices, π_init) = get_cali_auclert(ξ = 0.15, risk = false)

	model = HousingModel()

	guesses = (; K_supply = 19.8058, H_hh = 2.7714, L_eff = 2.2591478197110324)
	(; prices) = prices_from_guesses_nt(HousingModel(), guesses, par)

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
						details = 1, λ = 0.001, maxiter = 3000, tol = 5e-3
						)
		
	#	=#
end

# ╔═╡ b728930d-689c-4c4b-9e75-5d67763252d0
function even_simpler(m₀, m₁::Real, (; T̃))
	j_dim = DD.dims(m₀, :j)
	J = maximum(j_dim)
	
	borns = -J:1:T̃
	born_dim = Dim{:born}(borns)
	ms = cat(fill(m₀, born_dim)..., dims = born_dim)

	for j ∈ j_dim, born ∈ born_dim
		t = born + j
		if t ≥ 0 && j < J
			ms[j = At(j), born = At(born)] = m₁
		end
	end

	pmf₀ = pmf(m₀)
	births₀ = pmf₀[j = At(0)]

	out = map(groupby(ms, :born => identity)) do m
		surv = [births₀; dropdims(1 .- m, dims=:born)]
		DimVector(cumprod(surv)[1:end-1], j_dim)
	end

	pmfs = DimArray(
		cat(out..., dims = born_dim),
		name = :pmf
	)

	demo = DimStack(ms, )
end

# ╔═╡ 6acd7d7a-6b0d-45a8-adf6-27859ff6b6e5
function setup_simple_transition(model;
		demo = :perpetual_youth,
		T̃ = 300, #=constant_births=true,=# kwargs...)
 
	GEs = get_GEs(model; length, demo, #=constant_births,=# kwargs...)
	
	(; GE₀, GE₁, statespace) = GEs

	guess = GE₀.aggregates.updated
	guessed_path = dimstack_from_nt(guess, Dim{:t}(0:T̃))
	
	@info guessed_path
	
	j_dim = DD.dims(GE₁.par.m, :j) ## TODO CHECK THIS!!!
	J = maximum(j_dim)

	#T̃ = 200

	#@assert constant_births == :even_simpler
	demographics = let
		m₀ = GE₀.par.m
		m₁ = GE₁.par.m

		m₁ = m₁[j = At(0)]
		even_simpler(m₀, m₁, (; T̃))
	end

	(; guessed_path, T̃, demographics, GEs, statespace, model)

end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
DataFrameMacros = "75880514-38bc-4a95-a458-c2aea5a3a702"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Polynomials = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
QuantEcon = "fcd29c91-0bd7-5a09-975d-7ac3f643a60c"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
Sobol = "ed01d8cd-4d21-5b2a-85b4-cc3bdc58bad4"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
AlgebraOfGraphics = "~0.11.7"
CairoMakie = "~0.15.6"
Chain = "~1.0.0"
DataFrameMacros = "~0.4.1"
DataFrames = "~1.7.1"
DimensionalData = "~0.29.23"
GLM = "~1.9.0"
Interpolations = "~0.16.2"
LaTeXStrings = "~1.4.0"
PlutoLinks = "~0.1.6"
PlutoTest = "~0.2.2"
PlutoUI = "~0.7.71"
Polynomials = "~4.1.0"
QuantEcon = "~0.16.8"
Roots = "~2.2.10"
Sobol = "~1.5.0"
StatsBase = "~0.34.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "4319d525ea82e970095ec00809cc6778e1a0d917"

[[deps.ADTypes]]
git-tree-sha1 = "60665b326b75db6517939d0e1875850bc4a54368"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.17.0"

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
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

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
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
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
git-tree-sha1 = "748501513016edd2f15fa5ccb765e09d849d387b"
uuid = "cbdf2221-f076-402e-a563-3d30da359d67"
version = "0.11.7"

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
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "dbd8c3bbbdbb5c2778f85f4422c39960eac65a42"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.20.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
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
version = "1.11.0"

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
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

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
version = "1.11.0"

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
git-tree-sha1 = "f8caabc5a1c1fb88bcbf9bc4078e5656a477afd0"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.6"

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
git-tree-sha1 = "5ac098a7c8660e217ffac31dc2af0964a8c3182a"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "2.0.0"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "e771a63cc8b539eca78c85b0cabd9233d6c8f06f"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "a656525c8b46aa6a1c76891552ed5381bb32ae7b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.30.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

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
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
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
git-tree-sha1 = "cb1299fee09da21e65ec88c1ff3a259f8d0b5802"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.4"

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
git-tree-sha1 = "a37ac0840a1196cd00317b57e39d6586bf0fd6f6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "5620ff4ee0084a6ab7097a27ba0c19290200b037"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.4"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "a86af9c4c4f33e16a2b2ff43c2113b2f390081fa"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.5"

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
git-tree-sha1 = "16946a4d305607c3a4af54ff35d56f0e9444ed0e"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.7"

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
deps = ["Adapt", "ArrayInterface", "ConstructionBase", "DataAPI", "Dates", "Extents", "Interfaces", "IntervalSets", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "PrecompileTools", "Random", "RecipesBase", "Statistics", "TableTraits", "Tables"]
git-tree-sha1 = "611a26d31e6739dd7b6cf71f2a086d11f820ffdb"
uuid = "0703355e-b756-11e9-17c0-8b28908087d0"
version = "0.29.23"

    [deps.DimensionalData.extensions]
    DimensionalDataAbstractFFTsExt = "AbstractFFTs"
    DimensionalDataAlgebraOfGraphicsExt = "AlgebraOfGraphics"
    DimensionalDataCategoricalArraysExt = "CategoricalArrays"
    DimensionalDataDiskArraysExt = "DiskArrays"
    DimensionalDataMakie = "Makie"
    DimensionalDataNearestNeighborsExt = "NearestNeighbors"
    DimensionalDataPythonCall = "PythonCall"
    DimensionalDataSparseArraysExt = "SparseArrays"
    DimensionalDataStatsBase = "StatsBase"

    [deps.DimensionalData.weakdeps]
    AbstractFFTs = "621f4979-c628-5d54-868e-fcf4e3e8185c"
    AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
    CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
    DiskArrays = "3c3547ce-8d99-4f5e-a174-61eb10b00ae3"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
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
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3e6d038b77f22791b8e3472b7c633acea1ecac06"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.120"

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
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7bb1361afdb33c7f2b085aa49ea8fe1b0fb14e58"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.1+0"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "eaa040768ea663ca695d442be1bc97edfe6824f2"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.3+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "797762812ed063b9b94f6cc7742bc8883bb5e69e"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.9.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

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
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "31fd32af86234b6b71add76229d53129aa1b87a9"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.28.1"

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
git-tree-sha1 = "ce15956960057e9ff7f1f535400ffa14c92429a4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.1.0"
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
version = "1.11.0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "273bd1cd30768a2fddfa3fd63bbc746ed7249e5f"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.0"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "8e233d5167e63d708d41f87597433f59a0f213fe"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.4"

[[deps.GeoInterface]]
deps = ["DataAPI", "Extents", "GeoFormatTypes"]
git-tree-sha1 = "0f265264b9287a19715dc5d491dbe3aff00c1e71"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.5.0"
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
git-tree-sha1 = "35fbd0cefb04a516104b8e183ce0df11b70a3f1a"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.84.3+0"

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
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7a98c6502f4632dbe9fb1973a4244eaa3324e84d"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.1"

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
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

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
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

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
version = "1.11.0"

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
deps = ["CRlibm", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Random", "RoundingEmulator"]
git-tree-sha1 = "79342df41c3c24664e5bf29395cfdf2f2a599412"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.36"

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
git-tree-sha1 = "5fbb102dcb8b1a858111ae81d56682376130517d"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.11"
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
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

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
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e95866623950267c1e4878846f848d94810de475"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.2+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "d8337622fe53c05d16f031df24daf0270e53bc64"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.5"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "ba51324b894edaf1df3ab16e2cc6bc3280a2f1a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.10"

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
version = "1.11.0"

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
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

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
git-tree-sha1 = "706dfd3c0dd56ca090e86884db6eda70fa7dd4af"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d3c8af829abaeba27181db4acb485b18d15d89c6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.1+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "4adee99b7262ad2a1a4bbbc59d993d24e55ea96f"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.4.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "f749e7351f120b3566e5923fefdf8e52ba5ec7f9"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.6.4"

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
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "73b98709ad811a6f81d84e105f4f695c229385ba"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.4.3"

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
git-tree-sha1 = "368542cde25d381e44d84c3c4209764f05f4ef19"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.6"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "a370fef694c109e1950836176ed0d5eabbb65479"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.6"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

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
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2ae7d4ddec2e13ad3bddf5c0796f7547cf682391"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.2+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "EnumX", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "61942645c38dd2b5b78e2082c9b51ab315315d10"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.13.2"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

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
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

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
git-tree-sha1 = "275a9a6d85dc86c24d03d1837a0010226a96f540"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.3+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTest]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "Test"]
git-tree-sha1 = "17aa9b81106e661cffa1c4c36c17ee1c50a86eda"
uuid = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
version = "0.2.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

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
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

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
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

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
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

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
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "d852eba0cc08181083a58d5eb9dccaec3129cb03"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.9.0"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

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
git-tree-sha1 = "fea870727142270bdf7624ad675901a1ee3b4c87"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.1"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

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
version = "1.11.0"

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
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

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

[[deps.Sobol]]
deps = ["DelimitedFiles", "Random"]
git-tree-sha1 = "5a74ac22a9daef23705f010f72c81d6925b19df8"
uuid = "ed01d8cd-4d21-5b2a-85b4-cc3bdc58bad4"
version = "1.5.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "85a43f6fc30dd80c61342519ca2fc36dac912eae"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.6"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "8ad2e38cbb812e29348719cc63580ec1dfeb9de4"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.1"

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

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

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
version = "1.11.0"

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
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

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
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "6258d453843c466d84c17a58732dda5deeb8d3af"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.24.0"
weakdeps = ["ConstructionBase", "ForwardDiff", "InverseFunctions", "Printf"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    PrintfExt = "Printf"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

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
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

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
git-tree-sha1 = "4bba74fa59ab0755167ad24f98800fe5d727175b"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.12.1+0"

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
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

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
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

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
# ╠═6389c2d9-fc3b-4a1e-a771-4735b4ff6039
# ╠═3aa144a9-c675-43c7-ac55-a74307e48088
# ╠═3e877d30-74a1-4abe-9d38-83ecf820860f
# ╠═6bf61e31-e899-47d7-a770-d649ff3198da
# ╠═309a7d48-9693-4924-9748-73cae12a8739
# ╠═e9e1887c-ced6-4af7-b5fb-e938836e5713
# ╠═c40c0622-e228-4f70-bbf7-2ea0e8289f6d
# ╠═3ae08b70-3b24-4f6e-8d0d-16642e3a8904
# ╠═c6c967db-2d63-4178-ad5b-d6ff4d454300
# ╠═299ae2ba-91f9-4e58-89a3-188c7cc7926a
# ╠═a0cc6059-1f5c-44d7-9592-321b9f526736
# ╠═23aca92f-9ae2-4c3e-9762-613f7ab6d147
# ╟─f2d0e8ec-4be5-4a55-8539-b65eadc61304
# ╟─f91ce131-6387-4646-ae91-c2ffa7320496
# ╟─c27a5a24-6173-4d1c-baca-41efbdd2b07c
# ╟─a8d2e703-d6ae-459e-9892-2086d2ba38a6
# ╠═49c2d27d-862b-4e4e-b150-fbaad14dc476
# ╠═b298eafd-a826-4570-ac92-e4f9270dc9b9
# ╠═21c8e2d0-fe6a-4e7b-b07f-d1c85a861a00
# ╠═76dbf7ae-910a-4649-ad21-7d3814236f36
# ╠═290b76eb-e8f5-45ee-9f20-4f4c34787256
# ╠═708bd44f-bd01-4cfa-8f9c-b6f80515764a
# ╠═ab139efa-c2fe-4e73-a663-3b7b6ff07cc5
# ╠═d393045c-e3be-447a-a172-340f29878d9b
# ╠═ed3baf6e-8d45-46b7-a873-3f24d351d4ca
# ╠═84b89600-4713-4f51-8d7c-161545994b1a
# ╠═cf22e4a9-c04c-4747-bf8b-61df14444e9c
# ╠═edab0641-596e-4e0f-a90c-0b1a75db3264
# ╠═c6eaee11-dca8-4b7a-a4e6-2fc7fced8df8
# ╠═b9df5ec6-e00a-4b21-85be-24304a10f47c
# ╠═a3951f0c-20d0-4abb-90fb-2f61a0b50247
# ╠═9510b505-6504-40fd-948d-d130c0f47de5
# ╟─dcbeee05-1abb-4657-80ac-67b1dc8478e8
# ╠═34141abb-2723-44e8-8ba9-9d2345921212
# ╠═111639a7-125f-4908-841c-b3e045a2b719
# ╠═a8718f51-c43d-4832-8b73-7ba7a15dae10
# ╠═47f76c94-8667-4cfb-ba20-63ce7f34b420
# ╠═a9504b67-8109-4f0c-94cc-d362b2ddc09e
# ╟─b42e1037-84b5-48d6-8211-3fadd8372f07
# ╠═753ce801-bec3-4b1e-8184-5a00bc5d41b2
# ╠═10c45b23-e437-421f-baa0-97e91597f125
# ╠═debd6eea-7f66-4f16-b3d8-4a60835d33a8
# ╠═1453a3be-eb38-4e4f-a10a-d0122b75260a
# ╠═ee67f8fe-afa0-4b29-8d73-af93071b379e
# ╠═aa6cc2fb-89ba-4d29-8db8-2a38761397fb
# ╠═239a48fe-5004-45f8-9b15-e691941ce26e
# ╠═b876f4a9-424b-495e-80be-9330ad7d960d
# ╠═2059113e-6e49-487a-a920-2e7d17fc8257
# ╠═0a3d9b84-59f8-426c-a848-a1a9a645cb51
# ╟─96a63de4-87da-4aa2-bbb4-c7f5de256977
# ╟─69ca7b92-9b5a-46b5-8343-e9c59d445f17
# ╟─25add3f6-e250-4729-8380-1dd80afa1074
# ╠═a5014d64-97d7-4309-bced-97911603ca91
# ╟─55d6648b-0534-4d2a-b904-8708bc8346f9
# ╠═b0e87883-bc55-45b3-a290-220b24ae2c16
# ╠═d92a8170-1505-4d02-bdb5-77e0f53fdda9
# ╠═c32bad5e-3b7d-41a4-a876-c6d25024b346
# ╟─2ee9ab01-5f41-40a9-a8ef-f24ca0deacb0
# ╟─facbc90c-b8e2-4204-9cbf-547ce4372d35
# ╠═7df0f1a9-1461-4cee-9692-7941d67790d4
# ╠═7152e4f2-7600-4e83-8dbd-fb3e88f33e90
# ╠═8e7bd67c-10b2-4f1e-a2a4-26ec41a65e0f
# ╠═0eafc438-9f61-4f82-a231-89ca63d212f4
# ╠═0d13aba8-24a1-4d18-9ab2-ac09e8c21bd5
# ╠═7013ac3f-d193-405e-b941-d9c62156f16e
# ╠═b51f0bc6-417d-42c7-97a1-e15d001ca5cb
# ╠═3b616808-0e94-4f34-81f2-7835775e1fdb
# ╠═5623a579-d066-4743-adc1-4bd73ce47588
# ╠═558a92af-b868-4e3a-8c00-a5db713bc467
# ╠═05993d47-932b-46a7-a39e-beeca0c872f2
# ╟─b42f3a3f-ee8a-4adb-8e02-f5b019501470
# ╠═e5b26e79-d213-4f9e-a87b-e737cd12bd51
# ╠═9fbade6d-505c-4511-b929-b2bf1e5ee0b6
# ╠═13bf8c89-c974-4211-a4fc-812caed060f0
# ╠═8957a247-8e3c-404d-aa77-8d338f852baf
# ╠═c9c5d463-c557-45a7-ada4-a3dbbef06556
# ╠═5bc11374-27c6-4b94-a1e6-0825d68ae8cc
# ╠═1bd5dba4-18ac-436a-b553-9b1dc6a46b1b
# ╠═e4b287cb-525f-42be-97db-32c66bd00916
# ╠═22bc4986-1285-448b-90b9-58ba971c653a
# ╠═a8309fe0-a12c-456e-a525-c327fbe7fec8
# ╠═54332ff2-a35a-4ad1-9248-89c912033eb2
# ╠═8c2762d3-b315-424d-9d74-84ae89875e65
# ╠═253a308f-9dfe-4820-afb5-3a3d5d3a3d05
# ╠═ef84e72b-aca2-49fc-af78-904545e7ebee
# ╠═d93e0243-acfc-4dc9-8ae3-598042c7281b
# ╠═e8c5c3aa-cda6-4e04-a678-378c394f728a
# ╠═d030efff-5f4b-4540-9e12-183d8809f1b4
# ╠═ee5344fc-9efb-4de0-809a-fded19a4f0aa
# ╠═b26db514-377c-4425-982c-7e5b8ee742bb
# ╠═33144d79-63e0-4a0e-a88d-18592f385ab2
# ╠═5fd01ce3-6bad-452b-bdf5-dbf0826beb79
# ╠═20b876fc-7f11-4b86-9098-cccc177a1e54
# ╟─d6fdb670-c877-48d6-b76f-89d5b49db6bb
# ╠═eb350120-a8e1-433b-a123-b9d33ce4da9f
# ╠═e51d1336-5d0a-465e-91fc-83181511ba3f
# ╠═2cd527b2-cf2a-48ce-98cf-40f5c8d5d6c7
# ╠═661934d6-6895-45d4-95f2-6a539211e348
# ╠═3bcc33e3-9947-4676-9a09-edec34bd5df4
# ╠═2817bb63-b9af-485a-85f7-62056c843cbc
# ╟─84a3e143-847e-47ff-9dfd-628064e848b2
# ╠═7ad1c088-ab3d-4363-912b-2cb8344f2478
# ╠═8499faf3-9ff9-4c75-abc3-deb89be6dea9
# ╠═1afb5a95-ec6d-45f2-b7ac-4f92cab1a5fb
# ╠═dc5e2ce1-6547-4e03-a45e-5b2c0ad0a04f
# ╠═43864981-5b6d-47b7-b995-36923c77f0ed
# ╠═b4bb0db5-33a9-4da9-ab3c-a2e99b79115d
# ╠═841f0c36-9686-44a7-85b2-d655034e4d1d
# ╠═62192287-bcb7-4bf6-a4d0-fec028379cb3
# ╠═d3e791a3-0995-41a2-9102-1158fd88f36e
# ╠═62cf81a9-d6cc-4b83-ad00-c6485ccd0616
# ╟─860ce00d-66af-4e64-940b-0c73de646c75
# ╠═6411c948-d120-4671-8eec-9327e1b2b746
# ╠═786dd9e4-91f9-4d1b-94e4-27522bb6fcba
# ╠═5187e754-d917-4659-a0c0-c60fead97491
# ╠═d28aab10-5650-41eb-bce5-85be9481d067
# ╠═a86b4795-0925-4651-9495-6f5c2c2cc60a
# ╠═324cb71a-eb73-45bd-8aad-431ebdbb4ea1
# ╠═98a4e14e-29e0-4007-862c-2dbeef19ae8a
# ╟─b0a30542-ea8e-498b-9691-169646ca272d
# ╠═0cd9df59-0085-434b-9e1c-a27ac6e40f0e
# ╠═4be7b686-18dc-4143-b176-024f34d3fd9a
# ╠═f953815f-bbff-450a-83da-5b3584163c33
# ╠═3a6bd594-5eca-41e6-9fa7-593d92f214e6
# ╠═303dda81-267e-4f99-be90-0a6df21a0add
# ╟─2c27773a-9fdc-4021-a26a-d5407a216342
# ╠═ce5ad54a-e77a-4faa-9897-c580e810e894
# ╠═0663df9f-81c6-4901-8d6d-568616b21dc7
# ╠═40817425-b996-4528-8935-5ddaf73dedf1
# ╟─336795f7-8db3-4f3a-b312-4f7081e9a81b
# ╠═24ff640f-10a1-47cd-9dd3-c41eec6bd55c
# ╠═09271053-7b1d-4cad-bc1a-efc1c3f352c5
# ╠═a308aeb5-f284-4bfb-86fb-e993070ec4a7
# ╠═a38c8b5a-cd56-47de-92cd-b9dc86912ae5
# ╠═cc32766e-93c0-4967-929c-22f5e77fe29a
# ╠═aa0e739d-6e1d-4448-a9ff-3967891d10c8
# ╠═934b2b49-8d6b-43af-a1c2-38673cd4eaaa
# ╠═27e12095-6f63-4ab1-b58d-387c5865b652
# ╠═b617ccb3-e30f-4dcc-9994-bad7014c6f0b
# ╟─15ad0df7-a1ce-43c5-91cd-c8355a6d0361
# ╠═7fd6c219-35e9-4fff-9270-6b67010f6d88
# ╠═c70b7252-06ab-4fc6-97f0-43e28d277d98
# ╠═97beacd0-7a2e-4754-8c3d-017901019e16
# ╠═ee0b2ae8-72e3-4ec5-be71-addd268bb720
# ╟─a05f69cc-1aca-4aea-adda-06f0d340e4b6
# ╠═fab8ab09-6814-4c3d-bf89-8158937a1a43
# ╠═4b00a4c5-9fea-4373-836f-56e6722c367f
# ╠═5dba456e-052d-4d31-9bf6-a10ddb7b4191
# ╠═348f4e16-2cc3-46be-a975-0e93abfff8b6
# ╠═9b9626b2-03f4-4399-9b75-0549c91c99c9
# ╠═8ddec5f5-2107-469e-8bba-923b77153a8e
# ╠═37201f37-e890-409a-bd72-008a71d3b518
# ╠═9d1092f7-cd5a-4efa-bbb4-004e78e1ff1a
# ╠═93b12ad3-6107-403f-a721-b0ff78b0e872
# ╠═f5a2d497-ffdf-4817-b089-286af625d827
# ╠═1ee457b2-02ea-4965-978a-967530047c3c
# ╠═2ffc1a78-1c5d-4f31-8bda-6c7ebad94961
# ╠═608dc256-3db7-4dfc-94e2-4c1f34ff7a86
# ╠═6c20f2dc-f919-4b55-9cf1-b183f53ae269
# ╠═d6fb71b2-7481-4405-affd-7a8fc0ed7cd8
# ╠═94f097a4-beb5-4ce5-bdc9-19c7369333a7
# ╠═b1426980-0fe7-4d3a-9ed2-fb39e477c2f1
# ╠═088841e8-8112-49ff-b056-ce8d5ebae311
# ╠═6acd7d7a-6b0d-45a8-adf6-27859ff6b6e5
# ╠═93a1ce7b-d275-4f97-b28c-4e86683b4930
# ╠═151df92a-cdff-4286-8aab-ed5fa5dd429a
# ╠═8bc72a0f-77c1-468a-aff4-d5ba0e80a27a
# ╠═a163eb1d-07aa-43ea-b067-d6e5c4db2131
# ╠═e1601860-602e-4218-a51c-c905bcc8e228
# ╠═a044470b-02db-4fa5-bbfc-510422e989cd
# ╠═22786293-06b8-4701-bbf6-a4a7be0f5a13
# ╠═842119b1-0044-4034-a2c5-118c351aa8b8
# ╠═4fbca4ab-9e43-4b5d-94b0-e1765ab11d82
# ╠═de8617b6-ff90-4de3-b332-beaeb5a5921d
# ╠═c77d4e45-53e1-403e-8eb6-6850a1c42c43
# ╠═dc95d4a1-fd37-4c53-a6ab-96ef0c6f1bd2
# ╠═0cc7187e-bf5c-4609-baeb-d986e5da7c52
# ╠═f063455e-694d-42d3-8b1a-b154a9fc8b7f
# ╠═384cc606-fdb4-4fe5-8721-89acf4fbe832
# ╠═b728930d-689c-4c4b-9e75-5d67763252d0
# ╟─873a8e67-f903-4c5d-b3a8-16a34ee5a2b2
# ╠═1538744e-d666-414f-afeb-d92f09de5594
# ╠═bd9cb9ce-e97e-4d6e-a2fa-ebaa1ce9fa41
# ╠═37bf32d9-a199-469e-8141-fa7409c22b38
# ╠═2b1fee28-1ddf-454a-81c6-f652c41a4cdd
# ╠═d757776b-6651-404e-81d6-d5d91cd88ac9
# ╠═074b8836-fd0e-4920-8a4e-44cac30c0446
# ╠═a994c130-e869-4393-bc45-fff782a4a04f
# ╠═c49c5989-6855-4b2b-a5a6-c31f3c92224d
# ╠═2378c45e-2f5a-4046-8c2d-096cdee9542f
# ╠═1733f2bf-4a4e-49f8-8f89-9f74103759a5
# ╠═9d9c3fb1-a6a8-4f12-bf83-53033e9a33e4
# ╠═03d376b5-a7e5-4641-893a-b5e6670aa201
# ╠═0dabf828-554c-49de-b6d8-ff0e5561b3c2
# ╠═a7a4bc12-e8c6-4f7e-bd20-5fb97219f7ef
# ╠═f957f853-b32b-4a67-a7d0-3cc62cda70da
# ╠═6320fc56-94af-444f-807d-5a1b86113e16
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
