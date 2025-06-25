### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 8daf8548-2245-4271-80f7-d8a352687260
using QuantEcon: QuantEcon, rouwenhorst, MarkovChain, stationary_distributions

# ╔═╡ 45caaf75-c1bb-40d0-b4e7-5039d1ba3ba4
using Statistics: mean

# ╔═╡ 3128e1c7-2e13-4bd5-b6b7-9d3b90a126b3
using StatsBase: weights

# ╔═╡ c9da52e2-098a-46c4-acbf-173fce6965d8
using PlutoUI

# ╔═╡ acd91d10-53a9-44f2-8e91-ce058a30535b
using Chain, DataFrames, DataFrameMacros

# ╔═╡ 33b42231-663b-4e3f-ba4c-18f0a1ce24ea
using CairoMakie, AlgebraOfGraphics

# ╔═╡ 4741eb11-c44c-4141-bb02-269ff2b78ad2
using PlutoTest

# ╔═╡ d7bd570b-3e14-476c-8337-7edbab49980a
using DimensionalData

# ╔═╡ 91ddd974-9d67-4cd3-b4f8-60521c37baaa
using LinearAlgebra: dot

# ╔═╡ 2db71af0-fcc7-4822-a6cc-bbcf07107182
using Roots: find_zero

# ╔═╡ fd369634-4ddb-40cf-a1a0-e7e6282b7595
using Interpolations

# ╔═╡ 10d34c05-63e9-444e-aff8-bc7aa82858eb
md"""
# With Risk
"""

# ╔═╡ c9106199-39e7-4af5-bd33-16923e5b28ff


# ╔═╡ a2b8696d-ff6c-42e2-9c76-5dd45a882487
function Household(; σ = 1.0, β = 0.96,	
                      u = σ == 1 ? log : x -> (x^(1 - σ) - 1) / (1 - σ))
	(; β, u, σ)
end

# ╔═╡ 46170d0a-d23b-4e7e-bf34-bb181db8dd20
function exponential_grid(amin,amax,na)
	return exp.(range(0.0, stop=log(amax-amin+1.0), length = na)) .+ amin .- 1.0
end

# ╔═╡ 68c038ec-c197-4044-8b3a-26974f284e8b
function Statespace(; amin, amax, na, y_chain, exponential = true)
	if exponential
		a_grid = exponential_grid(amin, amax, na)
	else
		a_grid = range(amin, amax, length=na)
	end
	y_grid = y_chain.state_values
	
	a_dim = Dim{:a}(a_grid)
	y_dim = Dim{:y}(y_grid)
	dims = (a_dim, y_dim)
	
	(; a_grid = DimVector(a_grid, a_dim, name = :a),
	   y_grid = DimVector(y_grid, y_dim, name = :y), y_chain, dims)
end

# ╔═╡ d82f2dcd-6e81-459b-bd53-a31194176b7b
function log_rouwenhorst(args...; normalize_mean = true)
	mc₀ = rouwenhorst(args...)

	state_values 	= exp.(mc₀.state_values)

	π∞ = QuantEcon.stationary_distributions(mc₀) |> only

	if normalize_mean
		𝔼y = mean(state_values, weights(π∞))
		state_values ./= 𝔼y
	end
	
	return MarkovChain(mc₀.p, state_values)
end

# ╔═╡ 70fc73c5-4154-4a04-99fa-ce084d42ea93
function default_income_process(;
	ρ = 0.966, 	          #persistence of HH income process
    σ = (1.0-ρ^2)^0.5 * 0.5, #variance for household income process
	n = 7,
	normalize_mean = true
)
	log_rouwenhorst(n, ρ, σ; normalize_mean)
end

# ╔═╡ fc1fa5b1-4868-4951-ba14-14d07ccf2ee7
md"""
# Adapting the code
"""

# ╔═╡ 980ec36d-26b7-49b6-bafd-c31ea3a1befe
let
	statespace = Statespace(; amin = 0.0, amax = 150.0, na = 10, y_chain = default_income_process(normalize_mean = true, n = 6))

end

# ╔═╡ cac697fa-cde9-4b45-ae7a-e582388e491c
c_prev_risk(c, r, (; β, γ)) = c / (β * (1 + r))^(1/γ)

# ╔═╡ 7ff370ce-53e4-40d8-90e0-6712c844dc41
a_prev_risk(c_prev, a, y_prev, (; w_prev, r_prev, m_prev)) = (a * (1-m_prev) + c_prev - w_prev * y_prev)/(1+r_prev)

# ╔═╡ 66bc077a-8843-448d-b883-03619b21d6ff
c_curr_risk(a, a_next, y, (; w, r, m)) = (1+r) * a + y * w - (1-m) * a_next

# ╔═╡ c09a6bd6-9e5f-4265-9a52-71ea586a385d
# ╠═╡ disabled = true
#=╠═╡
function a_prev_c_prev_risk(cⱼ, aⱼ, yⱼ, par; r, r_prev, w_prev, m_prev)
	cⱼ₋₁ = c_prev_risk(cⱼ, r, par)
	aⱼ₋₁ = a_prev_risk(cⱼ₋₁, aⱼ; y_prev, 
				w_prev, r_prev, m_prev)
		
	(; cⱼ₋₁, aⱼ₋₁)
end
  ╠═╡ =#

# ╔═╡ 64055da7-95a9-4920-abcb-c88f2add29c1
c_J(a, y, (; r, w)) = (1 + r) * a + w * y

# ╔═╡ 0680a3c4-c437-4d67-bacb-49c34fdded1c
md"""
# Test
"""

# ╔═╡ 14c512b8-2f0e-4831-a1b4-4e66f3e061f5
md"""
### Budget constraint, parameters
"""

# ╔═╡ 91bbe376-ced2-4b88-b564-b72d7bf0900f
begin
	abstract type SolutionMethod end
	struct VFI <: SolutionMethod end
	struct EGM <: SolutionMethod end
end

# ╔═╡ db4b45b9-fc4a-457e-bfa2-9778d2002413
c(a, a_next; m, y, r, w) = w * y + (1+r) * a - a_next * (1-m)

# ╔═╡ d37ead9d-23e3-4492-863e-17c39c068f1c
"type of policy_plus named tuple"
PPT() = typeof(
	(a_i = 1, a_i_next = 1, c = 1.0, a = 1.0, a_next = 1.0,
	m = 1.0, y = 1.0, r = 1.0, w = 1.0, j = 1, t = 1)
)

# ╔═╡ ba2cd4ef-3acf-4d87-a7db-27c1bdf0994e
md"""
### Demographics
"""

# ╔═╡ 44dde0b2-ad39-4558-b260-41590796b2a7
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

# ╔═╡ 0e58d89d-0ce7-4ec4-b89d-e4db8fb03b3f
p_surv₀ = DimVector(
	[0.9945385, 0.9995935, 0.9997525, 0.999799, 0.999836, 0.9998605, 0.9998755, 0.999885, 0.99989, 0.99989, 0.999884, 0.9998745, 0.9998525, 0.9998115, 0.99975, 0.9996605, 0.999536, 0.9993885, 0.999241, 0.9991345, 0.99906, 0.998978, 0.9988925, 0.99881, 0.9987215, 0.998631, 0.9985435, 0.9984545, 0.998359, 0.998259, 0.998161, 0.9980625, 0.997962, 0.997868, 0.9977805, 0.997691, 0.9975995, 0.997493, 0.997366, 0.997226, 0.997077, 0.99692, 0.9967525, 0.9965905, 0.996419, 0.9962185, 0.995971, 0.995691, 0.9953685, 0.9950285, 0.994659, 0.9942635, 0.993815, 0.993334, 0.9927975, 0.9921995, 0.9915535, 0.9908825, 0.990155, 0.9893715, 0.988523, 0.987618, 0.9866885, 0.985767, 0.9848455, 0.983935, 0.982972, 0.9818665, 0.980645, 0.9793075, 0.9778105, 0.9760855, 0.9741015, 0.971813, 0.9691475, 0.9657655, 0.9623835000000001, 0.958681, 0.9545755, 0.9497485, 0.9445295, 0.9388595, 0.9326274999999999, 0.9255175, 0.9172435, 0.907863, 0.8973555, 0.885727, 0.873394, 0.859642, 0.844195, 0.827184, 0.8087880000000001, 0.78986, 0.7707634999999999, 0.7515835, 0.732595, 0.7140934999999999, 0.6963895, 0.6798, 0.662296, 0.6438275, 0.6243405, 0.6037785, 0.5820815, 0.559187, 0.5350275, 0.509533, 0.4826284999999999, 0.454237, 0.42427349999999997, 0.39265149999999993, 0.35927850000000006, 0.32405700000000004, 0.28970799999999997, 0.25419400000000003, 0.21690299999999996, 0.17774900000000005, 0.13663599999999998, 0.093468, 1.0],
	Dim{:j}(0:120)
)

# ╔═╡ 046d2a75-c05f-4c27-95c7-e960cb14f494
function mortality(model; m = 1/45, J = 120)
	
	j₀ = 0

	j_dim = Dim{:j}(j₀:J)
	
	if model == :perpetual_youth
		return DimVector([fill(m, J); 1.0], j_dim)
	elseif model == :lifecycle
		return 1 .- p_surv₀[j = At(j₀:J)]
	else
		@error "model ∉ [:perpetual_youth, :lifecycle]. Please fix!"
	end
	
end

# ╔═╡ 857e9793-6cc8-4c29-935c-493d88cc5da7
md"""
### Income profile
"""

# ╔═╡ 09c2d6c5-1268-44f7-89af-628ecba011a5
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

# ╔═╡ 4c27296c-aa27-4a48-816e-5d10ecceecf8
function simple_income_profile(J, JR; y=1.0, yR = 0.0)
	y = [fill(y, JR); fill(yR, J - JR)]

	DimArray(y, Dim{:j}(0:J-1), name = :y)
end

# ╔═╡ 2e34251a-e8e7-4fcf-9ac0-00509b5ae687
simple_income_profile(100, 50) |> lines

# ╔═╡ 83509c7d-2f15-4302-95ec-77b23f885bc2
income_profile(120, 41) |> lines

# ╔═╡ c5d76211-e9aa-4040-84f5-7417465d585f
md"""
### Equilibrium
"""

# ╔═╡ 46068b3b-fad4-4ba7-97bb-8c0e43589a96
wage(K, (; L, α, Θ, δ))          = K ≤ 0.0 ? NaN : (1-α) * Θ * (K/L)^(α)

# ╔═╡ 42ca4fde-502d-489f-979b-1213584bb72d
interest_rate(K, (; L, α, Θ, δ)) = K ≤ 0.0 ? NaN : α * Θ * (K/L)^(α-1) - δ

# ╔═╡ fd045d53-7d11-4dac-824c-0e382b80d4f5
inverse_interest_rate(r, (; L, α, Θ, δ)) = ((r + δ)/(α * Θ))^(1/(α - 1)) * L

# ╔═╡ 94c3c00a-389a-4cc8-97aa-c51339a3ef37
output(K, (; L, α, Θ, δ))           = K ≤ 0.0 ? NaN : Θ * K^α * L^(1-α)

# ╔═╡ 76a2bff9-6036-49a8-930a-551a23cd6f32
function aggregate(sim_df, pmf)
	agg_nt = @chain sim_df begin
		@select(#= :c, =# :a, :a_next, :j)
		#@select({Not(:m)})
		stack(Not(:j))
		leftjoin(DataFrame(pmf), on = :j)
		disallowmissing!
		@groupby(:variable)
		@combine(:value = dot(:value, :pmf))
		(; (Symbol.(_.variable) .=> _.value)...)
	end
end

# ╔═╡ df140406-28a5-49cc-b5c1-ecc53819802c
md"""
### Value function iteration
"""

# ╔═╡ 4e661c4d-41df-4b63-b188-2a52aa62be2d
function iterate_back!(v_curr, policy_curr, policy_plus_curr, a_grid, par, v_next;
						m, y, r, w, j, t)
	(; β, u) = par
	
	for (i, a) ∈ enumerate(a_grid)
		# for a given grid point a, find optimal choice
		(v_opt, a_i_opt) = findmax(
					(u ∘ c).(a, a_grid; m, y, r, w) + β * (1-m) .* v_next
				)

		# save interesting output
		v_curr[a = i]           = v_opt
		policy_curr[a = i]      = a_i_opt # a_next_index
		a_next = a_grid[a_i_opt]

		if !isnothing(policy_plus_curr)
			policy_plus_curr[a = i] = (;
						a_i = i,
						a_i_next = a_i_opt,
						c = c(a, a_next; m, y, r, w),
						a,
			            a_next,
						m, y, r, w, j, t
			)
		end

	end
end

# ╔═╡ dd7c5948-b335-44ea-909d-bcfd8be6b40e
md"""
### Endogenous grid method
"""

# ╔═╡ b04af990-5fb4-4877-9e6b-66fd6847e4d2
"Takes `price_paths`"
function solve_backward_forward_t(::EGM, par, grid; price_paths, init_state, j_init = 0, t_born = 0)
	(; statename) = Mo

	(; J, y, m) = par

	state_dim = :a # a or net worth
	j_dim = Dim{:j}(0:J)
	j_sim_dim = Dim{:j}(j_init:J)
	
	c = zeros(state_dim, j_dim, name = :c)
	
	## SOLVE BACKWARDS
	t_J = t_born + J
	prices_J = tuple_of_prices(Mo, price_paths, par; t=t_J, j=J)
	c[j = At(J)] .= c_J.(Ref(Mo), grid, Ref(prices_J), Ref(par))
	
	for j ∈ J:-1:j_init+1
		t = t_born + j
		prices = tuple_of_prices(Mo, price_paths, par; t, j)
		
		(; cⱼ₋₁, stateⱼ₋₁) = DataFrame(
			iterate_backward.(Ref(Mo), grid, c[j = At(j)], Ref(prices), Ref(par))
		)
			
		cⱼ₋₁_itp = LinearInterpolation(stateⱼ₋₁, cⱼ₋₁, extrapolation_bc = Line())
	
		c[j = At(j-1)] .= cⱼ₋₁_itp.(grid)
	end

	## SOLVE FORWARD
	nextstatename = Symbol(string(statename) * "_next") # e.g. a_next
	
	path_state      = zeros(j_dim, name = statename) # e.g. a
	path_next_state = zeros(j_dim, name = nextstatename) # e.g. a_next
	path_choice      = zeros(j_dim, name = :c)
		
	path_state[j = At(j_init)] = init_state

	TT = TOther(Mo)
	other_paths = DimVector(
					Vector{TT}(undef, J+1),
					j_dim, name = :other
				)
	
	(; r, w) = price_paths
	for j ∈ j_init:J
		t = t_born + j
		prices = tuple_of_prices(Mo, price_paths, par; t, j)
			
		stateⱼ = path_state[j = At(j)]
			
		cⱼ_itp = LinearInterpolation(grid, c[j = At(j)], extrapolation_bc = Line())
			
		cⱼ = cⱼ_itp(stateⱼ)

		(; cⱼ, stateⱼ₊₁, other) = iterate_forward(Mo, prices, par; cⱼ, stateⱼ)
		
		path_choice[j = At(j)] = cⱼ
		other_paths[ j = At(j)] = other
		path_next_state[j = At(j)] = stateⱼ₊₁
		
		if j < J
			path_state[j = At(j+1)] = stateⱼ₊₁
		end	
	end

	sim = DimStack(path_state, path_choice, path_next_state, other_paths)
	sim_df = DataFrame(sim)

	select!(sim_df, :other => AsTable, Not(:other))

	(; sim_df, sim, state_path=path_state, other_path=other_paths, c)
end

# ╔═╡ 21d8f2f0-8368-48dc-a625-eeb58d470bf1
function XXsolve_backward_forward(Me, par, grid; prices, init_state)
	(; J) = par

	price_paths = constant_price_paths(Mo, par, prices)

	solve_backward_forward_t(Mo, Me, par, grid; price_paths, init_state)
end

# ╔═╡ 072d96b3-75b3-4291-9220-219bdd6fe34f
a_next(c; a, w, y, r, m) = m == 1 ? 0.0 : ((1+r) * a + y * w - c)/(1-m)

# ╔═╡ 23f7eab4-70e7-4c35-8124-4cfb85095f95
c_curr(a, a_next; w, y, r, m) = (1+r) * a + y * w - (1-m) * a_next

# ╔═╡ 1fd2c215-37dd-42bb-81a5-a4e229a50580
c_prev(c, r, (; β, γ)) = c / (β * (1 + r))^(1/γ)

# ╔═╡ ee932ecb-1537-4bd2-a8c8-7b16c42e73e2
a_prev(c_prev, a; w_prev, y_prev, r_prev, m_prev) = (a * (1-m_prev) + c_prev - w_prev * y_prev)/(1+r_prev)

# ╔═╡ 16e4f838-00cc-46dc-b211-f2f492b67285
function a_prev_c_prev(cⱼ, aⱼ, par; r, r_prev, y_prev, w_prev, m_prev)
	cⱼ₋₁ = c_prev(cⱼ, r, par)
	aⱼ₋₁ = a_prev(cⱼ₋₁, aⱼ; y_prev, 
				w_prev, r_prev, m_prev)
		
	(; cⱼ₋₁, aⱼ₋₁)
end

# ╔═╡ 84e11eea-9626-4944-933e-226f29749da9
function _solve_backward_forward_(::EGM, par, grid; price_paths, init_state, j_init = 0, t_born = 0)
	(; J, y, m, a̲) = par
	(; rs, ws) = price_paths
	a_init = init_state
	
	a_dim = Dim{:a}(grid)
	j_dim = Dim{:j}(0:J)
	j_sim_dim = Dim{:j}(j_init:J)
	
	c = zeros(a_dim, j_dim, name = :c)
	
	## SOLVE BACKWARDS
	t_J = t_born + J
	c[j = At(J)] .= (1+rs[t=At(t_J)]) * grid .+ ws[t = At(t_J)] * y[j = At(J)]
	
	for j ∈ J:-1:1
		t = t_born + j
		
		cⱼ   = c[j = At(j)]

		cⱼ₋₁_aⱼ₋₁_df = a_prev_c_prev.(cⱼ, grid, Ref(par); 
				r = rs[t = At(t)], r_prev=rs[t = At(t-1)], w_prev=ws[t = At(t-1)],
				y_prev=y[j = At(j-1)], m_prev=m[j = At(j-1)]) |> DataFrame

		df₀ = cⱼ₋₁_aⱼ₋₁_df

		(; cⱼ₋₁, aⱼ₋₁) = df₀
			
		cⱼ₋₁_itp = LinearInterpolation(aⱼ₋₁, cⱼ₋₁, extrapolation_bc = Line())
	
		c[j = At(j-1)] .= cⱼ₋₁_itp.(grid)
	end

	## SOLVE FORWARD
	a_sim      = zeros(j_dim, name = :a)
	a_next_sim = zeros(j_dim, name = :a_next)
	c_sim      = zeros(j_dim, name = :c)
		
	a_sim[j = At(j_init)] = a_init
	
	for j ∈ j_init:J
		t = t_born + j
		aⱼ = a_sim[j = At(j)]
			
		cⱼ_itp = LinearInterpolation(grid, c[j = At(j)], extrapolation_bc = Line())
			
		cⱼ = cⱼ_itp(aⱼ)
		
		aⱼ₊₁ = a_next(cⱼ; a=aⱼ, w=ws[t = At(t)], y=y[j=At(j)], r=rs[t = At(t)], m=m[j=At(j)])

		#@info (; j, aⱼ₊₁)
		# handling the constraint
		if j == J
			a̲ = 0.0
		end
		if aⱼ₊₁ < a̲
			aⱼ₊₁ = a̲
			cⱼ = c_curr(aⱼ, aⱼ₊₁; w=ws[t = At(t)], y=y[j=At(j)], r=rs[t = At(t)], m=m[j=At(j)])
		end

		c_sim[j = At(j)] = cⱼ
		a_next_sim[j = At(j)] = aⱼ₊₁
		
		if j < J
			a_sim[j = At(j+1)] = aⱼ₊₁
		end	
	end

	sim_df = DataFrame(DimStack(a_sim, c_sim, a_next_sim))

	(; sim_df, path_state = a_sim, c)
end

# ╔═╡ 1748ac1d-5f4b-417b-9b11-d5bbbcb2c072
function solve_backward_forward(M::EGM, par, a_grid; r, w, a_init)
	(; J) = par
	
	rs = DimArray(fill(r, J+1), Dim{:t}(0:J))
	ws = DimArray(fill(w, J+1), Dim{:t}(0:J))

	price_paths = (; rs, ws)
	grid = a_grid
	init_state = a_init
	
	_solve_backward_forward_(M, par, grid; price_paths, init_state)
end

# ╔═╡ 2caa6111-33bf-4e90-b801-7b4655020444
md"""
# Appendix
"""

# ╔═╡ c45dcd06-b3a1-4045-89d4-b5632b988cf8
TableOfContents()

# ╔═╡ fbd9c561-24c8-4f76-9235-f59593cc298a
criterion(a, b) = (a - b)/(1 + max(abs(a), abs(b)))

# ╔═╡ 03d38330-6f14-46b5-86fd-620072ff102d
const DD = DimensionalData

# ╔═╡ 48a39844-3ec1-4300-9ee5-ade8c9a5c1d0
let
	statespace = Statespace(; amin = 0.0, amax = 150.0, na = 10, y_chain = default_income_process(normalize_mean = true, n = 4), exponential = false)
	household = Household(; β = 0.98, σ = 2.0)

	prices = (; r = 0.018, w = 1.0)

	(; dims) = statespace

	#DD.dims(dims, :a)
	#tmp = zeros(dims, name = :value)

	a_grid = DimVector(statespace.a_grid, DD.dims(dims, :a))
	y_grid = DimVector(statespace.y_grid, DD.dims(dims, :y))

	cⱼ = @d (1 + prices.r) .* a_grid .+ prices.w .* y_grid
	cⱼ = DimArray(cⱼ, name = :cⱼ)
	
	(; σ) = household

	P_dims = (DD.dims(dims, :y), DD.dims(dims, :y))
	P = DimArray(statespace.y_chain.p, P_dims)

	
	
	cⱼ₋₁ = let
		(; σ, β) = household
		(; r) = prices

		𝔼u′ = cⱼ .^ (-σ) * P'
	    
	    cⱼ₋₁ =  𝔼u′.^(-1/σ) * (β*(1+r)).^(-1/σ)
		cⱼ₋₁ = DimArray(cⱼ₋₁, name = :cⱼ₋₁)
	end

	w_prev = prices.w
	r_prev = prices.r

	aⱼ₋₁ = DimArray(
			@d((a_grid .+ cⱼ₋₁ .- w_prev .* y_grid ) ./ (1+r_prev)),
			name = :aⱼ₋₁
	)

	aⱼ₋₁_interpolated = similar(aⱼ₋₁)
	
	a_dim = collect(DD.dims(aⱼ₋₁, :a))
	for y ∈ DD.dims(aⱼ₋₁_interpolated, :y)
		
		itp = linear_interpolation(
			collect(
				aⱼ₋₁[y=At(y)]), 
				a_dim,
				extrapolation_bc=Line()
		)

		aⱼ₋₁_interpolated[y = At(y)] = itp(a_grid)
	end

	aⱼ₋₁_interpolated

	
end

# ╔═╡ ba932291-dd10-43a1-9f86-8659a66365f5
function _solve_backward_forward_risk_(::EGM, par, statespace; price_paths, init_state, j_init = 0, t_born = 0)
	(; J, y, m, a̲) = par
	(; rs, ws) = price_paths
	
	a_init = init_state

	#a_dim = statespace.dims
	(; a_grid, y_grid, dims) = statespace
	grid = a_grid
	a_min = 0.0
	
	j_dim = Dim{:j}(0:J)
	j_sim_dim = Dim{:j}(j_init:J)
	
	c_x    = zeros(dims..., j_dim, name = :c_x)
	c      = zeros(dims..., j_dim, name = :c)
	a_next = zeros(dims..., j_dim, name = :a_next)
	
	## SOLVE BACKWARDS ("solve policy functions")
	t_J = t_born + J
	prices_J = (; r = rs[t=At(t_J)], w = ws[t = At(t_J)])
	c_J_tmp = @d (1 + prices_J.r) .* a_grid .+ prices_J.w .* y_grid
	
	     c[j = At(J)] .= c_J_tmp # CHECK CORRECT DIMENSIONS ???
	   c_x[j = At(J)] .= c_J_tmp # CHECK CORRECT DIMENSIONS ???
	a_next[j = At(J)] .= 0.0
	
	P_dims = (DD.dims(dims, :y), DD.dims(dims, :y))
	P = DimArray(statespace.y_chain.p, P_dims)

	for j ∈ J:-1:1
		t = t_born + j
		cⱼ   = c[j = At(j)]

		prices      = (; r = rs[t=At(t)],   w = ws[t = At(t)],   m = m[j = At(j)])
		prices_prev = (; r_prev = rs[t = At(t-1)], 
					   	 w_prev = ws[t = At(t-1)],
					     m_prev =  m[j = At(j-1)]
					  )
		
		cⱼ₋₁ = let
			(; γ, β) = par
			(; r) = prices
	
			𝔼u′ = cⱼ .^ (-γ) * P'
		    
		    cⱼ₋₁ =  𝔼u′.^(-1/γ) * (β*(1+r)).^(-1/γ)
			cⱼ₋₁ = DimArray(cⱼ₋₁, name = :cⱼ₋₁)
		end
		
		aⱼ₋₁ = @d a_prev_risk.(cⱼ₋₁, a_grid, y_grid, Ref(prices_prev))
	
		for y ∈ DD.dims(dims, :y)
	
			cⱼ₋₁_itp = LinearInterpolation(
				aⱼ₋₁[y = At(y)], 
				cⱼ₋₁[y = At(y)],
				extrapolation_bc = Line()
			)

			aⱼ_itp = LinearInterpolation(
				aⱼ₋₁[y = At(y)], 
				parent(a_grid),
				extrapolation_bc = Line()
			)
	
			   c_x[j = At(j-1), y = At(y)] .= cⱼ₋₁_itp.(a_grid)
			a_next[j = At(j-1), y = At(y)] .= #max.(
				aⱼ_itp.(a_grid)#, a_min)
		end

		c[j = At(j-1)] = @d c_curr_risk.(a_grid, a_next[j = At(j-1)], y_grid, Ref((; r = prices_prev.r_prev, w = prices_prev.w_prev, m = prices_prev.m_prev)))
	end

	
	
	return (; c, c_x, a_next) #[j = At(J-1)]
	

	
	# =#


	############
	for j ∈ J:-1:1
		t = t_born + j
		
		cⱼ   = c[j = At(j)]

		cⱼ₋₁_aⱼ₋₁_df = 0.0 #a_prev_c_prev_risk.(cⱼ, grid, Ref(par); 
				#r = rs[t = At(t)], r_prev=rs[t = At(t-1)], w_prev=ws[t = At(t-1)],
				#y_prev=y[j = At(j-1)], m_prev=m[j = At(j-1)]) |> DataFrame

		df₀ = cⱼ₋₁_aⱼ₋₁_df

		(; cⱼ₋₁, aⱼ₋₁) = df₀
			
		cⱼ₋₁_itp = LinearInterpolation(aⱼ₋₁, cⱼ₋₁, extrapolation_bc = Line())
	
		c[j = At(j-1)] .= cⱼ₋₁_itp.(grid)
	end

	return c
	
	## SOLVE FORWARD "simulate"
	a_sim      = zeros(j_dim, name = :a)
	a_next_sim = zeros(j_dim, name = :a_next)
	c_sim      = zeros(j_dim, name = :c)
		
	a_sim[j = At(j_init)] = a_init
	
	for j ∈ j_init:J
		t = t_born + j
		aⱼ = a_sim[j = At(j)]
			
		cⱼ_itp = LinearInterpolation(grid, c[j = At(j)], extrapolation_bc = Line())
			
		cⱼ = cⱼ_itp(aⱼ)
		
		aⱼ₊₁ = a_next(cⱼ; a=aⱼ, w=ws[t = At(t)], y=y[j=At(j)], r=rs[t = At(t)], m=m[j=At(j)])

		#@info (; j, aⱼ₊₁)
		# handling the constraint
		if j == J
			a̲ = 0.0
		end
		if aⱼ₊₁ < a̲
			aⱼ₊₁ = a̲
			cⱼ = c_curr(aⱼ, aⱼ₊₁; w=ws[t = At(t)], y=y[j=At(j)], r=rs[t = At(t)], m=m[j=At(j)])
		end

		c_sim[j = At(j)] = cⱼ
		a_next_sim[j = At(j)] = aⱼ₊₁
		
		if j < J
			a_sim[j = At(j+1)] = aⱼ₊₁
		end	
	end

	sim_df = DataFrame(DimStack(a_sim, c_sim, a_next_sim))

	(; sim_df, path_state = a_sim, c)
end

# ╔═╡ 5a947389-9c8c-4ac3-b162-b93a30479d1f
function solve_backward_forward_risk(M::EGM, par, statespace; r, w, init_state)
	(; J) = par
	
	rs = DimArray(fill(r, J+1), Dim{:t}(0:J))
	ws = DimArray(fill(w, J+1), Dim{:t}(0:J))

	price_paths = (; rs, ws)
	
	_solve_backward_forward_risk_(M, par, statespace; price_paths, init_state)
end

# ╔═╡ ae0252ca-4a39-4581-9fd4-b99f635a6ca8
function partial_equilibrium_risk(par, statespace, (; K_guess, r, w); 
								  pmf = pmf(par.m), 
								  details = true, return_df = false,
								  init_state = statespace.a_grid[1],
								  solution_method = EGM()
								 )
	(; bonds2GDP) = par

	sol = solve_backward_forward_risk(
		solution_method, par, statespace; r, w, init_state
	)

	#=
	(; sim_df) = sol

	agg_nt = aggregate(sim_df, pmf)

	GDP = output(K_guess, par)
	B₀ = bonds2GDP * GDP
	
	K_hh = agg_nt.a
	
	K_supply = K_hh - B₀
	ζ = K_supply - K_guess

	if details
		#= @info @chain sim_df begin
			leftjoin(_, DataFrame(pmf), on = :j)
			stack([:a, :a_next, #= :c, :y, :m,=# :pmf], :j)
			data(_) * mapping(:j, :value, layout = :variable) * visual(Lines)
			draw(facet = (; linkyaxes = false))
		end
		=#

		#@info NamedTuple{(:K_guess, :r, :w)}(round.((K_guess, r, w), sigdigits = 4)) |> repr
		if return_df
			return (; ζ, K_guess, r, w, K_hh, K_supply, B₀, par, sol, pmf_df= DataFrame(pmf))
		else
			return (; ζ, K_guess, r, w, K_hh, K_supply, B₀, par, sol)
		end
	else
		return ζ
	end
	=#
end

# ╔═╡ 4e6216db-e6fe-4bdc-9bd4-d22e9edc532a
function get_par(; 
		demo = :perpetual_youth,
		mm = 1/45,
		y,
		γ = 2.0,
		J = length(y) - 1,
		β = 0.995,
		ρ = 1/β - 1,
		r = ρ,
        α = 0.33,
		δ = 0.1,
 		a̲ = -Inf,
		bonds2GDP = 1.0
	)
	m = mortality(demo; J, m=mm)
	J = maximum(DD.dims(m, :j))

	u(c) = c > 0 ? c^(1-γ)/(1-γ) : -Inf
		
	(; δ, α, Θ = 1, L = 1, β, ρ, r, bonds2GDP, 
		m, J, γ, a̲,
		y, u,
		w = 1.0)
end

# ╔═╡ 46964708-1e84-4e05-bf2d-440efa54efe8
let
	y = income_profile(120, 41)
	
	par = get_par(; demo = :lifecycle, y, a̲ = 0.0)

	statespace = Statespace(; amin = 0.0, amax = 150.0, na = 100, y_chain = default_income_process(normalize_mean = true, n = 6))
	
	#statespace = let
	#	a_grid = statespace.a_grid
	#	a_grid = range(0.0, 12.0, length = 600)
	#	(; a_grid, dims = (Dim{:a}(a_grid), ))
	#end
	
	K_guess = 4.522301994771901
	r = interest_rate(K_guess, par) 
	w = wage(K_guess, par)

	
	(; c, c_x, a_next) = partial_equilibrium_risk(par, statespace, (; K_guess, r, w), return_df = true)

	@chain DimStack(a_next, c, c_x) begin
		DataFrame
		@transform(:Δ = :c_x - :c)
		@subset(:j < 10, :a < 5)
		data(_) * mapping(
				:a, :Δ,
				group = :j => nonnumeric, color = :j,
				layout = :y => nonnumeric
		) * visual(Lines)
		draw(facet = (; linkyaxes = false))
	end
	#@info @test out.K_hh ≈ 6.169575851057367
	
	#(; sol, K_hh) = out
end

# ╔═╡ 13173588-b32f-48cd-8234-6a45673bfd01
function solve_backward(par, a_grid; r, w, t_born = 0, minimal = false)
	(; y, β, m) = par

	# j_dim
	j_dim = DD.dims(m, :j)
	J = maximum(j_dim)
	@assert collect(j_dim) == 0:J

	# t_dim
	let
		t_dim = DD.dims(r, :t)
		@assert t_dim == DD.dims(w, :t)
		T₀, T₁ = extrema(t_dim)
		@assert t_born     ≥ T₀
		@assert t_born + J ≤ T₁
	end
	
	#r = r[t = Begin]
	dim_a = Dim{:a}(a_grid)
	dims = (dim_a, j_dim)
	
	value       = zeros(     dims, name = :value)
	policy      = zeros(Int, dims, name = :policy)

	if !minimal
		policy_plus = DimArray(Matrix{PPT()}(undef, size(policy)), dims, name = :policy_plus)
	else
		policy_plus = nothing
	end

	grid = DimArray(a_grid, dim_a)
	
	value[j = At(J)] .= 0.0
	
	for j ∈ reverse(j_dim)
		t = t_born + j
		
		v_curr = @view value[j = At(j)]
		if j == J
			v_next = 0.0 * v_curr
		else
			v_next = @view value[j = At(j+1)]
		end

		policy_curr      = @view      policy[j = At(j)]

		if !minimal
			policy_plus_curr = @view policy_plus[j = At(j)]
		else
			policy_plus_curr = nothing
		end
		
		out = iterate_back!(
			v_curr, policy_curr, policy_plus_curr,
			grid, par, v_next; 
			m=m[j = At(j)],
			y=y[j = At(j)],
			w=w[t = At(t)],
			r=r[t = At(t)], j, t)
	end

	(; value, policy, policy_plus)
	
	#outs
end

# ╔═╡ 2ae464c4-a641-43af-8adc-066b9c0be788
function solve_forward(out, par, a_grid; a_i_init, j_init)
	(; value, policy, policy_plus) = out

	dim_j = DD.dims(policy, :j)

	# state
	T = typeof((; j=0, a_i = 1, a_i_next=1))

	path_state       = zeros(Int, dim_j)
	path_choice      = zeros(Int, dim_j)
	path_choice_nt = T[]
	
	for j ∈ j_init:par.J
		if j == 0
			curr_a = a_i_init
		else
			curr_a = path_choice[j = At(j-1)]
		end
		path_state[j = At(j)] = curr_a
		
		# choice = next state
		a_i_next = policy[a = curr_a, j = At(j)]
		path_choice[j = At(j)] = a_i_next
		push!(path_choice_nt, (; j, a_i=curr_a, a_i_next))
	end
	
	pp_df = @chain policy_plus begin
		DataFrame
		select!(:policy_plus => AsTable, Not(:policy_plus) )
	end

	df0 = DataFrame(path_choice_nt)
	
	sim_df = leftjoin!(df0, pp_df, on = [:a_i, :a_i_next, :j])

	(; sim_df, path_state)
end

# ╔═╡ 9a7fe74c-e8ac-47e2-9343-47ceced4207a
function solve_backward_forward(::VFI, par, a_grid; r, w, a_init)
	(; J) = par
	rs = DimArray(fill(r, J+1), Dim{:t}(0:J))
	ws = DimArray(fill(w, J+1), Dim{:t}(0:J))
	
	out_t = solve_backward(par, a_grid; r=rs, w=ws)		

	a_i_init = only(findall(a_grid .== a_init))
	
	(; path_state, sim_df) = solve_forward(out_t, par, a_grid; a_i_init, j_init = 0)

	(; sim_df, path_state = path_state, out_t, a_init)
end

# ╔═╡ 88d45c83-a253-4b2f-931a-b7bb5873fef4
function partial_equilibrium(par, a_grid, (; K_guess, r, w); pmf = pmf(par.m), details = true, return_df = false, a_init = a_grid[1], solution_method = VFI())
	(; bonds2GDP) = par

	sol = solve_backward_forward(solution_method, par, a_grid; r, w, a_init)

	(; sim_df) = sol

	agg_nt = aggregate(sim_df, pmf)

	GDP = output(K_guess, par)
	B₀ = bonds2GDP * GDP
	
	K_hh = agg_nt.a
	
	K_supply = K_hh - B₀
	ζ = K_supply - K_guess

	if details
		#= @info @chain sim_df begin
			leftjoin(_, DataFrame(pmf), on = :j)
			stack([:a, :a_next, #= :c, :y, :m,=# :pmf], :j)
			data(_) * mapping(:j, :value, layout = :variable) * visual(Lines)
			draw(facet = (; linkyaxes = false))
		end
		=#

		#@info NamedTuple{(:K_guess, :r, :w)}(round.((K_guess, r, w), sigdigits = 4)) |> repr
		if return_df
			return (; ζ, K_guess, r, w, K_hh, K_supply, B₀, par, sol, pmf_df= DataFrame(pmf))
		else
			return (; ζ, K_guess, r, w, K_hh, K_supply, B₀, par, sol)
		end
	else
		return ζ
	end
end

# ╔═╡ 0a4360fe-88a8-4b4a-bb60-dea0d975b8d1
function general_equilibrium(
			par, a_grid; 
			pmf = pmf(par.m),
			K_bracket = (1e-1, 10 * inverse_interest_rate(par.r , par)), 
			return_df = true,
			kwargs...
		)
		
	function objective(K_guess; details = false, return_df = false)
		r = interest_rate(K_guess, par)
		w = wage(K_guess, par)

		partial_equilibrium(par, a_grid, (; K_guess, r, w); details, pmf, return_df, kwargs...)
	end

	K_opt = find_zero(objective, K_bracket)
	
	objective(K_opt; details=true, return_df)
	
end

# ╔═╡ 67196d11-9bcc-4512-a707-189ab3f3104b
let
	y = income_profile(120, 41)
	
	par = get_par(; demo = :lifecycle, y, a̲ = 0.0)
	a_grid = range(0.0, 12.0, length = 600)
	
	K_guess = 4.522301994771901
	r = interest_rate(K_guess, par) 
	w = wage(K_guess, par)

	
	out = partial_equilibrium(par, a_grid, (; K_guess, r, w), return_df = true)
	
	(; sol, K_hh) = out

	@info @test out.K_supply ≈ 4.5218422030236365
	@info @test out.ζ        ≈ -0.0004597917482644931
	@info @test out.r        ≈ 0.020066827133991508
	@info @test out.K_hh     ≈ 6.167231451066009

	out_egm = general_equilibrium(par, a_grid, #=(; K_guess, r, w),=# return_df = true, solution_method = EGM())

	out_egm.ζ
	@info @test abs(criterion(out_egm.K_supply, out.K_supply)) < 5e-4
	@info @test abs(criterion(out_egm.r,        out.r))        < 5e-6
	@info @test abs(out_egm.ζ) < 1e-11
	
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
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuantEcon = "fcd29c91-0bd7-5a09-975d-7ac3f643a60c"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
AlgebraOfGraphics = "~0.10.8"
CairoMakie = "~0.13.10"
Chain = "~0.6.0"
DataFrameMacros = "~0.4.1"
DataFrames = "~1.7.0"
DimensionalData = "~0.29.17"
Interpolations = "~0.15.1"
PlutoTest = "~0.2.2"
PlutoUI = "~0.7.63"
QuantEcon = "~0.16.6"
Roots = "~2.2.7"
StatsBase = "~0.34.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "3895ea017024d9abe126e5d50ff5df835189b806"

[[deps.ADTypes]]
git-tree-sha1 = "e2478490447631aedba0823d4d7a80b2cc8cdb32"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.14.0"

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
git-tree-sha1 = "ccb66b5053b5870398e13eba7033a115f52ecde8"
uuid = "cbdf2221-f076-402e-a563-3d30da359d67"
version = "0.10.8"

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
git-tree-sha1 = "9606d7832795cbef89e06a550475be300364a8aa"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.19.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
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
git-tree-sha1 = "03fea4a4efe25d2069c2d5685155005fc251c0a1"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.3.0"

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
git-tree-sha1 = "9bd45574379e50579a78774334f4a1f1238c0af5"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.13.10"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.Chain]]
git-tree-sha1 = "9ae9be75ad8ad9d26395bf625dea9beac6d519f1"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.6.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "e771a63cc8b539eca78c85b0cabd9233d6c8f06f"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
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
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

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
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "0df00546373af8eee1598fb4b2ba480b1ebe895c"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.10"

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
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

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
git-tree-sha1 = "210933c93f39f832d92f9efbbe69a49c453db36d"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.1"

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
deps = ["Adapt", "ArrayInterface", "ConstructionBase", "DataAPI", "Dates", "Extents", "Interfaces", "IntervalSets", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "PrecompileTools", "Random", "RecipesBase", "SparseArrays", "Statistics", "TableTraits", "Tables"]
git-tree-sha1 = "b628bd06173897d44ab5cb5122e4a31509997c5a"
uuid = "0703355e-b756-11e9-17c0-8b28908087d0"
version = "0.29.17"

    [deps.DimensionalData.extensions]
    DimensionalDataAlgebraOfGraphicsExt = "AlgebraOfGraphics"
    DimensionalDataCategoricalArraysExt = "CategoricalArrays"
    DimensionalDataDiskArraysExt = "DiskArrays"
    DimensionalDataMakie = "Makie"
    DimensionalDataNearestNeighborsExt = "NearestNeighbors"
    DimensionalDataPythonCall = "PythonCall"
    DimensionalDataStatsBase = "StatsBase"

    [deps.DimensionalData.weakdeps]
    AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
    CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
    DiskArrays = "3c3547ce-8d99-4f5e-a174-61eb10b00ae3"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
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
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "8cc47f299902e13f90405ddb5bf87e5d474c0d38"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.2+0"

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
git-tree-sha1 = "f089ab1f834470c525562030c8cfde4025d5e915"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.27.0"

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
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "910febccb28d493032495b7009dce7d7f7aee554"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.0.1"
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
git-tree-sha1 = "294e99f19869d0b0cb71aef92f19d03649d028d5"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.4.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "2670cf32dcf0229c9893b895a9afe725edb23545"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.9"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "fee60557e4f19d0fe5cd169211fdda80e494f4e8"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.84.0+0"

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
git-tree-sha1 = "c5abfa0ae0aaee162a3fbb053c13ecda39be545b"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "dc6bed05c15523624909b3953686c5f5ffa10adc"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.1"

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
git-tree-sha1 = "6a9fde685a7ac1eb3495f8e812c5a7c3711c2d5e"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.3"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interfaces]]
git-tree-sha1 = "331ff37738aea1a3cf841ddf085442f31b84324f"
uuid = "85a1e053-f937-4924-92a5-1367d23b7b87"
version = "0.3.2"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Random", "RoundingEmulator"]
git-tree-sha1 = "694c52705f8b23dc5b39eeac629dc3059a168a40"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.35"
weakdeps = ["DiffRules", "ForwardDiff", "IntervalSets", "LinearAlgebra", "RecipesBase", "SparseArrays"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

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
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

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
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

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
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

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

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "1d7d16f0e02ec063becd7a140f619b2ffe5f2b11"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.22.10"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "c3159eb1e3aa3e409edbb71f4035ed8b1fc16e23"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.9.5"

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
git-tree-sha1 = "31a99cb7537f812e1d6be893a71804c35979f1be"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.4"

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
git-tree-sha1 = "35a8d661041aa6a237d10e12c29a7251a58bf488"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "1.1.4"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

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
git-tree-sha1 = "9216a80ff3682833ac4b733caa8c00390620ba5d"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.0+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "EnumX", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "31b3b1b8e83ef9f1d50d74f1dd5f19a37a304a1f"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.12.0"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

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

[[deps.PlutoTest]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "Test"]
git-tree-sha1 = "17aa9b81106e661cffa1c4c36c17ee1c50a86eda"
uuid = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
version = "0.2.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3876f0ab0390136ae0b5e3f064a109b87fa1e56e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.63"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "555c272d20fc80a2658587fb9bbda60067b93b7c"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.19"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
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
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

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
git-tree-sha1 = "13c5103482a8ed1536a54c08d0e742ae3dca2d42"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.4"

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
deps = ["DSP", "DataStructures", "Distributions", "FFTW", "Graphs", "LinearAlgebra", "Markdown", "NLopt", "Optim", "Pkg", "Primes", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Test"]
git-tree-sha1 = "034293b29fdbcae73aeb7ca0b2755e693f04701b"
uuid = "fcd29c91-0bd7-5a09-975d-7ac3f643a60c"
version = "0.16.6"

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
git-tree-sha1 = "3ac13765751ffc81e3531223782d9512f6023f71"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.7"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

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
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

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
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

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
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
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
git-tree-sha1 = "b81c5035922cc89c2d9523afc6c54be512411466"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.5"

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
git-tree-sha1 = "9022bcaa2fc1d484f1326eaa4db8db543ca8c66d"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.4"

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
git-tree-sha1 = "02aca429c9885d1109e58f400c333521c13d48a0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.4"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "cbbebadbcc76c5ca1cc4b4f3b0614b3e603b5000"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.2"

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
git-tree-sha1 = "02c1ac8104c9cf941395db79c611483909c04c7d"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.23.0"
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

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

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
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "002748401f7b520273e2b506f61cab95d4701ccf"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.48+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "d2408cac540942921e7bd77272c32e58c33d8a77"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.5.0+0"

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
git-tree-sha1 = "dcc541bb19ed5b0ede95581fb2e41ecf179527d2"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.6.0+0"
"""

# ╔═╡ Cell order:
# ╠═10d34c05-63e9-444e-aff8-bc7aa82858eb
# ╠═c9106199-39e7-4af5-bd33-16923e5b28ff
# ╠═a2b8696d-ff6c-42e2-9c76-5dd45a882487
# ╠═68c038ec-c197-4044-8b3a-26974f284e8b
# ╠═46170d0a-d23b-4e7e-bf34-bb181db8dd20
# ╠═70fc73c5-4154-4a04-99fa-ce084d42ea93
# ╠═d82f2dcd-6e81-459b-bd53-a31194176b7b
# ╠═8daf8548-2245-4271-80f7-d8a352687260
# ╠═45caaf75-c1bb-40d0-b4e7-5039d1ba3ba4
# ╠═3128e1c7-2e13-4bd5-b6b7-9d3b90a126b3
# ╠═fc1fa5b1-4868-4951-ba14-14d07ccf2ee7
# ╠═48a39844-3ec1-4300-9ee5-ade8c9a5c1d0
# ╠═980ec36d-26b7-49b6-bafd-c31ea3a1befe
# ╠═46964708-1e84-4e05-bf2d-440efa54efe8
# ╠═ba932291-dd10-43a1-9f86-8659a66365f5
# ╠═cac697fa-cde9-4b45-ae7a-e582388e491c
# ╠═7ff370ce-53e4-40d8-90e0-6712c844dc41
# ╠═66bc077a-8843-448d-b883-03619b21d6ff
# ╠═c09a6bd6-9e5f-4265-9a52-71ea586a385d
# ╠═64055da7-95a9-4920-abcb-c88f2add29c1
# ╠═ae0252ca-4a39-4581-9fd4-b99f635a6ca8
# ╠═5a947389-9c8c-4ac3-b162-b93a30479d1f
# ╠═0680a3c4-c437-4d67-bacb-49c34fdded1c
# ╠═67196d11-9bcc-4512-a707-189ab3f3104b
# ╠═14c512b8-2f0e-4831-a1b4-4e66f3e061f5
# ╠═91bbe376-ced2-4b88-b564-b72d7bf0900f
# ╠═db4b45b9-fc4a-457e-bfa2-9778d2002413
# ╠═4e6216db-e6fe-4bdc-9bd4-d22e9edc532a
# ╠═d37ead9d-23e3-4492-863e-17c39c068f1c
# ╠═ba2cd4ef-3acf-4d87-a7db-27c1bdf0994e
# ╠═44dde0b2-ad39-4558-b260-41590796b2a7
# ╠═046d2a75-c05f-4c27-95c7-e960cb14f494
# ╠═0e58d89d-0ce7-4ec4-b89d-e4db8fb03b3f
# ╠═857e9793-6cc8-4c29-935c-493d88cc5da7
# ╠═09c2d6c5-1268-44f7-89af-628ecba011a5
# ╠═4c27296c-aa27-4a48-816e-5d10ecceecf8
# ╠═2e34251a-e8e7-4fcf-9ac0-00509b5ae687
# ╠═83509c7d-2f15-4302-95ec-77b23f885bc2
# ╠═c5d76211-e9aa-4040-84f5-7417465d585f
# ╠═88d45c83-a253-4b2f-931a-b7bb5873fef4
# ╠═0a4360fe-88a8-4b4a-bb60-dea0d975b8d1
# ╠═46068b3b-fad4-4ba7-97bb-8c0e43589a96
# ╠═42ca4fde-502d-489f-979b-1213584bb72d
# ╠═fd045d53-7d11-4dac-824c-0e382b80d4f5
# ╠═94c3c00a-389a-4cc8-97aa-c51339a3ef37
# ╠═76a2bff9-6036-49a8-930a-551a23cd6f32
# ╟─df140406-28a5-49cc-b5c1-ecc53819802c
# ╠═4e661c4d-41df-4b63-b188-2a52aa62be2d
# ╠═13173588-b32f-48cd-8234-6a45673bfd01
# ╠═9a7fe74c-e8ac-47e2-9343-47ceced4207a
# ╠═2ae464c4-a641-43af-8adc-066b9c0be788
# ╟─dd7c5948-b335-44ea-909d-bcfd8be6b40e
# ╠═1748ac1d-5f4b-417b-9b11-d5bbbcb2c072
# ╠═21d8f2f0-8368-48dc-a625-eeb58d470bf1
# ╠═b04af990-5fb4-4877-9e6b-66fd6847e4d2
# ╠═84e11eea-9626-4944-933e-226f29749da9
# ╠═072d96b3-75b3-4291-9220-219bdd6fe34f
# ╠═23f7eab4-70e7-4c35-8124-4cfb85095f95
# ╠═1fd2c215-37dd-42bb-81a5-a4e229a50580
# ╠═ee932ecb-1537-4bd2-a8c8-7b16c42e73e2
# ╠═16e4f838-00cc-46dc-b211-f2f492b67285
# ╟─2caa6111-33bf-4e90-b801-7b4655020444
# ╠═c9da52e2-098a-46c4-acbf-173fce6965d8
# ╠═c45dcd06-b3a1-4045-89d4-b5632b988cf8
# ╠═fbd9c561-24c8-4f76-9235-f59593cc298a
# ╠═acd91d10-53a9-44f2-8e91-ce058a30535b
# ╠═33b42231-663b-4e3f-ba4c-18f0a1ce24ea
# ╠═4741eb11-c44c-4141-bb02-269ff2b78ad2
# ╠═d7bd570b-3e14-476c-8337-7edbab49980a
# ╠═91ddd974-9d67-4cd3-b4f8-60521c37baaa
# ╠═2db71af0-fcc7-4822-a6cc-bbcf07107182
# ╠═fd369634-4ddb-40cf-a1a0-e7e6282b7595
# ╠═03d38330-6f14-46b5-86fd-620072ff102d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
