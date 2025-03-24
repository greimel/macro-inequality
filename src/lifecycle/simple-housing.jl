### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ f2457772-3de4-47d2-946a-6f6e15a48fc4
using DimensionalData

# ╔═╡ ceda67ec-c351-4776-9980-0f2c27e0ea02
using PlutoTest

# ╔═╡ 8e018136-64f9-4a1f-975b-2d5ea5b08498
using CairoMakie, AlgebraOfGraphics

# ╔═╡ 829c6847-89a5-4a4a-807e-b65b04dedc0b
using Chain, DataFrames, DataFrameMacros

# ╔═╡ e8249393-d1bb-41b0-a36c-8447bd14bc5e
using LinearAlgebra: diag, I, /

# ╔═╡ 683216f2-51d7-4650-9263-8bc234539460
using StatsBase: weights

# ╔═╡ 0471e65c-5ca8-49a5-ac32-ebddb01b9738
using Statistics: mean

# ╔═╡ 62ca7314-4892-413c-98aa-dcdf05ab6a7a
using Interpolations

# ╔═╡ ba4d8cac-7493-4c69-83b5-78f7d45ff305
using PlutoUI

# ╔═╡ 134ba669-b2e2-40c3-872b-b79d14d16544
md"""
## Model

```math
\begin{align*}
&\max \sum_{j=0}^{J-1} \beta^j \textcolor{orange}{\Phi_j}u(c_j, s(h_j, \tilde h_j)) \\
&\begin{aligned}\text{subject to }
&x_j = h_j - (1-\delta)h_{t-1} \\
&c_j + p_{t(j)} x_j + a_{j+1} = y_j + (1+r_{t(j)}) a_{j} \\
&s(h_j, \tilde h_j) = h_j - \phi \tilde h_j \\
&a_{0}, h_{-1} \text{ given}
\end{aligned}
\end{align*}
```
"""

# ╔═╡ dd01a6ba-ea0b-4f10-8597-c7d5f36cf5fe
md"""
### Life-time budget constraint

```math
\begin{align*}
	\sum_{j=0}^{J-1} \Bigl(\frac{1}{1+r}\Bigr)^j (c_j + x_j) = \sum_{j=0}^{J-1} \Bigl(\frac{1}{1+r}\Bigr)^j (y_j) + (1+r) a_0 + 
\end{align*}
```
"""

# ╔═╡ ee8b2332-192c-4cd3-b89a-97fd5db07fa4
md"""
The Lagrangian is 

```math
\sum_{t=0}^\infty \Phi_j \beta^j \Biggl(u(c_j, h_j) - \lambda_j \Bigl(c_j + p_{t(j)} h_j + a_{j+1} - y_j - (1+r_{t(j)}) a_{j} - (1-\delta)p_{t(j)} h_{j-1}\Bigr) \Biggr)
```

* ``(a_{j+1})``: ``\textcolor{lightgray}{\beta^j} \Phi_j \lambda_j = \textcolor{purple}{\beta}^\textcolor{lightgray}{t+1} \textcolor{purple}{\Phi_{j+1} \lambda_{j+1}} (1+r_{t+1})``
* ``(c_j)``: ``u_{c_j} = \lambda_j``
* ``(h_j)``: ``\textcolor{lightgray}{\beta^j} \Phi_j (u_{h_j} - \lambda_j p_{t(j)}) + \textcolor{purple}{\beta}^\textcolor{lightgray}{t+1} \textcolor{purple}{\Phi_{j+1}}\textcolor{lightgray}{(-1)}\textcolor{purple}{\lambda_{j+1}}\textcolor{lightgray}{(-1)} (1-\delta) p_{t(j+1)} = 0 ``

"""

# ╔═╡ 2ce2146a-4158-4280-86d3-7d73937262e0
md"""
Plugging ``(a_{j+1})`` into ``(h_j)`` gives

```math
\begin{align}
0 &= \textcolor{lightgray}{\Phi_j} (u_{h_j} - \lambda_j p_{t(j)}) + \frac{\textcolor{lightgray}{\Phi_j} \lambda_j}{1+r_{t(j+1)}} (1-\delta) p_{t(j+1)} \\
\implies u_{h_j} &= \lambda_j p_{t(j)}\biggl(1 - \frac{1-\delta}{1+r_{t(j+1)}} \frac{p_{t(j+1)}}{p_{t(j)}}\biggr) \\
\implies \frac{u_{h_j}}{u_{c_j}} &= p_{t(j)}\biggl(1 - \frac{1-\delta}{1+r_{t(j+1)}} \frac{p_{t(j+1)}}{p_{t(j)}}\biggr) \\
\end{align}
```
"""

# ╔═╡ 71fcb5c5-073f-438c-bf17-0d8f666facac
md"""
Above, we computed ``\frac{u_{h_j}}{u_{c_j}}= \frac{\xi}{1-\xi}\frac{c_j}{h_j}``. Hence,

```math
\begin{align*}
\frac{\xi}{1-\xi}\frac{c_j}{h_j} &= p_{t(j)}\biggl(1 - \frac{1-\delta}{1+r_{t(j+1)}} \frac{p_{t(j+1)}}{p_{t(j)}}\biggr) \\
\implies \frac{c_j}{h_j} &= \underbrace{p_{t(j)} \frac{1-\xi}{\xi}\biggl(1 - \frac{1-\delta}{1+r_{t(j+1)}} \frac{p_{t(j+1)}}{p_{t(j)}}\biggr)}_{=: \tilde{\kappa}_j(p_{t(j)}, p_{t(j+1)}, r_{t(j+1)} )} \\
\implies \frac{c_j}{p_{t(j)} h_j} &= \underbrace{\frac{1-\xi}{\xi} \biggl(1 - \frac{1-\delta}{1+r_{t(j+1)}} \frac{p_{t(j+1)}}{p_{t(j)}}\biggr)}_{=: \kappa_j(p_{t(j)}, p_{t(j+1)}, r_{t(j+1)} )} \tag{$***$} \\
\end{align*}
```
"""

# ╔═╡ e49fee0a-148b-4a91-92f8-b9536f2a395d
md"""
```math
\begin{align}
u_{c_j} &= \frac{\textcolor{lightgray}{(1 - \xi)}}{c_j} (c_j^{1-\xi} \textcolor{skyblue}{(c_j/\tilde{\kappa}_j)}^{\xi})^{1-\sigma}\\
&=  \frac{\textcolor{lightgray}{(1 - \xi)}}{c_j} (c_j \textcolor{skyblue}{\tilde{\kappa}_j^{-\xi}})^{1-\sigma}\\
&= \textcolor{lightgray}{(1 - \xi)} {\tilde{\kappa}_j^{-\xi(1-\sigma)}} c_j^{-\sigma}\\
\end{align}
```

Recall ``(a_{j+1})``.

```math
\begin{align}
 \Phi_j \lambda_j &= \textcolor{purple}{\beta} \textcolor{purple}{\Phi_{j+1} \lambda_{j+1}} (1+r_{t(j+1)}) \\
\implies
\frac{\lambda_j}{\lambda_{j+1}} &= \frac{\Phi_{j+1}}{\Phi_j} \beta (1+r_{t(j+1)}) \\
\implies
\biggl(\frac{{\tilde{\kappa}_j}}{{\tilde{\kappa}_{j+1}}}\biggr)^{{-\xi(1-\sigma)}} \biggl(\frac{c_j}{ c_{j+1}}\biggr)^{-\sigma} &=  \frac{\Phi_{j+1}}{\Phi_j} \beta (1+r_{t(j+1)}) \\
\implies
 c_j &=  c_{j+1} \biggl(\frac{\Phi_{j+1}}{\Phi_j} \beta (1+r_{t(j+1)})\biggr)^{-1/\sigma} / \biggl(\frac{{\tilde{\kappa}_j}}{{\tilde{\kappa}_{j+1}}}\biggr)^{{\xi(1-\sigma)/\sigma}}\\
\end{align}
```
"""

# ╔═╡ 867e51e9-97df-4236-9c53-084990ccfe23
md"""
Note that ``\frac{\Phi_{j+1}}{\Phi_j}`` is the probability of surviving until age ``j+1`` conditional on having survived until age ``j``. Thus it is the survival probability at age ``j``.

```math
\frac{\Phi_{j+1}}{\Phi_j} = (1-m_j)
```
"""

# ╔═╡ 62fef877-a42e-4a86-af78-dcf4e54b779a
md"""
# The last period
"""

# ╔═╡ 6e02d06e-5ea0-4bcc-86a9-29182356cb3c
md"""
# VFI vs EGM: Testing one iteration
"""

# ╔═╡ 73f50ec7-6763-47d2-9560-167699c49161
md"""
# The penultimate period
"""

# ╔═╡ b0270296-aba9-4c53-87bb-550550944997
md"""
## Value function iteration
"""

# ╔═╡ 6f3e64fa-3280-4b12-997d-b7355e11b689
md"""
### Check choices in last period
"""

# ╔═╡ 99dcf6b6-104b-4ccc-afc1-72355dc0e21e
md"""
## EGM
"""

# ╔═╡ d5831143-df0d-43aa-9fc0-1a312feaa3dc
md"""
### Iterating backwards

* ``c', \omega'``
* ``c' \to c`` (Euler equation)
* BC: ``c + ph + a' = y + \omega``

```math
\begin{align}
\omega' &= (1-\delta) p' h + (1+r') a' \\
\iff a' &= \frac{1}{1+r'}\omega' - \frac{1-\delta}{1+r'}p'h
\end{align}
```
Plug in the budget constraint
```math
\begin{align}
y + \omega &= c + ph + a'  \\
y + \omega &= c + ph + \frac{1}{1+r'}\omega' - \frac{1-\delta}{1+r'}p'h \\
\omega &= c + ph\Bigl(1 - \frac{1-\delta}{1+r'}\frac{p'}{p}\Bigl) + \frac{1}{1+r'}\omega' - y
\end{align}
```
"""

# ╔═╡ b1b861ce-0659-11f0-094d-67da13f58563
md"""
# Housing
"""

# ╔═╡ bae86455-59b4-42fc-9f0e-b14ed67b9e5f
par = (; ξ = 0.1, δ = 0.1, γ = 2.0, β = 0.95, y = 1.0)

# ╔═╡ 9666b117-8273-4b32-a757-cb5a7b0ef107
Δr = 0.0

# ╔═╡ df8d0140-cd53-4864-b5f6-900661da5113
prices = (; p = 1.1, r = 1/par.β - 1 + Δr, w = 1.1)

# ╔═╡ 8f170e55-a231-44ba-981f-b3bca28bcf9c
u(c, h, (; ξ, γ)) = (c^(1-ξ) * h^ξ)^(1-γ)/(1-γ)

# ╔═╡ 0405682d-9757-48c2-81d9-03abe696af4b
function terminal_value(ω_grid, par, last_prices; what_is_zero = :ω)
	(; δ) = par
	(; rₜ₍ⱼ₊₁₎, pₜ₍ⱼ₎, pₜ₍ⱼ₊₁₎, w) = last_prices
	
	inc = par.y * w
	
	ω_dim = Dim{:ω}(ω_grid)
	
	v_terminal = zeros(ω_dim, name = :v)
	c_terminal = zeros(ω_dim, name = :c)
	h_terminal = zeros(ω_dim, name = :h)
	h̄s = zeros(ω_dim, name = :h̄)
	κs = zeros(ω_dim, name = :κ)
	a_terminal = zeros(ω_dim, name = :a_next)
	ω_terminal = zeros(ω_dim, name = :ω_next)
	
	
	for ωⱼ ∈ ω_grid
	
		aⱼ₊₁(h) = what_is_zero == :a ? 0.0 : - pₜ₍ⱼ₊₁₎ * (1 - δ)/(1 + rₜ₍ⱼ₊₁₎) * h
		c(h) = inc + ωⱼ - aⱼ₊₁(h) - pₜ₍ⱼ₎ * h

		h̄ = (inc + ωⱼ)/(pₜ₍ⱼ₎)
		hⱼ = range(0.01, h̄, length=100_000)
	
		cⱼ = c.(hⱼ)
			
		aⱼ₊₁ = @. inc + ωⱼ - cⱼ - pₜ₍ⱼ₎ * hⱼ
		ωⱼ₊₁ = @. pₜ₍ⱼ₊₁₎ * (1 - δ) * hⱼ + (1+rₜ₍ⱼ₊₁₎) * aⱼ₊₁

		close_to_zero(x) = abs(x) < 1e-12
		if what_is_zero == :ω
			@assert all(close_to_zero, ωⱼ₊₁)
		else what_is_zero == :a
			@assert all(close_to_zero, aⱼ₊₁)
		end
		
		uu = u.(cⱼ, hⱼ, Ref(par))

		u_opt, h_i_opt = findmax(uu)

		h̄s[ω = At(ωⱼ)] = h̄
		κs[ω = At(ωⱼ)] = cⱼ[h_i_opt] / hⱼ[h_i_opt]
		v_terminal[ω = At(ωⱼ)] = u_opt
		c_terminal[ω = At(ωⱼ)] = cⱼ[h_i_opt]
		h_terminal[ω = At(ωⱼ)] = hⱼ[h_i_opt]
		a_terminal[ω = At(ωⱼ)] = aⱼ₊₁[h_i_opt]
		ω_terminal[ω = At(ωⱼ)] = ωⱼ₊₁[h_i_opt]
		
	end

	out = DimStack(v_terminal, c_terminal, h_terminal, a_terminal, ω_terminal, h̄s, κs)
end

# ╔═╡ 31e0a1a9-2216-4a70-9f81-2f16267fcb5c
let
	what_is_zero = :ω
	par = (; γ = 2.0, ξ = 0.1, δ = 0.1, y = 1.0)
	
	inc = 1.0

	ω_grid = range(-0.04, -0.02, length = 100)
		
	prices = (; rₜ₍ⱼ₊₁₎ = 0.07, pₜ₍ⱼ₎ = 1.0, pₜ₍ⱼ₊₁₎ = 1.0, w = 1.0)

	out = terminal_value(ω_grid, par, prices; what_is_zero)

	@chain out begin
		DataFrame
		select(Not(string(what_is_zero) * "_next"))
		stack(Not(:ω))
		data(_) * mapping(:ω, :value, layout = :variable) * visual(Lines)
		draw(facet = (; linkyaxes = false))
	end
end

# ╔═╡ cb7f3832-c1e9-48de-b21e-2390b6d1a2e8
md"""
* definition of cash-at-hand ``\omega_{j+1} := p_{j+1} h_j (1-\delta) + (1+r) a_{j+1}``
* budget constraint ``c_j + p h_j + a_{j+1} = w\cdot y + \omega_j``

```math
\begin{align}
c_j + ph_j + a_{j+1} &= w \cdot y + \omega_j \\
\omega_{j+1} &:= p_{j+1} h_j (1-\delta) + (1+r) a_{j+1} \\
a_{j+1} &= \frac{\omega_{j+1} - p_{j+1} h_j (1-\delta)}{1+r} \\
c_j + p_j h_j + \frac{\omega_{j+1} - p_{j+1} h_j (1-\delta)}{1+r} &= w \cdot y + \omega_j \\
c_j + p_j h_j - p_{j+1} h_j \frac{1-\delta}{1+r} &= w \cdot y + \omega_j - \frac{1}{1+r} \omega_{j+1} \\
c_j + p_j h_j \Bigl(1 - \frac{1-\delta}{1+r}\frac{p_{j+1}}{p_j}\Bigr) &= w \cdot y + \omega_j - \frac{1}{1+r} \omega_{j+1}
\end{align}
```

Note that ``c = \kappa p h``. Thus

```math
\begin{align}
p_j h_j \Bigl(\kappa + 1 - \frac{1-\delta}{1+r}\frac{p_{j+1}}{p_j}\Bigr) &= w \cdot y + \omega_j - \frac{1}{1+r} \omega_{j+1}
\end{align}
```


"""

# ╔═╡ ba62cdca-ac5a-480b-b930-7b121156eba6
md"""
* ``c = \kappa (ph)``
"""

# ╔═╡ d00d0c99-44e8-4131-96d7-9f7956bf9629
#vⱼ₊₁ ωⱼ₋₁

# ╔═╡ c5ca1482-7252-4220-92ec-a67900f036e4
function κ₀((; δ, ξ), (; pₜ₍ⱼ₎, pₜ₍ⱼ₊₁₎, rₜ₍ⱼ₊₁₎); terminal) 
	#if terminal
	#	β = 0.0
	#end
    
	(1-ξ)/ξ * (1 - (1 - δ)/(1+rₜ₍ⱼ₊₁₎) * pₜ₍ⱼ₊₁₎/pₜ₍ⱼ₎) # c_by_ph
end

# ╔═╡ fd3cb5a8-8717-4d2f-8e58-4151a25f79dd
function choices(ωⱼ, ωⱼ₊₁, par, prices; terminal = false, details = false)
	(; rₜ₍ⱼ₊₁₎, w, pₜ₍ⱼ₎, pₜ₍ⱼ₊₁₎) = prices
	(; δ, y, ξ, β) = par
	
	inc = only(unique(y)) * w
	
	κ = κ₀(par, prices; terminal)

	pₜ₍ⱼ₎hⱼ = 1/(κ + 1 - (1-δ)/(1+rₜ₍ⱼ₊₁₎)*pₜ₍ⱼ₊₁₎/pₜ₍ⱼ₎) * (ωⱼ + inc - ωⱼ₊₁/(1+rₜ₍ⱼ₊₁₎))
	cⱼ = κ * pₜ₍ⱼ₎hⱼ
	hⱼ = pₜ₍ⱼ₎hⱼ/pₜ₍ⱼ₎

	if terminal
		aⱼ₊₁ = ωⱼ₊₁ + inc - cⱼ - pₜ₍ⱼ₊₁₎ * hⱼ
		#@info "ω_J+1 = $(pₜ₍ⱼ₎hⱼ * (1-δ) + (1+r) * aⱼ₊₁)"
	end
	
	uⱼ = if (terminal && ωⱼ₊₁ < 0) || cⱼ < 0
		-Inf
	else
		u(cⱼ, hⱼ, par)
	end
	
	aⱼ₊₁ = (ωⱼ₊₁ - (1-δ) * pₜ₍ⱼ₊₁₎ * hⱼ)/(1+rₜ₍ⱼ₊₁₎)
	
	let
		lhs = cⱼ + pₜ₍ⱼ₎hⱼ + aⱼ₊₁
		rhs = ωⱼ + inc
		#@info @test lhs ≈ rhs
	end

	if details
		return (; cⱼ, phⱼ = pₜ₍ⱼ₎hⱼ, aⱼ₊₁, hⱼ, uⱼ, ωⱼ₊₁, pₜ₍ⱼ₎, κ)
	else
		return uⱼ
	end
end

# ╔═╡ 9c9ceccb-fb28-419b-a9ed-37aaf6406f3c
function prices_from_price_paths(price_paths, t)
	(; 
		rₜ₍ⱼ₊₁₎ = price_paths.r, price_paths.w, 
		pₜ₍ⱼ₎   = price_paths.ps[t = At(t)],
		pₜ₍ⱼ₊₁₎ = price_paths.ps[t = At(t+1)]
	)
end

# ╔═╡ 1dbb9a6c-831e-4519-a862-668e61ea6a4e
ω_grid = sort([0.0; range(-1.0, 10.0, length = 1000)])

# ╔═╡ 53e1c988-fda6-473f-ba2c-b460483813d1
function iterate_backward(vⱼ₊₁, ω_grid, par, prices)
	
	(; β) = par
	ω_dim = Dim{:ω}(ω_grid)
	vⱼ       = zeros(ω_dim)

	TT = choices(0.0, 0.0, par, prices, details = true) |> typeof
	N = length(ω_grid)
	
	polⱼ = DimVector(Vector{TT}(undef, N), ω_dim)
	
	for ωⱼ ∈ ω_grid
		ωⱼ₊₁ = ω_grid
		
		us = choices.(ωⱼ, ωⱼ₊₁, Ref(par), Ref(prices))
		
		(v, ω_i_opt) = findmax(us .+ β .* vⱼ₊₁)
		vⱼ[      ω = At(ωⱼ)] = v
		ωⱼ₊₁_opt = ω_grid[ω_i_opt]
		polⱼ[ω = At(ωⱼ)] = choices(ωⱼ, ωⱼ₊₁_opt, par, prices, details = true)
	end

	(; vⱼ, polⱼ)
end

# ╔═╡ f61bf1a6-634f-4a47-8c28-efd42ce87747
function solve_backward_forward(par, ω_grid; price_paths, born = 0, init = nothing)
	(; J) = par
	j_dim = Dim{:j}(0:J-1)
	ω_dim = Dim{:ω}(ω_grid)

	prices = prices_from_price_paths(price_paths, (J-1) + born)
	
	TT = choices(0.0, 0.0, par, prices, details = true) |> typeof
	N = length(ω_grid)
	
	value  = zeros((ω_dim, j_dim))
	policy = DimArray(
		Array{TT}(undef, (N, J)), (ω_dim, j_dim))
	
	value[j = At(J-1)]  .= choices.(ω_grid, 0.0, Ref(par), Ref(prices); terminal = true)
	policy[j = At(J-1)] .= choices.(ω_grid, 0.0, Ref(par), Ref(prices); terminal = true, details = true)

	# solve backward
	for j ∈ J-2:-1:0
		t = j + born
		
		vⱼ₊₁ = value[j = At(j+1)]
		prices = prices_from_price_paths(price_paths, t)
 		(; vⱼ, polⱼ) = iterate_backward(vⱼ₊₁, ω_grid, par, prices)

		value[ j = At(j)] .= vⱼ
		policy[j = At(j)] .= polⱼ
	end

	if !isnothing(init)
		p₀ = price_paths.ps[t = At(0)]
		r₀ = prices.r
		ω_init = p₀ * (1-par.δ) * init.h + (1+r₀) * init.a
		@info (; p₀, ω_init, init)
	else
		ω_init = 0.0
	end
	pol_simulated = DimVector(Vector{TT}(undef, J), j_dim)
	ωⱼ = ω_grid[findfirst(ω_grid .≥ ω_init)]

	# solve forward
	for j ∈ 0:J-1
		polⱼ = policy[j = At(j), ω = At(ωⱼ)]
		pol_simulated[j = At(j)] = polⱼ

		ωⱼ = polⱼ.ωⱼ₊₁
	end

	pol_simulated

	df = select(DataFrame(DimTable(pol_simulated)), :value => AsTable, Not(:value))
	(; df, policy)
end

# ╔═╡ 4ee76793-be3c-4849-ab1c-cc64d0cdf906
md"""
# Testing against _Falling Behind_
"""

# ╔═╡ ee007b1f-e5b5-475e-b672-53483ad42f88


# ╔═╡ 230bdd4d-99fb-4adb-971b-dbef024b3e20
p_test = [0.5028480280461705, 0.5241238908263111, 0.5229295236881214, 0.5217017148142188, 0.5204099628296323, 0.5190844559661794, 0.5177911453101505, 0.5166033832040168, 0.5155792381994333, 0.5147496348256029, 0.514117292657164, 0.5136630071761439, 0.5133549339145355, 0.5131574170435576, 0.5130374257916149, 0.5129680665281533, 0.5129295872492532, 0.5129087271453475, 0.5128972868297691, 0.5128905760328677, 0.5128861057502261, 0.512882648490296, 0.5128796410208144, 0.5128768425830681, 0.512874157028558, 0.5128715481667272, 0.512869002797782, 0.5128665158769151, 0.5128640850404091, 0.5128617087497279, 0.5128593857139132, 0.512857114724707, 0.5128548946132658, 0.5128527242394967, 0.5128506024893068, 0.5128485282736247, 0.5128465005277771, 0.512844518210923, 0.5128425803055271, 0.5128406858168318, 0.5128388337723554, 0.5128370232213857, 0.5128352532344975, 0.5128335229030779, 0.5128318313388551, 0.5128301776734494, 0.5128285610579219, 0.5128269806623478, 0.5128254356753813, 0.5128239253038469, 0.5128224487723303, 0.51282100532278, 0.5128195942141227, 0.5128182147218784, 0.5128168661377885, 0.5128155477694639, 0.5128142589400108, 0.5128129989877024, 0.5128117672656282, 0.5128105631413655, 0.5128093859966565, 0.5128082352270891, 0.5128071102417884, 0.5128060104631144, 0.5128049353263597, 0.5128038842794712, 0.5128028567827553, 0.5128018523086093, 0.5128008703412458, 0.5127999103764325, 0.5127989719212271, 0.5127980544937343, 0.5127971576228445, 0.5127962808480111, 0.5127954237189918, 0.5127945857956379, 0.5127937666476585, 0.512792965854397, 0.512792183004622, 0.5127914176963164, 0.5127906695364635, 0.5127899381408534, 0.5127892231338809, 0.512788524148357, 0.5127878408253143, 0.5127871728138315, 0.5127865197708417, 0.5127858813609696, 0.5127852572563489, 0.5127846471364574, 0.512784050687956, 0.5127834676045216, 0.5127828975866932, 0.512782340341722, 0.5127817955834106, 0.5127812630319774, 0.5127807424139057, 0.512780233461807, 0.5127797359142817, 0.5127792495157849, 0.5127787740164927, 0.5127783091721871, 0.5127778547441065, 0.5127774104988446, 0.5127769762082233, 0.512776551649171, 0.5127761366036104, 0.512775730858354, 0.5127753342049791, 0.5127749464397375, 0.5127745673634349, 0.512774196781341, 0.5127738345030819, 0.512773480342542, 0.5127731341177729, 0.5127727956508938, 0.5127724647680043, 0.5127721412990883, 0.5127718250779322, 0.5127715159420378, 0.5127712137325312, 0.5127709182940903, 0.5127706294748553, 0.512770347126351, 0.5127700711034155, 0.5127698012641151, 0.5127695374696728, 0.5127692795843962, 0.512769027475605, 0.5127687810135566, 0.5127685400713826, 0.5127683045250161, 0.5127680742531386, 0.512767849137125, 0.5127676290610595, 0.5127674139118498, 0.512767203579637, 0.5127669979589149, 0.5127667969510747, 0.5127666004699587, 0.5127664084530146, 0.5127662208823485, 0.5127660378222231, 0.5127658594819225, 0.5127656863153942, 0.5127655191709162, 0.5127653595047076, 0.5127652096725649, 0.5127650733136256, 0.5127649558432817, 0.5127648650805959, 0.5127648120533261]

# ╔═╡ f2a067a9-b486-45dc-8d37-8b26b83ebfd2
let
	T = 300
	T₀ = length(p_test)
	
	born = 0
	t_dim = Dim{:t}(-1:T)
	ys = fill(1.19787, t_dim, name = :y)
	ys[t = At(-1)] = 1.14083

	ps = DimArray([p_test; fill(p_test[end], T - T₀ + 2)], t_dim, name = :p)

	J = 200 #T-born

	j_dim = Dim{:j}(0:J)
	
#	ps = DimVector(range(p₀, p₁, length = J+1), j_dim, name = :p)

	par = (γ = 2.0, β = 0.891944, ξ = 0.161566, δ = 0.103222, y = 1.19787, J)
	
	#par = (; ξ = 0.578, δ = 0.123, γ = 1.789, β = 0.95, y = 1.0)
	price_paths = (; ps, r = 1/par.β - 1, w = 1.0)

	# born in -1
	init = (; h = 1.83162, a = -0.736706)
	ω_grid = sort([0.0; range(-0.011, 0.05, length = 750)])
	# born in 0
	init = nothing
	ω_grid = sort([0.0; range(-0.011, 0.005, length = 250)])
	
	(; df) = solve_backward_forward(par, ω_grid; price_paths, born, init)

	@info df
	@chain df begin
		@subset(:j < 150)
		stack(Not(:j))
		data(_) * mapping(:j, :value, layout = :variable) * visual(Lines)
		draw(facet = (; linkyaxes = false))
	end # =#
end

# ╔═╡ c94afe0e-a8bf-4640-9874-f3cecab16b5f
p_test[1] * 1.83162

# ╔═╡ a4e70adc-7780-4456-96db-4e72feb2d032
function choices_test(p, (; r, β, δ, ξ, ε, Φ, G, #=group_weights, α, L̄,=# y_scale), y₀, a₋₁, h₋₁)
	
	κ₀ = ((1 - β * (1 - δ)) * (1-ξ)/ξ * p)^(1/(1-ε))
	@assert κ₀ ≈ ((r+δ)/(1+r) * (1-ξ)/ξ * p)^(1/(1-ε))

	κ₂ = 1 / (p * (r+δ) / (1+r) + κ₀)
	κ₁ = κ₀ * κ₂
	κ₃ = p * (1-δ)/(1+r) * κ₂
	
	if ε ≈ 0.0
		@assert κ₀ ≈ p * (1 - β * (1 - δ)) * (1-ξ)/ξ
		@assert κ₁ ≈ (1-ξ)
		@assert κ₂ ≈ ξ * (1+r)/(p * (δ + r))	
	end

	ϕs = diag(Φ)
	
	Θ = y_scale # productivity level
	y = y₀ .* Θ

	# initial wealth (including housing)
	ã₋₁ = @. a₋₁ + (1-δ)/(1+r) * p * h₋₁	
	𝒴 = @. r * ã₋₁ + y

	IpL2(Y) = (I - κ₁ * Φ * G) \ Y
	
	h = κ₂ * IpL2(𝒴)		
	
	h̃ = G * h

	c = κ₀ .* (h .- Φ * h̃)

	debt = (y - c - δ * p * h) / r
	#debt2 = (κ₃ / κ₂) .* h .- ã₋₁
	#@assert debt ≈ debt2
	
	ph = p .* h

	#if !(group_weights isa AbstractWeights)
	#	group_weights = weights(group_weights)
	#	@info "applied `weights()` to `group_weights`"
	#end

	ω = (1-δ) * ph + (1+r) * (-debt)
	return (; c, h, ph, debt, κ₀, ω)
	
	#=
	sensitivities = bellet_sensitivity.(h, h̃, [(; ϕ) for ϕ ∈ ϕs])
	
	avg_sensitivity = mean(sensitivities, group_weights)

	hx = δ .* ph .+ r .* debt
	tx = c + hx
	
	∑𝒴    = sum(𝒴, group_weights)
	∑h    = sum(h, group_weights)
	∑c    = sum(c, group_weights)
	∑ph   = p * ∑h
	∑debt = sum(debt, group_weights)

	housing_expenditures =  sum(hx, group_weights)
	total_expenditures = sum(tx, group_weights)

	Iₕ_demand = ∑h * δ
	Iₕ_supply = I_supply(p, (; α, L̄))

	Nₕ = p * α / Θ * Iₕ_supply
	ζₕ = Iₕ_demand - Iₕ_supply
	ζₕ_rel = ζₕ / maximum(abs, [Iₕ_demand, Iₕ_supply])

	group_tbl = (; 
			c, h, h̃, ph, 
			hx, tx, hx2y = hx ./ 𝒴, hx_share = hx ./ tx,
			debt, d2y = debt ./ 𝒴,
			𝒴, group = groups, group_weight = group_weights, sensitivities, ϕs
	)

	(; p, rhpi=p, Iₕ_demand, Iₕ_supply, ζₕ, ζₕ_rel, ∑h, ∑debt, ∑ph,
		d2y = ∑debt / ∑𝒴, d2ph = ∑debt / ∑ph, ph2y = ∑ph / ∑𝒴,
		hx2y = housing_expenditures / ∑𝒴,
		hx_share = housing_expenditures / total_expenditures,
		ϕs, ela, ε, group_tbl, avg_sensitivity, sensitivities, Nₕ
	)
	=#
end

# ╔═╡ 0af642fe-5083-4db8-8384-f0b572fa88a3
criterion(a, b) = abs(a - b) / (1 + max(abs(a), abs(b)))

# ╔═╡ e27ecff7-bf67-4c99-8970-001b078f48ff
let
	J = 100
	p = 1.123
	
	ω_grid = sort([0.0; range(-0.5, 0.5, length = 100)])
	t_dim = Dim{:t}(0:J)
	
	ps = fill(p, t_dim, name = :p)
	
	par = (; ξ = 0.578, δ = 0.123, γ = 1.789, β = 0.95, y = 1.0, J)
	price_paths = (; ps, r = 1/par.β - 1, w = 1.0)

	(; df) = solve_backward_forward(par, ω_grid; price_paths)

	(; cⱼ, hⱼ, phⱼ, aⱼ₊₁, ωⱼ₊₁) = first(df)

	out_test = let
		y₀ = [par.y]
		a₋₁ = 0.0
		h₋₁ = 0.0

		par_new = (; price_paths.r, ε = 0.0, Φ = zeros(1,1), G = zeros(1,1), y_scale = 1.0, par...)
	

		out = choices_test(p, par_new, y₀, a₋₁, h₋₁)

	end

	c_test = only(out_test.c)
	h_test = only(out_test.h)
	ph_test = only(out_test.ph)
	
	ω_test = only(out_test.ω)

	@info @test cⱼ ≈ c_test
	@info @test hⱼ ≈ h_test
	@info @test criterion(ωⱼ₊₁, ω_test) < 1e-10
	
	(; ωⱼ₊₁, ω_test, cⱼ, c_test, hⱼ, h_test)

	
end

# ╔═╡ d47576ee-d6e5-4fa3-b3c4-bee9ca49abd8
md"""
# EGM
"""

# ╔═╡ 1f66f226-aafc-4fa6-bfc3-2adc7f11d93d
function smootherstep(x₀, x₁, f₀, f₁, x)
  # Scale, and clamp x to 0..1 range
  x = clamp((x - x₀) / (x₁ - x₀), 0, 1)

  (x^3 * (x * (6x - 15) + 10)) * (f₁-f₀) + f₀
end

# ╔═╡ 1e248059-d315-4206-8fd2-a609954b46d7
prep = let
	T = 40
	t_dim = Dim{:t}(-1:T)
	
	#T₀ = length(p_test)
	#ps = DimArray([p_test; fill(p_test[end], T - T₀ + 2)], t_dim, name = :p)
	p₀ = 0.5
	p₁ = 1.0 * p₀
	T₀ = T ÷ 2
	ps₀ = smootherstep.(0, 1, p₀, p₁, range(0, 1, length=T₀))
	#ps₀ = [range(p₀, p₁, length = T₀); ]

	
	ps = DimVector([ps₀; fill(p₁, T + 2 - T₀)], t_dim, name = :p)
	
	born = 0

	ys = fill(1.0, t_dim, name = :y)
	#ys = fill(1.19787, t_dim, name = :y)
	#ys[t = At(-1)] = 1.14083

	J = 30 #T-born

	j_dim = Dim{:j}(0:J)
	
#	ps = DimVector(range(p₀, p₁, length = J+1), j_dim, name = :p)

	par = (; σ = 2.0, γ = 2.0, β = 0.891944, ξ = 0.161566, δ = 0.103222, y = fill(1.0, j_dim), m = fill(0.0, j_dim), J)

	#par = (; ξ = 0.578, δ = 0.123, γ = 1.789, β = 0.95, y = 1.0)
	price_paths = (; ps, r = 1/par.β - 1, w = 1.0)

	ωₘᵢₙ = -0.01
	ωₘₐₓ =  0.1
	grid_sparse = sort([0.0; range(ωₘᵢₙ, ωₘₐₓ, length = 100)])
	grid_dense  = sort([0.0; range(ωₘᵢₙ, ωₘₐₓ, length = 1000)])
	
	(; par, price_paths, T, j_dim, grid_sparse, grid_dense)
end

# ╔═╡ 4f54a3c5-b448-4987-8258-de77d69ef92f
prep.par

# ╔═╡ 8fd0bef0-40af-4b15-8d46-11caf051067f
penult₀ = let
	what_is_zero = :ω
	par = (; prep.par.γ, prep.par.β, prep.par.ξ, prep.par.δ, y = only(unique(prep.par.y)))
	ω̲, ω̅ = -0.1, 0.1

	#p″ = 1.567      # after last period
	#p′ = 0.9 * p″  # last period
	#p  = 0.8 * p″  # penultimate period
	p = p′ = p″ = prep.price_paths.ps |> unique |> only
	r = prep.price_paths.r

	#r = 1/par.β - 1
	prices_last   = (; rₜ₍ⱼ₊₁₎ = r, pₜ₍ⱼ₎ = p′, pₜ₍ⱼ₊₁₎ = p″, w = prep.price_paths.w)
	prices_penult = (; rₜ₍ⱼ₊₁₎ = r, pₜ₍ⱼ₎ = p,  pₜ₍ⱼ₊₁₎ = p′, w = prep.price_paths.w)

	(; par, prices_last, prices_penult, ω̲, ω̅, what_is_zero)
end

# ╔═╡ 5c197725-0cff-4fbd-9e29-cca5cc48f28c
vfi₀ = let
	(; par, prices_last, ω̲, ω̅, what_is_zero) = penult₀
	ω_grid = prep.grid_dense 
	#out_last = terminal_value(ω_grid, par, prices_last; what_is_zero)

	(; #=out_last,=# par, prices_last, penult₀.prices_penult, ω_grid)
end

# ╔═╡ 2fa9c3b9-1759-4c75-a264-40995942716c
vfi_penult = let
	(; #=out_last,=# ω_grid, par, prices_penult, prices_last) = vfi₀
	ω_dim = Dim{:ω}(ω_grid)
	ω_grid = DimVector(ω_grid, ω_dim)

	#vⱼ₊₁ = out_last.v

	(; prices_last, prices_penult) = let
		p, p′, p″ = prices_penult.pₜ₍ⱼ₎, prices_penult.pₜ₍ⱼ₊₁₎, prices_last.pₜ₍ⱼ₊₁₎
		r′ = prices_penult.rₜ₍ⱼ₊₁₎
		r″ = prices_last.rₜ₍ⱼ₊₁₎

		#(; rₜ₍ⱼ₊₁₎, w, pₜ₍ⱼ₎, pₜ₍ⱼ₊₁₎)
		
		prices_last = (; 
			pₜ₍ⱼ₎ = p′, pₜ₍ⱼ₊₁₎ = p″, rₜ₍ⱼ₊₁₎ = r″, mⱼ = 1.0, mⱼ₋₁ = 0.0, prices_last.w, yⱼ = par.y)
	
		prices_penult = (; 
			pₜ₍ⱼ₎ = p, pₜ₍ⱼ₊₁₎ = p′#=, rₜ₍ⱼ₎ = r′=#, rₜ₍ⱼ₊₁₎ = r′, mⱼ = 0.0, mⱼ₋₁ = 0.0, prices_penult.w, yⱼ₋₁ = par.y)

		(; prices_last, prices_penult)
	end
	
	vⱼ₊₁ = choices.(ω_grid, 0.0, Ref(par), Ref(prices_last); terminal = true)
	tmp = choices.(ω_grid, 0.0, Ref(par), Ref(prices_last); terminal = true, details = true)

	c_last = DimVector(
		select(DataFrame(DimTable(tmp)), Not(:layer1), :layer1 => AsTable).cⱼ,
		ω_dim, name = :c_last
	)
	
	# last period
	#p′ = prices.pₜ₍ⱼ₊₁₎ # last period
	#p  = prices.pₜ₍ⱼ₎   # penultimate period
	#prices_penult = (; prices.r, prices.w, pₜ₍ⱼ₎ = p, pₜ₍ⱼ₊₁₎ = p′)
	
	(; vⱼ, polⱼ) = iterate_backward(vⱼ₊₁, ω_grid, par, prices_penult)

	df = select(DataFrame(DimTable(polⱼ)), :value => AsTable, Not(:value))

	#c_last   = DimVector(out_last.c, name = :c_last)
	c_penult = DimVector(df.cⱼ, ω_dim, name = :c_penult)

	DimStack(c_last, c_penult) # =#
end

# ╔═╡ 3f0ad740-49d9-47f6-9e7c-336e45a43f34
# ╠═╡ disabled = true
#=╠═╡
let
	(; out_last, ω_grid, par, prices) = vfi₀
	ω_grid = DimVector(ω_grid, Dim{:ω}(ω_grid))

	# Version 1: `choices`
	out = choices.(ω_grid, 0.0, Ref(par), Ref(prices); terminal = true, details = true)
	df1 = select(DataFrame(DimTable(out)), :layer1 => AsTable, Not(:layer1))
	df1 = rename(df1, :hⱼ => :h, :cⱼ => :c)

	# Version 2: `out_last`
	df2 = DataFrame(out_last)
	
	df_both = vcat(
		stack(df1, [:h, :c, :κ], :ω),
		stack(df2, [:h, :c, :κ], :ω),
		source = :version => ["v1", "v2"]
	)
	
	data(df_both) * mapping(
		:ω, :value, color =:version, linestyle = :version, layout = :variable
	) * visual(Lines) |> draw	
end
  ╠═╡ =#

# ╔═╡ e10d7dfc-ae95-42cb-8ddf-8e98eb4eae53
egm₀ = let
	(; par, prices_last, ω̲, ω̅, what_is_zero) = penult₀
	ω_grid = prep.grid_sparse 
	#out_last = terminal_value(ω_grid, par, prices_last; what_is_zero)

	(; #=out_last,=# par, prices_last, penult₀.prices_penult, ω_grid)
end

# ╔═╡ 4e012287-9260-49a8-a0a8-f2b3abb6c189
egm_penult = let
	(; out_last, ω_grid, par, prices_penult, prices_last) = egm₀

	#r = prices.r
	(; w) = prices_penult
	y = 1.0
	inc = y * w

	# c = κₚ ⋅ h = κ ⋅ p ⋅ h
	κ( p, p′, r′, (; β, δ, ξ)) = (1-ξ)/ξ * (1 - (1-δ)/(1+r′) * p′/p)
	κₚ(p, p′, r′, par) = p * κ(p, p′, r′, par)
	
	# cₜ₊₁/cₜ = (κ(pₜ, pₜ₊₁, par) / κ(pₜ₊₁, pₜ₊₂, par))^(ξ * (1-σ)/σ)
	# c′/c = (κ(p, p′, par) / κ(p′, p′′, par))^(ξ * (1-σ)/σ)

	function get_c(c′, (; p, p′, p″, r′, r″), par)
		(; ξ, γ, β) = par
		σ = γ
		
		c′ * (β*(1+r′))^(-1/σ) / (κₚ(p, p′, r′, par) / κₚ(p′, p″, r″, par))^(ξ * (1-σ)/σ)
	end

	function get_h(c, (; p, p′, r′), par)
		c / κₚ(p, p′, r′, par)
	end

	function get_ω(c, h, ω′, (; p, p′, r′), par)
		(; δ) = par
		
		#inc = 1.0
		c + p * h * (1 - (1-δ)/(1+r′) * p′/p) + ω′/(1+r′) - inc
	end
	
	c′ = DimVector(out_last.c, name = :c_last)
	p, p′, p″ = prices_penult.pₜ₍ⱼ₎, prices_penult.pₜ₍ⱼ₊₁₎, prices_last.pₜ₍ⱼ₊₁₎
	r′ = prices_penult.rₜ₍ⱼ₊₁₎
	r″ = prices_last.rₜ₍ⱼ₊₁₎

	prices = (; p, p′, p″, r′, r″)
	
	# c′(c)
	c_ω′ =	DimVector(get_c.(c′, Ref(prices), Ref(par)), name = :c_penultimate)
	h_ω′ = DimVector(get_h.(c_ω′, Ref(prices), Ref(par)), name = :h_penultimate)
	ω′ = ω_grid
	ω = DimVector(get_ω.(c_ω′, h_ω′, ω′, Ref(prices), Ref(par)), name = :ω_penultimate)

	
	lines(collect(ω′), collect(c′), label = "last")
	lines!(collect(ω), collect(c_ω′), label = "penult")

	c_itp = linear_interpolation(ω, c_ω′, extrapolation_bc = Line())

	c_penult = DimVector(c_itp.(ω_grid), Dim{:ω}(ω_grid), name = :c_penult)

	DimStack(c′, c_penult)
end

# ╔═╡ 194a3206-07f0-44be-939b-b13b7202686d
md"""
## Test with VFI
"""

# ╔═╡ 67f9ea5a-157d-4d35-95e2-b0a57070a5f5
vfi_test = let
	(; price_paths, par, j_dim, grid_dense) = prep
	(; J) = par

	(; df, policy) = solve_backward_forward(par, grid_dense; price_paths)
end

# ╔═╡ ade5fcdb-59ae-4dec-9493-df9c2bc68ae9
let
	(; df) = vfi_test

	@chain df begin
		@aside @info _
	#	@subset(:j < J-1)
	#	@subset(:j < 0.8 * J)
		stack(Not(:j))
		data(_) * mapping(:j, :value, layout = :variable) * visual(Lines)
		draw(facet = (; linkyaxes = false))
	end 
	#(; cⱼ, hⱼ, phⱼ, aⱼ₊₁, ωⱼ₊₁) = first(out)
	
end

# ╔═╡ 8581b7ca-9297-4450-b774-471141e80aad


# ╔═╡ cdf1f14e-2ef5-47c1-bf0c-8ef1de9b97d0
let
	ph = 0.23611
	c = 1.2252
	(; ξ, δ, β) = prep.par
	r = 1/β - 1
	p = 0.5
	
	exp_Y = c / (1-ξ)
	@info exp_Y, ph / ξ

	ω = 0
	inc = 1
	a = (inc + ω) - c - ph

	(1+r) * a + (1-δ) * ph
	
end

# ╔═╡ c82bc57d-95eb-4832-9b79-70dd71781af5
md"""
```julia
let
	ωⱼ = 0.1
	inc = 1.0
	cⱼ = 1.0
	hⱼ = 0.5
	#ωⱼ₊₁ = (1+r) * aₜ₊₁ + (1-δ) * pₜ₊₁ * hⱼ == 0.0
	# ⟹
	# aₜ₊₁ = - (1-δ)/(1+r) * pₜ₊₁ * hⱼ

	# cⱼ + pₜ * hⱼ + aₜ₊₁  == inc + ωⱼ
	# ⟹
	#cⱼ + pₜ * hⱼ * (1 - (1-δ)/(1+r) * pₜ₊₁/pₜ) = inc + ωⱼ
	cⱼ + pₜ * hⱼ = (inc + ωⱼ) / (1 - (1-δ)/(1+r) * pₜ₊₁/pₜ)
end
```
"""

# ╔═╡ fb51b371-6b05-4da2-af1b-ef2d749e5af7
function c_last(ωⱼ, (; wₜ, yⱼ#=, rₜ, rₜ₊₁, pₜ₊₁, pₜ=#), (; δ, ξ)) 
	# ω_J == 0
	exp_J = (ωⱼ + wₜ * yⱼ) #/ (1 - (1-δ)/(1+rₜ₊₁) * pₜ₊₁/pₜ)
	(1-ξ) * exp_J
end

# ╔═╡ 91bb2ed6-deb6-43ff-815e-906be0f11516
aₜ_from_hₜ₋₁((; pₜ, rₜ), (; δ); ωₜ, hₜ₋₁) = 
				(; aₜ = (ωₜ - pₜ * (1-δ) * hₜ₋₁) / (1 + rₜ))

# ╔═╡ f7c86441-5ee4-494b-a11a-cdf14a1bea39
ωₜ₋₁_from_ahc((; pₜ₋₁, wₜ₋₁, yⱼ₋₁, mⱼ₋₁); aₜ, hₜ₋₁, cₜ₋₁) = 
				(; ωₜ₋₁ = (1-mⱼ₋₁) * aₜ + pₜ₋₁ * hₜ₋₁ + cₜ₋₁ - wₜ₋₁ * yⱼ₋₁)

# ╔═╡ e522d891-bd9d-4ae5-b13d-ff02e545aa3c
function κⱼ(pₜ₍ⱼ₎, pₜ₍ⱼ₊₁₎, rₜ₍ⱼ₊₁₎, mⱼ, (; ξ, δ))
	(1-ξ)/ξ * (
		pₜ₍ⱼ₎ - (1-mⱼ) * (1-δ)/(1+rₜ₍ⱼ₊₁₎) * pₜ₍ⱼ₊₁₎
	)
end

# ╔═╡ 2daa3181-79b5-4ae7-86fb-1b666be75e49
hⱼ_from_cⱼ((; pₜ, pₜ₊₁, rₜ₊₁, mⱼ), par; cⱼ) = (; hⱼ = cⱼ / κⱼ(pₜ, pₜ₊₁, rₜ₊₁, mⱼ, par))

# ╔═╡ b8d2fbac-5ef6-4900-a7f1-3e2633882f2a
function iterate_forward(prices, par; cⱼ, stateⱼ)
	ωⱼ = stateⱼ
	(; wₜ, yⱼ, rₜ, rₜ₊₁, mⱼ, pₜ, pₜ₊₁) = prices
	(; δ) = par
	
	(; hⱼ) = hⱼ_from_cⱼ(prices, par; cⱼ)

	if mⱼ == 1.0
		(; ξ) = par
		ωⱼ₊₁ = 0.0
		# expenditures in last period if assuming ωⱼ₊₁ = 0.0
		exp_J = (ωⱼ + wₜ * yⱼ) #/ (1 - (1-δ)/(1+rₜ₊₁) * pₜ₊₁/pₜ)
		cⱼ   = (1-ξ) * exp_J 
		hⱼ   =    ξ  * exp_J  / pₜ
		aⱼ₊₁ = -(1-δ)/(1+rₜ₊₁) * pₜ₊₁ * hⱼ
		#ωⱼ₊₁ = (1-δ) * pₜ₊₁ * hⱼ
		ωⱼ₊₁_0 = ωⱼ₊₁
	else
		aⱼ₊₁ = (ωⱼ + yⱼ * wₜ - cⱼ - pₜ * hⱼ) / (1-mⱼ)

		ωⱼ₊₁_0 = pₜ₊₁ * (1-δ) * hⱼ + (1+rₜ₊₁) * aⱼ₊₁
	end
	
	# ω = p * h * (1-δ) + (1+r) a
	# h = 
	# handling the constraint
	if ωⱼ₊₁_0 < 0
		#@info "ωⱼ₊₁ < 0"
		ωⱼ₊₁ = 0
		# ⟹₁ pₜ₊₁ * hⱼ * (1-δ)/(1+rₜ₊₁) = -aⱼ₊₁
		# ⟹₂ pₜ * hⱼ + cⱼ == yⱼ * wₜ + ωⱼ
		#⟹ pₜ * (cⱼ * κⱼ) + cⱼ = cⱼ(pₜ * κⱼ + 1 ) == yⱼ * wₜ + ωⱼ
		#⟹ cⱼ = yⱼ * wₜ + ωⱼ / (pₜ * κⱼ + 1)
		κ = κⱼ(pₜ, pₜ₊₁, rₜ, mⱼ, par)
		cⱼ = (yⱼ * wₜ + ωⱼ) / (pₜ * κ + 1)
		hⱼ = cⱼ * κ
		aⱼ₊₁ = - pₜ₊₁ * hⱼ * (1-δ)/(1+rₜ₊₁)
	else
		ωⱼ₊₁ = ωⱼ₊₁_0
	end

	other = (; ωⱼ₊₁, ωⱼ₊₁_0, cⱼ, hⱼ, aⱼ₊₁, pₜ)
	
	(; cⱼ, stateⱼ₊₁ = ωⱼ₊₁, other)
end

# ╔═╡ 87aa7ba3-ea06-454a-a7f6-b43f9b47ca26
function cₜ₋₁_from_cₜ((; pₜ₋₁, pₜ, pₜ₊₁, rₜ₊₁, rₜ, mⱼ, mⱼ₋₁), par; cₜ)
		(; ξ, σ, β) = par

		cₜ₋₁ = cₜ / (β * (1+rₜ))^(1/σ) / (
			κⱼ(pₜ₋₁, pₜ, rₜ,  mⱼ₋₁, par) / κⱼ(pₜ, pₜ₊₁, rₜ₊₁, mⱼ, par)
			)^(ξ * (1-σ) / σ)

		(; cₜ₋₁)
	end

# ╔═╡ cbe2abf3-6286-4cae-9452-88f66465a072
hₜ₋₁_from_cₜ₋₁((; pₜ₋₁, pₜ, rₜ, mⱼ₋₁), par; cₜ₋₁) = (; hₜ₋₁ = cₜ₋₁ / κⱼ(pₜ₋₁, pₜ, rₜ, mⱼ₋₁, par))

# ╔═╡ 71720b73-99cb-496e-9b7e-af913e4feb9b
function iterate_backward_(stateⱼ, cⱼ, prices, par)
	cₜ = cⱼ
	ωₜ = stateⱼ
	(; cₜ₋₁) = cₜ₋₁_from_cₜ(  prices, par; cₜ)
	(; hₜ₋₁) = hₜ₋₁_from_cₜ₋₁(prices, par; cₜ₋₁) 
	(; aₜ)   = aₜ_from_hₜ₋₁(  prices, par; ωₜ, hₜ₋₁)
	(; ωₜ₋₁) = ωₜ₋₁_from_ahc( prices;      aₜ, hₜ₋₁, cₜ₋₁)
		
	(; cⱼ₋₁=cₜ₋₁, hⱼ₋₁=hₜ₋₁, stateⱼ₋₁=ωₜ₋₁)
end

# ╔═╡ 99dd4792-3b16-456c-8671-bab6001c4663
egm_penult2 = let
	(; #=out_last, =# ω_grid, par, prices_penult, prices_last) = egm₀

	par = (; par..., σ = par.γ)
	#r = prices.r
	(; w) = prices_penult
	(; y) = par
	inc = y * w

	(; prices_last, prices_penult) = let
		p, p′, p″ = prices_penult.pₜ₍ⱼ₎, prices_penult.pₜ₍ⱼ₊₁₎, prices_last.pₜ₍ⱼ₊₁₎
		r′ = prices_penult.rₜ₍ⱼ₊₁₎
		r″ = prices_last.rₜ₍ⱼ₊₁₎
	
		prices_last = (; 
			pₜ₊₁ = p″, rₜ₊₁ = r″, mⱼ = 0.0, mⱼ₋₁ = 0.0, wₜ = prices_last.w, yⱼ = par.y)
	
		prices_penult = (; 
			pₜ₋₁ = p, pₜ = p′, pₜ₊₁ = p″, rₜ = r′, rₜ₊₁ = r″, mⱼ = 0.0, mⱼ₋₁ = 0.0, wₜ₋₁ = w, yⱼ₋₁ = par.y)

		(; prices_last, prices_penult)
	end
	
	# USES `c_last`
	c′ = DimVector(
		c_last.(ω_grid, Ref(prices_last), Ref(par)),
		Dim{:ω}(ω_grid), name = :c_last)

	# USES `iterate_backward`
	out = iterate_backward_.(ω_grid, c′, Ref(prices_penult), Ref(par))

	df = select(DataFrame(DimTable(out)), Not(:layer1), :layer1 => AsTable)

	c = DimVector(
		linear_interpolation(df.stateⱼ₋₁, df.cⱼ₋₁, extrapolation_bc = Line())(ω_grid),
		Dim{:ω}(ω_grid), name = :c_penult)

	DimStack(c, c′)
end

# ╔═╡ ba78d7cb-857d-45b3-a201-323e20b572fd
one_iteration = let
	df_vfi = DataFrame(vfi_penult)
	df_egm = DataFrame(egm_penult2)

	df = vcat(df_vfi, df_egm, source = :method => ["vfi", "egm"])
end	

# ╔═╡ fa69ae76-0b5b-42fd-9186-22cd0191153a
let
	@chain one_iteration begin
		stack([:c_last, :c_penult])
		@subset(-0.1 < :ω < 0.1)
		data(_) * mapping(:ω, :value, layout = :variable, 
			color = :method => sorter("vfi", "egm"),
			linestyle = :method => sorter("vfi", "egm")
		) * visual(Lines)
		draw
	end
end

# ╔═╡ 232386a5-72eb-44cc-b91e-06b6948c2baf
let
	one_iteration
	(; J) = prep.par
	(; policy) = vfi_test
	#policy.cⱼ

	df = @chain policy begin
		DimTable
		DataFrame
		select(:value => AsTable, Not(:value))
		@select(:j, :ω, :c = :cⱼ, :method = "vfi_2")
		@subset(:j ∈ J-2:J-1)
		unstack(:j, :c)
		rename(string(J-1) => :c_last, string(J-2) => :c_penult)
		vcat(_, one_iteration)
		#vcat(_, egm_pol, source = :method => ["vfi", "egm"])
	end

	@chain df begin
		stack([:c_last, :c_penult])
		#@subset(-0.1 < :ω < 0.1)
		data(_) * mapping(:ω, :value, layout = :variable, 
			color = :method, # => sorter("vfi", "egm"),
			linestyle = :method, # => sorter("vfi", "egm")
		) * visual(Lines)
		draw
	end
	#= @chain df begin
		@subset(:j > 20)# 10..15) #(par.J):-1:(par.J-10))
		data(_) * mapping(:ω, :c, layout = :j => nonnumeric, color = :method, linestyle = :method) * visual(Lines)
		draw(facet = (; linkyaxes = false))
	end =#
end

# ╔═╡ 8611066a-4ef4-47b5-8247-2f3268bf1d54
function tuple_of_prices((; ps, r, w), (; y, J); t, j)
	mⱼ = j == J ? 1.0 : 0.0
	mⱼ₋₁ = 0.0
	
	p = ps

	if t ≥ 1
		pₜ₋₁, pₜ, pₜ₊₁ = p[t = At(t-1:t+1)]
		rₜ₋₁, rₜ, rₜ₊₁ = r, r, r #[t = At(t-1:t+1)]
		wₜ₋₁, wₜ       = w, w #[t = At(t-1:t)]
		nt1 = (; pₜ₋₁, pₜ, pₜ₊₁, rₜ₋₁, rₜ, rₜ₊₁, wₜ₋₁, wₜ)
	else
		pₜ, pₜ₊₁ = p[t = At(t:t+1)]
		rₜ, rₜ₊₁ = r, r #[t = At(t:t+1)]
		wₜ       = w #[t = At(t)]
		nt1 = (; pₜ, pₜ₊₁, rₜ, rₜ₊₁, wₜ)
	end
	
	if j ≥ 1
		yⱼ₋₁, yⱼ       = y[j = At(j-1:j)]
		#mⱼ₋₁, mⱼ       = m, m #[j = At(j-1:j)]

		nt2 = (; yⱼ₋₁, yⱼ, mⱼ₋₁, mⱼ)
	else
		yⱼ       = y[j = At(j)]
		#mⱼ       = m #[j = At(j)]
		nt2 = (; yⱼ, mⱼ)
	end

	return (; nt1..., nt2...)
end

# ╔═╡ ca0fdef0-3469-4f67-9383-420b9bda296f
# ╠═╡ disabled = true
#=╠═╡
let
	(; price_paths, par, j_dim) = prep

	(; J, ξ) = par
	born = 1
	j = J
	t = born + j
	
	prices = tuple_of_prices(price_paths, par; t, j)

	ωⱼ = 5.0 # ω_J
	(; yⱼ, wₜ, pₜ) = prices # y_J

	cⱼ = c_last(ωⱼ, prices, par)
	#@assert cⱼ ≈ (1-ξ) * (ωⱼ + yⱼ * wₜ)
	
	c = zeros(j_dim, name = :c)
	c[j = At(J-1)] = cⱼ
	h = zeros(j_dim, name = :h)

	for j ∈ J:-1:1
		t = born + j
		prices = tuple_of_prices(price_paths, par; t, j)
		main = iterate_backward_(ωⱼ, cⱼ, prices, par)
		
		(; cⱼ₋₁, hⱼ₋₁, stateⱼ₋₁) = main
		@info (; t, j, cⱼ₋₁, hⱼ₋₁, stateⱼ₋₁, pₜ = prices.pₜ)

		c[j = At(j-1)] = cⱼ₋₁
		h[j = At(j-1)] = hⱼ₋₁
	
		ωⱼ = stateⱼ₋₁
		cⱼ = cⱼ₋₁
	end

	#lines(h[j = At(0:J-2)])
	#=
	for j ∈ J-10:J
		t = born + j
		prices = tuple_of_prices(M, price_paths, par; t, j)
		stateⱼ = ωⱼ
		main = iterate_forward(M, prices, par; cⱼ, stateⱼ)

		#(; other
		#other = (; ωⱼ₊₁, ωⱼ₊₁_0, cⱼ, hⱼ, aⱼ₊₁)
	
		(; cⱼ, stateⱼ₊₁, other) = main

		ωⱼ = stateⱼ₊₁

		@info (; j, cⱼ, ωⱼ₊₁ = stateⱼ₊₁)
		#c
	end #	=#
end
	
  ╠═╡ =#

# ╔═╡ 498eb9c4-f499-44df-901d-1078a5fa1902
"Takes `price_paths`"
function solve_backward_forward_egm(par, grid; price_paths, init_state, j_init = 0, t_born = 0)
	statename = :ω

	(; J, y, m) = par

	state_dim = Dim{statename}(grid) # a or net worth
	j_dim = Dim{:j}(0:J-1)
	j_sim_dim = Dim{:j}(j_init:J-1)
	
	c = zeros(state_dim, j_dim, name = :c)
	
	## SOLVE BACKWARDS
	t_J = t_born + J
	prices_J = tuple_of_prices(price_paths, par; t=t_J, j=J-1)
	c[j = At(J-1)] .= c_last.(grid, Ref(prices_J), Ref(par))
	
	
	for j ∈ (J-1):-1:(j_init+1)
		t = t_born + j
		prices = tuple_of_prices(price_paths, par; t, j)
		
		(; cⱼ₋₁, stateⱼ₋₁) = DataFrame(
			iterate_backward_.(grid, c[j = At(j)], Ref(prices), Ref(par))
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

	
	TT = typeof((; ωⱼ₊₁=1.0, ωⱼ₊₁_0=1.0, cⱼ=1.0, hⱼ=1.0, aⱼ₊₁=1.0, pₜ=1.0))
	other_paths = DimVector(
					Vector{TT}(undef, J),
					j_dim, name = :other
				)
	
	(; r, w) = price_paths
	
	for j ∈ j_init:(J-1)
		t = t_born + j
		prices = tuple_of_prices(price_paths, par; t, j)
			
		stateⱼ = path_state[j = At(j)]
			
		cⱼ_itp = LinearInterpolation(grid, c[j = At(j)], extrapolation_bc = Line())
			
		cⱼ = cⱼ_itp(stateⱼ)

		(; cⱼ, stateⱼ₊₁, other) = iterate_forward(prices, par; cⱼ, stateⱼ)
		
		path_choice[j = At(j)] = cⱼ
		other_paths[ j = At(j)] = other
		path_next_state[j = At(j)] = stateⱼ₊₁
		
		if j < J-1
			path_state[j = At(j+1)] = stateⱼ₊₁
		end	
	end

	sim = DimStack(path_state, path_choice, path_next_state, other_paths)
	sim_df = DataFrame(sim)

	 select!(sim_df, :other => AsTable, Not(:other))

	(; sim_df, sim, state_path=path_state, other_path=other_paths, c)

end

# ╔═╡ 0c134c1b-5fde-45a2-b43e-6edf6127599b
let
	(; price_paths, par, j_dim, grid_sparse) = prep
	#grid = sort([0.0; range(-1.0, 1.0, length = 100)])
	init_state = 0.0 #(; ω = 0.0)
	out = solve_backward_forward_egm(par, grid_sparse; price_paths, init_state)

	(; sim_df, c) = out

	egm_pol = @chain c begin
		DataFrame
		#@subset(:j ∈ (par.J):-1:(par.J-10))
		#data(_) * mapping(:ω, :c, group = :j => nonnumeric, color = :j) * visual(Lines)
		#draw
	end
	(; policy) = vfi_test
	df = @chain policy begin
		DimTable
		DataFrame
		select(:value => AsTable, Not(:value))
		@select(:j, :ω, :c = :cⱼ)
		vcat(_, egm_pol, source = :method => ["vfi", "egm"])
	end

	@chain df begin
		@subset(:j > 20)# 10..15) #(par.J):-1:(par.J-10))
		data(_) * mapping(:ω, :c, layout = :j => nonnumeric, color = :method, linestyle = :method) * visual(Lines)
		draw(facet = (; linkyaxes = false))
	end
#	lines(c[j = At(par.J-3)])
	
#=	@chain sim_df begin
		@aside @info @subset(_, :j ∈ [0, 1, 2, par.J-2, par.J-1])
		@subset(:j < par.J-1)
		stack([:cⱼ, :hⱼ, :aⱼ₊₁, :ω, :ωⱼ₊₁, :pₜ])
		data(_) * mapping(:j, :value, layout = :variable) * visual(Lines)
		draw(facet = (; linkyaxes = false))
	end
	# =#
end

# ╔═╡ 3ed4f678-f428-4732-b2d3-db68c22669d8
md"""
# Appendix
"""

# ╔═╡ a6264cf7-e6dc-410b-856b-241be2c858c9
TableOfContents()

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
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
AlgebraOfGraphics = "~0.9.2"
CairoMakie = "~0.13.1"
Chain = "~0.6.0"
DataFrameMacros = "~0.4.1"
DataFrames = "~1.7.0"
DimensionalData = "~0.29.12"
Interpolations = "~0.15.1"
PlutoTest = "~0.2.2"
PlutoUI = "~0.7.61"
Statistics = "~1.11.1"
StatsBase = "~0.34.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "a054e3bfb10392cf95fd805c54052a8f4bc55c06"

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
git-tree-sha1 = "cd8b948862abee8f3d3e9b73a102a9ca924debb0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.2.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AlgebraOfGraphics]]
deps = ["Accessors", "Colors", "Dates", "Dictionaries", "FileIO", "GLM", "GeoInterface", "GeometryBasics", "GridLayoutBase", "Isoband", "KernelDensity", "Loess", "Makie", "NaturalSort", "PlotUtils", "PolygonOps", "PooledArrays", "PrecompileTools", "RelocatableFolders", "StatsBase", "StructArrays", "Tables"]
git-tree-sha1 = "62c3acd999abce35d0ae164167838d7f9207b214"
uuid = "cbdf2221-f076-402e-a563-3d30da359d67"
version = "0.9.2"

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

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

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
git-tree-sha1 = "6d76f05dbc8b7a52deaa7cdabe901735ae7b6724"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.13.1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

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
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

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
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
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
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

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
git-tree-sha1 = "1cdab237b6e0d0960d5dcbd2c0ebfa15fa6573d9"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.4"

[[deps.DimensionalData]]
deps = ["Adapt", "ArrayInterface", "ConstructionBase", "DataAPI", "Dates", "Extents", "Interfaces", "IntervalSets", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "PrecompileTools", "Random", "RecipesBase", "SparseArrays", "Statistics", "TableTraits", "Tables"]
git-tree-sha1 = "89ce15a7893dd8cc80cc224942b7a5e0296bb1e8"
uuid = "0703355e-b756-11e9-17c0-8b28908087d0"
version = "0.29.12"

    [deps.DimensionalData.extensions]
    DimensionalDataAlgebraOfGraphicsExt = "AlgebraOfGraphics"
    DimensionalDataCategoricalArraysExt = "CategoricalArrays"
    DimensionalDataDiskArraysExt = "DiskArrays"
    DimensionalDataMakie = "Makie"
    DimensionalDataPythonCall = "PythonCall"
    DimensionalDataStatsBase = "StatsBase"

    [deps.DimensionalData.weakdeps]
    AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
    CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
    DiskArrays = "3c3547ce-8d99-4f5e-a174-61eb10b00ae3"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
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
git-tree-sha1 = "0b4190661e8a4e51a842070e7dd4fae440ddb7f4"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.118"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

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
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

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
git-tree-sha1 = "063512a13dbe9c40d999c439268539aa552d1ae6"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "8cc47f299902e13f90405ddb5bf87e5d474c0d38"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.2+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "7de7c78d681078f027389e067864a8d53bd7c3c9"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+3"

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

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "d52e255138ac21be31fa633200b65e4e71d26802"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.6"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

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
git-tree-sha1 = "f0895e73ba6c469ec8efaa13712eb5ee1a3647a3"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.2"

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
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

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
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

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
deps = ["CRlibm_jll", "LinearAlgebra", "MacroTools", "OpenBLASConsistentFPCSR_jll", "RoundingEmulator"]
git-tree-sha1 = "7b3603d3a5c52bcb18de8e46fa62e4176055f31e"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.25"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

    [deps.IntervalArithmetic.weakdeps]
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
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
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

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
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "df37206100d39f79b3376afb6b9cee4970041c61"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89211ea35d9df5831fca5d33552c02bd33878419"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e888ad02ce716b319e6bdb985d2ef300e7089889"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.3+0"

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
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "9680336a5b67f9f9f6eaa018f426043a8cd68200"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.22.1"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "c731269d5a2c85ffdc689127a9ba6d73e978a4b1"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.9.0"

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
git-tree-sha1 = "f45c8916e8385976e1ccd055c9874560c257ab13"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.2"

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

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

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
git-tree-sha1 = "5e1897147d1ff8d98883cda2be2187dcf57d8f0c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.15.0"
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
version = "0.8.1+4"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a9697f1d06cc3eb3fb3ad49cc67f2cfabaac31ea"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.16+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "966b85253e959ea89c53a9abebbf2e964fbf593b"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.32"

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
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

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
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

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
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

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
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
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
git-tree-sha1 = "5a3a31c41e15a1e042d60f2f4942adccba05d3c9"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.0"

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
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

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
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "f21231b166166bebc73b99cea236071eb047525b"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.3"

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
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

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
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

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

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56c6604ec8b2d82cc4cfe01aa03b00426aac7e1f"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.4+1"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e9216fdcd8514b7072b43653874fd688e4c6c003"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.12+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89799ae67c17caa5b3b5a19b8469eeee474377db"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.5+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c57201109a9e4c0585b208bb408bc41d205ac4e9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.2+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6dba04dbfb72ae3ebe5418ba33d087ba8aa8cb00"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.1+0"

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
git-tree-sha1 = "068dfe202b0a05b8332f1e8e6b4080684b9c7700"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.47+0"

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
# ╟─134ba669-b2e2-40c3-872b-b79d14d16544
# ╟─dd01a6ba-ea0b-4f10-8597-c7d5f36cf5fe
# ╟─ee8b2332-192c-4cd3-b89a-97fd5db07fa4
# ╟─2ce2146a-4158-4280-86d3-7d73937262e0
# ╟─71fcb5c5-073f-438c-bf17-0d8f666facac
# ╟─e49fee0a-148b-4a91-92f8-b9536f2a395d
# ╟─867e51e9-97df-4236-9c53-084990ccfe23
# ╟─62fef877-a42e-4a86-af78-dcf4e54b779a
# ╠═31e0a1a9-2216-4a70-9f81-2f16267fcb5c
# ╠═0405682d-9757-48c2-81d9-03abe696af4b
# ╟─6e02d06e-5ea0-4bcc-86a9-29182356cb3c
# ╟─73f50ec7-6763-47d2-9560-167699c49161
# ╠═4f54a3c5-b448-4987-8258-de77d69ef92f
# ╠═8fd0bef0-40af-4b15-8d46-11caf051067f
# ╟─b0270296-aba9-4c53-87bb-550550944997
# ╠═5c197725-0cff-4fbd-9e29-cca5cc48f28c
# ╠═2fa9c3b9-1759-4c75-a264-40995942716c
# ╟─6f3e64fa-3280-4b12-997d-b7355e11b689
# ╠═3f0ad740-49d9-47f6-9e7c-336e45a43f34
# ╟─99dcf6b6-104b-4ccc-afc1-72355dc0e21e
# ╟─d5831143-df0d-43aa-9fc0-1a312feaa3dc
# ╠═e10d7dfc-ae95-42cb-8ddf-8e98eb4eae53
# ╠═99dd4792-3b16-456c-8671-bab6001c4663
# ╠═4e012287-9260-49a8-a0a8-f2b3abb6c189
# ╟─b1b861ce-0659-11f0-094d-67da13f58563
# ╠═bae86455-59b4-42fc-9f0e-b14ed67b9e5f
# ╠═9666b117-8273-4b32-a757-cb5a7b0ef107
# ╠═df8d0140-cd53-4864-b5f6-900661da5113
# ╠═8f170e55-a231-44ba-981f-b3bca28bcf9c
# ╠═f2457772-3de4-47d2-946a-6f6e15a48fc4
# ╟─cb7f3832-c1e9-48de-b21e-2390b6d1a2e8
# ╠═ba62cdca-ac5a-480b-b930-7b121156eba6
# ╠═d00d0c99-44e8-4131-96d7-9f7956bf9629
# ╠═ceda67ec-c351-4776-9980-0f2c27e0ea02
# ╠═c5ca1482-7252-4220-92ec-a67900f036e4
# ╠═fd3cb5a8-8717-4d2f-8e58-4151a25f79dd
# ╠═8e018136-64f9-4a1f-975b-2d5ea5b08498
# ╠═829c6847-89a5-4a4a-807e-b65b04dedc0b
# ╠═9c9ceccb-fb28-419b-a9ed-37aaf6406f3c
# ╠═f61bf1a6-634f-4a47-8c28-efd42ce87747
# ╠═1dbb9a6c-831e-4519-a862-668e61ea6a4e
# ╠═53e1c988-fda6-473f-ba2c-b460483813d1
# ╟─4ee76793-be3c-4849-ab1c-cc64d0cdf906
# ╠═f2a067a9-b486-45dc-8d37-8b26b83ebfd2
# ╠═ee007b1f-e5b5-475e-b672-53483ad42f88
# ╠═c94afe0e-a8bf-4640-9874-f3cecab16b5f
# ╠═230bdd4d-99fb-4adb-971b-dbef024b3e20
# ╠═e27ecff7-bf67-4c99-8970-001b078f48ff
# ╠═a4e70adc-7780-4456-96db-4e72feb2d032
# ╠═e8249393-d1bb-41b0-a36c-8447bd14bc5e
# ╠═683216f2-51d7-4650-9263-8bc234539460
# ╠═0471e65c-5ca8-49a5-ac32-ebddb01b9738
# ╠═0af642fe-5083-4db8-8384-f0b572fa88a3
# ╟─d47576ee-d6e5-4fa3-b3c4-bee9ca49abd8
# ╠═1e248059-d315-4206-8fd2-a609954b46d7
# ╠═ca0fdef0-3469-4f67-9383-420b9bda296f
# ╠═1f66f226-aafc-4fa6-bfc3-2adc7f11d93d
# ╠═ba78d7cb-857d-45b3-a201-323e20b572fd
# ╠═fa69ae76-0b5b-42fd-9186-22cd0191153a
# ╠═232386a5-72eb-44cc-b91e-06b6948c2baf
# ╠═0c134c1b-5fde-45a2-b43e-6edf6127599b
# ╟─194a3206-07f0-44be-939b-b13b7202686d
# ╠═67f9ea5a-157d-4d35-95e2-b0a57070a5f5
# ╠═ade5fcdb-59ae-4dec-9493-df9c2bc68ae9
# ╠═8581b7ca-9297-4450-b774-471141e80aad
# ╠═cdf1f14e-2ef5-47c1-bf0c-8ef1de9b97d0
# ╠═62ca7314-4892-413c-98aa-dcdf05ab6a7a
# ╠═498eb9c4-f499-44df-901d-1078a5fa1902
# ╠═c82bc57d-95eb-4832-9b79-70dd71781af5
# ╠═fb51b371-6b05-4da2-af1b-ef2d749e5af7
# ╠═91bb2ed6-deb6-43ff-815e-906be0f11516
# ╠═f7c86441-5ee4-494b-a11a-cdf14a1bea39
# ╠═e522d891-bd9d-4ae5-b13d-ff02e545aa3c
# ╠═b8d2fbac-5ef6-4900-a7f1-3e2633882f2a
# ╠═2daa3181-79b5-4ae7-86fb-1b666be75e49
# ╠═87aa7ba3-ea06-454a-a7f6-b43f9b47ca26
# ╠═cbe2abf3-6286-4cae-9452-88f66465a072
# ╠═71720b73-99cb-496e-9b7e-af913e4feb9b
# ╠═8611066a-4ef4-47b5-8247-2f3268bf1d54
# ╟─3ed4f678-f428-4732-b2d3-db68c22669d8
# ╠═ba4d8cac-7493-4c69-83b5-78f7d45ff305
# ╠═a6264cf7-e6dc-410b-856b-241be2c858c9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
