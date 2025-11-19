### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ b0b92aec-7643-47b7-a0fc-b44637a6fe91
using PlutoUI: Slider

# ╔═╡ 4725c07e-580d-4bba-ad77-89a189491308
using AlgebraOfGraphics: AlgebraOfGraphics as AoG

# ╔═╡ 92a909f0-7697-4f3a-9c92-10003144783d
using CSV

# ╔═╡ 63333781-ee80-458f-9287-094e2e7529b8
using PlutoUI

# ╔═╡ abcd5019-bfaa-467e-903b-7e711ba0d3a4
using Chain, DataFrames, DataFrameMacros

# ╔═╡ 8ec8b772-69b1-4a06-b800-3bd02610a105
using CairoMakie, AlgebraOfGraphics

# ╔═╡ 313d3886-f84d-4561-85e2-9ff9cc60ffaf
using PlutoTest

# ╔═╡ 42c0154b-026d-4a26-b291-bbb87e124611
using StatsBase

# ╔═╡ 623ed842-a49c-4ac2-9a8b-8e69367ba517
using DimensionalData

# ╔═╡ 759aa4ff-7f40-428e-8285-066fdc02982b
using LinearAlgebra: dot, norm

# ╔═╡ a9d7e50a-eb07-4721-96bd-54765058ad60
using Roots: find_zero

# ╔═╡ 2b534ad4-a62d-4983-b43c-274f609c159d
using Interpolations

# ╔═╡ af6ef77a-f8e3-45ca-b99b-74994f247470
using QuantEcon: QuantEcon, rouwenhorst, MarkovChain, stationary_distributions

# ╔═╡ 5f6351f0-41ff-426e-be67-00572cf96f88
using Statistics: mean

# ╔═╡ 5641018b-f5ff-42aa-a153-83c2da5eea88
using PlutoLinks: ingredients, @ingredients

# ╔═╡ e5b31309-a7cb-43e5-8173-f5ed8bbf316e
md"""
# Adjust the code for period $\ne$ 1

* [x] discount factor β
* [x*] income: lifecycle component
* [x] income: risky component
* [x] depreciation rate ``\delta_P = 1 - (1-\delta)^P
* [x] mortality risk
* [ ] housing services? In a multi-year period, the services multiply with the length of the period
"""

# ╔═╡ 236eb5d3-65c2-409e-b5b9-b4006f04755a
md"""
## Discounting
"""

# ╔═╡ 516f737f-2729-4a7a-8395-f8227ba790b9
function β_AMMR(; age_min = 20, age_max = 96, β̄ = 0.9655, ξ = 0.00071)

	J = age_max - age_min
	 
	β = DimVector(
		[exp(j * log(β̄) + ξ * (j - (40 - age_min))^2) for j ∈ 0:J],
		Dim{:j}((0:J))
	)

	β[j = At(0:J-1)]

end

# ╔═╡ 60ba94e8-c7fd-4b3d-89fb-bec0d5bc815f
md"""
## Income: lifecycle component
"""

# ╔═╡ f3f4e738-89f5-4666-bbd9-cdc6c3650bb7
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

# ╔═╡ e26f2508-c8a4-4057-a5fd-740e19e42e12
#scatterlines(h̄_marcelo)

# ╔═╡ db007083-7d8f-4f8f-9629-96df08e514f4
md"""
## Income: transitory component
"""

# ╔═╡ 44909380-710e-4c4d-8ea9-cc03c62309e0
md"""
## Mortality
"""

# ╔═╡ 939a6edf-acf5-40e1-99a4-51e75394e55b
md"""
## Bequests
"""

# ╔═╡ c9ebec81-aa29-49db-8c8e-209741db0372
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

# ╔═╡ 1b7e6662-ab53-4a63-81a3-4b2b626e0e7e
md"""
**Note** ``F`` sums to 1.0. But shouldn't ``\sum_j \pi_j F_j = 1.0``? (So that ``F`` is the expected bequest per person, not the total bequests per cohort.)
"""

# ╔═╡ f40fa3a0-8064-4c6b-b5c2-9cd838919c9d
lines(F_marcelo)

# ╔═╡ 88d9865b-796d-4ed5-a2d5-862236f3ae54
md"""
## Check against simpler version
"""

# ╔═╡ a87f8586-fa96-4940-9f88-f97831ed9c1a
#=╠═╡
let
	(; par, out) = out_18
	
	par_beq = get_cali_test(; J_P = par.J +1, risk = true, bequests = true).par
	@assert par_beq.h == par.h
	
	π_jt = compute_π_jt(out, par_beq)
	
	(; inheritances, aggregate_inheritances) = compute_bequests_transition_simple(out, par_beq, π_jt)

	inheritances
	#=
	π_df = @chain out.sim_df begin
		@transform(:j, :t = :j + :born)
		@subset(0 ≤ :t ≤ out.T̃)
		@select(:j, :t, :born, :state, :ε, :π)		
		@subset(:π > 0)
	end
	
	@chain π_df begin
		leftjoin(_, DataFrame(inheritances), on = [:j, :t])
		@groupby(:t)
		@combine(:total_inheritance = sum(:inheritance, weights(:π)))
	end
	=#
end
  ╠═╡ =#

# ╔═╡ fb9f60d9-4f44-411f-9bfb-3b12830d4f78
function compute_π_jt((; sim_df, T̃), (; J))
	t_dim = Dim{:t}(0:T̃)
	j_dim = Dim{:j}(0:J)
	
	π_jt = @chain sim_df begin
		@groupby(:j, :t = :j + :born)
		@combine(:π = sum(:π))
		@subset(0 ≤ :t ≤ T̃)
		sparse(_.j .+ 1, _.t .+ 1, _.π)
		Matrix
		DimArray(_, (j_dim, t_dim))
	end
	
end

# ╔═╡ 6049b54f-6d12-4608-b5d4-9eea447a28f5
function compute_bequests_transition_simple((; GE₀, raw_aggregate_paths, sim_df, T̃), (; F, J), π_jt)

	t_dim = Dim{:t}(0:T̃)
	tₓ_dim = Dim{:t}(-1:T̃)

	aggregate_bequests = DimVector(
		[GE₀.raw_aggregates.bequests; raw_aggregate_paths.bequests], tₓ_dim, name = :bequests)

	# inheritances == bequests from last period
	aggregate_inheritances = DimVector( 
		parent(aggregate_bequests[t = At(-1:T̃-1)]), t_dim, name = :inheritances)

	inheritances = DimArray(@d(aggregate_inheritances .* F ./ π_jt), name = :inheritance)
	
	(; inheritances, aggregate_inheritances)
end

# ╔═╡ 07aed418-5f61-453e-bee0-d2e1c0a489c4
md"""
# Tests
"""

# ╔═╡ cf21558e-df00-4f36-8f11-0b2c407e0c16
md"""
## To do:

* [ ] use correct L path from first iteration
* [ ] what goes wrong with "good" guess?
* [x] re-use old solution 
* [~] read/write solutions from/to file (???)
* [x] use correct initial endowment (taking into account unexpected jump in prices)
* [~] compute value (for welfare comparisons)
"""

# ╔═╡ 6fd85349-9934-4fcf-a5d6-9e31175aa422
md"""
## Transition path
"""

# ╔═╡ f110e4f8-e3d7-4f38-af4d-39901ce3a13b
#=╠═╡
sprint_solution(out_18)
  ╠═╡ =#

# ╔═╡ c1a22777-0fbf-4366-a757-b38a02f79507
#=╠═╡
sprint_solution(out_18_noh)
  ╠═╡ =#

# ╔═╡ 9803fab5-2434-4b9c-91bc-869f95ed72f4


# ╔═╡ 4c02499c-b898-47c0-a447-cb9bd59da403
guesses_18_noh_05 = 
(; 
  stationary = (K_supply = 10.74605690956262, H_hh = 0.0, L_eff = 6.182668404249698),
  transition = DimStack(
      DimVector([
      10.746056881604218, 10.670272687242763, 10.70865004085076, 10.806471272463048, 10.928025312769348, 11.054352666764759, 11.17532916854795, 11.285414250463926, 11.382068833851642, 11.464473176771616, 11.533196636368352, 11.58951197881283, 11.635028666720778, 11.671376862992702, 11.70011740866541, 11.722699984511856, 11.740369965550235, 11.754129287469327, 11.764733748125547, 11.772542923953331, 11.778285314054678, 11.78249784053181, 11.785580083675772, 11.787831449247307, 11.789475445290373, 11.790677294266056, 11.791557947293533, 11.792205122307987, 11.792682232231682, 11.7930353030202, 11.793298169515658
    ], Dim{:t}(0:30), name = :K_supply),
      DimVector([
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ], Dim{:t}(0:30), name = :H_hh),
      DimVector([
      6.182668404245066, 6.335477429312836, 6.449805573998044, 6.534152753995002, 6.596055282755739, 6.6413090645145765, 6.674342456184469, 6.698403730395924, 6.715859015659292, 6.728598426325332, 6.737855217022292, 6.744522313544694, 6.749266833506258, 6.752610172941532, 6.75493333112776, 6.756496277885014, 6.757453148981151, 6.75789017297587, 6.757890172975861, 6.757890172975865, 6.757890172975868, 6.7578901729758725, 6.757890172975868, 6.757890172975866, 6.757890172975863, 6.75789017297587, 6.757890172975869, 6.757890172975872, 6.7578901729758645, 6.757890172975872, 6.757890172975872
    ], Dim{:t}(0:30), name = :L_eff),
  ),
  inheritances_θ = [0.8738136384902594, 1.75137230546213, 3.726223478042022],
inheritances_θt = [
  0.8738136350972242 0.5419093202201228 0.602643786643889 0.6484555270714716 0.6806731157498528 0.7021194793168098 0.715211331497029 0.7224927788753638 0.7264000929911315 0.7287910982869286 0.7313958883892043 0.7347365974132273 0.7385684835436841 0.742385959977411 0.7456842053038705 0.7481867031257601 0.7499813375124655 0.7513100875174172 0.7523611784312283 0.7528158069410953 0.7532314282391732 0.7536155537093521 0.7539579102157572 0.7542463862499665 0.7544754281053931 0.7546479276462208 0.7547729595943881 0.754862173320809 0.7549266917128699 0.7549752919846744 0.7550138060081534;
  1.7513722999984398 1.0907127929234142 1.2174371987414554 1.3135562393075164 1.381982359190052 1.4286880963659876 1.4586412897051466 1.4768507580691907 1.4879035121071598 1.4951224519164465 1.5017874534192999 1.5089867614366226 1.5164709383425607 1.5234869004732212 1.5293404055617341 1.5337431295953374 1.5369343360067278 1.5393357566048493 1.541224713938858 1.5418065334372357 1.5423253763552702 1.5427980069091891 1.5432145047825114 1.543560654789194 1.5438297016964042 1.5440253795950525 1.544159466466256 1.5442474009456677 1.5443043787993171 1.5443427440302795 1.544370898913835;
  3.7262234691777896 2.327622746099383 2.6056431325506546 2.817506183514487 2.9696524716703534 3.075482623426097 3.1458559264664485 3.191235544197693 3.220709405012037 3.2404937934693234 3.2571194080336463 3.2729977827370313 3.2881444936084714 3.30155496420773 3.3124032367067664 3.320520229037886 3.3265307319219852 3.3312234426567215 3.3349876937448433 3.335719535446571 3.3363395776525615 3.3368866397304426 3.337358097678035 3.3377408254521783 3.33802821117888 3.338225003712873 3.3383457435896466 3.3384099309796023 3.338437435584242 3.3384451728827536 3.338445028217382
 ]
)

# ╔═╡ edf8d0b2-86b9-4c43-9d9e-9fdc5dff20d8
guesses_18_bequests = 


(; 
  stationary = (K_supply = 8.830485846407845, H_hh = 1.4793952394473506, L_eff = 6.18266840425142),
  transition = DimStack(
      DimVector([
      8.83048584744193, 8.822483692594227, 8.831654265770288, 8.85178620389634, 8.873000854716635, 8.893324925402327, 8.911853007836037, 8.928201375796421, 8.94219995123569, 8.953878622223018, 8.963417974350117, 8.97107074378176, 8.977117522155405, 8.981837586122499, 8.985490970425449, 8.98830275352235, 8.99045648702224, 8.992095021725197, 8.993325986853442, 8.994215708317958, 8.994857451402074, 8.995321664160826, 8.995656495580096, 8.995897634844814, 8.996071357172667, 8.996196797684586, 8.996287740215791, 8.996354013806275, 8.996402636389266, 8.996438684370679, 8.996465945233629
    ], Dim{:t}(0:30), name = :K_supply),
      DimVector([
      1.4780432235191776, 1.4783209710632446, 1.479602636601362, 1.4813079484146336, 1.4830925456583772, 1.4847805579389286, 1.486296431733291, 1.487617011407501, 1.4887443911451357, 1.4896917472672488, 1.4904757420753179, 1.4911141165054747, 1.4916252456199852, 1.492027935819559, 1.4923407324986306, 1.4925808678247001, 1.4927632878089143, 1.4929002531656932, 1.4930015938305141, 1.493076240150873, 1.4931310780770861, 1.4931712303645215, 1.493200508958249, 1.4932217636437515, 1.4932371277067977, 1.4932481884381694, 1.4932561086679648, 1.4932617166260214, 1.4932655707768256, 1.493267999513191, 1.4932691100439368
    ], Dim{:t}(0:30), name = :H_hh),
      DimVector([
      6.182668404245034, 6.213225888798156, 6.234425748822419, 6.249297699821584, 6.259845500326865, 6.26737731314875, 6.272783631748733, 6.276673162577645, 6.279469700352515, 6.281489123634959, 6.282940656893956, 6.283975123413265, 6.2847023564676014, 6.285205460865165, 6.285544396220522, 6.285761150843893, 6.285884311204427, 6.285935361390635, 6.285935361390636, 6.285935361390635, 6.285935361390635, 6.285935361390634, 6.285935361390635, 6.285935361390632, 6.285935361390634, 6.285935361390636, 6.285935361390634, 6.285935361390638, 6.285935361390633, 6.285935361390633, 6.285935361390637
    ], Dim{:t}(0:30), name = :L_eff),
  ),
  inheritances_θ = [0.926533177879491, 1.7267687519050297, 3.107771442199534],
inheritances_θt = [
  0.9265331778795082 0.855518662667458 0.8712984463079874 0.8815789588679365 0.8880601750223602 0.892033218335623 0.8943477052447008 0.8956631282153669 0.8964810654180612 0.8971304830012765 0.8978419016789214 0.8986510902506645 0.8994898402783866 0.9002648375146061 0.9008987279554836 0.9013675896426102 0.9017038586872068 0.901953412865395 0.9021469541902178 0.90225709755205 0.9023547199206241 0.9024398778346056 0.9025113923538156 0.9025689571975936 0.9026135324014531 0.9026471008631276 0.9026720937654809 0.9026908276875851 0.9027051702646802 0.9027164474476823 0.9027255068618838;
  1.7267687519050419 1.5950255534575737 1.6255898547674619 1.6457791253192586 1.6587618576304273 1.6670060247866934 1.672098633003194 1.675239087408668 1.6773142892587072 1.6788905041495463 1.6803926457822689 1.6819146536514875 1.6833796699729782 1.6846664803516 1.6856884093797346 1.6864338818832363 1.6869629652464135 1.6873473226364568 1.6876314532643386 1.6877487546374286 1.6878505459196038 1.6879377626578476 1.688009287049394 1.6880649793485352 1.6881060713699445 1.688134971167462 1.688154607240004 1.688167789655769 1.688176799631715 1.6881832485226196 1.6881881226978954;
  3.107771442199544 2.870433608288773 2.9259227866515483 2.9633190713259148 2.987705594526034 3.003610137588961 3.013868108815866 3.0205531736669466 3.025143277899068 3.028543041372917 3.0314866028317593 3.0341811493140733 3.0365836389211545 3.038586892757421 3.0401365988835463 3.041266287267264 3.042082399231402 3.042685660626641 3.0431235260548584 3.043232107713885 3.0433179691409653 3.0433863219641557 3.0434380855949925 3.043474457444722 3.0434974059473685 3.0435096844246616 3.043514350892856 3.0435142569128675 3.043511695937528 3.0435082585061837 3.043504853529626
 ]

)

# ╔═╡ 5901b076-656d-4291-8a4f-dcd68efcf039
guesses_18_noh = (; 
  stationary = (K_supply = 7.81140214903537, H_hh = 9.89508367997814e-15, L_eff = 6.1826684042450495),
  transition = DimStack(
      DimVector([
      7.811397880732691, 7.78922587170626, 7.858569998419851, 7.9015764762100655, 7.929034047668358, 7.946830533561146, 7.958514940950549, 7.966224182187928, 7.971330847192476, 7.974764465795996, 7.977161812919831, 7.978868739007969, 7.9801026633935574, 7.981008071288078, 7.981683575858894, 7.982188572874373, 7.98255456891215, 7.982793858997483, 7.982913346410288, 7.982915406552861, 7.982924203577022, 7.982924388001349, 7.982922200684926, 7.982920207049919, 7.982919309603858, 7.98291953782927, 7.982920548797553, 7.9829219519945385, 7.982923409536978, 7.9829246905452145, 7.9829256811132066
    ], Dim{:t}(0:30), name = :K_supply),
      DimVector(0.0 .* [
      4.5156794860793157e-7, 4.527324768633717e-7, 4.5373437386621883e-7, 4.545217206299732e-7, 4.5511267905602856e-7, 4.555450498435379e-7, 4.5585692261641686e-7, 4.5608024246602257e-7, 4.562397217499737e-7, 4.5635362235449567e-7, 4.564349294200465e-7, 4.564928590055946e-7, 4.565339698671194e-7, 4.5656291384965805e-7, 4.565829502333203e-7, 4.5659635790098394e-7, 4.566047926860412e-7, 4.566096043941832e-7, 4.5661206614607553e-7, 4.5661332266464245e-7, 4.5661396615597164e-7, 4.5661430069930663e-7, 4.56614480600958e-7, 4.566145829023542e-7, 4.5661464534922414e-7, 4.566146861237364e-7, 4.566147139555013e-7, 4.5661473317451825e-7, 4.5661474611212685e-7, 4.5661475412565215e-7, 4.566147579161853e-7
    ], Dim{:t}(0:30), name = :H_hh),
      DimVector([
      6.182668404250349, 6.213230645462782, 6.234433805513665, 6.249308071520302, 6.2598575139242705, 6.267390499168172, 6.272797659330094, 6.276687795613729, 6.279484768705107, 6.281504506336361, 6.28295626554488, 6.283990893092, 6.284718239349369, 6.285221422061504, 6.285560410176443, 6.285777198540415, 6.285900378072408, 6.285951436205234, 6.285951436205238, 6.285951436205237, 6.285951436205231, 6.285951436205235, 6.285951436205238, 6.285951436205231, 6.28595143620523, 6.285951436205236, 6.2859514362052336, 6.285951436205233, 6.285951436205231, 6.285951436205233, 6.285951436205236
    ], Dim{:t}(0:30), name = :L_eff),
  )
)

# ╔═╡ 85465c5d-f4d5-4e60-8269-8293bb5d3e2a
guesses_18 = (; 
  stationary = (K_supply = 7.127752341937452, H_hh = 1.1212658132348279, L_eff = 6.182668404245626),
  inheritances_by_type = nothing,
  transition = DimStack(
      DimVector([
      7.127700125678019, 7.107433468422651, 7.174792033701194, 7.215516608289269, 7.241026838170942, 7.257270671743569, 7.267758147644637, 7.2745703273571936, 7.279016993292482, 7.281955759056319, 7.283983425533437, 7.285416527714739, 7.286450280997171, 7.287211969166251, 7.28778672895185, 7.288223618076419, 7.288546259955412, 7.288760963503013, 7.288869655680551, 7.288872015456595, 7.288882259888983, 7.288881882782235, 7.288878396092633, 7.288874943198147, 7.2888727282325965, 7.288871916772743, 7.288872208969871, 7.2888731783887435, 7.28887442518633, 7.288875648093575, 7.288876673044379
    ], Dim{:t}(0:30), name = :K_supply),
      DimVector([
      1.123077478265375, 1.126017739980421, 1.128795776177176, 1.131031018102039, 1.1327015241141056, 1.1339006001675, 1.1347405322490278, 1.1353203759427914, 1.1357181358809356, 1.1359917642134805, 1.136181283622833, 1.1363137844503552, 1.1364074620305373, 1.136474422083152, 1.1365224825762341, 1.1365565264003337, 1.136579585126063, 1.1365938188208966, 1.1366013372488986, 1.1366053953181985, 1.136607423822925, 1.1366083490652332, 1.136608727068646, 1.1366088689591787, 1.1366089328313547, 1.1366089872661402, 1.136609052724202, 1.1366091267379117, 1.136609198250094, 1.1366092546591702, 1.1366092835434203
    ], Dim{:t}(0:30), name = :H_hh),
      DimVector([
      6.182668771850203, 6.213231013571356, 6.234434173980124, 6.2493084402392185, 6.259857882821448, 6.267390868190918, 6.2727980284409055, 6.2766881647858535, 6.279485137919513, 6.28150487557965, 6.282956634807694, 6.283991262367838, 6.284718608633669, 6.2852217913511375, 6.285560779469282, 6.285777567835035, 6.285900747367915, 6.285951805501046, 6.2859518055010435, 6.285951805501045, 6.285951805501045, 6.2859518055010435, 6.285951805501044, 6.285951805501049, 6.285951805501048, 6.285951805501046, 6.285951805501045, 6.285951805501045, 6.285951805501046, 6.285951805501046, 6.285951805501045
    ], Dim{:t}(0:30), name = :L_eff),
  )
)

# ╔═╡ f1cd1f44-cd7b-4162-a51f-1c533cd79721
function sprint_dimstack(ds, prepend="")
	string = prepend
	for (key, val) ∈ pairs(ds)
		string *= prepend * "  DimVector([\n    " * prepend * join(repr.(val), ", ") * "\n$(prepend)  ], Dim{:t}(0:$(length(val)-1)), name = :$key),\n" * prepend
	end
	"DimStack(\n" * string * ")"
end

# ╔═╡ f83f05ae-4386-4b0c-97be-288b74fc6154
print_dimstack(ds) = sprint_dimstack(ds) |> Base.Text

# ╔═╡ 825e58e2-10b4-4a5f-99dc-573517fe32fe
md"""
# Bequests along the transition: Working proof of concept
"""

# ╔═╡ 4b58e7dc-2a33-4051-918c-a49d39cd6c29
md"""
* `transition_GE`
  * `π_θjt = compute_π_θjt((; demographics, GE₀, T̃), (; par.J), statespace)`
  * `inheritances_θt_new = compute_inheritance_θt(out_PE, statespace)`
* `transition_PE`
  * `inheritances_θtj = DimArray(@d(inheritances_θt .* par.F ./ π_θjt), name = :inheritance)`
"""

# ╔═╡ f44a4152-f111-4ca9-8fc5-7bf25bef7298
md"""
## End
"""

# ╔═╡ fea217b0-a5b9-4379-834b-393a2d0ad05e
function less_trivial_initial_distribution(statespace; init_state)
	(; dims, ε_chain, grid) = statespace
	
	π₀ = zeros(dims, name = :π₀)
	π₀[state = Near(init_state)] .= only(stationary_distributions(ε_chain))
	π₀
end

# ╔═╡ d4c3447e-8dae-4201-8d39-11bba68b56c5
md"""
* why is `a_next ≠ next_state` when there is no housing
* do I use the correct assets for computing GDP? should use `a_next` from previous period
* F matrix
"""

# ╔═╡ b0d59912-88b6-42bf-9328-aed96c0a3515
#=╠═╡
out_18.out.GE₀.raw_aggregates.a_next
  ╠═╡ =#

# ╔═╡ ee57a9c8-94ee-4c7b-8e53-20cab12fa64a
md"""
# To do

* growth ?!!??!!
"""

# ╔═╡ 5e4ea456-25d4-4138-8c0c-42d1a45828b7
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

# ╔═╡ 9e5d3e99-2fd5-4ef8-afe1-baeaf4339c84
# ╠═╡ disabled = true
#=╠═╡
out_18_noh = transition_test(18, guesses_trans=guesses_18_noh, tol_stat = 1e-12, tol_trans = 1e-5, ξ = 0.0, λ_trans = 0.2)
  ╠═╡ =#

# ╔═╡ f700462d-89af-4474-b063-92ddbfa4d079
# ╠═╡ disabled = true
#=╠═╡
out_18 = transition_test(18, guesses_trans=guesses_18, tol_stat = 1e-12, tol_trans = 1e-4) # 575 s
  ╠═╡ =#

# ╔═╡ 6355f453-5406-4899-a92e-9be360e9629e
EGMHousingRisk = ingredients("./egm-housing-risk.jl")

# ╔═╡ 441c1d74-5bbb-45f8-a150-4a66a5d51379
(; HousingModel, stationary_GE, stationary_PE, transition_PE, transition_GE) = EGMHousingRisk

# ╔═╡ a4342153-d03e-42f5-9448-b3ff632672ea
(; inheritances_stationary) = EGMHousingRisk

# ╔═╡ d0fbd753-27d5-4649-82b1-05a3e1cba2f8
(; get_states, prices_from_guesses_nt, constant_price_paths) = EGMHousingRisk

# ╔═╡ 6d7c2f32-6b22-4c72-87ee-5d71aa4fa60a
(; mortality, no_inheritances, trivial_initial_distribution, income_profile, no_income_risk, no_permanent_states, Statespace, get_par₀, visualize_stationary, permanent_states_AMMR, ε_chain_AMMR, dimstack_from_nt) = EGMHousingRisk

# ╔═╡ 6b5c2c47-66f8-4375-8850-efca4819e057
income_profile(90, 65) |> lines

# ╔═╡ 18070fd0-a0ce-4e28-ad5a-a39568460f43
income_profile(90, 89) |> lines

# ╔═╡ dcccf82a-e6f0-423d-b445-e09f7054caa4
(; weighted_neighbours) = EGMHousingRisk

# ╔═╡ cb517530-9ce3-4774-b6ed-ce2d636cb765
function sprint_matrix(mat)

	string = "[\n"
	nrows = size(mat, 1)
	for row ∈ 1:nrows
		string = string * "  " * join(repr.(mat[row,:]), " ")
		if row < nrows
			string = string * ";\n"
		else
			string = string * "\n"
		end
	end
	
	string * " ]"

	#=
	names = name(DD.dims(dv))

	
	string = "("

	for name ∈ names
		range = DD.dims(dv, name) |> parent |> parent |> repr
		string = string * "Dim{:$name}($range), "
	end

	string
	=#
	#"DimVector([\n    " * prepend * join(repr.(val), ", ") * "\n$(prepend)  ], Dim{:t}(0:$(length(val)-1)), name = :$key),\n" |> Base.Text

end

# ╔═╡ 11f86686-0165-4376-8847-42e68c30cc65
sprint_dimstack

# ╔═╡ 68b1fb68-61aa-4f1e-b4b9-c27eb9378059


# ╔═╡ eff5776e-7558-4c2b-8ce6-6b3d1107a511
md"""
## Stationary general equilibrium
"""

# ╔═╡ a1bf5a1b-36a8-4c65-bb6b-86847f25b569
#=╠═╡
visualize_stationary(
	solve_test(72; 
			   guesses = (; K_supply=9.51812, H_hh=2.56236e-11, L_eff=1.71987)
			  ).GE
)
  ╠═╡ =#

# ╔═╡ 959644f5-abc5-4bd5-8a43-3bf19776acb1
#=╠═╡
visualize_stationary(solve_test(18).GE)
  ╠═╡ =#

# ╔═╡ 5c88c45b-841a-4e7c-970b-6ae9a1ad38bd
#=╠═╡
visualize_stationary(out_no_risk.GE)
  ╠═╡ =#

# ╔═╡ eb6f43e9-7b05-4208-9165-5e3db9346151
#=╠═╡
out_no_risk = solve_test(18, risk = false, na = 500)
  ╠═╡ =#

# ╔═╡ 8a580804-4c9d-4eb7-b874-46f06282b4c0
# ╠═╡ disabled = true
#=╠═╡
function solve_test(J_P; amax = 100, na = 100, risk = true, ξ = 0.0, 
					tol_GE = 1e-4, λ = 0.1, details = 10,
					guesses = nothing)
	# equivalent to BaselineModel() in models_reduced.jl
	(; par, statespace, π_init, period) = get_cali_test(; amax, na, J_P, risk, ξ)

	Mo = HousingModel()

	if isnothing(guesses)
		K_guess = 3.5832334751343167
		guesses = (; K_supply = K_guess, H_hh = 8.55e-8, L_eff = period * 1.0)

		#(; prices) = prices_from_guesses_nt(Mo, guesses, par)
	
	
		#r = ((1 + prices.r)^(period) - 1) * 3
		#prices = (; r, prices.w, p = 1.2)
		#@info (; period, r, m = first(par.m))
	end
	#######################
	# Partial equilibrium #
	#######################
	
	GE = stationary_GE(Mo, par, statespace; 
						guesses,#= prices,=#
						π_init, tol = tol_GE, λ, details) # j_last XXX

	(; aggregates, prices) = GE
	(; K_supply) = aggregates.updated
	(; K_hh, ζ) = aggregates.aggregates
	(; state, c, ℓ_eff, a_next) = GE.raw_aggregates
	(; r) = prices  

	@info (; r = (1 + r)^(1/period)-1, K_hh, ℓ_eff = ℓ_eff / period)

	(; GE, statespace)
end
  ╠═╡ =#

# ╔═╡ a1a3aa17-a3c0-4503-910a-3caec607e9a6
#=╠═╡
let # test `iterate_cross_sectional_shock!`
	
	(; GE, statespace) = out_GE
	(; sim_df, prices, par, sols) = GE

	(; sol) = sols[1]
	
	(; π) = sol.sol_forward
	(; sol_backward) = sol
	(; m) = par

	π₋₁ = copy(π, name = :π₋₁)
	π₀  = zeros(DD.dims(π₋₁), name = :π₀)

	prices₀ = prices #(; prices.r, prices.w, p= * prices.p)
	
	iterate_cross_sectional_shock!(π₀, π₋₁, sol_backward, statespace, (; m), prices₀, par)

	@chain DimStack(sol_backward..., π₀, π₋₁) begin
		DataFrame
		@groupby(:j)
		@combine(
			:π₋₁ = mean(:π₋₁),
			:π₀  = mean(:π₀),
			:test = all(:π₋₁ ≈ :π₀)
		)
		@test all(_.test)
	end
end
  ╠═╡ =#

# ╔═╡ fe5d9a4a-6169-4090-828b-44d0367ea949


# ╔═╡ cf4f7625-a10c-4ccc-b259-b1e538e91246
# ╠═╡ disabled = true
#=╠═╡
out_GE = solve_test(36; guesses = (K_supply = 5.00, H_hh = 1.0, L_eff = 1.6), na = 200, ξ = 0.15, λ = 0.01)
  ╠═╡ =#

# ╔═╡ 408a192f-0844-4fa4-a58e-45498f9b3bbf
#=╠═╡
out_no_risk
  ╠═╡ =#

# ╔═╡ 0ff31629-855a-49a3-9b7d-cf4316c809da
#=╠═╡
let
	# XXXX
	(; GE) = out_no_risk
	
	(; sim_df, par) = GE

	@chain sim_df begin
		leftjoin(_, DataFrame(par.β), on = :j)
		@groupby(:j)
		@combine(
			:state = mean(:state, weights(:π)),
			:next_state = mean(:next_state, weights(:π)),
			:utility = mean(:utility, weights(:π)),
			:value = mean(:value, weights(:π)),
			:next_value = mean(:next_value, weights(:π)),
			:π = sum(:π),
			:m = only(unique(:m)),
			:β = only(unique(:β)),
			:β_over_β = only(unique(:β_over_β))
		)
		@transform(
			#:next_value = :next_value / (1-:m),
			:test = :utility + :β_over_β * (1 - :m) * :next_value - :value)
#		@transform(:test - :value)
		#@combine(sum(:β .* :π .* :utility))
		#@info _
	end

	#=@chain sim_df begin
		@subset(:j == 0)
		@combine(mean(:value, weights(:π)))
		@info _
	end=#
end

  ╠═╡ =#

# ╔═╡ f31b04f7-c119-4993-9e3d-f83d6bdce4be
#=╠═╡
let
	(; df, par) = out2
	(; value, utility, β_over_β, m, π) = df

	j_min = 0
	j_max = par.J + 1 - j_min
	
	l_surv = [1.0; df.surv[1:end-1]]
	β_over_βX = [1.0; β_over_β]
	
	@info @test value[j_max] ≈ utility[j_max]
	@info @test value[j_max-1] ≈ utility[j_max-1] + l_surv[j_max] * β_over_βX[j_max] * value[j_max]
	#@info @test value[17] ≈ utility[17] + l_surv[18] * β_over_βX[18] * utility[18]
	#@info @test value[16] ≈ utility[16] + l_surv[17] * β_over_βX[17] * value[17]
	#@info @test value[16] ≈ 
	#	 prod(l_surv[17:16]) * prod(β_over_βX[17:16]) * utility[16] + 
	#	 prod(l_surv[17:17]) * prod(β_over_βX[17:17]) * utility[17] + 
	#	 prod(l_surv[17:18]) * prod(β_over_βX[17:18]) * utility[18]

	i = 1
	value[i] ≈ sum(
		 prod(l_surv[(i+1):j]) * prod(β_over_βX[i+1:j]) * utility[j] for j ∈ i:18
	)

	@info @test value[1] ≈ sum(
		cumprod(l_surv)[j] * cumprod(β_over_βX)[j] * utility[j] for j ∈ 1:18
	)
	@info @test value[1] ≈ sum(
		π[j] * cumprod(β_over_βX)[j] * utility[j] for j ∈ 1:18
	)
	value[16]

	cumprod(β_over_β)
	i = 1
	j = 1
	prod(β_over_β[i:(j-1)])#, #cumprod(β_over_β)[j - 1]

	
	
end
  ╠═╡ =#

# ╔═╡ 34280c20-950c-4083-948b-98e8edce0724
#=╠═╡
out2 = let
	(; GE) = out_GE
	(; sim_df, par) = GE
	(; ξ, σ) = par

	df = @chain sim_df begin
		#DataFrame
		leftjoin(DataFrame(par.β), on = :j)
		leftjoin(rename(DataFrame(par.ξ), :value => :ξ), on = :j)
		
		@groupby(:j)
			@combine(
				:utility = mean(:utility, weights(:π)),
				:β = only(unique(:β)),
				:π = sum(:π),
				:next_value = mean(:next_value, weights(:π)),
				:β_over_β = only(unique(:β_over_β)),
				:m = only(unique(:m)),
				:value = mean(:value, weights(:π))
			)
			@transform(
				:surv = 1 - :m,
				:lsurv = @bycol [1.0; 1 .- :m[1:end-1]]
			)
			@transform(
				:test = :utility + :β_over_β * :surv * :next_value,
				:cum_surv = @bycol(cumprod(:surv)),
				:cum_lsurv = @bycol(cumprod(:lsurv))
			)
			@transform(
				:βX = @bycol cumprod([1.0; :β_over_β[1:end-1]])
			)
		
			end

	(; df, par)
end
  ╠═╡ =#

# ╔═╡ 6cf01d9d-f444-400a-9700-3e3e55cc1727
#=╠═╡
df = let
	out = out_GE

	(; sim_df, par) = out.GE

	#par.β, [0.995^j for j ∈ 0:(4*18)-1]

	#par.β,
	tmp = [0.995^j for j ∈ 0:(4*18)-1][4:4:end]

	β_over_β = tmp[10]/tmp[9] 
	@info @test β_over_β ≈ tmp[2]/tmp[1]

	(; ξ, σ) = par

	ξ, σ
	df = @chain sim_df begin
		#DataFrame
		leftjoin(DataFrame(par.β), on = :j)
		leftjoin(rename(DataFrame(par.ξ), :value => :ξ), on = :j)
		@transform(
			#:u = :co^(1-σ)/(1-σ),
			#:β2 = 0.995 ^ 3
		)
		
		@transform(
			#:βu = :β * :u,
			#:β2u = β_over_β * :u,
			#:β3u = :β_over_β * :u,
			
		)
		#@select(:value, :u, :u2=:utility, :j, :state, :next_state, :βu, :β2u, :β3u, :π, :β, :m)

		#@transform(:tmp = :j == 17 ? :u : β_over_β * :u)
		#@groupby(:j)
		#@combine(
		#	:v = mean(:value, weights(:π)),
		#	:u = mean(:u, weights(:π)),
		#	:π = sum(:π),
		#	:m = only(unique(:m))
		#)
		#@select(:u, :j, :π)

		
		@aside @chain _ begin
			@subset(:j == 0, :π > 0)
		
			@combine(
				:v = mean(:value, weights(:π))
			)
			@info _
		end
		
		
		#@aside @chain _ begin
			@groupby(:j)
			@combine(
				:utility = mean(:utility, weights(:π)),
				:β = only(unique(:β)),
				:π = sum(:π),
				:next_value = mean(:next_value, weights(:π)),
				:β_over_β = only(unique(:β_over_β)),
				:m = only(unique(:m)),
				:value = mean(:value, weights(:π))
			)
			@transform(
				:surv = 1 - :m,
				:lsurv = @bycol [1.0; 1 .- :m[1:end-1]]
			)
			@transform(
				:test = :utility + :β_over_β * :surv * :next_value,
				:cum_surv = @bycol(cumprod(:surv)),
				:cum_lsurv = @bycol(cumprod(:lsurv))
			)
			@transform(
				:βX = @bycol cumprod([1.0; :β_over_β[1:end-1]])
			)
			@aside @chain _ begin
				@combine(:value_by_hand = sum(:βX .* :utility .* :π))
				@info _
			end
			#@info _
		#end

		
		
		#@transform(:check = :value - :u)
		#@combine(maximum(:check))
		
		
		#=
		@subset(:j ≥ 16, :π > 0)
		#@aside @chain _ begin
			#@select(:j, :u, :value, :βu, :π, :\b)
			@groupby(:j)
			@combine(
				:value = mean(:value, weights(:π)),
				:u = mean(:u, weights(:π)),
				:u2 = mean(:utility, weights(:π)),
				:βu = mean(:βu, weights(:π)),
				:β_over_βu = β_over_β * mean(:u, weights(:π)),
				:β_over_βu2 = mean(:β_over_β .* :u, weights(:π)),
				:survβ_over_βu = β_over_β * mean((1 .- :m) .* :u, weights(:π)),
				:π = sum(:π),
				:β = only(unique(:β))
			)
		#	@info _
		#end
		=#

		
		#@subset(:j == 0, :π > 0)
		
		#@combine(
		#	:v = mean(:value, weights(:π))
		#)
		
	end
		

	#@chain df begin
	#	@transfrom(:u = )
	# =#
end
  ╠═╡ =#

# ╔═╡ f09814a6-82b8-457c-9d75-0c92d8612056
md"""
# Parameters
"""

# ╔═╡ e8496939-0407-4e92-a4cc-8d04bdb8a9c8
function get_statespace(; amax = 12.0, na = 100, 
						exponential = true, 
						period = 5, risk = true,
					   	n_ε = 3, n_std_ε = 2.0,
					   	n_θ = 3, n_std_θ = 2.0)
	if risk
		ε_chain = ε_chain_AMMR(1.0; n = n_ε, n_std = n_std_ε, period)
		permanent = permanent_states_AMMR(n = n_θ, n_std = n_std_ε)
	else
		ε_chain = no_income_risk()
		permanent = no_permanent_states()
	end
	
	statespace = Statespace(; 
							amin = 0.0, amax, na, 
							ε_chain, permanent, exponential)
end

# ╔═╡ 27523365-20a0-417f-a038-7e2b2f1bb832
get_statespace()

# ╔═╡ 7a39d667-e033-4915-94c1-41f12a02b944
md"""
## Save parameters for comparison with Marcelo
"""

# ╔═╡ 60356d6e-f242-4c22-bd29-47dc827a0d20
md"""
# Appendix
"""

# ╔═╡ 8daebfb1-30c8-46db-8a57-4d8969fd539b
TableOfContents()

# ╔═╡ 8ba3f40d-b249-4636-9df3-ccd315bc8fa6
fonts = (; regular = Makie.MathTeXEngine.texfont(:regular), bold = Makie.MathTeXEngine.texfont(:regular))

# ╔═╡ 32c861fb-4d5a-4886-aa04-2a7338525740
figure(size = (350, 250); figure_padding = 2, kwargs...) = (; size, fonts, figure_padding, kwargs...)

# ╔═╡ 0443aa96-730e-44d3-b372-cabe20bb5c1a
function visualize_transition((; sim_df, T̃), quantiles = [0.25, 0.5, 0.75])
	vars = [:z_next, :a_next, :ho, :income, :m]

	df = @chain sim_df begin
		@subset(:π > 0)
		@transform!(:t = :j + :born)
		@subset!(0 ≤ :t ≤ T̃)
		select!(vars..., :π, :t, :ε)
	end
#	@transform(:t )
#	df = 
	
	@chain df begin
		stack(vars, [:π, :t])
		@groupby(:variable, :t)
		@combine(
			:q = quantiles,
			:value = quantile(:value, weights(:π), quantiles))
		data(_) * mapping(:t, :value, 
						 # color = :q => nonnumeric,
						  row = :variable, col = :q
						 ) * visual(Lines)
		draw(; facet = (; linkyaxes = false), figure = figure((500, 500)))
	end
end

# ╔═╡ 56a4169a-7f74-4401-81a7-1355bfd4812e
criterion(a, b) = (a - b)/(1 + max(abs(a), abs(b)))

# ╔═╡ 44c4ddfe-3017-4e33-81e3-b067d88609fc
const DD = DimensionalData

# ╔═╡ 6773199d-d376-4b80-926a-8518feccd7dc
function adjust_period_discounting(βs, period)
	j_dim = DD.dims(βs, :j)
	j_min, j_max = extrema(j_dim)

	# shorten age grid
	sparse_js = period-1:period:j_max
	j_dim_new = Dim{:j}(0:length(sparse_js)-1)
	
	βs_itp = linear_interpolation(collect(j_dim), βs)
	
	βs_sparse₀ = βs_itp.(sparse_js)

	#if extend
	#	βs_sparse₀ = [βs_sparse₀; βs[j = At(j_max)]]
	#	sparse_js = [sparse_js; j_max]
	#end
	
	βs_sparse = DimVector(
		βs_sparse₀, j_dim_new,
		name = :β
	)

	(; βs_sparse, sparse_js)
end

# ╔═╡ 676fd408-d6fe-42fe-a8df-d79fa1007fc9
let
	βs = β_AMMR()
	
	βs = 0.95 .^ DD.dims(βs, :j)

	J = maximum(DD.dims(βs, :j))
	
	period = (J + 1) / 10
	
	j₀ = period
	(; βs_sparse, sparse_js) = adjust_period_discounting(βs, period)

	lines(βs)
	#js_sparse
	scatter!(sparse_js, parent(βs_sparse))

	#(βs_sparse)
	current_figure()
end

# ╔═╡ 18fe8d93-28c9-4da5-ad79-82150d2d13e8
function adjust_period_flow(flow, period; extend = false)
	j_dim = DD.dims(flow, :j)
	j_min, j_max = extrema(j_dim)

	# shorten age grid
	sparse_js = period-1:period:j_max
	j_dim_new = Dim{:j}(0:length(sparse_js)-1)
	
	# compute cumulative flow and interpolate
	cum_flow = cumsum(flow)
	cum_flow_itp = linear_interpolation(collect(j_dim), cum_flow)

	# compute cumulative flow at new j grid
	sparse_cum_flow = cum_flow_itp.(sparse_js)

	# take diff to get flow
	flow_sparse₀ = diff([0.0; sparse_cum_flow])
	
	flow_sparse = DimVector(
		flow_sparse₀, 
		j_dim_new,
		name = name(flow)
	)

	(; flow_sparse, sparse_js)

end

# ╔═╡ d8b0a2dd-3632-449b-a06d-509a555aaa31
function check_incomes(period; J_P = 60 ÷ period)
	
	J = period * J_P - 1
	JR = (J * 3) ÷ 4
	
	h = income_profile(J+1, JR)
	(; flow_sparse, sparse_js) = adjust_period_flow(h, period; extend = true)

	fig = Figure(size = (400, 200))
	ax = Axis(fig[1,1], title = "Compare cumulative flow")
	lines!(ax, 0:J, parent(cumsum(h)))
	scatter!(ax, sparse_js, parent(cumsum(flow_sparse)))

	ax2 = Axis(fig[1,2], title = "Compare flows")
	lines!(ax2, 0:J, parent(h))
	scatter!(ax2, sparse_js, parent(flow_sparse))
	
	fig
end

# ╔═╡ 55a26792-edd7-44d3-9ebc-1f3cc540101a
check_incomes(4, J_P = 45)

# ╔═╡ ada48a44-af10-4a91-b8f8-8aae066b5d53
function adjust_period_income_profile(h, period; kwargs...)
	(; flow_sparse, sparse_js) = adjust_period_flow(h, period; kwargs...)

	(; h_sparse = flow_sparse, sparse_js)
end

# ╔═╡ 40bcb115-eff9-4d1a-8da7-d4d51a9ea176
let
	period = 6
	J_P = 8
	J = period * J_P - 1
	JR = J

	period = 6
	j_grid₀  = 0:period:(J+1)
	j_grid = (period-1):period:J

	intervals = [j_grid₀[i]:j_grid[i] for i ∈ 1:length(j_grid₀)-1]

	@info last(intervals)
	
	h = income_profile(J+1, JR)

	@info DD.dims(h, :j)
	#lines(h)

	
	#intervals = [0:4, 5:9, 10:14, 15:19]

	only(unique(length.(intervals)))

	check = [sum(h[j = At(int)]) for int ∈ intervals]

	@info j_grid
	DimVector(check, Dim{:j}(maximum.(intervals)))
#=	
	scatter(j_grid₀, check)
	lines!(h, color = :orange)

	current_figure()
	#|> lines

	
	# =#


	
end

# ╔═╡ e6e752c3-411d-404f-aa85-77fdaba6ed59
let
	period = 1
	J_P = 8
	J = period * J_P - 1
	JR = J
	
	flow = income_profile(J+1, JR)
	
	j_dim = DD.dims(flow, :j)
	j_min, j_max = extrema(j_dim)
	
	sparse_js = period-1:period:J
	
	# compute cumulative flow and interpolate
	cum_flow = cumsum(flow)
	cum_flow_itp = linear_interpolation(collect(j_dim), cum_flow)

	sparse_cum_flow = [0.0; cum_flow_itp.(sparse_js)]

	# take diff to get flow
	flow_sparse₀ = diff(sparse_cum_flow)

	DimVector(flow_sparse₀, Dim{:j}(sparse_js))
end

# ╔═╡ 17367114-1f47-40eb-83f1-31fb88563e4d
function adjust_period_mortality(m, period; j₀=period)
	j_dim = DD.dims(m, :j)
	j_min, j_max = extrema(j_dim)

	# shorten age grid
	sparse_js = period-1:period:j_max
	j_dim_new = Dim{:j}(0:length(sparse_js)-1)

	# compute cumulative surv and interpolate
	cum_surv = cumprod(1 .- m)
	cum_surv_itp = linear_interpolation(collect(j_dim), cum_surv)

	# compute cumulative surv at new j grid
	sparse_cum_surv = [1.0; cum_surv_itp.(sparse_js)]

	#=
	extended = false
	
	if !(sparse_cum_surv[end] ≈ 0)
		sparse_cum_surv = [sparse_cum_surv; 0.0]
		sparse_js = [sparse_js; j_max]
		extended = true
	end
	=#
	
	# take diff to get flow
	surv_sparse = [sparse_cum_surv[i] / sparse_cum_surv[i-1] for i ∈ 2:length(sparse_cum_surv)]
	
	m_sparse = DimVector(1 .- surv_sparse, j_dim_new)

	(; m_sparse, sparse_js)
	
end

# ╔═╡ 5d127678-16a9-4845-a484-3c83278143d5
let
	period = 5
	m = mortality(:lifecycle)
	J = maximum(DD.dims(m, :j))
	JR = 45
	h = income_profile(J, JR) 
	
	(; m_sparse, sparse_js) = adjust_period_mortality(m, period)

#	cum_surv_test(death) = cumprod(1 .- death)

#	fig = Figure()
#	ax1 = Axis(fig[1,1])
#	lines!(ax1, cum_surv_test(m))
#	scatter!(ax1, js_sparse, parent(cum_surv_test(m_sparse)), color = :orange)

#	ax2 = Axis(fig[1,2])
	
#	fig
end

# ╔═╡ b6e810b1-cf37-4736-8f45-812290a6daf4
let
	m = mortality(:lifecycle)
	j_dim = DD.dims(m, :j)
	j_min, J = extrema(j_dim)
	
	period = (length(m) - 1)/7
	(; m_sparse, sparse_js) = 
		adjust_period_mortality(m, period, j₀ = period)

	
	cum_surv_test(death) = cumprod(1 .- death)

	fig = Figure(size = (500, 250))
	ax = Axis(fig[1,1], title = "Compare cumulative survival prop")
	lines!(ax, 0:J, parent(cum_surv_test(m)))
	scatter!(ax, sparse_js, parent(cum_surv_test(m_sparse)))

	ax2 = Axis(fig[1,2], title = "Compare mortality")
	lines!(ax2, 0:J, parent(m))
	scatter!(ax2, sparse_js, parent(m_sparse))

	fig
end

# ╔═╡ 6b3462fd-8d95-401f-be37-c394f4ae5fb3
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

# ╔═╡ abc2d0e1-17f2-49d3-8d1b-6bcdf3210f72
function sprint_inheritances_θt(inheritances_θt)
	t_range_repr = DD.dims(inheritances_θt, :t) |> parent |> parent |> repr
	t_dim_repr = "Dim{:t}($t_range_repr)"
	
	string = "inheritances_θt = "

	string = string * (inheritances_θt' |> Matrix |> sprint_matrix) 

	string = string * "\n"
#	string |> Base.Text

	
	
end

# ╔═╡ d6fab981-9367-46a2-bbbf-ff414069f144
function sprint_solution(out_trans)
	(; GE₀_etc, out) = out_trans
	(; GE₀, guessed_paths, inheritances_θt) = out

	stationary = "  stationary = " * repr(GE₀.guesses)

	transition = "  transition = " * sprint_dimstack(guessed_paths, "  ")

	inheritances_θ =  "  inheritances_θ = " * repr(parent(GE₀_etc.inheritances_etc.inheritances_θ))
	
	inheritances_θt = inheritances_θt |> sprint_inheritances_θt

	
	"(; \n" * stationary * ",\n" * transition * ",\n" * inheritances_θ * ",\n" * inheritances_θt *"\n)" |> Base.Text
end

# ╔═╡ 2e6a5fc2-ccc9-4afd-95ea-349856d86e86
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

# ╔═╡ e123bf7f-7113-44e0-a955-d20520f84872
function get_cali_test(; 
					   amax = 12.0, na = 500, exponential = true,
					   J_P = 71, risk = true, ξ = 0.0, bequests = false,
					   n_ε = 3, n_std_ε = 2.0,
					   n_θ = 3, n_std_θ = 2.0
			 )

	J = 71
	period = (J + 1) / J_P
	#J_P = J_approx ÷ period
	J = period * J_P - 1
	#JR = (J * 3) ÷ 4
	
	m = mortality(:marcelo, age_min = 20, age_max = 91)

	j_dim = DD.dims(m, :j)
	J = maximum(j_dim)
	#m[j = At(0:J-1)] .= 0.01

	@info J
	h = income_profile(J+1, 41)
	F = DimVector(F_marcelo[(1:72) .+ 2], j_dim, name = :F)
	
	βs = 0.995 .^ j_dim
	δ = 0.1
	δ = 1 - (1-δ) ^ period

	(; m_sparse) = adjust_period_mortality(m, period)
	(; h_sparse) = adjust_period_income_profile(h, period)
	(; βs_sparse) = adjust_period_discounting(βs, period)
	h_sparse2 = adjust_period_flow(h, period).flow_sparse
	F_sparse = adjust_period_flow(F, period).flow_sparse

	if !bequests
		F_sparse .= 0.0
	else
		F_sparse ./= sum(F_sparse)
	end
	
	@assert h_sparse2 ≈ h_sparse

	statespace = get_statespace(; amax, na, risk, exponential, period,  
								n_ε, n_std_ε, n_θ, n_std_θ)
	
	par = get_par₀(; h=h_sparse, m=m_sparse, ξ, δ, β = βs_sparse, 
				   bonds2GDP = 1.0, NFA2GDP = 0.0, τ = 0.0,
				   F = F_sparse, #ν₀ = 0.0,
				   annuities = false,
				   α = 0.35, θ = 0.0, Z̲ = 0.0, a̲ = 0.0)

	π_init = less_trivial_initial_distribution(statespace, init_state = 0.0)

	inheritances = no_inheritances(par, statespace)
	(; par, statespace, π_init, inheritances, period)
end

# ╔═╡ ea5552f4-e5d1-4a56-a655-5af6eb9cbf25
function transition_test(J_P; amax = 100, na = 100, risk = true, ξ = 0.15, guesses_trans=nothing, tol_stat = 1e-4, tol_trans = 1e-4, 
						 λ_trans = 0.02, λ_inherit = 1.0,
						 scale_m = 0.9, bequests = false, skip_transition = false, details = 20, 
						 maxiter_GE = 100, maxiter_trans = 400, PE = true
						)
	# equivalent to BaselineModel() in models_reduced.jl
	(; par, statespace, π_init, period) = get_cali_test(; amax, na, J_P, risk, ξ, bequests)

	Mo = HousingModel()

	if !isnothing(guesses_trans)
		@info "updated guesses_trans"
		#guesses_trans = guesses_trans
	end
	
	if !isnothing(guesses_trans)
		guesses = guesses_trans.stationary
		inheritances_θ_guess = guesses_trans.inheritances_θ
	else
		K_guess = 9.54318
		guesses = (; K_supply = K_guess, H_hh = 8.55e-8, L_eff = 1.7289)

		guesses = (K_supply=30.30629, H_hh=1.0, L_eff=6.18267)
		inheritances_θ_guess = nothing
	
	end

	prices = (r = 0.025, p = 1.2, w = 1.4)

	if PE
		GE₀_etc = stationary_PE(Mo, par, statespace, guesses, prices;
						#prices, #π_init
						#
						#guesses, inheritances_by_type_guess,#prices, 
						#(; r = -0.00015, w = 0.656348, p = 0.873666);
						π_init)#, maxiter = maxiter_GE, λ_inherit)#, tol = tol_stat, details = 30) # j_last XXX
	else
	
		GE₀_etc = stationary_GE(Mo, par, statespace; guesses,
						#prices, #π_init
						#
						#guesses, ,#prices, 
						#(; r = -0.00015, w = 0.656348, p = 0.873666);
							inheritances_θ_guess,	
							π_init, maxiter = maxiter_GE, λ_inherit, details=20, tol = tol_stat) # j_last XXX	
	end
	
	GE₀ = GE₀_etc.out_PE
	
	#return (; par, out = (; GE₀), period, statespace)
	
	(; aggregates, prices) = GE₀
	(; K_supply) = aggregates.updated
	(; K_hh, ζ) = aggregates.aggregates
	(; state, c, ℓ_eff, a_next) = GE₀.raw_aggregates
	(; r) = prices  

	if skip_transition
#		return (; par, out = (; GE₀), period, statespace)
		return (; par, out = (; GE₀), GE₀_etc, period, statespace)
	end
	
	
	T̃ = 30

	
	#####################################
	## TEST 2: REDUCE MORTALITY BY 10% ##
	
	

	#guessed_path = guess_auclert_trans
	
	demographics = let
		m₀ = par.m
		m₁ = scale_m * par.m
		m₁[end] = 1.0
		
		j_dim = DD.dims(m₀, :j)
		J = maximum(j_dim)
	
		borns = -J:1:T̃
		born_dim = Dim{:born}(borns)
		ms = DimArray(cat([m₁ for born ∈ born_dim]..., dims = born_dim), name = :m)

		
		demo = DimStack(ms, )
	end
	###########################

	t_dim = Dim{:t}(0:T̃)
	perm_dim = only(statespace.perm_dim)

	if isnothing(guesses_trans) || !hasproperty(guesses_trans, :inheritances_θt)
		inheritances_θt_guess = 
			(stack ∘ fill)(GE₀_etc.inheritances_etc.inheritances_θ, length(t_dim))'
	else
		@info "used inheritances_θt_guess"
		inheritances_θt_guess = DimArray(
			guesses_trans.inheritances_θt',
			(Dim{:t}(0:T̃), only(statespace.perm_dim))
		)
	end
	
	if isnothing(guesses_trans) || !hasproperty(guesses_trans, :transition)
		guess = GE₀.aggregates.updated
		guessed_path₀ = dimstack_from_nt(guess, Dim{:t}(0:T̃))
	

		guessed_paths = guessed_path₀
	else
		guessed_paths = guesses_trans.transition
	end

	out = transition_GE(Mo, T̃, par, statespace, demographics, GE₀, guessed_paths;
						normalize_population = false, inheritances_θt_guess,
						details, λ = λ_trans, maxiter = maxiter_trans, tol = tol_trans, λ_inh=λ_inherit
						)

	(; par, out=out.out_PE, GE₀_etc, out_full = out, period, statespace, demographics)
end

# ╔═╡ 8cb9b650-12fb-4691-925e-51e806cb12fd
out_18_noh_05 = transition_test(18; guesses_trans=guesses_18_noh_05,
								amax = 200, #na = 100,
								tol_stat = 1e-10, 
								tol_trans = 1e-4, ξ = 0.0, λ_trans = 0.2, λ_inherit = 1.0, scale_m = 0.5, bequests = true, skip_transition = false, PE = false,
								maxiter_GE = 100, 
								maxiter_trans = 10, details = 1
							   )

# ╔═╡ b15d0354-b4b8-4378-8a3c-350a81058b57
sprint_solution(out_18_noh_05)

# ╔═╡ 7a208658-8fb0-423c-bb95-95c0101fd98e
out_18_bequests = transition_test(18, guesses_trans = guesses_18_bequests,
								  maxiter_GE = 200,
								  maxiter_trans = 100,
								  PE = false,
								  tol_stat = 1e-9, tol_trans = 1e-4,
								  bequests = true, skip_transition = false, 
								  details = 1) # 575 s

# ╔═╡ e942a0e0-f194-4ccf-bade-254a7bdc0c02
sprint_solution(out_18_bequests)

# ╔═╡ 3dc546d1-10f4-4036-b8e2-93963b5e78e0
#out_X = out_18_noh_05
out_X = out_18_bequests

# ╔═╡ 7c79a861-356b-4694-8974-4ee748ffde10
sprint_solution(out_X)

# ╔═╡ d0737e80-f7ad-42d8-946d-fa584ac3e166
out_X.GE₀_etc.inheritances_etc.inheritances_θ |> parent |> repr

# ╔═╡ 624256ff-1209-44fc-aa85-3749b61e2186
let
	(; out) = out_X
	@chain out.sim_df begin
		@groupby(:t = :j + :born)
		@combine(
			:bequests    = sum(:bequests, weights(:π)),
			:inheritance = sum(:inheritance, weights(:π)),
			:population  = sum(:π)
		)
		@subset(0 ≤ :t ≤ out.T̃)
	end

end

# ╔═╡ d9cc41ce-4e3a-4e59-a1e7-180e06b15d55
let
	(; par) = out_X
	
	π = EGMHousingRisk.pmf(par.m)
	inh_θ = out_X.GE₀_etc.inheritances_etc.inheritances_θ

	@d inh_θ .* par.F ./ π
	
end

# ╔═╡ ef0dccc2-4dbf-4c2f-baf1-3ba015ac2bd8
let
	
	(; out, par, statespace) = out_X

	π_j = EGMHousingRisk.pmf(par.m)
	π_θ = statespace.π_permanent

	π_jθ = @d π_j .* π_θ
	
	#π = EGMHousingRisk.compute_π_θjt(out, par, statespace)

	#DataFrame(π)
	#π = EGMHousingRisk.pmf(par.m)
	
	inh_θ = out_X.out_full.inheritances_tθ
	(@d inh_θ .* par.F ./ π_jθ)[t = At(0)]
	
	 
end

# ╔═╡ 51ab33a4-bb9f-457f-8341-4859cb5c743b
out_X.GE₀_etc.out_PE.inheritances_θj

# ╔═╡ c8912513-9718-44b7-bc92-cf6636de3ecd
out_X.out.inheritances_θt

# ╔═╡ bf12a47d-436a-41fd-a145-656c58378852
out_X.out.raw_aggregate_paths.bequests |> lines

# ╔═╡ fb387b29-d335-4c84-bf2a-75f88a534f2b
out_X.out.GE₀.raw_aggregates.bequests

# ╔═╡ 4d6890fd-d7f6-432b-a744-2ea11da65ff1
out_X.out.GE₀.raw_aggregates.inheritance

# ╔═╡ ed4e6c7d-4741-4630-91e5-acac51f29e06
out_X.out.raw_aggregate_paths.inheritance

# ╔═╡ 99940e90-3580-446c-9637-7b39c1688935
let
	vars = [:inheritances, :F, :test]
	tmp = @chain out_X.out.GE₀.sim_df begin
		@transform(:π = @bycol :π ./ sum(:π))
		@groupby(:j, :θ = round(:permanent.θ, digits = 2))
		@combine(
			:bequests = mean(:bequests, weights(:π)),
			:inheritances = mean(:inheritance, weights(:π)),
			:π = sum(:π)
		)
		leftjoin(DataFrame(out_X.par.F), on = :j)
		@transform(:test =  :inheritances * :π / :F)
		data(_) * mapping(:j, vars, color = :θ => nonnumeric, row = AoG.dims(1) => renamer(vars)) * visual(Lines)
	   draw
	end
end

# ╔═╡ 27d2effa-2d30-4217-90f9-07f06eeefae4
let
	@chain out_X.out.GE₀.sim_df begin
		@transform(:π = @bycol :π / sum(:π))
		@groupby(:θ = :permanent.θ)
		@combine(
			:state = mean(:state, weights(:π)),
			:income = mean(:income, weights(:π)),
			:π = sum(:π)
		)

	end

end

# ╔═╡ 86c476ee-8da7-4e64-8259-784f4a6095f0
let
	@chain out_X.out.GE₀.sim_df begin
		@transform(:π = @bycol :π / sum(:π))
		@groupby(:θ = round(:permanent.θ, digits = 3), :j)
		@combine(
			:state = mean(:state, weights(:π)),
			:income = mean(:income, weights(:π)),
			:bequests = mean(:bequests, weights(:π)),
			:inheritance = mean(:inheritance, weights(:π)),
			:π = sum(:π)
		)
		stack([:bequests, :inheritance])
		data(_) * mapping(
			:j => L"age $j$", 
			:value => "",
			row = :variable, color = :θ => nonnumeric => L"type $\theta$") * visual(Lines)
		draw(; 
			 figure = figure(; size = (400, 300)),
			 facet = (; linkyaxes = false),
			)
	end

end

# ╔═╡ 2b1d18f2-b82c-4058-a6cb-65ff468ed1c4
let
	(; statespace, out) = out_X
	(; GE₀) = out

	π_θ = DimVector(out_X.statespace.π_permanent, name = :π)
	
	(; bequests_θ, inheritances_θ) = 
		inheritances_stationary(GE₀, statespace)

	df = (DataFrame ∘ DimStack)(bequests_θ, inheritances_θ, π_θ)

	@info @combine(df,
			 :bequests = sum(:bequests, weights(:π)),
			 :inheritances = sum(:inheritances, weights(:π))
			)

	df

	#inheritances_θj' |> Matrix

	@chain out_X.out.GE₀.sim_df begin
		@transform(:π = @bycol :π / sum(:π))
		@select(:j, :θ = :permanent.θ, :ε, :state, :income, :π, :next_state,
				:bequests, :inheritance_ref = :inheritance)
	#	leftjoin(DataFrame(inheritances_θj), on = [:j, :θ])
		#@groupby(:θ)
		@combine(
			:state = mean(:state, weights(:π)),
			:income = mean(:income, weights(:π)),
			:bequests = mean(:bequests, weights(:π)),
		#	:inheritance = mean(:inheritances, weights(:π)),
			:inheritance_ref = mean(:inheritance_ref, weights(:π)),
			
			:π = sum(:π)
		)

	end
	
end

# ╔═╡ 8daab1ff-f661-4772-89f6-569d9d0246f3
let
	
	tmp = @chain out_X.out.GE₀.sim_df begin
		#@transform(:π = @bycol :π ./ sum(:π))
		@groupby(:permanent)
		@combine(
			:bequests = mean(:bequests, weights(:π)), # / sum(:π),
			:inheritances = mean(:inheritance, weights(:π)), # / sum(:π),
			:π = sum(:π)
		)
		#=
		@aside @chain _ begin
			@combine(
				:bequests = sum(:bequests),
				:inheritances = sum(:inheritances),
				:π = sum(:π)
			)
			@info _
		end
		=#
	end	
end

# ╔═╡ 6bf82f04-b7f9-4ac4-a48a-948f7cdbad16
@chain out_X.out.raw_aggregate_paths begin
	DataFrame
	@select(:t, :bequests, :inheritance, :population)
end


# ╔═╡ e0ac337f-2653-4973-a2e6-6559ce58c428
out_X

# ╔═╡ 2e275057-1155-4edf-bb76-cf870092cb9a
out_X.GE₀_etc

# ╔═╡ 6de812cb-8d2b-45fc-bf99-685083b07a9a
@chain out_X.demographics begin
	DataFrame
	@transform(:t = :j + :born)
	@subset(:born == 0)
	#@groupby(:j, :t, :born)
	#data(_) * mapping(:j, :m, color = :born => nonnumeric) * visual(Lines)
	#draw
end

# ╔═╡ f3d805da-da19-4d9e-b4e6-69da4f19a833
@chain out_X.out_full.out_PE.sim_df begin
	@transform(:t = :j + :born)
	@groupby(:j, :t, :born)
	@combine(:population = sum(:π))
	@subset(:t > 0)
	@subset(:born ∈ [-10, 0, 10])
	#@subset(:)
	data(_) * mapping(:j, :population, color = :born => nonnumeric) * visual(Lines)
	draw
end

# ╔═╡ a08be402-718b-4e47-a68d-9d86e52a6b0a
let
	(; statespace, par, out) = out_X

	@chain out.sim_df begin
		@groupby(:t = :j + :born, :θ = :permanent.θ)
		@combine(
			:bequests_tot = sum(:bequests, weights(:π)),
			:bequests_avg = mean(:bequests, weights(:π)),
			:π = sum(:π)
		)
		@transform(:bequests_tot_test = :bequests_avg * :π)
		@subset(0 ≤ :t ≤ out.T̃)
		sort([:t, :θ])
		@groupby(:t)
		@combine(
			:bequests = sum(:bequests_avg, weights(:π)),
			:bequests2 = sum(:bequests_tot),
			:π = sum(:π)
		)
		@info _
	end
	bequests_θt = EGMHousingRisk.get_bequests_θt(out, statespace)
	bequests_θt[t = At(-1:1)]

end

# ╔═╡ 053c4e15-4fc7-4ef4-aa6a-26af5fa83fda
let
	(; statespace, par, out) = out_X

	@chain out.sim_df begin
		@transform(:t = :j + :born)
		@subset(0 ≤ :t ≤ out.T̃)
		@groupby(:t, :θ=:permanent.θ)
		@combine(
			:bequests = mean(:bequests, weights(:π)),
			:π = sum(:π)
		)
		sort([:t, :θ])
	end
end

# ╔═╡ 82865d9e-9329-41e6-a53a-a0da2daa9532
let
	(; statespace, par, out, out_full) = out_X

	@chain out.sim_df begin
		@groupby(:t = :j + :born)
		@combine(
			:bequests = sum(:bequests, weights(:π)),
			:inheritance = sum(:inheritance, weights(:π)),
			:π = sum(:π)
		)
		@subset(0 ≤ :t ≤ out.T̃)
	end
end

# ╔═╡ 9d734553-a5c8-4804-bf64-14dd1d04269d
let
	(; statespace, par, out, out_full) = out_X

	@chain out.GE₀.sim_df begin
		@transform(:π = @bycol :π / sum(:π))
		#@groupby(:t = :j + :born)
		@combine(
			:bequests = sum(:bequests, weights(:π)),
			:inheritance = sum(:inheritance, weights(:π)),
			:π = sum(:π)
		)
		#@subset(0 ≤ :t ≤ out.T̃)
	end
end

# ╔═╡ dc69dc02-37c0-40cb-bed8-9251d0285abf
visualize_transition(out_X.out)

# ╔═╡ ad93f5ad-b0ad-497c-9314-aaee78e25940
@chain out_X.out.price_paths begin
	DataFrame
	@subset(-1 ≤ :t ≤ 3)
	stack(Not(:t))
	@transform(:value = round(:value, digits = 3))
	unstack(:variable, :value)
	#data(_) * mapping(:t, :value, layout = :variable) * visual(ScatterLines)
	#draw(; facet = (; linkyaxes = false ))
end

# ╔═╡ 9fd5ad6b-aec7-4a5d-986b-0abec38da5f5
out_X.out.price_paths.r |> parent

# ╔═╡ cc7b46e7-3475-44a5-9684-471d11e24ad6
let
	(; price_paths, raw_aggregate_paths) = out_X.out

	df = innerjoin(DataFrame(price_paths), DataFrame(raw_aggregate_paths), on = :t)

	@chain df begin
		stack([:r, :p, #= :r, :w, :state, :co, :ho, :income, :m, :constrained,=# :population, :z, :a_next, :z_next], :t)
		@groupby(:variable)
		@transform(:value = @bycol :value ./ first(:value))
		data(_) * mapping(:t, :value, layout = :variable) * visual(Lines)
		draw(facet = (; linkyaxes = false))
	end
end

# ╔═╡ 2f587a99-6ff5-40cd-9415-4383f39571d4
let
	@chain out_X.out.GE₀.sim_df begin
		stack([:c, :ho, :z, :a_next, :z_next, :income], [:j, :π])
		@groupby(:variable, :j)
		@combine(:value = mean(:value, weights(:π)))
	#	unstack(:variable, :value)
		data(_) * mapping(:j, :value, layout = :variable) * visual(Lines)
		draw(facet = (; linkyaxes = false))
	end
	
end

# ╔═╡ 21a6230c-9284-4246-83d8-4eefd8616b90
out_X.out.GE₀

# ╔═╡ ebfdc2d6-431c-4ced-b8d2-0f534829d16e
out_X.out.sim_df

# ╔═╡ c2aa615c-4231-4173-a6db-81a5be35cabd
let
	df = DimStack(
		out_X.out.raw_aggregate_paths.population,
		out_X.out.raw_aggregate_paths.c,		
		out_X.out.raw_aggregate_paths.a_next,
		out_X.out.raw_aggregate_paths.next_state,
		out_X.out.raw_aggregate_paths.state,
		out_X.out.aggregate_paths.updated.L_eff,
		out_X.out.aggregate_paths.updated.K_supply,
	
		
	) |> DataFrame

	minus1 = DataFrame([(; t = -1, population = 1.0,
	out_X.out.GE₀.raw_aggregates.c,					 
	out_X.out.GE₀.raw_aggregates.a_next,
	out_X.out.GE₀.raw_aggregates.next_state,
	out_X.out.GE₀.raw_aggregates.state,
	out_X.out.GE₀.aggregates.updated.L_eff,
	out_X.out.GE₀.aggregates.updated.K_supply)])

	full = innerjoin([minus1; df], DataFrame(out_X.out.price_paths), on = :t)

	@chain full begin
		select(Not(:p, :next_state, :state))
		@subset(-1 ≤ :t ≤ 2)
	end
	
end

# ╔═╡ 211246a2-7bd6-47c3-882c-7804e31a012f
out_X.out.raw_aggregate_paths.state |> scatterlines

# ╔═╡ 3ea13784-73f0-4f70-bd72-edc4e81efa59
out_X.out.price_paths.p

# ╔═╡ 3bd0d3fe-619d-4f10-938c-c60d5cfb49e5
let
	layer1 = @chain out_X.out.sim_df begin
		@transform(:t = :j + :born)
		@subset(:t ≥ 0)
		@groupby(:j, :born, :t)
		@combine(:m = only(unique(:m)))
		data(_) * mapping(group = :born)
	end

	layer2 = @chain out_X.out.GE₀.sim_df begin
		@groupby(:j)
		@combine(:m = only(unique(:m)))
		data(_)
	end

	(layer1 + layer2) * mapping(:j, :m) * visual(Lines) |> draw
end

# ╔═╡ 6667de81-671d-454c-8d73-f0309f646b60
@chain out_X.out.GE₀.sim_df begin
	@groupby(:j)
	@combine(:m = only(unique(:m)))
	data(_) * mapping(:j, :m) * visual(Lines)
	draw
end

# ╔═╡ 3b124dcd-6e93-43d6-bb00-e144926a0a58
out_X.GE₀_etc

# ╔═╡ 0b68d40c-e7b0-4f4e-bf0d-d009a14707b3
@chain out_X.out.GE₀.sim_df begin
	@groupby(:permanent, :j)
	@combine(
		:inheritances = mean(:inheritance, weights(:π)),
		:bequests = mean(:bequests, weights(:π)),
		:state = mean(:state, weights(:π))
	)
	data(_) * mapping(:j, :state, color = :permanent) * visual(Lines)
	draw
end

# ╔═╡ c21b594c-a8de-4150-abf2-4d937052fdb7
let
	(; sim_df, T̃) = out_X.out
	@chain sim_df begin
		@transform(:t = :j + :born)
		@subset(0 ≤ :t ≤ T̃)
		@groupby(:t = :j + :born)
		@combine(
			:bequests = sum(:bequests, weights(:π)),
			:inheritance = sum(:inheritance, weights(:π)),
			:π = sum(:π)
		)
		@transform(
			:bequests * :π
		)
	end

end

# ╔═╡ 3add657e-ce09-4e4e-ac94-ac522e927ef8
let
	(; out, GE₀_etc, statespace) = out_X
	(; GE₀) = out

	@chain GE₀.sim_df begin
		@transform(:π = @bycol(:π ./ sum(:π)))
		@select(:state, :ε, :j, :permanent, :next_state, :bequests, :inheritance, :π)
		#@groupby(:permanent)
		@combine(
			:bequests = mean(:bequests, weights(:π)),
			:π_θ = sum(:π),
			:inheritances = mean(:inheritance, weights(:π))
		)
		#@transform(:bequests / :π_θ)
	end
end

# ╔═╡ 14b7d938-be7c-445b-b0a7-6088eb1a3fec
let
	(; out, GE₀_etc, statespace, par) = out_X
	π_jt = EGMHousingRisk.get_π_jt(out, par, statespace)

	@chain π_jt begin
		DataFrame
		@groupby(:t)
		@combine(:π_jt = @bycol :π_jt ./ sum(:π_jt))
	end
end

# ╔═╡ 702230c6-d538-4dd1-822b-57f2bb4c6818
let
	(; out, par, GE₀_etc, statespace) = out_X
	(; GE₀) = out

	bequests_θ = get_bequests_θ(GE₀.sim_df, statespace)

	inheritances_θ = get_inheritances_θ(bequests_θ, statespace)

	π_j = EGMHousingRisk.pmf(par.m)
	
	inheritances_θj = 
		DimArray(@d(inheritances_θ .* par.F ./ π_j), name = :inheritances)
	
	
	df = (DataFrame ∘ DimStack)(
		bequests_θ, inheritances_θ, DimVector(statespace.π_permanent, name = :π_perm)
	)

	@chain df begin
		@combine(
			:x = mean(:bequests, weights(:π_perm)),
			:y = mean(:inheritances, weights(:π_perm))
		)
	end
	#GE₀_etc.inheritances_etc.inheritances_θ
	# =#
end

# ╔═╡ 35976ef0-ab93-4192-abae-b36fad04ca99
let
	@chain out_X.out_full.inh_tθ_etc_new.inheritances_θt begin
		DataFrame
		data(_) * mapping(:t, :inheritances, color = :θ => nonnumeric) * visual(ScatterLines)
		draw
	end
end

# ╔═╡ d584e00d-3da1-44cf-b62a-1c4bc3144c90
let
	out_X.out.statespace.perm_dim
	out_X.out.inheritances_θt
end

# ╔═╡ d41e5a93-b200-4f35-a49b-0cddaac9bce7
let
	(; par) = get_cali_test(; J_P = 18, risk = true, bequests = true)

	df = DataFrame(DimStack(
		par.β, 
		DimVector(par.m, name = :m),
		par.h,
		par.F,
	))

	CSV.write("longer-period-marcelo.csv", df)	
end				   

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
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
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
AlgebraOfGraphics = "~0.11.7"
CSV = "~0.10.15"
CairoMakie = "~0.15.6"
Chain = "~1.0.0"
DataFrameMacros = "~0.4.1"
DataFrames = "~1.8.0"
DimensionalData = "~0.29.23"
Interpolations = "~0.16.2"
PlutoLinks = "~0.1.6"
PlutoTest = "~0.2.2"
PlutoUI = "~0.7.71"
QuantEcon = "~0.16.8"
Roots = "~2.2.10"
StatsBase = "~0.34.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "6f4b23ecd3b090e8dcc170005ad9ab2855ae927c"

[[deps.ADTypes]]
git-tree-sha1 = "27cecae79e5cc9935255f90c53bb831cc3c870d7"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.18.0"

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
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

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

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

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
git-tree-sha1 = "980f01d6d3283b3dbdfd7ed89405f96b7256ad57"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "2.0.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "e771a63cc8b539eca78c85b0cabd9233d6c8f06f"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.1"

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
git-tree-sha1 = "c967271c27a95160e30432e011b58f42cd7501b5"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6c72198e6a101cccdd4c9731d3985e904ba26037"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.1"

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
git-tree-sha1 = "cee1700673af54db57bd1c7fb834ad4ff31309a0"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.8"

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
git-tree-sha1 = "3bc002af51045ca3b47d2e1787d6ce02e68b943a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.122"

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
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

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
git-tree-sha1 = "173e4d8f14230a7523ae11b9a3fa9edb3e0efd78"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.14.0"
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
git-tree-sha1 = "dc41303865a16274ecb8450c220021ce1e0cf05f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.2.1"
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
git-tree-sha1 = "50c11ffab2a3d50192a228c313f05b5b5dc5acb2"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.0+0"

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
deps = ["CRlibm", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "815e74f416953c348c9da1d1bc977bbc97c84e18"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.0"

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
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

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
git-tree-sha1 = "3acf07f130a76f87c041cfb2ff7d7284ca67b072"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.2+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2a7a12fc0a4e7fb773450d17975322aa77142106"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.2+0"

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
git-tree-sha1 = "e24491cb83551e44a69b9106c50666dea9d953ab"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.4.4"

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
git-tree-sha1 = "f19301ae653233bc88b1810ae908194f07f8db9d"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

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
git-tree-sha1 = "1f7f9bbd5f7a2e5a9f7d96e51c9754454ea7f60b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.4+0"

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
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "5e9fe23c86d3ca630baa1efcad78575a27f158b2"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.0.11"

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
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"
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
git-tree-sha1 = "b117c1fe033a04126780c898e75c7980bf676df3"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.7"

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
git-tree-sha1 = "cec2df8cf14e0844a8c4d770d12347fda5931d72"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.25.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

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

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

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
# ╠═441c1d74-5bbb-45f8-a150-4a66a5d51379
# ╠═a4342153-d03e-42f5-9448-b3ff632672ea
# ╠═d0fbd753-27d5-4649-82b1-05a3e1cba2f8
# ╠═e5b31309-a7cb-43e5-8173-f5ed8bbf316e
# ╠═b0b92aec-7643-47b7-a0fc-b44637a6fe91
# ╠═6d7c2f32-6b22-4c72-87ee-5d71aa4fa60a
# ╠═dcccf82a-e6f0-423d-b445-e09f7054caa4
# ╠═236eb5d3-65c2-409e-b5b9-b4006f04755a
# ╠═676fd408-d6fe-42fe-a8df-d79fa1007fc9
# ╠═516f737f-2729-4a7a-8395-f8227ba790b9
# ╠═6773199d-d376-4b80-926a-8518feccd7dc
# ╟─60ba94e8-c7fd-4b3d-89fb-bec0d5bc815f
# ╠═5d127678-16a9-4845-a484-3c83278143d5
# ╠═d8b0a2dd-3632-449b-a06d-509a555aaa31
# ╠═f3f4e738-89f5-4666-bbd9-cdc6c3650bb7
# ╠═e26f2508-c8a4-4057-a5fd-740e19e42e12
# ╠═6b5c2c47-66f8-4375-8850-efca4819e057
# ╠═55a26792-edd7-44d3-9ebc-1f3cc540101a
# ╠═18fe8d93-28c9-4da5-ad79-82150d2d13e8
# ╠═40bcb115-eff9-4d1a-8da7-d4d51a9ea176
# ╠═e6e752c3-411d-404f-aa85-77fdaba6ed59
# ╠═ada48a44-af10-4a91-b8f8-8aae066b5d53
# ╟─db007083-7d8f-4f8f-9629-96df08e514f4
# ╟─44909380-710e-4c4d-8ea9-cc03c62309e0
# ╠═b6e810b1-cf37-4736-8f45-812290a6daf4
# ╠═17367114-1f47-40eb-83f1-31fb88563e4d
# ╟─939a6edf-acf5-40e1-99a4-51e75394e55b
# ╠═c9ebec81-aa29-49db-8c8e-209741db0372
# ╟─1b7e6662-ab53-4a63-81a3-4b2b626e0e7e
# ╠═f40fa3a0-8064-4c6b-b5c2-9cd838919c9d
# ╟─88d9865b-796d-4ed5-a2d5-862236f3ae54
# ╠═a87f8586-fa96-4940-9f88-f97831ed9c1a
# ╠═fb9f60d9-4f44-411f-9bfb-3b12830d4f78
# ╠═6049b54f-6d12-4608-b5d4-9eea447a28f5
# ╟─07aed418-5f61-453e-bee0-d2e1c0a489c4
# ╠═cf21558e-df00-4f36-8f11-0b2c407e0c16
# ╟─6fd85349-9934-4fcf-a5d6-9e31175aa422
# ╠═f110e4f8-e3d7-4f38-af4d-39901ce3a13b
# ╠═c1a22777-0fbf-4366-a757-b38a02f79507
# ╠═b15d0354-b4b8-4378-8a3c-350a81058b57
# ╠═9803fab5-2434-4b9c-91bc-869f95ed72f4
# ╠═4c02499c-b898-47c0-a447-cb9bd59da403
# ╠═e942a0e0-f194-4ccf-bade-254a7bdc0c02
# ╠═edf8d0b2-86b9-4c43-9d9e-9fdc5dff20d8
# ╠═5901b076-656d-4291-8a4f-dcd68efcf039
# ╠═85465c5d-f4d5-4e60-8269-8293bb5d3e2a
# ╠═d6fab981-9367-46a2-bbbf-ff414069f144
# ╠═f1cd1f44-cd7b-4162-a51f-1c533cd79721
# ╠═f83f05ae-4386-4b0c-97be-288b74fc6154
# ╠═3dc546d1-10f4-4036-b8e2-93963b5e78e0
# ╠═7c79a861-356b-4694-8974-4ee748ffde10
# ╠═d0737e80-f7ad-42d8-946d-fa584ac3e166
# ╠═624256ff-1209-44fc-aa85-3749b61e2186
# ╠═d9cc41ce-4e3a-4e59-a1e7-180e06b15d55
# ╠═ef0dccc2-4dbf-4c2f-baf1-3ba015ac2bd8
# ╠═51ab33a4-bb9f-457f-8341-4859cb5c743b
# ╠═c8912513-9718-44b7-bc92-cf6636de3ecd
# ╠═bf12a47d-436a-41fd-a145-656c58378852
# ╠═fb387b29-d335-4c84-bf2a-75f88a534f2b
# ╠═4d6890fd-d7f6-432b-a744-2ea11da65ff1
# ╠═ed4e6c7d-4741-4630-91e5-acac51f29e06
# ╠═4725c07e-580d-4bba-ad77-89a189491308
# ╠═99940e90-3580-446c-9637-7b39c1688935
# ╠═27d2effa-2d30-4217-90f9-07f06eeefae4
# ╠═86c476ee-8da7-4e64-8259-784f4a6095f0
# ╠═2b1d18f2-b82c-4058-a6cb-65ff468ed1c4
# ╠═8daab1ff-f661-4772-89f6-569d9d0246f3
# ╠═6bf82f04-b7f9-4ac4-a48a-948f7cdbad16
# ╠═e0ac337f-2653-4973-a2e6-6559ce58c428
# ╠═2e275057-1155-4edf-bb76-cf870092cb9a
# ╠═6de812cb-8d2b-45fc-bf99-685083b07a9a
# ╠═f3d805da-da19-4d9e-b4e6-69da4f19a833
# ╠═a08be402-718b-4e47-a68d-9d86e52a6b0a
# ╟─825e58e2-10b4-4a5f-99dc-573517fe32fe
# ╠═053c4e15-4fc7-4ef4-aa6a-26af5fa83fda
# ╠═4b58e7dc-2a33-4051-918c-a49d39cd6c29
# ╠═82865d9e-9329-41e6-a53a-a0da2daa9532
# ╠═9d734553-a5c8-4804-bf64-14dd1d04269d
# ╠═f44a4152-f111-4ca9-8fc5-7bf25bef7298
# ╠═dc69dc02-37c0-40cb-bed8-9251d0285abf
# ╠═0443aa96-730e-44d3-b372-cabe20bb5c1a
# ╠═ad93f5ad-b0ad-497c-9314-aaee78e25940
# ╠═9fd5ad6b-aec7-4a5d-986b-0abec38da5f5
# ╠═cc7b46e7-3475-44a5-9684-471d11e24ad6
# ╠═92a909f0-7697-4f3a-9c92-10003144783d
# ╠═18070fd0-a0ce-4e28-ad5a-a39568460f43
# ╠═fea217b0-a5b9-4379-834b-393a2d0ad05e
# ╠═2f587a99-6ff5-40cd-9415-4383f39571d4
# ╠═21a6230c-9284-4246-83d8-4eefd8616b90
# ╠═ebfdc2d6-431c-4ced-b8d2-0f534829d16e
# ╠═d4c3447e-8dae-4201-8d39-11bba68b56c5
# ╠═c2aa615c-4231-4173-a6db-81a5be35cabd
# ╠═211246a2-7bd6-47c3-882c-7804e31a012f
# ╠═3ea13784-73f0-4f70-bd72-edc4e81efa59
# ╠═3bd0d3fe-619d-4f10-938c-c60d5cfb49e5
# ╠═6667de81-671d-454c-8d73-f0309f646b60
# ╠═b0d59912-88b6-42bf-9328-aed96c0a3515
# ╠═8cb9b650-12fb-4691-925e-51e806cb12fd
# ╠═3b124dcd-6e93-43d6-bb00-e144926a0a58
# ╠═0b68d40c-e7b0-4f4e-bf0d-d009a14707b3
# ╠═c21b594c-a8de-4150-abf2-4d937052fdb7
# ╠═ee57a9c8-94ee-4c7b-8e53-20cab12fa64a
# ╠═3add657e-ce09-4e4e-ac94-ac522e927ef8
# ╠═14b7d938-be7c-445b-b0a7-6088eb1a3fec
# ╠═702230c6-d538-4dd1-822b-57f2bb4c6818
# ╠═5e4ea456-25d4-4138-8c0c-42d1a45828b7
# ╠═6b3462fd-8d95-401f-be37-c394f4ae5fb3
# ╠═35976ef0-ab93-4192-abae-b36fad04ca99
# ╠═9e5d3e99-2fd5-4ef8-afe1-baeaf4339c84
# ╠═f700462d-89af-4474-b063-92ddbfa4d079
# ╠═6355f453-5406-4899-a92e-9be360e9629e
# ╠═7a208658-8fb0-423c-bb95-95c0101fd98e
# ╠═d584e00d-3da1-44cf-b62a-1c4bc3144c90
# ╠═abc2d0e1-17f2-49d3-8d1b-6bcdf3210f72
# ╠═cb517530-9ce3-4774-b6ed-ce2d636cb765
# ╠═11f86686-0165-4376-8847-42e68c30cc65
# ╠═68b1fb68-61aa-4f1e-b4b9-c27eb9378059
# ╠═ea5552f4-e5d1-4a56-a655-5af6eb9cbf25
# ╟─eff5776e-7558-4c2b-8ce6-6b3d1107a511
# ╠═a1bf5a1b-36a8-4c65-bb6b-86847f25b569
# ╠═959644f5-abc5-4bd5-8a43-3bf19776acb1
# ╠═5c88c45b-841a-4e7c-970b-6ae9a1ad38bd
# ╠═eb6f43e9-7b05-4208-9165-5e3db9346151
# ╠═8a580804-4c9d-4eb7-b874-46f06282b4c0
# ╠═a1a3aa17-a3c0-4503-910a-3caec607e9a6
# ╠═fe5d9a4a-6169-4090-828b-44d0367ea949
# ╠═2e6a5fc2-ccc9-4afd-95ea-349856d86e86
# ╠═cf4f7625-a10c-4ccc-b259-b1e538e91246
# ╠═408a192f-0844-4fa4-a58e-45498f9b3bbf
# ╠═0ff31629-855a-49a3-9b7d-cf4316c809da
# ╠═f31b04f7-c119-4993-9e3d-f83d6bdce4be
# ╠═34280c20-950c-4083-948b-98e8edce0724
# ╠═6cf01d9d-f444-400a-9700-3e3e55cc1727
# ╠═f09814a6-82b8-457c-9d75-0c92d8612056
# ╠═e8496939-0407-4e92-a4cc-8d04bdb8a9c8
# ╠═27523365-20a0-417f-a038-7e2b2f1bb832
# ╠═e123bf7f-7113-44e0-a955-d20520f84872
# ╠═7a39d667-e033-4915-94c1-41f12a02b944
# ╠═d41e5a93-b200-4f35-a49b-0cddaac9bce7
# ╠═60356d6e-f242-4c22-bd29-47dc827a0d20
# ╠═63333781-ee80-458f-9287-094e2e7529b8
# ╠═8daebfb1-30c8-46db-8a57-4d8969fd539b
# ╠═32c861fb-4d5a-4886-aa04-2a7338525740
# ╠═8ba3f40d-b249-4636-9df3-ccd315bc8fa6
# ╠═56a4169a-7f74-4401-81a7-1355bfd4812e
# ╠═abcd5019-bfaa-467e-903b-7e711ba0d3a4
# ╠═8ec8b772-69b1-4a06-b800-3bd02610a105
# ╠═313d3886-f84d-4561-85e2-9ff9cc60ffaf
# ╠═42c0154b-026d-4a26-b291-bbb87e124611
# ╠═623ed842-a49c-4ac2-9a8b-8e69367ba517
# ╠═759aa4ff-7f40-428e-8285-066fdc02982b
# ╠═a9d7e50a-eb07-4721-96bd-54765058ad60
# ╠═2b534ad4-a62d-4983-b43c-274f609c159d
# ╠═44c4ddfe-3017-4e33-81e3-b067d88609fc
# ╠═af6ef77a-f8e3-45ca-b99b-74994f247470
# ╠═5f6351f0-41ff-426e-be67-00572cf96f88
# ╠═5641018b-f5ff-42aa-a153-83c2da5eea88
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
