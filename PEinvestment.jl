###############################################################################
#
# PEinvestment.jl solves the Benchmark PE Firm Investment Model
#
# Shane McMiken
# November 2021
# Boston College
#
################################################################################

#--------------------------------#
#         PREAMBLE
#--------------------------------#
using Plots
using Distributions
using QuantEcon
using Distributed

#--------------------------------#
#         PARAMETERIZATION
#--------------------------------#
α   = 0.7
p   = 1.2
δ   = 0.1
r   = 0.04
ρ   = 0.8
σ   = 0.2
γ_c = 2.5
γ_f = 0.0

#steady state
kbar = (α/(p*(r+δ)))^(1.0/(1.0-α))
Vbar = ((1+r)/r)*(kbar^α - δ*kbar)

#--------------------------------#
#  GRIDS and TRANSITION MATRIX
#--------------------------------#

# Tauchen (1986)
function MyTauchen(σ_a::Float64,ρ_a::Float64,na::Int64,m::Float64)

  # Discrete markov process approximation for AR(1)
  # y = ρ_a * y_  + σ_a * ϵ 
  # m: number of standard deviations
  # σ_a: standard deviations
  # ρ_a: AR(1) parameter
  # na: number of grid points

  vecsize = na
  σ_y = σ_a / sqrt(1-(ρ_a^2)) # Stationary distribution variance
  step = 2 * m * σ_y / (vecsize-1)
  agrid  = [-m*σ_y + (i-1)*step for i∈1:na]

  # Calculate transition probabilities, AR(1) - Tauchen
  P = zeros(na, na)

  mm = agrid[2] - agrid[1]; # re-compute step
  for j = 1:na
    for k = 1:na
      if(k == 1) # first column
        P[j, k] = cdf(Normal(), (agrid[k] - ρ_a*agrid[j] + (mm/2))/σ_a)
      elseif(k == na) # second-last column
        P[j, k] = 1 - cdf(Normal(), (agrid[k] - ρ_a*agrid[j] - (mm/2))/σ_a)
      else
        P[j, k] = cdf(Normal(), (agrid[k] - ρ_a*agrid[j] + (mm/2))/σ_a) - cdf(Normal(), (agrid[k] - ρ_a*agrid[j] - (mm/2))/σ_a)
      end
    end #k
  end #j

  # agrid = exp.(agrid) # Exponentiate grid

  return agrid, P #return grid and Markov transition matrix
end

# technology grid
cover = 5
nz    = 2*cover+1
m     = 2.5
zgrid, P = MyTauchen(σ,ρ,nz,m)
zgrid = exp.(zgrid) # exponentiate grid

# capital grid
nk = 100
kmax = ((maximum(zgrid)*α)/(p*(r+δ)))^(1.0/(1.0-α))
kgrid = [kmax*(1-δ)^i for i∈0:(nk-1)]

#--------------------------------#
#  VALUE FUNCTION ITERATION
#--------------------------------#
# RECURSIVE PROBLEM:
#
# V(z,k) = max_k' {R(z,k,k') + (1/(1+r))* E[V(z',k')|z] } 
# 
# WHERE R(z,k,k') = zk^α - pk + p(1-δ)k - AC(z,k,k')

struct state
  ind::Int64
  nz::Int64 # number of prod grid points
  nk::Int64 # number of capital grid points
  P::Array
  zgrid::Vector{Float64} # technology grid
  kgrid::Vector{Float64} # capital grid
  r::Float64 # interest rate rate
  δ::Float64 # Depreication rate
  α::Float64 # Capital share of income
  p::Float64 # price of investment goods
  γ_c::Float64 # variable adjustment cost
  γ_f::Float64 # fixed adjustment cost
  V::Array
end

function value(currentstate::state)

  ind     = currentstate.ind
  nz      = currentstate.nz
  nk      = currentstate.nk
  P       = currentstate.P
  zgrid   = currentstate.zgrid
  kgrid   = currentstate.kgrid
  r       = currentstate.r
  δ       = currentstate.δ
  α       = currentstate.α
  p       = currentstate.p
  γ_c     = currentstate.γ_c
  γ_f     = currentstate.γ_f
  VV      = currentstate.V

  # get states
  kk      = convert(Int, floor((ind-0.05)/nz))+1
  zz      = convert(Int, floor(mod(ind-0.05,nz))+1)
  kc      = kgrid[kk] # current capital
  zc      = zgrid[zz] # current technology

  # compute continuation value
  cont    = P * VV 
  cont    = cont[zz,:]

  # compute return function
  R       = zc*kc^α .- p.*kgrid .+ p*(1.0-δ)*kc .- γ_c/2.0 .* ((kgrid .- (1.0-δ)*kc)./kc).^2 .*kc - γ_f.*zc.*kc^α.*(kgrid.>(1-δ)*kc)

  # compute new value funciton
  VVp     = R .+ (1.0+r).^(-1) .* cont

  # find maximum
  Vnew, tempidx = findmax(vec(VVp))

  return Vnew, tempidx
end

#--------------------------------#
#  MAIN LOOP
#--------------------------------#

function vfi(Vbar::Float64, zgrid::Array, kgrid::Array, P::Array; α = α, r = r, δ = δ, γ_c = γ_c, γ_f = γ_f, p = p,crit = Inf, maxiter = 10000, iter = 0, tol = 1.0e-5, nfix = 10)
  
  nz = size(zgrid,1)
  nk = size(kgrid,1)
  V_init = fill(Vbar, (nz,nk))
  V_next = Array{Float64,1}
  index = ones(nz*nk,)

  # useful grids
  kkgr = vec(repeat(kgrid',nz))
  zzgr = repeat(zgrid,nk,1)

  while (crit > tol && iter < maxiter)

    V_next = copy(V_init)

    if mod(iter,nfix) == 0 # Howard acceleration

      for ind = 1:(nz*nk)

        modelstate = state(ind,nz,nk,P,zgrid,kgrid,r,δ,α,p,γ_c,γ_f,V_next)
        tempVal, tempIndex  = value(modelstate)

        V_next[ind] = tempVal
        index[ind] = tempIndex
      end

    else # on mod(iter,nfix) != 0, update the value function only.

      # Re-use conjectured index
      idx = convert(Array{Int64,1}, index)

      # E(V) condition on A(t) and each possible K(t+1)
      EVp = P*V_init
      evp_k = zeros(nk*nz,)
      for ind = 1:(nz*nk)
        kk = convert(Int, floor((ind-0.05)/nz))+1
        zz = convert(Int, floor(mod(ind-0.05,nz))+1)
        evp_k[ind] = EVp[zz,idx[ind]]
      end

      # Compute return function using conjectured policy - idx
      R = zzgr.*kkgr.^α .- p.*kgrid[idx].+ p.*(1.0-δ).*kkgr .- γ_c/2.0 .* ((kgrid[idx] .- (1.0-δ)*kkgr)./kkgr).^2 .*kkgr - γ_f.*zzgr.*kkgr.^α.*(kgrid[idx].>(1-δ)*kkgr)

      # Re-compute value function under conjectured optimal policy
      V_next = R + (1.0+r).^(-1) .* evp_k

    end

    # check tolerance
    V_next = reshape(V_next,nz,nk)
    crit = maximum(abs.(V_next .- V_init))
    V_init = copy(V_next)
    iter += 1

    print("iteration: ",iter, ", critical val: ", crit*1.0e5," times 10^5", "\n")
  end  

  return V_next, index, crit
end

@time valfunc, index, crit = vfi(Vbar,zgrid,kgrid,P)

#------------------------------------------#
#  VALUE FUNCTION ITERATION  - VECTORIZED
#------------------------------------------#

function vfivec(Vbar::Float64, zgrid::Array, kgrid::Array, P::Array; α = α, r = r, δ = δ, γ_c = γ_c, γ_f = γ_f, p = p,crit = Inf, maxiter = 10000, iter = 0, tol = 1.0e-5, nfix = 10)

  

end






