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
using Plots # plots
using Distributions # cdf(normal())
using Parameters  # macros: @with_kw & @unpack
using DelimitedFiles # writedlm 

#--------------------------------#
#         PARAMETERIZATIONS
#--------------------------------#

@with_kw struct NoAC
  α::Float64    = 0.7
  p::Float64    = 1.2
  δ::Float64    = 0.1
  r::Float64    = 0.04
  ρ::Float64    = 0.8
  σ::Float64    = 0.2
  γ_c::Float64  = 0.0
  γ_f::Float64  = 0.0
end

@with_kw struct ConvexAC
  α::Float64    = 0.7
  p::Float64    = 1.2
  δ::Float64    = 0.1
  r::Float64    = 0.04
  ρ::Float64    = 0.8
  σ::Float64    = 0.2
  γ_c::Float64  = 2.5
  γ_f::Float64  = 0.0
end

@with_kw struct FixedAC
  α::Float64    = 0.7
  p::Float64    = 1.2
  δ::Float64    = 0.1
  r::Float64    = 0.04
  ρ::Float64    = 0.8
  σ::Float64    = 0.2
  γ_c::Float64  = 0.0
  γ_f::Float64  = 0.05
end

#--------------------------------#
#  DESCRETIZE AR(1) PROCESS
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

  # compute new value funciton - if return is less than zero than value is -Inf
  VVp     = R .+ (1.0+r).^(-1) .* cont - Inf*(R.<0) 

  # find maximum
  Vnew, tempidx = findmax(vec(VVp))

  return Vnew, tempidx
end


function vfi(Vbar::Float64, zgrid::Array, kgrid::Array, P::Array, γ_c, γ_f, α, r, δ, p ; crit = Inf, maxiter = 10000, iter = 0, tol = 1.0e-6, nfix = 10)
  
  nz = length(zgrid)
  nk = length(kgrid)
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
      V_next = R + (1.0+r).^(-1) .* evp_k - Inf*(R.<0)  

    end

    # check tolerance
    V_next = reshape(V_next,nz,nk)
    crit = maximum(abs.(V_next .- V_init))
    V_init = V_next
    iter += 1

    #print("iteration: ",iter, ", critical val: ", crit*1.0e5," times 10^5", "\n")
  end  

  index = convert(Vector{Int64}, index)
  return V_next, index, crit
end

#------------------------------------------#
#  VFI FAST - VECTORIZED
#------------------------------------------#

function vfivec(Vbar::Float64, zgrid::Array, kgrid::Array, P::Array, γ_c, γ_f, α, r, δ, p ; crit = Inf, maxiter = 10000, iter = 0, tol = 1.0e-8, nfix = 10)

  nz = length(zgrid)
  nk = length(kgrid)
  V_init = fill(Vbar, (nz,nk))
  V_next = Array{Float64,1}
  index = Array{CartesianIndex{2},2}

  # useful grids
  kkgr = vec(repeat(kgrid',nz))
  zzgr = repeat(zgrid,nk,1)

  # compute model objects
  R = zzgr.*kkgr.^α .- p*kgrid' .+ p*(1.0-δ)*kkgr .- γ_c/2.0 .* ((kgrid' .- (1.0-δ)*kkgr)./kkgr).^2 .*kkgr - γ_f.*zzgr.*kkgr.^α.*(kgrid'.>(1-δ)*kkgr)  

  while (crit > tol && iter < maxiter)
    
    if mod(iter,nfix) == 0 # Howard acceleration
      
      # compute (nz*nk*nk)-dimensional value function
      obj = R + (1.0/(1.0+r)).*repeat(P*V_init,nk) - Inf*(R.<0)

      # Get maximum
      V_next, index = findmax(obj, dims = 2)
      
    else # on mod(iter,nfix) != 0, update the value function only.

      # Re-use conjectured index
      idx = [index[i][2] for i∈1:(nz*nk)]

      # E(V) condition on A(t) and each possible K(t+1)
      EVp = P*V_init
      evp_k = zeros(nk*nz,)
      for ind = 1:(nz*nk)
        zz = convert(Int, floor(mod(ind-0.05,nz))+1)
        evp_k[ind] = EVp[zz,idx[ind]]
      end

      # Compute return function using conjectured policy - idx
      R_idx = zzgr.*kkgr.^α .- p.*kgrid[idx].+ p.*(1.0-δ).*kkgr .- γ_c/2.0 .* ((kgrid[idx] .- (1.0-δ)*kkgr)./kkgr).^2 .*kkgr - γ_f.*zzgr.*kkgr.^α.*(kgrid[idx].>(1-δ)*kkgr)

      # Re-compute value function under conjectured optimal policy
      V_next = R_idx + (1.0+r).^(-1) .* evp_k - Inf*(R_idx.<0) 

    end

    # check tolerance
    V_next = reshape(V_next,nz,nk)
    crit = maximum(abs.(V_next .- V_init))
    V_init = copy(V_next)
    iter += 1
    
    #print("iteration: ",iter, ", critical val: ", crit*1.0e5," times 10^5", "\n")
  end

  index = [index[i][2] for i∈1:(nz*nk)]
  return V_init, index, crit

end

#------------------------------------------#
#  ERGODIC DISTRIBUTION - Young (2010)
#------------------------------------------#

function young(index::Array, P::Array, nz::Int64, nk::Int64 ; crit = Inf, maxiter = 10000, iter = 0, tol = 1.0e-6)

  # initialise mass
  μ_init = fill((1/(nz*nk)), (nz,nk))

  while (crit > tol && iter < maxiter)
    
    # intialise
    μ_next = zeros(nz,nk)

    # use policies from vfi to infer masses next period
    for ind = 1:(nz*nk)
      kk = convert(Int, floor((ind-0.05)/nz))+1
      zz = convert(Int, floor(mod(ind-0.05,nz))+1)
      μ_next[zz,index[ind]] =  μ_next[zz,index[ind]] + P[:,zz]'*μ_init[:,kk] 
    end
    
    # check tolerance
    crit = maximum(abs.(μ_init .- μ_next))
    μ_init = μ_next
    iter += 1
    
    #print("iteration: ",iter, ", critical val: ", crit*1.0e5," times 10^5", "\n")

  end

  return μ_init, crit, iter
end

#------------------------------------------#
#  EXTRACT AGGREGATES FROM THE MODEL
#------------------------------------------#

function moments(μ::Matrix,zgrid::Vector,kgrid::Vector,index::Vector, valfunc::Array, α, δ)
  
  nz = length(zgrid)
  nk = length(kgrid)

  #plot policy function
  pols = reshape(index,(nz,nk)) # policies

  # Get useful grids
  kkgr = log.(repeat(kgrid',nz))
  zzgr = log.(repeat(zgrid,1,nk))
  yygr = log.(exp.(zzgr).*exp.(kkgr).^α)
  iigr = (kgrid[pols]-(1-δ).*exp.(kkgr))./exp.(kkgr)

  # mean
  X = [kkgr, zzgr, yygr, iigr]
  EX = zeros(size(X),)
  for (ii, x) in enumerate(X)
    EX[ii] = sum(x.*μ)
  end
  
  #Std deviation
  STDX = zeros(size(X),)
  for (ii, x) in enumerate(X)
    STDX[ii] = sqrt.(sum(x.^2 .*μ) - EX[ii].^2)
  end 
  
  # correlation matrix
  CORRX = zeros(size(X)[1],size(X)[1])
  for (ii,xi) in enumerate(X)
    for (jj,xj) in enumerate(X)
      CORRX[ii,jj] = (sum(xi.*xj.*μ) - EX[ii]*EX[jj])/(STDX[ii]*STDX[jj])
    end
  end
  
  
  return EX, CORRX

end

#------------------------------------------#
#  OBTAIN PLOTS
#------------------------------------------#

# solve model
para = [NoAC, ConvexAC, FixedAC]
strpara = ["NoAC", "ConvexAC","FixedAC"]

for (ii,str) in enumerate(para)
  
  @unpack α, p, δ, r, ρ, σ, γ_c, γ_f= str()

  #steady state
  kbar = (α/(p*(r+δ)))^(1.0/(1.0-α))
  Vbar = ((1.0+r)/r)*(kbar^α - δ*kbar)

  # technology grid
  cover = 5
  nz    = 2*cover+1
  m     = 2.5
  zgrid, P = MyTauchen(σ,ρ,nz,m)
  zgrid = exp.(zgrid) # exponentiate grid

  # capital grid
  nk = 100
  kmax = ((maximum(zgrid)*α)/(p*(r+δ)))^(1.0/(1.0-α))
  kgrid = [kmax*(1-δ)^(nk-i-1) for i∈0:(nk-1)]

  #@time valfunc, index, crit = vfi(Vbar,zgrid,kgrid,P) #slower
  @time valfunc, index, crit = vfivec(Vbar, zgrid, kgrid, P, γ_c, γ_f, α, r, δ, p)
  @time μ_ergodic, crit, iterfin = young(index,P,nz,nk)

  ## (a)
  #plot ergodic distribution
  stationarydist = heatmap(kgrid,zgrid,μ_ergodic, title = "Stationary distribution - $(strpara[ii])", ylabel = "z", xlabel = "k")
  distribution = plot(stationarydist, xlabel = "K", ylabel = "Z")
  savefig(distribution,"./src/stationarydistribution_$(strpara[ii]).png")
  display(distribution)

  #plot policy function
  pols = reshape(index,(nz,nk)) # policies
  zlow = pols[cover-2,:] # fix productivity at low
  zmid = pols[cover+1,:] # fix productivity at steady state
  zhigh = pols[cover+4,:] # fix productivity high

  p1 = plot(kgrid,kgrid[zlow], ylabel = "low productivity", title = "k policy - $(strpara[ii])")
  p2 = plot(kgrid,kgrid[zmid], ylabel = "mid productivity")
  p3 = plot(kgrid,kgrid[zhigh], ylabel = "high productivity")
  p4 = heatmap(kgrid,zgrid,pols, ylabel = "Z", xlabel = "K")

  policyfunctions = plot(p1,p2,p3,p4, layout = (2,2), legend = false)
  savefig(policyfunctions,"./src/policyfunctions_$(strpara[ii]).png")
  display(policyfunctions)

  #plot value function 
  p5 = plot(kgrid,valfunc[cover-2,zlow], ylabel = "low productivity", title = "Value Function - $(strpara[ii])")
  p6 = plot(kgrid,valfunc[cover+1,zlow], ylabel = "mid productivity")
  p7 = plot(kgrid,valfunc[cover+4,zlow], ylabel = "high productivity")
  p8 = heatmap(kgrid,zgrid,valfunc, ylabel = "Z", xlabel = "K")

  valuefunction = plot(p5,p6,p7,p8, layout = (2,2), legend = false)
  savefig(valuefunction,"./src/valuefunction_$(strpara[ii]).png")
  display(valuefunction)

  ## (b) marginal Distributions
  marginal_k = sum(μ_ergodic, dims = 1)
  marginal_z = sum(μ_ergodic, dims = 2)
  p9 = plot(zgrid, marginal_z, ylabel = "probability density", xlabel = "productivity" )
  p10 = plot(kgrid, marginal_k', xlabel = "capital" )

  marginaldistributions = plot(p9,p10, layout = (1,2), title = "$(strpara[ii])" , legend = false)
  savefig(marginaldistributions,"./src/marginaldistributions_$(strpara[ii]).png")
  display(marginaldistributions)

  ## (c) moments

  EX, CORRX = moments(μ_ergodic, zgrid, kgrid, index, valfunc, α, δ)
  writedlm( "./src/correlationmatrix_$(strpara[ii]).csv", round.([EX' ; CORRX]; digits = 2), ',')

  ## (d) aggregates
  kkgr  = vec(repeat(kgrid',nz))
  zzgr  = repeat(zgrid,nk,1)
  μ_vec = vec(μ_ergodic)
  Kagg  = sum(μ_vec.*kkgr)
  Zagg  = sum(μ_vec.*zzgr)
  Yagg  = sum(μ_vec.*zzgr.*kkgr.^α)
  Iagg  = sum(μ_vec.*(kgrid[index]-(1-δ).*kkgr))
  Vagg  = sum(μ_vec.*vec(valfunc))
  TFP   = Yagg/Kagg^α
  writedlm("./src/aggregates_$(strpara[ii]).csv", round.([Yagg Iagg Kagg Vagg Zagg TFP]; digits = 2), ',')

  ## (f) implied TFP is larger than average of true underlying productivity because larger (more productive) firms produce more.

  ## (g)
  #find coordinates of maximal mass
  val, cord = findmax(μ_ergodic)
  p11 = plot(zgrid./kgrid[cord[2]],kgrid[pols[:,cord[2]]], ylabel = "k'(z=1,k)", xlabel = "z/k", title = "k(z=1,k) - $(strpara[ii]) ")
  eqpolicy = plot(p11, legend = false)
  savefig(eqpolicy,"./src/eqpolicy_$(strpara[ii]).png")
  display(eqpolicy)

end

