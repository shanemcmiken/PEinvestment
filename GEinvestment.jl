###############################################################################
#
# GEinvestment.jl solves the KT general equlibrium model of firm investment with labor
#
# Shane McMiken
# November 2021
# Boston College
#
# CONSUMER PROBLEM:
#   max_{N_{t+s}, B_{t+s+1}} \sum_{s ≥ 0} β^s (log C_{t+s} - ϕ N_{t+s})
#   s.t. C_{t+s} + B_{t+s+1} = W * N_{t+s} + R * B_{t+s} + ∫ d(z,k) dF(z,k)
#   
#   steady state: R = β^-1, W = ϕ * C
#
# RECURSIVE FIRM PROBLEM:
# V(Z,K) = max_K' {R(Z,K,K') + (1/(1+r))* E[V(Z',K')|Z] } 
# 
# WHERE R(z,k,k') = Z * K^α * N^ν - pK + p(1-δ)K - AC(Z,K,K')
#
################################################################################

#--------------------------------#
#         PREAMBLE
#--------------------------------#
using Plots # plots
using Distributions # cdf(normal())
using Parameters  # macros: @with_kw & @unpack
#using DelimitedFiles # writedlm 

#--------------------------------#
#       TAUCHEN (1986)
#--------------------------------#

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
#         PARAMETERIZATION
#--------------------------------#

@with_kw struct params
    
    # firm parameters 
    α::Float64      = 0.25 # capital share
    δ::Float64      = 0.1 # depreciation rate
    r::Float64      = 0.04 # interest rate
    ρ::Float64      = 0.8 # technology persistence 
    σ::Float64      = 0.2 # technology volatility
    γ_c::Float64    = 2.5 # variable investment adjustment cost
    γ_f::Float64    = 0.0 # fixed investment adjustment cost

    # household parameters
    β::Float64      = 1.0/0.04 # discount rate
    ϕ::Float64      = 2.0 #disutility of labor
    ν::Float64      = 0.5 #labor share

    # wage bisection bounds
    a::Float64      = 0.4 #wage lower bound
    c::Float64      = 1.8 #wage upper bound

end

#--------------------------------#
# INNER LOOP - VFI & ERGODIC DIST
#--------------------------------#

function vfivec(W::Float64, kkgr::Array, zzgr::Array, kgrid::Array, zgrid::Array;α = α, ν = ν, r = r, δ = δ, γ_f = γ_f, γ_c = γ_c, maxiter = 1000, tol = 1e-4, nfix = 10, iter = 0, verr = Inf) 

    # objects
    nz          = length(zgrid)
    nk          = length(kgrid)
    labor       = zzgr.^(1.0/(1.0-ν)).*(ν/W).^(1.0/(1.0-ν)).*kkgr.^(α/(1.0-ν))
    output      = zzgr .* kkgr.^α .* labor.^ν 
    investment  = kgrid' .- (1.0 - δ).*kkgr # kgrid - k' policy
    adjcost     = γ_c/2.0 .* (investment./kkgr).^2 .*kkgr + γ_f.*output.*(kgrid'.>(1-δ)*kkgr)  
    dividend    = repeat(output,1,nk) - W * repeat(labor,1,nk) - investment - adjcost
    

    # steady state
    kbar        = (α/(r+δ))^((1.0-ν)/(1.0-α-ν))*(ν/W)^(ν/(1.0-α-ν))
    nbar        = (ν/W)^(1.0/(1.0-ν))*kbar^(α/(1.0-ν))
    Vbar        = ((1.0+r)/r)*(kbar^α*nbar^ν - δ*kbar -W*nbar)

    # initialise
    V_init  = fill(Vbar, (nz,nk))
    V_next  = Array{Float64,1}
    index   = Array{CartesianIndex{2},2}

    while (verr > tol && iter < maxiter) # inner-loop value function

        if mod(iter,nfix) == 0 # Howard acceleration
            
            # obj dims: V(z×k,k')
            obj     = dividend + (1.0/(1.0+r)).*repeat(P*V_init,nk) # compute values for V(z×k,k')
            V_next, index = findmax(obj, dims = 2) # get maximum

        else

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
            investment_idx  = kgrid[idx] .- (1.0 - δ).*kkgr 
            adjcost_idx     = γ_c/2.0 .* ((kgrid[idx] .- (1.0-δ)*kkgr)./kkgr).^2 .*kkgr + γ_f.*output.*(kgrid[idx].>(1-δ)*kkgr)  
            dividend_idx    = output - W * labor - investment_idx - adjcost_idx

            # Re-compute value function under conjectured optimal policy
            V_next = dividend_idx + (1.0+r).^(-1) .* evp_k

        end

        # check tolerance
        V_next  = reshape(V_next,nz,nk)
        verr    = maximum(abs.(V_next .- V_init))
        V_init  = copy(V_next)
        iter   += 1
        #print("iteration: ",iter, ", critical val: ", verr*1.0e4," times 10^4", "\n")

    end # inner-loop: value function converges

    index = [index[i][2] for i∈1:(nz*nk)] # policy funciton   
    return index, labor, output, investment, adjcost

end

function young(index::Vector{Int64}; maxiter = 1000, tol = 1e-4, iter = 0, crit = Inf, nz = nz, nk = nk)

    # initialise
    μ_init  = fill((1/(nz*nk)), (nz,nk))

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
    μ_ergodic = vec(μ_init)
    return μ_ergodic
end

#--------------------------------#
# OUTER LOOP - WAGE
#--------------------------------#

function geinvest(a_init::Float64, c_init::Float64, kgrid::Array,zgrid::Array; wmaxiter::Int64 = 50, tol::Float64 = 1e-4, nk::Int64 = nk, nz::Int64 = nz, α::Float64=α, δ::Float64 = δ, r::Float64 = r, γ_c::Float64 = γ_c, γ_f::Float64 = γ_f, ϕ::Float64 = ϕ, ν::Float64 = ν, iter = 0, err = Inf)

    # wage initialisation
    a = copy(a_init) 
    c = copy(c_init)
    b = (a+c)/2.0
    W_next = (a_init + c_init)/2.0

    # useful grids
    kkgr = vec(repeat(kgrid',nz))
    zzgr = repeat(zgrid,nk,1)
    μ_ergodic = Vector{Float64}
    index = Vector{Int64}

    while (err > tol && iter < wmaxiter) #outer loop for wage
        
        # update prices
        W = copy(W_next)

        # value-function iteration
        index_stat, labor, output, investment, adjcost = vfivec(W,kkgr,zzgr,kgrid,zgrid) 
        index = copy(index_stat)

        # obtain ergodic distribution
        μ = young(index)
        μ_ergodic = copy(μ)

        #------------------------------------------#
        #  COMPUTE AGGREGATES
        #------------------------------------------#
    
        Lagg        = sum(μ_ergodic.*labor)
        Yagg        = sum(μ_ergodic.*output)
        Kagg        = sum(μ_ergodic.*kkgr)
        Zagg        = sum(μ_ergodic.*zzgr)
        Iagg        = sum(μ_ergodic.*investment[index])
        ACagg       = sum(μ_ergodic.*adjcost[index])
        Cagg        = Yagg .- Iagg .- ACagg
        
        # compute error for prices outer-loop
        diff        = W - ϕ * Cagg
        err         = abs(diff)
        iter        += 1 

        # update prices for outer loop
        if diff > 0
            c = copy(b)
        elseif diff < 0
            a = copy(b)
        end
        b = (a+c)/2.0
        W_next = b

        #print("iteration: ",iter, ", critical val: ", err*1.0e4," times 10^4 ", " wage ", W, " diff ", diff,   " Cagg ", Cagg,  "\n")

    end # outer-loop: wage converges

    return index,W_next,μ_ergodic
end

#--------------------------------#
#         RUN MODEL
#--------------------------------#

@unpack α, δ, r, ρ, σ, γ_c, γ_f, β, ϕ, ν, a, c= params()

# technology grid
cover = 5
nz    = 2*cover+1
m     = 2.5
zgrid, P = MyTauchen(σ,ρ,nz,m)
zgrid = exp.(zgrid) # exponentiate grid

# capital grid
nk = 100
kmax = (α/(r+δ))^((1.0-ν)/(1.0-α-ν))*(ν/a)^(ν/(1.0-α-ν))*maximum(zgrid)^(1.0/(1.0-α-ν))
kgrid = [kmax*(1-δ)^(nk-i-1) for i∈0:(nk-1)]

# Convex adjustment cost 
@time index, W, μ_ergodic  = geinvest(a, c, kgrid, zgrid)

# policy function

#ergodicdist = heatmap(kgrid,zgrid,reshape(μ_ergodic,nz,nk), title = "Ergodic distribution")
#display(ergodicdist)
#savefig(ergodicdist,"./src/GEergodicdistribution.png")

policyfunc = heatmap(kgrid,zgrid,reshape(index,(nz,nk)), title = "k'(z,k) Policy")
display(policyfunc)
savefig(policyfunc,"./src/GEpolicyfunction_ConvexAC.png")


# Fixed Adjustment cost 
γ_c, γ_f = 0.0, 0.05
@time index, W, μ_ergodic  = geinvest(a, c, kgrid, zgrid)

# policy function
policyfunc = heatmap(kgrid,zgrid,reshape(index,(nz,nk)), title = "k'(z,k) Policy")
display(policyfunc)
savefig(policyfunc,"./src/GEpolicyfunction_FixedAC.png")
