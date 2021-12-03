###############################################################################
#
# reiter.jl solves the KT general equlibrium model of firm investment with labor
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
# WHERE R(z,k,k') = Z * K^α * N^(1-α) - pK + p(1-δ)K - AC(Z,K,K')
#
################################################################################

#--------------------------------#
#         PREAMBLE
#--------------------------------#
using Plots # plots
using Distributions # cdf(normal())
#using QuantEcon
#using Distributed
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
    γ_f::Float64    = 0.05 # fixed investment adjustment cost

    # household parameters
    β::Float64      = 1.0/0.04 # discount rate
    ϕ::Float64      = 2.0 #disutility of labor
    ν::Float64        = 0.5 #labor share

    # wage bisection bounds
    a::Float64      = 0.6 #wage lower bound
    c::Float64      = 2.0 #wage upper bound

end

#--------------------------------#
#       MAIN FUNCTION
#--------------------------------#

function geinvest(kgrid,a,c; wmaxiter = 50, wtol = 1e-4, vmaxiter = 1000, vtol = 1e-4, nfix = 10)

    a_init = copy(a) 
    c_init = copy(c)
    
    # useful grids
    kkgr = vec(repeat(kgrid',nz))
    zzgr = repeat(zgrid,nk,1)

    # initialise 
    viter = 1
    witer = 1

    while (werr > wtol && witer < wmaxiter) #outer loop for wage
        
        W           = (a_init+c_init)/2.0 
        labor       = zzgr.^(1.0/(1.0-ν)).*(ν/W).^(1.0/(1.0-ν)).*kkgr.^(α/(1.0-ν))
        output      = zzgr .* kkgr.^α .* labor.^ν 
        investment  = kgrid' .- (1.0 - δ).*kkgr # kgrid - k' policy
        adjcost     = γ_c/2.0 .* ((kgrid' .- (1.0-δ)*kkgr)./kkgr).^2 .*kkgr + γ_f.*output.*(kgrid'.>(1-δ)*kkgr)  
        dividend    = repeat(output,1,nk) - W * repeat(labor,1,nk) - investment - adjcost

        # steady state
        kbar        = (α/(r+δ))^((1.0-ν)/(1.0-α-ν))*(ν/b)^(ν/(1.0-α-ν))
        nbar        = (ν/b)^(1.0/(1.0-ν))*kbar^(α/(1.0-ν))
        Vbar        = ((1.0+r)/r)*(kbar^α*nbar^ν - δ*kbar -b*nbar)

        # initialise
        V_init = fill(Vbar, (nz,nk))
        V_next = Array{Float64,1}

        while (verr > vtol && viter < maxiter)

            if mod(iter,nfix) == 0 # Howard acceleration
                
                obj     = dividend + (1.0/(1.0+r)).*repeat(P*V_init,nk) - Inf*(dividend.<0) # compute values for V(z×k,k')
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
                V_next = dividend + (1.0+r).^(-1) .* evp_k - Inf*(dividend_idx.<0) 

            end



        end


    end
end

#--------------------------------#
#         RUN MODEL
#--------------------------------#

@unpack α, δ, r, ρ, σ, γ_c, γ_f, β, ϕ, ν, a, c= params()

#steady state
b = (a+c)/2.0 # "middle wage"
kbar = (α/(r+δ))^((1.0-ν)/(1.0-α-ν))*(ν/b)^(ν/(1.0-α-ν))
nbar = (ν/b)^(1.0/(1.0-ν))*kbar^(α/(1.0-ν))
Vbar = ((1.0+r)/r)*(kbar^α*nbar^ν - δ*kbar -b*nbar)

# technology grid
cover = 2
nz    = 2*cover+1
m     = 2.5
zgrid, P = MyTauchen(σ,ρ,nz,m)
zgrid = exp.(zgrid) # exponentiate grid

# capital grid
nk = 50
kmax = (α/(r+δ))^((1.0-ν)/(1.0-α-ν))*(ν/a)^(ν/(1.0-α-ν))*maximum(zgrid)^(1.0/(1.0-α-ν))
kgrid = [kmax*(1-δ)^(nk-i-1) for i∈0:(nk-1)]