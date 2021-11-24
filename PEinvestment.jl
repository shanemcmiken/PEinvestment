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

#--------------------------------#
#         PARAMETERIZATION
#--------------------------------#
α   = 0.7
p   = 1.2
δ   = 0.1
r   = 0.04
ρ   = 0.8
σ   = 0.2
γ_c = 0.0
γ_f = 0.0

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
nz    = 5*cover+1
m     = 2.5
zgrid, P = MyTauchen(σ,ρ,nz,m)
zgrid = exp.(zgrid) # exponentiate grid

# capital grid
nk = 100
kmax = ((maximum(zgrid)*α)/(r+δ))^(1.0/(1.0-α))
kgrid = [kmax*(1-δ)^i for i∈0:(nk-1)]



