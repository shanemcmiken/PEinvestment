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

#--------------------------------#
#  GRIDS and TRANSITION MATRIX
#--------------------------------#

# Tauchen (1986)
function MyTauchen(σ_a::Float64,ρ_a::Float64,na::Int64,m::Float64)

  vecsize = na
  σ_y = sqrt(σ_a^2 / (1-(ρ_a^2))) # Stationary distribution variance
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

  agrid = exp.(agrid) # Exponentiate grid

  return agrid, P #return productivity grid and Markov transition matrix
end

# technology grid
cover = 5
nz    = 2*cover+1
σ     = 1.2
ρ     = 0.2
m     = 2.5

zgrid, P = MyTauchen(σ,ρ,na,m)
