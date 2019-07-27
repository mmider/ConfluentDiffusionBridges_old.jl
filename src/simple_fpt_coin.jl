using Random, Distributions
import Random.rand


function rand(::Dcoin, d0, dT, T)
    (sign(d0) != sign(dT)) && return true
    rand(Exponential()) ≥ abs(d0*dT)/T
end

function rand(::τᴰ, d0, dT, T)
    μ = abs(d0/dT)
    λ = 0.5*d0^2/T
    K = rand(InverseGaussian(μ, λ))
    T/(1.0 + 1.0/K)
end
