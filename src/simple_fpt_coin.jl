using Random, Distributions
import Random.rand


"""
    rand(::Dcoin, d0, dT, T)

Independent draw of a d-coin, which has a probability of success equal to the
probability of a Brownian bridge (scaled by √2) joining `d0` and `dT` over the
interval [0,`T`] to cross 0.
"""
function rand(::Dcoin, d0, dT, T)
    (sign(d0) != sign(dT)) && return true
    rand(Exponential()) ≥ abs(d0*dT)/T
end

"""
    rand(::τᴰ, d0, dT, T)

Draw from the distribution of the first-passage time to 0 of a Brownian bridge
(scaled by √2) joining `d0` and `dT` over the interval [0,`T`] conditioned on
hitting 0 during [0,`T`].
"""
function rand(::τᴰ, d0, dT, T)
    μ = abs(d0/dT)
    λ = 0.5*d0^2/T
    K = rand(InverseGaussian(μ, λ))
    T/(1.0 + 1.0/K)
end
