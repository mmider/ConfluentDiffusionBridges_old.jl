using AdaptiveRejectionSampling
using QuadGK
using Random
import Random.rand

# HARD CODED EXAMPLE

"""
    LangevinDoubleWell <: ContinuousTimeProcess{Float64}

Struct defining a Lengevin diffusion with double-well like behaviour
"""
struct LangevinDoubleWell <: ContinuousTimeProcess{Float64}
    v₁::Float64  # Only this is needed for simple diffusion bridges
    v₂::Float64
    μ₁::Float64
    μ₂::Float64
    n::Float64
    lowBd::Float64
    upBd::Float64

    function LangevinDoubleWell(v₁, μ₁, v₂, μ₂, n, λ=0.0)
        @assert λ ≥ 0.0

        lowBd = n*langevinT_lowBd(v₁) + n*langevinT_lowBd(v₂)
        upBd = n*langevinT_upBd(v₁) + n*langevinT_upBd(v₂) - lowBd + λ
        new(v₁, v₂, μ₁, μ₂, n, lowBd, upBd)
    end
end

langevinT_lowBd(v) = -0.25*(v+1.0)/v
langevinT_upBd(v) = 0.03125*(v+1.0)*(v+3.0)^2/(v^2+5.0*v)

langevinT_∇logp(v, μ, y) = -0.5*(v+1.0)*(y-μ)/(v + (y-μ)^2.0)
langevinT_ϕ(v, μ, y) = (v+1.0)*(-2.0*v+(y-μ)^2*(v+3.0))/(8.0*(v+(y-μ)^2)^2)
langevinT_A(v, μ, y) = -0.25*(v+1.0)*log(1.0 + (y-μ)^2/v)

"""
    drift(t, y::Float64, P::LangevinT)

Evaluate the drift function at (`t`,`y`)
"""
function drift(t, y::Float64, P::LangevinDoubleWell)
    P.n*langevinT_∇logp(P.v₁, P.μ₁, y) + P.n*langevinT_∇logp(P.v₂, P.μ₂, y)
end

"""
    vola(t, y::Float64, P::LangevinT)

Evaluate the volatility function at (`t`, `y`)
"""
vola(t, y::Float64, P::LangevinDoubleWell) = 1.0

"""
    ϕ(t::Float64, y::Float64, P::LangevinT)

Evaluate the phi function inside the exponent of a Radon-Niodym derivative at
(`t`, `y`)
"""
function ϕ(t::Float64, y::Float64, P::LangevinDoubleWell)
    P.n*langevinT_ϕ(P.v₁, P.μ₁, y) + P.n*langevinT_ϕ(P.v₂, P.μ₂, y) - P.lowBd
end

"""
    A(t, y::Float64, P::LangevinT)

Evaluate the anti-derivative of a drift function at (`t`, `y`)
"""
function A(t, y, P::LangevinDoubleWell)
    P.n*langevinT_A(P.v₁, P.μ₁, y) + P.n*langevinT_A(P.v₂, P.μ₂, y)
end

"""
    A(y::Float64, P::LangevinT)

Evaluate the anti-derivative of a drift function at `y`
"""
A(y, P::LangevinDoubleWell) = A(nothing,y,P)


"""
    rand(P::LangevinT, ::Invariant)

Sample from an invariant density of the diffusion `P`
"""
rand(P::LangevinDoubleWell, ::Invariant) = rand(TDist(P.v))


"""
    rand!(P::LangevinT, ::EndPoint, x₀, T)

Sample the proposal end-point at time `T` from a biased Bridge distribution for
a path that starts from `x₀`
"""
function rand!(P::LangevinDoubleWell, ::EndPoint, x₀, T)
    _f(x) = exp(A(x, P) - 0.5*(x-x₀)^2)
    𝓩, _ = quadgk(_f, -Inf, Inf)
    sampler = RejectionSampler(x->_f(x)/𝓩, (-Inf, Inf))
    sim = run_sampler!(sampler, 1)
    sim[1]
end

P = LangevinDoubleWell(3.0, 3.0, 3.0, -3.0, 3, 0.0)
N = 10000
samples = zeros(Float64, N)
for i in 1:N
    samples[i] = rand!(P, EndPoint(), 0.0, 10.0)
end

histogram(samples, normalize=:pdf)

samples
