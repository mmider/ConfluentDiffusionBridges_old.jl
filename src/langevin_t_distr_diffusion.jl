using Distributions, Random
import Random.rand


"""
    LangevinT <: ContinuousTimeProcess{Float64}

Struct defining a Lengevin diffusion which has t-distribution as its invariant
measure
"""
struct LangevinT <: ContinuousTimeProcess{Float64}
    v::Float64  # Only this is needed for simple diffusion bridges
    # ↓↓↓ the fields below are needed for confluent diffusion bridges ↓↓↓
    lowBd::Float64
    upBd::Float64

    lvls::NTuple{11,Float64}
    C::NTuple{10,Float64}
    D::NTuple{10,Float64}

    ζ::Vector{Truncated{Normal{Float64},Continuous}}
    ω::Vector{Float64}
    N::Int64

    function LangevinT(v)
        lowBd = -0.25*(v+1.0)/v
        upBd = 0.03125*(v+1.0)*(v+3.0)^2/(v^2+5.0*v) - lowBd

        d₁ = √(3.0*v)
        d₂ = (v+1.0)/√v
        d₃ = (v+1)/v

        lvls = (-Inf, -4.0*d₁, -2.0*d₁, -d₁, -0.5*√v, 0.0, √v, d₁, 2.0*d₁,
                3.0*d₁, Inf)

        # a bunch of nameless constants
        C = (2.0*√3.0/49.0*d₂, 72.0*√3.0/637.0*d₂, 9.0*√3.0/52.0*d₂, 0.25*d₂,
             0.0, 0.0, -0.125*√3.0/(√3.0-1.0)*d₂, -0.0625*3.0*√3.0*d₂,
             -24.0*√3.0/169.0*d₂, 0.0)
        D = (0.0, 23.0/2548.0*d₃, 5.0/208.0*d₃, 0.0, -0.25*d₃, -0.125*d₃,
             0.0625*d₃*(2.0-√3.0)/(√3.0-1.0), 0.03125*d₃, 11.0/676.0*d₃, 0.0)

        N = length(lvls)-1
        ζ = Vector{Truncated{Normal{Float64},Continuous}}(undef, N)
        ω = zeros(Float64, N)

        new(v, lowBd, upBd, lvls, C, D, ζ, ω, N)
    end
end

"""
    drift(t, y::Float64, P::LangevinT)

Evaluate the drift function at (`t`,`y`)
"""
drift(t, y::Float64, P::LangevinT) = -0.5*(P.v+1.0)*y/(P.v + y^2.0)
"""
    vola(t, y::Float64, P::LangevinT)

Evaluate the volatility function at (`t`, `y`)
"""
vola(t, y::Float64, P::LangevinT) = 1.0

"""
    ϕ(t::Float64, y::Float64, P::LangevinT)

Evaluate the phi function inside the exponent of a Radon-Niodym derivative at
(`t`, `y`)
"""
function ϕ(t::Float64, y::Float64, P::LangevinT)
    (P.v+1.0)*(-2.0*P.v+y^2*(P.v+3.0))/(8.0*(P.v+y^2)^2) - P.lowBd
end

"""
    A(t, y::Float64, P::LangevinT)

Evaluate the anti-derivative of a drift function at (`t`, `y`)
"""
A(t, y::Float64, P::LangevinT) = -0.25*(P.v+1.0)*log(1.0 + y^2/P.v)

"""
    A(y::Float64, P::LangevinT)

Evaluate the anti-derivative of a drift function at `y`
"""
A(y::Float64, P::LangevinT) = A(nothing,y,P)

"""
    _B(y::Float64, idx::Integer, P::LangevinT)

Convenience function needed for the end-point sampler
"""
_B(y::Float64, idx::Integer, P::LangevinT) = P.D[idx]*y^2 +P.C[idx]*y


"""
    rand(P::LangevinT, ::Invariant)

Sample from an invariant density of the diffusion `P`
"""
rand(P::LangevinT, ::Invariant) = rand(TDist(P.v))


"""
    rand!(P::LangevinT, ::EndPoint, x₀, T)

Sample the proposal end-point at time `T` from a biased Bridge distribution for
a path that starts from `x₀`
"""
function rand!(P::LangevinT, ::EndPoint, x₀, T)
    updateConstants!(P, x₀, T)
    while true
        lvlChoice = rand(Categorical(P.ω/sum(P.ω)))
        proposal = rand(P.ζ[lvlChoice])
        E = rand(Exponential())
        if lvlChoice == 1
            ( E ≥ _B(proposal, lvlChoice, P)-A(proposal, P) ) && return proposal
        elseif E ≥ ( _B(proposal, lvlChoice, P)
                    -_B(P.lvls[lvlChoice], lvlChoice, P)
                    + A(P.lvls[lvlChoice], P)
                    - A(proposal, P) )
            return proposal
        end
    end
end

"""
    updateConstants!(P::LangevinT, x₀, T)

Convenience function for sampling the end-points. Sets all the necessary
constants and partial, dominating samplers for a composite rejection sampler
"""
function updateConstants!(P::LangevinT, x₀, T)
    for i in 1:P.N
        μ = (x₀ + P.C[i]*T)/(1.0 - 2.0*T*P.D[i])
        σ = √( T/(1.0 - 2.0*T*P.D[i]) )
        P.ζ[i] = Truncated(Normal(μ, σ), P.lvls[i], P.lvls[i+1])

        d₀ = 0.5*(x₀ + P.C[i]*T)*μ/T
        # P.ζ[i].tp is the probability of falling inside truncated region
        P.ω[i] = σ * exp(d₀-0.5*x₀^2/T) * P.ζ[i].tp
        if i != 1
            P.ω[i] *= exp(-_B(P.lvls[i], i, P) + A(P.lvls[i], P))
        end
    end
end
