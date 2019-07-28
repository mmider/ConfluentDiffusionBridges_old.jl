import Base: length, resize!

"""
    ContinuousTimeProcess{T}

Types inheriting from this abstract type are continuous time processes
"""
abstract type ContinuousTimeProcess{T} end

"""
    Wiener{T}

Indicator for Wiener process
"""
struct Wiener{T} <: ContinuousTimeProcess{T} end

"""
    Wiener()

By default define one-dimensional Wiener process with Float64 data type
"""
Wiener() = Wiener{Float64}()

"""
    SDESolver

Types inheriting from this abstract type are discrete SDE solvers
"""
abstract type SDESolver end

"""
    EulerMaruyama <: SDESolver

Euler-Maruyama solver for SDEs
"""
struct EulerMaruyama <: SDESolver end

# define alias
Euler = EulerMaruyama

"""
    SamplePath{T}

Struct for sampling diffusion paths on discrete time grids
"""
struct SamplePath{T}
    yy::Vector{T}
    tt::Vector{Float64}

    SamplePath() = new{Float64}(Vector{Float64}(), Vector{Float64}())
    SamplePath(tt) = new{Float64}(zeros(Float64, length(tt)), collect(tt))
    SamplePath(tt, ::T) where T = new{T}(zeros(T, length(tt)), collect(tt))
end

"""
    length(X::SamplePath)

Return the length of the time grid
"""
length(X::SamplePath) = length(X.yy)

"""
    resize!(X::SamplePath, n)

Resize the internal containers of the SamplePath object
"""
function resize!(X::SamplePath, n)
    resize!(X.yy, n)
    resize!(X.tt, n)
end

"""
    UpdateType

Types inheriting from the abstract type `UpdateType` discriminate between
calls to rand! or rand
"""
abstract type UpdateType end

"""
    Proposal <: UpdateType

Flag indicating update of proposals
"""
struct Proposal <: UpdateType end

"""
    Auxiliary <: UpdateType

Flag indicating update of auxiliary variables
"""
struct Auxiliary <: UpdateType end

"""
    Coin

Types inheriting from the abstract type `Coin` discriminate between samplers
of coins (Bernoulli random variables)
"""
abstract type Coin end

"""
    Dcoin <: Coin

Flag indicating p_D coin---probability that a scaled Brownian bridge crosses 0
"""
struct Dcoin <: Coin end # this is 1-pD coin defined in the paper

"""
    FirstPassageTime

Types inheriting form `FirstPassageTime` discriminate between samplers of
first passage time events.
"""
abstract type FirstPassageTime end

"""
    τᴰ <: FirstPassageTime

Indicator for first passage time sampler of a conditioned, scaled Brownian
bridge to 0
"""
struct τᴰ <: FirstPassageTime end

"""
    Invariant

Flag used for discrimination of call to rand! or rand
"""
struct Invariant end

"""
    Invariant

Flag used for discrimination of call to rand! or rand
"""
struct EndPoint end
