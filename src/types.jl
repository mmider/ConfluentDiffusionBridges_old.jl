import Base: length, resize!

abstract type ContinuousTimeProcess{T} end

struct Wiener{T} <: ContinuousTimeProcess{T} end

Wiener() = Wiener{Float64}()

abstract type SDESolver end

struct EulerMaruyama <: SDESolver end

Euler = EulerMaruyama

struct SamplePath{T}
    yy::Vector{T}
    tt::Vector{Float64}

    SamplePath() = new{Float64}(Vector{Float64}(), Vector{Float64}())
    SamplePath(tt) = new{Float64}(zeros(Float64, length(tt)), collect(tt))
    SamplePath(tt, ::T) where T = new{T}(zeros(T, length(tt)), collect(tt))
end

length(X::SamplePath) = length(X.yy)

function resize!(X::SamplePath, n)
    resize!(X.yy, n)
    resize!(X.tt, n)
end


abstract type UpdateType end

struct Proposal <: UpdateType end

struct Auxiliary <: UpdateType end

abstract type Coin end

struct Dcoin <: Coin end # this is 1-pD coin defined in the paper

abstract type FirstPassageTime end

struct τᴰ <: FirstPassageTime end

struct Invariant end

struct EndPoint end
