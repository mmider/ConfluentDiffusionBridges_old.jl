using Distributions
using Random
import Random: rand!, rand

struct PathSegment
    yy::Vector{Float64}
    ppp::Vector{Float64}
    tt::Vector{Float64}
    dWt::Vector{Float64}
    t₀::Float64
    T::Float64
    κ::Vector{Int64}
    maxN::Vector{Int64}

    function PathSegment(t₀::Float64, T::Float64)
        N = 10
        yy = zeros(Float64, N)
        ppp = zeros(Float64, N)
        tt = zeros(Float64, N)
        dWt = zeros(Float64, N)
        κ = zeros(Int64, 1)
        κ[1] = 0
        maxN = zeros(Int64, 1)
        maxN[1] = N
        new(yy, ppp, tt, dWt, t₀, T, κ, maxN)
    end
end

function resize!(XX::PathSegment, N)
    if XX.maxN[1] < N
        resize!(XX.yy, N)
        resize!(XX.ppp, N)
        resize!(XX.tt, N)
        resize!(XX.dWt, N)
        XX.maxN[1] = N
    end
end


function rand!(XX::PathSegment, P::ContinuousTimeProcess, x₀=0.0; xₜ=nothing)
    accepted = false
    xT = 0.0
    while !accepted
        xT = sampleEndPt(P, x₀, XX.T, xₜ)
        κ = samplePPP!(XX, P.upBd)
        sampleBB!(x₀, xT, XX)
        accepted = true
        for i in 1:κ
            if ϕ(XX.tt[1+i], XX.yy[1+i], P) > XX.ppp[i]
                accepted = false
                break
            end
        end
    end
    xT
end

function rand!(XX::Vector{PathSegment}, P::ContinuousTimeProcess, x₀=0.0)
    N = length(XX)
    for i in 1:N
        x₀ = rand!(XX[i], P, x₀)
    end
    x₀
end


sampleEndPt(P, x0, T, xT::Float64) = xT
sampleEndPt(P, x0, T, xT::Nothing) = rand!(P, EndPoint(), x0, T)

function samplePPP!(XX::PathSegment, upBd::Float64)
    κ = rand(Poisson(XX.T*upBd))
    resize!(XX, κ+2) # +2 for start point and end point
    XX.tt[1] = XX.t₀
    T = XX.t₀ + XX.T
    rand!(Uniform(XX.t₀, T), view(XX.tt, 2:κ+1))
    XX.tt[κ+2] = T
    sort!(view(XX.tt, 2:κ+1))
    rand!(Uniform(0.0, upBd), view(XX.ppp, 1:κ))
    XX.κ[1] = κ
    κ
end

function sampleBB!(x₀::Float64, xT::Float64, XX::PathSegment)
    κ = XX.κ[1]
    XX.yy[1] = x₀
    if κ > 0
        dWt = view(XX.dWt, 1:κ+1)
        tt = view(XX.tt, 1:κ+2)
        xx = view(XX.yy, 2:κ+2)

        rand!(Normal(), dWt)
        dWt .*= sqrt.(diff(tt))
        xx .= cumsum(dWt)
        xx .+= ( ((xT-xx[end])/XX.T) .* (tt[2:end].-XX.t₀)
                 .+ (x₀/XX.T) .* ((XX.t₀+XX.T) .- tt[2:end]) )
    end
    XX.yy[κ+2] = xT
end
