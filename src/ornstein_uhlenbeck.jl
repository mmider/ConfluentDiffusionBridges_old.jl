using Random
import Random.rand
import Distributions: cov, mean


struct OrnsteinUhlenbeck <: ContinuousTimeProcess{Float64}
    θ::Float64
    μ::Float64
    σ::Float64
end

drift(t, y::Float64, P::OrnsteinUhlenbeck) = P.θ*(P.μ-y)
vola(t, y::Float64, P::OrnsteinUhlenbeck) = P.σ

rand(P::OrnsteinUhlenbeck, ::Invariant) = rand(Normal(P.μ, P.σ/√(2.0*P.θ)))


function cov(P::OrnsteinUhlenbeck, t, T)
    P.σ^2/(2.0*P.θ)*( exp(-P.θ*(T-t)) - exp(-P.θ*(T+t)) )
end

mean(P::OrnsteinUhlenbeck, t, x₀) = exp(-P.θ*t)*x₀

function condpdf(P::OrnsteinUhlenbeck, x0, xt, xT, t, T)
    μ = mean(P, t, x0) + cov(P, t, T)/cov(P, T, T)*(xT - mean(P, T, x0))
    σ² = cov(P, t, t) - cov(P, t, T)^2/cov(P, T, T)
    pdf(Normal(μ, √σ²), xt)
end
