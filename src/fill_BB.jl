using Distributions
using Random

"""
    fillBB(tt, x₀, xₜ, rtt, rxx)

Sub-sample path on a time grid `tt`. It is assumed that the sample has been
sampled using rejection sampling on a path space
...
# Arguments
- tt: time grid at which to sample the path
- x₀: starting point of the path
- xₜ: end-point of the path
- rtt: random time grid at which the path has already been revealed by PSRS
- rxx: values of the path taken at a random time grid `rtt`
...
"""
function fillBB(tt, x₀, xₜ, rtt, rxx)
    N = length(tt)
    xx = zeros(Float64, N)
    xx[1] = x₀
    xx[end] = xₜ

    idx = 1
    idx_prev = 1

    rN = length(rtt)
    if rN > 0
        for i in 1:rN
            x0 = (i == 1 ? x₀ : rxx[i-1])
            t0 = (i == 1 ? tt[1] : rtt[i-1])
            while idx < N && tt[idx] <= rtt[i]
                idx += 1
            end
            xx[idx_prev:idx-1] = _sampleBB(x0, rxx[i], t0, rtt[i], tt[idx_prev:idx-1])
            idx_prev = idx
        end
        xx[idx_prev:end] = _sampleBB(rxx[end], xₜ, rtt[end], tt[end], tt[idx_prev:end])
    else
        xx[2:end-1] = _sampleBB(xx[1], xx[end], tt[1], tt[end], tt[2:end-1])
    end
    xx
end

"""
    _sampleBB(x0::Float64, xT::Float64, t0::Float64, T::Float64, tt)

Convenience function for sampling Brownian Bridges joining `x0` and `xT` on
[`t0`,`T`]. Path is revealed on at a time grid `tt`.
"""
function _sampleBB(x0::Float64, xT::Float64, t0::Float64, T::Float64, tt)
    if length(tt)>0
        N = length(tt)
        noise = rand(Normal(), N+1)
        noise[2:end-1] .*= sqrt.(diff(tt))
        noise[1] *= sqrt(tt[1]-t0)
        noise[end] *= sqrt(T-tt[end])
        BM = cumsum(noise)
        BM[1:end-1] .-= (BM[end]/(T-t0)) .* (tt .- t0)
        BM[end] = 0
        xx = BM[1:end-1] .+ (xT/(T-t0)) .* (tt .- t0) .+ (x0/(T-t0)) .* (T .- tt)
    else
        xx = zeros(Float64,0)
    end
    xx
end
