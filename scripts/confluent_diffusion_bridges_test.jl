SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "wiener.jl"))
include(joinpath(SRC_DIR, "langevin_t_distr_diffusion.jl"))
include(joinpath(SRC_DIR, "simple_fpt_coin.jl"))
include(joinpath(SRC_DIR, "path_space_rejection_sampler.jl"))
include(joinpath(SRC_DIR, "confluent_diffusion_bridges.jl"))
include(joinpath(SRC_DIR, "fill_BB.jl"))

using Plots

# Define diffusion to sample
x₀, xₜ, T = 2.0, 3.3, 4.0
P = LangevinT(3.0)


# sample paths using Rejection sampling on a path space
function samplePathsExactly(x0, xT, T, numPaths=100)
    XX = PathSegment(0.0, T)
    pathSamples = Vector{Any}(undef, numPaths)
    timeElapsed = zeros(Float64, numPaths)
    endPts = zeros(Float64, numPaths)
    for i in 1:numPaths
        start = time()
        endPts[i] = rand!(XX, P, x0; xₜ=xT)
        timeElapsed[i] = time() - start
        iRange = 2:XX.κ[1]+1
        pathSamples[i] = (copy(XX.tt[iRange]), copy(XX.yy[iRange]))
    end
    print("Average time to simulate a single path: ", mean(timeElapsed[4:end]))
    pathSamples, endPts
end

function plotPaths(tt, pathSamples, alpha=0.2)
    @assert tt[end] == T
    xx = fillBB(tt, x₀, xₜ, pathSamples[1]...)
    p = plot(tt, xx, alpha=alpha, color="steelblue", label="")
    numPaths = length(pathSamples)
    for i in 2:numPaths
        xx = fillBB(tt, x₀, xₜ, pathSamples[i]...)
        plot!(tt, xx, alpha=alpha, color="steelblue", label="")
    end
    p
end


tt = 0.0:0.01:T
pathSamples, _ = samplePathsExactly(x₀, xₜ, T, 100)
plotPaths(tt, pathSamples)


function extractMidPts(pathSamples)
    N = length(pathSamples)
    midPts = zeros(Float64, N)
    tt = [0.0, 0.5*T, T]
    for i in 1:N
        xx = fillBB(tt, x₀, xₜ, pathSamples[i]...)
        midPts[i] = xx[2]
    end
    midPts
end


# sample mid points using Rejection sampling on a path space
pathSamples, _ = samplePathsExactly(x₀, xₜ, T, 10^6)
midPtsExact = extractMidPts(pathSamples)
histogram(midPtsExact, normalize=:pdf)
plot!([2.0, 2.0], [0.0, 0.4])


include(joinpath(SRC_DIR, "euler.jl"))
function sampleEulerUncond(x0, tt, numPaths)
    WW = SamplePath(tt)
    XX = SamplePath(tt)

    endPts = zeros(Float64, numPaths)
    for i in 1:numPaths
        rand!(Wiener(), WW)
        solve!(Euler(), P, XX, WW, x0)
        endPts[i] = XX.yy[end]
    end
    endPts
end

# sample paths using Rejection sampling on a path space
function samplePathsInSegments(x0, T, numSegments, numPaths=100)
    dt = T/numSegments
    XX = [PathSegment((i-1)*dt, dt) for i in 1:numSegments]
    endPts = zeros(Float64, numPaths)
    for i in 1:numPaths
        endPts[i] = rand!(XX, P, x0)
    end
    endPts
end

pathSamples, endPts = samplePathsExactly(x₀, nothing, T, 10^6)
endPtsEuler = sampleEulerUncond(x₀, 0.0:0.01:T, 10^6)
endPtsSeg = samplePathsInSegments(x₀, T, 4, 10^6)
histogram(endPts, normalize=:pdf, alpha=0.5, label="Path space rejection sampler")
histogram!(endPtsEuler, normalize=:pdf, alpha=0.5, label="euler")
histogram!(endPtsSeg, normalize=:pdf, alpha=0.5, label="psrs, segments")

# Confluent diffusion bridges
XX = ConfluentDiffBridge(16.0, 4)
rand!(XX, P, Proposal(), x₀, xₜ)


function plotMe(i, add=true)
    fw = XX.fw[i]
    κ = fw.κ[1]
    if add
        p = scatter!(fw.tt[1:κ+2], fw.yy[1:κ+2], color="steelblue", label="", alpha=0.5)
    else
        p = scatter(fw.tt[1:κ+2], fw.yy[1:κ+2], color="steelblue", label="", alpha=0.5)
    end
    p
end
function plotMeBw(i, T)
    bw = XX.bw[i]
    κ = bw.κ[1]
    p = scatter!(T.-bw.tt[1:κ+2], bw.yy[1:κ+2], color="red", label="", alpha=0.5)
    p
end

function plotMeProp(i)
    prop = XX.prop[i]
    κ = prop.κ[1]
    p = scatter!(prop.tt[1:κ+2], prop.yy[1:κ+2], color="orange", label="", alpha=0.7)
    p
end



plotMe(1, false)
plotMe(2)
plotMe(3)
plotMe(4)

plotMeBw(1, 16.0)
plotMeBw(2, 16.0)
plotMeBw(3, 16.0)
plotMeBw(4, 16.0)

plotMeProp(1)
plotMeProp(2)
plotMeProp(3)
plotMeProp(4)
