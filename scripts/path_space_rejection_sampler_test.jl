SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "wiener.jl"))
include(joinpath(SRC_DIR, "euler.jl"))
include(joinpath(SRC_DIR, "langevin_t_distr_diffusion.jl"))
include(joinpath(SRC_DIR, "path_space_rejection_sampler.jl"))
include(joinpath(SRC_DIR, "fill_BB.jl"))
include(joinpath(AUX_DIR, "path_space_rejection_sampling_convenience_fns.jl"))

using Plots

# Define diffusion to sample
x₀, xₜ, T = 2.0, 3.3, 4.0
P = LangevinT(3.0)

# Let's visualise some paths
# --------------------------
tt = 0.0:0.01:T # set the time-grid
pathSamples, _ = samplePathsExactly(x₀, xₜ, T, P, 100) # sample 100 paths
plotPaths(tt, pathSamples) # plot them

# Let's check if the sampler works correctly by sampling unconditioned paths
# using PSRS and Euler-Maruyama and compare the distribution of the end-points
# Let's also do the same with PSRS with additional segmentation of sampling
# based on the Markov property
#-----------------------------------------------------------------------------
# First, define routine for sampling paths using Euler scheme (only end points
# are returned)
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
# Sampler using PSRS is already defined in convenience functions, but we still
# need a routine for using PSRS with segmentation of [0,T]
function samplePathsInSegments(x0, T, numSegments, numPaths=100)
    dt = T/numSegments
    XX = [PathSegment((i-1)*dt, dt) for i in 1:numSegments]
    endPts = zeros(Float64, numPaths)
    for i in 1:numPaths
        endPts[i] = rand!(XX, P, x0)
    end
    endPts
end
# OK, now we can sample the end-points. Let's sample 1mil paths for each method
pathSamples, endPts = samplePathsExactly(x₀, nothing, T, P, 10^6)
endPtsEuler = sampleEulerUncond(x₀, 0.0:0.01:T, 10^6)
endPtsSeg = samplePathsInSegments(x₀, T, 4, 10^6)
# and let's see how the distributions of the end-points compare
histogram(endPts, normalize=:pdf, alpha=0.5, label="Path space rejection sampler")
histogram!(endPtsEuler, normalize=:pdf, alpha=0.5, label="euler")
histogram!(endPtsSeg, normalize=:pdf, alpha=0.5, label="psrs, segments")
# Great!
