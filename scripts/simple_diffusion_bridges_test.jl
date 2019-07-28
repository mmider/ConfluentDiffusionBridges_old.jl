SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "wiener.jl"))
include(joinpath(SRC_DIR, "euler.jl"))
include(joinpath(SRC_DIR, "simple_diffusion_bridges.jl"))
include(joinpath(SRC_DIR, "langevin_t_distr_diffusion.jl"))

using Plots


# pseudo-marginal mcmc sampler using simple diffusion bridges. Return chain
# of mid-points
function mcmc(P, x0, xT, tt, numMCMCsteps=1000)
    start = time()
    # workspace
    XX = SimpleDiffBridge(tt)
    XXᵒ = SimpleDiffBridge(tt)
    WWᵒ = SimpleDiffBridge(tt)
    midi = Integer((length(tt)-1)/2)+1

    # initialise
    rand!(XX, P, Proposal(), x0, xT, WWᵒ)
    𝓣 = rand!(XX, P, Auxiliary(), WWᵒ)

    numAccepted = 0
    midis = zeros(Float64, numMCMCsteps)
    for i in 1:numMCMCsteps
        rand!(XXᵒ, P, Proposal(), x0, xT, WWᵒ)
        𝓣ᵒ = rand!(XXᵒ, P, Auxiliary(), WWᵒ)
        if rand() ≤ 𝓣ᵒ/𝓣
            XXᵒ, XX = XX, XXᵒ
            𝓣ᵒ, 𝓣 = 𝓣, 𝓣ᵒ
            numAccepted += 1
        end
        midis[i] = XX.prop.yy[midi]
    end
    elapsed = time() - start
    print("Time elapsed: ", elapsed, "\n")
    print("Acceptance rate: ", numAccepted/numMCMCsteps, "\n")
    XX, midis
end


# Define diffusion to sample
x0, xT, T = 2.0, 3.3, 4.0
tt = 0.0:0.01:T # time grid
P = LangevinT(3.0)

# Run simple diffusion bridges
XX, mid_pts_SDB = mcmc(P, x0, xT, tt, 10^6)

# Let's compare to the true distribution of mid-points
include(joinpath(SRC_DIR, "path_space_rejection_sampler.jl"))
include(joinpath(SRC_DIR, "fill_BB.jl"))
include(joinpath(AUX_DIR, "path_space_rejection_sampling_convenience_fns.jl"))
# sample mid points using Rejection sampling on a path space
pathSamples, _ = samplePathsExactly(x0, xT, T, P, 10^6)
mid_pts_exact = extractMidPts(x0, xT, T, pathSamples)
# Let's compare the empirical distributions
p = histogram(mid_pts_SDB, normalize=:pdf, alpha=0.5, label="simple diffusion bridges")
histogram!(mid_pts_exact, normalize=:pdf, alpha=0.5, label="path space rejection sampler")
plot!([2.0, 2.0], [0.0, 0.4])
savefig(p, "sdb_vs_truth.png")




# Let's check if the diffusions are spliced together correctly
WW = SimpleDiffBridge(tt)
rand!(XX, P, Proposal(), x0, xT, WW) # sample proposal
p = plot(XX.fw.tt, XX.fw.yy) # plot forward path
plot!(XX.bw.tt, reverse(XX.bw.yy)) # plot backward path
plot!(XX.prop.tt, XX.prop.yy) # plot proposal path
