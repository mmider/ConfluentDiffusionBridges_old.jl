SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "wiener.jl"))
include(joinpath(SRC_DIR, "euler.jl"))
include(joinpath(SRC_DIR, "simple_diffusion_bridges.jl"))
include(joinpath(SRC_DIR, "ornstein_uhlenbeck.jl"))

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
x0, xT, T = -2.0, -2.0, 1.0
tt = 0.0:0.01:T # time grid
P = OrnsteinUhlenbeck(0.5, 0.0, 1.0)

# Run simple diffusion bridges
XX, mid_pts_SDB = mcmc(P, x0, xT, tt, 10^6)
XX, mid_pts_SDB_mod = mcmc(P, x0, xT, tt, 10^6)

# Let's compare to the true distribution of mid-points
# Let's compare the empirical distributions
p = histogram(mid_pts_SDB, normalize=:pdf, alpha=0.5, label="simple diffusion bridges (original)",legend=:bottomleft)
histogram!(mid_pts_SDB_mod, normalize=:pdf, alpha=0.5, label="simple diffusion bridges (modified)")
xaxis = -4.0:0.01:0.0
yaxis = [condpdf(P, x0, x, xT, 0.5, 1.0) for x in xaxis]
plot!(xaxis, yaxis, label="exact pdf")
#plot!([2.0, 2.0], [0.0, 0.4])
savefig(p, "sdb_vs_truth_ou.png")




# Let's check if the diffusions are spliced together correctly
WW = SimpleDiffBridge(tt)
rand!(XX, P, Proposal(), x0, xT, WW) # sample proposal
p = plot(XX.fw.tt, XX.fw.yy) # plot forward path
plot!(XX.bw.tt, reverse(XX.bw.yy)) # plot backward path
plot!(XX.prop.tt, XX.prop.yy) # plot proposal path
