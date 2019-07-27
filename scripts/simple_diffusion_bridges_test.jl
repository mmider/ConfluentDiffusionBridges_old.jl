SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "wiener.jl"))
include(joinpath(SRC_DIR, "euler.jl"))
include(joinpath(SRC_DIR, "simple_diffusion_bridges.jl"))
include(joinpath(SRC_DIR, "langevin_t_distr_diffusion.jl"))


function mcmc(P, x0, xT, tt, numMCMCsteps=1000)
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
    print("Acceptance rate: ", numAccepted/numMCMCsteps)
    XX, midis
end

tt = 0.0:0.01:4.0
x0 = 2.0
xT = 3.3
P = LangevinT(3.0)
start = time()
XX, midis3 = mcmc(P, x0, xT, tt, Integer(1e6))
elapsed = time() - start
print("Time elapsed: ", elapsed)

using Plots
WW = SimpleDiffBridge(tt)
rand!(XX, P, Proposal(), x0, xT, WW)
p = plot(XX.fw.tt, XX.fw.yy)
plot!(XX.bw.tt, reverse(XX.bw.yy))
plot!(XX.prop.tt, XX.prop.yy)
plot!(XX.aux.tt, XX.aux.yy)
idx = 49
plot!([XX.fw.tt[idx], XX.fw.tt[idx]], [-2,3])

p = histogram(midis3, normalize=:pdf, alpha=0.5, label="simple diffusion bridges")
histogram!(mid_pts, normalize=:pdf, alpha=0.5, label="truth")
plot!([2.0, 2.0], [0.0, 0.4], label="", linewidth=2.0)

savefig(p, "sdb_vs_truth.png")
