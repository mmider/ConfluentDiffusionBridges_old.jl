SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "types.jl"))
include(joinpath(SRC_DIR, "indexing.jl"))
include(joinpath(SRC_DIR, "wiener.jl"))
include(joinpath(SRC_DIR, "langevin_t_distr_diffusion.jl"))
include(joinpath(SRC_DIR, "simple_fpt_coin.jl"))
include(joinpath(SRC_DIR, "path_space_rejection_sampler.jl"))
include(joinpath(SRC_DIR, "confluent_diffusion_bridges.jl"))
include(joinpath(SRC_DIR, "fill_BB.jl"))

using Plots

# Define diffusion to sample
x₀, xₜ, T = 2.0, 3.3, 4.0
λ = 0.0 # artificial inflator of number of poisson points (for debugging only)
P = LangevinT(3.0, λ)

# Confluent diffusion bridges
XX = ConfluentDiffBridge(16.0, 4)
Random.seed!(4)
rand!(XX, P, Proposal(), x₀, xₜ)

function plotMe(i, add=true)
    fw = XX.fwc[i]
    κ = fw.κ[1]
    if add
        p = scatter!(fw.tt[1:κ+2], fw.yy[1:κ+2], color="steelblue", label="", alpha=0.5)
    else
        p = scatter(fw.tt[1:κ+2], fw.yy[1:κ+2], color="steelblue", label="", alpha=0.5)
    end
    p
end
function plotMeBw(i, T)
    bw = XX.bwc[i]
    κ = bw.κ[1]
    p = scatter!(bw.tt[1:κ+2], bw.yy[1:κ+2], color="red", label="", alpha=0.5)
    p
end



p = plotMe(1, false)
plotMe(2)
plotMe(3)
plotMe(4)

plotMeBw(1, 16.0)
plotMeBw(2, 16.0)
plotMeBw(3, 16.0)
plotMeBw(4, 16.0)

scatter!([XX.τ[1][3]],[XX.τ[1][4]])
show(p)


# Let's test auxiliary sampler now
Random.seed!(4)
samples = zeros(Int64, 10000)
for i in 1:10000
    samples[i] = rand!(XX, P, Auxiliary())
end
samples[samples.≥5]

function plotMeᵒ(i, add=true)
    fw = XX.fwcᵒ[i]
    κ = fw.κ[1]
    if add
        p = scatter!(fw.tt[1:κ+2], fw.yy[1:κ+2], color="steelblue", label="", alpha=0.5)
    else
        p = scatter(fw.tt[1:κ+2], fw.yy[1:κ+2], color="steelblue", label="", alpha=0.5)
    end
    p
end
function plotMeBwᵒ(i)
    bw = XX.bwcᵒ[i]
    κ = bw.κ[1]
    p = scatter!(bw.tt[1:κ+2], bw.yy[1:κ+2], color="red", label="", alpha=0.5)
    p
end
function plotMeAuxᵒ(i)
    aux = XX.auxᵒ[i]
    κ = aux.κ[1]
    p = scatter!(aux.tt[1:κ+2], aux.yy[1:κ+2], color="violet", label="", alpha=0.5)
    p
end
XX.fwcᵒ[1]

p = plotMeᵒ(1, false)
plotMeᵒ(2)
plotMeᵒ(3)
plotMeᵒ(4)

plotMeBwᵒ(1)
plotMeBwᵒ(2)
plotMeBwᵒ(3)
plotMeBwᵒ(4)
scatter!([XX.τ[1][3]],[XX.τ[1][4]])

plotMeAuxᵒ(1)
plotMeAuxᵒ(2)
plotMeAuxᵒ(3)
plotMeAuxᵒ(4)
