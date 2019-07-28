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
