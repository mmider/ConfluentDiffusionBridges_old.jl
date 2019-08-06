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
include(joinpath(SRC_DIR, "coin_container.jl"))
include(joinpath(SRC_DIR, "confluent_diffusion_bridges.jl"))
include(joinpath(SRC_DIR, "aux_fpt_coins.jl"))
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
function plotMeBw(i)
    bw = XX.bwc[i]
    κ = bw.κ[1]
    p = scatter!(bw.tt[1:κ+2], bw.yy[1:κ+2], color="red", label="", alpha=0.5)
    p
end

plot!([1.664771820124903, 1.664771820124903], [-0.5, 3.0])
plot!([2.382836705251509, 2.382836705251509], [-0.5, 3.0])
print(XX.fwc[1].tt[1:XX.fwc[1].κ[1]+2])
print(XX.fwc[1].yy[1:XX.fwc[1].κ[1]+2])
print(XX.fw[1].tt[1:XX.fw[1].κ[1]+2])
print(XX.fw[1].yy[1:XX.fw[1].κ[1]+2])

print(XX.bwc[1].tt[1:XX.bwc[1].κ[1]+2])
print(XX.bwc[1].yy[1:XX.bwc[1].κ[1]+2])
print(16.0.-reverse(XX.bw[4].tt[1:XX.bw[4].κ[1]+2]))
print(reverse(XX.bw[4].yy[1:XX.bw[4].κ[1]+2]))

p = plotMe(1, false)
plotMe(2)
plotMe(3)
plotMe(4)

plotMeBw(1)
plotMeBw(2)
plotMeBw(3)
plotMeBw(4)

scatter!([XX.τ[1][3]],[XX.τ[1][4]])
show(p)


# Let's test auxiliary sampler now
Random.seed!(4)
N = 10000
samples = zeros(Int64, N)
for i in 1:N
    samples[i] = rand!(XX, P, Auxiliary())
end
probs = [length(samples[samples.≥i])/N for i in 2:8]

Random.seed!(4)

rand!(XX, P, Auxiliary())

ttᵒ, yyᵒ = path!(XX, 0.0:0.01:16.0)
plot!(ttᵒ, yyᵒ)

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
