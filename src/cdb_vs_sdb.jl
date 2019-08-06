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
include(joinpath(SRC_DIR, "simple_diffusion_bridges.jl"))
include(joinpath(SRC_DIR, "euler.jl"))

# pseudo-marginal mcmc sampler using simple diffusion bridges. Return chain
# of mid-points
function mcmc_sdb(P, x0, xT, tt, numMCMCsteps=1000)
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
        rand!(XX, P, Proposal(), x0, xT, WWᵒ)
#        rand!(XXᵒ, P, Proposal(), x0, xT, WWᵒ)
#        𝓣ᵒ = rand!(XXᵒ, P, Auxiliary(), WWᵒ)
#        if rand() ≤ 𝓣ᵒ/𝓣
#            XXᵒ, XX = XX, XXᵒ
#            𝓣ᵒ, 𝓣 = 𝓣, 𝓣ᵒ
#            numAccepted += 1
#        end
        midis[i] = XX.prop.yy[midi]
    end
    elapsed = time() - start
    print("Time elapsed: ", elapsed, "\n")
    print("Acceptance rate: ", numAccepted/numMCMCsteps, "\n")
    XX, midis
end

# pseudo-marginal mcmc sampler using confluent diffusion bridges. Return chain
# of mid-points
function mcmc_cdb(P, x0, xT, T, numSegments, numMCMCsteps=1000; cutoff=Inf)
    start = time()
    # workspace
    XX = ConfluentDiffBridge(T, numSegments)
    XXᵒ = ConfluentDiffBridge(T, numSegments)

    # initialise
    rand!(XX, P, Proposal(), x0, xT)
    𝓣 = rand!(XX, P, Auxiliary())

    numAccepted = 0
    midis = zeros(Float64, numMCMCsteps)
    for i in 1:numMCMCsteps
        rand!(XX, P, Proposal(), x0, xT)
#        rand!(XXᵒ, P, Proposal(), x0, xT)
#        𝓣ᵒ = rand!(XXᵒ, P, Auxiliary(), cutoff=cutoff)
#        if rand() ≤ 𝓣ᵒ/𝓣
#            XXᵒ, XX = XX, XXᵒ
#            𝓣ᵒ, 𝓣 = 𝓣, 𝓣ᵒ
#            numAccepted += 1
#        end
        ttᵒ, yyᵒ = path!(XX, [0.0, 0.5*T, T])
        midis[i] = yyᵒ[ttᵒ.==0.5*T][1]
    end
    elapsed = time() - start
    print("Time elapsed: ", elapsed, "\n")
    print("Acceptance rate: ", numAccepted/numMCMCsteps, "\n")
    XX, midis
end

x₀, xₜ, T = 2.0, 3.3, 4.0
λ = 0.0 # artificial inflator of number of poisson points (for debugging only)
P = LangevinT(3.0, λ)

_, mid_sdb = mcmc_sdb(P, x₀, xₜ, 0.0:0.001:T, 1000000)

_, mid_cdb = mcmc_cdb(P, x₀, xₜ, T, 1, 1000000; cutoff=50)

using Plots
histogram(mid_sdb, label="simple diff bridges", alpha=0.5, normalize=:pdf)
histogram!(mid_cdb, label="confluent diff bridges", alpha=0.5, normalize=:pdf)






d0 = 1.0
dT = 0.5
T = 1.0
num_samples = 100000
samples_b = zeros(Bool, num_samples)
samples_v = zeros(Float64, num_samples)
for i in 1:num_samples
    samples_b[i] = rand(Dcoin(), d0, dT, T)
    samples_v[i] = rand(τᴰ(), d0, dT, T)
end
print(sum(samples_b)/length(samples_b))

histogram(samples_v, normalize=:pdf)


num_samples = 100000
samples_b = zeros(Bool, num_samples)
cc = CoinContainer()
x0_fw, x0_bw, x0_aux = 1.45795, -0.68, 1.39092
xT_fw, xT_bw, xT_aux = 1.56026, -0.6, 1.38236
T = 0.0296597
for i in 1:num_samples
    samples_b[i] = rand!(Bcoin(), cc, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux, 0.0, T)
end
print(1.0 - sum(samples_b)/length(samples_b))


num_samples = 100000
samples_b = zeros(Bool, num_samples)
cc = CoinContainer()
x0_fw, x0_bw, x0_aux = -0.5, -1.8, -2.3
xT_fw, xT_bw, xT_aux = -1.2, -1.2, -1.5
T = 0.7
for i in 1:num_samples
    samples_b[i] = rand!(Ccoin(), cc, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux, 0.0, T)
end
print(1.0 - sum(samples_b)/length(samples_b))
