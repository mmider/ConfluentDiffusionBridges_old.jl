import Random.rand!

struct ConfluentDiffBridge
    fw::Vector{PathSegment}
    fwc::Vector{PathSegment}
    fwcᵒ::Vector{PathSegment}
    bw::Vector{PathSegment}
    bwc::Vector{PathSegment}
    bwcᵒ::Vector{PathSegment}
    aux::Vector{PathSegment}
    auxᵒ::Vector{PathSegment}
    τ::Vector{Tuple{Int64, Int64, Float64, Float64}}
    τᵒ::Vector{Tuple{Int64, Int64, Float64, Float64}}
    coin::CoinContainer

    function ConfluentDiffBridge(T::Number, numSegments::Integer)
        dt = T/numSegments

        c = [PathSegment((i-1)*dt, dt) for i in 1:numSegments]
        θ = [deepcopy(c) for i in 1:8]
        τIdx = [(1,1,1.0,1.0)]
        τIdxᵒ = [(1,1,1.0,1.0)]

        new(θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7], θ[8], τIdx, τIdxᵒ,
            CoinContainer())
    end
end

"""
    length(XX::ConfluentDiffBridge)

Return the number of segments that each container is made out of
"""
length(XX::ConfluentDiffBridge) = length(XX.fw)

"""
    rand!(XX::ConfluentDiffBridge, P::ContinuousTimeProcess, ::Proposal, x0, xT)

Draw a proposal diffusion according to the confluent diffusion bridge simulation
algorithm
"""
function rand!(XX::ConfluentDiffBridge, P::ContinuousTimeProcess, ::Proposal,
               x0, xT)
    diffsCross = false
    while !diffsCross
        rand!(XX.fw, P, x0) # call to path space rejection sampler
        rand!(XX.bw, P, xT)
        crossPopulate!(XX)
        diffsCross, crossIdx, τ, x_τ = diffusionsCross(XX.fwc, XX.bwc)
        diffsCross && (XX.τ[1] = (crossIdx..., τ, x_τ))
    end
end

"""
    crossPopulate!(XX::ConfluentDiffBridge)

Reveal forward and backward diffusion at additional time points, so that they
are both revealed on a common time-grid.
"""
function crossPopulate!(XX::ConfluentDiffBridge)
    N = length(XX)
    T = XX.fw[end].t₀ + XX.fw[end].T
    for i in 1:N
        crossPopulate!(XX.fw[i], XX.bw[N+1-i], XX.fwc[i], XX.bwc[i], T)
    end
end

"""
    crossPopulate!(fw::PathSegment, bw, fwᵒ, bwᵒ, T::Float64)

Reveal forward and backward diffusion at additional time points of a given
segment, so that on this segment they are both revealed on a common time-grid.
...
# Arguments
- fw: segment of a forward diffusion
- bw: segment of a backward diffusion
- fwᵒ: container where revealed forward diffusion will be stored
- bwᵒ: container where revealed backward diffusion will be stored
- T: length of the bridges
...
"""
function crossPopulate!(fw::PathSegment, bw, fwᵒ, bwᵒ, T::Float64)
    κ₁, κ₂ = fw.κ[1], bw.κ[1]
    resize!(κ₁+κ₂, fwᵒ, bwᵒ)

    i_fw = Idx(1,1,false,κ₁+2)
    i_bw = Idx(κ₂+2,1,true,1)

    i_fw, i_bw = initSegments!(fwᵒ, bwᵒ, fw, bw, i_fw, i_bw)

    # iterate over elements of the forward diffusion
    while moreLeft(i_fw)
        i_fw, i_bw = fillInFw!(fw.tt[i_fw.i_1], fw.tt[i_fw.i],
                               fw.yy[i_fw.i_1], fw.yy[i_fw.i],
                               fwᵒ, bwᵒ, bw, i_fw, i_bw, T)
        if !lastIntv(i_fw)
            bwᵒ.yy[i_bw.iᵒ] = sampleBB(bwᵒ.yy[i_bw.iᵒ_1], bw.yy[i_bw.i],
                                       bwᵒ.tt[i_bw.iᵒ_1], T-bw.tt[i_bw.i],
                                       fw.tt[i_fw.i])
            i_bw = nextᵒ(i_bw)
        else
            @assert lastIntv(i_bw)
            bwᵒ.yy[i_bw.iᵒ] = bw.yy[i_bw.i]
            bwᵒ.tt[i_bw.iᵒ] = fwᵒ.tt[i_fw.iᵒ_1] # to avoid numerical surprises
        end
        i_fw = next(i_fw)
    end
end

"""
    resize!(κ, fwᵒ, bwᵒ, auxᵒ=nothing)

Resize internal containers
...
# Arguments
- κ: total number of random (interior) points that need to be stored
- fwᵒ: container with forward path
- bwᵒ: container with backward path
- auxᵒ: container with auxiliary path
...
"""
function resize!(κ::Integer, fwᵒ, bwᵒ, auxᵒ=nothing)
    resize!(fwᵒ, κ+2)
    resize!(bwᵒ, κ+2)
    fwᵒ.κ[1] = bwᵒ.κ[1] = κ
    if auxᵒ != nothing
        resize!(auxᵒ, κ+2)
        auxᵒ.κ[1] = κ
    end
end

"""
    initSegments!(seg₁ᵒ, seg₂ᵒ, seg₁, seg₂, i₁₃, i₂, seg₃ᵒ=nothing, seg₃=nothing)

Initialise the proposal `ᵒ` segments by copying the first elements from the
corresponding regular segments
"""
function initSegments!(seg₁ᵒ, seg₂ᵒ, seg₁, seg₂, i₁₃, i₂, seg₃ᵒ=nothing,
                       seg₃=nothing)
    seg₁ᵒ.tt[i₁₃.iᵒ] = seg₂ᵒ.tt[i₂.iᵒ] = seg₁.tt[i₁₃.i]
    seg₁ᵒ.yy[i₁₃.iᵒ] = seg₁.yy[i₁₃.i]
    seg₂ᵒ.yy[i₂.iᵒ] = seg₂.yy[i₂.i]

    if seg₃ != nothing
        seg₃ᵒ.tt[i₁₃.iᵒ] = seg₃.tt[i₁₃.i]
        seg₃ᵒ.yy[i₁₃.iᵒ] = seg₃.yy[i₁₃.i]
    end

    i₁₃ = next_nextᵒ(i₁₃)
    i₂ = next_nextᵒ(i₂)
    i₁₃, i₂
end

"""
    fillInFw!(t0_fw, T_fw, x0_fw, xT_fw, fwᵒ, bwᵒ, bw, i_fw, i_bw, T)

Reveal forward diffusion at all time points inside the time interval
[`t0_fw`, `t_fw`] at which backward diffusion is already revealed at.
"""
function fillInFw!(t0_fw, T_fw, x0_fw, xT_fw, fwᵒ, bwᵒ, bw, i_fw, i_bw, T)
    t = T - bw.tt[i_bw.i]
    while T_fw > t
        fwᵒ.tt[i_fw.iᵒ] = t
        bwᵒ.tt[i_bw.iᵒ] = t

        fwᵒ.yy[i_fw.iᵒ] = sampleBB(x0_fw, xT_fw, t0_fw, T_fw, t)
        bwᵒ.yy[i_bw.iᵒ] = bw.yy[i_bw.i]

        t0_fw = t
        x0_fw = fwᵒ.yy[i_fw.iᵒ]

        i_fw = nextᵒ(i_fw)
        i_bw = next_nextᵒ(i_bw)
        t = T - bw.tt[i_bw.i]
    end
    fwᵒ.tt[i_fw.iᵒ] = T_fw
    bwᵒ.tt[i_bw.iᵒ] = T_fw
    fwᵒ.yy[i_fw.iᵒ] = xT_fw # bwc.yy[iₓ] is set outside the function
    i_fw = nextᵒ(i_fw)  # also incrementing next(i_fw) is done outside
    i_fw, i_bw
end

"""
    diffusionsCross(fwᵒ::Vector{PathSegment}, bwᵒ::Vector{PathSegment})

Check if forward diffusion path `fwᴼ` and backward diffusion path `bwᴼ` cross,
if so, sample the exact time of the first crossing from the left
"""
function diffusionsCross(fwᵒ::Vector{PathSegment}, bwᵒ::Vector{PathSegment})
    N = length(fwᵒ)
    for i in 1:N
        diffsCross, crossIdx, τ, x_τ = diffusionsCross(fwᵒ[i], bwᵒ[i])
        if diffsCross
            return true, (i, crossIdx), τ, x_τ
        end
    end
    return false, (nothing, nothing), nothing, nothing
end

"""
    diffusionsCross(fwᵒ::PathSegment, bwᵒ::PathSegment)

Check if the segment `fwᵒ` of the forward path and the segment `bwᵒ` of the
backward path cross, if so sample the exact time of the first crossing from the
left
"""
function diffusionsCross(fwᵒ::PathSegment, bwᵒ::PathSegment)
    N = fwᵒ.κ[1] + 2
    for i in 1:N-1
        d0 = fwᵒ.yy[i] - bwᵒ.yy[i]
        dT = fwᵒ.yy[i+1] - bwᵒ.yy[i+1]
        T = fwᵒ.tt[i+1] - fwᵒ.tt[i]
        if rand(Dcoin(), d0, dT, T)
            s0 = fwᵒ.yy[i] + bwᵒ.yy[i]
            sT = fwᵒ.yy[i+1] + bwᵒ.yy[i+1]
            τ = rand(τᴰ(), d0, dT, T)
            x_τ = 0.5*sampleBB(s0, sT, 0.0, T, τ; σ=√2.0)
            return true, i, fwᵒ.tt[i] + τ, x_τ
        end
    end
    return false, nothing, nothing, nothing
end


"""
    sampleBB(x0::Float64, xT::Float64, t0::Float64, T::Float64, t::Float64;
             σ=1.0)

Sample scaled Brownian bridge (scaled by `σ`) joining `x0` and `xT` on the time
interval [`t0`,`T`] at time `t`
"""
function sampleBB(x0::Float64, xT::Float64, t0::Float64, T::Float64, t::Float64;
                  σ=1.0)
    midPt = σ*√(t-t0)*randn(Float64)
    endPt = midPt + σ*√(T-t)*randn(Float64)
    midPt += x0*(T-t)/(T-t0) + (xT-endPt)*(t-t0)/(T-t0)
    midPt
end

"""
    rand!(XX::ConfluentDiffBridge, P::ContinuousTimeProcess, ::Auxiliary)

Sample auxiliary diffusions according to the law `P` until the first one that
hits the proposal path stored inside the container `XX`
"""
function rand!(XX::ConfluentDiffBridge, P::ContinuousTimeProcess, ::Auxiliary; cutoff=Inf)
    numAuxSamples = 0
    while true
        y = rand(P, Invariant())
        numAuxSamples += 1
        if abs(y) < cutoff # fast rejection (for numerical purposes)
            rand!(XX.aux, P, y) # call to path space rejection sampler
            crossPopulateAux!(XX)
            auxCross(XX) && return numAuxSamples
        end
    end
end


function auxCross(XX::ConfluentDiffBridge)
    diffsCross = ( simpleCrossing(XX) || rand(Acoin(), XX)
                   || rand!(Bcoin(), XX.coin, XX) || rand!(Ccoin(), XX.coin, XX) )
    return diffsCross
end

"""
    crossPopulateAux!(XX::ConfluentDiffBridge)

Reveal the forward, backward and auxiliary diffusions at a common time-grid
"""
function crossPopulateAux!(XX::ConfluentDiffBridge)
    iᵒ, _, _, _ = XX.τ[1]
    N = length(XX)
    for i in 1:iᵒ-1
        crossPopulateAuxLR!(XX.fwc[i], XX.bwc[i], XX.aux[i], XX.fwcᵒ[i],
                              XX.bwcᵒ[i], XX.auxᵒ[i], sampleCondBB)
    end
    crossPopulateAuxMid!(XX, XX.fwc[iᵒ], XX.bwc[iᵒ], XX.aux[iᵒ], XX.fwcᵒ[iᵒ],
                         XX.bwcᵒ[iᵒ], XX.auxᵒ[iᵒ], XX.τ[1])
    for i in iᵒ+1:N
        crossPopulateAuxLR!(XX.fwc[i], XX.bwc[i], XX.aux[i], XX.fwcᵒ[i],
                               XX.bwcᵒ[i], XX.auxᵒ[i], sampleBB)
    end
end


"""
    crossPopulateAuxLeft!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ)

Reveal the forward, backward and auxiliary diffusions at a common time-grid, on
a segment known to be lying to the left of the first crossing time.
"""
function crossPopulateAuxLR!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, fillingFn)
    κ₁, κ₂ = fw.κ[1], aux.κ[1]
    resize!(κ₁+κ₂, fwᵒ, bwᵒ, auxᵒ)

    i_fw = Idx(1,1,false,κ₁+2)
    i_aux = Idx(1,1,false,κ₂+2)

    i_fw, i_aux = initSegments!(fwᵒ, auxᵒ, fw, aux, i_fw, i_aux, bwᵒ, bw)

    while moreLeft(i_fw)
        i_fw, i_aux = fillInFwBwAux!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, i_fw, i_aux,
                                     fillingFn)
    end
end


"""
    fillInFwBwAux!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, i_fw, i_aux, fillingFn)

Reveal the forward, backward and auxiliary diffusions at a common time-grid, on
a segment known not to contain the first crossing time.
"""
function fillInFwBwAux!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, i_fw, i_aux, fillingFn)
    i_fw, i_aux = fillInFwBw!(fw.tt[i_fw.i_1], fw.tt[i_fw.i],
                              fw.yy[i_fw.i_1], fw.yy[i_fw.i],
                              bw.yy[i_fw.i_1], bw.yy[i_fw.i],
                              fwᵒ, bwᵒ, auxᵒ, aux, i_fw, i_aux, fillingFn)
    if !lastIntv(i_fw)
        auxᵒ.yy[i_aux.iᵒ] = sampleBB(auxᵒ.yy[i_aux.iᵒ_1], aux.yy[i_aux.i],
                                     auxᵒ.tt[i_aux.iᵒ_1], aux.tt[i_aux.i],
                                     fw.tt[i_fw.i])
        i_aux = nextᵒ(i_aux)
    else
        @assert lastIntv(i_aux)
        auxᵒ.yy[i_aux.iᵒ] = aux.yy[i_aux.i]
        auxᵒ.tt[i_aux.iᵒ] = fwᵒ.tt[i_fw.iᵒ_1]
    end
    i_fw = next(i_fw)
    i_fw, i_aux
end


"""
    fillInFwBw!(t0_fw, T_fw, x0_fw, xT_fw, x0_bw, xT_bw, fwᵒ, bwᵒ, auxᵒ, aux,
                i_fw, i_aux, fillingFn)

Reveal the forward and backward diffusions at all points that the auxliary
diffusions has already been revealed at on the interval [`t0_fw`, `T_fw`]
"""
function fillInFwBw!(t0_fw, T_fw, x0_fw, xT_fw, x0_bw, xT_bw, fwᵒ, bwᵒ,
                         auxᵒ, aux, i_fw, i_aux, fillingFn)
    t = aux.tt[i_aux.i]
    while T_fw > t
        fwᵒ.tt[i_fw.iᵒ] = t
        bwᵒ.tt[i_fw.iᵒ] = t
        auxᵒ.tt[i_aux.iᵒ] = t

        fwᵒ.yy[i_fw.iᵒ], bwᵒ.yy[i_fw.iᵒ] = fillingFn(x0_fw, xT_fw, x0_bw, xT_bw,
                                                     t0_fw, T_fw, t)
        auxᵒ.yy[i_aux.iᵒ] = aux.yy[i_aux.i]

        t0_fw = t
        x0_fw, x0_bw = fwᵒ.yy[i_fw.iᵒ], bwᵒ.yy[i_fw.iᵒ]

        i_fw = nextᵒ(i_fw)
        i_aux = next_nextᵒ(i_aux)
        t = aux.tt[i_aux.i]
    end
    fwᵒ.tt[i_fw.iᵒ] = T_fw
    bwᵒ.tt[i_fw.iᵒ] = T_fw
    auxᵒ.tt[i_aux.iᵒ] = T_fw

    fwᵒ.yy[i_fw.iᵒ] = xT_fw
    bwᵒ.yy[i_fw.iᵒ] = xT_bw
    i_fw = nextᵒ(i_fw)
    i_fw, i_aux
end

"""
    crossPopulateAuxMid!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, τIdx)

Reveal the forward, backward and auxiliary diffusions at a common time-grid, on
a segment known to contain the first crossing time.
"""
function crossPopulateAuxMid!(XX, fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, τIdx)
    crossIntv, crossIdx, τ, x_τ = τIdx

    κ₁, κ₂ = fw.κ[1], aux.κ[1]
    resize!(κ₁+κ₂+1, fwᵒ, bwᵒ, auxᵒ)

    i_fw = Idx(1,1,false,κ₁+2)
    i_aux = Idx(1,1,false,κ₂+2)

    i_fw, i_aux = initSegments!(fwᵒ, auxᵒ, fw, aux, i_fw, i_aux, bwᵒ, bw)

    while i_fw.i ≤ crossIdx
        i_fw, i_aux = fillInFwBwAux!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, i_fw, i_aux,
                                     sampleCondBB)
    end

    i_fw, i_aux, idx_of_τ = fillInFwBwAuxᵒ!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, i_fw,
                                            i_aux, τ, x_τ)

    XX.τᵒ[1] = (crossIntv, idx_of_τ, τ, x_τ)

    while moreLeft(i_fw)
        i_fw, i_aux = fillInFwBwAux!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, i_fw, i_aux,
                                     sampleBB)
    end
end


"""
    fillInFwBwAuxᵒ!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, i_fw, i_aux, τ, x_τ)

Reveal the forward, backward and auxiliary diffusions at a common time-grid, on
a segment known to contain the first crossing time.
"""
function fillInFwBwAuxᵒ!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, i_fw, i_aux, τ, x_τ)
    i_fw, i_aux = fillInFwBw!(fw.tt[i_fw.i_1], τ,
                              fw.yy[i_fw.i_1], x_τ, bw.yy[i_fw.i_1], x_τ,
                              fwᵒ, bwᵒ, auxᵒ, aux, i_fw, i_aux, sampleBessel)

    auxᵒ.yy[i_aux.iᵒ] = sampleBB(auxᵒ.yy[i_aux.iᵒ_1], aux.yy[i_aux.i],
                                 auxᵒ.tt[i_aux.iᵒ_1], aux.tt[i_aux.i], τ)
    idx_of_τ = i_aux.iᵒ
    i_aux = nextᵒ(i_aux)
    i_fw, i_aux = fillInFwBw!(τ, fw.tt[i_fw.i],
                              x_τ, fw.yy[i_fw.i], x_τ, bw.yy[i_fw.i],
                              fwᵒ, bwᵒ, auxᵒ, aux, i_fw, i_aux, sampleBB)
    if !lastIntv(i_fw)
        auxᵒ.yy[i_aux.iᵒ] = sampleBB(auxᵒ.yy[i_aux.iᵒ_1], aux.yy[i_aux.i],
                                     auxᵒ.tt[i_aux.iᵒ_1], aux.tt[i_aux.i],
                                     fw.tt[i_fw.i])
        i_aux = nextᵒ(i_aux)
    else
        @assert lastIntv(i_aux)
        auxᵒ.yy[i_aux.iᵒ] = aux.yy[i_aux.i]
        auxᵒ.tt[i_aux.iᵒ] = fwᵒ.tt[i_fw.iᵒ_1]
    end
    i_fw = next(i_fw)
    i_fw, i_aux, idx_of_τ
end

function sampleBB(x0_fw, xT_fw, x0_bw, xT_bw, t0_fw, T_fw, t)
    sampleBB(x0_fw, xT_fw, t0_fw, T_fw, t), sampleBB(x0_bw, xT_bw, t0_fw, T_fw, t)
end


function sampleCondBB(x0_fw, xT_fw, x0_bw, xT_bw, t0_fw, T_fw, t)
    d0 = x0_fw-x0_bw
    dT = xT_fw-xT_bw
    t₁ = t-t0_fw
    t₂ = T_fw-t
    while true
        xt_fw = sampleBB(x0_fw, xT_fw, t0_fw, T_fw, t)
        xt_bw = sampleBB(x0_bw, xT_bw, t0_fw, T_fw, t)
        dt = xt_fw-xt_bw
        if ( sign(d0) == sign(dt) && !rand(Dcoin(), d0, dt, t₁)
             && !rand(Dcoin(), dt, dT, t₂) )
            return xt_fw, xt_bw
        end
    end
end

function sampleBessel(x0_fw, xT_fw, x0_bw, xT_bw, t0_fw, T_fw, t)
    s0 = x0_fw + x0_bw
    sT = xT_fw + xT_bw
    st = sampleBB(s0, sT, t0_fw, T_fw, t; σ=√2.0)

    dsign = sign(x0_fw - x0_bw)
    d0 = abs(x0_fw - x0_bw)
    dt = sampleBesselBridge(d0, t0_fw, T_fw, t; σ=√2.0)
    0.5*(st + dsign*dt), 0.5*(st-dsign*dt)
end


function sampleBesselBridge(x0, t0, T, t; σ=1.0)
    sqdt = √(t-t0)
    x⁽¹⁾_t = σ*sqdt*randn(Float64)
    x⁽²⁾_t = σ*sqdt*randn(Float64)
    x⁽³⁾_t = σ*sqdt*randn(Float64)

    sqdt = √(T-t)
    x⁽¹⁾_T = x⁽¹⁾_t + σ*sqdt*randn(Float64)
    x⁽²⁾_T = x⁽²⁾_t + σ*sqdt*randn(Float64)
    x⁽³⁾_T = x⁽³⁾_t + σ*sqdt*randn(Float64)

    θₜ = (t-t0)/(T-t0)
    x⁽¹⁾_t -= θₜ * x⁽¹⁾_T
    x⁽²⁾_t -= θₜ * x⁽²⁾_T
    x⁽³⁾_t -= θₜ * x⁽³⁾_T

    √((x⁽¹⁾_t + x0*(1.0-θₜ))^2 + x⁽²⁾_t^2 + x⁽³⁾_t^2)
end


function simpleCrossing(XX::ConfluentDiffBridge)
    N = length(XX)
    iᵒ = XX.τᵒ[1][1]
    τIdx = XX.τᵒ[1][2]
    for i in 1:iᵒ-1
        simpleCrossing(XX.fwcᵒ[i], XX.auxᵒ[i], 1:XX.fwcᵒ[i].κ[1]+1) && return true
    end
    simpleCrossing(XX.fwcᵒ[iᵒ], XX.auxᵒ[iᵒ], 1:τIdx-1) && return true
    simpleCrossing(XX.bwcᵒ[iᵒ], XX.auxᵒ[iᵒ], τIdx:XX.bwcᵒ[iᵒ].κ[1]+1) && return true
    for i in iᵒ+1:N
        simpleCrossing(XX.bwcᵒ[i], XX.auxᵒ[i], 1:XX.bwcᵒ[i].κ[1]+1) && return true
    end
    return false
end


function simpleCrossing(seg₁::PathSegment, seg₂::PathSegment, iRange)
    for i in iRange
        if ((seg₁.yy[i] ≤ seg₂.yy[i] && seg₁.yy[i+1] ≥ seg₂.yy[i+1]) ||
            (seg₁.yy[i] ≥ seg₂.yy[i] && seg₁.yy[i+1] ≤ seg₂.yy[i+1]))
            return true
        end
    end
    return false
end


function path!(XX::ConfluentDiffBridge, tt)
    set_artificial_aux!(XX, tt)
    crossPopulateAux!(XX)
    iᵒ, τIdx, _, _ = XX.τᵒ[1]
    N = length(XX)
    ttᵒ = zeros(Float64, 0)
    yyᵒ = zeros(Float64, 0)
    for i in 1:iᵒ-1
        fw = XX.fwcᵒ[i]
        M = fw.κ[1]+1
        ttᵒ = vcat(ttᵒ, fw.tt[1:M])
        yyᵒ = vcat(yyᵒ, fw.yy[1:M])
    end
    fw = XX.fwcᵒ[iᵒ]
    ttᵒ = vcat(ttᵒ, fw.tt[1:τIdx])
    yyᵒ = vcat(yyᵒ, fw.yy[1:τIdx])

    bw = XX.bwcᵒ[iᵒ]
    M = bw.κ[1]+1
    ttᵒ = vcat(ttᵒ, bw.tt[τIdx+1:M])
    yyᵒ = vcat(yyᵒ, bw.yy[τIdx+1:M])
    for i in iᵒ+1:N
        bw = XX.bwcᵒ[i]
        M = bw.κ[1]+1
        ttᵒ = vcat(ttᵒ, bw.tt[1:M])
        yyᵒ = vcat(yyᵒ, bw.yy[1:M])
    end
    bw = XX.bwcᵒ[end]
    ttᵒ = vcat(ttᵒ, bw.tt[bw.κ[1]+2])
    yyᵒ = vcat(yyᵒ, bw.yy[bw.κ[1]+2])
    ttᵒ, yyᵒ
end


function set_artificial_aux!(XX::ConfluentDiffBridge, tt)
    N = length(XX)
    for i in 1:N
        fw = XX.fwc[i]
        t₀, T = fw.tt[1], fw.tt[fw.κ[1]+2]
        ttᵒ = tt[t₀ .≤ tt .≤ T]
        if t₀ != ttᵒ[1]
            ttᵒ = vcat(t₀, ttᵒ)
        end
        if T != ttᵒ[end]
            ttᵒ = vcat(ttᵒ, T)
        end
        M = length(ttᵒ)
        resize!(XX.aux[i], M)
        XX.aux[i].tt[1:M] .= ttᵒ
        XX.aux[i].yy[1:M] .= 0.0
        XX.aux[i].κ[1] = M-2
    end
end
