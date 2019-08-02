import Random.rand!

struct ConfluentDiffBridge
    fw::Vector{PathSegment}
    fwc::Vector{PathSegment}
    fwcᵒ::Vector{PathSegment}
    bw::Vector{PathSegment}
    bwc::Vector{PathSegment}
    bwcᵒ::Vector{PathSegment}
    prop::Vector{PathSegment}
    aux::Vector{PathSegment}
    auxᵒ::Vector{PathSegment}
    τIdx::Vector{Tuple{Int64, Int64, Float64, Float64}}

    function ConfluentDiffBridge(T::Number, numSegments::Integer)
        dt = T/numSegments

        c = [PathSegment((i-1)*dt, dt) for i in 1:numSegments]
        θ = [deepcopy(c) for i in 1:9]
        τIdx = [(1,1)]

        new(θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7], θ[8], θ[9], τIdx)
    end
end

length(XX::ConfluentDiffBridge) = length(XX.fw)

function rand!(XX::ConfluentDiffBridge, P::ContinuousTimeProcess, ::Proposal,
               x0, xT)
    diffsCross = false
    N = length(XX)
    while !diffsCross
        rand!(XX.fw, P, x0)
        rand!(XX.bw, P, xT)
        crossPopulate!(XX)
        diffsCross, crossIdx, τ = diffusionsCross(XX.fw, XX.bw)
        diffsCross && buildConcat!(crossIdx, τ, XX)
    end
end

function crossPopulate!(XX::ConfluentDiffBridge)
    N = length(XX)
    T = XX.fw[end].t₀ + XX.fw[end].T
    for i in 1:N
        crossPopulate!(XX.fw[i], XX.bw[N+1-i], XX.fwc[i], XX.bwc[i], T)
    end
end


function resize!(κ₁::Integer, κ₂::Integer, fwᵒ, bwᵒ, auxᵒ=nothing; extra=0)
    κ = κ₁ + κ₂ + extra
    resize!(fwᵒ, κ+2)
    resize!(bwᵒ, κ+2)
    fwᵒ.κ[1] = bwᵒ.κ[1] = κ
    if auxᵒ != nothing
        resize!(auxᵒ, κ+2)
        auxᵒ.κ[1] = κ
    end
end


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


function crossPopulate!(fw::PathSegment, bw, fwᵒ, bwᵒ, T::Float64)
    κ₁, κ₂ = fw.κ[1], bw.κ[1]
    resize!(κ₁, κ₂, fwᵒ, bwᵒ)

    i_fw = Idx(1,1,false)
    i_bw = Idx(κ₂+2,1,true)

    i_fw, i_bw = initSegments!(fwᵒ, bwᵒ, fw, bw, i_fw, i_bw)

    # iterate over elements of the forward diffusion
    while i_fw.i < κ₁+3
        i_fw, i_bw = fillInForwardDiff!(fw.tt[i_fw.i_1], fw.tt[i_fw.i],
                                        fw.yy[i_fw.i_1], fw.yy[i_fw.i],
                                        fwᵒ, bwᵒ, bw, i_fw, i_bw, T)
        if i_fw.i < κ₁+2
            bwᵒ.yy[i_bw.iᵒ] = sampleBB(bwᵒ.yy[i_bw.iᵒ_1], bw.yy[i_bw.i],
                                       bwᵒ.tt[i_bw.iᵒ_1], T-bw.tt[i_bw.i],
                                       fw.tt[i_fw.i])
            i_bw = nextᵒ(i_bw)
        else
            @assert i_bw.i == 1
            bwᵒ.yy[i_bw.iᵒ] = bw.yy[i_bw.i]
            bwᵒ.tt[i_bw.iᵒ] = fwᵒ.tt[i_fw.iᵒ_1] # to avoid numerical surprises
        end
        i_fw = next(i_fw)
    end
end


function fillInForwardDiff!(t0_fw, T_fw, x0_fw, xT_fw, fwᵒ, bwᵒ, bw, i_fw, i_bw, T)
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
    i_fw = nextᵒ(i_fw)
    i_fw, i_bw
end


function diffusionsCross(fwc::Vector{PathSegment}, bwc::Vector{PathSegment})
    N = length(fwc)
    for i in 1:N
        diffsCross, crossIdx, τ = diffusionsCross(fwc[i], bwc[i])
        if diffsCross
            return true, (i, crossIdx), τ
        end
    end
    return false, (nothing, nothing), nothing
end

# this is the simple crossing check
function diffusionsCross(fwc::PathSegment, bwc::PathSegment)
    N = fwc.κ[1] + 2
    for i in 1:N-1
        # transform to diffusion G
        g0 = fwc.yy[i] - bwc.yy[i]
        gT = fwc.yy[i+1] - bwc.yy[i+1]
        T = fwc.tt[i+1] - fwc.tt[i]
        if sign(g0) != sign(gT) || rand(Dcoin(), g0, gT, T)
            τ = rand(τᴰ(), g0, gT, T)
            return true, i, τ
        end
    end
    return false, nothing, nothing
end

function copySegments!(copyTo, copyFrom, iRange)
    for i in iRange
        resize!(copyTo[i], copyFrom[i].κ[1]+2)
        copyTo[i] .= copyFrom[i]
    end
end

function copyPartOfSegment!(copyTo, copyFrom, iᵒRange, iRange)
    copyTo.yy[iᵒRange] .= copyFrom.yy[iRange]
    copyTo.tt[iᵒRange] .= copyFrom.tt[iRange]
end


function buildConcat!((crossIntv, crossIdx), τ, XX::ConfluentDiffBridge)
    N = length(XX)
    copySegments!(XX.prop, XX.fwc, 1:crossIntv-1)

    # re-labeling
    prop, fw, bw = XX.prop[crossIntv], XX.fwc[crossIntv], XX.bwc[crossIntv]

    resize!(prop, fw.κ[1]+3) # +2 for start and end point +1 for crossing time
    prop.κ[1] = fw.κ[1]+1

    copyPartOfSegment!(prop, fw, 1:crossIdx, 1:crossIdx)

    s0 = fw.yy[crossIdx] + bw.yy[crossIdx]
    sT = fw.yy[crossIdx+1] + bw.yy[crossIdx+1]
    T = fw.tt[crossIdx+1] - fw.tt[crossIdx]
    prop.yy[crossIdx+1] = sampleAtCrossing(s0, sT, T, τ)
    prop.tt[crossIdx+1] = fw.tt[crossIdx] + τ

    # store crossing info
    XX.τIdx[1] = (crossIntv, crossIdx, prop.tt[crossIdx+1], prop.yy[crossIdx+1])

    # copy backward diffusion from then on
    copyPartOfSegment!(prop, bw, crossIdx+2:prop.κ[1]+2, crossIdx+1:prop.κ[1]+1)
    copySegments!(XX.prop, XX.bwc, crossIntv+1:N)
end


function sampleAtCrossing(s0, sT, T, τ)
    sτ = sampleBB(s0, sT, 0.0, T, τ; σ=√2.0)
    0.5*sτ
end


function sampleBB(x0::Float64, xT::Float64, t0::Float64, T::Float64, t::Float64;
                  σ=1.0)
    midPt = σ*√(t-t0)*randn(Float64)
    endPt = midPt + σ*√(T-t0)*randn(Float64)
    midPt += x0*(T-t)/(T-t0) + (xT-endPt)*(t-t0)/(T-t0)
    midPt
end


function rand!(XX::ConfluentDiffBridge, P::ContinuousTimeProcess, ::Auxiliary)
    numAuxSamples = 0
    while true
        y = rand(P, Invariant())
        rand!(XX.aux, P, y)
        numAuxSamples += 1
        crossPopulateAux!(XX)
        auxCross(XX) && return numAuxSamples
    end
end


function auxCross(XX::ConfluentDiffBridge)
    diffsCross = ( simpleCrossing(XX) || rand(Acoin(), XX)
                   || rand(Bcoin(), XX) || rand(Coin(), XX) )
    return diffsCross
end

function crossPopulateAux!(XX::ConfluentDiffBridge)
    N = length(XX)
    crossIntv, crossIdx = XX.τIdx
    iᵒ = crossIntv # just shortening the name
    for i in 1:iᵒ-1
        crossPopulateAuxLeft!(XX.fwc[i], XX.bwc[i], XX.aux[i], XX.fwcᵒ[i],
                              XX.bwcᵒ[i], XX.auxᵒ[i])
    end
    crossPopulateAuxMid!(XX.fwc[iᵒ], XX.bwc[iᵒ], XX.aux[iᵒ], XX.fwcᵒ[iᵒ],
                         XX.bwcᵒ[iᵒ], XX.auxᵒ[iᵒ], crossIdx)
    for i in iᵒ+1:N
        crossPopulateAuxRight!(XX.fwc[i], XX.bwc[i], XX.aux[i], XX.fwcᵒ[i],
                               XX.bwcᵒ[i], XX.auxᵒ[i])
    end
end



function crossPopulateAuxLeft!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ)
    κ₁, κ₂ = fw.κ[1], aux.κ[1]
    resize!(κ₁, κ₂, fwᵒ, bwᵒ, auxᵒ)

    i_fw = Idx(1,1,false)
    i_aux = Idx(1,1,false)

    i_fw, i_aux = initSegments!(fwᵒ, auxᵒ, fw, aux, i_fw, i_aux, bwᵒ, bw)

    for i in 2:κ₁+2
        i_fw, i_aux = fillInFwBw!(fw.tt[i_fw.i_1], fw.tt[i_fw.i],
                                      fw.yy[i_fw.i_1], fw.yy[i_fw.i],
                                      bw.yy[i_fw.i_1], bw.yy[i_fw.i],
                                      fwᵒ, bwᵒ, auxᵒ, aux, i_fw, i_aux,
                                      sampleCondBB)
        if i < κ₁+2
            auxᵒ.yy[i_aux.iᵒ] = sampleBB(auxᵒ.yy[i_aux.iᵒ_1], aux.yy[i_aux.i],
                                         auxᵒ.tt[i_aux.iᵒ_1], aux.tt[i_aux.i],
                                         fw.tt[i_fw.i])
            i_aux = nextᵒ(i_aux)
        else
            @assert i_aux.i = κ₂+2
            auxᵒ.yy[i_aux.iᵒ] = aux.yy[i_aux.i]
            auxᵒ.tt[i_aux.iᵒ] = fwᵒ.tt[i_fw.iᵒ_1]
        end
        i_fw = next(i_fw)
    end
end

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


function crossPopulateAuxMid!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, τIdx)
    crossIntv, crossIdx, τ, x_τ = τIdx

    κ₁, κ₂ = fw.κ[1], aux.κ[1]
    resize!(κ₁, κ₂, fwᵒ, bwᵒ, auxᵒ; extra=1)

    i_fw = Idx(1,1,false,κ₁+2)
    i_aux = Idx(1,1,false,κ₂+2)

    i_fw, i_aux, moreLeft = initSegments!(fwᵒ, auxᵒ, fw, aux, i_fw, i_aux, bwᵒ, bw)

    while i_fw.i ≤ crossIdx
        i_fw, i_aux = fillInFwBwAux!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, i_fw, i_aux, sampleCondBB)
    end

    fillInFwBwAuxᵒ!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, i_fw, i_aux, τ, x_τ)

    while moreLeft(i_fw)
        i_fw, i_aux = fillInFwBwAux!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, i_fw, i_aux, sampleBB)
    end
end


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

function fillInFwBwAuxᵒ!(fw, bw, aux, fwᵒ, bwᵒ, auxᵒ, i_fw, i_aux, τ, x_τ)
    i_fw, i_aux = fillInFwBw!(fw.tt[i_fw.i_1], fw.tt[i_fw.i_1] + τ,
                              fw.yy[i_fw.i_1], x_τ,
                              bw.yy[i_fw.i_1], x_τ,
                              fwᵒ, bwᵒ, auxᵒ, aux, i_fw, i_aux, sampleBessel)
    auxᵒ.yy[i_aux.iᵒ] = sampleBB(auxᵒ.yy[i_aux.iᵒ_1], aux.yy[i_aux.i],
                                 auxᵒ.tt[i_aux.iᵒ_1], aux.tt[i_aux.i],
                                 fw.tt[i_fw.i_1] + τ)
    i_aux = nextᵒ(i_aux)
    i_fw, i_aux = fillInFwBw!(fw.tt[i_fw.i_1] + τ, fw.tt[i_fw.i],
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
    i_fw, i_aux
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
        if (sign(x0_fw-x0_bw) == sign(xt_fw-xt_bw) &&
            !rand(Doin(), d0, dt, t₁) && !rand(Dcoin(), d0, dt, t₂))
            return xt_fw, xt_bw
        end
    end
end

function sampleBessel(x0_fw, xT_fw, x0_bw, xT_bw, t0_fw, T_fw, t)
    s0 = x0_fw + x0_bw
    sT = xT_bw + xT_bw
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
