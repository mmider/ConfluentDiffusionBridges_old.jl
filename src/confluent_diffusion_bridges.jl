import Random.rand!

struct ConfluentDiffBridge
    fw::Vector{PathSegment}
    fwc::Vector{PathSegment}
    bw::Vector{PathSegment}
    bwc::Vector{PathSegment}
    prop::Vector{PathSegment}
    aux::Vector{PathSegment}
    τIdx::Vector{Tuple{Int64, Int64}}

    function ConfluentDiffBridge(T::Number, numSegments::Integer)
        dt = T/numSegments

        c = [PathSegment((i-1)*dt, dt) for i in 1:numSegments]
        θ = [deepcopy(c) for i in 1:6]
        τIdx = [(1,1)]

        new(θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], τIdx)
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
        crossPopulate!(XX.fw[i], XX.fwc[i], XX.bw[N+1-i], XX.bwc[i], T)
    end
end

function crossPopulate!(fw::PathSegment, fwc, bw, bwc, T::Float64)
    κ₁, κ₂ = fw.κ[1], bw.κ[1]
    κ = κ₁ + κ₂
    resize!(fwc, κ+2)
    resize!(bwc, κ+2)
    fwc.κ[1] = bwc.κ[1] = κ

    fwc.tt[1] = bwc.tt[1] = fw.tt[1]
    fwc.yy[1] = fw.yy[1]
    bwc.yy[1] = bw.yy[κ₂+2]

    # iterator scanning through backward diffusion
    i_bw = κ₂+1
    # iterator scanning through cross-populated diffusions
    iₓ = 2

    # iterate over elements of the forward diffusion
    for i in 2:κ₁+2
        i_bw, iₓ = fillInForwardDiff!(fw.tt[i-1], fw.tt[i], fw.yy[i-1],
                                      fw.yy[i], fwc, bwc, bw, i_bw, iₓ, T)
        if i < κ₁+2
            bwc.yy[iₓ] = sampleBB(bw.yy[i_bw+1], bw.yy[i_bw], T-bw.tt[i_bw+1],
                                  T-bw.tt[i_bw], fw.tt[i])
            iₓ += 1
        else
            @assert i_bw == 1
            bwc.yy[iₓ] = bw.yy[1]
            bwc.tt[iₓ] = fwc.tt[iₓ] # to avoid numerical surprises
        end
    end

end

function fillInForwardDiff!(t0_fw, T_fw, x0_fw, xT_fw, fwc, bwc, bw, i_bw, iₓ, T)
    t = T - bw.tt[i_bw]
    while T_fw > t
        fwc.tt[iₓ] = t
        bwc.tt[iₓ] = t

        fwc.yy[iₓ] = sampleBB(x0_fw, xT_fw, t0_fw, T_fw, t)
        bwc.yy[iₓ] = bw.yy[i_bw]

        t0_fw = t
        x0_fw = fwc.yy[iₓ]

        iₓ += 1
        i_bw -= 1
        t = T - bw.tt[i_bw]
    end
    fwc.tt[iₓ] = T_fw
    bwc.tt[iₓ] = T_fw
    fwc.yy[iₓ] = xT_fw # bwc.yy[iₓ] is set outside the function
    i_bw, iₓ
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


function buildConcat!((crossIntv, crossIdx), τ, XX::ConfluentDiffBridge)
    N = length(XX)
    for i in 1:crossIntv-1
        resize!(XX.prop[i], XX.fwc[i].κ[1]+2)
        XX.prop[i], XX.fwc[i] = XX.fwc[i], XX.prop[i]
    end
    # re-labeling
    prop = XX.prop[crossIntv]
    fw = XX.fwc[crossIntv]
    bw = XX.bwc[crossIntv]

    resize!(prop, fw.κ[1]+3) # +2 for start and end point +1 for crossing time
    prop.κ[1] = fw.κ[1]+1

    prop.yy[1:crossIdx] .= fw.yy[1:crossIdx] # copy forward diffusion until crossing
    prop.tt[1:crossIdx] .= fw.tt[1:crossIdx]

    s0 = fw.yy[crossIdx] + bw.yy[crossIdx]
    sT = fw.yy[crossIdx+1] + bw.yy[crossIdx+1]
    T = fw.tt[crossIdx+1] - fw.tt[crossIdx]
    prop.yy[crossIdx+1] = sampleAtCrossing(s0, sT, T, τ)
    prop.tt[crossIdx+1] = fw.tt[crossIdx] + τ

    # store crossing info
    XX.τIdx[1] = (crossIntv, crossIdx)

    # copy backward diffusion from then on
    iRange = crossIdx+1:prop.κ[1]+1
    prop.yy[iRange.+1] .= bw.yy[iRange]
    prop.tt[iRange.+1] .= bw.tt[iRange]

    for i in crossIntv+1:N
        resize!(XX.prop[i], XX.bwc[i].κ[1]+2)
        XX.prop[i], XX.bwc[i] = XX.bwc[i], XX.prop[i]
    end
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
