#NOTE this is deprecated, as this step is not actually needed
"""
    buildConcat!((crossIntv, crossIdx), τ, XX::ConfluentDiffBridge)

Build a proposal path out of
"""
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
