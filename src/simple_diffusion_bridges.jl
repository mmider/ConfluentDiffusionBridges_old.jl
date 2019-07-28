import Random.rand!

struct SimpleDiffBridge
    fw::SamplePath{Float64}
    bw::SamplePath{Float64}
    prop::SamplePath{Float64}
    aux::SamplePath{Float64}

    function SimpleDiffBridge(tt)
        ttShifted = tt.-(tt[end]-tt[1])
        ttˣ = vcat(ttShifted[1:end-1], tt)
        new(SamplePath(tt), SamplePath(tt), SamplePath(tt), SamplePath(ttˣ))
    end
end

function rand!(XX::SimpleDiffBridge, P::ContinuousTimeProcess, ::Proposal,
               x0, xT, WW)
    diffsCross = false
    while !diffsCross
        rand!(Wiener(), WW.fw)
        solve!(Euler(), P, XX.fw, WW.fw, x0)

        rand!(Wiener(), WW.bw)
        solve!(Euler(), P, XX.bw, WW.bw, xT)

        diffsCross, crossIdx = diffusionsCross(XX.fw, XX.bw)
        buildConcat!(diffsCross, crossIdx, XX)
    end
end

function rand!(XX::SimpleDiffBridge, P::ContinuousTimeProcess, ::Auxiliary, WW)
    #XX.bw.yy[1]
    numAuxSamples = 0
    while true
        y = rand(P, Invariant())
        rand!(Wiener(), WW.aux)
        solve!(Euler(), P, XX.aux, WW.aux, y)
        numAuxSamples += 1
        auxCross(XX.prop, XX.aux) && return numAuxSamples
    end
end

function diffusionsCross(fw::SamplePath, bw::SamplePath)
    m = length(fw)
    for i in 1:m-1
        if ( (fw.yy[i] ≥ bw.yy[m+1-i] && fw.yy[i+1] ≤  bw.yy[m-i]) ||
             (fw.yy[i] ≤ bw.yy[m+1-i] && fw.yy[i+1] ≥  bw.yy[m-i]) )
             return true, i
        end
    end
    return false, nothing
end

function auxCross(X::SamplePath, aux::SamplePath)
    m = length(X)
    for i in 1:m-1
        if ( (X.yy[i] ≥ aux.yy[m-1+i] && X.yy[i+1] ≤  aux.yy[m+i]) ||
             (X.yy[i] ≤ aux.yy[m-1+i] && X.yy[i+1] ≥  aux.yy[m+i]) )
             return true
        end
    end
    return false
end

function buildConcat!(diffsCross, crossIdx, XX)
    if diffsCross
        m = length(XX.fw)
        XX.prop.yy[1:crossIdx] .= XX.fw.yy[1:crossIdx]
        XX.prop.yy[crossIdx+1:m] .= XX.bw.yy[m-crossIdx:-1:1]
    end
end
