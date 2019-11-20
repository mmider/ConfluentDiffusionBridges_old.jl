import Random.rand!

"""
    SimpleDiffBridge

Struct
```
struct SimpleDiffBridge
    fw::SamplePath{Float64}     # forward diffusion path
    bw::SamplePath{Float64}     # backward diffusion path
    prop::SamplePath{Float64}   # proposal diffusion path
    aux::SamplePath{Float64}    # auxiliary diffusion path
end
```
consists of containers needed for the Simple Diffusion Bridges algorithm
"""
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

"""
    rand!(XX::SimpleDiffBridge, P::ContinuousTimeProcess, ::Proposal, x0, xT,
          WW)

Draw proposal path using Simple diffusion bridges algorithm
...
# Arguments
- XX: object with all relevant containers
- P: target diffusion law
- ::Proposal: flag that a proposal bridge is to be drawn
- x0: starting point
- xT: end-point
- WW: container with the Wiener path
...
"""
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

"""
    rand!(XX::SimpleDiffBridge, P::ContinuousTimeProcess, ::Auxiliary, WW)

Sample auxiliary diffusion paths until the first crossing time of the proposal
...
# Arguments
- XX: object with all relevant containers
- P: target diffusion law
- ::Proposal: flag that auxiliary diffusions are to be drawn
- WW: container with the Wiener path
...
"""
function rand!(XX::SimpleDiffBridge, P::ContinuousTimeProcess, ::Auxiliary, WW)
    #y = XX.bw.yy[1]     # start from the end-point of the bridge
    numAuxSamples = 0
    while true
        y = rand(P, Invariant())    # start from the invariant measure
        rand!(Wiener(), WW.aux)
        solve!(Euler(), P, XX.aux, WW.aux, y)
        numAuxSamples += 1
        auxCross(XX.prop, XX.aux) && return numAuxSamples
    end
end

"""
    diffusionsCross(fw::SamplePath, bw::SamplePath)

Check if the forward diffusion path `fw` and backward diffusion path `bw` cross.
Return the index of a time point at which the first crossing occurs
"""
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

"""
    auxCross(X::SamplePath, aux::SamplePath)

Check if the proposal path `XX.prop` and auxiliary path `aux` cross
"""
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

"""
    buildConcat!(diffsCross, crossIdx, XX)

Splice forward and backward diffusions to create a proposal
...
# Arguments
- diffsCross: flag for whether diffusions cross, if the don't, do nothing
- crossIdx: index of the time point at which the first crossing occurs
- XX: object with all relevant containers
...
"""
function buildConcat!(diffsCross, crossIdx, XX)
    if diffsCross
        m = length(XX.fw)
        XX.prop.yy[1:crossIdx] .= XX.fw.yy[1:crossIdx]
        XX.prop.yy[crossIdx+1:m] .= XX.bw.yy[m-crossIdx:-1:1]
    end
end
