"""
    samplePathsExactly(x0, xT, T, P, numPaths=100)

Sample `numPaths`-many independent paths of a diffusion bridge joining `x0` and
`xT` on the interval [0,`T`]. The unconditioned paths are distributed according
to the law `P`. Use path space rejection sampling
"""
function samplePathsExactly(x0, xT, T, P, numPaths=100)
    XX = PathSegment(0.0, T)
    pathSamples = Vector{Any}(undef, numPaths)
    timeElapsed = zeros(Float64, numPaths)
    endPts = zeros(Float64, numPaths)
    for i in 1:numPaths
        start = time()
        endPts[i] = rand!(XX, P, x0; xₜ=xT)
        timeElapsed[i] = time() - start
        iRange = 2:XX.κ[1]+1
        pathSamples[i] = (copy(XX.tt[iRange]), copy(XX.yy[iRange]))
    end
    print("Average time to simulate a single path: ", mean(timeElapsed[4:end]))
    pathSamples, endPts
end


"""
    plotPaths(tt, pathSamples, alpha=0.2)

Draw paths sampled using Rejection sampling on a path space. `tt` is the mesh
at which to sub-sample the paths, `pathSamples` is a collection of the sampled
paths and `alpha` sets the alpha channel for each path in the plot
"""
function plotPaths(tt, pathSamples, alpha=0.2)
    @assert tt[end] == T
    xx = fillBB(tt, x₀, xₜ, pathSamples[1]...)
    p = plot(tt, xx, alpha=alpha, color="steelblue", label="")
    numPaths = length(pathSamples)
    for i in 2:numPaths
        xx = fillBB(tt, x₀, xₜ, pathSamples[i]...)
        plot!(tt, xx, alpha=alpha, color="steelblue", label="")
    end
    p
end


"""
    extractMidPts(x0, xT, T, pathSamples)

Extract mid-points from the path sampled using rejection sampling on a path
space
"""
function extractMidPts(x0, xT, T, pathSamples)
    N = length(pathSamples)
    midPts = zeros(Float64, N)
    tt = [0.0, 0.5*T, T]
    for i in 1:N
        xx = fillBB(tt, x0, xT, pathSamples[i]...)
        midPts[i] = xx[2]
    end
    midPts
end
