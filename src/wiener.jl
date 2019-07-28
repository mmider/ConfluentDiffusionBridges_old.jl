using Random
import Random.rand!

"""
    rand!(::Wiener{T}, WW, y=zero(T)) where T

Draw a path of a Wiener process
"""
function rand!(::Wiener{T}, WW, y=zero(T)) where T
    N = length(WW.tt)
    WW.yy[1] = y
    for i in 1:N-1
        rootdt = √(WW.tt[i+1]-WW.tt[i])
        WW.yy[i+1] = WW.yy[i] + rootdt*randn(T)
    end
    WW
end
