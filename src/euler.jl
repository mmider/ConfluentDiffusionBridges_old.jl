function solve!(::EulerMaruyama, P::ContinuousTimeProcess, X::SamplePath,
                W::SamplePath, u::Float64)
    N = length(W)
    N != length(X) && error("X and W differ in length.")

    ww = W.yy
    yy = X.yy
    tt = X.tt

    y::Float64 = u
    yy[1] = u

    for i in 1:N-1
        dWt = ww[i+1]-ww[i]
        dt = tt[i+1]-tt[i]
        y = y + drift(tt[i], y, P) * dt + vola(tt[i], y, P) * dWt
        yy[i+1] = y
    end
end
