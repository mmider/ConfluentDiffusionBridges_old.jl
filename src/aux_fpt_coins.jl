import Random.rand
import SpecialFunctions.gamma

function rand(::Acoin, XX::ConfluentDiffBridge)
    iᵒ = XX.τᵒ[1][1]
    τIdx = XX.τᵒ[1][2]
    N = length(XX)
    rand(Acoin(), XX.bwcᵒ[iᵒ], XX.auxᵒ[iᵒ], τIdx:XX.bwcᵒ[iᵒ].κ[1]+1) && return true
    for i in iᵒ+1:N
        rand(Acoin(), XX.bwcᵒ[i], XX.auxᵒ[i], 1:XX.bwcᵒ[i].κ[1]+1) && return true
    end
    return false
end

function rand(::Acoin, A::PathSegment, B::PathSegment, iRange)
    for i in iRange
        d0 = A.yy[i]-B.yy[i]
        dT = A.yy[i+1] - B.yy[i+1]
        T = A.tt[i+1] - A.tt[i]
        rand(Dcoin(), d0, dT, T) && return true
    end
    return false
end

function rand!(pB::Bcoin, cc::CoinContainer, XX::ConfluentDiffBridge)
    iᵒ = XX.τᵒ[1][1]
    τIdx = XX.τᵒ[1][2]
    for i in 1:iᵒ-1
        rand!(pB, XX.fwcᵒ[i], XX.bwcᵒ[i], XX.auxᵒ[i], cc, 1:XX.fwcᵒ[i].κ[1]+1) && return true
    end
    rand!(pB, XX.fwcᵒ[iᵒ], XX.bwcᵒ[iᵒ], XX.auxᵒ[iᵒ], cc, 1:τIdx-2) && return true
    return false
end

function rand!(pC::Ccoin, cc::CoinContainer, XX::ConfluentDiffBridge)
    iᵒ = XX.τᵒ[1][1]
    τIdx = XX.τᵒ[1][2]
    return rand!(pC, XX.fwcᵒ[iᵒ], XX.bwcᵒ[iᵒ], XX.auxᵒ[iᵒ], cc, τIdx-1:τIdx-1)
end


function rand!(coin::S, fw::PathSegment, bw::PathSegment, aux::PathSegment,
               cc::CoinContainer, iRange) where S <: Union{Bcoin, Ccoin}
    for i in iRange
        rand!(coin, cc, fw.yy[i], fw.yy[i+1], bw.yy[i], bw.yy[i+1], aux.yy[i],
              aux.yy[i+1], fw.tt[i], fw.tt[i+1]) && return true
    end
    return false
end


function rand!(p::S, cc::CoinContainer, x0_fw::Float64, xT_fw, x0_bw,
               xT_bw, x0_aux, xT_aux, t0, T) where S <: Union{Bcoin, Ccoin}
    α, r0, rT, r, θ₀, Tᵒ = set_constants!(cc, x0_fw, xT_fw, x0_bw, xT_bw,
                                          x0_aux, xT_aux, t0, T)
    θₜ = compute_θₜ(p, cc, Tᵒ)
    logc = compute_logc(p, cc, Tᵒ, α, r0, rT)

    too_far_apart(Tᵒ, cc.g0[2], cc.gT[2]) && return false
    too_far_apart(Tᵒ, cc.g0[1], cc.gT[1]) && return rand(Dcoin(), cc.g0[2], cc.gT[2], Tᵒ)

    U = rand(Uniform())

    reset_error!(cc, r)
    M, N = 0, 0
    N = append_multipliers!(p, cc, N, θ₀, θₜ, α)
    bessel_term!(cc, N, M, α, r)
    update_error!(cc, N, M, r)
    while true
        total = 0.0
        for i in 1:N
            total += cc.multipliers[i] * cc.bessel_func[i]
        end
        total = sign(total)*exp(log(abs(total)) + logc)
        error = get_error(cc, logc)

        (U ≤ total - error) && return false
        (U > total + error) && return true

        update_error!(cc, N+1, M, r)
        m_increm = 0
        while cc.errors[2] < cc.errors[3]
            m_increm += 1
            update_error!(cc, N+1, M+m_increm, r)
        end
        N = append_multipliers!(p, cc, N, θ₀, θₜ, α)
        for i in 1:N-1
            for m in 1:m_increm
                bessel_term!(cc, i, M+m, α, r)
            end
        end
        M += m_increm
        for m in 0:M
            bessel_term!(cc, N, m, α, r)
        end
    end
end


function set_constants!(cc::CoinContainer, x0_fw::Float64, xT_fw, x0_bw, xT_bw,
                        x0_aux, xT_aux, t0, T) where S <: Union{Bcoin, Ccoin}
    # set the end points of the G process: G:=(X^1-X^2,X^1-X^3)
    set_end_pts_G!(cc, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux, false)

    # find the type of coin that needs to be sampled (out of four types)
    determine_signs!(cc)

    # set the end points of the X process, possibly reflecting the values
    set_end_pts_X!(cc, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux, cc.signs[1])

    # redefine the G process for new X and reflect the second coordinate of the
    # G process if needed
    set_end_pts_G!(cc, cc.x0[1], cc.xT[1], cc.x0[2], cc.xT[2], cc.x0[3],
                   cc.xT[3], cc.signs[2])

    # find the constants
    α = compute_α(cc)
    Tᵒ = T-t0
    r0, rT, r = compute_r(cc, Tᵒ)
    θ₀ = compute_θ(cc.g0, Tᵒ, cc.signs[2])
    α, r0, rT, r, θ₀, Tᵒ
end

"""
    set_end_pts_G!(cc::CoinContainer, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux,
                   xT_aux, opposite_signs)

Set the end points of the G process, possibly inverting the sign of the second
coordinate
"""
function set_end_pts_G!(cc::CoinContainer, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux,
                        xT_aux, opposite_signs)
    cc.g0[1] = x0_fw - x0_bw
    cc.g0[2] = (-1)^opposite_signs * (x0_fw - x0_aux)
    cc.gT[1] = xT_fw - xT_bw
    cc.gT[2] = (-1)^opposite_signs * (xT_fw - xT_aux)
end

"""
    determine_signs!(cc::CoinContainer)

Determine which coin needs to be sampled:
    1 - G^1>0, G^2>0 ()
    2 - G^1>0, G^2<0 (opposite_signs)
    3 - G^1<0, G^2>0 (inverted signs, opposite signs)
    4 - G^1<0, G^2<0 (inverted_signs)
"""
function determine_signs!(cc::CoinContainer)
    cc.signs[1] = cc.g0[1] < 0                       # inverted signs
    cc.signs[2] = sign(cc.g0[1]) != sign(cc.g0[2])   # opposite signs
end

"""
    set_end_pts_X!(cc::CoinContainer, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux,
                   xT_aux, inverted_signs)

Set the end points of the X process, reflecting them if necessary
"""
function set_end_pts_X!(cc::CoinContainer, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux,
                        xT_aux, inverted_signs)
    s = inverted_signs ? -1 : 1
    cc.x0[1] = s * x0_fw
    cc.x0[2] = s * x0_bw
    cc.x0[3] = s * x0_aux

    cc.xT[1] = s * xT_fw
    cc.xT[2] = s * xT_bw
    cc.xT[3] = s * xT_aux
end

compute_α(cc::CoinContainer) = π/3.0*(1+!cc.signs[2])

function compute_r(cc::CoinContainer, T)
    s = cc.signs[2] ? 1 : -1
    r0 = compute_r(cc.g0, s)
    rT = compute_r(cc.gT, s)
    r0, rT, 0.5*r0*rT/T
end

compute_r(g::Vector{Float64}, s) = √(2.0/3.0*(g[1]^2 + g[2]^2+s*g[1]*g[2]))


compute_θₜ(pB::Bcoin, cc::CoinContainer, T) = compute_θ(cc.gT, T, cc.signs[2])
compute_θₜ(pB::Ccoin, cc::CoinContainer, T) = nothing

function compute_θ(g::Vector{Float64}, T::Float64, opposite_signs::Bool)
    s = opposite_signs ? -1 : 1
    s1 = opposite_signs ? 1 : -1

    d₀ = s*0.5*g[2]
    d₁ = √3.0 * g[2]/(2.0*g[1] + s1*g[2])

    (g[1] < d₀) && return π + atan(d₁)
    (g[1] == d₀) && return 0.5*π
    return atan(d₁)
end

function compute_logc(pB::Bcoin, cc::CoinContainer, T, α, r0, rT)
    s1 = cc.signs[2] ? 1 : -1
    d1 = cc.gT[1]-cc.g0[1]
    d2 = cc.gT[2]-cc.g0[2]
    temp₁ = ( -1.0/(3.0*T) * (d1^2 + s1*d1*d2 + d2^2)
              + log(1.0-exp(-cc.g0[1]*cc.gT[1]/T)) )
    temp₂ = log(4.0*π) - log(α) - (rT^2 + r0^2)/(2.0*T)
    temp₂ - temp₁
end

function compute_logc(pB::Ccoin, cc::CoinContainer, T, α, r0, rT)
    s = cc.signs[2] ? -1 : 1
    d1 = cc.g0[1]
    d2 = cc.gT[2]-cc.g0[2]
    temp₁ = 1.0/(3.0*T) * (d1^2 + s*d1*d2 + d2^2)
    temp₂ = ( log(4.0) - 0.5*log(2.0) + 2.0*log(π) + log(T) - 2.0*log(α)
              - log(rT) - log(d1) - (rT^2 + r0^2)/(2.0*T) )
    temp₂ + temp₁
end

"""
    too_far_apart(T, g0, gT)

Check if the distances are so large, that the numerical approx will fail
"""
too_far_apart(T, g0, gT) = 1.5*√T < min(abs(g0),abs(gT))

function reset_error!(cc::CoinContainer, r)
    cc.MN[1] = -1
    cc.MN[2] = -1
    prepare_error!(cc, r)
end

function prepare_error!(cc::CoinContainer, r)
    cc.errors[1] = r^(3.0/(2.0-cc.signs[2])) + r^2 # log_e_trail
    cc.errors[2] = 0.0                             # log_e1
    cc.errors[3] = 3.0/(2.0-cc.signs[2]) * log(r)  # log_e2
end

function update_error!(cc::CoinContainer, N, M, r)
    while cc.MN[2] < N
        cc.MN[2] += 1
        cc.errors[2] += 3.0/(2.0-cc.signs[2]) * log(r) - log(cc.MN[2]+1.0)
    end
    while cc.MN[1] < M
        cc.MN[1] += 1
        cc.errors[3] += 2.0*log(r) - log(cc.MN[1]+1.0) - log(cc.MN[1]+2.0)
    end
end

function get_error(cc::CoinContainer, logc)
    exp(logc + cc.errors[1] + cc.errors[2]) + exp(logc + cc.errors[1] + cc.errors[3])
end

function append_multipliers!(p::S, cc::CoinContainer, N, θ₀, θₜ, α
                             ) where S <: Union{Bcoin, Ccoin}
    N += 1
    resize!(cc.multipliers, N)
    cc.multipliers[N] = get_multiplier(p, N, θ₀, θₜ, α)
    N
end

get_multiplier(pB::Bcoin, n, θ₀, θₜ, α) = sin(n*π*θₜ/α) * sin(n*π*θ₀/α)

get_multiplier(pC::Coin, n, θ₀, ::Any, α) = n*sin(n*π*(α-θ₀)/α)

function bessel_term!(cc::CoinContainer, n, m, α, r)
    if m == 0
        resize!(cc.bessel_terms, n)
        cc.bessel_terms[n] = n*π/α*log(r) - log(gamma(n*π/α+1.0))
        resize!(cc.bessel_func, n)
        cc.bessel_func[n] = exp(cc.bessel_terms[n])
    else
        cc.bessel_terms[n] += 2.0*log(r) - log(m) - log(m+n*π/α)
        cc.bessel_func[n] += exp(cc.bessel_terms[n])
    end
end
