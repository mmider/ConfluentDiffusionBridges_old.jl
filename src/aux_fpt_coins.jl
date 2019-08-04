import Random.rand
using SpecialFunctions

function rand(::Acoin, XX::ConfluentDiffBridge)
    iᵒ = XX.τ[1][1]
    τIdx = XX.τ[1][2]
    N = length(XX)
    rand(Acoin(), XX.bwcᵒ[iᵒ], XX.auxᵒ[iᵒ], τIdx+1:XX.bwcᵒ[iᵒ].κ[1]+1) && return true
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


function rand!(pB::Bcoin, XX::ConfluentDiffBridge)

end

function rand!(pC::Ccoin, XX::ConfluentDiffBridge)

end

function rand!(pB::Bcoin, x0_fw::Float64, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux,
               t0, T)
    α, r0, rT, r, θ₀ = set_constants!(pB, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux, t0, T)
    Tᵒ = T-t0
    θₜ = set_θₜ!(pB, pB.gT, Tᵒ)
    logc = compute_logc!(pB, Tᵒ, α, r0, rT)

    too_far_apart(Tᵒ, pB.g0[2], pB.gT[2]) && return true
    too_far_apart(Tᵒ, pB.g0[1], pB.gT[1]) && return rand(Dcoin(), pB.g0[2], pB.gT[2], Tᵒ)

    reset_error(pB, r)
    M, N = 0
    N = append_multipliers(pB, N, θ₀, θₜ, α)
    bessel_term!(pB, N, M, α, r)
    update_error!(pB, N, M, r)
    while true
        total = 0.0
        for i in 1:N
            total += pB.multipliers[i] * pB.bessel_func[i]
        end
        total = exp(log(total) + logc)
        error = get_error(pB, logc)
        U = rand(Uniform())
        if U ≤ total - error
            return true
        elseif U > total + error
            return false
        end
        update_error!(pB, N+1, M, r)
        m_increm = 0
        while pB.errors[2] < pB.errors[3]
            m_increm += 1
            update_error!(pB, N+1, M+m_increm, r)
        end
        N = append_multipliers(pB, N, θ₀, θₜ, α)
        for i in 1:N
            for m in 1:m_increm+1
                bessel_term!(pB, i, M+m, α, r)
            end
        end
        M += m_increm
        for m in 0:M+1 # fix numbering from 0
            bessel_term!(pB, M, m, α, r)
        end
    end
end

function rand!(pC::Ccoin, x0_fw::Float64, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux,
               t0, T)
    α, r0, rT, r, θ₀ = set_constants!(pC, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux, t0, T)
    logc = compute_logc!(pC, T-t0, α, r0, rT)
end



function set_constants!(p::S, x0_fw::Float64, xT_fw, x0_bw, xT_bw, x0_aux,
                        xT_aux, t0, T) where S <: Union{Bcoin, Ccoin}
    # set the end points of the G process: G:=(X^1-X^2,X^1-X^3)
    set_end_pts_G!(pB, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux, false)

    # find the type of coin that needs to be sampled (out of four types)
    determine_signs!(pB)

    # set the end points of the X process, possibly reflecting the values
    set_end_pts_X!(pB, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux)

    # redefine the G process for new X and reflect the second coordinate of the
    # G process if needed
    set_end_pts_G!(pB, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux, pB.signs[2])

    # find the constants
    α = compute_α(pB)
    r0, rT, r = compute_r(pB, T-t0)
    θ₀ = set_θ!(pB, pB.g0, T-t0)
    α, r0, rT, r, θ₀
end

function set_end_pts_G!(pB, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux, opposite_signs)
    pB.g0[1] = x0_fw - x0_bw
    pB.g0[2] = (-1)^opposite_signs * (x0_fw - x0_aux)
    pB.gT[1] = xT_fw - xT_bw
    pB.gT[2] = (-1)^opposite_signs * (xT_fw - xT_aux)
end

function determine_signs!(pB)
    pB.signs[1] = pB.g0[1] < 0 # inverted signs
    pB.signs[2] = sign(pB.g0[1]) != sign(pB.g0[2]) # opposite signs
end

function set_end_pts_X!(pB, x0_fw, xT_fw, x0_bw, xT_bw, x0_aux, xT_aux, inverted_signs)
    pB.x0[1] = (-1)^inverted_signs * x0_fw
    pB.x0[2] = (-1)^inverted_signs * x0_bw
    pB.x0[3] = (-1)^inverted_signs * x0_aux

    pB.xT[1] = (-1)^inverted_signs * xT_fw
    pB.xT[2] = (-1)^inverted_signs * xT_bw
    pB.xT[3] = (-1)^inverted_signs * xT_aux
end

function compute_α(pB)
    π/3.0*(1+!pB.signs[2])
end

function compute_r(pB, T)
    s = (-1)^(pB.signs[2]+1)
    r0 = √(2.0/3.0*(pB.g0[1]^2 + pB.g0[2]^2+s*pB.g0[1]*pB.g0[2]))
    rT = √(2.0/3.0*(pB.gT[1]^2 + pB.gT[2]^2+s*pB.gT[1]*pB.gT[2]))
    r0, rT, 0.5*r0*rT/T
end

compute_θₜ(pB::Bcoin, g, T) = compute_θ(pB, g, T)
compute_θₜ(pB::Ccoin, g, T) = nothing

function compute_θ(pB, g, T)
    s = (-1)^pB.signs[2]
    s1 = (-1)^(pB.signs[2]+1)

    d₀ = s*0.5*g[2]
    d₁ = √3.0 * g[2]/(2.0*g[1] + s1*g[2])

    (g[1] < d₀) && return π + atan(d₁)
    (g[1] == d₀) && return 0.5*π
    return atan(d₁)
end

function compute_logc(pB::Bcoin, T, α, r0, rT)
    s1 = (-1)^(pB.signs[2]+1)
    d1 = pB.gT[1]-pB.g0[1]
    d2 = pB.gT[2]-pB.g0[2]
    temp₁ = ( -1.0/(3.0*T) * (d1^2 + s1*d1*d2 + d2^2)
              + log(1.0-exp(-pB.g0[1]*pB.gT[1]/T)) )
    temp₂ = log(4.0*π) - log(α) - (rT^2 + r0^2)/(2.0*T)
    temp₂ - temp₁
end

function compute_logc(pB::Ccoin, T, α, r0, rT)
    s = (-1)^(pB.signs[2])
    d1 = pB.g0[1]
    d2 = pB.gT[2]-pB.g0[2]
    temp₁ = ( 1.0/(3.0*T) * (d1^2 + s*d1*d2 + d2^2)
              + log(1.0-exp(-pB.g0[1]*pB.gT[1]/T)) )
    temp₂ = ( log(4.0) - 0.5*log(2.0) + 2.0*log(π) + log(T) - 2.0*log(α)
              - log(rT) - log(d1) - (rT^2 + r0^2)/(2.0*T) )
    temp₂ + temp₁
end


too_far_apart(T, g0, gT) = 1.5*√T < fmin(abs(g0),abs(gT))

function reset_error!(pB, r)
    pB.errors_MN[1] = -1
    pB.errors_MN[2] = -1
    prepare_error!(pB, r)
end


function prepare_error!(pB, r)
    pB.errors[1] = r^(3.0/(2.0-pB.signs[2])) + r^2 # log_e_trail
    pB.errors[2] = 0.0                             # log_e1
    pB.errors[3] = 3.0/(2.0-pB.signs[2]) * log(r)  # log_e2
end

function update_error!(pB, N, M, r)
    while pB.errors_MN[2] < N
        pB.errors_MN[2] += 1
        pB.errors[2] += 3.0/(2.0-pB.signs[2]) * log(r) - log(pB.errors_MN[2])
    end
    while pB.errors_MN[1] < M
        pB.errors_MN[1] += 1
        pB.errors[3] += ( 2.0*log(r) - log(pB.errors_MN[1]+1.0)
                          - log(pB.errors_MN[1]+2.0) )
    end
end

function get_error(pB, logc)
    exp(logc + pB.errors[1] + pB.errors[2]) + exp(logc + pB.errors[1] + pB.errors[3])
end

function append_multipliers!(pB, N, θ₀, θₜ, α)
    N += 1
    resize!(pB.multipliers, N+1)
    pB.multipliers[N] = get_multiplier(N)
    N
end

get_multiplier(pB::Bcoin, n, θ₀, θₜ, α) = sin(n*π*θₜ/α) * sin(n*π*θ₀/α)

get_multiplier(pC::Coin, n, θ₀, ::Any, α) = n*sin(n*π*(α-θ₀)/α)

function bessel_term(pB, n, m, α, r)
    if m == 0
        resize!(pB.bessel_terms, n)
        pB.bessel_terms[n] = n*π/α*log(r) - log(gamma(n*π/α+1.0))
        resize!(pB.bessel_func, n)
        pB.bessel_func[n] = exp(pB.bessel_terms[n])
    else
        pB.bessel_terms[n] += 2.0*log(r) - log(m) - log(m+n*π/α)
        pB.bessel_func[n] += exp(pB.bessel_terms[n])
    end
end
