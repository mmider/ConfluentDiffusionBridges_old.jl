import Base: getindex, setindex!, resize!, length


struct CustomContainer{T}
    x::Vector{T}
    N::Vector{Int}

    function CustomContainer{T}() where T
        x = zeros(T,1)
        N = [1]
        new{T}(x, N)
    end
end

getindex(cc::CustomContainer, i::Int64) = cc.x[i]
function setindex!(cc::CustomContainer{T}, x::S, i::Int64) where {T,S}
    cc.x[i] = x
end

function resize!(cc::CustomContainer, n)
    if n > cc.N[1]
        while n > cc.N[1]
            cc.N[1] *= 2
        end
        resize!(cc.x, cc.N[1])
    end
end
length(cc::CustomContainer) = cc.N[1]

"""
    CoinContainer

Flag indicating p_B coin
"""
struct CoinContainer
    g0::Vector{Float64}
    gT::Vector{Float64}
    x0::Vector{Float64}
    xT::Vector{Float64}
    signs::Vector{Bool}
    MN::Vector{Int64}
    errors::Vector{Float64}
    multipliers::CustomContainer{Float64}
    bessel_terms::CustomContainer{Float64}
    bessel_func::CustomContainer{Float64}

    function CoinContainer()
        g0 = zeros(Float64,2)
        gT = zeros(Float64,2)
        x0 = zeros(Float64,3)
        xT = zeros(Float64,3)
        signs = [false, false]
        MN = zeros(Int64,2)
        errors = zeros(Float64,3)
        multipliers = CustomContainer{Float64}()
        bessel_terms = CustomContainer{Float64}()
        bessel_func = CustomContainer{Float64}()

        new(g0, gT, x0, xT, signs, MN, errors, multipliers, bessel_terms,
            bessel_func)
    end
end
