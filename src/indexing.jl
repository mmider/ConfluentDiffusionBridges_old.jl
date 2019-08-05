import Base: last

struct Idx{T}
    i::Int64
    i_1::Int64
    iᵒ::Int64
    iᵒ_1::Int64
    i_bd::Int64
    function Idx(i::Int64, iᵒ::Int64, rev::Bool, i_bd::Int64)
        i_1 = rev ? i+1 : i-1
        iᵒ_1 = iᵒ-1
        new{Val{rev}}(i, i_1, iᵒ, iᵒ_1, i_bd)
    end

    function Idx(idx::Idx{T}, incr, incrᵒ) where T<:Val
        new{T}(idx.i + incr, idx.i_1 + incr, idx.iᵒ + incrᵒ, idx.iᵒ_1 + incrᵒ, idx.i_bd)
    end
end

nextᵒ(idx::Idx) = Idx(idx, 0, 1)
next(idx::Idx{Val{true}}) = Idx(idx, -1, 0)
next(idx::Idx{Val{false}}) = Idx(idx, 1, 0)

next_nextᵒ(idx::Idx{Val{true}}) = Idx(idx, -1, 1)
next_nextᵒ(idx::Idx{Val{false}}) = Idx(idx, 1, 1)

lastIntv(idx::Idx) = idx.i == idx.i_bd

moreLeft(idx::Idx{Val{true}}) = idx.i ≥ idx.i_bd
moreLeft(idx::Idx{Val{false}}) = idx.i ≤ idx.i_bd
