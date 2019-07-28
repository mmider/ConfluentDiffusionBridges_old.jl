struct Idx{T} where T<:Val
    i::Int64
    i_1::Int64
    iᵒ::Int64
    iᵒ_1::Int64
    function Idx(i::Int64, iᵒ::Int64, rev::Bool)
        i_1 = i-1
        iᵒ_1 = rev ? iᵒ+1 : iᵒ-1
        new{Val{rev}}(i, i_1, iᵒ, iᵒ_1)
    end

    function Idx(idx::Idx{T}, incr, incrᵒ)
        new{T}(idx.i + incr, idx.i_1 + incr, idx.iᵒ + incrᵒ, idx.iᵒ_1 + incrᵒ)
    end
end

nextᵒ(idx::Idx) = Idx(idx, 0, 1)
next(idx::Idx{Val{true}}) = Idx(idx, -1, 0)
next(idx::Idx{Val{false}}) = Idx(idx, 1, 0)

next_nextᵒ(idx::Idx{Val{true}}) = Idx(idx, -1, 1)
next_nextᵒ(idx::Idx{Val{false}}) = Idx(idx, 1, 1)
