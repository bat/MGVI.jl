# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


# Workaround for missing SVector() pullback in Zygote:
_svector(x::NTuple{N,T}) where {N,T<:Real} = SVector(x)

function ChainRulesCore.rrule(::typeof(_svector), x::NTuple{N,T}) where {N,T<:Real}
    function _svector_pullback(thunked_Δy)
        Δy = unthunk(thunked_Δy)
        return ChainRulesCore.NoTangent(), (Δy...,)
    end

    return _svector(x), _svector_pullback
end


# ToDo: Check Zygote.pullback performance of `reduce(vcat, A)`
_flatten_vec_of_vec(A::AbstractVector{<:AbstractVector{<:Real}}) = reduce(vcat, A)

_flatten_vec_of_vec(A::AbstractVector{U}) where {N,T<:Real,U<:Union{NTuple{N,T},StaticVector{N,T}}} = copy(reinterpret(T, A))

function ChainRulesCore.rrule(::typeof(_flatten_vec_of_vec), A::AbstractVector{U}) where {N,T<:Real,U<:Union{NTuple{N,T},StaticVector{N,T}}}
    function _flatten_vec_of_vec_pullback(thunked_Δy)
        Δy = unthunk(thunked_Δy)
        return ChainRulesCore.NoTangent(), copy(reinterpret(U, Δy))
    end

    return _flatten_vec_of_vec(A), _flatten_vec_of_vec_pullback
end

# ToDo (if ArraysOfArrays gets added to deps): _flatten_vec_of_vec(A::VectorOfSimilarVectors{<:Real}) = vec(flatview(C))
