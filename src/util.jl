# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


const DiagMatLike{T} = Union{Diagonal{T}, PDiagMat{T}}

# get_diagonal is only defined for diagonal matrices, unlike LinearAlgebra.diag:
get_diagonal(A::DiagMatLike) = diag(A)


without_chol(A::AbstractMatrix) = A
without_chol(A::PDMat) = A.mat
without_chol(A::PDiagMat) = Diagonal(diag(A))
without_chol(A::PDSparseMat) = A.mat
without_chol(A::ScalMat) = Diagonal(Fill(A.value, A.dim))

without_chol_pullback(thunked_Δy) = (NoTangent(), thunked_Δy)
function ChainRulesCore.rrule(::typeof(without_chol), A::AbstractMatrix)
    return without_chol(A), without_chol_pullback
end


cholesky_L(A::AbstractMatrix) = cholesky(A).L
cholesky_L(A::AbstractSparseMatrix) = sparse(cholesky(A).L)
cholesky_L(A::DiagMatLike) = Diagonal(sqrt.(get_diagonal(A)))



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
