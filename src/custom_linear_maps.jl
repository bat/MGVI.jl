# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


"""
    struct MGVI.PDLinMapWithChol{T} <: LinearMaps.LinearMap{T}

A `LinearMap` that stores both a map and the lower-tringangular map
of its Cholesky decomposition.
"""
struct PDLinMapWithChol{
    T,
    PMap<:LinearMaps.LinearMap{T},
    CMap<:LinearMaps.LinearMap{T}
} <: LinearMaps.LinearMap{T}
    _linmap::PMap
    _chol_L::CMap
end

function PDLinMapWithChol(A::AbstractMatrix, chol_A::AbstractMatrix)
    chol_L = LinearMap(
        chol_A,
        issymmetric = false, ishermitian = false, isposdef = false
    )

    wrapped_A = LinearMap(
        A,
        issymmetric = true, ishermitian = true, isposdef = true
    )

    PDLinMapWithChol(wrapped_A, chol_L)
end

PDLinMapWithChol(A::AbstractMatrix) = PDLinMapWithChol(A, cholesky_L(A))


without_chol(A::PDLinMapWithChol) = A._linmap
cholesky_L(A::PDLinMapWithChol) = A._chol_L


Base.Matrix(A::PDLinMapWithChol) = Matrix(without_chol(A))

Base.convert(::Type{AbstractMatrix}, A::PDLinMapWithChol) =
    convert(AbstractMatrix, without_chol(A))
Base.convert(::Type{Matrix}, A::PDLinMapWithChol) =
    convert(Matrix, without_chol(A))
Base.convert(::Type{SparseMatrixCSC}, A::PDLinMapWithChol) =
    convert(SparseMatrixCSC, without_chol(A))
SparseArrays.sparse(A::PDLinMapWithChol) =
    sparse(without_chol(A))


Base.size(A::PDLinMapWithChol) = size(without_chol(A))

LinearAlgebra.issymmetric(A::PDLinMapWithChol) = true
LinearAlgebra.ishermitian(A::PDLinMapWithChol) = true
LinearAlgebra.isposdef(A::PDLinMapWithChol) = true

LinearAlgebra.adjoint(A::PDLinMapWithChol) = A

LinearAlgebra.transpose(A::PDLinMapWithChol{<:Real}) = A
LinearAlgebra.transpose(A::PDLinMapWithChol) = PDLinMapWithChol(transpose(without_chol(A)), cholesky_L(A))

LinearMaps.MulStyle(A::PDLinMapWithChol) = LinearMaps.MulStyle(without_chol(A))

Base.@propagate_inbounds LinearAlgebra.mul!(
    y::AbstractVector, A::PDLinMapWithChol, x::AbstractVector, α::Number, β::Number
) = mul!(y, without_chol(A), x, α, β)

Base.@propagate_inbounds LinearAlgebra.mul!(
    y::AbstractMatrix, A::PDLinMapWithChol, x::AbstractMatrix, α::Number, β::Number
) = mul!(y, without_chol(A), x, α, β)

Base.:(*)(A::LinearMap, B::PDLinMapWithChol) = A * without_chol(B)
Base.:(*)(A::PDLinMapWithChol, B::LinearMap) = without_chol(A) * B
Base.:(*)(A::PDLinMapWithChol, B::PDLinMapWithChol) = without_chol(A) * without_chol(B)

function _blockdiag(As::NTuple{N,PDLinMapWithChol}) where N
    PDLinMapWithChol(blockdiag(map(without_chol, As)...), blockdiag(map(cholesky_L, As)...))
end

const DiagLinearMap{T} = LinearMaps.WrappedMap{T,<:Diagonal}
const DiagPDLinMapWithChol = PDLinMapWithChol{T,<:DiagLinearMap,<:DiagLinearMap} where T

function _blockdiag(As::NTuple{N, DiagPDLinMapWithChol}) where N
    PDLinMapWithChol(
        LinearMap(Diagonal(vcat(map(A -> get_diagonal(without_chol(A).lmap), As)...)), issymmetric = true, ishermitian = true, isposdef = true),
        LinearMap(Diagonal(vcat(map(A -> get_diagonal(cholesky_L(A).lmap), As)...)), issymmetric = true, ishermitian = true, isposdef = true)
    )
end

function _blockdiag(A::AbstractVector{<:DiagPDLinMapWithChol})
    d = _flatten_vec_of_vec(map(x -> get_diagonal(without_chol(x).lmap), A))
    PDLinMapWithChol(Diagonal(d))
end

blockdiag(As::PDLinMapWithChol...) = _blockdiag(As)
