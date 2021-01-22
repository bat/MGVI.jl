# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

function cholesky_L(m::AbstractSparseMatrix)
    sparse(cholesky(m).L)
end

function cholesky_L(m::AbstractMatrix)
    cholesky(m).L
end

function cholesky_L(m::Diagonal)
    Diagonal(sqrt.(m.diag))
end


struct PDLinMapWithChol{
    T,
    PMap<:LinearMaps.LinearMap{T},
    CMap<:LinearMaps.LinearMap{T}
} <: LinearMaps.LinearMap{T}
    parent::PMap
    chol_L::CMap
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


Base.parent(A::PDLinMapWithChol) = A.parent
cholesky_L(A::PDLinMapWithChol) = A.chol_L


Base.Matrix(A::PDLinMapWithChol) = Matrix(parent(A))

Base.convert(::Type{AbstractMatrix}, A::PDLinMapWithChol) =
    convert(AbstractMatrix, parent(A))
Base.convert(::Type{Matrix}, A::PDLinMapWithChol) =
    convert(Matrix, parent(A))
Base.convert(::Type{SparseMatrixCSC}, A::PDLinMapWithChol) =
    convert(SparseMatrixCSC, parent(A))
SparseArrays.sparse(A::PDLinMapWithChol) =
    sparse(parent(A))


Base.size(A::PDLinMapWithChol) = size(parent(A))

LinearAlgebra.issymmetric(A::PDLinMapWithChol) = true
LinearAlgebra.ishermitian(A::PDLinMapWithChol) = true
LinearAlgebra.isposdef(A::PDLinMapWithChol) = true

LinearAlgebra.adjoint(A::PDLinMapWithChol) = A

LinearAlgebra.transpose(A::PDLinMapWithChol{<:Real}) = A
LinearAlgebra.transpose(A::PDLinMapWithChol) = PDLinMapWithChol(transpose(A.parent), A.chol_L)

Base.@propagate_inbounds LinearMaps.A_mul_B!(
    y::AbstractVector, A::PDLinMapWithChol, x::AbstractVector
) = LinearMaps.A_mul_B!(y, parent(A), x)

Base.@propagate_inbounds LinearMaps.At_mul_B!(
    y::AbstractVector, A::PDLinMapWithChol, x::AbstractVector
) = LinearMaps.At_mul_B!(y, parent(A), x)

Base.@propagate_inbounds LinearMaps.Ac_mul_B!(
    y::AbstractVector, A::PDLinMapWithChol, x::AbstractVector
) = LinearMaps.Ac_mul_B!(y, parent(A), x)

LinearMaps.MulStyle(A::PDLinMapWithChol) = LinearMaps.MulStyle(parent(A))

Base.@propagate_inbounds LinearAlgebra.mul!(
    y::AbstractVector, A::PDLinMapWithChol, x::AbstractVector, α::Number, β::Number
) = mul!(y, parent(A), x, α, β)

Base.@propagate_inbounds LinearAlgebra.mul!(
    y::AbstractMatrix, A::PDLinMapWithChol, x::AbstractMatrix, α::Number, β::Number
) = mul!(y, parent(A), x, α, β)

Base.:(*)(A::LinearMap, B::PDLinMapWithChol) = A * parent(B)
Base.:(*)(A::PDLinMapWithChol, B::LinearMap) = parent(A) * B
Base.:(*)(A::PDLinMapWithChol, B::PDLinMapWithChol) = parent(A) * parent(B)

function _blockdiag(As::Tuple{Vararg{<:PDLinMapWithChol}})
    PDLinMapWithChol(blockdiag(map(parent, As)...), blockdiag(map(A -> A.chol_L, As)...))
end

const DiagLinearMap{T} = LinearMaps.WrappedMap{T,<:Diagonal}
const DiagPDLinMapWithChol = PDLinMapWithChol{T,<:DiagLinearMap,<:DiagLinearMap} where T

function _blockdiag(As::Tuple{Vararg{<:DiagPDLinMapWithChol}}) where T
    PDLinMapWithChol(
        LinearMap(Diagonal(vcat(map(A -> A.parent.lmap.diag, As)...)), issymmetric = true, ishermitian = true, isposdef = true),
        LinearMap(Diagonal(vcat(map(A -> A.chol_L.lmap.diag, As)...)), issymmetric = true, ishermitian = true, isposdef = true)
    )
end

blockdiag(As::PDLinMapWithChol...) = _blockdiag(As)

function _blockdiag(A::AbstractVector{<:DiagPDLinMapWithChol})
    d = reduce(vcat, map(x -> parent(x).lmap.diag, A))
    PDLinMapWithChol(Diagonal(d))
end
