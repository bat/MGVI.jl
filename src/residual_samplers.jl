# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

abstract type AbstractResidualSampler{C, S} <: Sampleable{C, S} end

struct FullResidualSampler <: AbstractResidualSampler{Multivariate, Continuous}
    λ_information_map::LinearMap
    jac_dλ_dθ_map::LinearMap
end

Base.length(rs::FullResidualSampler) = size(rs.jac_dλ_dθ_map, 2)

function Distributions._rand!(rng::AbstractRNG, s::FullResidualSampler, x::AbstractVector{T}) where T<:Real
    θ_information_map = assemble_fisher_information(s.λ_information_map, s.jac_dλ_dθ_map)
    covariance = Symmetric(inv(Matrix(θ_information_map) + I))
    dist = MvNormal(zeros(eltype(covariance), size(covariance, 1)), covariance)
    rand!(rng, dist, x)
end

struct ImplicitResidualSampler <: AbstractResidualSampler{Multivariate, Continuous}
    λ_information_map::LinearMap
    jac_dλ_dθ_map::LinearMap
    cg_params::NamedTuple
end

const _default_cg_params=(;)

ImplicitResidualSampler(λ_information_map::LinearMap,
                        jac_dλ_dθ_map::LinearMap;
                        cg_params::NamedTuple=_default_cg_params) = ImplicitResidualSampler(λ_information_map,
                                                                                            jac_dλ_dθ_map,
                                                                                            cg_params)

Base.length(rs::ImplicitResidualSampler) = size(rs.jac_dλ_dθ_map, 2)

function cholesky_sparse_L(lm::LinearMaps.WrappedMap{T, PDSparseMat{A, B}}) where {T, A, B}
    lm.lmap |> cholesky |> (fac -> fac.L) |> sparse |> LinearMap
end

function cholesky_sparse_L(lm::LinearMap)
    convert(SparseMatrixCSC, lm) |> cholesky |> (fac -> fac.L) |> sparse |> LinearMap
end

function cholesky_sparse_L(bd::LinearMaps.BlockDiagonalMap)
    blockdiag(map(cholesky_sparse_L, bd.maps)...)
end

function _implicit_rand_common_params(s::ImplicitResidualSampler)
    num_λs = size(s.jac_dλ_dθ_map, 1)
    num_θs = size(s.jac_dλ_dθ_map, 2)
    root_Id = cholesky_sparse_L(s.λ_information_map)
    invcov_estimate = assemble_fisher_information(s.λ_information_map, s.jac_dλ_dθ_map) + I
    (num_λs, num_θs, root_Id, invcov_estimate)
end

function _implicit_rand_common!(rng::AbstractRNG, s::ImplicitResidualSampler, x::AbstractVector{T},
                                num_λs, num_θs, root_Id, invcov_estimate) where T<:Real
    sample_n = randn(rng, num_λs)
    sample_eta = randn(rng, num_θs)
    Δφ = adjoint(s.jac_dλ_dθ_map) * (root_Id * sample_n) + sample_eta
    x[:] = cg(invcov_estimate, Δφ; s.cg_params...)  # Δξ
end

function Distributions._rand!(rng::AbstractRNG, s::ImplicitResidualSampler, x::AbstractVector{T}) where T<:Real
    params = _implicit_rand_common_params(s)
    _implicit_rand_common!(rng, s, x, params...)
end

function Distributions._rand!(rng::AbstractRNG, s::Sampleable{Multivariate}, A::DenseMatrix{T}) where T<:Real
    params = _implicit_rand_common_params(s)
    for i = 1:size(A,2)
        _implicit_rand_common!(rng, s, view(A,:,i), params...)
    end
    return A
end

export ImplicitResidualSampler, FullResidualSampler
