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

function mgvi_residual_sampler(f::Function, center_p::Vector)
    fisher_map, jac_map = fisher_information_components(f, center_p)
    FullResidualSampler(fisher_map, jac_map)
end

export mgvi_residual_sampler
