# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

"""
    abstract type AbstractResidualSampler <: Sampleable{Multivariate, Continuous} end

Generate zero-mean samples from Gaussian posterior, assuming priors are standard gaussians.

# Example
```julia
rng = MersenneTwister(145)
rs = FullResidualSampler(λ_information, jac_dλ_dθ)
samples = rand(rng, rs, 10)  # generate 10 samples
```
"""
abstract type AbstractResidualSampler <: Sampleable{Multivariate, Continuous} end

"""
    FullResidualSampler(λ_information, jac_dλ_dθ)

Gaussian posterior's covariance approximated by the Fisher Information

**This sampler constructs covariance matrix explicitly, so memory use grows
quadratically with the number of model parameters**

Fisher information in canonical coordinates and coordinate transformation jacobian are provided
as arguments. Since `` J \\cdot I_{canon} \\cdot J^T `` does not depend on the definition of canonical coordinates,
we omit description of what canonical coordinates are

# Arguments

* `λ_information::LinearMap`: Fisher Information in canonical coordinates. Coordinates are chosen in the way
  that this matrix is very simple, and diagonal for the univariate distributions
* `jac_dλ_fθ::LinearMap`: Coordinate transformation jacobian between canonical coordinates and
  coordinates of the model
"""
struct FullResidualSampler <: AbstractResidualSampler
    λ_information::LinearMap
    jac_dλ_dθ::LinearMap
end

Base.length(rs::FullResidualSampler) = size(rs.jac_dλ_dθ, 2)

function Distributions._rand!(rng::AbstractRNG, s::FullResidualSampler, x::AbstractVector{T}) where T<:Real
    θ_information = fisher_information_in_parspace(s.λ_information, s.jac_dλ_dθ)
    root_covariance = cholesky(PositiveFactorizations.Positive, inv(Matrix(θ_information) + I)).L
    x[:] = root_covariance * randn(eltype(root_covariance), size(root_covariance, 1))
end

"""
    ImplicitResidualSampler(λ_information, jac_dλ_dθ; cg_params=NamedTuple())

Memory efficient Gaussian posterior's covariance approximated by the Fisher Information

This sampler uses Conjugate Gradients to iteratively invert Fisher information, and
never instantiates entire jacobian or fisher information / covariance in memory.

# Arguments

* `λ_information::LinearMap`: Fisher Information in canonical coordinates.
* `jac_dλ_fθ::LinearMap`: Coordinate transformation jacobian between canonical coordinates and
* `cg_params::NamedTuple`: Keyword arguments passed to `cg`. Useful for enabling verbose mode
  or to set accuracy/number of iterations
"""
struct ImplicitResidualSampler <: AbstractResidualSampler
    λ_information::LinearMap
    jac_dλ_dθ::LinearMap
    cg_params::NamedTuple
end

const _default_cg_params=NamedTuple()

ImplicitResidualSampler(λ_information::LinearMap,
                        jac_dλ_dθ::LinearMap;
                        cg_params::NamedTuple=_default_cg_params) = ImplicitResidualSampler(λ_information,
                                                                                            jac_dλ_dθ,
                                                                                            cg_params)

Base.length(rs::ImplicitResidualSampler) = size(rs.jac_dλ_dθ, 2)

function _implicit_rand_impl_args(s::ImplicitResidualSampler)
    num_λs = size(s.jac_dλ_dθ, 1)
    num_θs = size(s.jac_dλ_dθ, 2)
    sqrt_Id = cholesky_L(s.λ_information)
    invcov_estimate = fisher_information_in_parspace(s.λ_information, s.jac_dλ_dθ) + I
    (num_λs, num_θs, sqrt_Id, invcov_estimate)
end

function _implicit_rand_impl!(rng::AbstractRNG, s::ImplicitResidualSampler, x::AbstractVector{T},
                              num_λs, num_θs, root_Id, invcov_estimate) where T<:Real
    sample_n = randn(rng, num_λs)
    sample_eta = randn(rng, num_θs)
    Δφ = adjoint(s.jac_dλ_dθ) * (root_Id * sample_n) + sample_eta
    x[:] = cg(invcov_estimate, Δφ; s.cg_params...)  # Δξ
end

function Distributions._rand!(rng::AbstractRNG, s::ImplicitResidualSampler, x::AbstractVector{T}) where T<:Real
    args = _implicit_rand_impl_args(s)
    _implicit_rand_impl!(rng, s, x, args...)
end

function Distributions._rand!(rng::AbstractRNG, s::ImplicitResidualSampler, A::DenseMatrix{T}) where T<:Real
    args = _implicit_rand_impl_args(s)
    for i = 1:size(A,2)
        _implicit_rand_impl!(rng, s, view(A,:,i), args...)
    end
    return A
end

export FullResidualSampler,
       ImplicitResidualSampler
