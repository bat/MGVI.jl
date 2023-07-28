# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


function _fisher_information_and_jac(fwd_model::Function, ξ::AbstractVector, OP, context::MGVIContext)
    ℐ_λ = fisher_information(fwd_model(ξ))
    _, dλ_dξ = with_jacobian(flat_params ∘ fwd_model, ξ, OP, context.ad)
    ℐ_λ, dλ_dξ
end



"""
    abstract type LinearSolverAlg

Abstract supertype for linear solver algorithms.
"""
abstract type LinearSolverAlg end


"""
    struct MatrixInversion <: LinearSolverAlg

Solve linear systems by direct matrix inversion.

Note: Will instantiate implicit matrices/operators in memory explicitly.
"""
struct MatrixInversion <: LinearSolverAlg end



"""
    struct IterativeSolversCG <: LinearSolverAlg

Solve linear systems using `IterativeSolvers.gc`.
"""

struct IterativeSolversCG{OPTS<:NamedTuple} <: LinearSolverAlg
    cgopts::OPTS
end

IterativeSolversCG() = IterativeSolversCG(NamedTuple())



"""
    struct ResidualSampler

Generates zero-centered samples from the posterior's covariance approximated
by the Fisher information.

This sampler uses Conjugate Gradients to iteratively invert  the Fisher information,
never instantiating the covariance in memory explicitly.

The Fisher information in canonical coordinates and Jacobian of the coordinate transformation
are provided as arguments.

Constructor:

```julia
ResidualSampler(f_model::Function, center_point::Vector{<:Real}, solver::MGVI.LinearSolverAlg, context::MGVIContext)
```

Call `MGVI.sample_residuals(s::ResidualSampler[, n::Integer])` to generate a
single or `n` samples.
"""
struct ResidualSampler{F,RV<:AbstractVector{<:Real},SLV<:LinearSolverAlg,OPL<:LinearMap,OPJ<:LinearMap,CTX<:MGVIContext}
    f_model::F
    center_point::RV
    solver::SLV
    λ_information::OPL
    jac_dλ_dθ::OPJ
    context::CTX
end
export ResidualSampler


_get_operator_type(::MatrixInversion) = Matrix
_get_operator_type(::IterativeSolversCG) = LinearMap

function ResidualSampler(f_model::Function, center_point::Vector{<:Real}, solver::LinearSolverAlg, context::MGVIContext)
    OP = _get_operator_type(solver)
    ℐ_λ, dλ_dξ = _fisher_information_and_jac(f_model, center_point, OP, context)
    ResidualSampler(f_model, center_point, solver, convert(LinearMap, ℐ_λ), convert(LinearMap, dλ_dξ), context)
end


function sample_residuals(s::ResidualSampler, n::Integer)
    m = size(s.jac_dλ_dθ, 2)
    A = allocate_array(s.context.gen, (m, n))
    Base.Threads.@threads for i in 1:size(A,2)
        view(A, :, i) .= sample_residuals(s)
    end
    return A
end


function sample_residuals(s::ResidualSampler{<:Any,<:AbstractVector{<:Real},<:MatrixInversion})
    genctx = s.context.gen

    ℐ_λ = s.λ_information
    dλ_dθ = s.jac_dλ_dθ
    n_λ, n_θ = size(dλ_dθ)

    Σ⁻¹_θ_est = dλ_dθ' * ℐ_λ * dλ_dθ + I
    Σ⁻¹_θ_est_matrix = allocate_array(genctx, (n_θ, n_θ))
    mul!(Σ⁻¹_θ_est_matrix, Σ⁻¹_θ_est, one(eltype(Σ⁻¹_θ_est_matrix)))
    root_covariance = cholesky(PositiveFactorizations.Positive, inv(Σ⁻¹_θ_est_matrix)).L
    root_covariance * randn(genctx, size(root_covariance, 1))
end


function sample_residuals(s::ResidualSampler{<:Any,<:AbstractVector{<:Real},<:IterativeSolversCG})
    genctx = s.context.gen

    ℐ_λ = s.λ_information
    dλ_dθ = s.jac_dλ_dθ
    n_λ, n_θ = size(dλ_dθ)
    Σ⁻¹_θ_est = dλ_dθ' * ℐ_λ * dλ_dθ + I

    dλ_dθ = s.jac_dλ_dθ
    sample_n = randn(genctx, n_λ)
    sample_eta = randn(genctx, n_θ)
    Δφ = dλ_dθ' * (cholesky_L(ℐ_λ) * sample_n) + sample_eta
    IterativeSolvers.cg(Σ⁻¹_θ_est, Δφ; s.solver.cgopts...)  # Δξ
end
