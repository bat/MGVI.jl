# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


function _fisher_information_and_jac(fwd_model::Function, ξ::AbstractVector, OP, context::MGVIContext)
    ℐ_λ = fisher_information(fwd_model(ξ))
    _, dλ_dξ = with_jacobian(flat_params ∘ fwd_model, ξ, OP, context.ad)
    ℐ_λ, dλ_dξ
end



"""
    struct MatrixInversion
Solve linear systems by direct matrix inversion.

Note: Will instantiate implicit matrices/operators in memory explicitly.
"""
struct MatrixInversion end


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
ResidualSampler(f_model::Function, center_point::Vector{<:Real}, linear_solver, context::MGVIContext)
```

`linear_solver` must be a solver supported by [`LinearSolve`](https://github.com/SciML/LinearSolve.jl) or
[`MGVI.MatrixInversion`](@ref). Use `MatrixInversion` only for low-dimensional problems.

Call `MGVI.sample_residuals(s::ResidualSampler[, n::Integer])` to generate a
single or `n` samples.
"""
struct ResidualSampler{F,RV<:AbstractVector{<:Real},SLV,OPL<:LinearMap,OPJ<:LinearMap,CTX<:MGVIContext}
    f_model::F
    center_point::RV
    linear_solver::SLV
    λ_information::OPL
    jac_dλ_dθ::OPJ
    context::CTX
end
export ResidualSampler


@inline _get_operator_type(::MatrixInversion) = DenseMatrix
@inline _get_operator_type(::Any) = LinearMap

function ResidualSampler(f_model::Function, center_point::Vector{<:Real}, linear_solver, context::MGVIContext)
    OP = _get_operator_type(linear_solver)
    ℐ_λ, dλ_dξ = _fisher_information_and_jac(f_model, center_point, OP, context)
    ResidualSampler(f_model, center_point, linear_solver, convert(LinearMap, ℐ_λ), convert(LinearMap, dλ_dξ), context)
end


function sample_residuals(s::ResidualSampler{<:Any,<:AbstractVector{<:Real},<:Any})
    genctx = s.context.gen

    ℐ_λ = s.λ_information
    dλ_dθ = s.jac_dλ_dθ
    n_λ, n_θ = size(dλ_dθ)
    Σ⁻¹_θ_est = dλ_dθ' * ℐ_λ * dλ_dθ + I

    dλ_dθ = s.jac_dλ_dθ
    sample_n = randn(genctx, n_λ)
    sample_eta = randn(genctx, n_θ)
    Δφ = dλ_dθ' * (cholesky_L(ℐ_λ) * sample_n) + sample_eta

    prob = LinearProblem{false}(Σ⁻¹_θ_est, Δφ)
    sol = solve(prob, s.linear_solver)
    Δξ = sol.u
    return Δξ
end

function sample_residuals(s::ResidualSampler, n::Integer)
    m = size(s.jac_dλ_dθ, 2)
    A = allocate_array(s.context.gen, (m, n))
    Base.Threads.@threads for i in 1:size(A,2)
        view(A, :, i) .= sample_residuals(s)
    end
    return A
end


function residual_pushfwd_operator(s::ResidualSampler{<:Any,<:AbstractVector{<:Real},<:MatrixInversion})
    genctx = s.context.gen

    ℐ_λ = s.λ_information
    dλ_dθ = s.jac_dλ_dθ
    n_λ, n_θ = size(dλ_dθ)

    Σ⁻¹_θ_est = dλ_dθ' * ℐ_λ * dλ_dθ + I
    Σ⁻¹_θ_est_matrix = allocate_array(genctx, (n_θ, n_θ))
    mul!(Σ⁻¹_θ_est_matrix, Σ⁻¹_θ_est, one(eltype(Σ⁻¹_θ_est_matrix)))
    Σ_θ_est_matrix = inv(Σ⁻¹_θ_est_matrix)
    Σ_θ_est_chol_l = cholesky(PositiveFactorizations.Positive, Σ_θ_est_matrix).L

    return Σ_θ_est_chol_l
end

function sample_residuals(s::ResidualSampler{<:Any,<:AbstractVector{<:Real},<:MatrixInversion})
    genctx = s.context.gen
    op = residual_pushfwd_operator(s)
    Δξ =  op * randn(genctx, size(op, 2))
    return Δξ
end

function sample_residuals(s::ResidualSampler{<:Any,<:AbstractVector{<:Real},<:MatrixInversion}, n::Integer)
    genctx = s.context.gen
    op = residual_pushfwd_operator(s)
    Δξ =  op * randn(genctx, size(op, 2), n)
    return Δξ
end
