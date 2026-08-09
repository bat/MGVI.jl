# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


"""
    struct MatrixInversion
Solve linear systems by direct matrix inversion.

Note: Will instantiate implicit matrices/operators in memory explicitly.
"""
struct MatrixInversion end

# MatrixInversion instantiates the posterior precision explicitly, so
# for it the Jacobian is computed as an explicit matrix up front - a
# single batched AD pass that works with forward-mode as well as
# reverse-mode-only selectors. Iterative solvers use the implicit
# Jacobian operator and never materialize:
_jacobian_op_type(::MatrixInversion) = AbstractMatrix
_jacobian_op_type(::Any) = MatrixShapedOperator

_as_jac_operator(J::MatrixShapedOperator) = J
_as_jac_operator(J::AbstractMatrix) = asoperator(J)

# fisher_information specializations may return a plain hermitian
# matrix, which gets wrapped; anything else that is not already a
# matrix-shaped operator is rejected with a clear error at this
# boundary instead of a method error deep inside the samplers:
_as_fisher_operator(ℐ::MatrixShapedOperator) = ℐ
_as_fisher_operator(ℐ::AbstractMatrix{<:Real}) = asoperator(ℐ, IsHermitian())
_as_fisher_operator(@nospecialize(ℐ)) = throw(ArgumentError(
    "fisher_information must return a MatrixShapedOperator that supports rowgram_factor, or a hermitian AbstractMatrix, got a $(nameof(typeof(ℐ)))"
))

function _fisher_information_and_jac(
    fwd_model::Function, ξ::AbstractVector, context::MGVIContext, JOP::Type = MatrixShapedOperator
)
    ℐ_λ = _as_fisher_operator(fisher_information(fwd_model(ξ)))
    _, dλ_dξ = with_jacobian(flat_params ∘ fwd_model, ξ, JOP, context.ad)
    ℐ_λ, _as_jac_operator(dλ_dξ)
end


"""
    struct ResidualSampler

Generates zero-centered samples from the posterior's covariance approximated
by the Fisher information.

This sampler uses Conjugate Gradients to iteratively invert the Fisher information,
never instantiating the covariance in memory explicitly.

The Fisher information in canonical coordinates and Jacobian of the coordinate transformation
are provided as arguments.

Constructor:

```julia
ResidualSampler(
    f_model::Function, center_point::Vector{<:Real}, linear_solver, context::MGVIContext;
    linear_solver_opts::NamedTuple = (;)
)
```

`linear_solver` must be a solver supported by [`LinearSolve`](https://github.com/SciML/LinearSolve.jl) or
[`MGVI.MatrixInversion`](@ref). Use `MatrixInversion` only for low-dimensional problems.
`linear_solver_opts` is passed to `LinearSolve.solve` as keyword arguments.

Call `MGVI.sample_residuals(s::ResidualSampler[, n::Integer])` to generate a
single or `n` samples.
"""
struct ResidualSampler{F,RV<:AbstractVector{<:Real},SLV,SLO<:NamedTuple,OPL<:MatrixShapedOperator,OPJ<:MatrixShapedOperator,CTX<:MGVIContext}
    f_model::F
    center_point::RV
    linear_solver::SLV
    linear_solver_opts::SLO
    λ_information::OPL
    jac_dλ_dθ::OPJ
    context::CTX
end
export ResidualSampler


function ResidualSampler(
    f_model::Function, center_point::Vector{<:Real}, linear_solver, context::MGVIContext;
    linear_solver_opts::NamedTuple = (;)
)
    ℐ_λ, dλ_dξ = _fisher_information_and_jac(f_model, center_point, context, _jacobian_op_type(linear_solver))
    ResidualSampler(f_model, center_point, linear_solver, linear_solver_opts, ℐ_λ, dλ_dξ, context)
end


function sample_residuals(s::ResidualSampler{<:Any,<:AbstractVector{<:Real},<:Any})
    genctx = s.context.gen

    ℐ_λ = s.λ_information
    dλ_dθ = s.jac_dλ_dθ
    n_θ = size(dλ_dθ, 2)
    F_λ = rowgram_factor(ℐ_λ)
    Σ⁻¹_θ_est = rowgram_operator(dλ_dθ' * F_λ) + 𝟙

    # The generalized Cholesky factor of the Fisher information may be
    # rectangular, the metric noise lives in its column space:
    sample_n = randn(genctx, size(F_λ, 2))
    sample_eta = randn(genctx, n_θ)
    Δφ = dλ_dθ' * (F_λ * sample_n) + sample_eta

    prob = LinearProblem{false}(Σ⁻¹_θ_est, Δφ)
    # LinearSolve's default maxiters (== n_θ) leaves iterative solvers
    # unconverged in floating point on ill-conditioned systems:
    sol = solve(prob, s.linear_solver; maxiters = 4 * n_θ, s.linear_solver_opts...)
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


"""
    MGVI._batched_cg(
        apply_A, B::AbstractMatrix{<:Real}, max_iter::Integer, rtol::Real
    )

Solve `A * X == B` column-wise via conjugate-gradient iterations, with
`apply_A(P)` applying the positive-definite `A` to the columns of `P`.

Iterates until the residual norms of all columns have been reduced by the
factor `rtol`, but at most `max_iter` times. Free of array mutation and,
when compiled via program tracing, of scalar control flow (`max_iter`
unrolled cg iterations then, masked to no-ops for converged columns), so
it is suitable for compilation e.g. via Reactant, unlike dynamically
terminated solvers.
"""
function _batched_cg(apply_A, B::AbstractMatrix{<:RealLike}, max_iter::Integer, rtol::Real)
    tiny = eps(float(one(eltype(B))))
    X = zero(B)
    R = copy(B)
    P = copy(B)
    rs₀ = sum(abs2, B; dims = 1)
    rs = copy(rs₀)
    threshold = oftype(tiny, rtol^2)
    # ToDo: Use the early-terminating while loop in all cases once traced
    # loops support callables that carry traced state (like apply_A):
    if within_compile()
        for _ in 1:max_iter
            X_next, R_next, P_next, rs_next = _batched_cg_iteration(apply_A, X, R, P, rs, tiny)
            active = (rs ./ (rs₀ .+ tiny)) .> threshold
            X = ifelse.(active, X_next, X)
            R = ifelse.(active, R_next, R)
            P = ifelse.(active, P_next, P)
            rs = ifelse.(active, rs_next, rs)
        end
    else
        conv = one(tiny)
        i = 0
        while (i < max_iter) & (conv > threshold)
            X, R, P, rs = _batched_cg_iteration(apply_A, X, R, P, rs, tiny)
            conv = maximum(rs ./ (rs₀ .+ tiny))
            i += 1
        end
    end
    return X
end

function _batched_cg_iteration(apply_A, X, R, P, rs, tiny)
    AP = apply_A(P)
    α = rs ./ (sum(P .* AP; dims = 1) .+ tiny)
    X = X .+ P .* α
    R = R .- AP .* α
    rs_next = sum(abs2, R; dims = 1)
    β = rs_next ./ (rs .+ tiny)
    P = R .+ P .* β
    return X, R, P, rs_next
end


"""
    MGVI.sample_residuals(
        f_model, center::AbstractVector{<:Real},
        sample_n::AbstractMatrix{<:Real}, sample_η::AbstractMatrix{<:Real},
        ad::ADSelector;
        cg_max_iterations::Integer = 4 * length(center),
        cg_rtol::Real = sqrt(eps(Float64))
    )

Generate zero-centered residual samples like
`sample_residuals(::ResidualSampler, n)`, but as a pure function of
pre-drawn standard normal samples `sample_n` (data-parameter space, size
`m × k`) and `sample_η` (parameter space, size `n × k`), solving all `k`
linear systems together with [`MGVI._batched_cg`](@ref).

Returns an `n × k` matrix of residual samples.

Free of RNG state and array mutation, so suitable for program tracing and
compilation, e.g. via Reactant with an Enzyme-based `ad`.
"""
function sample_residuals(
    f_model, center::AbstractVector{<:RealLike},
    sample_n::AbstractMatrix{<:RealLike}, sample_η::AbstractMatrix{<:RealLike},
    ad::ADSelector;
    cg_max_iterations::Integer = 4 * length(center),
    cg_rtol::Real = sqrt(eps(Float64))
)
    f_flat = flat_params ∘ f_model
    _, J = with_jacobian(f_flat, center, MatrixShapedOperator, ad)
    ℐ_λ = fisher_information(f_model(center))
    L = rowgram_factor(ℐ_λ)
    Δφ = J' * (L * sample_n) .+ sample_η
    apply_M(P) = J' * (ℐ_λ * (J * P)) .+ P
    return _batched_cg(apply_M, Δφ, cg_max_iterations, cg_rtol)
end


function residual_pushfwd_operator(s::ResidualSampler{<:Any,<:AbstractVector{<:Real},<:MatrixInversion})
    genctx = s.context.gen

    ℐ_λ = s.λ_information
    dλ_dθ = s.jac_dλ_dθ
    n_θ = size(dλ_dθ, 2)

    Σ⁻¹_θ_est = rowgram_operator(dλ_dθ' * rowgram_factor(ℐ_λ)) + 𝟙
    Σ⁻¹_θ_est_matrix = allocate_array(genctx, (n_θ, n_θ))
    copyto!(Σ⁻¹_θ_est_matrix, AbstractMatrix(Σ⁻¹_θ_est))
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
