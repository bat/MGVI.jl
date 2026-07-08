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
struct ResidualSampler{F,RV<:AbstractVector{<:Real},SLV,SLO<:NamedTuple,OPL<:LinearMap,OPJ<:LinearMap,CTX<:MGVIContext}
    f_model::F
    center_point::RV
    linear_solver::SLV
    linear_solver_opts::SLO
    λ_information::OPL
    jac_dλ_dθ::OPJ
    context::CTX
end
export ResidualSampler


@inline _get_operator_type(::MatrixInversion) = DenseMatrix
@inline _get_operator_type(::Any) = LinearMap

function ResidualSampler(
    f_model::Function, center_point::Vector{<:Real}, linear_solver, context::MGVIContext;
    linear_solver_opts::NamedTuple = (;)
)
    OP = _get_operator_type(linear_solver)
    ℐ_λ, dλ_dξ = _fisher_information_and_jac(f_model, center_point, OP, context)
    ResidualSampler(f_model, center_point, linear_solver, linear_solver_opts, convert(LinearMap, ℐ_λ), convert(LinearMap, dλ_dξ), context)
end


function sample_residuals(s::ResidualSampler{<:Any,<:AbstractVector{<:Real},<:Any})
    genctx = s.context.gen

    ℐ_λ = s.λ_information
    dλ_dθ = s.jac_dλ_dθ
    n_λ, n_θ = size(dλ_dθ)
    Σ⁻¹_θ_est = dλ_dθ' * ℐ_λ * dλ_dθ + I

    sample_n = randn(genctx, n_λ)
    sample_eta = randn(genctx, n_θ)
    Δφ = dλ_dθ' * (cholesky_L(ℐ_λ) * sample_n) + sample_eta

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
via `ReactantCore.@trace`, of scalar control flow, so it is suitable for
program tracing and compilation (e.g. via Reactant), unlike dynamically
terminated solvers.
"""
function _batched_cg(apply_A, B::AbstractMatrix{<:Real}, max_iter::Integer, rtol::Real)
    tiny = eps(float(one(eltype(B))))
    X = zero(B)
    R = copy(B)
    P = copy(B)
    rs₀ = sum(abs2, B; dims = 1)
    rs = copy(rs₀)
    threshold = oftype(tiny, rtol^2)
    conv = one(tiny)
    i = 0
    @trace while (i < max_iter) & (conv > threshold)
        X, R, P, rs = _batched_cg_iteration(apply_A, X, R, P, rs, tiny)
        conv = maximum(rs ./ (rs₀ .+ tiny))
        i += 1
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
    f_model, center::AbstractVector{<:Real},
    sample_n::AbstractMatrix{<:Real}, sample_η::AbstractMatrix{<:Real},
    ad::ADSelector;
    cg_max_iterations::Integer = 4 * length(center),
    cg_rtol::Real = sqrt(eps(Float64))
)
    f_flat = flat_params ∘ f_model
    jvp = jvp_func(f_flat, center, ad)
    _, vjp = with_vjp_func(f_flat, center, ad)
    fi = fisher_information(f_model(center))
    ℐ_λ = _fisher_repr(fi)
    L = _fisher_repr(cholesky_L(fi))
    Δφ = _mapcols(vjp, _fisher_apply(L, sample_n)) .+ sample_η
    apply_M(P) = _mapcols(vjp, _fisher_apply(ℐ_λ, _mapcols(jvp, P))) .+ P
    return _batched_cg(apply_M, Δφ, cg_max_iterations, cg_rtol)
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
