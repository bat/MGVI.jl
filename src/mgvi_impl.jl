# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

function _mv_normal_logdensity(x::AbstractVector{<:RealLike})
    T = eltype(x)
    r = - dot(x, x)/2 - T(length(x)) * T(log2π/2)
    return r
end

function posterior_loglike(model, p, data)
    likelihood_value = logdensityof(model(p), data)
    # standard multivariate normal prior:
    prior_density = _mv_normal_logdensity(p)
    return likelihood_value + prior_density
end

# Equivalent to -(ELBO - H(q)), can't calculate H(q) efficiently for MGVI,
# but it's constant, so we'll use the mean of the negative non-normalized
# log-posterior over the samples instead:
function _mean_neg_log_pstr(f::Function, data, residual_samples::AbstractMatrix{<:RealLike}, center::AbstractVector{<:RealLike})
    mnlp_contribution(residual::AbstractVector) = - (
        posterior_loglike(f, center + residual, data) +
        posterior_loglike(f, center - residual, data)
    )
    res = sum(i -> mnlp_contribution(residual_samples[:, i]), axes(residual_samples, 2))
    n = 2 * size(residual_samples, 2)
    kl = res / n
    return kl
end


"""
    struct MGVI.MGVIKLTarget <: Function

The optimization target of [`mgvi_step`](@ref): the sampled KL estimator
(mean negative non-normalized log-posterior over the samples) as a function
of the variational mean.

Construct with [`mgvi_kl_target`](@ref).
"""
struct MGVIKLTarget{F,D,S<:AbstractMatrix{<:RealLike}} <: Function
    f_model::F
    data::D
    residual_samples::S
end

function (t::MGVIKLTarget)(center::AbstractVector{<:RealLike})
    return _mean_neg_log_pstr(t.f_model, t.data, t.residual_samples, center)
end

"""
    mgvi_kl_target(forward_model, data, residual_samples::AbstractMatrix{<:Real})

Return the function of the variational mean that [`mgvi_step`](@ref)
minimizes, given zero-centered residual samples (as matrix columns).

The result is a pure function of its argument and suitable for program
tracing and compilation (e.g. via Reactant, with an Enzyme-based gradient),
provided `forward_model` is.
"""
function mgvi_kl_target(forward_model, data, residual_samples::AbstractMatrix{<:RealLike})
    return MGVIKLTarget(forward_model, data, residual_samples)
end
export mgvi_kl_target


"""
    struct MGVIConfig

MGVI algorithm configuration.

Fields:

* `linsolver`: Linear solver to use, must be suitable for positive-definite operators
* `linsolver_opts`: Linear solver options
* `optimizer`: Optimization solver to use
* `optimizer_opts`: Optimization solver options

`linsolver` must be a solver supported by
[`LinearSolve`](https://github.com/SciML/LinearSolve.jl) or
[`MGVI.MatrixInversion`](@ref). Use `MatrixInversion` only for low-dimensional
problems. `linsolver_opts` is passed to `LinearSolve.solve` as keyword
arguments.

`optimizer` may be [`MGVI.NewtonCG()`](@ref) or an optimization
algorithm supported by `OptimizationBase` or `Optim`. `optimizer_opts` is
algorithm-specific.
"""
@with_kw struct MGVIConfig{LS, LSO<:NamedTuple, OS, OP<:NamedTuple}
    linsolver::LS = KrylovJL_CG()
    linsolver_opts::LSO = (;)
    optimizer::OS = MGVI.NewtonCG()
    optimizer_opts::OP = (;)
end
export MGVIConfig


"""
    struct MGVIResult

State resulting from [`mgvi_step`](@ref).

Fields:

* `samples`: The samples drawn by MGVI
* `mnlp`: The mean of the negative non-normalized log-posterior over the samples
* `info`: Additional information given by the linear solver and optimization
  algorithm.
"""
struct MGVIResult{
    T<:RealLike, TM<:AbstractMatrix{T}, U<:RealLike,
    AUX<:NamedTuple
}
    samples::TM
    mnlp::U
    info::AUX
end


"""
    mgvi_step(
        forward_model, data, n_residuals::Integer, center_init::AbstractVector{<:Real},
        config::MGVIConfig, context::MGVIContext
    )

Performs one MGVI step and returns a tuple
`(result::MGVIResult, updated_center::AbstractVector{<:Real})`.

The posterior distribution is approximated with a multivariate normal distribution.
The covariance is approximated with the inverse Fisher information evaluated at
`center_init`. Samples are drawn according to this covariance, which are then
used to estimate and minimize the KL divergence between the true posterior and the
approximation.

Note: The prior is implicit, it is a standard (uncorrelated) multivariate
normal distribution of the same dimensionality as `center_init`.

# Example

```julia
using Random, Distributions, MGVI
import LinearSolve, Zygote

context = MGVIContext(ADSelector(Zygote))

model(x::AbstractVector) = Normal(x[1], 0.2)
true_param = [2.0]
data = rand(model(true_param), 1)[1]
center = [1.3]

config = MGVIConfig(
    linsolver = LinearSolve.KrylovJL_CG(),
    optimizer = MGVI.NewtonCG()
)
n_residuals = 12
n_steps = 5

res, center = mgvi_step(model, data, n_residuals, center, config, context)
for i in 1:n_steps-1
    res, center = mgvi_step(model, data, n_residuals, center, config, context)
end

samples_from_est_covariance = res.samples
```
"""
function mgvi_step(
    forward_model, data, n_residuals::Integer, center_init::AbstractVector{<:Real},
    config::MGVIConfig, context::MGVIContext
)
    n_residuals > 0 || throw(ArgumentError(
        "n_residuals must be positive, got $n_residuals"
    ))
    residual_sampler = ResidualSampler(forward_model, center_init, config.linsolver, context; linear_solver_opts = config.linsolver_opts)
    residual_samples = sample_residuals(residual_sampler, n_residuals)
    mnlp = mgvi_kl_target(forward_model, data, residual_samples)
    Σ⁻¹(ξ) = _inv_cov_est(forward_model, ξ, context, _jacobian_op_type(config.linsolver))
    # average the metric over both members of the antithetic sample pairs,
    # like the KL estimator does:
    Σ̅⁻¹(ξ) = mean(Σ⁻¹.(collect.(vcat(eachcol(ξ .+ residual_samples), eachcol(ξ .- residual_samples)))))
    center_updated, min_mnlp, optres = _optimize(mnlp, context.ad, Σ̅⁻¹, center_init, config.optimizer, config.optimizer_opts)
    samples = _build_samples(residual_samples, center_updated)
    info = (linsolver_output = nothing, optimizer_output = optres)
    result = MGVIResult(samples, min_mnlp, info)
    return result, oftype(center_init, center_updated)
end
export mgvi_step

# The posterior-covariance-inverse estimate J'ℐJ + I as a row-Gram
# operator plus identity, positive definite by construction:
function _inv_cov_est(fwd_model::Function, ξ::AbstractVector, context::MGVIContext, JOP::Type = MatrixShapedOperator)
    ℐ_λ, dλ_dξ = _fisher_information_and_jac(fwd_model, ξ, context, JOP)
    rowgram_operator(dλ_dξ' * rowgram_factor(ℐ_λ)) + 𝟙
end

function _build_samples(residual_samples::AbstractMatrix{<:Real}, center::AbstractVector{<:Real})
    a = center .+ residual_samples
    b = center .- residual_samples
    reshape(permutedims(stack([a, b]), (1,3,2)), (size(a, 1), size(a,2) + size(b,2)))
end


"""
    mgvi_sample(
        forward_model, data, n_residuals::Integer, center::AbstractVector{<:Real},
        config::MGVIConfig, context::MGVIContext
    )

Draws samples from the MGVI posterior approximation centered at `center`
without performing an optimization step.

Returns a matrix whose `2 * n_residuals` columns are the samples
`center ± residual`.
"""
function mgvi_sample(
    forward_model, data, n_residuals::Integer, center::AbstractVector{<:Real},
    config::MGVIConfig, context::MGVIContext
)
    n_residuals > 0 || throw(ArgumentError(
        "n_residuals must be positive, got $n_residuals"
    ))
    residual_sampler = ResidualSampler(forward_model, center, config.linsolver, context; linear_solver_opts = config.linsolver_opts)
    residual_samples = sample_residuals(residual_sampler, n_residuals)
    smpls = _build_samples(residual_samples, center)
    return smpls
end
export mgvi_sample


"""
    mgvi_mvnormal_pushfwd_function(
        forward_model, data, config::MGVIConfig,
        center_point::AbstractVector{<:Real}, context::MGVIContext
    )

Returns a function that pushes a multivariate normal distribution forward
to the MGVI posterior approximation.

This currently instantiates the full Jacobian of the forward model as
a matrix in memory, and so should not be used for very high-dimensional
problems.
"""
function mgvi_mvnormal_pushfwd_function(
    forward_model, data, config::MGVIConfig,
    center_point::AbstractVector{<:Real}, context::MGVIContext
)
    residual_sampler = ResidualSampler(forward_model, center_point, MatrixInversion(), context)
    op = residual_pushfwd_operator(residual_sampler)
    MulAdd(op, center_point)
end
export mgvi_mvnormal_pushfwd_function
