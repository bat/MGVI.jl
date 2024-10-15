# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

function posterior_loglike(model, p, data)
    likelihood_value = logdensityof(model(p), data)
    # normal prior, leaving out `+ length(p)*log2π/2` normalization constant
    prior_density = - dot(p, p)/2
    return likelihood_value + prior_density
end

# Equivalent to -(ELBO - H(q)), can't calculate H(q) efficiently for MGVI,
# but it's constant, so we'll use the mean of the negative non-normalized
# log-posterior over the samples instead:
function _mean_neg_log_pstr(f::Function, data, residual_samples::AbstractMatrix{<:Real}, center::AbstractVector{<:Real})
    mnlp_contribution(residual::AbstractVector) = - (
        posterior_loglike(f, center + residual, data) +
        posterior_loglike(f, center - residual, data)
    )
    res = sum(mnlp_contribution, eachcol(residual_samples))
    n = 2 * size(residual_samples, 2)
    kl = res / n
    return kl
end


"""
    struct MVGIConfig

MGVI clgorithm configuration.

Fields:

* `linar_solver`: Linear solver to use, must be suitable for positive-definite operators
* `optimization_alg`: Optimization solver to use
* `optimization_opts`: Optimization solver options

`linear_solver` must be a solver supported by
[`LinearSolve`](https://github.com/SciML/LinearSolve.jl) or
[`MGVI.MatrixInversion`](@ref). Use `MatrixInversion` only for low-dimensional
problems.

`optimization_alg` nay be [`MGVI.NewtonCG()`](@ref) or an optimization
algorithm supported by `Optimization` or `Optim`. `optimization_opts` is
algorithm-specific.
"""
@with_kw struct MGVIConfig{LS, OS, OP<:NamedTuple}
    linear_solver::LS = KrylovJL_CG()
    optimization_alg::OS = MGVI.NewtonCG()
    optimization_opts::OP::NamedTuple = (;)
end
export MGVIConfig


"""
    struct MGVIResult

State resulting from [`mgvi_step`](@ref).

Fields:

* `smples`: The samples drawn by MVGI
* `mnlp`: The mean of the negative non-normalized log-posterior over the samples
"""
struct MGVIResult{
    T<:Real, TM<:AbstractMatrix{T}, TV<:AbstractMatrix{T}, S<:ResidualSampler,
    C<:MGVIConfig, CTX<:MGVIContext
}
    smpls::TM
    center::TV
    mnlp::U
    info::AUX
end


"""
    struct MGVIState

State resulting from [`mgvi_step`](@ref).

Fields:

* `center`: The center/mean of the current MGVI posterior approximation.
* `resisual_sampler`: The current [`ResidualSampler`](@ref).
* `config`: The current [`MGVIConfig`](@ref).
"""
struct MGVIState{
    T<:Real, TV<:AbstractVector{T}, S<:ResidualSampler,
    C<:MGVIConfig, CTX<:MGVIContext
}
    center::TV
    resisual_sampler::S
    config::C
    context::CTX
end


"""
    mgvi_step(
        forward_model::Function, data, center_init::AbstractVector{<:Real},
        num_residuals::Integer, context::MGVIContext;
        linear_solver = KrylovJL_CG(),
        optimization_alg = MGVI.NewtonCG(), optimization_opts::NamedTuple = (;)
    )

    mgvi_step(
        forward_model::Function, data, state::MGVIState, context::MGVIContext;
        linear_solver = KrylovJL_CG(),
        optimization_alg = MGVI.NewtonCG(), optimization_opts::NamedTuple = (;)
    )

Performs one MGVI step and returns an [`MGVIState`](@ref) object.

The posterior distribution is approximated with a multivariate normal distribution.
The covariance is approximated with the inverse Fisher information valuated at
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
init_param = [1.3]

res = mgvi_step(
    model, data, init_param, context;
    num_residuals = 5,
    linear_solver = LinearSolve.KrylovJL_CG(),
    optimization_alg = MGVI.NewtonCG(),
)

next_param_point = res.center

optim_optimized_object = res.info
Optim.summary(optim_optimized_object)

samples_from_est_covariance = res.samples
```
"""
function mgvi_step end
export mgvi_step

function mgvi_step(
    forward_model::Function, data, center_init::AbstractVector{<:Real},
    num_residuals::Integer, context::MGVIContext;
    linear_solver = KrylovJL_CG(),
    optimization_alg = MGVI.NewtonCG(), optimization_opts::NamedTuple = (;)
)
    residual_sampler = ResidualSampler(forward_model, center_init, linear_solver, context)
    residual_samples = sample_residuals(residual_sampler, num_residuals)
    mnlp(params::AbstractVector) = _mean_neg_log_pstr(forward_model, data, residual_samples, params)
    OP = _get_operator_type(linear_solver)
    Σ⁻¹(ξ) = _inv_cov_est(forward_model, ξ, OP, context)
    Σ̅⁻¹(ξ) = mean(Σ⁻¹.(collect.(eachcol(ξ .+ residual_samples))))
    center_updated, min_mnlp, optres = _optimize(mnlp, context.ad, Σ̅⁻¹, center_init, optimization_alg, optimization_opts)
    smpls = hcat(center_updated .+ residual_samples, center_updated .- residual_samples)

    return (samples = smpls, mnlp = min_mnlp, optres = optres), MGVIState(center, residual_sampler, smpls, )
end

function mgvi_step(
    forward_model::Function, data, state::MGVIState, context::MGVIContext;
    linear_solver = KrylovJL_CG(),
    optimization_alg = MGVI.NewtonCG(), optimization_opts::NamedTuple = (;)
)
    new_state = mgvi_step(
        forward_model, data, state.center, size(state.samples, 2) ÷ 2, context;
        linear_solver = linear_solver, optimization_alg = optimization_alg, optimization_opts = optimization_opts
    )
    return new_state
end


function _inv_cov_est(fwd_model::Function, ξ::AbstractVector, OP, context::MGVIContext)
    ℐ_λ, dλ_dξ = _fisher_information_and_jac(fwd_model, ξ, OP, context)
    dλ_dξ' * ℐ_λ * dλ_dξ + I
end
