# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


function posterior_loglike(model, p, data)
    logpdf(model(p), data) - dot(p, p)/2
end

function mgvi_kl(f::Function, data, residual_samples::AbstractMatrix{<:Real}, center_p::AbstractVector{<:Real})
    kl_component(p::AbstractVector) = -posterior_loglike(f, p, data)
    kl_comp_both(rs::AbstractVector) = kl_component(center_p + rs) + kl_component(center_p - rs)
    res = 0
    for rs in eachcol(residual_samples)
        res += kl_comp_both(rs)
    end
    res/size(residual_samples, 2)/2
end


"""
    function mgvi_optimize_step(
        forward_model::Function, data, init_param_point::AbstractVector{<:Real}, context::MGVIContext;
        num_residuals::Integer = 3,
        lcenter_pointinear_solver::LinearSolverAlg = IterativeSolversCG(),
        optim_solver::Union{Optim.AbstractOptimizer,MGVI.NewtonCG} = MGVI.NewtonCG(),
        optim_options::Union{Optim.Options, Nothing} = Optim.Options(),
    )

Performs one MGVI iteration.

The posterior distribution is approximated with a multivariate normal distribution.
The covariance is approximated with the inverse Fisher information valuated at
`init_param_point`. Samples are drawn according to this covariance, which are then
used to estimate and minimize the KL divergence between the true posterior and the
approximation.

Note: The prior is implicit, it is a standard (uncorrelated) multivariate
normal distribution of the same dimensionality as `init_param_point`.

# Example

```julia
using Random, Distributions, MGVI
import Zygote

context = MGVIContext(ADModule(:Zygote))

model(x::AbstractVector) = Normal(x[1], 0.2)
true_param = [2.0]
data = rand(model(true_param), 1)[1]
init_param = [1.3]

res = mgvi_optimize_step(
    model, data, init_param, context;
    num_residuals = 5,
    linear_solver = MGVI.IterativeSolversCG(),
    optim_solver = MGVI.NewtonCG(),
)

next_param_point = res.result

optim_optimized_object = res.optimized
Optim.summary(optim_optimized_object)

samples_from_est_covariance = res.samples
```
"""
function mgvi_optimize_step end
export mgvi_optimize_step

function mgvi_optimize_step(
    forward_model::Function, data, init_point::AbstractVector{<:Real}, context::MGVIContext;
    num_residuals::Integer = 3,
    linear_solver::LinearSolverAlg = IterativeSolversCG(),
    optim_solver::Union{Optim.AbstractOptimizer,NewtonCG} = MGVI.NewtonCG(),
    optim_options::Union{Optim.Options, Nothing} = Optim.Options(),
)
    res_sampler = ResidualSampler(forward_model, init_point, linear_solver, context)
    residual_samples = sample_residuals(res_sampler, num_residuals)
    kl(params::AbstractVector) = mgvi_kl(forward_model, data, residual_samples, params)
    ∇kl! = gradient!_func(kl, context.ad)
    OP = _get_operator_type(linear_solver)
    Σ⁻¹(ξ) = _inv_cov_est(forward_model, ξ, OP, context)
    Σ̅⁻¹(ξ) = mean(Σ⁻¹.(collect.(eachcol(ξ .+ residual_samples))))
    res = _optimize(
        kl, ∇kl!, Σ̅⁻¹, init_point, optim_solver, optim_options)
    updated_point = res.minimizer

    (result=updated_point, optimized=res, samples=hcat(updated_point .+ residual_samples, updated_point .- residual_samples))
end

function _inv_cov_est(fwd_model::Function, ξ::AbstractVector, OP, context::MGVIContext)
    ℐ_λ, dλ_dξ = _fisher_information_and_jac(fwd_model, ξ, OP, context)
    dλ_dξ' * ℐ_λ * dλ_dξ + I
end
