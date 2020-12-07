# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

const rs_default_options=NamedTuple()
const optim_default_options = Optim.Options()
const optim_default_solver = LBFGS()

function _create_residual_sampler(f::Function, center_p::Vector;
                                  residual_sampler::Type{RS}=ImplicitResidualSampler,
                                  jacobian_func::Type{JF}=FwdDerJacobianFunc,
                                  residual_sampler_options::NamedTuple
                                 ) where RS <: AbstractResidualSampler where JF <: AbstractJacobianFunc
    fisher, jac = fisher_information_and_jac(f, center_p; jacobian_func=jacobian_func)
    residual_sampler(fisher, jac; residual_sampler_options...)
end

function mgvi_kl(f::Function, data, residual_samples::AbstractMatrix{<:Real}, center_p::AbstractVector{<:Real})
    kl_component(p::AbstractVector) = -logpdf(f(p), data) + dot(p, p)/2
    kl_comp_both(rs::AbstractVector) = kl_component(center_p + rs) + kl_component(center_p - rs)
    res = 0
    for rs in eachcol(residual_samples)
        res += kl_comp_both(rs)
    end
    res/size(residual_samples, 2)/2
end

function _gradient_for_optim(kl::Function)
    (res::AbstractVector, x::AbstractVector) -> begin
        res .= Zygote.gradient(kl, x)[1]
    end
end

"""
    mgvi_kl_optimize_step(rng, forward_model, data, init_param_point;
                          jacobian_func=jacobian_func,
                          residual_sampler=residual_sampler,
                          [num_residuals=3,]
                          [residual_sampler_options=NamedTuple(),]
                          [optim_solver=LBFGS(),]
                          [optim_options=Optim.Options()])

Make one step of MGVI iterative procedure

We approximate posterior distribution with a multivariate normal. Estimate covariance of it around
`init_param_point` with inverse fisher information, then build next guess of model params
by choosing `param_point` that minimizes the KL divergence between the estimated
multivariate normal and the `forward_model`.

# Arguments

* `rng::AbstractRNG`: instance of the random number generator
* `forward_model::Function`: gets model parameters and returns instance of Distribution
* `data::AbstractVector`: data on which model's pdf is evaluated
* `init_param_point::Vector`: best guess for model parameters from which to start iterative optimization procedure
* `jacobian_func::Type{<:AbstractJacobianFunc}`: type of the calculator to use for finding jacobian
  matrix for the transition from canonical to model parametrization of the distributions
* `residual_sampler::Type{<:AbstractResidualSampler}`: type of sampler to use for sampling from
  estimated covariance of the forward model
* `num_residuals::Integer = 3`: number of samples to average KL divergence on
* `residual_sampler_options::NamedTuple = NamedTuple()`: options to pass to the residual sampler. Useful option
  is `cg_params` that is passed to CG solver used inside of ImplicitResidualSampler
* `optim_solver::Optim.AbstractOptimizer = LBFGS()`: optimizer used for minimizing KL divergence
* `optim_options::Optim.Options = Optim.Options()`: options to pass to the KL optimizer

# Example

```julia
using Random, Distributions, MGVI

model(x::AbstractVector) = Normal(x[1], 0.2)
true_param = [2.0]
data = rand(model(true_param), 1)[1]
init_param = [1.3]

res = mgvi_kl_optimize_step(Random.GlobalRNG, model, data, init_param;
                            jacobian_func=FwdRevADJacobianFunc,
                            residual_sampler=ImplicitResidualSampler,
                            num_residuals=5,
                            residual_sampler_options=(;cg_params=(;maxiter=10, verbose=true))),
                            optim_solver=LBFGS(;m=5),
                            optim_options=Optim.Options(iterations=7, show_trace=true))

next_param_point = res.result

optim_optimized_object = res.optimized
Optim.summary(optim_optimized_object)

samples_from_est_covariance = res.samples
```

# See also

* Residual samplers: [`AbstractResidualSampler`](@ref), [`ImplicitResidualSampler`](@ref), [`FullResidualSampler`](@ref)
* Jacobian functions: [`AbstractJacobianFunc`](@ref), [`FwdRevADJacobianFunc`](@ref), [`FwdDerJacobianFunc`](@ref)

"""
function mgvi_kl_optimize_step(rng::AbstractRNG,
                               forward_model::Function, data, init_param_point::AbstractVector;
                               jacobian_func::Type{JF},
                               residual_sampler::Type{RS},
                               num_residuals::Integer=3,
                               residual_sampler_options::NamedTuple=rs_default_options,
                               optim_solver::Optim.AbstractOptimizer=optim_default_solver,
                               optim_options::Optim.Options=optim_default_options
                              ) where RS <: AbstractResidualSampler where JF <: AbstractJacobianFunc
    est_res_sampler = _create_residual_sampler(forward_model, init_param_point;
                                               residual_sampler=residual_sampler,
                                               jacobian_func=jacobian_func,
                                               residual_sampler_options=residual_sampler_options)
    residual_samples = rand(rng, est_res_sampler, num_residuals)
    kl(params::AbstractVector) = mgvi_kl(forward_model, data, residual_samples, params)
    mgvi_kl_grad! =  _gradient_for_optim(kl)
    res = optimize(kl, mgvi_kl_grad!,
                   init_param_point, optim_solver, optim_options)
    updated_p = Optim.minimizer(res)

    (result=updated_p, optimized=res, samples=residual_samples .+ updated_p)
end

export mgvi_kl_optimize_step
