# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

rs_default_options=(;)
optim_default_options = Optim.Options()
optim_default_solver = LBFGS()

function _get_residual_sampler(f::Function, center_p::Vector;
                               residual_sampler::Type{RS}=ImplicitResidualSampler,
                               jacobian_func::Type{JF}=FwdDerJacobianFunc,
                               residual_sampler_options::NamedTuple
                              ) where RS <: AbstractResidualSampler where JF <: AbstractJacobianFunc
    fisher_map, jac_map = fisher_information_components(f, center_p; jacobian_func=jacobian_func)
    residual_sampler(fisher_map, jac_map; residual_sampler_options...)
end

function mgvi_kl(f::Function, data, residual_samples::Array, center_p)
    res = 0.
    for residual_sample in eachcol(residual_samples)
        p = center_p + residual_sample
        res += -logpdf(f(p), data) + dot(p, p)/2
    end
    res/size(residual_samples, 2)
end

_fill_grad(f::Function, grad_f::Function) = function (res::AbstractVector, x::AbstractVector)
    res[:] = grad_f(f, x)
end

function mgvi_kl_optimize_step(rng::AbstractRNG,
                               f::Function, data, center_p::Vector;
                               num_residuals=15,
                               residual_sampler::Type{RS},
                               jacobian_func::Type{JF},
                               residual_sampler_options::NamedTuple=rs_default_options,
                               optim_options::Optim.Options=optim_default_options,
                               optim_solver::Optim.AbstractOptimizer=optim_default_solver
                              ) where RS <: AbstractResidualSampler where JF <: AbstractJacobianFunc
    estimated_dist = _get_residual_sampler(f, center_p;
                                           residual_sampler=residual_sampler,
                                           jacobian_func=jacobian_func,
                                           residual_sampler_options=residual_sampler_options)
    residual_samples = rand(rng, estimated_dist, num_residuals)
    residual_samples = hcat(residual_samples, -residual_samples)
    mgvi_kl_simple(params::Vector) = mgvi_kl(f, data, residual_samples, params)
    mgvi_kl_grad! =  _fill_grad(mgvi_kl_simple, first ∘ gradient)
    res = optimize(mgvi_kl_simple, mgvi_kl_grad!,
                   center_p, optim_solver, optim_options)
    updated_p = Optim.minimizer(res)

    (result=updated_p, optimized=res, samples=residual_samples .+ updated_p)
end

export mgvi_kl_optimize_step
