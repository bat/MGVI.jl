# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

const rs_default_options=(;)
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
    res = 0
    kl_component(p::AbstractVector) = -logpdf(f(p), data) + dot(p, p)/2
    for residual_sample in eachcol(residual_samples)
        res += kl_component(center_p + residual_sample) + kl_component(center_p - residual_sample)
    end
    res/size(residual_samples, 2)/2
end

_fill_grad(f::Function, grad_f::Function) = function (res::AbstractVector, x::AbstractVector)
end

function _gradient_for_optim(kl::Function)
    (res::AbstractVector, x::AbstractVector) -> begin
        res[:] = Zygote.gradient(kl, x)[1]
    end
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
    est_res_sampler = _create_residual_sampler(f, center_p;
                                               residual_sampler=residual_sampler,
                                               jacobian_func=jacobian_func,
                                               residual_sampler_options=residual_sampler_options)
    residual_samples = rand(rng, est_res_sampler, num_residuals)
    kl(params::Vector) = mgvi_kl(f, data, residual_samples, params)
    mgvi_kl_grad! =  _gradient_for_optim(kl)
    res = optimize(kl, mgvi_kl_grad!,
                   center_p, optim_solver, optim_options)
    updated_p = Optim.minimizer(res)

    (result=updated_p, optimized=res, samples=residual_samples .+ updated_p)
end

export mgvi_kl_optimize_step
