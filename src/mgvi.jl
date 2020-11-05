# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

function _get_residual_sampler(f::Function, center_p::Vector;
                               residual_sampler::Type{RS}=ImplicitResidualSampler,
                               jacobian_func::Type{JF}=FwdDerJacobianFunc) where RS <: AbstractResidualSampler where JF <: AbstractJacobianFunc
    fisher_map, jac_map = fisher_information_components(f, center_p; jacobian_func=jacobian_func)
    residual_sampler(fisher_map, jac_map)
end

function mgvi_kl(f::Function, data, residual_samples::Array, center_p)
    res = 0.
    for residual_sample in eachcol(residual_samples)
        p = center_p + residual_sample
        res += -logpdf(f(p), data) + dot(p, p)/2
    end
    res/size(residual_samples, 2)
end

function mgvi_kl_optimize_step(f::Function, data, center_p::Vector;
                               num_residuals=15,
                               residual_sampler::Type{RS}=ImplicitResidualSampler,
                               jacobian_func::Type{JF}=FwdDerJacobianFunc) where RS <: AbstractResidualSampler where JF <: AbstractJacobianFunc
    estimated_dist = _get_residual_sampler(f, center_p; residual_sampler=residual_sampler, jacobian_func=jacobian_func)
    residual_samples = rand(estimated_dist, num_residuals)
    residual_samples = hcat(residual_samples, -residual_samples)
    res = optimize(params -> mgvi_kl(f, data, residual_samples, params),
                   center_p, LBFGS(); autodiff=:forward)
    updated_p = Optim.minimizer(res)
    updated_p
end

export mgvi_kl_optimize_step
