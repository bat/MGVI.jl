# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

function mgvi_kl(f::Function, data, residual_samples::Array, center_p)
    res = 0.
    for residual_sample in eachcol(residual_samples)
        p = center_p + residual_sample
        res += -logpdf(f(p), data) + dot(p, p)/2
    end
    res/size(residual_samples, 2)
end

function mgvi_kl_optimize_step(f::Function, data, center_p::Vector; num_residuals=15)
    estimated_dist = mgvi_residual_sampler(f, center_p)
    residual_samples = rand(estimated_dist, num_residuals)
    residual_samples = hcat(residual_samples, -residual_samples)
    res = optimize(params -> mgvi_kl(f, data, residual_samples, params),
                   center_p, LBFGS(); autodiff=:forward)
    updated_p = Optim.minimizer(res)
    updated_p
end

export mgvi_kl_optimize_step
