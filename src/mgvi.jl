# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

function mgvi_kl(f::Function, data, residual_samples::Array, center_p)
    res = 0.
    for residual_sample in eachcol(residual_samples)
        p = center_p + residual_sample
        res += -sum(map(d -> logpdf(f(p), d), data)) + sum(p .* p)/2
    end
    res/size(residual_samples, 2)
end

function mgvi_kl_optimize_step(f::Function, data, center_p; num_residuals=15)
    estimated_covariance = inv(model_fisher_information(f, center_p) + I)
    estimated_dist = MvNormal(zeros(Float64, size(center_p, 1)), estimated_covariance)
    residual_samples = rand(estimated_dist, num_residuals)
    residual_samples = hcat(residual_samples, -residual_samples)
    res = optimize(params -> mgvi_kl(f, data, residual_samples, params),
                   center_p, LBFGS(); autodiff=:forward)
    updated_p = Optim.minimizer(res)
    updated_p
end

export mgvi_kl_optimize_step
