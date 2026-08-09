# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


"""
    pareto_diagnostic(log_ratios::AbstractVector{<:Real})

Run Pareto-smoothed importance sampling (PSIS) on unnormalized log
importance ratios `log p - log q`.

Returns `(; pareto_shape, weights, result)` with the Pareto shape `k̂`,
the normalized Pareto-smoothed importance weights and the full
`PSIS.PSISResult`. `k̂` diagnoses how well the variational distribution
`q` covers the posterior `p`: `k̂ < 0.5` is good, values up to `0.7` are
usable, higher values indicate that `q` misses posterior mass.
"""
function pareto_diagnostic(log_ratios::AbstractVector{<:Real})
    result = PSIS.psis(collect(float.(log_ratios)); warn = false)
    return (; pareto_shape = result.pareto_shape, weights = result.weights, result)
end
export pareto_diagnostic


"""
    pareto_diagnostic(
        forward_model, data, samples::AbstractMatrix{<:Real},
        center::AbstractVector{<:Real}, context::MGVIContext
    )

PSIS diagnostic of the metric-Gaussian posterior approximation
`q = N(center, (J'ℐJ + 𝟙)⁻¹)` for posterior `samples` (as matrix
columns) drawn around `center`, as returned by [`mgvi_step`](@ref).

The log-density of `q` enters only up to its sample-independent
normalization, which cancels in the diagnostic and in the
self-normalized weights. Exact for MGVI samples of a converged fit;
for geoVI samples the per-sample nonlinear transport is ignored,
making the diagnostic approximate.
"""
function pareto_diagnostic(
    forward_model, data, samples::AbstractMatrix{<:Real},
    center::AbstractVector{<:Real}, context::MGVIContext
)
    apply_M = _posterior_metric_fn(forward_model, center, context.ad)
    log_ratios = map(eachcol(samples)) do ξ
        δ = ξ - center
        log_q = -dot(δ, apply_M(δ)) / 2
        posterior_loglike(forward_model, ξ, data) - log_q
    end
    return pareto_diagnostic(log_ratios)
end


# Posterior metric M = J'ℐJ + I at center, as a matrix-free apply function:
function _posterior_metric_fn(f_model, center::AbstractVector{<:Real}, ad::ADSelector)
    f_flat = flat_params ∘ f_model
    _, J = with_jacobian(f_flat, center, MatrixShapedOperator, ad)
    ℐ = fisher_information(f_model(center))
    apply_M(v::AbstractVector) = J' * (ℐ * (J * v)) .+ v
    return apply_M
end
