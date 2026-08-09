# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


# Mean posterior-covariance-inverse estimate over the samples (the
# antithetic pairs for MGVI, the sample columns themselves for geoVI), as
# a nested function ξ -> (v -> Σ̅⁻¹(ξ) * v). Equivalent to the operator
# curvature used by mgvi_step(::MGVIConfig), expressed via closures so
# it stays suitable for program tracing:
function _mean_fisher_curvature(f_model, ad::ADSelector, residual_samples::AbstractMatrix{<:RealLike}, signs::Tuple)
    f_flat = flat_params ∘ f_model
    function curvature(ξ::AbstractVector)
        ops = [
            begin
                ξᵢ = ξ + s * residual_samples[:, i]
                _, Jᵢ = with_jacobian(f_flat, ξᵢ, MatrixShapedOperator, ad)
                ℐᵢ = fisher_information(f_model(ξᵢ))
                (Jᵢ, ℐᵢ)
            end
            for i in axes(residual_samples, 2), s in signs
        ]
        function apply_curvature(v::AbstractVector)
            acc = sum(Jᵢ' * (ℐᵢ * (Jᵢ * v)) for (Jᵢ, ℐᵢ) in ops)
            return acc ./ length(ops) .+ v
        end
        return apply_curvature
    end
    return curvature
end


# The complete math of one MGVI step as a pure function of the current
# center and pre-drawn standard normal samples:
struct _MGVIStepFn{F,D,AD<:ADSelector,OPT<:NewtonCG} <: Function
    forward_model::F
    data::D
    ad::AD
    optimizer::OPT
    cg_max_iterations::Int
    cg_rtol::Float64
end

# Adapt's generic closure rule does not support callable structs:
Adapt.adapt_structure(to, s::_MGVIStepFn) = _MGVIStepFn(
    Adapt.adapt(to, s.forward_model), Adapt.adapt(to, s.data),
    s.ad, s.optimizer, s.cg_max_iterations, s.cg_rtol
)

function (s::_MGVIStepFn)(
    center::AbstractVector{<:RealLike},
    sample_n::AbstractMatrix{<:RealLike}, sample_η::AbstractMatrix{<:RealLike}
)
    residual_samples = sample_residuals(
        s.forward_model, center, sample_n, sample_η, s.ad;
        cg_max_iterations = s.cg_max_iterations, cg_rtol = s.cg_rtol
    )
    mnlp = mgvi_kl_target(s.forward_model, s.data, residual_samples)
    ∇mnlp = gradient_func(mnlp, s.ad, center)
    Σ̅⁻¹ = _mean_fisher_curvature(s.forward_model, s.ad, residual_samples, (+1, -1))
    center_updated, min_mnlp, _ = _newtoncg_optimize(mnlp, ∇mnlp, Σ̅⁻¹, center, s.optimizer, (;))
    return center_updated, min_mnlp, residual_samples
end


"""
    struct MGVIPreparedStep

An MGVI step prepared via [`mgvi_prepare`](@ref) for a fixed model, data,
number of residual samples and parameter dimensionality.

Run steps with `mgvi_step(prepared::MGVIPreparedStep, center)`.
"""
struct MGVIPreparedStep{SF,CTX<:MGVIContext}
    step_fn::SF
    n_z::Int
    n_θ::Int
    n_residuals::Int
    context::CTX
end
export MGVIPreparedStep


"""
    mgvi_prepare(
        forward_model, data, n_residuals::Integer,
        center_proto::AbstractVector{<:Real},
        config::MGVIConfig, context::MGVIContext;
        device = nothing
    )

Prepare repeated MGVI steps for the given model and data and return a
[`MGVIPreparedStep`](@ref).

`center_proto` provides the shape and type of the (later) center points,
its values are not used. The number of residual samples and the parameter
dimensionality are fixed at preparation time.

If `device` (an `MLDataDevices.AbstractDevice`) is given, the step
computation is set up on the device via `HeterogeneousComputing.on_device`.
For a Reactant device it is compiled once and then reused for every
subsequent step, which requires `config.optimizer` to be a
[`NewtonCG`](@ref) with a [`MGVI.BacktrackingLineSearch`](@ref)
linesearcher and an Enzyme-based `context.ad`.

The prepared step uses [`MGVI._batched_cg`](@ref) for residual sampling
instead of `config.linsolver`; `maxiters` and `reltol` in
`config.linsolver_opts` are honored.
"""
function mgvi_prepare(
    forward_model, data, n_residuals::Integer, center_proto::AbstractVector{<:Real},
    config::MGVIConfig, context::MGVIContext;
    device = nothing
)
    n_residuals > 0 || throw(ArgumentError(
        "n_residuals must be positive, got $n_residuals"
    ))
    optimizer = config.optimizer
    optimizer isa NewtonCG || throw(ArgumentError(
        "mgvi_prepare requires config.optimizer to be a MGVI.NewtonCG"
    ))
    if nameof(typeof(device)) == :ReactantDevice && !(optimizer.linesearcher isa BacktrackingLineSearch)
        throw(ArgumentError(
            "Reactant-compiled MGVI steps require a NewtonCG optimizer with a MGVI.BacktrackingLineSearch linesearcher"
        ))
    end

    n_θ = length(center_proto)
    # The metric noise lives in the column space of the (possibly
    # rectangular) generalized Cholesky factor of the Fisher information:
    n_z = size(rowgram_factor(fisher_information(forward_model(center_proto))), 2)
    cg_max_iterations = Int(get(config.linsolver_opts, :maxiters, 4 * n_θ))
    cg_rtol = Float64(get(config.linsolver_opts, :reltol, sqrt(eps(Float64))))

    step_fn = _MGVIStepFn(forward_model, data, context.ad, optimizer, cg_max_iterations, cg_rtol)

    genctx = context.gen
    dummy_center = collect(center_proto)
    dummy_n = randn(genctx, n_z, n_residuals)
    dummy_η = randn(genctx, n_θ, n_residuals)
    prepared_fn = _maybe_on_device(step_fn, device, dummy_center, dummy_n, dummy_η)

    return MGVIPreparedStep(prepared_fn, n_z, n_θ, Int(n_residuals), context)
end
export mgvi_prepare

_maybe_on_device(f, ::Nothing, dummy_args...) = f
_maybe_on_device(f, device, dummy_args...) = on_device(f, device, dummy_args...)


"""
    mgvi_step(prepared::MGVIPreparedStep, center::AbstractVector{<:Real})

Performs one MGVI step like
`mgvi_step(forward_model, data, n_residuals, center, config, context)`,
using a step prepared with [`mgvi_prepare`](@ref).

Returns a tuple `(result::MGVIResult, updated_center)`.
"""
function mgvi_step(prepared::MGVIPreparedStep, center::AbstractVector{<:Real})
    genctx = prepared.context.gen
    sample_n = randn(genctx, prepared.n_z, prepared.n_residuals)
    sample_η = randn(genctx, prepared.n_θ, prepared.n_residuals)
    center_updated, min_mnlp, residual_samples = prepared.step_fn(center, sample_n, sample_η)
    samples = _build_samples(residual_samples, center_updated)
    info = (linsolver_output = nothing, optimizer_output = nothing)
    result = MGVIResult(samples, min_mnlp, info)
    return result, oftype(center, center_updated)
end
