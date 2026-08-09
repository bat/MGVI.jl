# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


"""
    struct GeoVIConfig

geoVI algorithm configuration.

Fields:

* `linsolver`: Linear solver to use, must be suitable for positive-definite operators
* `linsolver_opts`: Linear solver options
* `optimizer`: Optimization solver for the KL minimization
* `optimizer_opts`: Optimization solver options
* `sampling_optimizer`: [`MGVI.NewtonCG`](@ref) used for the nonlinear
  per-sample solves

`linsolver` must be a solver supported by
[`LinearSolve`](https://github.com/SciML/LinearSolve.jl), `linsolver_opts`
is passed to `LinearSolve.solve` as keyword arguments. `optimizer` and
`optimizer_opts` behave as in [`MGVIConfig`](@ref).
"""
@with_kw struct GeoVIConfig{LS, LSO<:NamedTuple, OS, OP<:NamedTuple}
    linsolver::LS = KrylovJL_CG()
    linsolver_opts::LSO = (;)
    optimizer::OS = MGVI.NewtonCG()
    optimizer_opts::OP = (;)
    sampling_optimizer::NewtonCG = MGVI.NewtonCG()
end
export GeoVIConfig


"""
    struct GeoVISampler

Generates zero-centered samples from the geoVI posterior approximation
around a given expansion point.

geoVI locally isometrizes the Fisher metric `M(ξ) = J'ℐJ + 𝟙` via the
embedding `emb(ξ) = (euclidean_coords(model(ξ)), ξ)`: with the embedding
Jacobian `C = ∂emb₁/∂ξ` frozen at the expansion point `ξ̄`, each sample
solves the nonlinear least-squares problem

```
min_ξ ½ ‖ g̃(ξ) - t ‖²,   g̃(ξ) = (ξ - ξ̄) + C'(emb₁(ξ) - emb₁(ξ̄))
```

with target `t ~ N(0, M(ξ̄))`, starting from the linear (MGVI) residual
`M(ξ̄)⁻¹ t`. For forward models that are linear in `ξ` this reduces
exactly to MGVI residual sampling.

Constructor:

```julia
GeoVISampler(
    f_model::Function, center_point::Vector{<:Real}, linear_solver,
    sampling_optimizer::MGVI.NewtonCG, context::MGVIContext;
    linear_solver_opts::NamedTuple = (;)
)
```

`linear_solver` must be a solver supported by
[`LinearSolve`](https://github.com/SciML/LinearSolve.jl).

Call `MGVI.sample_geovi_residuals(s::GeoVISampler, n::Integer)` to generate
`2n` samples (each target `t` is solved for both signs `±t`).
"""
struct GeoVISampler{
    F,RV<:AbstractVector{<:Real},SLV,SLO<:NamedTuple,
    XV<:AbstractVector{<:Real},OPJ<:MatrixShapedOperator,CTX<:MGVIContext
}
    f_model::F
    center_point::RV
    linear_solver::SLV
    linear_solver_opts::SLO
    euclidean_center::XV
    jac_dx_dθ::OPJ
    optimizer::NewtonCG
    context::CTX
end
export GeoVISampler

_euclidean_coords_flat(f_model) = ξ -> euclidean_coords(f_model(ξ))

function GeoVISampler(
    f_model::Function, center_point::Vector{<:Real}, linear_solver,
    sampling_optimizer::NewtonCG, context::MGVIContext;
    linear_solver_opts::NamedTuple = (;)
)
    x̄, C = with_jacobian(_euclidean_coords_flat(f_model), center_point, MatrixShapedOperator, context.ad)
    GeoVISampler(
        f_model, center_point, linear_solver, linear_solver_opts,
        x̄, C, sampling_optimizer, context
    )
end


# Draw t ~ N(0, M̄) and the corresponding linear (MGVI) residual M̄⁻¹t:
function _geovi_sample_pair(s::GeoVISampler)
    genctx = s.context.gen
    C = s.jac_dx_dθ
    n_x, n_θ = size(C)
    M̄ = colgram_operator(C) + 𝟙
    t = C' * randn(genctx, n_x) + randn(genctx, n_θ)
    prob = LinearProblem{false}(M̄, t)
    sol = solve(prob, s.linear_solver; maxiters = 4 * n_θ, s.linear_solver_opts...)
    return t, sol.u
end

# Solve the nonlinear geoVI sampling problem for one target:
function _geovi_residual(s::GeoVISampler, t::AbstractVector{<:Real}, Δξ_init::AbstractVector{<:Real})
    ξ̄ = s.center_point
    x̄ = s.euclidean_center
    C = s.jac_dx_dθ
    x_fn = _euclidean_coords_flat(s.f_model)
    adsel = s.context.ad

    g̃_residual(ξ) = (ξ - ξ̄) + C' * (x_fn(ξ) - x̄) - t

    function f(ξ::AbstractVector)
        r = g̃_residual(ξ)
        dot(r, r) / 2
    end

    function ∇f(ξ::AbstractVector)
        x_ξ, J_x = with_jacobian(x_fn, ξ, MatrixShapedOperator, adsel)
        r = (ξ - ξ̄) + C' * (x_ξ - x̄) - t
        r + J_x' * (C * r)
    end

    # Gauss-Newton metric G'G with G = ∂g̃/∂ξ = I + C'J_x(ξ):
    function curvature(ξ::AbstractVector)
        _, J_x = with_jacobian(x_fn, ξ, MatrixShapedOperator, adsel)
        G = C' * J_x + 𝟙
        colgram_operator(G)
    end

    ξ_res, _, _ = _newtoncg_optimize(f, ∇f, curvature, ξ̄ + Δξ_init, s.optimizer, (;))
    return ξ_res - ξ̄
end

"""
    MGVI.sample_geovi_residuals(s::GeoVISampler, n::Integer)

Generate `2n` zero-centered geoVI residual samples, as columns of a matrix.

Each of the `n` targets is solved for both signs, sharing the underlying
random draw, so columns `2i-1` and `2i` form an antithetic pair (only
approximately mirrored, due to the nonlinearity).
"""
function sample_geovi_residuals(s::GeoVISampler, n::Integer)
    pairs = [_geovi_sample_pair(s) for _ in 1:n]
    n_θ = size(s.jac_dx_dθ, 2)
    A = allocate_array(s.context.gen, (n_θ, 2*n))
    Base.Threads.@threads for i in 1:n
        t, Δξ = pairs[i]
        view(A, :, 2*i - 1) .= _geovi_residual(s, t, Δξ)
        view(A, :, 2*i) .= _geovi_residual(s, -t, -Δξ)
    end
    return A
end


# Explicit accumulation loop instead of sum over a column-index range, for
# AD- and program-tracing-compatibility:
function _mean_neg_log_pstr_cols(f::Function, data, residual_samples::AbstractMatrix{<:RealLike}, center::AbstractVector{<:RealLike})
    res = -posterior_loglike(f, center + residual_samples[:, 1], data)
    for i in 2:size(residual_samples, 2)
        res = res - posterior_loglike(f, center + residual_samples[:, i], data)
    end
    return res / size(residual_samples, 2)
end


"""
    geovi_step(
        forward_model, data, n_residuals::Integer, center_init::AbstractVector{<:Real},
        config::GeoVIConfig, context::MGVIContext
    )

Performs one geoVI step and returns a tuple
`(result::MGVIResult, updated_center::AbstractVector{<:Real})`.

Like [`mgvi_step`](@ref), but the samples are drawn from the geoVI
approximation: the posterior is approximated by pulling a standard normal
distribution back through a local isometry of the Fisher metric (see
[`GeoVISampler`](@ref)), instead of by a multivariate normal distribution.
This improves the approximation for forward models with significant
nonlinearity at the posterior's scale.

Requires `MGVI.euclidean_coords` to be defined for the distribution type
returned by `forward_model`.

Note: The prior is implicit, it is a standard (uncorrelated) multivariate
normal distribution of the same dimensionality as `center_init`.
"""
function geovi_step(
    forward_model, data, n_residuals::Integer, center_init::AbstractVector{<:Real},
    config::GeoVIConfig, context::MGVIContext
)
    n_residuals > 0 || throw(ArgumentError(
        "n_residuals must be positive, got $n_residuals"
    ))
    smplr = GeoVISampler(
        forward_model, center_init, config.linsolver, config.sampling_optimizer, context;
        linear_solver_opts = config.linsolver_opts
    )
    residual_samples = sample_geovi_residuals(smplr, n_residuals)
    mnlp(params::AbstractVector) = _mean_neg_log_pstr_cols(forward_model, data, residual_samples, params)
    Σ⁻¹(ξ) = _inv_cov_est(forward_model, ξ, context)
    Σ̅⁻¹(ξ) = mean(Σ⁻¹.(collect.(eachcol(ξ .+ residual_samples))))
    center_updated, min_mnlp, optres = _optimize(mnlp, context.ad, Σ̅⁻¹, center_init, config.optimizer, config.optimizer_opts)
    samples = center_updated .+ residual_samples
    info = (linsolver_output = nothing, optimizer_output = optres)
    result = MGVIResult(samples, min_mnlp, info)
    return result, oftype(center_init, center_updated)
end
export geovi_step


"""
    geovi_sample(
        forward_model, data, n_residuals::Integer, center::AbstractVector{<:Real},
        config::GeoVIConfig, context::MGVIContext
    )

Draws samples from the geoVI posterior approximation centered at `center`
without performing an optimization step.

Returns a matrix whose `2 * n_residuals` columns are the samples.
"""
function geovi_sample(
    forward_model, data, n_residuals::Integer, center::AbstractVector{<:Real},
    config::GeoVIConfig, context::MGVIContext
)
    n_residuals > 0 || throw(ArgumentError(
        "n_residuals must be positive, got $n_residuals"
    ))
    smplr = GeoVISampler(
        forward_model, center, config.linsolver, config.sampling_optimizer, context;
        linear_solver_opts = config.linsolver_opts
    )
    residual_samples = sample_geovi_residuals(smplr, n_residuals)
    return center .+ residual_samples
end
export geovi_sample


# Like _geovi_residual, but as a pure function of its operator and
# vector arguments, suitable for program tracing:
function _geovi_residual_pure(
    x_fn, ξ̄::AbstractVector{<:RealLike}, x̄::AbstractVector{<:RealLike}, C::MatrixShapedOperator,
    t::AbstractVector{<:RealLike}, Δξ_init::AbstractVector{<:RealLike},
    ad::ADSelector, optimizer::NewtonCG
)
    g̃_residual(ξ) = (ξ - ξ̄) + C' * (x_fn(ξ) - x̄) - t

    function f(ξ::AbstractVector)
        r = g̃_residual(ξ)
        dot(r, r) / 2
    end

    function ∇f(ξ::AbstractVector)
        x_ξ, vjp_ξ = with_vjp_func(x_fn, ξ, ad)
        r = (ξ - ξ̄) + C' * (x_ξ - x̄) - t
        r + vjp_ξ(C * r)
    end

    # Gauss-Newton metric G'G with G = ∂g̃/∂ξ = I + C'J_x(ξ):
    function curvature(ξ::AbstractVector)
        _, J_ξ = with_jacobian(x_fn, ξ, MatrixShapedOperator, ad)
        function apply_curvature(v::AbstractVector)
            Gv = v + C' * (J_ξ * v)
            Gv + J_ξ' * (C * Gv)
        end
        return apply_curvature
    end

    ξ_res, _, _ = _newtoncg_optimize(f, ∇f, curvature, ξ̄ + Δξ_init, optimizer, (;))
    return ξ_res - ξ̄
end

# Like sample_geovi_residuals, but as a pure function of pre-drawn standard
# normal samples. The solves loop is a traced loop under compilation, so
# the (large) nonlinear solve is only traced once instead of being
# unrolled over the samples:
function _geovi_residual_samples(
    f_model, center::AbstractVector{<:RealLike},
    sample_n::AbstractMatrix{<:RealLike}, sample_η::AbstractMatrix{<:RealLike},
    ad::ADSelector, optimizer::NewtonCG, cg_max_iterations::Integer, cg_rtol::Real
)
    x_fn = _euclidean_coords_flat(f_model)
    x̄, C = with_jacobian(x_fn, center, MatrixShapedOperator, ad)
    T = C' * sample_n .+ sample_η
    apply_M(P) = C' * (C * P) .+ P
    ΔΞ = _batched_cg(apply_M, T, cg_max_iterations, cg_rtol)
    n_smpls = size(T, 2)
    X = zero(hcat(ΔΞ, ΔΞ))
    # ToDo: Run the solves as a traced loop when compiled, once traced
    # loops support callables that carry traced state (like x_fn); the
    # loop is unrolled under program tracing for now:
    for j in 1:(2 * n_smpls)
        i = (j + 1) ÷ 2
        s = ifelse(isodd(j), 1.0, -1.0)
        r_j = _geovi_residual_pure(
            x_fn, center, x̄, C, s .* T[:, i], s .* ΔΞ[:, i], ad, optimizer
        )
        X[:, j] = r_j
    end
    return X
end

struct _GeoVIKLTarget{F,D,S<:AbstractMatrix{<:RealLike}} <: Function
    f_model::F
    data::D
    residual_samples::S
end

function (t::_GeoVIKLTarget)(center::AbstractVector{<:RealLike})
    return _mean_neg_log_pstr_cols(t.f_model, t.data, t.residual_samples, center)
end

# The complete math of one geoVI step as a pure function of the current
# center and pre-drawn standard normal samples:
struct _GeoVIStepFn{F,D,AD<:ADSelector,OPT<:NewtonCG,SOPT<:NewtonCG} <: Function
    forward_model::F
    data::D
    ad::AD
    optimizer::OPT
    sampling_optimizer::SOPT
    cg_max_iterations::Int
    cg_rtol::Float64
end

# Adapt's generic closure rule does not support callable structs:
Adapt.adapt_structure(to, s::_GeoVIStepFn) = _GeoVIStepFn(
    Adapt.adapt(to, s.forward_model), Adapt.adapt(to, s.data),
    s.ad, s.optimizer, s.sampling_optimizer, s.cg_max_iterations, s.cg_rtol
)

function (s::_GeoVIStepFn)(
    center::AbstractVector{<:RealLike},
    sample_n::AbstractMatrix{<:RealLike}, sample_η::AbstractMatrix{<:RealLike}
)
    residual_samples = _geovi_residual_samples(
        s.forward_model, center, sample_n, sample_η,
        s.ad, s.sampling_optimizer, s.cg_max_iterations, s.cg_rtol
    )
    mnlp = _GeoVIKLTarget(s.forward_model, s.data, residual_samples)
    ∇mnlp = gradient_func(mnlp, s.ad, center)
    Σ̅⁻¹ = _mean_fisher_curvature(s.forward_model, s.ad, residual_samples, (+1,))
    center_updated, min_mnlp, _ = _newtoncg_optimize(mnlp, ∇mnlp, Σ̅⁻¹, center, s.optimizer, (;))
    return center_updated, min_mnlp, residual_samples
end


"""
    struct GeoVIPreparedStep

A geoVI step prepared via [`geovi_prepare`](@ref) for a fixed model, data,
number of residual samples and parameter dimensionality.

Run steps with `geovi_step(prepared::GeoVIPreparedStep, center)`.
"""
struct GeoVIPreparedStep{SF,CTX<:MGVIContext}
    step_fn::SF
    n_x::Int
    n_θ::Int
    n_residuals::Int
    context::CTX
end
export GeoVIPreparedStep


"""
    geovi_prepare(
        forward_model, data, n_residuals::Integer,
        center_proto::AbstractVector{<:Real},
        config::GeoVIConfig, context::MGVIContext;
        device = nothing
    )

Prepare repeated geoVI steps for the given model and data and return a
[`GeoVIPreparedStep`](@ref), like [`mgvi_prepare`](@ref) does for MGVI
steps.

If `device` (an `MLDataDevices.AbstractDevice`) is given, the step
computation is set up on the device via `HeterogeneousComputing.on_device`.
Unlike for [`mgvi_prepare`](@ref), Reactant compilation is not supported
yet for geoVI steps and is rejected.

The prepared step uses [`MGVI._batched_cg`](@ref) for the linear part of
the residual sampling instead of `config.linsolver`; `maxiters` and
`reltol` in `config.linsolver_opts` are honored.
"""
function geovi_prepare(
    forward_model, data, n_residuals::Integer, center_proto::AbstractVector{<:Real},
    config::GeoVIConfig, context::MGVIContext;
    device = nothing
)
    n_residuals > 0 || throw(ArgumentError(
        "n_residuals must be positive, got $n_residuals"
    ))
    optimizer = config.optimizer
    optimizer isa NewtonCG || throw(ArgumentError(
        "geovi_prepare requires config.optimizer to be a MGVI.NewtonCG"
    ))
    # The in-trace Jacobian-adjoint applications of the per-sample geoVI
    # solves currently hit an Enzyme-Reactant autodiff-op limitation
    # ("Too few arguments to autodiff op" during MLIR compilation):
    if nameof(typeof(device)) == :ReactantDevice
        throw(ArgumentError(
            "Reactant compilation of geoVI steps is not supported yet"
        ))
    end

    n_θ = length(center_proto)
    n_x = length(euclidean_coords(forward_model(center_proto)))
    cg_max_iterations = Int(get(config.linsolver_opts, :maxiters, 4 * n_θ))
    cg_rtol = Float64(get(config.linsolver_opts, :reltol, sqrt(eps(Float64))))

    step_fn = _GeoVIStepFn(
        forward_model, data, context.ad, optimizer, config.sampling_optimizer,
        cg_max_iterations, cg_rtol
    )

    genctx = context.gen
    dummy_center = collect(center_proto)
    dummy_n = randn(genctx, n_x, n_residuals)
    dummy_η = randn(genctx, n_θ, n_residuals)
    prepared_fn = _maybe_on_device(step_fn, device, dummy_center, dummy_n, dummy_η)

    return GeoVIPreparedStep(prepared_fn, n_x, n_θ, Int(n_residuals), context)
end
export geovi_prepare


"""
    geovi_step(prepared::GeoVIPreparedStep, center::AbstractVector{<:Real})

Performs one geoVI step like
`geovi_step(forward_model, data, n_residuals, center, config, context)`,
using a step prepared with [`geovi_prepare`](@ref).

Returns a tuple `(result::MGVIResult, updated_center)`.
"""
function geovi_step(prepared::GeoVIPreparedStep, center::AbstractVector{<:Real})
    genctx = prepared.context.gen
    sample_n = randn(genctx, prepared.n_x, prepared.n_residuals)
    sample_η = randn(genctx, prepared.n_θ, prepared.n_residuals)
    center_updated, min_mnlp, residual_samples = prepared.step_fn(center, sample_n, sample_η)
    samples = center_updated .+ residual_samples
    info = (linsolver_output = nothing, optimizer_output = nothing)
    result = MGVIResult(samples, min_mnlp, info)
    return result, oftype(center, center_updated)
end
