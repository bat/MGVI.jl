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

geoVI locally isometrizes the Fisher metric `M(ξ) = J'ℐJ + I` via the
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
    XV<:AbstractVector{<:Real},OPJ<:LinearMap,CTX<:MGVIContext
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
    x̄, C = with_jacobian(_euclidean_coords_flat(f_model), center_point, LinearMap, context.ad)
    GeoVISampler(
        f_model, center_point, linear_solver, linear_solver_opts,
        x̄, convert(LinearMap, C), sampling_optimizer, context
    )
end


# Draw t ~ N(0, M̄) and the corresponding linear (MGVI) residual M̄⁻¹t:
function _geovi_sample_pair(s::GeoVISampler)
    genctx = s.context.gen
    C = s.jac_dx_dθ
    n_x, n_θ = size(C)
    M̄ = C' * C + I
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
        x_ξ, J_x = with_jacobian(x_fn, ξ, LinearMap, adsel)
        r = (ξ - ξ̄) + C' * (x_ξ - x̄) - t
        r + J_x' * (C * r)
    end

    # Gauss-Newton metric G'G with G = ∂g̃/∂ξ = I + C'J_x(ξ):
    function curvature(ξ::AbstractVector)
        _, J_x = with_jacobian(x_fn, ξ, LinearMap, adsel)
        G = C' * J_x + I
        G' * G
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


function _mean_neg_log_pstr_cols(f::Function, data, residual_samples::AbstractMatrix{<:Real}, center::AbstractVector{<:Real})
    res = sum(i -> -posterior_loglike(f, center + residual_samples[:, i], data), axes(residual_samples, 2))
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
    smplr = GeoVISampler(
        forward_model, center_init, config.linsolver, config.sampling_optimizer, context;
        linear_solver_opts = config.linsolver_opts
    )
    residual_samples = sample_geovi_residuals(smplr, n_residuals)
    mnlp(params::AbstractVector) = _mean_neg_log_pstr_cols(forward_model, data, residual_samples, params)
    OP = _get_operator_type(config.linsolver)
    Σ⁻¹(ξ) = _inv_cov_est(forward_model, ξ, OP, context)
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
    smplr = GeoVISampler(
        forward_model, center, config.linsolver, config.sampling_optimizer, context;
        linear_solver_opts = config.linsolver_opts
    )
    residual_samples = sample_geovi_residuals(smplr, n_residuals)
    return center .+ residual_samples
end
export geovi_sample
