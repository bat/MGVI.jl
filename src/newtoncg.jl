# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


"""
    struct NewtonCG

Inexact-Newton optimizer with a matrix-free conjugate-gradient inner solver.

Each step solves the Newton system approximately, with the cg residual
target set by the Eisenstat–Walker forcing sequence
`‖r‖ <= min(1/2, √‖∇f‖) * ‖∇f‖`, and finishes with a line search along the
resulting direction.

Constructors:

* '''$(FUNCTIONNAME)(; fields...)'''

$(TYPEDFIELDS)

"""
@with_kw struct NewtonCG
    "number of NewtonCG steps"
    steps::Int64 = 4

    "stop early when a NewtonCG step decreases the target by no more than
    this (in uncompiled operation only); if positive, cg iterations also
    stop once their quadratic-energy decrease becomes negligible compared
    to the previous NewtonCG step's improvement"
    absdelta::Float64 = 0.0

    "maximum number of cg iterations per NewtonCG step, `0` chooses
    automatically based on the problem dimensionality"
    cg_maxiter::Int64 = 0

    "LineSearcher that will be used after cg iterations are finished"
    linesearcher = StrongWolfe{Float64}()
end


"""
    struct MGVI.BacktrackingLineSearch

Simple Armijo backtracking line search.

Unlike the line searches from
[`LineSearches`](https://github.com/JuliaNLSolvers/LineSearches.jl) it is
free of scalar control flow, so a [`NewtonCG`](@ref) using it (with
suitable target functions) is suitable for program tracing and
compilation, e.g. via Reactant.

Constructors:

* '''$(FUNCTIONNAME)(; fields...)'''

$(TYPEDFIELDS)
"""
@with_kw struct BacktrackingLineSearch
    "sufficient-decrease constant"
    c::Float64 = 1e-4

    "step reduction factor"
    ρ::Float64 = 0.5

    "maximum number of step reductions"
    maxsteps::Int64 = 20
end

function (ls::BacktrackingLineSearch)(f_uni, df, f_and_df, β₀::RealLike, f₀::RealLike, dϕ₀::RealLike)
    c, ρ, maxsteps = ls.c, ls.ρ, ls.maxsteps
    β = oftype(f₀ / β₀, β₀)
    fβ = f_uni(β)
    k = 0
    @trace while (fβ > f₀ + c * β * dϕ₀) & (k < maxsteps)
        β = ρ * β
        fβ = f_uni(β)
        k += 1
    end
    return β, fβ
end


mutable struct NewtonCGResults{O, Tx, Tf, M}
    method::O
    initial_x::Tx
    minimizer::Tx
    initial_f::Tf
    minimum::Tf
    iterations::Int
    trace::M
    f_calls::Int
    g_calls::Int
    cg_iterations::Int
    absolute_reduction::Tf
end

function Base.show(io::IO, r::NewtonCGResults)
    println(io, "NewtonCG Results:")
    println(io, "------------------------------")
    println(io, "f value start: $(r.initial_f)")
    println(io, "f value end: $(r.minimum)")
    println(io, "absolute reduction: $(r.absolute_reduction)")
    println(io, "------------------------------")
    println(io, "f calls: $(r.f_calls)")
    println(io, "∇f calls: $(r.g_calls)")
    println(io, "------------------------------")
    f_hist, cg_hist = r.trace
    for (fₙ, cg_iter, n) in zip(f_hist, cg_hist, 0:r.iterations)
        println(io, "n: $n   fₙ: $fₙ    cg_iter: $cg_iter")
    end
end

function linesearch_args(
        f, ∇f, x::AbstractVector, Δx::AbstractVector, f_x, ∇f_x)
    # build univariate functions
    f_uni(α::Real) = f(x + α*Δx)
    df(α::Real) = dot(∇f(x + α*Δx), Δx)
    f_and_df(α::Real) = (f_uni(α), df(α))
    return (f_uni, df, f_and_df, 1.0, f_x, dot(∇f_x, Δx))
end

# LineSearches.jl searchers throw when no acceptable step exists; near a
# stationary point (directional derivative at floating-point noise level, as
# in the geoVI sampling solves) that means the step has effectively converged,
# so treat it as a zero step rather than failing:
function _run_linesearch(ls, f_uni, df, f_and_df, β₀, f₀, dϕ₀)
    try
        ls(f_uni, df, f_and_df, β₀, f₀, dϕ₀)
    catch err
        err isa LineSearches.LineSearchException || rethrow()
        (zero(f₀), f₀)
    end
end

_run_linesearch(ls::BacktrackingLineSearch, args...) = ls(args...)

struct EvalCount
    f::Function
    counter::Base.Threads.Atomic{Int}
end

EvalCount(f::Function) = EvalCount(f, Base.Threads.Atomic{Int}(0))

function (F::EvalCount)(x::AbstractVector)
    Base.Threads.atomic_add!(F.counter, 1)
    F.f(x)
end


_apply_curvature(A, v::AbstractVector) = A * v
_apply_curvature(A::Function, v::AbstractVector) = A(v)

# One conjugate-gradient iteration for `A * Δx == b`, in purely functional
# form (suitable for program tracing):
function _cg_update(A, Δx, r, p, rs)
    tiny = eps(float(one(rs)))
    Ap = _apply_curvature(A, p)
    γ = rs / (dot(p, Ap) + tiny)
    Δx = Δx + γ .* p
    r = r - γ .* Ap
    rs_next = dot(r, r)
    δ = rs_next / (rs + tiny)
    p = r + δ .* p
    return Δx, r, p, rs_next
end


function _optimize(
    f::Function, adsel::ADSelector, Σ̅⁻¹::Function,
    x₀::AbstractVector, optimizer::NewtonCG, optimization_opts::NamedTuple
)
    ∇f = gradient_func(f, adsel, x₀)
    _newtoncg_optimize(f, ∇f, Σ̅⁻¹, x₀, optimizer, optimization_opts)
end

function _newtoncg_optimize(
    f::Function, ∇f::Function, Σ̅⁻¹::Function,
    x₀::AbstractVector, optimizer::NewtonCG, optimization_opts::NamedTuple
)
    # fetch parameters from optimizer struct
    steps = optimizer.steps
    absdelta = optimizer.absdelta
    ls = optimizer.linesearcher
    # iteration cap and minimum iterations before the energy criterion may
    # fire, matching NIFTy:
    cg_maxiter = optimizer.cg_maxiter > 0 ? optimizer.cg_maxiter : min(200, 20 * length(x₀))
    cg_ad_miniter = min(6, cg_maxiter)

    # logging information, not available when compiled via program tracing:
    record_trace = !within_compile()
    cg_iterations=Int64[]
    f_history=Float64[]

    f_counted = EvalCount(f)
    ∇f_counted = EvalCount(∇f)
    # eval counters are incompatible with program tracing, count only
    # when running normally (within_compile is constant-foldable):
    f_used = within_compile() ? f : f_counted
    ∇f_used = within_compile() ? ∇f : ∇f_counted

    # value of f before any optimization steps
    f⁰ = fⁿ⁻¹ = f_used(x₀)
    record_trace && push!(f_history, f⁰)

    fⁿ = Δfⁿ = zero(f⁰)
    xₙ = x₀
    n_done = 0
    for n in 1:steps
        ∇f_at_xₙ = ∇f_used(xₙ)

        # solve Σ̅⁻¹(xₙ) * Δx == ∇f(xₙ) approximately via conjugate gradients,
        # the residual target given by the Eisenstat–Walker forcing sequence
        # ‖r‖ <= min(1/2, √‖∇f‖) * ‖∇f‖ (inexact Newton, cf. NIFTy):
        A = Σ̅⁻¹(xₙ)
        g = ∇f_at_xₙ
        Δx = zero(xₙ)
        r = copy(g)
        p = copy(r)
        rs = dot(r, r)
        rs_target = min(rs / 4, rs * sqrt(rs))
        k = 0
        if absdelta > 0
            # also stop cg once its quadratic-energy decrease per iteration
            # becomes negligible: below absdelta / 100 in the first NewtonCG
            # step, below a tenth of the previous step's improvement after:
            cg_absdelta = n == 1 ? oftype(fⁿ⁻¹, absdelta / 100) : max(zero(Δfⁿ), Δfⁿ) / 10
            E = zero(rs)
            ΔE = oftype(rs, Inf)
            @trace while (k < cg_maxiter) & (rs > rs_target) &
                    ((k < cg_ad_miniter) | (ΔE >= cg_absdelta))
                Δx, r, p, rs = _cg_update(A, Δx, r, p, rs)
                k += 1
                # cg quadratic energy ½ Δx'AΔx - g'Δx == -½ ⟨Δx, g + r⟩:
                E_next = -dot(Δx, g + r) / 2
                ΔE = E - E_next
                E = E_next
            end
        else
            @trace while (k < cg_maxiter) & (rs > rs_target)
                Δx, r, p, rs = _cg_update(A, Δx, r, p, rs)
                k += 1
            end
        end
        record_trace && push!(cg_iterations, Int(k))

        # finish NewtonCG step with line search in -Δx direction
        β, fⁿ = _run_linesearch(ls, linesearch_args(f_used, ∇f_used, xₙ, -Δx, fⁿ⁻¹, ∇f_at_xₙ)...)
        xₙ = xₙ - β .* Δx
        Δfⁿ = fⁿ⁻¹ - fⁿ
        fⁿ⁻¹ = fⁿ
        record_trace && push!(f_history, fⁿ)
        n_done = n
        # a step without meaningful improvement means convergence, don't
        # spend the remaining steps (not possible when compiled, program
        # tracing requires a fixed number of steps):
        !within_compile() && Δfⁿ <= absdelta && break
    end
    trace = (f_history=f_history, cg_iterations=cg_iterations)
    res = NewtonCGResults{typeof(optimizer), typeof(x₀), typeof(f⁰), typeof(trace)}(
        optimizer,
        x₀,
        xₙ,
        f⁰,
        fⁿ,
        n_done,
        trace,
        f_counted.counter[],
        ∇f_counted.counter[],
        sum(cg_iterations),
        abs(fⁿ - f⁰)
    )

    x_res = oftype(x₀, xₙ)
    f_x_res = fⁿ
    #@assert f_x_res == f(x_res)
    return x_res, f_x_res, res
end
