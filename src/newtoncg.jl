# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


"""
    struct NewtonCG

Constructors:

* '''$(FUNCTIONNAME)(; fields...)'''

$(TYPEDFIELDS)

"""
@with_kw struct NewtonCG
    "amount of previous NewtonCG improvement guarding the lower
    bound to the improvement between consecutive cg iterations from
    the second NewtonCG step on"
    α::Float64=0.1

    "Number of total NewtonCG steps"
    steps::Int64=4

    "maximum number of cg iterations in the first NewtonCG step"
    i₀::Int64=5

    "maximum number of cg iterations from the second NewtonCG step on"
    i₁::Int64=50

    "LineSearcher that will be used after cg iterations are finished"
    linesearcher=StrongWolfe{Float64}()
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
    ∇f = gradient_func(f, adsel)
    _newtoncg_optimize(f, ∇f, Σ̅⁻¹, x₀, optimizer, optimization_opts)
end

function _newtoncg_optimize(
    f::Function, ∇f::Function, Σ̅⁻¹::Function,
    x₀::AbstractVector, optimizer::NewtonCG, optimization_opts::NamedTuple
)
    # fetch parameters from optimizer struct
    α = optimizer.α
    steps = optimizer.steps
    i₀ = optimizer.i₀
    i₁ = optimizer.i₁
    ls = optimizer.linesearcher

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
    for n in 1:steps
        ∇f_at_xₙ = ∇f_used(xₙ)

        # solve Σ̅⁻¹(xₙ) * Δx == ∇f(xₙ) approximately via conjugate gradients:
        A = Σ̅⁻¹(xₙ)
        Δx = zero(xₙ)
        r = ∇f_at_xₙ
        p = copy(r)
        rs = dot(r, r)
        k = 0
        if n == 1
            # do i₀ iterations of cg
            for _ in 1:i₀
                Δx, r, p, rs = _cg_update(A, Δx, r, p, rs)
                k += 1
            end
        else
            # do at most i₁ cg iterations, move on early if the improvement
            # falls below a fraction α of the previous NewtonCG step improvement:
            fᵏ = fⁿ⁻¹
            Δfᵏ = oftype(fⁿ⁻¹, Inf)
            @trace while (k < i₁) & (Δfᵏ >= α * Δfⁿ)
                Δx, r, p, rs = _cg_update(A, Δx, r, p, rs)
                k += 1
                fᵏ_next = f_used(xₙ - Δx)
                Δfᵏ = abs(fᵏ_next - fᵏ)
                fᵏ = fᵏ_next
            end
        end
        record_trace && push!(cg_iterations, Int(k))

        # finish NewtonCG step with line search in -Δx direction
        β, fⁿ = _run_linesearch(ls, linesearch_args(f_used, ∇f_used, xₙ, -Δx, fⁿ⁻¹, ∇f_at_xₙ)...)
        xₙ = xₙ - β .* Δx
        Δfⁿ = abs(fⁿ - fⁿ⁻¹)
        fⁿ⁻¹ = fⁿ
        record_trace && push!(f_history, fⁿ)
    end
    trace = (f_history=f_history, cg_iterations=cg_iterations)
    res = NewtonCGResults{typeof(optimizer), typeof(x₀), typeof(f⁰), typeof(trace)}(
        optimizer,
        x₀,
        xₙ,
        f⁰,
        fⁿ,
        steps,
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
