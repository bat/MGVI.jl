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


mutable struct NewtonCGResults{O, Tx, Tf, M} <: Optim.OptimizationResults
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

struct EvalCount
    f::Function
    counter::Base.Threads.Atomic{Int}
end

EvalCount(f::Function) = EvalCount(f, Base.Threads.Atomic{Int}(0))

function (F::EvalCount)(x::AbstractVector)
    Base.Threads.atomic_add!(F.counter, 1)
    F.f(x)
end

function _optimize(f::Function, ∇f!::Function, Σ̅⁻¹::Function, 
        x₀::AbstractVector, optimizer::NewtonCG, optim_options)
    # fetch parameters from optimizer struct
    α = optimizer.α
    steps = optimizer.steps
    i₀ = optimizer.i₀
    i₁ = optimizer.i₁
    ls = optimizer.linesearcher
    # logging information
    cg_iterations=Int64[]
    f_history=Float64[]

    # gradient function of f
    ∇f(x) = ∇f!(similar(x), x)

    f = EvalCount(f)
    ∇f = EvalCount(∇f)
    
    # value of f before any optimization steps
    f⁰ = fⁿ⁻¹ = f(x₀)
    push!(f_history, f⁰)
    push!(cg_iterations, i₀)

    fⁿ = Δfⁿ = zero(f⁰)
    xₙ = x₀
    for n in 1:steps
        # preallocate/reset vector of descent
        Δx = zero(xₙ)
        ∇f_at_xₙ = ∇f(xₙ)

        # initialize cg_iterator with our mean fisher metric as the
        # hessian and our gradient of f at xₙ as our gradient
        # A = Σ̅⁻¹, b = ∇f(xₙ)
        cgiterator = cg_iterator!(Δx, Σ̅⁻¹(xₙ), ∇f_at_xₙ, initially_zero=true)
        if n == 1
            # do i₀ iterations of cg
            for (iteration, residual) in enumerate(cgiterator)
                if iteration >= i₀
                    break
                end
            end
        else
            fᵏ⁻¹ = fⁿ⁻¹
            # do at most i₁ cg iterations or move on if our improvement
            # is below α*100 percent of our previous NewtonCG step
            for (k, residual) in enumerate(cgiterator)
                fᵏ = f(xₙ - Δx)
                if k >= i₁ || abs(fᵏ - fᵏ⁻¹) < α*Δfⁿ
                    push!(cg_iterations, k)
                    break
                end
                fᵏ⁻¹ = fᵏ
            end
        end
        # finish NewtonCG step with line search in -Δx direction
        β, fⁿ = ls(linesearch_args(f, ∇f, xₙ, -Δx, fⁿ⁻¹, ∇f_at_xₙ)...)
        xₙ -= β*Δx
        Δfⁿ = abs(fⁿ - fⁿ⁻¹)
        fⁿ⁻¹ = fⁿ
        push!(f_history, fⁿ)
    end
    trace = (f_history=f_history, cg_iterations=cg_iterations)
    NewtonCGResults{typeof(optimizer), typeof(x₀), typeof(f⁰), typeof(trace)}(
        optimizer,
        x₀,
        xₙ,
        f⁰,
        fⁿ,
        steps,
        trace,
        f.counter[],
        ∇f.counter[],
        sum(cg_iterations),
        abs(fⁿ - f⁰)
    )
end

function _optimize(f::Function, ∇f!::Function, curvature::Function, 
        x₀::AbstractVector, optimizer::Optim.AbstractOptimizer, 
        optim_options::Optim.Options)
    optimize(f, ∇f!, x₀, optimizer, optim_options)
end