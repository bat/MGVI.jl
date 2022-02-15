using LineSearches
using Parameters

export _optimize, NewtonCG

@with_kw struct NewtonCG
    α::Float64=0.1
    steps::Int64=4
    i₀::Int64=5
    i₁::Int64=10
    linesearcher=StrongWolfe{Float64}()
    verbose::Bool=false
end

function linesearch_args(
        f, ∇f, x::AbstractVector, Δx::AbstractVector, f_x, ∇f_x)
    # build univariate functions
    f_uni(α::Real) = f(x + α*Δx)
    df(α::Real) = dot(∇f(x + α*Δx), Δx)
    f_and_df(α::Real) = (f_uni(α), df(α))
    return (f_uni, df, f_and_df, 1.0, f_x, dot(∇f_x, Δx))
end

mutable struct KL_
    kl::Function
    count::Int
end

mutable struct ∇KL_
    ∇kl::Function 
    count::Int
end

function (f::KL_)(x::AbstractVector)
    f.count += 1
    f.kl(x)
end

function (f::∇KL_)(x::AbstractVector)
    f.count += 1
    f.∇kl(x)
end

function _optimize(optimizer::NewtonCG, optim_options, kl::Function, 
        ∇kl!::Function, Σ̅⁻¹::LinearMap, ξ̅⁰::AbstractVector)
    # fetch parameters from optimizer struct
    α = optimizer.α
    steps = optimizer.steps
    i₀ = optimizer.i₀
    i₁ = optimizer.i₁
    ls = optimizer.linesearcher
    verbose = optimizer.verbose
    # logging information
    kl_calls=0
    ∇kl_calls=0
    cg_iterations=Int64[]
    Δkl_history=Float64[]

    # gradient function of our kl
    ∇kl(ξ) = ∇kl!(similar(ξ), ξ)

    kl = KL_(kl, 0)
    ∇kl = ∇KL_(∇kl, 0)
    
    # value of kl divergence before any optimization steps
    kl⁰ = klⁿ⁻¹ = kl(ξ̅⁰)

    klⁿ = Δklⁿ = zero(kl⁰)
    ξ̅ⁿ = ξ̅⁰
    for n in 1:steps
        # preallocate/reset vector of descent
        Δξ̅ = zero(ξ̅ⁿ)
        ∇kl_at_ξ̅ⁿ = ∇kl(ξ̅ⁿ)

        # initialize cg_iterator with our mean fisher metric as the
        # hessian and our gradient of our kl at ξ̅ⁿ as our gradient
        # A = Σ̅⁻¹, b = ∇kl(ξ̅ⁿ)
        cgiterator = cg_iterator!(Δξ̅, Σ̅⁻¹, ∇kl_at_ξ̅ⁿ, initially_zero=true)
        if n == 1
            # do i₀ iterations of cg
            for (iteration, residual) in enumerate(cgiterator)
                if iteration >= i₀
                    break
                end
            end
        else
            klᵏ⁻¹ = klⁿ⁻¹
            # do at most i₁ cg iterations or move on if our improvement
            # is below α*100 percent of our previous NewtonCG step
            for (k, residual) in enumerate(cgiterator)
                klᵏ = kl(ξ̅ⁿ - Δξ̅)
                if k >= i₁ || abs(klᵏ - klᵏ⁻¹) < α*Δklⁿ
                    push!(cg_iterations, k)
                    break
                end
                klᵏ⁻¹ = klᵏ
            end
        end
        # finish NewtonCG step with line search in -Δξ̅ direction
        β, klⁿ = ls(linesearch_args(kl, ∇kl, ξ̅ⁿ, -Δξ̅, klⁿ⁻¹, ∇kl_at_ξ̅ⁿ)...)
        ξ̅ⁿ -= β*Δξ̅
        Δklⁿ = abs(klⁿ - klⁿ⁻¹)
        push!(Δkl_history, Δklⁿ)
        klⁿ⁻¹ = klⁿ
    end
    kl_calls = kl.count
    ∇kl_calls = ∇kl.count
    rel_reduction = sum(Δkl_history) / kl⁰
    if verbose
        println("NewtonCG Results:")
        println("------------------------------")
        println("kl value start: $kl⁰")
        println("kl value end: $klⁿ")
        println("relative reduction: $rel_reduction")
        println("------------------------------")
        println("kl calls: $(kl_calls)")
        println("∇kl calls: $(∇kl_calls)")
        println("------------------------------")
        println("1.step: $(i₀) cg iterations Δkl_1 $(Δkl_history[1])")
        for i in 2:steps
            print("$i.step: $(cg_iterations[i-1]) cg iterations")
            println(" Δkl_$i $(Δkl_history[i])")
        end
    end
    return (
        minimizer=ξ̅ⁿ,
        minimum=klⁿ,
        kl_start=kl⁰,
        kl_calls=kl_calls,
        ∇kl_calls=∇kl_calls,
        cg_iterations=cg_iterations,
        Δkl_history=Δkl_history,
        rel_reduction=rel_reduction
    )
end

#
# _optimize dispatch for Optim.optimizer
#
function _optimize(optimizer::Optim.AbstractOptimizer, 
        optim_options::Optim.Options, kl::Function, ∇kl!::Function, 
        curvature::LinearMap, ξ::AbstractVector)
    optimize(kl, ∇kl!, ξ, optimizer, optim_options)
end