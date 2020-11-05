abstract type AbstractJacobianFunc <: Function end

struct FullJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

function (jf::FullJacobianFunc)(θ::Vector)
    λs = jf.f
    jac = ForwardDiff.jacobian(λs, θ)
    LinearMap(jac)
end

struct FwdRevADJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

function (jf::FwdRevADJacobianFunc)(θ::Vector)
    λs = jf.f
    num_λs = size(λs(θ), 1)
    dual_along(f::Function, x::Vector, δ::Vector) = map(ForwardDiff.Dual, x, δ) |> f
    grad_along(f::Function, x::Vector, δ::Vector) = mapreduce(ForwardDiff.partials, vcat, dual_along(f, x, δ))
    jvd(δ::Vector) = grad_along(λs, θ, δ)
    vjd(δ::Vector) = first(Zygote.pullback(λs, θ)[2](δ))
    LinearMap(jvd, vjd, num_λs, size(θ, 1))
end

struct FwdJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

function (jf::FwdJacobianFunc)(θ::Vector)
    λs = jf.f
    num_θs = size(θ, 1)
    num_λs = size(λs(θ), 1)

    dual_along(f::Function, x::Vector, δ::Vector) = map(ForwardDiff.Dual, x, δ) |> f
    grad_along(f::Function, x::Vector, δ::Vector) = mapreduce(ForwardDiff.partials, vcat, dual_along(f, x, δ))
    jvd(δ::Vector) = grad_along(λs, θ, δ)

    grad_i(i) = map(first ∘ ForwardDiff.partials,
                    λs([θ[1:i-1]..., ForwardDiff.Dual(θ[i], 1.), θ[i+1:end]...]))
    grad_i_along(i, δ::Vector) = dot(δ, grad_i(i))
    vjd(δ::Vector) = mapreduce(p -> grad_i_along(p...), vcat, enumerate(repeated(δ, num_θs)))

    LinearMap(jvd, vjd, num_λs, num_θs)
end

struct FwdDerJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

function (jf::FwdDerJacobianFunc)(θ::Vector)
    λs = jf.f
    num_θs = size(θ, 1)
    num_λs = size(λs(θ), 1)

    dual_along(f::Function, x::Vector, δ::Vector) = map(ForwardDiff.Dual, x, δ) |> f
    grad_along(f::Function, x::Vector, δ::Vector) = mapreduce(ForwardDiff.partials, vcat, dual_along(f, x, δ))
    jvd(δ::Vector) = grad_along(λs, θ, δ)
    vjd(δ::Vector) = ForwardDiff.gradient(t -> dot(δ, jvd(t)), zeros(num_θs))

    LinearMap(jvd, vjd, num_λs, num_θs)
end

export FwdJacobianFunc,
       FwdDerJacobianFunc,
       FwdRevADJacobianFunc,
       FullJacobianFunc,
       AbstractJacobianFunc
