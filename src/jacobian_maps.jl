# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

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
    # TODO: maybe put special tag for Dual here?
    dual_along(f::Function, x::Vector, δ::Vector) = ForwardDiff.Dual.(x, δ) |> f
    grad_along(f::Function, x::Vector, δ::Vector) = ForwardDiff.partials.(dual_along(f, x, δ), 1)
    jvd(δ::Vector) = grad_along(λs, θ, δ)
    vjd(δ::Vector) = first(Zygote.pullback(λs, θ)[2](δ))
    LinearMap(jvd, vjd, num_λs, size(θ, 1))
end

struct FwdDerJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

function (jf::FwdDerJacobianFunc)(θ::Vector)
    λs = jf.f
    num_θs = size(θ, 1)
    num_λs = size(λs(θ), 1)

    dual_along(f::Function, x::Vector, δ::Vector) = map(ForwardDiff.Dual, x, δ) |> f
    grad_along(f::Function, x::Vector, δ::Vector) = vcat(map(ForwardDiff.partials, dual_along(f, x, δ))...)
    jvd(δ::Vector) = grad_along(λs, θ, δ)
    vjd(δ::Vector) = ForwardDiff.gradient(t -> dot(δ, jvd(t)), zeros(num_θs))

    LinearMap(jvd, vjd, num_λs, num_θs)
end
