# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

abstract type AbstractJacobianFunc <: Function end

struct JacVecProdTag{F, T} end

function _dual_along(f::F, x::AbstractVector{T1}, δ::AbstractVector{T2}) where {F, T1, T2}
    T =  promote_type(T1, T2)
    T_Dual = JacVecProdTag{F, T}
    f(ForwardDiff.Dual{T_Dual}.(x, δ))
end

struct FullJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

function (jf::FullJacobianFunc)(θ::Vector)
    λs = jf.f
    jac = ForwardDiff.jacobian(λs, θ)
    LinearMap(jac, isposdef=false, issymmetric=false, ishermitian=false)
end

struct FwdRevADJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

function (jf::FwdRevADJacobianFunc)(θ::Vector)
    λs = jf.f
    num_λs = size(λs(θ), 1)
    grad_along(f::Function, x::Vector, δ::Vector) = ForwardDiff.partials.(_dual_along(f, x, δ), 1)
    jvd(δ::Vector) = grad_along(λs, θ, δ)
    vjd(δ::Vector) = first(Zygote.pullback(λs, θ)[2](δ))
    LinearMap(jvd, vjd, num_λs, size(θ, 1), isposdef=false, issymmetric=false, ishermitian=false)
end

struct FwdDerJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

function (jf::FwdDerJacobianFunc)(θ::Vector)
    λs = jf.f
    num_θs = size(θ, 1)
    num_λs = size(λs(θ), 1)
    grad_along(f::Function, x::Vector, δ::Vector) = ForwardDiff.partials.(_dual_along(f, x, δ), 1)
    jvd(δ::Vector) = grad_along(λs, θ, δ)
    vjd(δ::Vector) = ForwardDiff.gradient(t -> dot(δ, jvd(t)), zeros(num_θs))
    LinearMap(jvd, vjd, num_λs, num_θs, isposdef=false, issymmetric=false, ishermitian=false)
end
