# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

"""
    abstract type AbstractJacobianFunc <: Function end

Abstract type of the jacobian calculators.

Represents the jacobian of the function. The instance can be called at the point `θ`,
a LinearMap representing the jacobian is returned.
"""
abstract type AbstractJacobianFunc <: Function end

struct JacVecProdTag{F, T} end

function _dual_along(f::F, x::AbstractVector{T1}, δ::AbstractVector{T2}) where {F, T1, T2}
    T =  promote_type(T1, T2)
    T_Dual = JacVecProdTag{F, T}
    f(ForwardDiff.Dual{T_Dual}.(x, δ))
end

"""
    FullJacobianFunc(f)

Construct of the Jacobian with ForwardDiff.

When called at point `θ`, the Jacobian matrix being fully instantiated
and stored explicitly in memory.

# Examples
```julia
# forward_model: θ -> Distribution
jacobian_func = FullJacobianFunc(forward_model)
jacobian = jacobian_func(θ)  # LinearMap
jacobian * v  # act on vector
```
"""
struct FullJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

function (jf::FullJacobianFunc)(θ::AbstractVector{T}) where T
    λs = jf.f
    jac = ForwardDiff.jacobian(λs, θ)
    LinearMap{T}(jac, isposdef=false, issymmetric=false, ishermitian=false)
end

"""
    FwdRevADJacobianFunc(f)

Construct of the Jacobian with ForwardDiff for direct action and Zygote.pullback for the adjoint

The Jacobian action is computed on the fly, no matrix is stored in memory at any time.

# Examples
```julia
# forward_model: θ -> Distribution
jacobian_func = FwdRevADJacobianFunc(forward_model)
jacobian = jacobian_func(θ)  # LinearMap
jacobian * v  # act on vector as if it was a matrix
```
"""
struct FwdRevADJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

function (jf::FwdRevADJacobianFunc)(θ::AbstractVector{T}) where T
    λs = jf.f
    num_λs = size(λs(θ), 1)
    grad_along(f, x, δ) = ForwardDiff.partials.(_dual_along(f, x, δ), 1)
    jvd(δ) = grad_along(λs, θ, δ)
    vjd(δ) = first(Zygote.pullback(λs, θ)[2](δ))
    LinearMap{T}(jvd, vjd, num_λs, size(θ, 1), isposdef=false, issymmetric=false, ishermitian=false)
end

"""
    FwdDerJacobianFunc(f)

Construct of the Jacobian with ForwardDiff for the direct action, and twice applied
ForwardDiff for the adjoint action

Adjoint is implemented by introducing a placeholder parametric vector ``\\vec{t}``:

`` \\frac{d}{d\\vec{t}} \\vec{x} \\cdot (A \\vec{t}) = A^{T} \\vec{x} ``

This allows to implement both, the direct and adjoint actions, using
ForwardDiff, without instantiating full Jacobian at any point.
"""
struct FwdDerJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

function (jf::FwdDerJacobianFunc)(θ::AbstractVector{T}) where T
    λs = jf.f
    num_θs = size(θ, 1)
    num_λs = size(λs(θ), 1)
    grad_along(f, x, δ) = ForwardDiff.partials.(_dual_along(f, x, δ), 1)
    jvd(δ) = grad_along(λs, θ, δ)
    vjd(δ) = ForwardDiff.gradient(t -> dot(δ, jvd(t)), zeros(num_θs))
    LinearMap{T}(jvd, vjd, num_λs, num_θs, isposdef=false, issymmetric=false, ishermitian=false)
end

export FullJacobianFunc,
       FwdRevADJacobianFunc,
       FwdDerJacobianFunc
