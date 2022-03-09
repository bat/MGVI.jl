# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

"""
    abstract type AbstractJacobianFunc <: Function end

Abstract type of the jacobian calculators.

Represents the jacobian of the function. The instance can be called at the point `x`,
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

When called at point `x`, the Jacobian matrix is fully instantiated
and stored explicitly in memory.

# Examples
```julia
# forward_model: x -> Distribution
jacobian_func = FullJacobianFunc(forward_model)
jacobian = jacobian_func(x)  # LinearMap
jacobian * δ  # act on vector
```
"""
struct FullJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

export FullJacobianFunc

function (jf::FullJacobianFunc)(x::AbstractVector{<:Real})
    f = jf.f
    y = f(x)
    U = promote_type(eltype(x), eltype(y))
    # ForwardDiff.jacobian is not type stable, so convert and fix type:
    jac = convert(Matrix{U}, ForwardDiff.jacobian(f, x))::Matrix{U}
    LinearMap{U}(jac, isposdef=false, issymmetric=false, ishermitian=false)
end



struct ForwardDiffJVP{F<:Function,T<:AbstractVector{<:Real}}
    f::F
    x::T
end

function (jvp::ForwardDiffJVP)(δ::AbstractVector{<:Real})
    ForwardDiff.partials.(_dual_along(jvp.f, jvp.x, δ), 1)
end



struct ForwardDiffVJP{F<:Function,T<:AbstractVector{<:Real}}
    f::F
    x::T
end

function (vjp::ForwardDiffVJP)(δ::AbstractVector{<:Real})
    f, x = vjp.f, vjp.x

    # Mathematically elegant, but not type stable and uses double differentiation:
    #jvp = ForwardDiffJVP(f, x)
    #g(t) = dot(δ, jvp(t))
    #ForwardDiff.gradient(g, zero(vjp.x))

    jvp = ForwardDiffJVP(f, x)
    U = promote_type(eltype(f(x)), eltype(x))
    result = similar(x, U)
    tmp = similar(x, U)
    for i in eachindex(x)
        tmp .= 0
        tmp[i] = 1
        result[i] = dot(δ, jvp(tmp))
    end
    result
end



struct ZygoteVJP{F<:Function,T,G<:Function}
    f::F
    x::T
    pullback::G
end

ZygoteVJP(f::Function, x) = ZygoteVJP(f, x, Zygote.pullback(f, x)[2])

@inline (vjp::ZygoteVJP)(δ) = first(vjp.pullback(δ))



"""
    FwdRevADJacobianFunc(f)

Construct of the Jacobian with ForwardDiff for direct action and Zygote.pullback for the adjoint

The Jacobian action is computed on the fly, no matrix is stored in memory at any time.

# Examples
```julia
# forward_model: x -> Distribution
jacobian_func = FwdRevADJacobianFunc(forward_model)
jacobian = jacobian_func(x)  # LinearMap
jacobian * δ  # act on vector as if it was a matrix
```
"""
struct FwdRevADJacobianFunc{F<:Function} <: AbstractJacobianFunc
    f::F
end

export FwdRevADJacobianFunc

function (jf::FwdRevADJacobianFunc)(x::AbstractVector{T}) where T
    f = jf.f
    y = f(x)
    jvp = ForwardDiffJVP(f, x)
    vjp = ZygoteVJP(f, x)
    LinearMap{T}(jvp, vjp, size(y, 1), size(x, 1), isposdef=false, issymmetric=false, ishermitian=false)
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

export FwdDerJacobianFunc

function (jf::FwdDerJacobianFunc)(x::AbstractVector{T}) where T
    f = jf.f
    y = f(x)
    jvp = ForwardDiffJVP(f, x)
    vjp = ForwardDiffVJP(f, x)
    LinearMap{T}(jvp, vjp, size(y, 1), size(x, 1), isposdef=false, issymmetric=false, ishermitian=false)
end
