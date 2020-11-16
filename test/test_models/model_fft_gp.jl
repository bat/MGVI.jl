# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

module ModelFFTGP

import Base: *, adjoint
import AbstractFFTs
import FFTW: plan_r2r, DHT
import ForwardDiff
import Random: randn, MersenneTwister
import Distributions: Normal
import ValueShapes: NamedTupleDist
import Zygote
import LinearAlgebra: Diagonal

export model, true_params, starting_point

_dims = 40
_k = [i < _dims / 2 ? i : _dims-i for i = 0:_dims-1]

# Define the harmonic transform operator as a matrix-like object
_ht = plan_r2r(zeros(_dims), DHT)

# Unfortunately neither Zygote nor ForwardDiff support planned Hartley
# transformations. While Zygote does not support AbstractFFTs.ScaledPlan,
# ForwardDiff does not overload the appropriate methods from AbstractFFTs.
function _plan_dual_product(trafo::AbstractFFTs.Plan, u::Vector{ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    # Unpack AoS -> SoA
    vs = ForwardDiff.value.(u)
    ps = mapreduce(ForwardDiff.partials, hcat, u)
    # Actual computation
    val = trafo * vs
    jvp = [trafo*t[:] for t in eachrow(ps)]
    # Pack SoA -> AoS (depending on jvp, might need `eachrow`)
    return map((v, p) -> ForwardDiff.Dual{T}(v, p...), val, zip(jvp...))
end


function *(trafo::AbstractFFTs.Plan, u::Vector{ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    _plan_dual_product(trafo, u)
end

function *(trafo::AbstractFFTs.ScaledPlan, u::Vector{ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    _plan_dual_product(trafo, u)
end

Zygote.@adjoint function *(trafo::AbstractFFTs.Plan, xs)
    # return trafo * xs, Δ -> (nothing, adjoint(trafo) * Δ)
    return trafo * xs, Δ -> (nothing, trafo * Δ)
end

Zygote.@adjoint function inv(trafo::AbstractFFTs.Plan)
    inv_t = inv(trafo)
    return inv_t, function (Δ)
        return (- inv_t * Δ * inv_t,)
    end
end

function _correlated_field(ξ::Vector)
    loglogslope = 2.3
    P = @. 50 / (_k^loglogslope + 1)
    return inv(_ht) * (P .* ξ)
end

function _mean(ξ::Vector)
    return exp.(_correlated_field(ξ))
end

function model(ξ::Vector)
    return NamedTupleDist(unnamed=Normal.(_mean(ξ), 0.4))
end

# ξ := latent variables
true_params = randn(MersenneTwister(128), _dims)
starting_point = randn(MersenneTwister(12), _dims)

end  # module
