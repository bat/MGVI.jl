# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

module ModelFFTGP

import FFTW: fft
import ForwardDiff
import Random: randn, MersenneTwister
import Distributions: Normal
import ValueShapes: NamedTupleDist

export model, true_params, starting_point

const _dims = 40
const _k = [i < _dims / 2 ? i : _dims-i for i = 0:_dims-1]

# Discrete Hartley transform via FFT: real-to-real, self-inverse up to a
# factor of `length(x)`, and (unlike FFTW's real-to-real plans) composable
# with automatic differentiation and program tracing:
function dht(x::AbstractArray)
    F = fft(Complex.(x))
    return real.(F) .- imag.(F)
end

# The DHT is linear, so it transforms dual values and partials independently:
function dht(x::AbstractArray{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
    val = dht(ForwardDiff.value.(x))
    parts = ntuple(i -> dht(ForwardDiff.partials.(x, i)), Val(N))
    return ForwardDiff.Dual{T}.(val, parts...)
end

idht(x::AbstractArray) = dht(x) ./ length(x)

function _correlated_field(ξ::Vector)
    loglogslope = 2.3
    P = @. 50 / (_k^loglogslope + 1)
    return idht(P .* ξ)
end

function _mean(ξ::Vector)
    return exp.(_correlated_field(ξ))
end

function model(ξ::Vector)
    return NamedTupleDist(unnamed=Normal.(_mean(ξ), 0.4))
end

# ξ := latent variables
const true_params = randn(MersenneTwister(128), _dims)
const starting_point = randn(MersenneTwister(12), _dims)

end # module
