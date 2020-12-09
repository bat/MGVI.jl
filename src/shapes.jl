# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).
#
function _unshaped(x::Number)
    [x]
end

function _unshaped(x::Tuple)
    vcat(x...)
end

function _unshaped(x::NamedTuple)
    vcat(values(x)...)
end

function _unshaped(x::AbstractVector)
    x
end

function unshaped_params(d::Distribution)
    vcat(map(_unshaped, params(d))...)
end

function unshaped_params(ntd::NamedTupleDist)
    vcat(map(unshaped_params, values(ntd))...)
end

function unshaped_params(dp::Product)
    reduce(vcat, map(unshaped_params, dp.v))
end

function _uppertriang_to_vec(m::AbstractMatrix)
    reduce(vcat, [m[1:i, i] for i in 1:size(m, 1)])
end

function unshaped_params(d::MvNormal)
    μ, σ = params(d)
    vcat(μ, _uppertriang_to_vec(σ))
end
