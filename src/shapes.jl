# This file is a part of MGVI.jl, licensed under the MIT License (MIT).
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

function ChainRulesCore.rrule(::typeof(MGVI._uppertriang_to_vec), m::AbstractMatrix)
    res = MGVI._uppertriang_to_vec(m)

    function _uppertriang_to_vec_pullback(x)
        triang_n = size(res, 1)
        n = size(m, 1)
        pb_res = zero(m)
        for j in 1:n
            pb_res[j, 1:j] .= x[j*(j-1)÷2+1:j*(j+1)÷2]
        end
        ChainRulesCore.NoTangent(), pb_res
    end

    res, _uppertriang_to_vec_pullback
end

function unshaped_params(d::MvNormal)
    μ, σ = params(d)
    vcat(μ, _uppertriang_to_vec(σ))
end

function unshaped_params(d::TuringDenseMvNormal)
    μ = d.m
    σ = d.C.L*d.C.U
    vcat(μ, _uppertriang_to_vec(σ))
end
