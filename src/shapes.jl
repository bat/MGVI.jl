# This file is a part of MGVI.jl, licensed under the MIT License (MIT).
#
function _unshaped(x::Number)
    [x]
end

function _unshaped(x)
    unshaped(x)
end

function unshaped_params(d::Distribution)
    reduce(vcat, map(_unshaped, params(d)))
end

function unshaped_params(ntd::NamedTupleDist)
    map(unshaped_params, ntd)
end

function unshaped_params(dp::Product)
    reduce(vcat, map(unshaped_params, dp.v))
end
