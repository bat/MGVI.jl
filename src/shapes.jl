# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).
#
function unshaped(x::Number)
    [x]
end

function unshaped(x::Array{Number})
    reshape(x, :)
end

function unshaped(x::PDMat)
    reshape(x, :)
end

function unshaped(x)
    reduce(vcat, map(unshaped, x))
end


function unshaped_params(d::Distribution)
    reduce(vcat, map(unshaped, params(d)))
end

function unshaped_params(ntd::NamedTupleDist)
    map(unshaped_params, ntd)
end

function unshaped_params(dp::Product)
    unshaped(map(unshaped_params, dp.v))
end
