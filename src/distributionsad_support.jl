# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


function unshaped_params(d::DistributionsAD.TuringDenseMvNormal)
    μ = d.m
    σ = convert(AbstractMatrix, d.C)
    vcat(μ, _uppertriang_to_vec(σ))
end
