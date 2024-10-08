# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module MGVIOptimExt

import Optim

import MGVI

using AutoDiffOperators: ADSelector, gradient!_func


function MGVI._optimize(f::Function, adsel::ADSelector, curvature::Function, 
    x₀::AbstractVector, optimizer::Optim.AbstractOptimizer, 
    optim_options::NamedTuple
)
    ∇f! = gradient!_func(f, adsel)
    res = Optim.optimize(f, ∇f!, x₀, optimizer, Optim.Options(;optim_options...))
    x_res = oftype(x₀, Optim.minimizer(res))
    x_res, res
end


end # module MGVIOptimExt
