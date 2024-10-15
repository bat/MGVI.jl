# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module MGVIOptimExt

import Optim

import MGVI

using AutoDiffOperators: ADSelector, gradient!_func


function MGVI._optimize(f::Function, adsel::ADSelector, curvature::Function, 
    x₀::AbstractVector, optimizer::Optim.AbstractOptimizer, 
    optimization_opts::NamedTuple
)
    ∇f! = gradient!_func(f, adsel)
    res = Optim.optimize(f, ∇f!, x₀, optimizer, Optim.Options(;optimization_opts...))
    x_res = oftype(x₀, Optim.minimizer(res))
    f_x_res = Optim.minimum(res)
    # @assert f_x_res == f(x_res)
    return x_res, f_x_res, res
end


end # module MGVIOptimExt
