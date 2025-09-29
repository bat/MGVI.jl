# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module MGVIOptimExt

using Optim: Optim, OnceDifferentiable

import MGVI

using AutoDiffOperators: ADSelector, reverse_adtype


function MGVI._optimize(f::Function, adsel::ADSelector, curvature::Function, 
    x₀::AbstractVector, optimizer::Optim.AbstractOptimizer, 
    optimization_opts::NamedTuple
)
    ad = reverse_adtype(adsel)
    target = OnceDifferentiable(f, x₀, autodiff = ad)
    res = Optim.optimize(target, x₀, optimizer, Optim.Options(;optimization_opts...))
    x_res = oftype(x₀, Optim.minimizer(res))
    f_x_res = Optim.minimum(res)
    # @assert f_x_res == f(x_res)
    return x_res, f_x_res, res
end


end # module MGVIOptimExt
