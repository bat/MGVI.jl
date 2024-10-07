# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module MGVIOptimizationExt

import Optimization

import MGVI

function MGVI._optimize(f::Function, adsel::ADSelector, curvature::Function, 
    x₀::AbstractVector, optimizer, optim_options::NamedTuple
)
    # optim_options = (maxiters = ..., maxtime = ..., abstol = ..., reltol = ...)
   
    # ToDo: Forward curvature/Hessian?

    adsel = reverse_ad_selector(adsel)
    optfunc = Optimization.OptimizationFunction(f, adsel)
    optprob = Optimization.OptimizationProblem(optfunc, x₀)

    optres = Optimization.solve(optprob, optimizer; optim_options...)
    x_result = oftype(x₀, optres.u)
    return x_result, optres
end


end # module MGVIOptimizationExt
