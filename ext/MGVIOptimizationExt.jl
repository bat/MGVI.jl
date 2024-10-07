# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module MGVIOptimizationExt

import Optimization

import MGVI
using AutoDiffOperators: ADSelector, reverse_ad_selector


struct _OptimizationTargetFunc{F} <: Function
    f::F
end

_OptimizationTargetFunc(::Type{F}) where F = _OptimizationTargetFunc{Type{F}}(F)

(ft::_OptimizationTargetFunc)(x, p) = ft.f(x)

function MGVI._optimize(f::Function, adsel::ADSelector, curvature::Function, 
    x₀::AbstractVector, optimizer, optimization_opts::NamedTuple
)
    # optimization_opts = (maxiters = ..., maxtime = ..., abstol = ..., reltol = ...)
   
    # ToDo: Forward curvature/Hessian?

    adsel = reverse_ad_selector(adsel)
    optfunc = Optimization.OptimizationFunction(_OptimizationTargetFunc(f), adsel)
    optprob = Optimization.OptimizationProblem(optfunc, x₀)

    res = Optimization.solve(optprob, optimizer; optimization_opts...)
    x_res = oftype(x₀, res.u)
    # ToDo: Is there a way to make Optimization return f(x_res)?
    f_x_res = f(x_res)
    return x_res, f_x_res, res
end


end # module MGVIOptimizationExt
