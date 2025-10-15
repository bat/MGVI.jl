# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module MGVIOptimizationBaseExt

import OptimizationBase

import MGVI
using AutoDiffOperators: ADSelector, reverse_adtype


struct _OptimizationTargetFunc{F} <: Function
    f::F
end
_OptimizationTargetFunc(::Type{F}) where F = _OptimizationTargetFunc{Type{F}}(F)

(ft::_OptimizationTargetFunc)(x, ::Any) = ft.f(x)


function MGVI._optimize(f::Function, adsel::ADSelector, curvature::Function, 
    x₀::AbstractVector, optimizer, optimization_opts::NamedTuple
)
    # optimization_opts = (maxiters = ..., maxtime = ..., abstol = ..., reltol = ...)
   
    # ToDo: Forward curvature/Hessian?

    f_target = _OptimizationTargetFunc(f)
    ad = reverse_adtype(adsel)
    optfunc = OptimizationBase.OptimizationFunction(f_target, ad)
    optprob = OptimizationBase.OptimizationProblem(optfunc, x₀)

    res = OptimizationBase.solve(optprob, optimizer; optimization_opts...)
    x_res = oftype(x₀, res.u)
    # ToDo: Is there a way to make OptimizationBase return f(x_res)?
    f_x_res = f(x_res)
    return x_res, f_x_res, res
end


end # module MGVIOptimizationBaseExt
