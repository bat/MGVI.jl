# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI

using Random
using AutoDiffOperators
import LinearSolve, Zygote
import Optimization, Optim#, OptimizationOptimJL

if !isdefined(Main, :ModelPolyfit)
    include("test_models/model_polyfit.jl")
    import .ModelPolyfit
end

Test.@testset "test_mgvi_optimize_step" begin
    context = MGVIContext(ADSelector(Zygote))

    model = ModelPolyfit.model
    true_params = ModelPolyfit.true_params
    starting_point = ModelPolyfit.starting_point

    rng = MersenneTwister(145)
    data = rand(rng, model(true_params), 1)[1]

    first_iteration = mgvi_step(
        model, data, starting_point, context;
        linear_solver = LinearSolve.KrylovJL_CG(),
        optim_solver = MGVI.NewtonCG()
    )
    next_iteration = first_iteration

    next_iteration = mgvi_step(
        model, data, next_iteration.result, context;
        linear_solver = LinearSolve.KrylovJL_CG(),
        optim_solver = MGVI.NewtonCG()
    )

    next_iteration = mgvi_step(
        model, data, next_iteration.result, context;
        linear_solver = LinearSolve.KrylovJL_CG(),
        optim_solver = Optimization.LBFGS()
    )

    next_iteration = mgvi_step(
        model, data, next_iteration.result, context;
        linear_solver = LinearSolve.KrylovJL_CG(),
        optim_solver = Optim.LBFGS()
    )
end
