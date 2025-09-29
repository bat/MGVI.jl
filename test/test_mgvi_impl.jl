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
    center = ModelPolyfit.starting_point

    rng = Xoshiro(145)
    data = rand(rng, model(true_params), 1)[1]

    config = MGVIConfig(
        linsolver = MGVI.MatrixInversion(),
        optimizer = MGVI.NewtonCG()
    )
    result, center = mgvi_step(model, data, 12, center, config, context)
    @test result.mnlp isa Real
    @test result.samples isa AbstractMatrix{<:Real}
    @test center isa AbstractVector{<:Real}

    config = MGVIConfig(
        linsolver = LinearSolve.KrylovJL_CG(),
        optimizer = MGVI.NewtonCG()
    )
    result, center = mgvi_step(model, data, 12, center, config, context)
    @test result.mnlp isa Real
    @test result.samples isa AbstractMatrix{<:Real}
    @test center isa AbstractVector{<:Real}

    config = MGVIConfig(
        linsolver = LinearSolve.KrylovJL_CG(),
        optimizer = Optimization.LBFGS()
    )
    result, center = mgvi_step(model, data, 12, center, config, context)
    @test result.mnlp isa Real
    @test result.samples isa AbstractMatrix{<:Real}
    @test center isa AbstractVector{<:Real}

    config = MGVIConfig(
        linsolver = LinearSolve.KrylovJL_CG(),
        optimizer = Optim.LBFGS()
    )
    result, center = mgvi_step(model, data, 12, center, config, context)
    @test result.mnlp isa Real
    @test result.samples isa AbstractMatrix{<:Real}
    @test center isa AbstractVector{<:Real}
end
