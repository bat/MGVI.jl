# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI

using AutoDiffOperators
using Random
import DistributionsAD  # required for Zygote AD through Distributions
import Zygote

if !isdefined(Main, :ModelPolyfit)
    include("test_models/model_polyfit.jl")
    import .ModelPolyfit
end

Test.@testset "test_pareto_diagnostic" begin
    # raw log-ratio method:
    log_ratios = 0.1 .* randn(Xoshiro(21), 200)
    diag = pareto_diagnostic(log_ratios)
    @test diag.pareto_shape isa Real
    @test length(diag.weights) == length(log_ratios)
    @test sum(diag.weights) ≈ 1

    # metric-Gaussian diagnostic on an MGVI fit:
    context = MGVIContext(ADSelector(Zygote))
    model = ModelPolyfit.model
    center = ModelPolyfit.starting_point
    rng = Xoshiro(145)
    data = rand(rng, model(ModelPolyfit.true_params), 1)[1]

    config = MGVIConfig(optimizer = MGVI.NewtonCG())
    result = nothing
    for _ in 1:5
        result, center = mgvi_step(model, data, 12, center, config, context)
    end

    diag = pareto_diagnostic(model, data, result.samples, center, context)
    @test diag.pareto_shape isa Real
    @test isfinite(diag.pareto_shape)
    @test length(diag.weights) == size(result.samples, 2)
    @test sum(diag.weights) ≈ 1
end
