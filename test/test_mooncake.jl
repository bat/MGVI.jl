# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI

using Random
using AutoDiffOperators
import Mooncake

if !isdefined(Main, :ModelPolyfit)
    include("test_models/model_polyfit.jl")
    import .ModelPolyfit
end

Test.@testset "test_mgvi_step_mooncake" begin
    # The full pipeline with a DifferentiationInterface-based backend that
    # relies on preparation: jvp/vjp for the Fisher metric, prepared
    # gradients in NewtonCG.
    context = MGVIContext(ADSelector(Mooncake))

    model = ModelPolyfit.model
    center = ModelPolyfit.starting_point
    rng = Xoshiro(145)
    data = rand(rng, model(ModelPolyfit.true_params), 1)[1]

    config = MGVIConfig(optimizer = MGVI.NewtonCG())
    result, center = mgvi_step(model, data, 3, center, config, context)
    @test result.mnlp isa Real
    @test result.samples isa AbstractMatrix{<:Real}
    @test center isa AbstractVector{<:Real}
end
