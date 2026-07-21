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

Test.@testset "test_mgvi_prepared_step_mooncake" begin
    # The prepared path builds ADJacobian operators and applies them to
    # sample batches via DI's batched pushforward/pullback:
    context = MGVIContext(ADSelector(Mooncake))

    model = ModelPolyfit.model
    center = ModelPolyfit.starting_point
    rng = Xoshiro(145)
    data = rand(rng, model(ModelPolyfit.true_params), 1)[1]

    config = MGVIConfig(
        optimizer = MGVI.NewtonCG(linesearcher = MGVI.BacktrackingLineSearch())
    )
    prepared = mgvi_prepare(model, data, 12, center, config, context)

    first_mnlp = nothing
    result = nothing
    for i in 1:5
        result, center = mgvi_step(prepared, center)
        i == 1 && (first_mnlp = result.mnlp)
    end
    @test result.mnlp isa Real
    @test result.samples isa AbstractMatrix{<:Real}
    @test center isa AbstractVector{<:Real}
    @test result.mnlp < first_mnlp
end

Test.@testset "test_geovi_prepared_step_mooncake" begin
    context = MGVIContext(ADSelector(Mooncake))

    model = ModelPolyfit.model
    center = ModelPolyfit.starting_point
    rng = Xoshiro(145)
    data = rand(rng, model(ModelPolyfit.true_params), 1)[1]

    config = GeoVIConfig(
        optimizer = MGVI.NewtonCG(linesearcher = MGVI.BacktrackingLineSearch()),
        sampling_optimizer = MGVI.NewtonCG(linesearcher = MGVI.BacktrackingLineSearch())
    )
    prepared = geovi_prepare(model, data, 8, center, config, context)

    first_mnlp = nothing
    result = nothing
    for i in 1:4
        result, center = geovi_step(prepared, center)
        i == 1 && (first_mnlp = result.mnlp)
    end
    @test result.mnlp isa Real
    @test center isa AbstractVector{<:Real}
    @test result.mnlp < first_mnlp
end
