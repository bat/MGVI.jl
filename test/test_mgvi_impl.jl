# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI

using Random
using LinearAlgebra
using AutoDiffOperators
import DistributionsAD  # required for Zygote AD through Distributions
import LinearSolve, Zygote
import OptimizationLBFGSB, Optim#, OptimizationOptimJL

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
        optimizer = OptimizationLBFGSB.LBFGSB()
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

Test.@testset "test_newtoncg_linesearch_robustness" begin
    # A line search that fails (e.g. StrongWolfe unable to satisfy the Wolfe
    # conditions at a near-stationary point, as in the geoVI sampling solves)
    # must be treated as a zero step, not crash the optimization:
    throwing_ls(args...) = throw(MGVI.LineSearches.LineSearchException("forced failure", 1.0))
    f = x -> sum(abs2, x) / 2
    curvature = x -> Diagonal(ones(length(x)))
    optimizer = MGVI.NewtonCG(linesearcher = throwing_ls, steps = 3)
    x₀ = [1.0, -2.0, 0.5]
    x_res, f_res, _ = MGVI._optimize(f, ADSelector(Zygote), curvature, x₀, optimizer, (;))
    @test x_res == x₀        # every step reduces to a zero step
    @test f_res == f(x₀)
end
