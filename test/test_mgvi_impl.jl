# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI

using Random
using LinearAlgebra
using AutoDiffOperators
import DistributionsAD  # required for Zygote AD through Distributions
import LinearSolve, Mooncake, Zygote
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

    # mgvi_kl_target is the mean negative log-posterior over antithetic pairs:
    residuals = rand(Xoshiro(7), 5, 3)
    mnlp = mgvi_kl_target(model, data, residuals)
    ref = -sum(
        MGVI.posterior_loglike(model, center + s * residuals[:, i], data)
        for i in 1:3, s in (+1, -1)
    ) / 6
    @test mnlp(center) ≈ ref
end

Test.@testset "test_mgvi_prepared_step" begin
    context = MGVIContext(ADSelector(Zygote))

    model = ModelPolyfit.model
    true_params = ModelPolyfit.true_params
    center = ModelPolyfit.starting_point

    rng = Xoshiro(145)
    data = rand(rng, model(true_params), 1)[1]

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
    @test size(result.samples) == (5, 24)
    @test center isa AbstractVector{<:Real}
    @test result.mnlp < first_mnlp
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

Test.@testset "test_newtoncg_linesearch_robustness" begin
    # A line search that fails (e.g. StrongWolfe unable to satisfy the Wolfe
    # conditions at a near-stationary point, as in the geoVI sampling solves)
    # must be treated as a zero step, not crash the optimization:
    throwing_ls(args...) = throw(MGVI.LineSearches.LineSearchException("forced failure", 1.0))
    f = x -> sum(abs2, x) / 2
    curvature = x -> Diagonal(ones(length(x)))
    optimizer = MGVI.NewtonCG(linesearcher = throwing_ls, steps = 3)
    x₀ = [1.0, -2.0, 0.5]
    x_res, f_res, res = MGVI._optimize(f, ADSelector(Zygote), curvature, x₀, optimizer, (;))
    @test x_res == x₀        # the failed search reduces to a zero step
    @test f_res == f(x₀)
    @test res.iterations == 1  # a zero step stops the optimization early
end

Test.@testset "test_newtoncg_forcing" begin
    # On a quadratic with exact curvature the Eisenstat–Walker residual
    # target is reached after a single cg iteration and the first NewtonCG
    # step lands on the optimum; the second step yields no improvement and
    # stops the optimization early:
    f = x -> sum(abs2, x) / 2
    curvature = x -> Diagonal(ones(length(x)))
    x₀ = [1.0, -2.0, 0.5]
    optimizer = MGVI.NewtonCG(absdelta = 1e-12)
    x_res, f_res, res = MGVI._optimize(f, ADSelector(Zygote), curvature, x₀, optimizer, (;))
    @test f_res ≈ 0 atol = 1e-20
    @test isapprox(x_res, zero(x₀), atol = 1e-10)
    @test res.iterations < optimizer.steps
    @test res.cg_iterations <= 10
end
