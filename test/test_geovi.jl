# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI

using Distributions
using LinearAlgebra
using Random
using Statistics
using ValueShapes
using AutoDiffOperators
using PDMats: PDiagMat
import ForwardDiff, Zygote
import LinearSolve

if !isdefined(Main, :ModelPolyfit)
    include("test_models/model_polyfit.jl")
    import .ModelPolyfit
end

Test.@testset "test_euclidean_coords" begin
    # C'C == fisher_information for flat distribution families:
    for (d, p2d) in (
        (Normal(0.0, 0.7), p -> Normal(p...)),
        (Exponential(0.8), p -> Exponential(p...)),
        (Poisson(3.4), p -> Poisson(p...)),
        (MvNormal(zeros(2), PDiagMat([0.8, 1.7])), p -> MvNormal(p[1:2], PDiagMat(p[3:4]))),
    )
        λ = collect(MGVI.flat_params(d))
        C = ForwardDiff.jacobian(p -> MGVI.euclidean_coords(p2d(p)), λ)
        @test C'C ≈ Matrix(MGVI.fisher_information(d))
    end

    # blockwise stacking matches fisher_information for composite distributions:
    let
        d = NamedTupleDist(
            a = Poisson(3.4),
            b = product_distribution([Exponential(0.8), Exponential(1.2)])
        )
        p2d = p -> NamedTupleDist(
            a = Poisson(p[1]),
            b = product_distribution([Exponential(p[2]), Exponential(p[3])])
        )
        λ = collect(MGVI.flat_params(d))
        C = ForwardDiff.jacobian(p -> MGVI.euclidean_coords(p2d(p)), λ)
        @test C'C ≈ Matrix(MGVI.fisher_information(d))
    end

    # Zygote-traced evaluation (TuringDiagMvNormal path) matches the primal:
    let
        model = ModelPolyfit.model
        p = ModelPolyfit.true_params
        ec_primal = MGVI.euclidean_coords(model(p))
        ec_traced, _ = Zygote.pullback(ξ -> MGVI.euclidean_coords(model(ξ)), p)
        @test ec_traced ≈ ec_primal
    end
end


Test.@testset "test_geovi_linear_model" begin
    # For a forward model that is linear in the parameters the geoVI
    # residuals are exact MGVI residuals, distributed as N(0, (A'A/v + I)^-1):
    context = MGVIContext(ADSelector(Zygote))
    Random.seed!(42)

    A = randn(15, 3)
    v = 0.6
    lin_model(p::AbstractVector) = MvNormal(A * p, PDiagMat(fill(oftype(p[1] * 1.0, v), 15)))
    center = [0.3, -0.2, 0.5]

    smplr = GeoVISampler(lin_model, center, LinearSolve.KrylovJL_CG(), MGVI.NewtonCG(), context)
    residuals = MGVI.sample_geovi_residuals(smplr, 500)
    @test size(residuals) == (3, 1000)
    # antithetic pairs mirror exactly in the linear case:
    @test residuals[:, 1:2:end] ≈ -residuals[:, 2:2:end] rtol = 1e-4

    Σ_exact = inv(A'A ./ v + I)
    Σ_est = cov(residuals; dims = 2)
    @test norm(Σ_est - Σ_exact) / norm(Σ_exact) < 0.15
end


Test.@testset "test_geovi_nonlinear" begin
    # Log-rate Poisson model: the exact posterior is skewed, geoVI captures
    # the asymmetry while MGVI is symmetric by construction.
    context = MGVIContext(ADSelector(Zygote))
    Random.seed!(21)

    pois_model(ξ::AbstractVector) = Poisson(exp(ξ[1]))
    kdata = 3

    lp(ξ) = logpdf(Poisson(exp(ξ)), kdata) - ξ^2 / 2
    ξs = range(-5, 5; length = 20001)
    w = exp.(lp.(ξs))
    w ./= sum(w)
    m_exact = sum(w .* ξs)
    sd_exact = sqrt(sum(w .* (ξs .- m_exact) .^ 2))

    config = GeoVIConfig()
    center = [1.0]
    result = nothing
    for _ in 1:6
        result, center = geovi_step(pois_model, kdata, 50, center, config, context)
    end
    @test result.mnlp isa Real
    @test size(result.samples, 2) == 100

    smpls = vec(geovi_sample(pois_model, kdata, 1000, center, config, context))
    m, sd = mean(smpls), std(smpls)
    skew = mean(((smpls .- m) ./ sd) .^ 3)
    @test abs(m - m_exact) < 0.15
    @test abs(sd - sd_exact) < 0.15
    @test skew < -0.1
end


Test.@testset "test_geovi_step" begin
    context = MGVIContext(ADSelector(Zygote))

    model = ModelPolyfit.model
    true_params = ModelPolyfit.true_params
    center = ModelPolyfit.starting_point

    rng = Xoshiro(145)
    data = rand(rng, model(true_params), 1)[1]

    config = GeoVIConfig()
    first_mnlp = nothing
    result = nothing
    for i in 1:4
        result, center = geovi_step(model, data, 8, center, config, context)
        i == 1 && (first_mnlp = result.mnlp)
    end
    @test result.mnlp isa Real
    @test result.samples isa AbstractMatrix{<:Real}
    @test size(result.samples, 2) == 16
    @test center isa AbstractVector{<:Real}
    @test result.mnlp < first_mnlp
end
