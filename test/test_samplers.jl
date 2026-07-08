# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI

using AutoDiffOperators
using Distributions
using HypothesisTests
using LinearAlgebra
using Random
using PDMats: PDiagMat

import DistributionsAD  # required for Zygote AD through Distributions
import ForwardDiff
import LinearSolve, Zygote

if :ModelPolyfit ∉ names(Main)
    include("test_models/model_polyfit.jl")
    import .ModelPolyfit
end

Test.@testset "test_cmp_residual_samplers" begin
    context = MGVIContext(ADSelector(Zygote))

    model = ModelPolyfit.model
    true_params = ModelPolyfit.true_params
    starting_point = ModelPolyfit.starting_point

    Random.seed!(42)
    num_of_test_samples = 60

    full_rs = MGVI.ResidualSampler(model, true_params, MGVI.MatrixInversion(), context)
    full_samples_1 = MGVI.sample_residuals(full_rs, num_of_test_samples)
    full_samples_2 = MGVI.sample_residuals(full_rs, num_of_test_samples)
    B0 = BartlettTest(full_samples_1', full_samples_2')
    Test.@test pvalue(B0) > 1E-2

    implicit_rs = MGVI.ResidualSampler(model, true_params, LinearSolve.KrylovJL_CG(), context)
    implicit_samples = MGVI.sample_residuals(implicit_rs, num_of_test_samples)
    B1 = BartlettTest(full_samples_1', implicit_samples')

    Test.@test B1.L′ > B0.L′/10
    Test.@test B1.L′ < B0.L′*10
end

Test.@testset "test_batched_residual_sampling" begin
    # The pure batched kernel must match a direct dense solve:
    Random.seed!(47)
    A = randn(8, 3)
    kmodel(ξ::AbstractVector) = product_distribution(Normal.(A * ξ, exp.(0.15 .* (A * ξ))))
    center = randn(3)
    n_smpls = 6
    sample_n = randn(16, n_smpls)
    sample_η = randn(3, n_smpls)

    fi = MGVI.fisher_information(kmodel(center))
    ℐm = Matrix(fi)
    Lm = Matrix(MGVI.cholesky_L(fi))
    J = ForwardDiff.jacobian(p -> MGVI.flat_params(kmodel(p)), center)
    X_ref = (J' * ℐm * J + I) \ (J' * (Lm * sample_n) + sample_η)

    for ad in (ADSelector(ForwardDiff), ADSelector(Zygote))
        X = MGVI.sample_residuals(kmodel, center, sample_n, sample_η, ad)
        Test.@test X ≈ X_ref rtol = 1e-8
    end

    # _batched_cg solves column-wise:
    M = J' * ℐm * J + I
    B = randn(3, 4)
    X = MGVI._batched_cg(P -> M * P, B, 20)
    Test.@test X ≈ M \ B rtol = 1e-8
end
