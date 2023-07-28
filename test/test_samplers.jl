# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI

using HypothesisTests
using Random

import Zygote

if :ModelPolyfit ∉ names(Main)
    include("test_models/model_polyfit.jl")
    import .ModelPolyfit
end

Test.@testset "test_cmp_residual_samplers" begin
    context = MGVIContext(ADModule(:Zygote))

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

    implicit_rs = MGVI.ResidualSampler(model, true_params, MGVI.IterativeSolversCG(), context)
    implicit_samples = MGVI.sample_residuals(implicit_rs, num_of_test_samples)
    B1 = BartlettTest(full_samples_1', implicit_samples')

    Test.@test B1.L′ > B0.L′/10
    Test.@test B1.L′ < B0.L′*10
end
