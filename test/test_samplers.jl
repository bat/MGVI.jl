# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

Test.@testset "test_cmp_residual_samplers" begin

    model = ModelPolyfit.model
    true_params = ModelPolyfit.true_params
    starting_point = ModelPolyfit.starting_point

    Random.seed!(42)
    num_of_test_samples = 60

    fisher, jac = MGVI.fisher_information_and_jac(model, true_params; jacobian_func=FullJacobianFunc)

    full_rs = FullResidualSampler(fisher, jac)
    full_samples_1 = rand(Random.GLOBAL_RNG, full_rs, num_of_test_samples)
    full_samples_2 = rand(Random.GLOBAL_RNG, full_rs, num_of_test_samples)
    p0 = BartlettTest(full_samples_1', full_samples_2') |> pvalue

    Test.@test p0 > 5E-2

    implicit_rs = ImplicitResidualSampler(fisher, jac)
    implicit_samples = rand(Random.GLOBAL_RNG, implicit_rs, num_of_test_samples)
    p = BartlettTest(full_samples_1', implicit_samples') |> pvalue

    Test.@test p > p0/10

end
