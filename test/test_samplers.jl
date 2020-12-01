# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

Test.@testset "test_cmp_residual_samplers" begin

    model = ModelPolyfit.model
    true_params = ModelPolyfit.true_params
    starting_point = ModelPolyfit.starting_point

    Random.seed!(42)
    num_of_test_samples = 60

    fisher, jac = MGVInference.fisher_information_and_jac(model, true_params; jacobian_func=FullJacobianFunc)

    full_rs = FullResidualSampler(fisher, jac)
    full_samples_1 = rand(Random.GLOBAL_RNG, full_rs, num_of_test_samples)
    full_samples_2 = rand(Random.GLOBAL_RNG, full_rs, num_of_test_samples)
    p0 = BartlettTest(full_samples_1', full_samples_2') |> pvalue

    Test.@test p0 > 1E-3

    implicit_rs = ImplicitResidualSampler(fisher, jac)
    implicit_samples = rand(Random.GLOBAL_RNG, implicit_rs, num_of_test_samples)
    p = BartlettTest(full_samples_1', implicit_samples') |> pvalue

    Test.@test p > p0/10

end

Test.@testset "test_num_residual_sampler_full" begin

    Random.seed!(42)
    epsilon = 1E-5

    _simple_model_params = [1, 2.]

    function simple_model(p)
        Normal(p[1], p[2])
    end

    fisher, jac = MGVInference.fisher_information_and_jac(simple_model, _simple_model_params; jacobian_func=FullJacobianFunc)
    full_fisher = Matrix(jac' * fisher * jac)

    truth = [1/_simple_model_params[2]^2 0; 0 1/_simple_model_params[2]^4/2]

    Test.@test norm(truth - full_fisher) < epsilon

end
