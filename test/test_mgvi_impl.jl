# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

Test.@testset "test_mgvi_optimize_step" begin

    model = ModelPolyfit.model
    true_params = ModelPolyfit.true_params
    starting_point = ModelPolyfit.starting_point

    rng = MersenneTwister(145)
    data = rand(rng, model(true_params), 1)[1]

    first_iteration = mgvi_kl_optimize_step(rng,
                                            model, data, starting_point;
                                            jacobian_func=FwdRevADJacobianFunc,
                                            residual_sampler=ImplicitResidualSampler)

    next_iteration = first_iteration
    next_iteration = mgvi_kl_optimize_step(rng,
                                           model, data, next_iteration.result;
                                           jacobian_func=FwdRevADJacobianFunc,
                                           residual_sampler=ImplicitResidualSampler)

end
