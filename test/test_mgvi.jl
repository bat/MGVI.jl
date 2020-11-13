# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

using Random
using Distributions
using ValueShapes

using MGVInference
using Test

include("test_models/polyfit.jl")

@testset "test_mgvi_optimize_step" begin

    rng = MersenneTwister(145)
    data = rand(rng, full_model(true_params), 1)[1]

    first_iteration = mgvi_kl_optimize_step(rng,
                                            full_model, data, starting_point;
                                            jacobian_func=FwdRevADJacobianFunc,
                                            residual_sampler=ImplicitResidualSampler)

    next_iteration = first_iteration
    next_iteration = mgvi_kl_optimize_step(rng,
                                           full_model, data, next_iteration.result;
                                           jacobian_func=FwdRevADJacobianFunc,
                                           residual_sampler=ImplicitResidualSampler)

end
