# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

using Random
using Distributions
using ValueShapes

using MGVInference
include("test_models/model_polyfit.jl")

using Test

@testset "test_mgvi_optimize_step" begin

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
