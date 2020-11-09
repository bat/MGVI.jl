# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

using Random
using Distributions
using ValueShapes

using MGVInference
using Test

include("test_models/polyfit.jl")

@testset "test_mgvi_optimize_step" begin

    data = rand(full_model(true_params), 1)[1]

    first_iteration = mgvi_kl_optimize_step(full_model, data, starting_point)

    next_iteration = first_iteration
    for i in 1:20
        next_iteration = mgvi_kl_optimize_step(full_model, data, next_iteration)
    end

end
