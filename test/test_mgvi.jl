# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

using Distributions
using ValueShapes

using MGVInference
using Test


true_params = [1.8, 0.5, 1.4, 2.5, 1.2]

function full_model(p)
    x1_grid = [Float64(i) for i in 1:10]
    dist1 = Product(Normal.(p[1] .+ p[2] .* x1_grid .+ p[3] .* x1_grid.^2/300. .+ p[4] .* x1_grid.^3/1000., exp(p[5])))

    x2_grid = [i + 0.5 for i in 1:10]
    dist2 = Product(Normal.(p[1] .+ p[2] .* x2_grid .+ p[3] .* x2_grid.^2/300. .+ p[4] .* x2_grid.^3/1000., exp(p[5])))

    NamedTupleDist(a=dist1,
                   b=dist2)
end

@testset "test_mgvi_optimize_step" begin
    data = sample_data(full_model, true_params, 100)
    starting_point = sample_params(true_params, 1)

    first_iteration = mgvi_kl_optimize_step(full_model, data, starting_point)
    next_iteration = mgvi_kl_optimize_step(full_model, data, first_iteration)
end
