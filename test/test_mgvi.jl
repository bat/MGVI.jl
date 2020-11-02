# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

using Random
using Distributions
using ValueShapes

using MGVInference
using Test


param_size = 4

function full_model(p)
    x1_grid = [Float64(i) for i in 1:25]
    dist1 = Product(Normal.(p[1]*10 .+ p[2] .* x1_grid .+ p[3] .* x1_grid.^2/5. .+ p[4] .* x1_grid.^3/10., 1.2))

    x2_grid = [i + 0.5 for i in 1:15]
    dist2 = Product(Normal.(p[1]*10 .+ p[2] .* x2_grid .+ p[3] .* x2_grid.^2/5. .+ p[4] .* x2_grid.^3/10., 1.2))

    NamedTupleDist(a=dist1,
                   b=dist2)
end

@testset "test_mgvi_optimize_step" begin

    true_params = randn(param_size)
    data = rand(full_model(true_params), 1)[1]
    starting_point = randn(param_size)*0.1

    first_iteration = mgvi_kl_optimize_step(full_model, data, starting_point)
    next_iteration = first_iteration

    for i in 1:20
        next_iteration = mgvi_kl_optimize_step(full_model, data, next_iteration)
    end
    next_iteration

end
