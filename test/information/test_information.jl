# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using MGVI

using LinearAlgebra
using Random
using SparseArrays
using ValueShapes

Test.@testset "test_fisher_values" begin

    include("information_utils.jl")

    Random.seed!(42)
    epsilon = 5E-2
    num_runs = 100000

    function test_univariate(dist, params)
        model = p -> dist(p...)
        res = MGVI.fisher_information(model(params))
        truth = fisher_information_mc(model, params, num_runs)
        Test.@test norm((Matrix(res) - truth)) / norm(truth) < epsilon
    end

    test_univariate(Normal, [0.1, 0.2])
    test_univariate(Exponential, [0.3])
    test_univariate(Poisson, [5.75])

    function test_mvnormal(dim)
        dim = 1
        cov = I*5 + Symmetric(rand(dim, dim))
        mean = rand(dim)
        params = vcat(mean, cov[:])
        model = p -> MvNormal(p[1:dim], reshape(p[dim+1:end], dim, dim))
        res = MGVI.fisher_information(model(params))
        explicit = explicit_mv_normal_fi(cov)
        truth = fisher_information_mc(model, params, num_runs)
        Test.@test norm((Matrix(res) - truth)) / norm(truth) < epsilon
        Test.@test norm((Matrix(res) - explicit)) / norm(explicit) < epsilon
    end

    test_mvnormal(1)  # test mvnormal 1d
    test_mvnormal(2)  # test mvnormal 2d
    test_mvnormal(3)  # test mvnormal 3d
end


Test.@testset "test_fisher_information_combinations" begin
    epsilon = 1E-5

    # test product_distribution(Univariates)
    μ1, σ1 = 0.1, 0.2
    μ2, σ2 = 0.1, 0.3
    dists = [Normal(μ1, σ1), Normal(μ2, σ2)]
    res = MGVI.fisher_information(Distributions.Product{Continuous, Normal{Float64}, Vector{Normal{Float64}}}(dists))
    truth = blockdiag(MGVI.fisher_information.(dists)...)
    Test.@test norm(Matrix(res) - Matrix(truth)) < epsilon

    # test product_distribution(Univariates)
    μ1, σ1 = 0.1, 0.2
    μ2, σ2 = 0.1, 0.3
    dists = [Normal(μ1, σ1) Normal(μ2, σ2); Normal(μ1, σ1) Normal(μ2, σ2)]
    res = MGVI.fisher_information(Distributions.Distributions.ProductDistribution(dists))
    truth = blockdiag(MGVI.fisher_information.(dists)...)
    Test.@test norm(Matrix(res) - Matrix(truth)) < epsilon

    # test NamedTupleDist
    dists = NamedTupleDist(a=Normal(0.1, 0.2),
                           b=Distributions.Product{Continuous, Normal{Float64}, Vector{Normal{Float64}}}([Normal(0.1, 0.2), Normal(0.3, 0.1)]),
                           c=MvNormal([0.2, 0.3], [2. 0.1; 0.1 4.5]))
    res = MGVI.fisher_information(dists)
    truth = blockdiag((MGVI.without_chol ∘ MGVI.fisher_information).(values(dists))...)
    Test.@test norm(Matrix(res) - Matrix(truth)) < epsilon
end
