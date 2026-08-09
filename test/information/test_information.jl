# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using MGVI
using Test

using Distributions
using LinearAlgebra
using Random
using ValueShapes
import MatrixShapedOperators

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

    function test_mvnormal_diag(dim)
        vars = rand(dim) .+ 0.5
        mn = rand(dim)
        params = vcat(mn, vars)
        model = p -> MvNormal(p[1:dim], PDiagMat(p[dim+1:end]))
        res = MGVI.fisher_information(model(params))
        truth = fisher_information_mc(model, params, num_runs)
        Test.@test norm((Matrix(res) - truth)) / norm(truth) < epsilon
    end

    test_mvnormal_diag(2)
    test_mvnormal_diag(3)

    function test_mvnormal_scal(dim)
        v = rand() + 0.5
        mn = rand(dim)
        params = vcat(mn, v)
        model = p -> MvNormal(p[1:dim], ScalMat(dim, p[dim+1]))
        res = MGVI.fisher_information(model(params))
        truth = fisher_information_mc(model, params, num_runs)
        Test.@test norm((Matrix(res) - truth)) / norm(truth) < epsilon
    end

    test_mvnormal_scal(3)
end


Test.@testset "test_fisher_information_combinations" begin
    epsilon = 1E-5

    _dense_blockdiag(As...) = cat(As...; dims = (1, 2))

    # test product_distribution(Univariates)
    μ1, σ1 = 0.1, 0.2
    μ2, σ2 = 0.1, 0.3
    dists = [Normal(μ1, σ1), Normal(μ2, σ2)]
    res = MGVI.fisher_information(Distributions.Product{Continuous, Normal{Float64}, Vector{Normal{Float64}}}(dists))
    truth = _dense_blockdiag(Matrix.(MGVI.fisher_information.(dists))...)
    Test.@test norm(Matrix(res) - truth) < epsilon
    # all-diagonal factors collapse into a single diagonal factor:
    Test.@test MatrixShapedOperators.asmatrix(MGVI.rowgram_factor(res)) isa Diagonal

    # test product_distribution(Univariates)
    μ1, σ1 = 0.1, 0.2
    μ2, σ2 = 0.1, 0.3
    dists = [Normal(μ1, σ1) Normal(μ2, σ2); Normal(μ1, σ1) Normal(μ2, σ2)]
    res = MGVI.fisher_information(Distributions.Distributions.ProductDistribution(dists))
    truth = _dense_blockdiag(Matrix.(MGVI.fisher_information.(vec(dists)))...)
    Test.@test norm(Matrix(res) - truth) < epsilon

    # test NamedTupleDist
    dists = NamedTupleDist(a=Normal(0.1, 0.2),
                           b=Distributions.Product{Continuous, Normal{Float64}, Vector{Normal{Float64}}}([Normal(0.1, 0.2), Normal(0.3, 0.1)]),
                           c=MvNormal([0.2, 0.3], [2. 0.1; 0.1 4.5]))
    res = MGVI.fisher_information(dists)
    Test.@test res isa MatrixShapedOperators.RowGramOperator
    truth = _dense_blockdiag(Matrix.(map(MGVI.fisher_information, values(dists)))...)
    Test.@test norm(Matrix(res) - truth) < epsilon
end


Test.@testset "fisher information boundary" begin
    # Plain-matrix returns from fisher_information specializations get
    # wrapped and support rowgram_factor, unsupported return types get
    # a clear error at the boundary:
    M = [2.0 0.5; 0.5 1.0]
    op = MGVI._as_fisher_operator(M)
    Test.@test op isa MatrixShapedOperators.MatrixShapedOperator
    F = Matrix(MGVI.rowgram_factor(op))
    Test.@test F * F' ≈ M
    Test.@test MGVI._as_fisher_operator(op) === op
    Test.@test_throws ArgumentError MGVI._as_fisher_operator((1.0, 2.0))
end
