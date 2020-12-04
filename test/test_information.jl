# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

Test.@testset "test_fisher_values" begin

    Random.seed!(42)
    epsilon = 5E-2
    num_runs = 100000

    # test univariate normal
    params = [0.1, 0.2]
    model = p -> Normal(p...)
    res = MGVInference.fisher_information(model(params))
    truth = fisher_information_mc(model, params, num_runs)
    Test.@test norm((Matrix(res) - truth)) / norm(truth) < epsilon

    # test exponential
    params = [0.3]
    model = p -> Exponential(p...)
    res = MGVInference.fisher_information(model(params))
    truth = fisher_information_mc(model, params, num_runs)
    Test.@test norm((Matrix(res) - truth)) / norm(truth) < epsilon

    # test mv normal
    dim = 1
    cov = I*5 + Symmetric(rand(dim, dim))
    mean = rand(dim)
    params = vcat(mean, cov[:])
    model = p -> MvNormal(p[1:dim], reshape(p[dim+1:end], dim, dim))
    res = MGVInference.fisher_information(model(params))
    explicit = explicit_mv_normal_fi(cov)
    truth = fisher_information_mc(model, params, num_runs)
    Test.@test norm((Matrix(res) - truth)) / norm(truth) < epsilon
    Test.@test norm((Matrix(res) - explicit)) / norm(explicit) < epsilon

    # test mv normal
    dim = 2
    cov = Matrix(I*5 + Symmetric(rand(dim, dim)))
    mean = rand(dim)
    params = vcat(mean, cov[:])
    model = p -> MvNormal(p[1:dim], reshape(p[dim+1:end], dim, dim))
    res = MGVInference.fisher_information(model(params))
    truth = fisher_information_mc(model, params, num_runs)
    explicit = explicit_mv_normal_fi(cov)
    Test.@test norm((Matrix(res) - truth)) / norm(truth) < epsilon
    Test.@test norm((Matrix(res) - explicit)) / norm(explicit) < epsilon

    # test mv normal
    dim = 3
    cov = I*5 + Symmetric(rand(dim, dim))
    mean = rand(dim)
    params = vcat(mean, cov[:])
    model = p -> MvNormal(p[1:dim], reshape(p[dim+1:end], dim, dim))
    res = MGVInference.fisher_information(model(params))
    truth = fisher_information_mc(model, params, num_runs)
    explicit = explicit_mv_normal_fi(cov)
    Test.@test norm((Matrix(res) - truth)) / norm(truth) < epsilon
    Test.@test norm((Matrix(res) - explicit)) / norm(explicit) < epsilon

end

Test.@testset "test_fisher_information_combinations" begin

    epsilon = 1E-5

    MGVInference.fisher_information(MvNormal([0.1, 0.2], [2. 0.1; 0.1 4]))

    # test Product(Univariates)
    μ1, σ1 = 0.1, 0.2
    μ2, σ2 = 0.1, 0.3
    dists = [Normal(μ1, σ1), Normal(μ2, σ2)]
    res = MGVInference.fisher_information(Product(dists))
    truth = blockdiag(MGVInference.fisher_information.(dists)...)
    Test.@test norm(Matrix(res) - Matrix(truth)) < epsilon

    # test NamedTupleDist
    dists = NamedTupleDist(a=Normal(0.1, 0.2),
                           b=Product([Normal(0.1, 0.2), Normal(0.3, 0.1)]),
                           c=MvNormal([0.2, 0.3], [2. 0.1; 0.1 4.5]))
    res = MGVInference.fisher_information(dists)
    truth = blockdiag((parent ∘ MGVInference.fisher_information).(values(dists))...)
    Test.@test norm(Matrix(res) - Matrix(truth)) < epsilon

end
