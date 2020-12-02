# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


# fisher information MC implementation

function _grad_logpdf(model::Function, params::AbstractVector, data_point)
    g = Zygote.gradient(dp -> logpdf(model(dp), data_point), params)[1]
    g_flat = reduce(vcat, g)
    g_flat * g_flat'
end

function fisher_information_mc(model::Function, params::AbstractVector, n::Integer)
    dist = model(params)
    sample() = _grad_logpdf(model, params, rand(dist))

    res = [sample()/n for _ in 1:Threads.nthreads()]
    Threads.@threads for i in 1:n-Threads.nthreads()
        res[Threads.threadid()] += sample()/n
    end

    sum(res)
end

# end fisher information mc


Test.@testset "test_fisher_with_mc" begin

    Random.seed!(42)
    epsilon = 5E-2
    num_runs = 100000

    # test univariate normal
    params = [0.1, 0.2]
    model = p -> Normal(p...)
    res = MGVI.fisher_information(model(params))
    truth = fisher_information_mc(model, params, num_runs)
    Test.@test norm((Matrix(res) - truth)) / norm(truth) < epsilon

    # test exponential
    params = [0.3]
    model = p -> Exponential(p...)
    res = MGVI.fisher_information(model(params))
    truth = fisher_information_mc(model, params, num_runs)
    Test.@test norm((Matrix(res)[1] - truth)) / norm(truth) < epsilon

end

Test.@testset "test_fisher_mvnormal_explicit" begin

    Random.seed!(42)
    epsilon = 1E-5

    mean = rand(2)
    variance_sqrt = rand(2, 2)
    variance = adjoint(variance_sqrt)*variance_sqrt

    mean_dims = length(mean)
    var_dims = length(variance)
    dims = mean_dims + var_dims

    res = zeros(dims, dims)
    inv_var = inv(variance)
    res[1:mean_dims, 1:mean_dims] .= inv_var
    for m in 1:var_dims
        for n in 1:var_dims
            m_mat = zeros(mean_dims, mean_dims)
            m_mat[m] = 1
            n_mat = zeros(mean_dims, mean_dims)
            n_mat[n] = 1
            res[mean_dims + m, mean_dims + n] = tr(inv_var * m_mat * inv_var * n_mat)/2
        end
    end

    mgvi_fi = MGVI.fisher_information(MvNormal(mean, variance))

    Test.@test norm(Matrix(mgvi_fi - res)) < epsilon

end

Test.@testset "test_fisher_information_combinations" begin

    epsilon = 1E-5

    MGVI.fisher_information(MvNormal([0.1, 0.2], [2. 0.1; 0.1 4]))

    # test Product(Univariates)
    μ1, σ1 = 0.1, 0.2
    μ2, σ2 = 0.1, 0.3
    dists = [Normal(μ1, σ1), Normal(μ2, σ2)]
    res = MGVI.fisher_information(Product(dists))
    truth = blockdiag(MGVI.fisher_information.(dists)...)
    Test.@test norm(Matrix(res) - Matrix(truth)) < epsilon

    # test NamedTupleDist
    dists = NamedTupleDist(a=Normal(0.1, 0.2),
                           b=Product([Normal(0.1, 0.2), Normal(0.3, 0.1)]),
                           c=MvNormal([0.2, 0.3], [2. 0.1; 0.1 4.5]))
    res = MGVI.fisher_information(dists)
    truth = blockdiag((parent ∘ MGVI.fisher_information).(values(dists))...)
    Test.@test norm(Matrix(res) - Matrix(truth)) < epsilon

end
