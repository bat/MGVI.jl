# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

Test.@testset "test_fisher_information_value" begin

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

    μ, σ = 0.1, 0.2
    res = MGVI.fisher_information(Normal(μ, σ))
    truth = Diagonal([1/σ^2, 1/σ^4/2])
    Test.@test norm(Matrix(res) - truth) < epsilon

    MGVI.fisher_information(MvNormal([0.1, 0.2], [2. 0.1; 0.1 4]))

    μ1, σ1 = 0.1, 0.2
    μ2, σ2 = 0.1, 0.3
    res = MGVI.fisher_information(Product([Normal(0.1, 0.2), Normal(0.1, 0.3)]))
    truth = Diagonal([1/σ1^2, 1/σ1^4/2, 1/σ2^2, 1/σ2^4/2])
    Test.@test norm(Matrix(res) - truth) < epsilon

    MGVI.fisher_information(NamedTupleDist(a=Normal(0.1, 0.2),
                                                   b=Product([Normal(0.1, 0.2), Normal(0.3, 0.1)]),
                                                   c=MvNormal([0.2, 0.3], [2. 0.1; 0.1 4.5])))

end
