# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

Test.@testset "test_fisher_information" begin

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

    mgvi_fi = MGVInference.fisher_information(MvNormal(mean, variance))

    Test.@test norm(Matrix(mgvi_fi - res)) < epsilon

end
