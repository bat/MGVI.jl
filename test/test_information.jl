# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Random
using Distributions
using LinearAlgebra

using MGVI

using Test

Random.seed!(42)

@testset "test_fisher_information" begin
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

    @test sum((mgvi_fi - res) .* (mgvi_fi - res)) < 1E-5
end
