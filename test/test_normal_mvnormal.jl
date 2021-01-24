# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using MGVI

using Distributions
using LinearAlgebra
using Random

Test.@testset "test_normal_mvnormal_jac" begin

    epsilon = 1E-5
    Random.seed!(145)

    true_params = randn(4)

    function _common_params(p)
        μ1 = p[1] - p[2]
        μ2 = p[2] * p[1]
        σ1 = p[3]^2 + 10.
        σ2 = (p[3] + p[4])^2 + 10.

        μ1, σ1, μ2, σ2
    end

    function normal_model(p)
        μ1, σ1, μ2, σ2 = _common_params(p)
        n1 = Normal(μ1, σ1)
        n2 = Normal(μ2, σ2)
        Product([n1, n2])
    end
    _flat_normal_model = MGVI._dists_flat_params_getter(normal_model)

    # NOTE: models are not the same. Σ[1, 1] = σ1^2, while here we assign σ1.
    # We do this because in this way it is easier to compare jacobians.
    # While in Normal, parameters are μ and σ, in MvNormal corresponding paramters
    # are μ and σ^2.
    function mvnormal_model(p)
        μ1, σ1, μ2, σ2 = _common_params(p)
        MvNormal([μ1, μ2], [σ1 0.;0. σ2])
    end
    _flat_mvnormal_model = MGVI._dists_flat_params_getter(mvnormal_model)

    normal_full_jac = FullJacobianFunc(_flat_normal_model)(true_params)
    normal_fwdder_jac = FwdDerJacobianFunc(_flat_normal_model)(true_params)
    normal_fwdrevad_jac = FwdRevADJacobianFunc(_flat_normal_model)(true_params)

    mvnormal_full_jac = FullJacobianFunc(_flat_mvnormal_model)(true_params)
    mvnormal_fwdder_jac = FwdDerJacobianFunc(_flat_mvnormal_model)(true_params)
    mvnormal_fwdrevad_jac = FwdRevADJacobianFunc(_flat_mvnormal_model)(true_params)

    for i in 1:min(size(normal_full_jac)...)

        vec = rand(size(normal_full_jac, 2))

        # We have to reorder jacobian for MvNormal and pick only relevant parts
        # While stacked Normals have parameters [μ1, σ1, μ2, σ2]
        # MvNormal is parametrized as [μ1, μ2, Σ11, Σ12, Σ22]
        # Moreover. In our example Σ12 = 0. Thus we skip this element
        mv_reorder = [1, 3, 2, 5]

        Test.@test norm(normal_full_jac*vec - (mvnormal_full_jac*vec)[mv_reorder]) < epsilon
        Test.@test norm(normal_fwdder_jac*vec - (mvnormal_fwdder_jac*vec)[mv_reorder]) < epsilon
        Test.@test norm(normal_fwdrevad_jac*vec - (mvnormal_fwdrevad_jac*vec)[mv_reorder]) < epsilon

    end

end

Test.@testset "test_normal_mvnormal_logpdf_der" begin


    epsilon = 1E-5
    Random.seed!(145)

    dim = 2
    data = randn(dim)
    pardim = dim + dim
    true_params = randn(pardim)

    function _common_params(p)
        μ1 = p[1] - p[2]
        μ2 = p[2] * p[1]
        σ1 = p[3]^2 + 10.
        σ2 = (p[3] + p[4])^2 + 10.

        μ1, σ1, μ2, σ2
    end

    function mvnormal_model(p)
        μ1, σ1, μ2, σ2 = _common_params(p)
        MvNormal([μ1, μ2], [σ1^2 0.;0. σ2^2])
    end
    mvnormal_logpdf = p -> logpdf(mvnormal_model(p), data)
    mvnormal_grad = zeros(pardim)
    MGVI._gradient_for_optim(mvnormal_logpdf)(mvnormal_grad, true_params)

    function normal_model(p)
        μ1, σ1, μ2, σ2 = _common_params(p)
        n1 = Normal(μ1, σ1)
        n2 = Normal(μ2, σ2)
        Product([n1, n2])
    end
    normal_par_dim = dim + dim
    normal_logpdf = p -> logpdf(normal_model(p), data)
    normal_grad = zeros(pardim)
    MGVI._gradient_for_optim(normal_logpdf)(normal_grad, true_params)

    Test.@test norm(normal_grad - mvnormal_grad) < epsilon

end
