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
    _flat_normal_model = MGVI.flat_params ∘ normal_model

    # NOTE: models are not the same. Σ[1, 1] = σ1^2, while here we assign σ1.
    # We do this because in this way it is easier to compare jacobians.
    # While in Normal, parameters are μ and σ, in MvNormal corresponding paramters
    # are μ and σ^2.
    function mvnormal_model(p)
        μ1, σ1, μ2, σ2 = _common_params(p)
        MvNormal([μ1, μ2], [σ1 0.;0. σ2])
    end
    _flat_mvnormal_model = MGVI.flat_params ∘ mvnormal_model

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

    function normal_model(p)
        μ1, σ1, μ2, σ2 = _common_params(p)
        n1 = Normal(μ1, σ1)
        n2 = Normal(μ2, σ2)
        Product([n1, n2])
    end

    # est_res_sampler = MGVI._create_residual_sampler(normal_model, true_params;
                                               # residual_sampler=ImplicitResidualSampler,
                                               # jacobian_func=FwdRevADJacobianFunc,
                                               # residual_sampler_options=(;))
    # residual_samples = rand(Random.GLOBAL_RNG, est_res_sampler, 5)

    residual_samples = [
        1.57327   -0.545016   -0.468532  -0.148169  -0.0476087
        0.889662  -0.32206    -0.776208  -1.3703    -0.262721
        0.535246  -0.0152493   0.847152   0.723876  -0.0277677
        2.13597    0.252785   -0.278254   1.11853   -0.189659
    ]

    normal_par_dim = dim + dim
    normal_kl(params::AbstractVector) = MGVI.mgvi_kl(normal_model, data, residual_samples, params)
    normal_grad = zeros(pardim)
    MGVI._gradient_for_optim(normal_kl)(normal_grad, true_params)


    function mvnormal_model(p)
        μ1, σ1, μ2, σ2 = _common_params(p)
        MvNormal([μ1, μ2], [σ1^2 0.;0. σ2^2])
    end
    mvnormal_kl(params::AbstractVector) = MGVI.mgvi_kl(mvnormal_model, data, residual_samples, params)
    mvnormal_grad = zeros(pardim)
    MGVI._gradient_for_optim(mvnormal_kl)(mvnormal_grad, true_params)

    Test.@test norm(normal_grad - mvnormal_grad) < epsilon

end
