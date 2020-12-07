# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

using MGVInference

using LinearAlgebra
using Random

if :ModelPolyfit ∉ names(Main)
    include("test_models/model_polyfit.jl")
    import .ModelPolyfit
end

Test.@testset "test_jacobians_consistency" begin


    epsilon = 1E-5
    Random.seed!(145)

    _flat_model = MGVInference._dists_flat_params_getter(ModelPolyfit.model)
    true_params = ModelPolyfit.true_params

    full_jac = FullJacobianFunc(_flat_model)(true_params)
    fwdder_jac = FwdDerJacobianFunc(_flat_model)(true_params)
    fwdrevad_jac = FwdRevADJacobianFunc(_flat_model)(true_params)

    for i in 1:min(size(full_jac)...)

        vec = rand(size(full_jac, 2))

        Test.@test norm(fwdder_jac*vec - full_jac*vec) < epsilon

        Test.@test norm(fwdrevad_jac*vec - full_jac*vec) < epsilon

    end

end
