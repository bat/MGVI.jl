# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI

using Distributions
using LinearAlgebra
using Random
using ForwardDiff

if :ModelPolyfit ∉ names(Main)
    include("test_models/model_polyfit.jl")
    import .ModelPolyfit
end

Test.@testset "test_jacobians_consistency" begin
    let
        A = rand(4,6)
        f = let A=A; x -> (A*(x.^2).^2) end
        x = rand(6); l = rand(4); r = rand(6)
        J_ref = ForwardDiff.jacobian(f, x)
        
        J1 = @inferred FullJacobianFunc(f)(x)
        J2 = @inferred FwdRevADJacobianFunc(f)(x)
        J3 = @inferred FwdDerJacobianFunc(f)(x)

        for J in (J1, J2, J3)
            @test @inferred(Array(J)) ≈ J_ref
            @test @inferred(J' * l) ≈ J_ref' * l
            @test @inferred(J * r) ≈ J_ref * r
        end
    end

    epsilon = 1E-5
    Random.seed!(145)

    _flat_model = MGVI.flat_params ∘ ModelPolyfit.model
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
