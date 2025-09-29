# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI

using Distributions
using LinearAlgebra
using Random
using AutoDiffOperators, LinearMaps
import ForwardDiff, Zygote

if !isdefined(Main, :ModelPolyfit)
    include("test_models/model_polyfit.jl")
    import .ModelPolyfit
end

Test.@testset "test_jacobians_consistency" begin
    let
        A = rand(4,6)
        f = let A=A; x -> (A*(x.^2).^2) end
        x = rand(6); l = rand(4); r = rand(6)
        J_ref = ForwardDiff.jacobian(f, x)
        
        _, J1 = @inferred with_jacobian(f, x, Matrix, ADSelector(ForwardDiff))
        _, J2 = @inferred with_jacobian(f, x, LinearMap, ADSelector(Zygote))
        _, J3 = @inferred with_jacobian(f, x, LinearMap, ADSelector(ForwardDiff))

        for J in (J1, J2, J3)
            @test @inferred(Matrix(J)) ≈ J_ref
            @test @inferred(J' * l) ≈ J_ref' * l
            @test @inferred(J * r) ≈ J_ref * r
        end
    end

    epsilon = 1E-5
    Random.seed!(145)

    _flat_model = MGVI.flat_params ∘ ModelPolyfit.model
    true_params = ModelPolyfit.true_params

    _, full_jac = @inferred with_jacobian(_flat_model, true_params, Matrix, ADSelector(ForwardDiff))
    _, fwdder_jac = @inferred with_jacobian(_flat_model, true_params, LinearMap, ADSelector(ForwardDiff))
    _, fwdrevad_jac = @inferred with_jacobian(_flat_model, true_params, LinearMap, ADSelector(Zygote))

    for i in 1:min(size(full_jac)...)
        vec = rand(size(full_jac, 2))
        @test norm(fwdder_jac*vec - full_jac*vec) < epsilon
        @test norm(fwdrevad_jac*vec - full_jac*vec) < epsilon
    end
end
