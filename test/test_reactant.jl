# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI
using LinearAlgebra
using Random
using AutoDiffOperators: ADSelector
using HeterogeneousComputing: GenContext
import DensityInterface
import Enzyme, Reactant, MLDataDevices

# CPU is available everywhere and sufficient to test tracing compatibility:
Reactant.set_default_backend("cpu")

# A diagonal-normal forward model that avoids Distributions/PDMats, which
# are not compatible with program tracing yet:
struct _DiagNormalLike{TM,TV}
    μ::TM
    var::TV
end
DensityInterface.DensityKind(::_DiagNormalLike) = DensityInterface.HasDensity()
function DensityInterface.logdensityof(d::_DiagNormalLike, x)
    -(sum((x .- d.μ) .^ 2 ./ d.var .+ log.(d.var)) + length(d.μ) * log(2π)) / 2
end
MGVI.flat_params(d::_DiagNormalLike) = vcat(d.μ, d.var)
MGVI.fisher_information(d::_DiagNormalLike) =
    MGVI.PDLinMapWithChol(Diagonal(vcat(1 ./ d.var, 1 ./ (2 .* d.var .^ 2))))

const _rct_A = randn(Xoshiro(11), 12, 4)
_rct_model(ξ) = _DiagNormalLike(_rct_A * ξ, exp.(0.1 .* (_rct_A * ξ)) .+ 0.5)

Test.@testset "test_reactant" begin
    d_true = _rct_model(randn(Xoshiro(2), 4))
    data = d_true.μ .+ sqrt.(d_true.var) .* randn(Xoshiro(5), 12)
    center₀ = 0.1 .* randn(Xoshiro(7), 4)

    config = MGVIConfig(optimizer = MGVI.NewtonCG(
        steps = 2, cg_maxiter = 4, cg_steps_traced = 4,
        linesearcher = MGVI.BacktrackingLineSearch(maxsteps = 10)
    ))
    # identically seeded contexts make host and compiled steps draw the
    # same residual samples:
    ctx_host = MGVIContext(GenContext(Xoshiro(99)), ADSelector(Enzyme))
    ctx_dev = MGVIContext(GenContext(Xoshiro(99)), ADSelector(Enzyme))

    prep_host = mgvi_prepare(_rct_model, data, 3, center₀, config, ctx_host)
    res_h1, center_h1 = mgvi_step(prep_host, center₀)
    res_h2, center_h2 = mgvi_step(prep_host, center_h1)

    prep_dev = mgvi_prepare(
        _rct_model, data, 3, center₀, config, ctx_dev;
        device = MLDataDevices.ReactantDevice()
    )
    res_d1, center_d1 = mgvi_step(prep_dev, center₀)
    res_d2, center_d2 = mgvi_step(prep_dev, center_d1)

    @test res_d1.mnlp ≈ res_h1.mnlp rtol = 1e-6
    @test res_d2.mnlp ≈ res_h2.mnlp rtol = 1e-6
    @test center_d1 ≈ center_h1 rtol = 1e-6
    @test center_d2 ≈ center_h2 rtol = 1e-6
end
