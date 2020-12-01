# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Random
using Distributions
using HypothesisTests
using LinearAlgebra
using ValueShapes

using MGVI

include("test_models/model_polyfit.jl")
import .ModelPolyfit

include("test_models/model_fft_gp.jl")
import .ModelFFTGP

import Test

Test.@testset "Package MGVI" begin

include("test_mgvi_impl.jl")
include("test_jacobians.jl")
include("test_information.jl")
include("test_samplers.jl")

end # testset
