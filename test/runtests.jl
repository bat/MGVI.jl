# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

using Random
using Distributions
using DistributionsAD
using HypothesisTests
using LinearAlgebra
using SparseArrays
using ValueShapes
import Zygote

using MGVInference

include("test_models/model_polyfit.jl")
import .ModelPolyfit

include("test_models/model_fft_gp.jl")
import .ModelFFTGP

import Test

Test.@testset "Package MGVInference" begin

include("utils.jl")
include("test_mgvi.jl")
include("test_jacobians.jl")
include("test_information.jl")
include("test_samplers.jl")

end # testset
