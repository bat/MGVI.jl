# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package MGVInference" begin

include("test_mgvi.jl")
include("test_jacobians.jl")
include("information/test_information.jl")
include("test_samplers.jl")

end # testset
