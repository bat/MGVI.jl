# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

import Test
Test.@testset "Package MGVInference" begin

include("test_mgvi.jl")
include("test_information.jl")

end # testset
