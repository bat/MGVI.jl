# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

import Test
Test.@testset "Package MGVI" begin

include("test_mgvi_impl.jl")
include("test_information.jl")

end # testset
