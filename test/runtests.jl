# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package MGVI" begin

include("test_mgvi_impl.jl")
include("test_jacobians.jl")
include("information/test_information.jl")
include("test_samplers.jl")
include("test_normal_mvnormal.jl")

end # testset
