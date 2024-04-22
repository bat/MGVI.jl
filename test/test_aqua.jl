# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import MGVI

#Test.@testset "Package ambiguities" begin
#    Test.@test isempty(Test.detect_ambiguities(MGVI))
#end # testset

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        MGVI,
        ambiguities = false,
        unbound_args = false # Detects unbounds args in with {N,T} where N might be zero
    )
end # testset
