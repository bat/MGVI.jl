# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package MGVI" begin
    include("test_aqua.jl")
    include("test_mgvi_impl.jl")
    # Mooncake rule compilation stalls on Julia 1.10 already during
    # precompilation and tends to lag behind Julia prereleases, so it
    # can't be a static test dependency:
    if VERSION >= v"1.11" && isempty(VERSION.prerelease)
        import Pkg
        Base.identify_package("Mooncake") === nothing && Pkg.add("Mooncake")
        include("test_mooncake.jl")
    end
    include("test_jacobians.jl")
    include("information/test_information.jl")
    include("test_samplers.jl")
    include("test_geovi.jl")
    include("test_diagnostics.jl")
    # Reactant only supports 64-bit Linux and macOS, and some of its
    # dependencies break already during precompilation on other platforms,
    # so it can't be a static test dependency:
    if Sys.WORD_SIZE == 64 && (Sys.islinux() || Sys.isapple()) && isempty(VERSION.prerelease)
        import Pkg
        Base.identify_package("Reactant") === nothing && Pkg.add("Reactant")
        include("test_reactant.jl")
    end
    include("test_docs.jl")
end # testset
