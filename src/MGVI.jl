# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

__precompile__(true)

"""
    MGVI

An implementation of the Metric Gaussian Variational Inference algorithm.
"""
module MGVI

import ChainRulesCore
using Distributed
using LinearAlgebra
using Random
using SparseArrays
using Base.Iterators
using Distributions
import ForwardDiff
using LinearMaps
using IterativeSolvers
using Optim
using PositiveFactorizations
using Requires
import SparseArrays: blockdiag
using SparseArrays
using StaticArrays
using ValueShapes
import Zygote

import Requires

include("custom_linear_maps.jl")
include("shapes.jl")
include("jacobian_maps.jl")
include("information.jl")
include("residual_samplers.jl")
include("mgvi_impl.jl")

function __init__()
    @require DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c" include("distributionsad_support.jl")
end

end # module
