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
using DistributionsAD
import ForwardDiff
using LinearMaps
using IterativeSolvers
using Optim
using PDMats
using PositiveFactorizations
import SparseArrays: blockdiag
using SparseArrays
using StaticArrays
using ValueShapes
import Zygote

using ChainRulesCore: AbstractTangent, Tangent, NoTangent, ZeroTangent, ProjectTo, AbstractThunk, unthunk
import Statistics: mean

include("util.jl")
include("custom_linear_maps.jl")
include("shapes.jl")
include("jacobian_maps.jl")
include("information.jl")
include("residual_samplers.jl")
include("mgvi_impl.jl")
include("newtoncg.jl")

end # module
