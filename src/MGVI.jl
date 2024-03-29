# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

__precompile__(true)

"""
    MGVI

An implementation of the Metric Gaussian Variational Inference algorithm.
"""
module MGVI

import AutoDiffOperators: ADSelector, with_jacobian, gradient!_func
import ChainRulesCore
using Distributed
using LinearAlgebra
using Random
using SparseArrays
using Base.Iterators
using Distributions
using DistributionsAD
using FillArrays
using HeterogeneousComputing: GenContext, allocate_array
using LineSearches
using LinearMaps
using IterativeSolvers
using Optim
using Parameters
using PDMats
using PositiveFactorizations
import SparseArrays: blockdiag
using SparseArrays
using StaticArrays
using ValueShapes
using DocStringExtensions

using ChainRulesCore: AbstractTangent, Tangent, NoTangent, ZeroTangent, ProjectTo, AbstractThunk, unthunk
import Statistics: mean

include("util.jl")
include("mgvi_context.jl")
include("custom_linear_maps.jl")
include("shapes.jl")
include("information.jl")
include("residual_samplers.jl")
include("newtoncg.jl")
include("mgvi_impl.jl")

end # module
