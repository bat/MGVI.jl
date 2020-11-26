# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

__precompile__(true)

"""
    MGVInference

An implementation of the Metric Gaussian Variational Inference algorithm.
"""
module MGVInference

using Distributed
using LinearAlgebra
using Random
using SparseArrays
using Base.Iterators
using BlockDiagonals
using Distributions
import ForwardDiff
using LinearMaps
using IterativeSolvers
using Optim
using PDMats
using PositiveFactorizations
using SparseArrays
using StaticArrays
using ValueShapes
import Zygote

include("shapes.jl")
include("jacobian_maps.jl")
include("information.jl")
include("residual_samplers.jl")
include("mgvi.jl")

end # module
