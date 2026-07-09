# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

__precompile__(true)

"""
    MGVI

An implementation of the Metric Gaussian Variational Inference algorithm.
"""
module MGVI

import Adapt
using AffineMaps: MulAdd
using AutoDiffOperators: ADSelector, with_jacobian, gradient_func, jvp_func, with_vjp_func
import ChainRulesCore
using LinearAlgebra
using Random
using SparseArrays
using DensityInterface
using Distributions
using FillArrays
using HeterogeneousComputing: GenContext, allocate_array, on_device
using IrrationalConstants: log2π, sqrt2, invsqrt2
using LineSearches
using LinearMaps
using Parameters
using PDMats
using PositiveFactorizations
import PSIS
import SparseArrays: blockdiag
using StaticArrays
using ValueShapes
using DocStringExtensions

import LinearSolve
using LinearSolve: solve, LinearProblem, KrylovJL_CG

using ChainRulesCore: AbstractTangent, Tangent, NoTangent, ZeroTangent, ProjectTo, AbstractThunk, unthunk
import Statistics: mean

using ReactantCore: within_compile

# Reactant's traced and concrete number types do not subtype Real yet,
# accept Number in the compilation-relevant signatures for now.
# ToDo: Change back to Real once Reactant number types subtype Real:
const RealLike = Number

include("util.jl")
include("mgvi_context.jl")
include("custom_linear_maps.jl")
include("shapes.jl")
include("information.jl")
include("residual_samplers.jl")
include("newtoncg.jl")
include("mgvi_impl.jl")
include("mgvi_prepared.jl")
include("geovi_impl.jl")
include("diagnostics.jl")

end # module
