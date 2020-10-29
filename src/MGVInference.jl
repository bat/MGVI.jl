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
using Statistics

using ArgCheck
using ArraysOfArrays
using BlockDiagonals
using ChainRules
using ChainRulesCore
using DiffResults
using Distributions
using DistributionsAD
using FFTW
using FillArrays
using FiniteDiff
using ForwardDiff
using NLSolversBase
using Optim
using Parameters
using PositiveFactorizations
using ProgressMeter
using Random123
using SparseDiffTools
using SparsityDetection
using Statistics
using StatsBase
using ValueShapes
using Zygote
using ZygoteRules

include("mgvi.jl")

end # module
