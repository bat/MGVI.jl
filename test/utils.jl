# This file is a part of MGVInference.jl, licensed under the MIT License (MIT).

using Distributions
using LinearAlgebra

function sample_params(p_size::Int, nsamples::Int)
    p_dist = MvNormal(zeros(p_size), I)
    params = rand(p_dist, nsamples)
    params
end

function sample_data(f::Function, p_size::Int, nsamples::Int)
    params = sample_params(p_size, nsamples)
    data_samples = []
    for param in eachcol(params)
        model = f(param)
        data_sample = rand(model, 1)
        push!(data_samples, data_sample)
    end
    vcat(data_samples...)
end

export sample_params, sample_data
