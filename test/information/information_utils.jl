# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Distributions
using DistributionsAD
using LinearAlgebra
import Zygote

"""
    _fi_cov_only_upper_params(s; offset=0)

Return indices of the covariance part of the FI matrix that
correspond to upper triangular part of the covariance matrix.

Specify offset to get indices shifted to the right and bottom by
the same offset. (useful to remove irrelevant cols from full FI)
"""
function _fi_cov_only_upper_params(s; offset=0)
    slices = []
    for i in 1:s
        start = (i-1)*s+1
        push!(slices, (start:start+i-1))
    end
    cov_slices = union(slices...) .+ offset
    relevant_slices = union(1:offset, cov_slices)
    relevant_slices
end

# fisher information MC implementation

function _grad_logpdf(model::Function, params::AbstractVector, data_point)
    g = Zygote.gradient(dp -> logpdf(model(dp), data_point), params)[1]
    g_flat = collect(g)
    g_flat * g_flat'
end

_cut_params(res, dist::Distribution) = res

"""
    _cut_params(res, dist::Distribution)

noop for most of the distributions.

for MvNormal it removes rows and cols from FI that correspond to lower triangular
part of the covariance matrix. Covariance matrix is symmetric, so lower triangular
parameters are redundand.
"""
function _cut_params(res, dist::MvNormal)
    mean_size = length(dist.μ)
    slices = _fi_cov_only_upper_params(mean_size; offset=mean_size)
    res[slices, slices]
end

function fisher_information_mc(model::Function, params::AbstractVector, n::Integer)
    dist = model(params)
    sample() = _grad_logpdf(model, params, rand(dist))/n

    res = zero(sample())
    res_lock = ReentrantLock()

    Threads.@threads for _ in 1:n
        s = sample()
        @lock res_lock res .+= s
    end

    _cut_params(res, dist)
end

# end fisher information mc

"""
    explicit_mv_normal_fi(variance::AbstractMatrix)

Compute fisher information of MvNormal based on the wiki article:

https://en.wikipedia.org/wiki/Fisher_information#Multivariate_normal_distribution
"""
function explicit_mv_normal_fi(variance::AbstractMatrix)
    var_dims = length(variance)
    mean_dims = size(variance, 1)
    dims = mean_dims + var_dims

    res = zeros(dims, dims)
    inv_var = inv(variance)
    res[1:mean_dims, 1:mean_dims] .= inv_var
    for m in 1:var_dims
        for n in 1:var_dims
            m_mat = zeros(mean_dims, mean_dims)
            m_mat[m] = 1
            m_mat = Symmetric(m_mat)
            n_mat = zeros(mean_dims, mean_dims)
            n_mat[n] = 1
            n_mat = Symmetric(n_mat)
            res[mean_dims + m, mean_dims + n] = tr(inv_var * m_mat * inv_var * n_mat)/2
        end
    end

    slices = _fi_cov_only_upper_params(mean_dims; offset=mean_dims)
    res[slices, slices]
end
