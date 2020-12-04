# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

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

function _cut_params(res, dist::MvNormal)
    mean_size = length(dist.μ)
    slices = _fi_cov_only_upper_params(mean_size; offset=mean_size)
    res[slices, slices]
end

function fisher_information_mc(model::Function, params::AbstractVector, n::Integer)
    dist = model(params)
    sample() = _grad_logpdf(model, params, rand(dist))/n

    res = [sample() for _ in 1:Threads.nthreads()]
    for i in 1:n-Threads.nthreads()
        res[Threads.threadid()] .+= sample()
    end

    _cut_params(sum(res), dist)
end

# end fisher information mc

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
