# ## Introduction

# In this tutorial, we will fit the [coal mining disaster dataset](coal_mining_data.tsv) with a Gaussian process modulated
# Poisson process.

# ## Prepare the environment

# We start by importing:
# * MGVI for the posterior fit
# * `Distributions.jl` and `FFTW.jl` to define the statistical model
# * `Optim.jl` to pass `Optim.Options` to MGVI and to find Maximum a posteriori fit that we will use for comparison
# * `StatsBase.jl` for histogram construction from the data and also for error bands visualization
# * `Plots.jl` for visualization

using MGVI

using FillArrays
using DelimitedFiles
using LinearAlgebra
using Random
using StatsBase
using Distributions
using Optim

using Plots
using Plots.PlotMeasures
Plots.default(legendfontsize=10, tickfontsize=10, grid=false, dpi=120, size=(500, 300))

using FFTW

import ForwardDiff
import Zygote

#

Random.seed!(84612);

# ## Load data

# The dataset, which is included with this repository, contains intervals in days between
# disasters occuring at British coal mines between March 1851 and March 1962.
# We build a model by splitting the entire time range into intervals of 365 days.

# ## Global parameters and the grid

# Now we define several model properties:
# * `DATA_DIM` is the shape of the dataset
# * `DATA_XLIM` specifies the time range of the data
#    This is useful when there are several datasets defined on different grids.
# * `GP_PADDING` adds empty paddings to the dataset. We use a Fourier transform to sample from the Gaussian process
#    with a finite correlation length. `GP_PADDING` helps us to ensure that periodic boundary conditions
#    imposed by a Fourier transform won't affect the data region.

DATA_DIM = (200, 200);

DATA_XLIM = ((-1., 1.), (0., 2.));

GP_PADDING = (2., 1.);

#

function produce_bins_1d(data_xlim, data_dim, gp_padding)
    binsize = (data_xlim[2] - data_xlim[1])/data_dim
    gp_dim = Integer(((data_xlim[2] - data_xlim[1]) + 2*gp_padding) ÷ binsize)
    gp_left_bin_offset = gp_right_bin_offset = (gp_dim - data_dim) ÷ 2
    if (2*gp_left_bin_offset + data_dim) % 2 == 1
        gp_left_bin_offset += 1
    end
    gp_left_xlim = data_xlim[1] - gp_left_bin_offset*binsize
    gp_right_xlim = data_xlim[2] + gp_right_bin_offset*binsize
    gp_left_xs = collect(gp_left_xlim + binsize/2:binsize:data_xlim[1])
    gp_right_xs = collect(data_xlim[2] + binsize/2:binsize:gp_right_xlim)
    gp_data_xs = collect(data_xlim[1] + binsize/2:binsize:data_xlim[2])
    gp_xs = [gp_left_xs; gp_data_xs; gp_right_xs]
    data_idxs = collect(gp_left_bin_offset+1:gp_left_bin_offset+data_dim)
    gp_xs, binsize, data_idxs
end;

function produce_bins()
    all_gp_xs = []
    all_gp_binsize = []
    all_data_idxs = []
    for i in 1:size(DATA_DIM, 1)
        gp_xs, gp_binsize, data_idxs = produce_bins_1d(DATA_XLIM[i], DATA_DIM[i], GP_PADDING[i])
        push!(all_gp_xs, gp_xs)
        push!(all_gp_binsize, gp_binsize)
        push!(all_data_idxs, data_idxs)
    end
    tuple(all_gp_xs...), tuple(all_gp_binsize...), tuple(all_data_idxs...)
end;

# Based on the defined model properties, we generate the grid. GP grid is the fine-grained grid
# with offsets added to the data range.
# * `_GP_XS` represent bin centers of such a fine-grained grid
# * `_GP_BINSIZE` is the width of the bin (that is 1/`GP_GRAIN_FACTOR` of data bin size)
# * `_DATA_IDXS` - integer indices of the left edges of the data bins

_GP_XS, _GP_BINSIZE, _DATA_IDXS = produce_bins();
_GP_DIM = length.(_GP_XS);
_HARMONIC_DIST = 1 ./ (_GP_DIM .* _GP_BINSIZE);

_HARMONIC_DIST

# ## Model parameters

# The Gaussian process in this tutorial is modeled in the Fourier space with zero mean
# and two hyperparameters defining properties of its kernel. To sample from this
# Gaussian process, we also need a parameter per bin that will represent the particular
# realization of the GP in the bin.

function assemble_paridx(;kwargs...)
    pos = 0
    res = []
    for (k, v) in kwargs
        new_start, new_stop = v.start+pos, v.stop+pos
        push!(res, (k, (v.start+pos):(v.stop+pos)))
        pos = new_stop
    end
    (;res...)
end;

# MGVI is an iterative procedure, so we will need to introduce an initial guess for the state of the model.
# We create a vector with size equal to the count of all parameters' `starting_point` and a NamedTuple
# `PARDIX` that assigns names to the sub-regions in this vector. In the correct case:
# * `gp_hyper` is two hyperparameters of the Gaussian process stored in the first two cells of the parameter vector
# * `gp_latent` `_GP_DIM` are parameters used to define the particular realization of the gaussian process,
#    stored at indices between `3` to `2 + _GP_DIM`.
#
# Function `assemble_paridx` is responsible for constructing such a NamedTuple from the parameter specification.

PARIDX = assemble_paridx(gp_hyper=1:4, gp_latent=1:prod(_GP_DIM));

starting_point = randn(last(PARIDX).stop);

#

# ## Model implementation

function map_idx(idx::Real, idx_range::AbstractUnitRange{<:Integer})
    i = idx - minimum(idx_range)
    n = length(eachindex(idx_range))
    n_2 = n >> 1
    ifelse(i <= n_2, i, i - n)
end

function dist_k(idx::CartesianIndex, ax::NTuple{N,<:AbstractUnitRange{<:Integer}}, harmonic_distances::NTuple{N,<:Real}) where N
    mapped_idx = map(map_idx, Tuple(idx), ax)
    norm(map(*, mapped_idx, harmonic_distances))
end

function dist_array(dims::NTuple{N,<:Real}, harmonic_distances::NTuple{N,<:Real}) where N
    cart_idxs = CartesianIndices(map(Base.OneTo, dims))
    dist_k.(cart_idxs, Ref(axes(cart_idxs)), Ref(harmonic_distances))
end;

#

# A Gaussian process's covariance in the Fourier space is represented with a diagonal matrix. Values
# on the diagonal follow a squared exponential function with parameters depending on priors.
# A kernel that is diagonal and mirrored around the center represents a periodic and translationally invariant function
# in the coordinate space. This property restricts covariance to have a finite correlation length in the coordinate
# space.
#
# The kernel in the Fourier space is defined on the domain of wave numbers `k`. We model the mirror-symmetrical kernel
# by imposing the mirror symmetry on the vector of the wave numbers. (See `map_idx` for the symmetry implementation)

k = dist_array(_GP_DIM, _HARMONIC_DIST);

heatmap(k)

# MGVI assumes that all priors are distributed as standard normals `N(0, 1)`; thus,
# to modify the shapes of the priors, we explicitly rescale them at the model implementation phase.
#
# We also exponentiate each prior before using it to tune the squared exponential shape. In doing so,
# we ensure only positive values for the kernel's hyperparameters.
#
# Actually, for the sake of numeric stability we model already square root of the covariance.
# This can be traced by missing `sqrt` in the next level, where we sample from the Gaussian process.

function amplitude_spectrum(d::Real, zero_mode_std::Real, slope::Real, offset::Real)
    # ampl * sqrt(2 * π * corrlen) * exp( -π^2 * d^2 * corrlen^2)
    ifelse(d ≈ 0, promote(zero_mode_std, exp(offset + slope * log(d)))...)
end;

function sqrt_kernel(p)
    _, kernel_zero_mode_std_c, kernel_slope_c, kernel_offset_c = p[PARIDX.gp_hyper]
    kernel_zero_mode_std = exp(kernel_zero_mode_std_c)*0.5 + 0.2
    kernel_slope = kernel_slope_c/5 - 2
    kernel_offset = kernel_offset_c/5 - 2
    amplitude_spectrum.(k, kernel_zero_mode_std, kernel_slope, kernel_offset)
end;

#

# As a Fourier transform we choose the Discrete Hartley Transform, which ensures that Fourier
# coefficients of the real valued function remain real valued.

ht = FFTW.plan_r2r(zeros(_GP_DIM), FFTW.DHT);

# Before we proceed, let's have a brief look at the kernel's shape. Below
# we plot the kernel in the coordinate space `K(r) = K(x2 - x1)` as a function of time in years
# between two points. As we go further along the `x`-axis, the time interval will increase, and
# the covariance will decrease.

function plot_kernel_model_x(p, x_i; plot_args=(;))
    xs = _GP_XS[2]
    plot!(xs, (ht * (sqrt_kernel(p)))[x_i, 1:end] .* _HARMONIC_DIST[1], label=nothing, linewidth=2.5; plot_args...)
end

function plot_kernel_model_y(p, y_i; plot_args=(;))
    xs = _GP_XS[1]
    plot!(xs, (ht * (sqrt_kernel(p)))[1:end, y_i] .* _HARMONIC_DIST[2], label=nothing, linewidth=2.5; plot_args...)
end


plot()
p1 = plot_kernel_model_x(starting_point, 5)
plot()
p2 = plot_kernel_model_y(starting_point, 10)
plot(p1, p2, layout=2)

#

# To make it even more visual, we also plot the structure of the covariance matrix as a heatmap.
# We see that the finite correlation length shows up as a band around the diagonal. We also
# see small artifacts in the antidiagonal corners. These come from the assumption that the
# kernel is periodic.

function plot_kernel_matrix(p)
    xkernel = ht * (sqrt_kernel(p)) .* prod(_HARMONIC_DIST)
    heatmap!(_GP_XS[1], _GP_XS[2], reshape(xkernel, _GP_DIM); yflip=true, xmirror=true, tick_direction=:out, top_margin=20px, right_margin=30px)
end

plot()
plot_kernel_matrix(starting_point)

#

# After we defined the square root of the kernel function (`sqrt_kernel`),
# we just follow the regular procedure of sampling from the normal distribution.
# Since the covariance matrix in the Fourier space is diagonal, Gaussian variables
# in each bin are independent of each other. Thus, sampling ends up rescaling
# the `gp_latent` part of the prior vector responsible for the Gaussian process state.
#
# After we produced a sample of Gaussian random values following the kernel model,
# we apply a Fourier transform to return back to the coordinate space.

zero_mode_matrix = zeros(_GP_DIM)
zero_mode_matrix[1,1] = 1;

function gp_sample(p)
    zero_mode_mean = p[PARIDX.gp_hyper][1]
    flat_gp = sqrt_kernel(p) .* reshape(p[PARIDX.gp_latent], _GP_DIM)
    flat_gp = flat_gp + zero_mode_matrix*exp(zero_mode_mean/10+0.3)*60
    pixel_volume = prod(_HARMONIC_DIST)
    (ht * flat_gp) .* pixel_volume
end;

# Together with the implementation of `gp_sample` we also need
# to define its version of the `Dual`s. This will allow our
# application of the Hartley transform to be differentiatiable.

function gp_sample(dp::Vector{ForwardDiff.Dual{T, V, N}}) where {T,V,N}
    pixel_volume = prod(_HARMONIC_DIST)
    zero_mode_mean = dp[PARIDX.gp_hyper][1]
    flat_gp_duals = sqrt_kernel(dp) .* reshape(dp[PARIDX.gp_latent], _GP_DIM)
    flat_gp_duals = flat_gp_duals + zero_mode_matrix*exp(zero_mode_mean/10+0.3)*60
    val_res = (ht*ForwardDiff.value.(flat_gp_duals)) .* pixel_volume
    psize = size(ForwardDiff.partials(flat_gp_duals[1]), 1)
    ps = x -> ForwardDiff.partials.(flat_gp_duals, x)
    val_ps = map((x -> ht*ps(x) .* pixel_volume), 1:psize)
    ForwardDiff.Dual{T}.(val_res, val_ps...)
end;

# Gaussian process realization is meant to serve as a Poisson rate of the Poisson
# process. Since the Gaussian process is not restricted to positive values, we
# exponentiate its values to forcefully make the function positive.

function poisson_gp_link(fs)
    exp.(fs)
end;

#

# Finally, we define the model by using the building blocks defined above:
# * `gp_sample` sample from the Gaussian process with defined `sqrt_kernel` covariance
# * `poisson_gp_link` ensures Gaussian process is positive
# * `model` maps parameters into the product of the Poisson distribution's counting events in each bin.

function model(params)
    fs = gp_sample(params)
    lambdas = poisson_gp_link(fs)
    Product(Poisson.(lambdas*prod(_GP_BINSIZE))[:])
end;

true_params = randn(last(PARIDX).stop);

model(true_params)

true_params

data = rand(model(true_params));

heatmap(reshape(data, DATA_DIM))

function compute_avg_likelihood(model, samples, data)
    tot = 0
    for sample in eachcol(samples)
        tot += -MGVI.posterior_loglike(model, sample, data)
    end
    tot/size(samples, 2)
end;

function show_avg_likelihood(series)
    scatter!(1:size(series, 1), series, label="-loglike")
end;

#

first_iteration = mgvi_kl_optimize_step(Random.GLOBAL_RNG,
                                        model, data,
                                        starting_point;
                                        num_residuals=3,
                                        jacobian_func=FwdRevADJacobianFunc,
                                        residual_sampler=ImplicitResidualSampler,
                                        optim_options=Optim.Options(iterations=1, show_trace=false),
                                        residual_sampler_options=(;cg_params=(;abstol=1E-2,verbose=false)));

next_iteration = first_iteration;
avg_likelihood_series = [];
push!(avg_likelihood_series, compute_avg_likelihood(model, next_iteration.samples, data));
for i in 1:100
    tmp_iteration = mgvi_kl_optimize_step(Random.GLOBAL_RNG,
                                          model, data,
                                          next_iteration.result;
                                          num_residuals=6,
                                          jacobian_func=FwdRevADJacobianFunc,
                                          residual_sampler=ImplicitResidualSampler,
                                          optim_options=Optim.Options(iterations=10, show_trace=false),
                                          residual_sampler_options=(;cg_params=(;abstol=1E-4,verbose=false)))
    global next_iteration = tmp_iteration
    push!(avg_likelihood_series, compute_avg_likelihood(model, next_iteration.samples, data))
end;

plot(yscale=:log)
show_avg_likelihood(avg_likelihood_series)

for i in avg_likelihood_series
    print(i, "\n")
end
