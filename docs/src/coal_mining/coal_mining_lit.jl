#md # # Advanced Tutorial
#md # Notebook [download](advanced_tutorial.ipynb) [nbviewer](@__NBVIEWER_ROOT_URL__/advanced_tutorial.ipynb) [source](advanced_tutorial.jl)

# In this tutorial we will fit coal mining disaster dataset with a Gaussian process modulated
# Poisson process.

# We start by importing:
# * MGVI that we will use for posterior fit
# * Distributions and FFTW to define the statistical model
# * Optim to pass Optim.Options to MGVI and to find Maximum A-Posteriori fit that we will use for comparison
# * StatsBase for histogram preparation from the data and also for error bands visualization
# * Plots.jl for visualization

using MGVI

using Distributions
using DelimitedFiles
using Random
using Optim
using StatsBase

using Plots

using FFTW

import ForwardDiff
#-
Random.seed!(84612);

# Dataset is attached to the repository and contains intervals in days between
# disasters happend at british coal mines between March 1851 and March 1962.
# We split the entire time range into intervals of 365 days, then number of events
# in each interval costitute the measurement we are going to model.

function read_coal_mining_data(filepath, binsize)
    init_year = empty
    data = empty
    open(filepath) do io
        raw = readline(io)
        while ! occursin("init_date", raw)
            raw = readline(io)
        end

        init_year = parse(Float64, split(split(strip(raw[2:end]), "\t")[2], "-")[1])
        data = readdlm(io, '\t', Int, '\n', comments=true)[:]
    end
    dates_fract_years = init_year .+ cumsum(data)/365
    left_edge = dates_fract_years[1]
    num_bins = ((dates_fract_years[end] - left_edge) ÷ binsize)
    right_edge = left_edge + binsize*num_bins
    fit(Histogram, dates_fract_years, left_edge:binsize:right_edge).weights
end

coal_mine_disaster_data = read_coal_mining_data("src/coal_mining/intervals.tsv", 1);

# Now we define several model properties:
# * `DATA_DIM` is just a size of the dataset
# * `DATA_XLIM` specifies the time range of the data
# * `GP_GRAIN_FACTOR` determines numbers of finer bins into which data bin is split.
# This is useful when there are several datasets defined on different grids.
# * `GP_PADDING` adds empty paddings to the dataset. We use Fourier transform to sample from the Gaussian process
# with a finite correlation length. `GP_PADDING` helps us to ensure that periodic boundary conditions
# imposed by Fourier transform won't affect data region.

DATA_DIM = size(coal_mine_disaster_data, 1);

data = coal_mine_disaster_data;

DATA_XLIM = [1851., 1962.];

GP_GRAIN_FACTOR = 3;
GP_PADDING = 80;
#-
function produce_bins()  #hide
    data_binsize = (DATA_XLIM[2] - DATA_XLIM[1])/DATA_DIM  #hide
    gp_binsize = data_binsize/GP_GRAIN_FACTOR  #hide
    gp_dim = Integer(((DATA_XLIM[2] - DATA_XLIM[1]) + 2*GP_PADDING) ÷ gp_binsize)  #hide
    gp_left_bin_offset = gp_right_bin_offset = (gp_dim - DATA_DIM) ÷ 2  #hide
    if (2*gp_left_bin_offset + DATA_DIM*GP_GRAIN_FACTOR) % 2 == 1  #hide
        gp_left_bin_offset += 1  #hide
    end  #hide
    gp_left_xlim = DATA_XLIM[1] - gp_left_bin_offset*gp_binsize  #hide
    gp_right_xlim = DATA_XLIM[2] + gp_right_bin_offset*gp_binsize  #hide
    gp_left_xs = collect(gp_left_xlim + gp_binsize/2:gp_binsize:DATA_XLIM[1])  #hide
    gp_right_xs = collect(DATA_XLIM[2] + gp_binsize/2:gp_binsize:gp_right_xlim)  #hide
    gp_data_xs = collect(DATA_XLIM[1] + gp_binsize/2:gp_binsize:DATA_XLIM[2])  #hide
    gp_xs = [gp_left_xs; gp_data_xs; gp_right_xs]  #hide
    data_idxs = collect(gp_left_bin_offset+-1:GP_GRAIN_FACTOR:gp_left_bin_offset+DATA_DIM*GP_GRAIN_FACTOR)  #hide
    gp_xs, gp_binsize, data_idxs  #hide
end;  #hide

# Based on the defined model properties we generate the grid. GP grid is the fine grained grid
# with offsets added to the data range. `_GP_XS` represent bin centers of such a fine grained grid,
# `_GP_BINSIZE` is the width of the bin (that is 1/`GP_GRAIN_FACTOR` of data bin size),
# `_DATA_IDXS` - integer indices of the left edges of the data bins.
#
# Implementation of the `produce_bins()` is hidden, but one can find it in the github repo.

_GP_XS, _GP_BINSIZE, _DATA_IDXS = produce_bins();
_GP_DIM = length(_GP_XS);

function assemble_paridx(;kwargs...)  #hide
    pos = 0  #hide
    res = []  #hide
    for (k, v) in kwargs  #hide
        new_start, new_stop = v.start+pos, v.stop+pos  #hide
        push!(res, (k, (v.start+pos):(v.stop+pos)))  #hide
        pos = new_stop  #hide
    end  #hide
    (;res...)  #hide
end;  #hide

# Now we have to define the vector of parameters that we will use as an initial guess.
# For this we create one vector of the size of the count of all parameters `starting_point` and a NamedTuple
# `PARDIX` that assignes names to the sub-regions in the vector of parameters. In the currect case:
# * `gp_hyper` two hyperparameters of the Gaussian process stored in the first two cells of the parameter vector
# * `gp_latent` `_GP_DIM` parameters used to define the particular realization of the gaussian process,
# stored at indices 3 to 2+`_GP_DIM`.
#
# Function `assemble_paridx` is responsible for constructing such a NamedTuple from the parameter specification. We omit
# details of its implementation in the tutorial, but those who interested can find the details in the github repo.

PARIDX = assemble_paridx(gp_hyper=1:2, gp_latent=1:_GP_DIM);

starting_point = randn(last(PARIDX).stop);
#-
k = collect(0:(_GP_DIM)÷2 -1);

# squared exp model. sqrt of the covariance matrix in the Fourier space
function kernel_model(p)
    kernel_A_c, kernel_l_c = p[PARIDX.gp_hyper]
    kernel_A = 60*exp(kernel_A_c*0.9)*GP_GRAIN_FACTOR
    kernel_l = 0.025*exp(kernel_l_c/15)/(GP_GRAIN_FACTOR^0.3)
    positive_modes = kernel_A .* sqrt(2 * π * kernel_l) .* exp.( -π^2 .* k.^2 .* kernel_l^2)
    negative_modes = positive_modes[end:-1:1]
    [positive_modes; negative_modes]
end;
#-
ht = FFTW.plan_r2r(zeros(_GP_DIM), FFTW.DHT);

function gp_sample(p)
    flat_gp = kernel_model(p) .* p[PARIDX.gp_latent]
    (ht * flat_gp) ./ _GP_DIM
end;
#-
function gp_sample(dp::Vector{ForwardDiff.Dual{T, V, N}}) where {T,V,N}
    flat_gp_duals = kernel_model(dp) .* dp[PARIDX.gp_latent]
    val_res = ht*ForwardDiff.value.(flat_gp_duals) ./ _GP_DIM
    psize = size(ForwardDiff.partials(flat_gp_duals[1]), 1)
    ps = x -> ForwardDiff.partials.(flat_gp_duals, x)
    val_ps = map((x -> ht*ps(x) ./ _GP_DIM), 1:psize)
    ForwardDiff.Dual{T}.(val_res, val_ps...)
end;
#-
function poisson_gp_link(fs)
    exp.(fs)
end;
#-
function _forward_agg(data, idxs, steps_forward)
    [sum(data[i:i+steps_forward-1]) for i in idxs]
end;

function agg_lambdas(lambdas)
    gps = _forward_agg(lambdas, _DATA_IDXS, GP_GRAIN_FACTOR) .* _GP_BINSIZE
    xs = _GP_XS[_DATA_IDXS .+ (GP_GRAIN_FACTOR ÷ 2)]
    xs, gps
end;

function agg_full_lambdas(lambdas)
    left_idxs = 1:GP_GRAIN_FACTOR:(_DATA_IDXS[1]-GP_GRAIN_FACTOR)
    left_gp = _forward_agg(lambdas, left_idxs, GP_GRAIN_FACTOR) .* _GP_BINSIZE
    left_xs = _GP_XS[left_idxs .+ (GP_GRAIN_FACTOR ÷ 2)]
    right_idxs = (_DATA_IDXS[end]+1):GP_GRAIN_FACTOR:(size(lambdas, 1) - GP_GRAIN_FACTOR)
    right_gp = _forward_agg(lambdas, right_idxs, GP_GRAIN_FACTOR) .* _GP_BINSIZE
    right_xs = _GP_XS[right_idxs .+ (GP_GRAIN_FACTOR ÷ 2)]
    middle_xs, middle_gp = agg_lambdas(lambdas)
    full_xs = [left_xs; middle_xs; right_xs]
    full_gp = [left_gp; middle_gp; right_gp]
    full_xs, full_gp
end;
#-
function model(params)
    fs = gp_sample(params)
    fine_lambdas = poisson_gp_link(fs)
    _, lambdas = agg_lambdas(fine_lambdas)
    Product(Poisson.(lambdas))
end;
#-
function _mean(p; full=false)
    agg_func = if (!full) agg_lambdas else agg_full_lambdas end
    xs, gps = agg_func(poisson_gp_link(gp_sample(p)))
    xs, gps
end;

function plot_mean(p, label="mean"; plot_args=(;), full=false)
    plot!(_mean(p; full=full)..., label=label, linewidth=2; plot_args...)
end;

function plot_prior_samples(num_samples)
    for _ in 1:num_samples
        p = randn(last(PARIDX).stop)
        plot_mean(p, nothing)
    end
end;

function plot_data(; scatter_args=(;), smooth_args=(;))
    scatter!(_GP_XS[_DATA_IDXS .+ (GP_GRAIN_FACTOR ÷ 2)], data, la=0, markersize=2., markerstrokewidth=0, label="data"; scatter_args...)
    smooth_step = 4
    smooth_xs = _GP_XS[_DATA_IDXS .+ (GP_GRAIN_FACTOR ÷ 2)][(smooth_step+1):(end-smooth_step)]
    smooth_data = [sum(data[i-smooth_step:i+smooth_step])/(2*smooth_step+1) for i in (smooth_step+1):(size(data, 1)-smooth_step)]
    plot!(smooth_xs, smooth_data, linewidth=2, linealpha=1, ls=:dash, label="smooth data"; smooth_args...)
end;

function plot_mgvi_samples(params)
    for sample in eachcol(params.samples)
        if any(isnan.(sample))
            print("nan found in samples", "\n")
            continue
        end
        plot!(_mean(Vector(sample))..., linealpha=0.5, linewidth=1, label=nothing)
    end
    plot!()
end;

function produce_posterior_samples(p, num_residuals)
    batch_size = 10

    if num_residuals <= 2*batch_size
        batch_size = num_residuals ÷ 2
    end

    est_res_sampler = MGVI._create_residual_sampler(model, p;
                                                    residual_sampler=ImplicitResidualSampler,
                                                    jacobian_func=FwdRevADJacobianFunc,
                                                    residual_sampler_options=(;cg_params=(;abstol=1E-2)))
    batches = []
    for _ in 1:(num_residuals ÷ batch_size ÷ 2)
        batch_residual_samples = MGVI.rand(Random.GLOBAL_RNG, est_res_sampler, batch_size)
        push!(batches, p .+ batch_residual_samples)
        push!(batches, p .- batch_residual_samples)
        if VERBOSE
            print(sum(size.(batches, 2)), "\n")
        end
    end
    reduce(hcat, batches)
end

function _extract_quantile(sorted_gp_realizations, p)
    map(s -> quantile(s, p; sorted=true), eachrow(sorted_gp_realizations))
end;

function plot_posterior_bands(p, num_samples; full=false)
    bands = [(0.997, :red), (0.955, :goldenrod1), (0.683, :green)]
    samples = produce_posterior_samples(p, num_samples)
    xs, first_gp = _mean(samples[1:end, 1]; full=full)
    gp_realizations = reduce(hcat, [_mean(Vector(sample); full=full)[2] for sample in eachcol(samples[1:end, 2:end])]; init=first_gp)
    for (i, one_x_sample) in enumerate(eachrow(gp_realizations))
        gp_realizations[i, 1:end] .= sort(Vector(one_x_sample))
    end
    for (band, color) in bands
        quant_l = _extract_quantile(gp_realizations, (1-band)/2)
        quant_u = _extract_quantile(gp_realizations, (1+band)/2)
        plot!(xs, quant_l; fillrange=quant_u, fillcolor=color, linealpha=0, label=band)
    end
    sample_median = _extract_quantile(gp_realizations, 0.5)
    plot!(xs, sample_median; linewidth=2, linecolor=:grey25, label="median")
end;
#-
plot()
plot_prior_samples(200)
plot_data()
plot!(ylim=[0, 8])
#-
plot()
plot_mean(starting_point, "starting_point")
plot_data()
#-
plot()
plot_mean(starting_point, "full gp"; full=true)
plot_mean(starting_point, "starting_point")
plot_data()
#-
first_iteration = mgvi_kl_optimize_step(Random.GLOBAL_RNG,
                                        model, data,
                                        starting_point;
                                        num_residuals=3,
                                        jacobian_func=FwdRevADJacobianFunc,
                                        residual_sampler=ImplicitResidualSampler,
                                        optim_options=Optim.Options(iterations=1, show_trace=false),
                                        residual_sampler_options=(;cg_params=(;abstol=1E-2,verbose=false)));
#-
plot()
plot_mean(first_iteration.result, "first_iteration")
plot_data()
#-
plot()
plot_data()
plot_mean(first_iteration.result, "full gp"; full=true)
plot_mean(first_iteration.result, "first_iteration")
#-
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
#-
next_iteration = first_iteration;
avg_likelihood_series = [];
push!(avg_likelihood_series, compute_avg_likelihood(model, next_iteration.samples, data));
for i in 1:30
    tmp_iteration = mgvi_kl_optimize_step(Random.GLOBAL_RNG,
                                          model, data,
                                          next_iteration.result;
                                          num_residuals=8,
                                          jacobian_func=FwdRevADJacobianFunc,
                                          residual_sampler=ImplicitResidualSampler,
                                          optim_options=Optim.Options(iterations=1, show_trace=false),
                                          residual_sampler_options=(;cg_params=(;abstol=1E-2,verbose=false)))
    global next_iteration = tmp_iteration
    push!(avg_likelihood_series, compute_avg_likelihood(model, next_iteration.samples, data))
end;
#-
plot(yscale=:log)
show_avg_likelihood(avg_likelihood_series)
#-
plot(ylim=[0,8])
plot_mgvi_samples(next_iteration)
plot_mean(next_iteration.result, "many_iterations", plot_args=(color=:deepskyblue2, linewidth=3.5))
plot_data(scatter_args=(;color=:blue2, marker_size=3.5), smooth_args=(;color=:deeppink3, linewidth=3))
#-
plot(ylim=[0,8])
plot_posterior_bands(next_iteration.result, 400)
plot_mean(next_iteration.result, "many_iterations", plot_args=(color=:deepskyblue2, linewidth=3.5))
plot_data(scatter_args=(;color=:blue2, marker_size=3.5), smooth_args=(;color=:deeppink3, linewidth=3))
#-
plot()
plot_data()
plot_mean(next_iteration.result; full=true)
plot_mean(next_iteration.result, "many_iterations")
#-
max_posterior = Optim.optimize(x -> -MGVI.posterior_loglike(model, x, data), starting_point, LBFGS(), Optim.Options(show_trace=false, g_tol=1E-10, iterations=300));
#-
plot()
plot_mean(Optim.minimizer(max_posterior), "map")
plot_data()
#-
plot()
plot_data()
plot_mean(Optim.minimizer(max_posterior), "full gp"; full=true)
plot_mean(Optim.minimizer(max_posterior), "map")
