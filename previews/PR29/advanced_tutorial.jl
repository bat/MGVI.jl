using MGVI

using Distributions
using DelimitedFiles
using Random
using Optim
using StatsBase

using Plots
using Plots.PlotMeasures
Plots.default(legendfontsize=24, tickfontsize=24, grid=false, dpi=100, size=(1300, 700))

using FFTW

import ForwardDiff

Random.seed!(84612);
mkpath(joinpath(@__DIR__, "plots"));

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

coal_mine_disaster_data = read_coal_mining_data(joinpath(@__DIR__, "coal_mining_data.tsv"), 1);

DATA_DIM = size(coal_mine_disaster_data, 1);

data = coal_mine_disaster_data;

DATA_XLIM = [1851., 1962.];

GP_GRAIN_FACTOR = 3;
GP_PADDING = 80;

function produce_bins()
    data_binsize = (DATA_XLIM[2] - DATA_XLIM[1])/DATA_DIM
    gp_binsize = data_binsize/GP_GRAIN_FACTOR
    gp_dim = Integer(((DATA_XLIM[2] - DATA_XLIM[1]) + 2*GP_PADDING) ÷ gp_binsize)
    gp_left_bin_offset = gp_right_bin_offset = (gp_dim - DATA_DIM) ÷ 2
    if (2*gp_left_bin_offset + DATA_DIM*GP_GRAIN_FACTOR) % 2 == 1
        gp_left_bin_offset += 1
    end
    gp_left_xlim = DATA_XLIM[1] - gp_left_bin_offset*gp_binsize
    gp_right_xlim = DATA_XLIM[2] + gp_right_bin_offset*gp_binsize
    gp_left_xs = collect(gp_left_xlim + gp_binsize/2:gp_binsize:DATA_XLIM[1])
    gp_right_xs = collect(DATA_XLIM[2] + gp_binsize/2:gp_binsize:gp_right_xlim)
    gp_data_xs = collect(DATA_XLIM[1] + gp_binsize/2:gp_binsize:DATA_XLIM[2])
    gp_xs = [gp_left_xs; gp_data_xs; gp_right_xs]
    data_idxs = collect(gp_left_bin_offset+1:GP_GRAIN_FACTOR:gp_left_bin_offset+DATA_DIM*GP_GRAIN_FACTOR)
    gp_xs, gp_binsize, data_idxs
end;

_GP_XS, _GP_BINSIZE, _DATA_IDXS = produce_bins();
_GP_DIM = length(_GP_XS);

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

PARIDX = assemble_paridx(gp_hyper=1:2, gp_latent=1:_GP_DIM);

starting_point = randn(last(PARIDX).stop);

k = collect(0:(_GP_DIM)÷2 -1);

function sqrt_kernel(p)
    kernel_A_c, kernel_l_c = p[PARIDX.gp_hyper]
    kernel_A = 60*exp(kernel_A_c*0.9)*GP_GRAIN_FACTOR
    kernel_l = 0.025*exp(kernel_l_c/15)/(GP_GRAIN_FACTOR^0.3)
    positive_modes = kernel_A .* sqrt(2 * π * kernel_l) .* exp.( -π^2 .* k.^2 .* kernel_l^2)
    negative_modes = positive_modes[end:-1:1]
    [positive_modes; negative_modes]
end;

ht = FFTW.plan_r2r(zeros(_GP_DIM), FFTW.DHT);

function plot_kernel_model(p, width; plot_args=(;))
    xs = collect(1:Int(floor(width/_GP_BINSIZE)))
    plot!(xs .* _GP_BINSIZE, (ht * (sqrt_kernel(p) .^ 2))[xs] ./ _GP_DIM, label=nothing, linewidth=2.5; plot_args...)
end

plot()
plot_kernel_model(starting_point, 20)

function plot_kernel_matrix(p)
    xkernel = ht * (sqrt_kernel(p) .^ 2) ./ _GP_DIM
    res = reduce(hcat, [circshift(xkernel, i) for i in 0:(_GP_DIM-1)])'
    heatmap!(_GP_XS, _GP_XS, res; yflip=true, xmirror=true, tick_direction=:out, top_margin=20px, right_margin=30px)
end

plot()
plot_kernel_matrix(starting_point)
png(joinpath(@__DIR__, "plots/gp-covariance-matrix.png"))

function gp_sample(p)
    flat_gp = sqrt_kernel(p) .* p[PARIDX.gp_latent]
    (ht * flat_gp) ./ _GP_DIM
end;

function gp_sample(dp::Vector{ForwardDiff.Dual{T, V, N}}) where {T,V,N}
    flat_gp_duals = sqrt_kernel(dp) .* dp[PARIDX.gp_latent]
    val_res = ht*ForwardDiff.value.(flat_gp_duals) ./ _GP_DIM
    psize = size(ForwardDiff.partials(flat_gp_duals[1]), 1)
    ps = x -> ForwardDiff.partials.(flat_gp_duals, x)
    val_ps = map((x -> ht*ps(x) ./ _GP_DIM), 1:psize)
    ForwardDiff.Dual{T}.(val_res, val_ps...)
end;

function poisson_gp_link(fs)
    exp.(fs)
end;

function _forward_agg(data, idxs, steps_forward)
    [sum(data[i:i+steps_forward-1]) for i in idxs]
end;

function agg_lambdas(lambdas)
    gps = _forward_agg(lambdas, _DATA_IDXS, GP_GRAIN_FACTOR) .* _GP_BINSIZE
    xs = _GP_XS[_DATA_IDXS .+ (GP_GRAIN_FACTOR ÷ 2)]
    xs, gps
end;

function model(params)
    fs = gp_sample(params)
    fine_lambdas = poisson_gp_link(fs)
    _, lambdas = agg_lambdas(fine_lambdas)
    Product(Poisson.(lambdas))
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

function _mean(p; full=false)
    agg_func = if (!full) agg_lambdas else agg_full_lambdas end
    xs, gps = agg_func(poisson_gp_link(gp_sample(p)))
    xs, gps
end;

function plot_mean(p, label="mean"; plot_args=(;), full=false)
    plot!(_mean(p; full=full)...; label=label, linewidth=3, plot_args...)
end;

function plot_prior_samples(num_samples; mean_plot_args=(;))
    for _ in 1:num_samples
        p = randn(last(PARIDX).stop)
        plot_mean(p, nothing; plot_args=mean_plot_args)
    end
end;

function plot_kernel_prior_samples(num_samples, width)
    for _ in 1:num_samples
        p = randn(last(PARIDX).stop)
        plot_kernel_model(p, width)
    end
    plot!()
end;

function plot_data(; scatter_args=(;), smooth_args=(;))
    bar!(_GP_XS[_DATA_IDXS .+ (GP_GRAIN_FACTOR ÷ 2)], data, color=:deepskyblue2, la=0, markersize=2., markerstrokewidth=0, alpha=0.4, label="data"; scatter_args...)
    smooth_step = 4
    smooth_xs = _GP_XS[_DATA_IDXS .+ (GP_GRAIN_FACTOR ÷ 2)][(smooth_step+1):(end-smooth_step)]
    smooth_data = [sum(data[i-smooth_step:i+smooth_step])/(2*smooth_step+1) for i in (smooth_step+1):(size(data, 1)-smooth_step)]
    plot!(smooth_xs, smooth_data, color=:deeppink3, linewidth=3, linealpha=1, ls=:dash, label="smooth data"; smooth_args...)
end;

function plot_mgvi_samples(samples)
    for sample in eachcol(samples)
        if any(isnan.(sample))
            print("nan found in samples", "\n")
            continue
        end
        plot!(_mean(Vector(sample))..., linealpha=0.5, linewidth=2, label=nothing)
    end
    plot!()
end;

function plot_kernel_mgvi_samples(samples, width)
    for sample in eachcol(samples)
        if any(isnan.(sample))
            print("nan found in samples", "\n")
            continue
        end
        plot_kernel_model(sample, width; plot_args=(linealpha=0.5, linewidth=2, label=nothing))
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

plot()
plot_data(;scatter_args=(;alpha=0.7))
plot_prior_samples(200, mean_plot_args=(;alpha=0.5))
plot!(ylim=[0, 8])
png(joinpath(@__DIR__, "plots/poisson-dynamic-range.png"))

plot()
plot_kernel_prior_samples(200, 20)
png(joinpath(@__DIR__, "plots/gp-kernel-dynamic-range.png"))

plot()
plot_data()
plot_mean(starting_point, "starting point"; plot_args=(;color=:darkorange2))
plot_mgvi_samples(produce_posterior_samples(starting_point, 6))
png(joinpath(@__DIR__, "plots/res-starting-point.png"))

plot()
plot_data()
plot_mean(starting_point, "full gp"; full=true, plot_args=(;color=:pink))
plot_mean(starting_point, "starting point"; plot_args=(;color=:darkorange2))

plot()
plot_kernel_model(starting_point, 20; plot_args=(;label="kernel model"))
plot_kernel_mgvi_samples(produce_posterior_samples(starting_point, 6), 20)
png(joinpath(@__DIR__, "plots/kernel-starting-point.png"))

first_iteration = mgvi_kl_optimize_step(Random.GLOBAL_RNG,
                                        model, data,
                                        starting_point;
                                        num_residuals=3,
                                        jacobian_func=FwdRevADJacobianFunc,
                                        residual_sampler=ImplicitResidualSampler,
                                        optim_options=Optim.Options(iterations=1, show_trace=false),
                                        residual_sampler_options=(;cg_params=(;abstol=1E-2,verbose=false)));

plot()
plot_data()
plot_mean(first_iteration.result, "first iteration"; plot_args=(;color=:darkorange2))
plot_mgvi_samples(first_iteration.samples)

plot()
plot_data()
plot_mean(first_iteration.result, "full gp"; full=true, plot_args=(;color=:pink))
plot_mean(first_iteration.result, "first iteration"; plot_args=(;color=:darkorange2))

plot()
plot_kernel_model(first_iteration.result, 20; plot_args=(;label="kernel model"))
plot_kernel_mgvi_samples(first_iteration.samples, 20)

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

next_iteration = first_iteration;
avg_likelihood_series = [];
push!(avg_likelihood_series, compute_avg_likelihood(model, next_iteration.samples, data));
for i in 1:30
    tmp_iteration = mgvi_kl_optimize_step(Random.GLOBAL_RNG,
                                          model, data,
                                          next_iteration.result;
                                          num_residuals=3,
                                          jacobian_func=FwdRevADJacobianFunc,
                                          residual_sampler=ImplicitResidualSampler,
                                          optim_options=Optim.Options(iterations=1, show_trace=false),
                                          residual_sampler_options=(;cg_params=(;abstol=1E-2,verbose=false)))
    global next_iteration = tmp_iteration
    push!(avg_likelihood_series, compute_avg_likelihood(model, next_iteration.samples, data))
end;

plot(yscale=:log)
show_avg_likelihood(avg_likelihood_series)
png(joinpath(@__DIR__, "plots/convergence.png"))

plot(ylim=[0,8])
plot_data()
plot_mgvi_samples(next_iteration.samples)
plot_mean(next_iteration.result, "many iterations"; plot_args=(;color=:darkorange2))
png(joinpath(@__DIR__, "plots/res-many-iter.png"))

plot(ylim=[0,8])
plot_posterior_bands(next_iteration.result, 400)
plot_data()
plot_mean(next_iteration.result, "many iterations"; plot_args=(;color=:darkorange2))
png(joinpath(@__DIR__, "plots/res-bands.png"))

plot()
plot_data()
plot_mean(next_iteration.result; full=true, plot_args=(;color=:pink))
plot_mean(next_iteration.result, "many iterations"; plot_args=(;color=:darkorange2))

plot()
plot_kernel_model(next_iteration.result, 20; plot_args=(;label="kernel model"))
plot_kernel_mgvi_samples(next_iteration.samples, 20)
png(joinpath(@__DIR__, "plots/kernel-many-iter.png"))

max_posterior = Optim.optimize(x -> -MGVI.posterior_loglike(model, x, data), starting_point, LBFGS(), Optim.Options(show_trace=false, g_tol=1E-10, iterations=300));

plot()
plot_data()
plot_mean(next_iteration.result, "mgvi mean"; plot_args=(;color=:darkorange2))
plot_mean(Optim.minimizer(max_posterior), "map")
png(joinpath(@__DIR__, "plots/map.png"))

plot()
plot_data()
plot_mean(Optim.minimizer(max_posterior), "full gp"; full=true, plot_args=(;color=:darkorange2))
plot_mean(next_iteration.result, "mgvi full gp"; full=true)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

