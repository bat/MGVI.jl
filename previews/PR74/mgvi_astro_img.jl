using MGVI

using Distributions
using FFTW
using Random
using Statistics
using Plots
using ForwardDiffPullbacks

import ForwardDiff, Zygote
import DistributionsAD # provides the AD rules Zygote needs for Distributions
using AutoDiffOperators

context = MGVIContext(ADSelector(Zygote))

Plots.default(dpi=100, size=(1300, 700))

Random.seed!(122);

struct SkyModel <: Function
    data_dim::NTuple{2,Int}
    gp_dim::NTuple{2,Int}
    k::Matrix{Float64}
    k_ref::Float64
end

function SkyModel(; data_dim::NTuple{2,<:Integer}, pad_factor::Integer = 2)
    gp_dim = pad_factor .* data_dim
    kfreq(n) = [(i <= n ÷ 2 ? i : i - n) / n for i in 0:(n - 1)]
    k = hypot.(kfreq(gp_dim[1]), permutedims(kfreq(gp_dim[2])))
    SkyModel(data_dim, gp_dim, k, 1 / maximum(gp_dim))
end;

hyper_idxs(::SkyModel) = 1:3
gp_idxs(m::SkyModel) = (1:prod(m.gp_dim)) .+ last(hyper_idxs(m))
star_idxs(m::SkyModel) = (1:prod(m.data_dim)) .+ last(gp_idxs(m))
n_params(m::SkyModel) = last(star_idxs(m));

data_idxs(m::SkyModel) = map((o, n) -> (o + 1):(o + n), (m.gp_dim .- m.data_dim) .÷ 2, m.data_dim);

function dht(x::AbstractArray)
    F = fft(Complex.(x))
    return real.(F) .- imag.(F)
end;

function dht(x::AbstractArray{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
    val = dht(ForwardDiff.value.(x))
    parts = ntuple(i -> dht(ForwardDiff.partials.(x, i)), Val(N))
    return ForwardDiff.Dual{T}.(val, parts...)
end;

function diffuse_emission(m::SkyModel, p::AbstractVector)
    ξ_offset, ξ_fluct, ξ_slope = p[hyper_idxs(m)]
    offset = log(3.0) + 1.0 * ξ_offset
    fluctuations = 1.5 * exp(0.3 * ξ_fluct)
    α = 1.5 + 0.25 * ξ_slope
    k_safe = ifelse.(m.k .== 0, one.(m.k), m.k)
    A_raw = ifelse.(m.k .== 0, zero(α), (k_safe ./ m.k_ref) .^ -α)
    A = fluctuations .* A_raw ./ sqrt(sum(abs2, A_raw) / length(A_raw))
    ξ_gp = reshape(p[gp_idxs(m)], m.gp_dim)
    log_brightness = offset .+ dht(A .* ξ_gp) ./ sqrt(prod(m.gp_dim))
    exp.(log_brightness[data_idxs(m)...])
end;

function star_emission(m::SkyModel, p::AbstractVector)
    exp.(-6.5 .+ 3.2 .* reshape(p[star_idxs(m)], m.data_dim))
end;

total_emission(m::SkyModel, p::AbstractVector) = diffuse_emission(m, p) .+ star_emission(m, p);

function (m::SkyModel)(p::AbstractVector)
    product_distribution(fwddiff(Poisson).(vec(total_emission(m, p))))
end;

sky = SkyModel(data_dim = (96, 96))

true_params = randn(n_params(sky));
data = rand(sky(true_params));

plot(
    heatmap(diffuse_emission(sky, true_params), title="true diffuse emission"),
    heatmap(star_emission(sky, true_params), title="true stars"),
    heatmap(log10.(total_emission(sky, true_params)), title="true total emission (log10)"),
    heatmap(log10.(reshape(data, sky.data_dim) .+ 1), title="observed counts (log10(1+n))"),
    layout=(2, 2), size=(900, 700)
)
savefig("astro-img-plot1.png")

config = MGVIConfig(
    linsolver_opts = (;maxiters = 30),
    optimizer = MGVI.NewtonCG(steps = 3, cg_maxiter = 20)
)

center = zeros(n_params(sky))
prepared = mgvi_prepare(sky, data, 4, center, config, context)

mnlp_series = Float64[]
result = nothing
for i in 1:12
    global result, center = mgvi_step(prepared, center)
    push!(mnlp_series, result.mnlp)
end

plot(mnlp_series, yscale=:log10, marker=:circle, label="mnlp", xlabel="iteration")
savefig("astro-img-plot2.pdf")

plot(
    heatmap(diffuse_emission(sky, true_params), title="true diffuse emission"),
    heatmap(diffuse_emission(sky, center), title="reconstructed diffuse emission"),
    heatmap(star_emission(sky, true_params), title="true stars"),
    heatmap(star_emission(sky, center), title="reconstructed stars"),
    layout=(2, 2), size=(900, 700)
)
savefig("astro-img-plot3.png")

star_sel = findall(vec(star_emission(sky, true_params)) .> 1);

scatter(
    vec(star_emission(sky, true_params))[star_sel], vec(star_emission(sky, center))[star_sel],
    xscale=:log10, yscale=:log10, xlabel="true star brightness",
    ylabel="reconstructed star brightness", label=nothing
)
plot!(identity, label="truth")
savefig("astro-img-plot4.pdf")

residuals = MGVI.sample_residuals(
    sky, center,
    randn(prod(sky.data_dim), 16), randn(n_params(sky), 16),
    context.ad
);
post_samples = hcat(center .+ residuals, center .- residuals);

sample_log_emissions = [log10.(total_emission(sky, Vector(s))) for s in eachcol(post_samples)];

plot(
    heatmap(log10.(total_emission(sky, center)), title="reconstructed total emission (log10)"),
    heatmap(std(sample_log_emissions), title="posterior uncertainty (dex)"),
    layout=(1, 2), size=(900, 350)
)
savefig("astro-img-plot5.png")
