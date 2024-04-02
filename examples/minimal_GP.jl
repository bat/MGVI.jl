using BAT
using Random, LinearAlgebra
using Distributions
using ValueShapes
using Plots
using FFTW
import ForwardDiff
using DelimitedFiles
using PDMats
using MGVI
using Optim
# using Zygote: @adjoint, @ignore, gradient
using Zygote


n_data = 100
dims = (n_data,)
distances = (1/dims[1],)

harmonic_pad_distances = 1 ./ (2 .* dims .* distances)


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
end

function my_mask(x, n_data::Integer)
    #  y = x[mask]
     y = x[1:n_data] #FIXME
     y
end


function amplitude_forward_model(parameters)
    corr = (parameters.fluctuations ./(1 .+ (D./0.1).^(-parameters.slope)))[2:end]
    corr = vcat(parameters.zero_mode, corr)
    # corr[1] = parameters.zero_mode
    corr.^0.5 
end


function gp_forward_model(parameters::NamedTuple, n_data::Integer, n_x_pad::Integer, harmonic_pad_distances::Tuple, ht::FFTW.r2rFFTWPlan)
    amplitude = amplitude_forward_model(parameters)
    # amplitude = 1.
    harmonic_gp = amplitude .* parameters.ξ
    gp = apply_ht(ht, harmonic_gp) * (harmonic_pad_distances[1] / sqrt(n_x_pad))
    my_mask(gp, n_data)
end

function apply_ht(ht::FFTW.r2rFFTWPlan, dp::Vector{Float64})
    ht * dp
end


function apply_ht(ht::FFTW.r2rFFTWPlan, dp::Vector{ForwardDiff.Dual{T, V, N}}) where {T,V,N}
    val_res = ht *  ForwardDiff.value.(dp)
    psize = size(ForwardDiff.partials(dp[1]), 1)
    ps = x -> ForwardDiff.partials.(dp, x)
    val_ps = map((x -> ht*ps(x)), 1:psize)
    ForwardDiff.Dual{T}.(val_res, val_ps...)
end

rng = Random.default_rng()

# trafo_dht = FFTW.plan_r2r(zeros(2 .* dims), FFTW.DHT)
x = collect(distances[1]:distances[1]:distances[1] * dims[1];)
x_pad = collect(distances[1]:distances[1]:distances[1] * dims[1]*2;)

ht = FFTW.plan_r2r(zeros(length(x_pad)), FFTW.DHT);
D = dist_array(2 .* dims, harmonic_pad_distances)


mask = collect(1:length(x_pad))
mask = mask .<= length(mask)÷2

prior = NamedTupleDist(
    ξ = BAT.StandardMvNormal(length(D)), 
    fluctuations = Uniform(0, 100000), #FIXME something is off with parametrzization
    zero_mode = Uniform(0.1, 200),
    slope = Uniform(-4, -2),
)

bwd_trafo = BAT.DistributionTransform(Normal, prior)
fwd_trafo = inv(bwd_trafo)


truth = rand(prior)
standard_truth = bwd_trafo(truth)
N = 1
n = N * randn(n_data)

true_gp = gp_forward_model(truth,  n_data, length(x_pad), harmonic_pad_distances, ht)
data = true_gp + n 


model = let fwd_trafo = fwd_trafo, n_data = length(data), n_x_pad = length(x_pad), harmonic_pad_distances = harmonic_pad_distances, ht = ht
    function (stand_pars)
        parameters = fwd_trafo(stand_pars)[]
        Product(Normal.(gp_forward_model(parameters, n_data, n_x_pad, harmonic_pad_distances, ht), N))
    end
end

N_samps = 5
starting_point = bwd_trafo(rand(prior))
first_iteration = mgvi_kl_optimize_step(rng,
                                        model, data,
                                        starting_point;                                     
                                        jacobian_func=FwdRevADJacobianFunc,
                                        residual_sampler=ImplicitResidualSampler,
                                        num_residuals=N_samps,
                                        optim_options=Optim.Options(iterations=10, show_trace=true),
                                        residual_sampler_options=(;cg_params=(;maxiter=100)))

next_iteration = first_iteration

for i in 1:5
    global next_iteration = mgvi_kl_optimize_step(rng,
                                                    model, data,
                                                    next_iteration.result;
                                                    jacobian_func=FwdRevADJacobianFunc,
                                                    residual_sampler=ImplicitResidualSampler,
                                                    num_residuals=N_samps,
                                                    optim_options=Optim.Options(iterations=20, show_trace=true),
                                                    residual_sampler_options=(;cg_params=(;maxiter=100)))
end


res = fwd_trafo(next_iteration.result)[1]
plot(data,label="data",seriestype = :scatter)
for i in 1:N_samps*2
    plot!(gp_forward_model(fwd_trafo(next_iteration.samples[:,i])[1],  length(data), length(x_pad), harmonic_pad_distances, ht),color="black",alpha=0.3)
end
plot!(true_gp,label="truth", color="green")
plot!(gp_forward_model(res,  length(data), length(x_pad), harmonic_pad_distances, ht),label="mean", color="red")
