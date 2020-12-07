#md # # Tutorial
#md # Notebook [download](tutorial.ipynb) [nbviewer](@__NBVIEWER_ROOT_URL__/tutorial.ipynb) [source](tutorial.jl)

using MGVI
#
using Distributions
using Random
using ValueShapes
using LinearAlgebra
using Optim

# We define model of two sets of measurements (`a` and `b`) of the same 3-degree polynomial. MGVI takes model
# expressed as a function of model parameters returning an instance of the Distribution. In this example, since
# we have two sets of independent measurements, we pack them into the ValueShapes.NamedTupleDist.
#
# We assume errors are normally distributed with unknown variance that we also would like to fit.

const _x1_grid = [Float64(i)/10 for i in 1:25]
const _x2_grid = [Float64(i)/10 + 0.1 for i in 1:15]
const _common_grid = sort(vcat(_x1_grid, _x2_grid))

function _mean(x_grid, p)
    p[1]*10 .+ p[2]*40 .* x_grid .+ p[3]*600 .* x_grid.^2 .+ p[4]*80 .* x_grid.^3
end

function model(p)
    dist1 = Product(Normal.(_mean(_x1_grid, p), p[5]^2*60))
    dist2 = Product(Normal.(_mean(_x2_grid, p), p[5]^2*60))
    NamedTupleDist(a=dist1,
                   b=dist2)
end

# When model is defined, we also provide some real values of the parameters. Also, since MGVI is an iterative procedure,
# we define starting point — best guess of the true parameters. Each iteration will return new best guess that we can use
# as a starting point to construct next step.

const true_params =  [
 -0.3
 -1.5
 0.2
 -0.5
 0.3]

const starting_point = [
  0.2
  0.5
  -0.1
  0.3
 -0.6
];
#-
function pprintln(obj)
    show(stdout, "text/plain", obj)
    println()
end;
#-
using Plots
gr(size=(400, 300), dpi=700, fmt=:png)
#-
rng = MersenneTwister(157);

# We also sample data directly from the model:

data = rand(rng, model(true_params), 1)[1];
#-
function _mean(x::Vector)
    _mean(_common_grid, x)
end

init_plots =() -> let
    truth = _mean(true_params)
    plot!(_common_grid, truth, markercolor=:blue, linecolor=:blue, label="truth")
    scatter!(_common_grid, _mean(starting_point), markercolor=:orange, markerstrokewidth=0, markersize=3, label="init")
    scatter!(vcat(_x1_grid, _x2_grid), reduce(vcat, data), markercolor=:black, markerstrokewidth=0, markersize=3, label="data")
end;

# Before we start optimization, let's have a look at the data first. Also it is interesting to see how does our starting guess
# differ from true parameters

p = plot()
init_plots()

# Now we are ready to run one iteration of the MGVI. In the output we could print next best guess (`first_iteration.result`)
# and compare it to the true parameters.

first_iteration = mgvi_kl_optimize_step(rng,
                                        model, data,
                                        starting_point;
                                        jacobian_func=FwdRevADJacobianFunc,
                                        residual_sampler=ImplicitResidualSampler,
                                        optim_options=Optim.Options(iterations=10, show_trace=true),
                                        residual_sampler_options=(;cg_params=(;maxiter=10)))
pprintln(hcat(first_iteration.result, true_params))
p
#-
plot_iteration = (params, label) -> let
    #error_mat = mgvi_kl_errors(full_model, params)
    #display(error_mat)
    #errors = sqrt.(error_mat[diagind(error_mat)])
    #yerr = abs.(line(common_grid, params+errors) - line(common_grid, params-errors))
    #scatter!(common_grid, line(common_grid, params), markercolor=:green, label=label, yerr=yerr)
    for sample in eachcol(params.samples)
        scatter!(_common_grid, _mean(Vector(sample)), markercolor=:gray, markeralpha=0.3, markersize=2, label=nothing)
    end
    scatter!(_common_grid, _mean(params.result), markercolor=:green, label=label)
end;

# Now let's also plot the curve built on the estimated parameters after the first iteration:

p = plot()
init_plots()
plot_iteration(first_iteration, "first")
p
#-
plot_iteration_light = (params, counter) -> let
    scatter!(_common_grid, _mean(params.result), markercolor=:green, markersize=3, markeralpha=2*atan(counter/18)/π, label=nothing)
end;

# From the plot above we see that 1 iteration is not enough. Let's do 5 more steps and plot the evolution of guesses

init_plots()
plt = scatter()
next_iteration = first_iteration
for i in 1:5
    pprintln(minimum(next_iteration.optimized))
    pprintln(hcat(next_iteration.result, true_params))
    global next_iteration = mgvi_kl_optimize_step(rng,
                                                  model, data,
                                                  next_iteration.result;
                                                  jacobian_func=FwdRevADJacobianFunc,
                                                  residual_sampler=ImplicitResidualSampler,
                                                  optim_options=Optim.Options(iterations=10, show_trace=true),
                                                  residual_sampler_options=(;cg_params=(;maxiter=10)))
    plot_iteration_light(next_iteration, i)
end
pprintln(minimum(next_iteration.optimized))
pprintln(hcat(next_iteration.result, true_params))
plt

# Finally, let's plot the last best guess and compare it to the truth. Also, notice, that gray dots that represent samples from
# the covariance, became less spread after few iterations, so we reduced error estimate of our guess.

p = plot()
init_plots()
plot_iteration(next_iteration, "last")
p
