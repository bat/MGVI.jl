using MGVInference
#
include("../../../../test/test_models/model_polyfit.jl")
#
using Distributions
using Random
using ValueShapes
using LinearAlgebra
using Optim
#-
function pprintln(obj)
    show(stdout, "text/plain", obj)
    println()
end
#-
using Plots
gr(size=(400, 300), dpi=700)
#-
rng = MersenneTwister(157);
#-
data = rand(rng, model(true_params), 1)[1];
#-
function _mean(x::Vector)
    _mean(_common_grid, x)
end
#-
init_plots =() -> let
    truth = _mean(true_params)
    plot!(_common_grid, truth, markercolor=:blue, linecolor=:blue, label="truth")
    scatter!(_common_grid, _mean(starting_point), markercolor=:orange, markerstrokewidth=0, markersize=3, label="init")
    scatter!(vcat(_x1_grid, _x2_grid), MGVInference.unshaped(data), markercolor=:black, markerstrokewidth=0, markersize=3, label="data")
end
#-
plot()
init_plots()
#-
first_iteration = mgvi_kl_optimize_step(rng,
                                        model, data,
                                        starting_point;
                                        jacobian_func=FwdRevADJacobianFunc,
                                        residual_sampler=ImplicitResidualSampler,
                                        optim_options=Optim.Options(iterations=10, show_trace=true),
                                        residual_sampler_options=(;cg_params=(;maxiter=10)))
pprintln(hcat(first_iteration.result, true_params))
#-
plot_iteration = (params, label) -> let
    ##error_mat = mgvi_kl_errors(full_model, params)
    ##display(error_mat)
    ##errors = sqrt.(error_mat[diagind(error_mat)])
    ##yerr = abs.(line(common_grid, params+errors) - line(common_grid, params-errors))
    ##scatter!(common_grid, line(common_grid, params), markercolor=:green, label=label, yerr=yerr)
    for sample in eachcol(params.samples)
        scatter!(_common_grid, _mean(Vector(sample)), markercolor=:gray, markeralpha=0.3, markersize=2, label=nothing)
    end
    scatter!(_common_grid, _mean(params.result), markercolor=:green, label=label)
end
#-
plot()
init_plots()
plot_iteration(first_iteration, "first")
#-
plot_iteration_light = (params, counter) -> let
    scatter!(_common_grid, _mean(params.result), markercolor=:green, markersize=3, markeralpha=2*atan(counter/18)/π, label=nothing)
end
#-
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
#-
plot()
init_plots()
plot_iteration(next_iteration, "last")
