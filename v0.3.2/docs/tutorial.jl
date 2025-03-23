using MGVI

using Distributions
using Random
using ValueShapes
using LinearAlgebra
using Optim
using AutoDiffOperators
import Zygote

context = MGVIContext(ADSelector(Zygote))

const _x1_grid = [Float64(i)/10 for i in 1:25]
const _x2_grid = [Float64(i)/10 + 0.1 for i in 1:15]
const _common_grid = sort(vcat(_x1_grid, _x2_grid))

function _mean(x_grid, p)
    p[1]*10 .+ p[2]*40 .* x_grid .+ p[3]*600 .* x_grid.^2 .+ p[4]*80 .* x_grid.^3
end

function model(p)
    dist1 = product_distribution(Normal.(_mean(_x1_grid, p), p[5]^2*60))
    dist2 = product_distribution(Normal.(_mean(_x2_grid, p), p[5]^2*60))
    NamedTupleDist(a=dist1,
                   b=dist2)
end

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

function pprintln(obj)
    show(stdout, "text/plain", obj)
    println()
end;

using Plots

rng = MersenneTwister(157);

data = rand(rng, model(true_params), 1)[1];

function _mean(x::Vector)
    _mean(_common_grid, x)
end

init_plots =() -> let
    truth = _mean(true_params)
    plot!(_common_grid, truth, markercolor=:blue, linecolor=:blue, label="truth")
    scatter!(_common_grid, _mean(starting_point), markercolor=:orange, markerstrokewidth=0, markersize=3, label="init")
    scatter!(vcat(_x1_grid, _x2_grid), reduce(vcat, data), markercolor=:black, markerstrokewidth=0, markersize=3, label="data")
end;

p = plot()
init_plots()
savefig("tutorial-plot1.pdf")

first_iteration = mgvi_optimize_step(
    model, data, starting_point, context;
    linear_solver=MGVI.IterativeSolversCG((;maxiter=10)),
    optim_options=Optim.Options(iterations=10, show_trace=true),
)
@info hcat(first_iteration.result, true_params)
p
savefig("tutorial-plot2.pdf")

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

p = plot()
init_plots()
plot_iteration(first_iteration, "first")
p
savefig("tutorial-3.pdf")

plot_iteration_light = (params, counter) -> let
    scatter!(_common_grid, _mean(params.result), markercolor=:green, markersize=3, markeralpha=2*atan(counter/18)/π, label=nothing)
end;

init_plots()
plt = scatter()
next_iteration = first_iteration
for i in 1:5
    @info minimum(next_iteration.optimized)
    @info hcat(next_iteration.result, true_params)
    global next_iteration = mgvi_optimize_step(
        model, data, next_iteration.result, context;
        linear_solver=MGVI.IterativeSolversCG((;maxiter=10)),
        optim_options=Optim.Options(iterations=10, show_trace=true)
    )
    plot_iteration_light(next_iteration, i)
end
@info minimum(next_iteration.optimized)
@info hcat(next_iteration.result, true_params)
plt
savefig("tutorial-plot4.pdf")

p = plot()
init_plots()
plot_iteration(next_iteration, "last")
p
savefig("tutorial-5.pdf")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
