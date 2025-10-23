# Deep Echo State Networks

In this example we showcase how to build a deep echo state network (DeepESN)
following the work of [Gallicchio2017](@cite). The DeepESN stacks reservoirs
on top of each other, feeding the output from one into the next.
In the version implemented in ReservoirComputing.jl the final state is the state
used for training.

## Lorenz Example

We are going to reuse the Lorenz data used in the
[Lorenz System Forecasting](@ref) example.

```@example deep_lorenz
using OrdinaryDiffEq

#define lorenz system
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

#solve and take data
prob = ODEProblem(lorenz!, [1.0, 0.0, 0.0], (0.0, 200.0))
data = solve(prob, ABM54(); dt=0.02)
data = reduce(hcat, data.u)

#determine shift length, training length and prediction length
shift = 300
train_len = 5000
predict_len = 1250

#split the data accordingly
input_data = data[:, shift:(shift + train_len - 1)]
target_data = data[:, (shift + 1):(shift + train_len)]
test_data = data[:, (shift + train_len + 1):(shift + train_len + predict_len)]
```

The call for the DeepESN works similarly to the ESN.
The only difference is that the reservoir (and corresponding kwargs)
can be fed as an array.

```@example deep_lorenz
using ReservoirComputing
input_size = 3
res_size = 300
desn = DeepESN(input_size, [res_size, res_size], input_size;
    init_reservoir=rand_sparse(; radius=1.2, sparsity=6/300),
    state_modifiers=[NLAT2, ExtendedSquare]
)

```

The training and prediction follow the usual framework:

```@example deep_lorenz
using Random
Random.seed!(42)
rng = MersenneTwister(17)

ps, st = setup(rng, desn)
ps, st = train!(desn, input_data, target_data, ps, st)

output, st = predict(desn, 1250, ps, st; initialdata=test_data[:, 1])
```

Plotting the results:

```@example deep_lorenz
using Plots

ts = 0.0:0.02:200.0
lorenz_maxlyap = 0.9056
predict_ts = ts[(shift + train_len + 1):(shift + train_len + predict_len)]
lyap_time = (predict_ts .- predict_ts[1]) * (1 / lorenz_maxlyap)

p1 = plot(lyap_time, [test_data[1, :] output[1, :]]; label=["actual" "predicted"],
    ylabel="x(t)", linewidth=2.5, xticks=false, yticks=-15:15:15);
p2 = plot(lyap_time, [test_data[2, :] output[2, :]]; label=["actual" "predicted"],
    ylabel="y(t)", linewidth=2.5, xticks=false, yticks=-20:20:20);
p3 = plot(lyap_time, [test_data[3, :] output[3, :]]; label=["actual" "predicted"],
    ylabel="z(t)", linewidth=2.5, xlabel="max(λ)*t", yticks=10:15:40);

plot(p1, p2, p3; plot_title="Lorenz System Coordinates",
    layout=(3, 1), xtickfontsize=12, ytickfontsize=12, xguidefontsize=15,
    yguidefontsize=15,
    legendfontsize=12, titlefontsize=20)
```

## References

```@bibliography
Pages = ["deep_esn.md"]
Canonical = false
```
