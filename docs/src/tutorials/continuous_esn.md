# Continuous ESN: forecasting Lorenz

[`ContinuousESN`](@ref) is a continuous-time Echo State Network that
implements the ODE of [Lukosevicius2012](@cite):

```math
\dot{\mathbf{x}}(t) = -\mathbf{x}(t) + \tanh\!\left(
    \mathbf{W}_{\text{in}}\,\mathbf{u}(t) + \mathbf{W}_r\,\mathbf{x}(t)
    + \mathbf{b}\right)
```

This tutorial trains a `ContinuousESN` on Lorenz-63 data and rolls it
forward autoregressively to reproduce the attractor. The training and
prediction pipeline is the same as for [`ESN`](@ref).

## Building a Lorenz dataset

```@example continuous-esn-lorenz
using ReservoirComputing
using LuxCore: setup
using SciMLBase
using DataInterpolations
using OrdinaryDiffEqTsit5
using Plots
using Random

Random.seed!(42)
rng = MersenneTwister(17)

function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end
data_prob = ODEProblem(
    lorenz!, [1.0, 0.0, 0.0], (0.0, 40.0), [10.0, 28.0, 8 / 3]
)
data = Array(solve(data_prob, Tsit5(); saveat = 0.02))

shift, train_len, predict_len = 300, 1000, 250
input_data = data[:, shift:(shift + train_len - 1)]
target_data = data[:, (shift + 1):(shift + train_len)]
test = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]
```

## Constructing the `ContinuousESN`

```@example continuous-esn-lorenz
N_res = 300
res_radius = 0.9
res_sparsity = 6 / N_res

# Float64 initialisers so the reservoir, the solve, and the input all
# share a numeric type. Without these the cell would default to
# Float32 via `scaled_rand` / `rand_sparse` / `zeros32`.
init_input_f64(rng, d...) = scaled_rand(rng, Float64, d...)
init_reservoir_f64(rng, d...) = rand_sparse(
    rng, Float64, d...; radius = res_radius, sparsity = res_sparsity
)

esn_train = ContinuousESN(
    3, N_res, 3, (0.0, Float64(train_len)), Tsit5();
    init_input = init_input_f64,
    init_reservoir = init_reservoir_f64,
    state_modifiers = (NLAT2(),),
    reltol = 1.0e-6, abstol = 1.0e-8
)
esn_pred = ContinuousESN(
    3, N_res, 3, (0.0, Float64(predict_len)), Tsit5();
    init_input = init_input_f64,
    init_reservoir = init_reservoir_f64,
    state_modifiers = (NLAT2(),),
    reltol = 1.0e-6, abstol = 1.0e-8
)

ps, st = setup(rng, esn_train)
```

## Training

```@example continuous-esn-lorenz
ps, st = train(esn_train, input_data, target_data, ps, st;
    objective = RidgeRegression(1.0e-6))
```

## Autoregressive rollout

```@example continuous-esn-lorenz
output, _ = predict(
    esn_pred, predict_len, ps, st; initialdata = test[:, 1]
)
```

```@example continuous-esn-lorenz
using Plots.PlotMeasures

dt = 0.02
lorenz_maxlyap = 0.9056
lyap_time = (0:(predict_len - 1)) .* dt .* (1 / lorenz_maxlyap)

p1 = plot(lyap_time, [test[1, :] output[1, :]]; label = ["actual" "predicted"],
    ylabel = "x(t)", linewidth = 2.5, xticks = false, yticks = -15:15:15);
p2 = plot(lyap_time, [test[2, :] output[2, :]]; label = ["actual" "predicted"],
    ylabel = "y(t)", linewidth = 2.5, xticks = false, yticks = -20:20:20);
p3 = plot(lyap_time, [test[3, :] output[3, :]]; label = ["actual" "predicted"],
    ylabel = "z(t)", linewidth = 2.5, xlabel = "max(λ)*t", yticks = 10:15:40);

plot(p1, p2, p3; plot_title = "Lorenz System Coordinates",
    layout = (3, 1), xtickfontsize = 12, ytickfontsize = 12, xguidefontsize = 15,
    yguidefontsize = 15,
    legendfontsize = 12, titlefontsize = 20)
```

The two trajectories agree on the early portion of the rollout before
chaotic divergence dominates — the same behaviour the discrete-ESN
tutorial produces. The point of the example is that nothing in the
training loop changes between discrete ESN, `SciMLProblemReservoir`
with hand-rolled equations, and `ContinuousESN`: the same `train` /
`predict` pipeline drives all three.

## When to reach for `ContinuousESN` vs `SciMLProblemReservoir`

* `ContinuousESN` pre-bakes the continuous ESN ODE; use it when the
  standard continuous ESN is what you want.
* [`SciMLProblemReservoir`](@ref) is the generic building block; use it
  when the reservoir ODE is not the standard eq (5) — bespoke RHS, SDE,
  DDE, or non-standard parameter layout.
