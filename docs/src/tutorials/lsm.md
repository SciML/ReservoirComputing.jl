# Liquid State Machine: a periodic orbit

[`LSM`](@ref) is a Liquid State Machine ([Maass2002](@cite)): an
[`LSMCell`](@ref), optional `state_modifiers`, and a
[`LinearReadout`](@ref). The default neuron [`LIFNeuron`](@ref) is
leaky integrate-and-fire, not [`LIFESN`](@ref) (Local Information Flow).

This tutorial trains an `LSM` on a 2D periodic orbit and rolls it
forward autoregressively. The training and prediction pipeline is the
same as for [`ESN`](@ref).

## Building the dataset

```@example lsm-orbit
using ReservoirComputing
using LuxCore: setup
using SciMLBase
using DataInterpolations
using OrdinaryDiffEqTsit5
using Plots
using Random

rng = MersenneTwister(42)

dt = 0.01
train_len, predict_len = 400, 160
period = 0.8
t = range(0.0; step = dt, length = train_len + predict_len + 1)
theta = 2 * π .* t ./ period
data = vcat(reshape(sin.(theta), 1, :), reshape(cos.(theta), 1, :))

input_data = data[:, 1:train_len]
target_data = data[:, 2:(train_len + 1)]
seed = data[:, train_len + 1]
test = data[:, (train_len + 2):(train_len + predict_len + 1)]
```

## Constructing the `LSM`

```@example lsm-orbit
N_res = 80

init_input_f64(rng, d...) = scaled_rand(rng, Float64, d...)
init_reservoir_f64(rng, d...) = dale_sparse(rng, Float64, d...)
init_state_f64(rng, d...) = zeros(Float64, d...)

lsm_train = LSM(
    2, N_res, 2, (0.0, train_len * dt), Tsit5();
    feature_map = MembraneVoltageFeature(),
    init_input = init_input_f64,
    init_reservoir = init_reservoir_f64,
    init_state = init_state_f64,
    dtmax = 5.0e-4,
    reltol = 1.0e-6, abstol = 1.0e-8
)
lsm_pred = LSM(
    2, N_res, 2, (0.0, predict_len * dt), Tsit5();
    feature_map = MembraneVoltageFeature(),
    init_input = init_input_f64,
    init_reservoir = init_reservoir_f64,
    init_state = init_state_f64,
    dtmax = 5.0e-4,
    reltol = 1.0e-6, abstol = 1.0e-8
)

ps, st = setup(rng, lsm_train)
```

## Training

```@example lsm-orbit
ps, st = train(lsm_train, input_data, target_data, ps, st;
    washout = 50, objective = RidgeRegression(1.0e-5))
```

## Autoregressive rollout

```@example lsm-orbit
output, _ = predict(
    lsm_pred, predict_len, ps, st; initialdata = seed
)

p1 = plot(test[1, :], test[2, :]; label = "actual", linewidth = 2.5,
    aspect_ratio = 1, xlabel = "x", ylabel = "y")
plot!(p1, output[1, :], output[2, :]; label = "predicted", linewidth = 2.5)

ts = (0:(predict_len - 1)) .* dt
p2 = plot(ts, test[1, :]; label = "actual", linewidth = 2.5,
    xlabel = "t", ylabel = "x(t)")
plot!(p2, ts, output[1, :]; label = "predicted", linewidth = 2.5)

plot(p1, p2; layout = (1, 2), size = (800, 350))
```
