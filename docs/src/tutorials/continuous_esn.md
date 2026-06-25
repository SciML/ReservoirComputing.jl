# Continuous ESN: forecasting Lorenz

[`ContinuousESN`](@ref) is a thin wrapper around
[`SciMLProblemReservoir`](@ref) that pre-bakes the leaky-integrator
continuous Echo State Network ODE of [Lukosevicius2012](@cite) §3.2.6
eq (5):

```math
\dot{\mathbf{x}}(t) = \alpha \left(
    -\mathbf{x}(t) + \tanh\!\left(
        \mathbf{W}_{\text{in}}\,\mathbf{u}(t) + \mathbf{W}_r\,\mathbf{x}(t)
        + \mathbf{b}
    \right)
\right)
```

with leaking rate `α`. Forward-Euler discretisation at step `Δt = 1`
recovers the discrete leaky ESN `x(n+1) = (1-α) x(n) + α tanh(...)`
exactly. The reservoir matrices `W_r`, `W_in`, and optional bias `b`
live in `ps.reservoir` and are constructed by `setup(rng, rc)`; the
ODE solver and any solve-time keyword arguments are captured at
construction. Under the hood the same `RCODEReservoirExt` extension
that powers `SciMLProblemReservoir` also runs `ContinuousESN`.

This tutorial walks through training a `ContinuousESN` on Lorenz-63
data and rolling it forward autoregressively to reproduce the
attractor. `SciMLBase` provides `solve` / `remake`, `DataInterpolations`
backs the per-window input signal in autoregressive mode, and an
OrdinaryDiffEq solver package (e.g. `OrdinaryDiffEqTsit5`) supplies the
concrete solver type.

## Building a Lorenz dataset

```@example continuous-esn-lorenz
using ReservoirComputing
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

The constructor mirrors `SciMLProblemReservoir`: the integration
`tspan` and the solver positional argument are captured at
construction time, exactly as in DiffEqFlux's `NeuralODE`.

```@example continuous-esn-lorenz
N_res = 100
ce_train = ContinuousESN(
    3, N_res, (0.0, Float64(train_len)), Tsit5();
    leak_coefficient = 1.0, use_bias = true, T = Float64,
    reltol = 1.0e-6, abstol = 1.0e-8
)
ce_pred = ContinuousESN(
    3, N_res, (0.0, Float64(predict_len)), Tsit5();
    leak_coefficient = 1.0, use_bias = true, T = Float64,
    reltol = 1.0e-6, abstol = 1.0e-8
)

rc_train = ReservoirComputer(ce_train, (NLAT2(),), LinearReadout(N_res => 3))
rc_pred = ReservoirComputer(ce_pred, (NLAT2(),), LinearReadout(N_res => 3))

ps, st = setup(rng, rc_train)
```

## Training

`train!` is sampler-agnostic: it routes through the continuous
`_collectstates` provided by the extension and fits a linear readout
on the collected states.

```@example continuous-esn-lorenz
ps, st = train!(rc_train, input_data, target_data, ps, st)
```

## Autoregressive rollout

`predict(rc, steps, ps, st; initialdata)` splits the predict-time
`tspan` into `steps` equal sub-intervals; on each sub-interval the
previous readout output is held constant as the input. The default
initial reservoir state is `prob.u0` (zeros) — if you want to continue
from the trained reservoir's terminal state, `remake` the prob before
constructing the predict reservoir.

```@example continuous-esn-lorenz
ps_pred, st_pred = setup(rng, rc_pred)
ps_pred = merge(ps_pred, (readout = ps.readout,))
st_pred = merge(st_pred, (readout = st.readout,))

output, _ = predict(
    rc_pred, predict_len, ps_pred, st_pred; initialdata = test[:, 1]
)

plot(
    transpose(output)[:, 1], transpose(output)[:, 2],
    transpose(output)[:, 3]; label = "predicted"
)
plot!(
    transpose(test)[:, 1], transpose(test)[:, 2],
    transpose(test)[:, 3]; label = "actual"
)
```

The two trajectories agree on the early portion of the rollout before
chaotic divergence dominates — the same behaviour the discrete-ESN
tutorial produces. The point of the example is that nothing in the
training loop changes between discrete ESN, `SciMLProblemReservoir`
with hand-rolled equations, and `ContinuousESN`: the same `train!` /
`predict` pipeline drives all three.

## When to reach for `ContinuousESN` vs `SciMLProblemReservoir`

* `ContinuousESN` pre-bakes eq (5); reach for it when the standard
  leaky-integrator continuous ESN is what you want and you'd otherwise
  be hand-rolling the same RHS.
* [`SciMLProblemReservoir`](@ref) is the generic building block; reach
  for it when the reservoir ODE is *not* eq (5) — bespoke RHS, SDE,
  DDE, or non-standard parameter layout.

Both go through the same `_collectstates` / `_predict` dispatch and
the same protected-`saveat` discipline.
