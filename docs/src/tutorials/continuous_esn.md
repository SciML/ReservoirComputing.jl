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
N_res = 100

# Float64 initialisers so the reservoir, the solve, and the input all
# share a numeric type. Without these the cell would default to
# Float32 via `scaled_rand` / `rand_sparse` / `zeros32`.
init_input_f64(rng, d...)     = scaled_rand(rng, Float64, d...)
init_reservoir_f64(rng, d...) = rand_sparse(rng, Float64, d...)
init_bias_f64(rng, d...)      = zeros(Float64, d...)

esn_train = ContinuousESN(
    3, N_res, 3, (0.0, Float64(train_len)), Tsit5();
    use_bias = true,
    init_input = init_input_f64,
    init_reservoir = init_reservoir_f64,
    init_bias = init_bias_f64,
    state_modifiers = (NLAT2(),),
    reltol = 1.0e-6, abstol = 1.0e-8
)
esn_pred = ContinuousESN(
    3, N_res, 3, (0.0, Float64(predict_len)), Tsit5();
    use_bias = true,
    init_input = init_input_f64,
    init_reservoir = init_reservoir_f64,
    init_bias = init_bias_f64,
    state_modifiers = (NLAT2(),),
    reltol = 1.0e-6, abstol = 1.0e-8
)

ps, st = setup(rng, esn_train)
```

## Training

```@example continuous-esn-lorenz
ps, st = train!(esn_train, input_data, target_data, ps, st)
```

## Autoregressive rollout

```@example continuous-esn-lorenz
ps_pred, st_pred = setup(rng, esn_pred)
ps_pred = merge(ps_pred, (readout = ps.readout,))
st_pred = merge(st_pred, (readout = st.readout,))

output, _ = predict(
    esn_pred, predict_len, ps_pred, st_pred; initialdata = test[:, 1]
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

* `ContinuousESN` pre-bakes the continuous ESN ODE; use it when the
  standard continuous ESN is what you want.
* [`SciMLProblemReservoir`](@ref) is the generic building block; use it
  when the reservoir ODE is not the standard eq (5) — bespoke RHS, SDE,
  DDE, or non-standard parameter layout.
