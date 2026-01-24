# Saving and loading models

ReservoirComputing.jl borrows the same structure as Lux.jl,
where each model is defined by its own parameters `ps` and states `st`.
Therefore, in order to save a model, it suffices to save the hyperparameters that
define the model, its parameters `ps`, and its states `st`. In this example we are
going to show how saving and loading a model can be done leveraging
[JLD2.jl](https://github.com/JuliaIO/JLD2.jl).

Let's assume you have trained an [ESN](@ref), and you want to save it. Following the
[getting started](getting_started.md) example we are going to train the ESN on the Lorenz system:

```@example saveload
using OrdinaryDiffEq
using Plots
using Random
using ReservoirComputing

Random.seed!(42)
rng = MersenneTwister(17)

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

prob = ODEProblem(lorenz, [1.0f0, 0.0f0, 0.0f0], (0.0, 200.0), Float32[10.0, 28.0, 8/3])
data = Array(solve(prob, ABM54(); dt=0.02f0))
shift = 300
train_len = 5000
predict_len = 1250

input_data = data[:, shift:(shift + train_len - 1)]
target_data = data[:, (shift + 1):(shift + train_len)]
test = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]

esn = ESN(3, 300, 3; init_reservoir=rand_sparse(; radius=1.2, sparsity=6/300),
    state_modifiers=NLAT2)

ps, st = setup(rng, esn)
ps, st = train!(esn, input_data, target_data, ps, st)
```

Now that we have a trained model we want to save both the parameters and states, as well as
the hyperparameters that define the model. We can do so by creating an additional `NamedTuple`
for the hyperparameters

```@example saveload
spec = (
  in_size = 3,
  res_size = 300,
  out_size = 3,
  radius = 1.2,
  sparsity = 6/300,
  state_modifiers = :NLAT2,
  # include any non-default knobs you used:
  # leak_coefficient = 1.0, input_scaling = 0.1, use_bias=false, etc.
)
```

We can now save the model:

```@example saveload
using JLD2
@save "esn_trained.jld2" ps st spec
```

In order to load the model and use it we still rely on JLD2.jl, using the `@load` macro:

```@example saveload
@load "esn_trained.jld2" ps st spec

# Rebuild the same ESN architecture (must match ps structure)
esn = ESN(spec.in_size, spec.res_size, spec.out_size;
          init_reservoir=rand_sparse(; radius=spec.radius, sparsity=spec.sparsity),
          state_modifiers = getfield(ReservoirComputing, spec.state_modifiers))

# Now you can predict using the loaded ps/st
output, st = predict(esn, predict_len, ps, st; initialdata=test[:, 1])
```
