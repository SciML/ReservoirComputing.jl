# Getting Started with ReservoirComputing.jl

This is an introductory tutorial for ReservoirComputing.jl. We will showcase
a typical use case by creating an Echo State Network (ESN) and training it
to reproduce the dynamics of the chaotic Lorenz system.

## Installing ReservoirComputing

ReservoirComputing.jl is registered in the General registry, so it can be installed
through the Julia package manager:

```julia
using Pkg
Pkg.add("ReservoirComputing")
```

## Copy-Pastable Simplified Example

If you wish to just get some code running to get started, the following code
block provides an end-to-end simplified runnable example. The rest of this
page will delve into more details, expanding on various aspects of the example.

```@example first-esn
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
output, st = predict(esn, predict_len, ps, st; initialdata=test[:, 1])

plot(transpose(output)[:, 1], transpose(output)[:, 2], transpose(output)[:, 3];
    label="predicted")
plot!(transpose(test)[:, 1], transpose(test)[:, 2], transpose(test)[:, 3];
    label="actual")
```

Congrats, you trained your first ESN! Now let's go into more detail.

## Generating the data

Starting off the workflow, the first step is to obtain the data.
We use `OrdinaryDiffEq` to derive the Lorenz system data.
The data is passed to the model as a matrix, where the rows are
the features and the columns represent the time steps.

```@example lorenz
using OrdinaryDiffEq

# define lorenz system
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

# solve and take data
prob = ODEProblem(lorenz!, [1.0, 0.0, 0.0], (0.0, 200.0))
data = solve(prob, ABM54(); dt=0.02)
data = reduce(hcat, data.u)
```

Now we split the data into training and testing. To do an autoregressive
forecast we want the model to be trained on the next step, so we are
going to shift the target data by one. Additionally, we discard the
transient period.

```@example lorenz
#determine shift length, training length and prediction length
shift = 300
train_len = 5000
predict_len = 1250

#split the data accordingly
input_data = data[:, shift:(shift + train_len - 1)]
target_data = data[:, (shift + 1):(shift + train_len)]
test_data = data[:, (shift + train_len + 1):(shift + train_len + predict_len)]
```

It is **important** to notice that the data needs to be formatted in a
matrix with the features as rows and time steps as columns as in this example.
This is needed even if the time series consists of single values (ie, the data
can be represented by a vector).

## Building the Echo State Network

Once the data is ready, it is possible to define the parameters for
the ESN and the `ESN` struct itself. In this example, the values
from [Pathak2017](@cite) are loosely followed as general guidelines.

```@example lorenz
using ReservoirComputing

#define ESN parameters
res_size = 300
in_size = 3
res_radius = 1.2
res_sparsity = 6 / 300
input_scaling = 0.1

#build ESN struct
esn = ESN(in_size, res_size, in_size; #autoregressive so in_size = out_size
    init_reservoir = rand_sparse(; radius = res_radius, sparsity = res_sparsity),
    init_input = weighted_init(; scaling = input_scaling),
    state_modifiers = NLAT2,
    leak_coefficient=1.0, # default value
    init_state = randn32, # default value
    use_bias=false, # default value
    init_bias = zeros32, # default value, not used since use_bias=false
    readout_activation=identity, # default value
)
```

In this case, a size of 300 has been chosen, so the reservoir matrix will be 300 x 300.
However, this is not always the case, since some input layer
constructions can modify the dimensions of the reservoir. Please make sure to read the
API documentation of the initializer you intend to use if you think that
is cause of errors.

The `res_radius` determines the scaling of the spectral radius of the reservoir matrix;
a proper scaling is necessary to assure the echo state property.
The default value in the [`rand_sparse`](@ref) method is 1.0 in accordance with the most
commonly followed guidelines found in the literature (see [Lukoeviius2012](@cite)
and references therein).

The value of `input_scaling` determines the upper and lower bounds of the
uniform distribution of the weights in the [`weighted_init`](@ref).
The value of 0.1 represents the default. The default input layer is
the [`scaled_rand`](@ref), a dense matrix. The details of the weighted version
can be found in [Lu2017](@cite). For this example, this version returns
the best results.

## Training and Prediction

Training for ESNs usually means solving a linear regression. The library supports
solvers from [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl),
in addition to a custom implementation of ridge regression [`StandardRidge`](@ref).
In this example we will use the latter.

Since `ReservoirComputing.jl` builds on
[`LuxCore.jl`](https://lux.csail.mit.edu/stable/api/Building_Blocks/LuxCore)
we first need to setup the state and the parameters

```@example lorenz
using Random
Random.seed!(42)
rng = MersenneTwister(17)

ps, st = setup(rng, esn)
```

Now we can proceed with training the ESN model. Usually an initial transient
is discarded, to account for the dynamics of the ESN to settle. This can
be done by passing the `washout` keyword argument to `train`.

```@example lorenz
#define training method
training_method = StandardRidge(0.0)

ps, st = train!(esn, input_data, target_data, ps, st, training_method;
    washout = 0 # we use no washout
)
```

`ps` now contains the trained parameters for the ESN.

!!! info "Returning training states"

    The ESN states are internally used the training, however they are not returned by
    default. To inspect the states, it is necessary to set the boolean keyword
    argument `return_states` as `true` in the [`train!`](@ref) call.
    
    ```julia
    (ps, st), states = train!(esn, input_data, target_data, ps, st, training_method;
        return_states = true
    )
    ```

ReservoirComputing.jl provides
additional utilities functions for autoregressive forecasting:

```@example lorenz
output, st = predict(esn, predict_len, ps, st; initialdata=test_data[:, 1])
```

To inspect the results, they can easily be plotted using an external library.
In this case, we will use `Plots.jl`:

```@example lorenz
using Plots, Plots.PlotMeasures

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
Pages = ["getting_started.md"]
Canonical = false
```
