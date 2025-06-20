# Using different layers

A great deal of efforts in the ESNs field are devoted to finding an ideal construction
for the reservoir matrices. ReservoirComputing.jl offers multiple implementation of
reservoir and input matrices initializations found in the literature.
The API is standardized, and follows 
[WeightInitializers.jl](https://github.com/LuxDL/Lux.jl/tree/main/lib/WeightInitializers):

```julia
weights = init(rng, dims...)
#rng is optional
weights = init(dims...)
```

Additional keywords can be added when needed:

```julia
weights_init = init(rng; kwargs...)
weights = weights_init(rng, dims...)
# or
weights_init = init(; kwargs...)
weights = weights_init(dims...)
```

Custom layers only need to follow these APIs to be compatible with ReservoirComputing.jl.

## Example of minimally complex ESN

Using [Rodan2012](@cite) and [Rodan2011](@cite) as references this section will provide an
example on how to change both the input layer and the reservoir for ESNs.

The task for this example will be the one step ahead prediction of the Henon map.
To obtain the data one can leverage the package
[PredefinedDynamicalSystems.jl](https://juliadynamics.github.io/PredefinedDynamicalSystems.jl/dev/).
The data is scaled to be between -1 and 1.

```@example minesn
using PredefinedDynamicalSystems
train_len = 3000
predict_len = 2000

ds = Systems.henon()
traj, t = trajectory(ds, 7000)
data = Matrix(traj)'
data = (data .- 0.5) .* 2
shift = 200

training_input = data[:, shift:(shift + train_len - 1)]
training_target = data[:, (shift + 1):(shift + train_len)]
testing_input = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]
testing_target = data[:, (shift + train_len + 1):(shift + train_len + predict_len)]
```

Now it is possible to define the input layers and reservoirs we want to compare and run
the comparison in a simple for loop. The accuracy will be tested using the mean squared
deviation msd from StatsBase.

```@example minesn
using ReservoirComputing, StatsBase

res_size = 300
input_layer = [minimal_init(; weight=0.85, sampling_type=:irrational_sample!),
    minimal_init(; weight=0.95, sampling_type=:irrational_sample!)]
reservoirs = [simple_cycle(; weight=0.7),
    cycle_jumps(; cycle_weight=0.7, jump_weight=0.2, jump_size=5)]

for i in 1:length(reservoirs)
    esn = ESN(training_input, 2, res_size;
        input_layer=input_layer[i],
        reservoir=reservoirs[i])
    wout = train(esn, training_target, StandardRidge(0.001))
    output = esn(Predictive(testing_input), wout)
    println(msd(testing_target, output))
end
```

As it is possible to see, changing layers in ESN models is straightforward.
Be sure to check the API documentation for a full list of reservoir and layers.

## References

```@bibliography
Pages = ["change_layers.md"]
Canonical = false
```
