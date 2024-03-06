# Using Different Layers

A great deal of effort in the ESNs field is devoted to finding the ideal construction for the reservoir matrices. With a simple interface using ReservoirComputing.jl it is possible to leverage the currently implemented matrix construction methods for both the reservoir and the input layer. On this page, it is showcased how it is possible to change both of these layers.

ReservoirComputing.jl follows the standard set by [WeightInitializers.jl](https://github.com/LuxDL/WeightInitializers.jl) to define the initialization functions for both reservoirs and input layers. 

## Example of a minimally complex ESN

Using [^1] and [^2] as references, this section will provide an example of how to change both the input layer and the reservoir for ESNs. The full script for this example can be found [here](https://github.com/MartinuzziFrancesco/reservoir-computing-examples/blob/main/change_layers/layers.jl). This example was run on Julia v1.7.2.

The task for this example will be the one step ahead prediction of the Henon map. To obtain the data, one can leverage the package [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/dev/). The data is scaled to be between -1 and 1.

```@example mesn
using PredefinedDynamicalSystems
train_len = 3000
predict_len = 2000

ds = PredefinedDynamicalSystems.henon()
traj, time = trajectory(ds, 7000)
data = Matrix(traj)'
data = (data .- 0.5) .* 2
shift = 200

training_input = data[:, shift:(shift + train_len - 1)]
training_target = data[:, (shift + 1):(shift + train_len)]
testing_input = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]
testing_target = data[:, (shift + train_len + 1):(shift + train_len + predict_len)]
```

Now it is possible to define the input layers and reservoirs we want to compare and run the comparison in a simple for loop. The accuracy will be tested using the mean squared deviation `msd` from [StatsBase](https://juliastats.org/StatsBase.jl/stable/).

```@example mesn
using ReservoirComputing, StatsBase

res_size = 300
input_layer = [
    minimal_init(;weight=0.85, sampling_type=:irrational),
    minimal_init(;weight=0.95, sampling_type=:irrational)
]
reservoirs = [simple_cycle(;weight=0.7),
    cycle_jumps(;cycle_weight = 0.7, jump_weight = 0.2, jump_size = 5)]

for i in 1:length(reservoirs)
    esn = ESN(size(training_input, 1), res_size, training_input;
        input_layer = input_layer[i],
        reservoir = reservoirs[i])
    wout = train(esn, training_target, StandardRidge(0.001))
    output = esn(Predictive(testing_input), wout)
    println(msd(testing_target, output))
end
```

As it is possible to see, changing layers in ESN models is straightforward. Be sure to check the API documentation for a full list of reservoirs and layers.

## Bibliography

[^1]: Rodan, Ali, and Peter Tiňo. “Simple deterministically constructed cycle reservoirs with regular jumps.” Neural computation 24.7 (2012): 1822-1852.
[^2]: Rodan, Ali, and Peter Tiňo. “Minimum complexity echo state network.” IEEE transactions on neural networks 22.1 (2010): 131-144.
