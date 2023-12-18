# Using Different Layers

A great deal of effort in the ESNs field is devoted to finding the ideal construction for the reservoir matrices. With a simple interface using ReservoirComputing.jl it is possible to leverage the currently implemented matrix construction methods for both the reservoir and the input layer. On this page, it is showcased how it is possible to change both of these layers.

The `input_init` keyword argument provided with the `ESN` constructor allows for changing the input layer. The layers provided in ReservoirComputing.jl are the following:

  - `WeightedLayer(scaling)`
  - `DenseLayer(scaling)`
  - `SparseLayer(scaling, sparsity)`
  - `MinimumLayer(weight, sampling)`
  - `InformedLayer(model_in_size; scaling=0.1, gamma=0.5)`
    In addition, the user can define a custom layer following this workflow:

```julia
#creation of the new struct for the layer
struct MyNewLayer <: AbstractLayer
    #the layer params go here
end

#dispatch over the function to actually build the layer matrix
function create_layer(input_layer::MyNewLayer, res_size, in_size)
    #the new algorithm to build the input layer goes here
end
```

Similarly the `reservoir_init` keyword argument provides the possibility to change the construction for the reservoir matrix. The available reservoir are:

  - `RandSparseReservoir(res_size, radius, sparsity)`
  - `PseudoSVDReservoir(res_size, max_value, sparsity, sorted, reverse_sort)`
  - `DelayLineReservoir(res_size, weight)`
  - `DelayLineBackwardReservoir(res_size, weight, fb_weight)`
  - `SimpleCycleReservoir(res_size, weight)`
  - `CycleJumpsReservoir(res_size, cycle_weight, jump_weight, jump_size)`
    And, like before, it is possible to build a custom reservoir by following this workflow:

```julia
#creation of the new struct for the reservoir
struct MyNewReservoir <: AbstractReservoir
    #the reservoir params go here
end

#dispatch over the function to build the reservoir matrix
function create_reservoir(reservoir::AbstractReservoir, res_size)
    #the new algorithm to build the reservoir matrix goes here
end
```

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
    MinimumLayer(0.85, IrrationalSample()),
    MinimumLayer(0.95, IrrationalSample()),
]
reservoirs = [SimpleCycleReservoir(res_size, 0.7),
    CycleJumpsReservoir(res_size, cycle_weight = 0.7, jump_weight = 0.2, jump_size = 5)]

for i in 1:length(reservoirs)
    esn = ESN(training_input;
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
