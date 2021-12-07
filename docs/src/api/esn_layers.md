# ESN Layers

## Input Layers
```@docs
    WeightedLayer
    DenseLayer
    SparseLayer
    InformedLayer
    MinimumLayer
```
The sign in the ```MinimumLayer``` are chosen based on the following methods:
```@docs
    BernoulliSample
    IrrationalSample
```
To derive the matrix one can call the following function:
```@docs
    create_layer
```
To create new input layers it suffice to define a new struct containing the needed parameters of the new input layer. This struct wiil need to be an ```AbstractLayer```, so the ```create_layer``` function can be dispatched over it. The workflow should follow this snippet:
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

## Reservoirs
```@docs
    RandSparseReservoir
    PseudoSVDReservoir
    DelayLineReservoir
    DelayLineBackwardReservoir
    SimpleCycleReservoir
    CycleJumpsReservoir
```

Like for the input layers, to actually build the matrix of the reservoir one can call the following function:
```@docs
    create_reservoir
```

To create a new reservoir the procedure is imilar to the one for the input layers. First the definition of the new struct of type ```AbstractReservoir``` with the reservoir parameters is needed. Then the dispatch over the ```create_reservoir``` function makes the model actually build the reservoir matrix. An example of the workflow is given in the following snippet:
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
