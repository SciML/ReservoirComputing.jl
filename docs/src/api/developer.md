# Developer Interfaces

!!! warning
    These interfaces describe implementation contracts used to maintain and
    extend ReservoirComputing.jl. They are versioned and tested, but they are
    not ordinary end-user model constructors. Prefer the documented concrete
    models, cells, and [`ReservoirComputer`](@ref) composition API unless you
    are implementing a new reservoir family.

## Reservoir Containers

```@docs
ReservoirComputing.AbstractReservoirComputer
ReservoirComputing.AbstractEchoStateNetwork
```

## Recurrent Cells

```@docs
ReservoirComputing.AbstractReservoirRecurrentCell
ReservoirComputing.AbstractEchoStateNetworkCell
ReservoirComputing.AbstractReservoirCollectionLayer
ReservoirComputing.AbstractReservoirTrainableLayer
```

## Continuous and Cellular Automata Reservoirs

```@docs
ReservoirComputing.AbstractSciMLProblemReservoir
ReservoirComputing.AbstractSampler
ReservoirComputing.AbstractInputEncoding
ReservoirComputing.AbstractEncodingData
```

## Training

```@docs
ReservoirComputing.AbstractReservoirComputingSolver
```
