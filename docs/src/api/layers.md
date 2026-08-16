# Layers

## Base Layers

```@docs
    ReservoirComputer
    ReservoirChain
    Collect
    StatefulLayer
    DelayLayer
    NonlinearFeaturesLayer
```

## Readout Layers

```@docs
    LinearReadout
    SVMReadout
```

## Echo State Networks

```@docs
    AdditiveEIESNCell
    EIESNCell
    ES2NCell
    ESNCell
    EuSNCell
    MemoryESNCell
    MemoryResESNCell
```

```@docs
    RMNCell
    ResESNCell
    LIFESNCell
```

## Continuous-time Layers

```@docs
    ContinuousESNCell
    LSMCell
```

## Spiking

```@docs
    AbstractSpikingNeuron
    LIFNeuron
    AbstractInputEncoder
    CurrentInjection
    PoissonRateEncoder
    AbstractSpikeFeature
    SpikeCountFeatures
    ExponentialSpikeFilter
    MembraneVoltageFeature
```

## Wrappers

```@docs
    LocalInformationFlow
```

## Continuous-Time Reservoirs

```@docs
    SciMLProblemReservoir
    TerminalStateSampling
```

## Reservoir computing with cellular automata

```@docs
    RECACell
```
