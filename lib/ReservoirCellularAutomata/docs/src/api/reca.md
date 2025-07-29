# Reservoir Computing with Cellular Automata

```@docs
    RECA
```

The input encodings are the equivalent of the input matrices of the ESNs. These are the available encodings:

```@docs
    RandomMapping
```

The training and prediction follow the same workflow as the ESN. It is important to note that currently we were unable to find any papers using these models with a `Generative` approach for the prediction, so full support is given only to the `Predictive` method.
