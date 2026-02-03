# Building a model from scratch

ReservoirComputing.jl provides utilities to build reservoir
computing models from scratch. In this tutorial, we are going to build
an echo state network ([`ESN`](@ref)) and showcase how this custom
implementation is equivalent to the provided model (minus some comfort
utilities).

## Using provided layers: ReservoirChain, ESNCell, and LinearReadout

The library provides a [`ReservoirChain`](@ref), which is virtually
equivalent to Lux's `Chain`. Passing layers, or functions,
to the chain will concatenate them, and will allow the flow of the input
data through the model.

To build an ESN we also need a [`ESNCell`](@ref) to provide the ESN
forward pass. However, the cell is stateless, so to keep the memory of
the input we need to wrap it in a [`StatefulLayer`](@ref), which saves the
internal state in the model states `st` and feeds it to the cell in the
next step.

Finally, we need the trainable readout for the reservoir computing.
The library provides [`LinearReadout`](@ref), a dense layer the weights
of which will be trained using linear regression.

Putting it all together, we get:

```@example scratch
using ReservoirComputing

esn_scratch = ReservoirChain(
    StatefulLayer(
        ESNCell(3=>50)
    ),
    LinearReadout(50=>1)
)
```

Now, this implementation, elements naming aside, is completely equivalent to
the following

```@example scratch
esn = ESN(3, 50, 1)
```

and we can check it by initializing the two models and comparing, for instance,
the weights of the input layer:

```@example scratch
using Random
Random.seed!(43)

rng = MersenneTwister(17)
ps_s, st_s = setup(rng, esn_scratch)

rng = MersenneTwister(17)
ps, st = setup(rng, esn)

ps_s.layer_1.input_matrix == ps.reservoir.input_matrix
```

Both the models can be trained using [`train!`](@ref), and predictions can be
obtained with [`predict`](@ref). The internal states collected for linear
regression are computed by traversing the [`ReservoirChain`](@ref), and
stopping right before the [`LinearReadout`](@ref).

## Manual state collection with Collect

For more complicated models usually you would want to control when the state
collection happens. In a [`ReservoirChain`](@ref), the collection of states is
controlled by the layer [`Collect`](@ref). The role of this layer is to tell
the [`collectstates`](@ref) function where to stop for state collection. All
the readout layers have a `include_collect=true` keyword, which forces a
[`Collect`](@ref) layer before the readout. The model we wrote before can
be written as

```@example scratch
esn_scratch = ReservoirChain(
    StatefulLayer(
        ESNCell(3=>50)
    ),
    Collect(),
    LinearReadout(50=>1; include_collect=false)
)
```

to make the collection explicit. This layer is useful in case one needs to build
more complicated models such as a [`DeepESN`](@ref). We can build a deep model
in multiple ways:

```@example scratch
deepesn_scratch = ReservoirChain(
    StatefulLayer(
        ESNCell(3=>50)
    ),
    StatefulLayer(
        ESNCell(50=>50)
    ),
    StatefulLayer(
        ESNCell(50=>50)
    ),
    Collect(),
    LinearReadout(50=>1; include_collect=false)
)
```

this first approach is the one provided by default in the library through
[`DeepESN`](@ref). However, you could want the state collection to be after each
cell

```@example scratch
deepesn_scratch = ReservoirChain(
    StatefulLayer(
        ESNCell(3=>50)
    ),
    Collect(),
    StatefulLayer(
        ESNCell(50=>50)
    ),
    Collect(),
    StatefulLayer(
        ESNCell(50=>50)
    ),
    Collect(),
    LinearReadout(50=>1; include_collect=false)
)
```

With this approach, the resulting state will be a concatenation of the states at each
[`Collect`](@ref) point. So the resulting states for this architecture will be vector of
size 150.

```@example scratch
ps, st = setup(rng, deepesn_scratch)
states, st = collectstates(deepesn_scratch, rand(3, 300), ps, st)
size(states[:,1])
```

This allows for even more complex constructions, where the
state collection follows specific patterns

```@example scratch
deepesn_scratch = ReservoirChain(
    StatefulLayer(
        ESNCell(3=>50)
    ),
    StatefulLayer(
        ESNCell(50=>50)
    ),
    Collect(),
    StatefulLayer(
        ESNCell(50=>50)
    ),
    Collect(),
    LinearReadout(50=>1; include_collect=false)
)
```

Here, for instance, we have a [`Collect`](@ref) after the first two cells and then one
at the very end. You can see how the size of the states is now 100:

```@example scratch
ps, st = setup(rng, deepesn_scratch)
states, st = collectstates(deepesn_scratch, rand(3, 300), ps, st)
size(states[:,1])
```

Similar approaches could be leveraged, for instance, when the data show
multiscale dynamics that require specific modeling approaches.
