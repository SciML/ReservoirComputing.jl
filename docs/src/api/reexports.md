# Reexported API

`using ReservoirComputing` also brings a small, fixed set of names from
[LuxCore](https://lux.csail.mit.edu/stable/api/Building_Blocks/LuxCore) and
[WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers)
into scope, so the tutorials on this site run on `using ReservoirComputing` alone.
ReservoirComputing does not document these names — it only reexports them. Their
documentation lives with the packages that own them, linked below.

## Model lifecycle (owned by LuxCore)

Every ReservoirComputing model is an `AbstractLuxLayer`, so it is built with the
standard LuxCore lifecycle rather than a ReservoirComputing-specific one:

  - `setup(rng, model)` — allocate a model's parameters and states. This is the
    second line of essentially every tutorial: `ps, st = setup(rng, esn)`, feeding
    [`train`](train.md) and [`predict`](predict.md).
  - `apply(model, x, ps, st)` — run a model on an input, returning the output and
    the updated state.

Owned and documented by
[LuxCore](https://lux.csail.mit.edu/stable/api/Building_Blocks/LuxCore).

## Custom layer interface (owned by LuxCore)

These are the two methods a custom reservoir cell implements; see
[Building a model to add to ReservoirComputing.jl](../examples/model_es2n.md) and
[Developer Interfaces](developer.md):

  - `initialparameters(rng, layer)` — the layer's trainable parameters.
  - `initialstates(rng, layer)` — the layer's non-trainable state.

Owned and documented by
[LuxCore](https://lux.csail.mit.edu/stable/api/Building_Blocks/LuxCore).

Anything else from LuxCore — `AbstractLuxLayer`, `statelength`, `outputsize`,
`replicate` and the container-layer supertypes — must be imported from LuxCore
directly. Those are extension hook points rather than names a model user calls.

## Default weight initializers (owned by WeightInitializers)

The cell constructors take `init_input`, `init_reservoir`, `init_bias`, `init_state`
and `init_orthogonal` keyword arguments. Most defaults are ReservoirComputing's own
initializers (see [Initializers](inits.md)), but five come from WeightInitializers and
are reexported so that overriding one keyword does not force you to import the
package just to name the defaults for the others:

  - `zeros32` — default `init_bias` on most cells, default `init_state` on
    `LIFESNCell`, and default `init_delay` on `DelayLayer`.
  - `randn32` — default `init_state` on `ESNCell`, `ES2NCell`, `ResESNCell` and
    `MemoryESNCell`.
  - `rand32` — default `init_weight`/`init_bias` on `LinearReadout` and default
    `init_bias` on `MemoryESNCell`.
  - `orthogonal` — default `init_orthogonal` on `ES2NCell` and `ResESNCell`.
  - `sparse_init` — the sparse sampler behind `rand_sparse` and `dale_sparse`.

Owned and documented by
[WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).

The rest of the WeightInitializers surface — `glorot_normal`, `glorot_uniform`,
`kaiming_normal`, `kaiming_uniform`, `identity_init`, `truncated_normal` and the
16/64-bit and complex `ones*`/`rand*`/`randn*`/`zeros*` variants — is **not**
reexported and must be imported from WeightInitializers directly. They are all valid
`init_*` arguments; ReservoirComputing simply does not name any of them as a default.
