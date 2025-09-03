abstract type AbstractReservoirCollectionLayer <: AbstractLuxLayer end
abstract type AbstractReservoirRecurrentCell <: AbstractLuxLayer end
abstract type AbstractReservoirTrainableLayer <: AbstractLuxLayer end

### Readout
# adapted from lux layers/basic Dense
@doc raw"""
    Readout(in_dims => out_dims, [activation];
            use_bias=false, include_collect=true)

Linear readout layer with optional bias and elementwise activation. Intended as
the final, trainable mapping from collected features (e.g., reservoir state) to
outputs. When `include_collect=true`, training will collect features immediately
before this layer (logically inserting a [`Collect()`](@ref) right before it).

## Equation

```math
\mathbf{y} = \psi\!\left(\mathbf{W}\,\mathbf{z} + \mathbf{b}\right)
```

## Arguments

- `in_dims`: Input/feature dimension (e.g., reservoir size).
- `out_dims`: Output dimension (e.g., number of targets).
- `activation`: Elementwise output nonlinearity. Default: `identity`.

## Keyword arguments

- `use_bias`: Include an additive bias vector `b`. Default: `false`.
- `include_collect`: If `true` (default), training collects features immediately
  before this layer (as if a [`Collect()`](@ref) were inserted right before it).

## Parameters

- `weight :: (out_dims × in_dims)`
- `bias   :: (out_dims,)` — present only if `use_bias=true`

## States

- None.

## Notes

- In ESN workflows, readout weights are typically set via ridge regression in
  `train!(...)`. Therefore, how `Readout` gets initialized is of no consequence.
- If you set `include_collect=false`, make sure a [`Collect()`](@ref) appears earlier in the chain.
  Otherwise training may operate on the post-readout signal,
  which is usually unintended.
"""
@concrete struct Readout <: AbstractReservoirTrainableLayer
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    use_bias <: StaticBool
    include_collect <: StaticBool
end

function Base.show(io::IO, ro::Readout)
    print(io, "Readout($(ro.in_dims) => $(ro.out_dims)")
    (ro.activation == identity) || print(io, ", $(ro.activation)")
    has_bias(ro) || print(io, ", use_bias=false")
    ic = known(getproperty(ro, Val(:include_collect)))
    ic === true && print(io, ", include_collect=true")
    return print(io, ")")
end

function Readout(mapping::Pair{<:IntegerType,<:IntegerType}, activation=identity; kwargs...)
    return Readout(first(mapping), last(mapping), activation; kwargs...)
end

function Readout(in_dims::IntegerType, out_dims::IntegerType, activation=identity;
    include_collect::BoolType=True(), use_bias::BoolType=False())
    return Readout(activation, in_dims, out_dims, static(use_bias), static(include_collect))
end

function initialparameters(rng::AbstractRNG, ro::Readout)
    weight = rand(rng, Float32, ro.out_dims, ro.in_dims)

    if has_bias(ro)
        return (; weight, bias=rand(rng, Float32, ro.out_dims))
    else
        return (; weight)
    end
end

parameterlength(ro::Readout) = ro.out_dims * ro.in_dims + has_bias(ro) * ro.out_dims
statelength(ro::Readout) = 0

outputsize(ro::Readout, _, ::AbstractRNG) = (ro.out_dims,)

function (ro::Readout)(inp::AbstractArray, ps, st::NamedTuple)
    out_tmp = ps.weight * inp
    if has_bias(ro)
        out_tmp += ps.bias
    end
    output = ro.activation.(out_tmp)
    return output, st
end

@doc raw"""
    Collect()

Marker layer that passes data through unchanged but marks a feature
checkpoint for [`collectstates`](@ref). At each time step, whenever a `Collect()` is
encountered in the chain, the current vector is recorded as part of the feature
vector used to train the readout. If multiple `Collect()` layers exist, their
vectors are concatenated with `vcat` in order of appearance.

## Arguments

- None.

## Keyword arguments

- None.

## Inputs

- `x :: AbstractArray (d, batch)` — the current tensor flowing through the chain.

## Returns

- `(x, st)` — the same tensor `x` and the **unchanged** state `st`.

## Parameters

- None (`Collect` has no trainable parameters).

## States

- None (state is passed through unchanged).

## Notes

- When used with a single `Collect()` before a [`Readout`](@ref), training uses exactly
  the tensor right before the readout (e.g., the reservoir state).
- With **multiple** `Collect()` layers (e.g., after different submodules), the
  per-step features are `vcat`-ed in chain order to form one feature vector.
- If the readout is constructed with `include_collect=true`, an *implicit*
  collection point is assumed immediately before the readout. Use an explicit
  `Collect()` only when you want to control where/what is collected (or to stack
  multiple features).

  ```julia
  rc = ReservoirChain(
          StatefulLayer(ESNCell(3 => 300)),
          NLAT2(),
          Collect(), # <-- collect the 300-dim reservoir after NLAT2
          Readout(300 => 3; include_collect=false) # <-- toggle off the default Collect()
      )
```
"""
struct Collect <: AbstractReservoirCollectionLayer end

function (cl::Collect)(inp::AbstractArray, ps, st::NamedTuple)
    return inp, st
end

Base.show(io::IO, cl::Collect) = print(io, "Collection point of states")

@doc raw"""
    collectstates(rc, data, ps, st)

Run the sequence `data` once through the reservoir chain `rc`, advancing the
model state over time, and collect feature vectors at every [`Collect()`](@ref) layer.
If more than one [`Collect()`](ref) is encountered in a step, their vectors are
concatenated with `vcat` in order of appearance. If no [`Collect()`](@ref) is seen
in a step, the feature defaults to the final vector exiting the chain for
that time step.

!!! note
    If your [`Readout`](@ref) layer was created with `include_collect=true`
    (default behaviour), a collection point is placed immediately before the readout,
    so the collected features are the inputs to the readout.

## Arguments

- `rc`: A [`ReservoirChain`](@ref) (or compatible [`AbstractLuxLayer`](@extref) with `.layers`).
- `data`: Input sequence of shape `(in_dims, T)`, where columns are time steps.
- `ps`, `st`: Current parameters and state for `rc`.

## Returns

- `states`: Reservoir states, i.e. a feature matrix with one column per
  time step. The feature dimension `n_features` equals the vertical concatenation
  of all vectors captured at [`Collect()`](@ref) layers in that step.
- `st`: Updated model states.

"""
function collectstates(rc::AbstractLuxLayer, data::AbstractArray, ps, st::NamedTuple)
    newst = st
    collected = Any[]
    for inp in eachcol(data)
        inp_tmp = inp
        state_vec = nothing
        for (name, layer) in pairs(rc.layers)
            if layer isa AbstractReservoirTrainableLayer
                break
            end
            inp_tmp, st_i = layer(inp_tmp, ps[name], newst[name])
            newst = merge(newst, (; name => st_i))
            if layer isa AbstractReservoirCollectionLayer
                state_vec = state_vec === nothing ? copy(inp_tmp) : vcat(state_vec, inp_tmp)
            end
        end
        push!(collected, state_vec === nothing ? copy(inp_tmp) : state_vec)
    end
    states = eltype(data).(reduce(hcat, collected))
    return states, newst
end
