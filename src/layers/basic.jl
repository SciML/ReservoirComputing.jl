abstract type AbstractReservoirCollectionLayer <: AbstractLuxLayer end
abstract type AbstractReservoirRecurrentCell <: AbstractLuxLayer end
abstract type AbstractReservoirTrainableLayer <: AbstractLuxLayer end

### LinearReadout
# adapted from lux layers/basic Dense
@doc raw"""
    LinearReadout(in_dims => out_dims, [activation];
            use_bias=false, include_collect=true)

Linear readout layer with optional bias and elementwise activation. Intended as
the final, trainable mapping from collected features (e.g., reservoir state) to
outputs. When `include_collect=true`, training will collect features immediately
before this layer (logically inserting a [`Collect`](@ref) right before it).

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
  before this layer (as if a [`Collect`](@ref) were inserted right before it).

## Parameters

- `weight :: (out_dims × in_dims)`
- `bias   :: (out_dims,)` — present only if `use_bias=true`

## States

- None.

## Notes

- In ESN workflows, readout weights are typically replaced via ridge regression in
  [`train!`](@ref). Therefore, how `LinearReadout` gets initialized is of no consequence.
  Additionally, the dimensions will also not be taken into account, as [`train!`](@ref)
  will replace the weights.
- If you set `include_collect=false`, make sure a [`Collect`](@ref) appears earlier in the chain.
  Otherwise training may operate on the post-readout signal,
  which is usually unintended.
"""
@concrete struct LinearReadout <: AbstractReservoirTrainableLayer
    activation::Any
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_weight::Any
    init_bias::Any
    use_bias <: StaticBool
    include_collect <: StaticBool
end

function LinearReadout(
        mapping::Pair{<:IntegerType, <:IntegerType}, activation = identity; kwargs...)
    return LinearReadout(first(mapping), last(mapping), activation; kwargs...)
end

function LinearReadout(in_dims::IntegerType, out_dims::IntegerType, activation = identity;
        init_weight = rand32, init_bias = rand32, include_collect::BoolType = True(),
        use_bias::BoolType = False())
    return LinearReadout(activation, in_dims, out_dims, init_weight,
        init_bias, static(use_bias), static(include_collect))
end

function initialparameters(rng::AbstractRNG, ro::LinearReadout)
    weight = ro.init_weight(rng, ro.out_dims, ro.in_dims)

    if has_bias(ro)
        return (; weight, bias = ro.init_bias(rng, ro.out_dims))
    else
        return (; weight)
    end
end

parameterlength(ro::LinearReadout) = ro.out_dims * ro.in_dims + has_bias(ro) * ro.out_dims
statelength(ro::LinearReadout) = 0

outputsize(ro::LinearReadout, _, ::AbstractRNG) = (ro.out_dims,)

function (ro::LinearReadout)(inp::AbstractArray, ps, st::NamedTuple)
    out_tmp = ps.weight * inp
    if has_bias(ro)
        out_tmp += ps.bias
    end
    output = ro.activation.(out_tmp)
    return output, st
end

function Base.show(io::IO, ro::LinearReadout)
    print(io, "LinearReadout($(ro.in_dims) => $(ro.out_dims)")
    (ro.activation == identity) || print(io, ", $(ro.activation)")
    has_bias(ro) || print(io, ", use_bias=false")
    ic = known(getproperty(ro, Val(:include_collect)))
    ic === true && print(io, ", include_collect=true")
    return print(io, ")")
end

@doc raw"""
    Collect()

Marker layer that passes data through unchanged but marks a feature
checkpoint for [`collectstates`](@ref). At each time step, whenever a `Collect` is
encountered in the chain, the current vector is recorded as part of the feature
vector used to train the readout. If multiple `Collect` layers exist, their
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

- None.

## States

- None.

## Notes

- When used with a single `Collect` before a [`LinearReadout`](@ref), training uses exactly
  the tensor right before the readout (e.g., the reservoir state).
- With **multiple** `Collect` layers (e.g., after different submodules), the
  per-step features are `vcat`-ed in chain order to form one feature vector.
- If the readout is constructed with `include_collect=true`, an *implicit*
  collection point is assumed immediately before the readout. Use an explicit
  `Collect` only when you want to control where/what is collected (or to stack
  multiple features).

  ```julia
  rc = ReservoirChain(
          StatefulLayer(ESNCell(3 => 300)),
          NLAT2(),
          Collect(), # <-- collect the 300-dim reservoir after NLAT2
          LinearReadout(300 => 3; include_collect=false) # <-- toggle off the default Collect()
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
model state over time, and collect feature vectors at every [`Collect`](@ref) layer.
If more than one [`Collect`](@ref) is encountered in a step, their vectors are
concatenated with `vcat` in order of appearance. If no [`Collect`](@ref) is seen
in a step, the feature defaults to the final vector exiting the chain for
that time step.

!!! note
    If your [`LinearReadout`](@ref) layer was created with `include_collect=true`
    (default behaviour), a collection point is placed immediately before the readout,
    so the collected features are the inputs to the readout.

## Arguments

- `rc`: A [`ReservoirChain`](@ref) (or compatible `AbstractLuxLayer` with `.layers`).
- `data`: Input sequence of shape `(in_dims, T)`, where columns are time steps.
- `ps`, `st`: Current parameters and state for `rc`.

## Returns

- `states`: Reservoir states, i.e. a feature matrix with one column per
  time step. The feature dimension `n_features` equals the vertical concatenation
  of all vectors captured at [`Collect`](@ref) layers in that step.
- `st`: Updated model states.

"""
function collectstates(rc::AbstractLuxLayer, data::AbstractMatrix, ps, st::NamedTuple)
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
    @assert !isempty(collected)
    firstcol = collected[1]
    Tcol = eltype(firstcol)
    empty_mat = zeros(Tcol, length(firstcol), 0)
    states_raw = reduce(hcat, collected; init = empty_mat)
    states = eltype(data).(states_raw)
    return states, newst
end

function collectstates(rc::AbstractLuxLayer, data::AbstractVector, ps, st::NamedTuple)
    return collectstates(rc, reshape(data, :, 1), ps, st)
end

@doc raw"""
    DelayLayer(input_dim; num_delays=2, stride=1)

Stateful delay layer that augments the current vector with a fixed number of
time-delayed copies of itself. Intended to be used as a `state_modifier` in a
`ReservoirComputer`, for example to build NVAR-style feature vectors.

At each call, the layer:

1. Takes the current input vector `h(t)` of length `input_dim`.
2. Produces an output vector that concatenates:
- the current input `h(t)`, and
- `num_delays` previous inputs stored in an internal buffer.
3. Updates its internal delay buffer with `h(t)` every `stride` calls.

Newly initialized buffers are filled with zeros, so at the beginning of a
sequence, delayed entries correspond to zero padding.

## Arguments

- `input_dim`: Dimension of the input/state vector at each time step.

## Keyword arguments

  - `num_delays`: Number of delayed copies to keep. The output
    will have `(num_delays + 1) * input_dim` entries: the current vector plus
    `num_delays` past vectors. Default is 2.
  - `stride`: Delay stride in layer calls. The internal buffer
    is updated only when `clock % stride == 0`. Default is 1.
  - `init_delay`: Initializer(s) for the delays. Must be either a
    single function (e.g. `zeros32`, `randn32`) or an `NTuple` of
    `num_delays` functions, e.g. `(zeros32, randn32)`. If a single function
    `fn` is provided, it is automatically expanded into a
    `num_delays`-element tuple `(fn, fn, ..., fn)`.
    Default is `zeros32`.

## Inputs

- `h(t) :: AbstractVector (input_dim,)`
Typically the current reservoir state at time `t`.

## Returns

- `z :: AbstractVector ((num_delays + 1) * input_dim,)`
Concatenation of the current input and its delayed copies.

## Parameters

None

## States

  - `history`: a matrix whose column j holds the j-th most recent stored input.
    On a stride update, the columns are shifted and the current input is placed
    in column 1.
  - `clock`: A counter that updates each call of the layer. The delay buffer is
    updated when `clock % stride == 0`.
"""
@concrete struct DelayLayer <: AbstractLuxLayer
    in_dims <: Int
    num_delays <: Int
    stride <: Int
    init_delay::Any
end

function DelayLayer(in_dims; num_delays::Int = 2, stride::Int = 1, init_delay = zeros32)
    if init_delay isa Tuple
        @assert length(init_delay) == num_delays
    else
        init_delay = ntuple(_ -> init_delay, num_delays)
    end

    return DelayLayer(in_dims, num_delays, stride, init_delay)
end

function initialparameters(rng::AbstractRNG, dl::DelayLayer)
    return NamedTuple()
end

function initialstates(rng::AbstractRNG, dl::DelayLayer)
    history = nothing
    clock = 0

    return (history = history, clock = clock, rng = rng)
end

function init_delay_history(::Nothing, rng::AbstractRNG, dl::DelayLayer, inp::AbstractVecOrMat)
    history = similar(inp, dl.in_dims, dl.num_delays)
    for (init, delay_col) in zip(dl.init_delay, eachcol(history))
        delay_col .= init(rng, dl.in_dims, 1)
    end

    return history
end

function init_delay_history(history::AbstractMatrix, rng::AbstractRNG, dl::DelayLayer, inp::AbstractVecOrMat)
    return history
end

function (dl::DelayLayer)(inp::AbstractVecOrMat, ps, st::NamedTuple)
    @assert size(inp, 1) == dl.in_dims
    history = init_delay_history(st.history, st.rng, dl, inp)
    inp_with_delay = vcat(inp, vec(history))
    clock = st.clock + 1
    if dl.num_delays > 0 && (clock % dl.stride == 0)
        @views history[:, 2:end] .= history[:, 1:(end - 1)]
        @views history[:, 1] .= inp
    end

    return inp_with_delay, (history = history, clock = clock, rng = st.rng)
end
