
@doc raw"""
    DelayESN(in_dims, res_dims, out_dims, activation=tanh;
             num_delays=1, stride=1, leak_coefficient=1.0,
             init_reservoir=rand_sparse, init_input=scaled_rand,
             init_bias=zeros32, init_state=randn32, use_bias=false,
             state_modifiers=(), readout_activation=identity)

Echo State Network with state delays [Fleddermann2025](@cite).

`DelayESN` composes:
  1) a stateful [`ESNCell`](@ref) (reservoir),
  2) a [`DelayLayer`](@ref) applied to the reservoir state to build
     tapped-delay features,
  3) zero or more additional `state_modifiers` applied after the delay, and
  4) a [`LinearReadout`](@ref) mapping delayed reservoir features to outputs.

At each time step, the reservoir produces a state vector `h(t)` of length
`res_dims`. The `DelayLayer` then constructs a feature vector that stacks
`h(t)` together with `num_delays` past states, spaced according to `stride`,
before passing it on to any further modifiers and the readout.

## Arguments

  - `in_dims`: Input dimension.
  - `res_dims`: Reservoir (hidden state) dimension.
  - `out_dims`: Output dimension.
  - `activation`: Reservoir activation (for [`ESNCell`](@ref)). Default: `tanh`.

## Keyword arguments

Reservoir (passed to [`ESNCell`](@ref)):

  - `leak_coefficient`: Leak rate in `(0, 1]`. Default: `1.0`.
  - `init_reservoir`: Initializer for `W_res`. Default: [`rand_sparse`](@ref).
  - `init_input`: Initializer for `W_in`. Default: [`scaled_rand`](@ref).
  - `init_bias`: Initializer for reservoir bias (used iff `use_bias=true`).
    Default: `zeros32`.
  - `init_state`: Initializer used when an external state is not provided.
    Default: `randn32`.
  - `use_bias`: Whether the reservoir uses a bias term. Default: `false`.

Delay expansion:

  - `num_delays`: Number of past reservoir states to include in the tapped-delay
    vector. The `DelayLayer` output has `(num_delays + 1) * res_dims` entries
    (current state plus `num_delays` past states). Default: `1`.
  - `stride`: Delay stride in layer calls. The delay buffer is updated only when
    the internal clock is a multiple of `stride`. Default: `1`.

Composition:

  - `state_modifiers`: A layer or collection of layers applied to the delayed
    reservoir features before the readout. These run **after** the internal
    `DelayLayer`. Accepts a single layer, an `AbstractVector`, or a `Tuple`.
    Default: empty `()`.
  - `readout_activation`: Activation for the linear readout. Default: `identity`.

## Inputs

  - `x :: AbstractArray (in_dims, batch)`

## Returns

  - Output `y :: (out_dims, batch)`.
  - Updated layer state (NamedTuple).

## Parameters

  - `reservoir` — parameters of the internal [`ESNCell`](@ref), including:
      - `input_matrix :: (res_dims × in_dims)` — `W_in`
      - `reservoir_matrix :: (res_dims × res_dims)` — `W_res`
      - `bias :: (res_dims,)` — present only if `use_bias=true`
  - `states_modifiers` — a `Tuple` with parameters for:
      1. the internal [`DelayLayer`](@ref), and
      2. any user-provided modifier layers (may be empty).
  - `readout` — parameters of [`LinearReadout`](@ref), typically:
      - `weight :: (out_dims × ((num_delays + 1) * res_dims))` — `W_out`
      - `bias :: (out_dims,)` — `b_out` (if the readout uses bias)

> Exact field names for modifiers/readout follow their respective layer
> definitions.

## States

  - `reservoir` — states for the internal [`ESNCell`](@ref) (e.g. `rng` used to
    sample initial hidden states).
  - `states_modifiers` — a `Tuple` with states for the internal `DelayLayer`
    (its delay buffer and clock) and each additional modifier layer.
  - `readout` — states for [`LinearReadout`](@ref) (typically empty).
"""
@concrete struct DelayESN <:
                 AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

function DelayESN(
        in_dims::IntegerType, res_dims::Int, out_dims::IntegerType, activation = tanh;
        num_delays::Int = 2, stride::Int = 1, readout_activation = identity,
        state_modifiers = (), kwargs...)
    cell = StatefulLayer(ESNCell(in_dims => res_dims, activation; kwargs...))
    delay = DelayLayer(res_dims; num_delays = num_delays, stride = stride)
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
                 (delay, state_modifiers...) : (delay, state_modifiers)
    mods = _wrap_layers(mods_tuple)
    ro_in_dims = res_dims * (num_delays + 1)
    ro = LinearReadout(ro_in_dims => out_dims, readout_activation)

    return DelayESN(cell, mods, ro)
end

function Base.show(io::IO, esn::DelayESN)
    print(io, "DelayESN(\n")

    print(io, "    reservoir = ")
    show(io, esn.reservoir)
    print(io, ",\n")

    print(io, "    state_modifiers = ")
    if isempty(esn.states_modifiers)
        print(io, "()")
    else
        print(io, "(")
        for (i, m) in enumerate(esn.states_modifiers)
            i > 1 && print(io, ", ")
            show(io, m)
        end
        print(io, ")")
    end
    print(io, ",\n")

    print(io, "    readout = ")
    show(io, esn.readout)
    print(io, "\n)")

    return
end
