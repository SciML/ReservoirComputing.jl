@doc raw"""
    ES2N(in_dims, res_dims, out_dims, activation=tanh;
        proximity=1.0, init_reservoir=rand_sparse, init_input=scaled_rand,
        init_bias=zeros32, init_state=randn32, use_bias=False(),
        state_modifiers=(), readout_activation=identity,
        init_orthogonal=orthogonal,)

Edge of Stability Echo State Network (ES2N) [Ceni2025](@cite).


## Equations

```math
\begin{aligned}
x(t) = \beta\, \phi\!\left( \rho\, \mathbf{W}_r x(t-1) + \omega\,
    \mathbf{W}_{in} u(t) \right) + (1-\beta)\, \mathbf{O}\, x(t-1),
\end{aligned}
```

## Arguments

  - `in_dims`: Input dimension.
  - `res_dims`: Reservoir (hidden state) dimension.
  - `out_dims`: Output dimension.
  - `activation`: Reservoir activation (for [`ESNCell`](@ref)). Default: `tanh`.

## Keyword arguments

  - `proximity`: proximity `α ∈ (0,1]`. Default: `1.0`.
  - `init_reservoir`: Initializer for `W_res`. Default: [`rand_sparse`](@ref).
  - `init_input`: Initializer for `W_in`. Default: [`scaled_rand`](@ref).
  - `init_orthogonal`: Initializer for `O`. Default: [`orthogonal`].
  - `init_bias`: Initializer for reservoir bias (used if `use_bias=true`).
    Default: `zeros32`.
  - `init_state`: Initializer used when an external state is not provided.
    Default: `randn32`.
  - `use_bias`: Whether the reservoir uses a bias term. Default: `false`.
  - `state_modifiers`: A layer or collection of layers applied to the reservoir
    state before the readout. Accepts a single layer, an `AbstractVector`, or a
    `Tuple`. Default: empty `()`.
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
      - `orthogonal_matrix :: (res_dims × res_dims)` — `O`
      - `bias :: (res_dims,)` — present only if `use_bias=true`
  - `states_modifiers` — a `Tuple` with parameters for each modifier layer (may be empty).
  - `readout` — parameters of [`LinearReadout`](@ref), typically:
      - `weight :: (out_dims × res_dims)` — `W_out`
      - `bias :: (out_dims,)` — `b_out` (if the readout uses bias)

> Exact field names for modifiers/readout follow their respective layer
> definitions.

## States

  - `reservoir` — states for the internal [`ES2NCell`](@ref) (e.g. `rng` used to sample initial hidden states).
  - `states_modifiers` — a `Tuple` with states for each modifier layer.
  - `readout` — states for [`LinearReadout`](@ref).

"""
@concrete struct ES2N <:
                 AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

function ES2N(in_dims::IntegerType, res_dims::IntegerType,
        out_dims::IntegerType, activation = tanh;
        readout_activation = identity,
        state_modifiers = (),
        kwargs...)
    cell = StatefulLayer(ES2NCell(in_dims => res_dims, activation; kwargs...))
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
                 Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return ES2N(cell, mods, ro)
end

function Base.show(io::IO, esn::ES2N)
    print(io, "ES2N(\n")

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
