@doc raw"""
    ESN(in_dims, res_dims, out_dims, activation=tanh;
        leak_coefficient=1.0, init_reservoir=rand_sparse, init_input=scaled_rand,
        init_bias=zeros32, init_state=randn32, use_bias=false,
        state_modifiers=(), readout_activation=identity)

Echo State Network (ESN): a reservoir (recurrent) layer followed by an optional
sequence of state-modifier layers and a linear readout.

`ESN` composes:
  1) a stateful [`ESNCell`](@ref) (reservoir),
  2) zero or more `state_modifiers` applied to the reservoir state, and
  3) a [`LinearReadout`](@ref) mapping reservoir features to outputs.

## Equations

For input `\mathbf{x}(t) ∈ \mathbb{R}^{in\_dims}`, reservoir state
`\mathbf{h}(t) ∈ \mathbb{R}^{res\_dims}`, and output
`\mathbf{y}(t) ∈ \mathbb{R}^{out\_dims}`:

```math
\begin{aligned}
    \tilde{\mathbf{h}}(t) &= \phi\!\left(\mathbf{W}_{in}\,\mathbf{x}(t) +
        \mathbf{W}_{res}\,\mathbf{h}(t-1) + \mathbf{b}\right) \\
    \mathbf{h}(t) &= (1-\alpha)\,\mathbf{h}(t-1) + \alpha\,\tilde{\mathbf{h}}(t) \\
    \mathbf{z}(t) &= \psi\!\left(\mathrm{Mods}\big(\mathbf{h}(t)\big)\right) \\
    \mathbf{y}(t) &= \rho\!\left(\mathbf{W}_{out}\,\mathbf{z}(t) + \mathbf{b}_{out}\right)
\end{aligned}
```

## Arguments

  - `in_dims`: Input dimension.
  - `res_dims`: Reservoir (hidden state) dimension.
  - `out_dims`: Output dimension.
  - `activation`: Reservoir activation (for [`ESNCell`](@ref)). Default: `tanh`.

## Keyword arguments

Reservoir (passed to [`ESNCell`](@ref)):

  - `leak_coefficient`: Leak rate `α ∈ (0,1]`. Default: `1.0`.
  - `init_reservoir`: Initializer for `W_res`. Default: [`rand_sparse`](@ref).
  - `init_input`: Initializer for `W_in`. Default: [`scaled_rand`](@ref).
  - `init_bias`: Initializer for reservoir bias (used iff `use_bias=true`).
    Default: [`zeros32`](@extref).
  - `init_state`: Initializer used when an external state is not provided.
    Default: [`randn32`](@extref).
  - `use_bias`: Whether the reservoir uses a bias term. Default: `false`.

Composition:

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

  - `cell` — parameters of the internal [`ESNCell`](@ref), including:
      - `input_matrix :: (res_dims × in_dims)` — `W_in`
      - `reservoir_matrix :: (res_dims × res_dims)` — `W_res`
      - `bias :: (res_dims,)` — present only if `use_bias=true`
  - `states_modifiers` — a `Tuple` with parameters for each modifier layer (may be empty).
  - `readout` — parameters of [`LinearReadout`](@ref), typically:
      - `weight :: (out_dims × res_dims)` — `W_out`
      - `bias :: (out_dims,)` — `b_out` (if the readout uses bias)

> Exact field names for modifiers/readout follow their respective layer
> definitions.

## States

Created by `initialstates(rng, esn)`:

  - `cell` — states for the internal [`ESNCell`](@ref) (e.g. `rng` used to sample initial hidden states).
  - `states_modifiers` — a `Tuple` with states for each modifier layer.
  - `readout` — states for [`LinearReadout`](@ref).

"""
@concrete struct ESN <: AbstractEchoStateNetwork{(:cell, :states_modifiers, :readout)}
    cell
    states_modifiers
    readout
end

function ESN(in_dims::IntegerType, res_dims::IntegerType,
    out_dims::IntegerType, activation=tanh;
    readout_activation=identity,
    state_modifiers=(),
    kwargs...)
    cell = StatefulLayer(ESNCell(in_dims => res_dims, activation; kwargs...))
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
                 Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return ESN(cell, mods, ro)
end

function initialparameters(rng::AbstractRNG, esn::ESN)
    ps_cell = initialparameters(rng, esn.cell)
    ps_mods = map(l -> initialparameters(rng, l), esn.states_modifiers) |> Tuple
    ps_ro = initialparameters(rng, esn.readout)
    return (cell=ps_cell, states_modifiers=ps_mods, readout=ps_ro)
end

function initialstates(rng::AbstractRNG, esn::ESN)
    st_cell = initialstates(rng, esn.cell)
    st_mods = map(l -> initialstates(rng, l), esn.states_modifiers) |> Tuple
    st_ro = initialstates(rng, esn.readout)
    return (cell=st_cell, states_modifiers=st_mods, readout=st_ro)
end

function _partial_apply(esn::ESN, inp, ps, st)
    out, st_cell = apply(esn.cell, inp, ps.cell, st.cell)
    out, st_mods = _apply_seq(
        esn.states_modifiers, out, ps.states_modifiers, st.states_modifiers)
    return out, (cell=st_cell, states_modifiers=st_mods)
end

function (esn::ESN)(inp, ps, st)
    out, new_st = _partial_apply(esn, inp, ps, st)
    out, st_ro = apply(esn.readout, out, ps.readout, st.readout)
    return out, merge(new_st, (readout=st_ro,))
end
