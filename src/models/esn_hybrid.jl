
@doc raw"""
    HybridESN(km, km_dims, in_dims, res_dims, out_dims, [activation];
        state_modifiers=(), readout_activation=identity,
        include_collect=true, kwargs...)

Hybrid Echo State Network (HybridESN): an Echo State Network augmented with a
knowledge model whose outputs are concatenated to the ESN’s input and used
throughout the reservoir and readout computations.

`HybridESN` composes:
  1) a knowledge model `km` producing auxiliary features from the input,
  2) a stateful [`ESNCell`](@ref) that receives the concatenated input
     `[km(x(t)); x(t)]`,
  3) zero or more `state_modifiers` applied to the reservoir state, and
  4) a [`LinearReadout`](@ref) mapping the combined features `[km(x(t)); h*(t)]`
     to the output.

## Arguments

- `km`: Knowledge model applied to the input (e.g. a physical model, neural
    submodule, or differentiable function). May be a `WrappedFunction` or any
    callable layer.
- `km_dims`: Output dimension of the knowledge model `km`.
- `in_dims`: Input dimension.
- `res_dims`: Reservoir (hidden state) dimension.
- `out_dims`: Output dimension.
- `activation`: Reservoir activation (for [`ESNCell`](@ref)). Default: `tanh`.

## Keyword arguments

- `leak_coefficient`: Leak rate `α ∈ (0,1]`. Default: `1.0`.
- `init_reservoir`: Initializer for `W_res`. Default: [`rand_sparse`](@ref).
- `init_input`: Initializer for `W_in`. Default: [`scaled_rand`](@ref).
- `init_bias`: Initializer for reservoir bias (used if `use_bias=true`).
    Default: [`zeros32`](@extref).
- `init_state`: Initializer used when an external state is not provided.
    Default: [`randn32`](@extref).
- `use_bias`: Whether the reservoir uses a bias term. Default: `false`.
- `state_modifiers`: A layer or collection of layers applied to the reservoir
    state before the readout. Accepts a single layer, an `AbstractVector`, or a
    `Tuple`. Default: empty `()`.
- `readout_activation`: Activation for the linear readout. Default: `identity`.
- `include_collect`: Whether the readout should include collection mode.
    Default: `true`.

## Inputs

- `x :: AbstractArray (in_dims, batch)`

## Returns

- Output `y :: (out_dims, batch)`.
- Updated layer state (NamedTuple).

## Parameters

- `knowledge_model` — parameters of the knowledge model `km`.
- `cell` — parameters of the internal [`ESNCell`](@ref), including:
    - `input_matrix :: (res_dims × (in_dims + km_dims))` — `W_in`
    - `reservoir_matrix :: (res_dims × res_dims)` — `W_res`
    - `bias :: (res_dims,)` — present only if `use_bias=true`
- `states_modifiers` — a `Tuple` with parameters for each modifier layer (may be empty).
- `readout` — parameters of [`LinearReadout`](@ref), typically:
    - `weight :: (out_dims × (res_dims + km_dims))` — `W_out`
    - `bias :: (out_dims,)` — `b_out` (if the readout uses bias)

> Exact field names for modifiers/readout follow their respective layer
> definitions.

## States

Created by `initialstates(rng, hesn)`:

- `knowledge_model` — states for the internal knowledge model.
- `cell` — states for the internal [`ESNCell`](@ref).
- `states_modifiers` — a `Tuple` with states for each modifier layer.
- `readout` — states for [`LinearReadout`](@ref).
"""
@concrete struct HybridESN <: AbstractEchoStateNetwork{(
    :cell, :states_modifiers, :readout, :knowledge_model)}
    cell::Any
    knowledge_model::Any
    states_modifiers::Any
    readout::Any
end

function HybridESN(km,
        km_dims::IntegerType, in_dims::IntegerType,
        res_dims::IntegerType, out_dims::IntegerType,
        activation = tanh;
        state_modifiers = (),
        readout_activation = identity,
        include_collect::BoolType = True(),
        kwargs...)
    esn_inp_size = in_dims + km_dims
    cell = StatefulLayer(ESNCell(esn_inp_size => res_dims, activation; kwargs...))
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
                 Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims + km_dims => out_dims, readout_activation;
        include_collect = static(include_collect))
    km_layer = km isa WrappedFunction ? km : WrappedFunction(km)
    return HybridESN(cell, km_layer, mods, ro)
end

function initialparameters(rng::AbstractRNG, hesn::HybridESN)
    ps_cell = initialparameters(rng, hesn.cell)
    ps_km = initialparameters(rng, hesn.knowledge_model)
    ps_mods = map(l -> initialparameters(rng, l), hesn.states_modifiers) |> Tuple
    ps_ro = initialparameters(rng, hesn.readout)
    return (cell = ps_cell, knowledge_model = ps_km,
        states_modifiers = ps_mods, readout = ps_ro)
end

function initialstates(rng::AbstractRNG, hesn::HybridESN)
    st_cell = initialstates(rng, hesn.cell)
    st_km = initialstates(rng, hesn.knowledge_model)
    st_mods = map(l -> initialstates(rng, l), hesn.states_modifiers) |> Tuple
    st_ro = initialstates(rng, hesn.readout)
    return (cell = st_cell, knowledge_model = st_km,
        states_modifiers = st_mods, readout = st_ro)
end

function _partial_apply(hesn::HybridESN, inp, ps, st)
    k_t, st_km = hesn.knowledge_model(inp, ps.knowledge_model, st.knowledge_model)
    xin = vcat(k_t, inp)
    r, st_cell = apply(hesn.cell, xin, ps.cell, st.cell)
    rstar,
    st_mods = _apply_seq(hesn.states_modifiers, r, ps.states_modifiers, st.states_modifiers)
    feats = vcat(k_t, rstar)
    return feats, (cell = st_cell, states_modifiers = st_mods, knowledge_model = st_km)
end

function (hesn::HybridESN)(inp, ps, st)
    feats, new_st = _partial_apply(hesn, inp, ps, st)
    y, st_ro = apply(hesn.readout, feats, ps.readout, st.readout)
    return y, merge(new_st, (readout = st_ro,))
end
