@doc raw"""
        AdditiveEIESN(in_dims, res_dims, out_dims, activation=tanh_fast;
                exc_recurrence_scale=0.9, inh_recurrence_scale=0.5, exc_output_scale=1.0, inh_output_scale=1.0,
                input_scale=1.0,
                init_reservoir=rand_sparse, init_input=scaled_rand,
                init_state=randn32,
                readout_activation=identity,
                state_modifiers=(),
                kwargs...)

Excitatory-Inhibitory Echo State Network (EIESN) with Additive Input [Panahi2025](@cite).

## Equations

```math
\begin{aligned}
        \mathbf{x}(t) &= b_{\mathrm{ex}} \, \phi\!\left(a_{\mathrm{ex}} \mathbf{A} \mathbf{x}(t-1)\right)
        - b_{\mathrm{inh}} \, \phi\!\left(a_{\mathrm{inh}} \mathbf{A} \mathbf{x}(t-1)\right)
        + g \, \mathbf{W}_{\mathrm{in}} \mathbf{u}(t) \\
        \mathbf{z}(t) &= \mathrm{Mods}\!\left(\mathbf{x}(t)\right) \\
        \mathbf{y}(t) &= \rho\!\left( \mathbf{W}_{\text{out}}\, \mathbf{z}(t) + \mathbf{b}_{\text{out}} \right)
\end{aligned}
```

## Arguments

    - `in_dims`: Input dimension.
    - `res_dims`: Reservoir (hidden state) dimension.
    - `out_dims`: Output dimension.
    - `activation`: Reservoir activation (for [`AdditiveEIESNCell`](@ref)). Default: `tanh_fast`.

## Keyword arguments

    - `exc_recurrence_scale`: Excitatory recurrence scaling factor. Default: `0.9`.
    - `inh_recurrence_scale`: Inhibitory recurrence scaling factor. Default: `0.5`.
    - `exc_output_scale`: Excitatory output scaling factor. Default: `1.0`.
    - `inh_output_scale`: Inhibitory output scaling factor. Default: `1.0`.
    - `input_scale`: Input scaling factor ($g$). Default: `1.0`.
    - `init_reservoir`: Initializer for the reservoir matrix. Default: [`rand_sparse`](@ref).
    - `init_input`: Initializer for the input matrix. Default: [`scaled_rand`](@ref).
    - `init_state`: Initializer used when an external state is not provided. Default: `randn32`.
    - `readout_activation`: Activation for the linear readout. Default: `identity`.
    - `state_modifiers`: A layer or collection of layers applied to the reservoir state before the readout. Accepts a single layer, an `AbstractVector`, or a `Tuple`. Default: empty `()`.

## Inputs

    - `x :: AbstractArray (in_dims, batch)`

## Returns

    - Output `y :: (out_dims, batch)`.
    - Updated layer state (NamedTuple).

## Parameters

    - `reservoir` — parameters of the internal [`AdditiveEIESNCell`](@ref).
    - `states_modifiers` — a `Tuple` with parameters for each modifier layer (may be empty).
    - `readout` — parameters of [`LinearReadout`](@ref).

## States

    - `reservoir` — states for the internal [`AdditiveEIESNCell`](@ref) (e.g. `rng`).
    - `states_modifiers` — a `Tuple` with states for each modifier layer.
    - `readout` — states for [`LinearReadout`](@ref).
"""

@concrete struct AdditiveEIESN <:
    AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

function AdditiveEIESN(
        in_dims::IntegerType, res_dims::IntegerType,
        out_dims::IntegerType, activation = tanh_fast;
        readout_activation = identity,
        state_modifiers = (),
        kwargs...
    )

    cell = StatefulLayer(AdditiveEIESNCell(in_dims => res_dims, activation; kwargs...))
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
        Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return AdditiveEIESN(cell, mods, ro)
end

function Base.show(io::IO, esn::AdditiveEIESN)
    print(io, "AdditiveEIESN(\n")

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
