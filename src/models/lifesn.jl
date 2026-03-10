@doc raw"""
    LIFESNCell(in_dims => out_dims, args...; lookback_horizon=2, kwargs...)

Convenience constructor for a [`LocalInformationFlow`](@ref)-wrapped [`ESNCell`](@ref).

Equivalent to `LocalInformationFlow(ESNCell, in_dims => out_dims, lookback_horizon, args...; kwargs...)`.

See [`LocalInformationFlow`](@ref) and [`ESNCell`](@ref) for full argument descriptions.
"""
function LIFESNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        args...; lookback_horizon::IntegerType = 2, kwargs...
    )
    return LocalInformationFlow(
        ESNCell, (in_dims => out_dims), lookback_horizon, args...; kwargs...
    )
end

@doc raw"""
    LIFESN(in_dims, res_dims, out_dims, activation=tanh;
        lookback_horizon=2, readout_activation=identity,
        state_modifiers=(), kwargs...)

Local Information Flow Echo State Network [Liu2025](@cite).

`LIFESN` composes:
  1) a stateful [`LIFESNCell`](@ref) (a [`LocalInformationFlow`](@ref)-wrapped
     [`ESNCell`](@ref)),
  2) zero or more `state_modifiers` applied to the reservoir state, and
  3) a [`LinearReadout`](@ref) mapping reservoir features to outputs.

At each time step, the reservoir state is reconstructed from scratch using only
the most recent `lookback_horizon` inputs, restricting the effective memory of
the network to a local window.

## Equations

```math
\begin{aligned}
    \mathbf{x}(t) &= f\!\left(
        \mathbf{W}_{\text{in}}\, \mathbf{u}(t)
        + \mathbf{W}\, \mathbf{z}(t-1) \right) \\
    \mathbf{z}(t-1) &= f\!\left(
        \mathbf{W}_{\text{in}}\, \mathbf{u}(t-1)
        + \mathbf{W}\, \mathbf{z}(t-2) \right) \\
    &\ \vdots \\
    \mathbf{z}(t-k+1) &= f\!\left(
        \mathbf{W}_{\text{in}}\, \mathbf{u}(t-k+1)
        + \mathbf{W}\, \mathbf{z}_0 \right) \\
    \hat{\mathbf{x}}(t) &= \mathrm{Mods}\!\left(\mathbf{x}(t)\right) \\
    \mathbf{y}(t) &= \rho\!\left(
        \mathbf{W}_{\text{out}}\, \hat{\mathbf{x}}(t)
        + \mathbf{b}_{\text{out}} \right)
\end{aligned}
```

## Arguments

  - `in_dims`: Input dimension.
  - `res_dims`: Reservoir (hidden state) dimension.
  - `out_dims`: Output dimension.
  - `activation`: Reservoir activation (for [`ESNCell`](@ref)). Default: `tanh`.

## Keyword arguments

Reservoir (passed to [`ESNCell`](@ref) via [`LIFESNCell`](@ref)):

  - `lookback_horizon`: Number of most recent inputs used to reconstruct the
    recurrent state. Default: `2`.
  - `leak_coefficient`: Leak rate `α ∈ (0,1]`. Default: `1.0`.
  - `init_reservoir`: Initializer for `W_res`. Default: [`rand_sparse`](@ref).
  - `init_input`: Initializer for `W_in`. Default: [`scaled_rand`](@ref).
  - `init_bias`: Initializer for reservoir bias (used if `use_bias=true`).
    Default: `zeros32`.
  - `init_state`: Initializer used when an external state is not provided.
    Default: `randn32`.
  - `use_bias`: Whether the reservoir uses a bias term. Default: `false`.

Composition:

  - `state_modifiers`: A layer or collection of layers applied to the reservoir
    state before the readout. Accepts a single layer, an `AbstractVector`, or a
    `Tuple`. Default: empty `()`.
  - `readout_activation`: Activation for the linear readout. Default: `identity`.
"""
@concrete struct LIFESN <:
    AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

function LIFESN(
        in_dims::IntegerType, res_dims::IntegerType,
        out_dims::IntegerType, activation = tanh;
        lookback_horizon::IntegerType = 2,
        readout_activation = identity,
        state_modifiers = (),
        kwargs...
    )
    cell = StatefulLayer(
        LIFESNCell(in_dims => res_dims, activation; lookback_horizon = lookback_horizon, kwargs...)
    )
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
        Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return LIFESN(cell, mods, ro)
end

function Base.show(io::IO, lifesn::LIFESN)
    print(io, "LIFESN(\n")

    print(io, "    reservoir = ")
    show(io, lifesn.reservoir)
    print(io, ",\n")

    print(io, "    state_modifiers = ")
    if isempty(lifesn.states_modifiers)
        print(io, "()")
    else
        print(io, "(")
        for (i, m) in enumerate(lifesn.states_modifiers)
            i > 1 && print(io, ", ")
            show(io, m)
        end
        print(io, ")")
    end
    print(io, ",\n")

    print(io, "    readout = ")
    show(io, lifesn.readout)
    print(io, "\n)")

    return
end
