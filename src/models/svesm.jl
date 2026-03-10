@doc raw"""
    SVESM(in_dims, res_dims, out_dims, activation=tanh;
        leak_coefficient=1.0, init_reservoir=rand_sparse, init_input=scaled_rand,
        init_bias=zeros32, init_state=randn32, use_bias=false,
        state_modifiers=())

Support Vector Echo-State Machine [ShiHan2007](@cite).

`SVESM` replaces the linear readout of an [`ESN`](@ref) with a
[`SVMReadout`](@ref), performing support vector regression in the
high-dimensional reservoir state space (the "reservoir trick").
Training requires LIBSVM.jl to be loaded and a `LIBSVM.AbstractSVR`
instance to be passed as the `train_method` argument to [`train!`](@ref).

## Equations

```math
\begin{aligned}
    \mathbf{x}(t) &= (1-\alpha)\, \mathbf{x}(t-1) + \alpha\, \phi\!\left(
        \mathbf{W}_{\text{in}}\, \mathbf{u}(t) + \mathbf{W}_r\, \mathbf{x}(t-1)
        + \mathbf{b} \right) \\
    \mathbf{z}(t) &= \mathrm{Mods}\!\left(\mathbf{x}(t)\right) \\
    \mathbf{y}(t) &= \mathrm{SVR}\!\left(\mathbf{z}(t)\right)
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
  - `init_bias`: Initializer for reservoir bias (used if `use_bias=true`).
    Default: `zeros32`.
  - `init_state`: Initializer used when an external state is not provided.
    Default: `randn32`.
  - `use_bias`: Whether the reservoir uses a bias term. Default: `false`.

Composition:

  - `state_modifiers`: A layer or collection of layers applied to the reservoir
    state before the readout. Accepts a single layer, an `AbstractVector`, or a
    `Tuple`. Default: empty `()`.
"""
@concrete struct SVESM <:
    AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

function SVESM(
        in_dims::IntegerType, res_dims::IntegerType,
        out_dims::IntegerType, activation = tanh;
        state_modifiers = (),
        kwargs...
    )
    cell = StatefulLayer(ESNCell(in_dims => res_dims, activation; kwargs...))
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
        Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = SVMReadout(res_dims => out_dims)
    return SVESM(cell, mods, ro)
end

function Base.show(io::IO, svesm::SVESM)
    print(io, "SVESM(\n")

    print(io, "    reservoir = ")
    show(io, svesm.reservoir)
    print(io, ",\n")

    print(io, "    state_modifiers = ")
    if isempty(svesm.states_modifiers)
        print(io, "()")
    else
        print(io, "(")
        for (i, m) in enumerate(svesm.states_modifiers)
            i > 1 && print(io, ", ")
            show(io, m)
        end
        print(io, ")")
    end
    print(io, ",\n")

    print(io, "    readout = ")
    show(io, svesm.readout)
    print(io, "\n)")

    return
end
