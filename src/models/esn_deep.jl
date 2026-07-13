@doc raw"""
    DeepESN(cells::Tuple, readout; states_modifiers=nothing)

Deep Echo State Network wrapper [Gallicchio2017](@cite).

`DeepESN` acts as a generic wrapper that composes, for `L = length(cells)` layers:
  1) a sequence of arbitrary Lux layers (typically stateful `ESNCell`s or custom dynamical systems),
  2) zero or more per-layer `states_modifiers[ℓ]` applied to the layer's state, and
  3) a final `readout` layer from the last layer's features to the output.

## Equations

```math
\begin{aligned}
    \mathbf{x}^{(1)}(t) &= (1-\alpha_1)\, \mathbf{x}^{(1)}(t-1)
        + \alpha_1\, \phi_1\!\left(\mathbf{W}^{(1)}_{\text{in}}\, \mathbf{u}(t)
        + \mathbf{W}^{(1)}_r\, \mathbf{x}^{(1)}(t-1) + \mathbf{b}^{(1)} \right), \\
    \mathbf{u}^{(1)}(t) &= \mathrm{Mods}_1\!\left(\mathbf{x}^{(1)}(t)\right), \\
    \mathbf{x}^{(\ell)}(t) &= (1-\alpha_\ell)\, \mathbf{x}^{(\ell)}(t-1)
        + \alpha_\ell\, \phi_\ell\!\left(\mathbf{W}^{(\ell)}_{\text{in}}\,
        \mathbf{u}^{(\ell-1)}(t) + \mathbf{W}^{(\ell)}_r\, \mathbf{x}^{(\ell)}(t-1)
        + \mathbf{b}^{(\ell)} \right), \quad \ell = 2,\dots,L, \\
    \mathbf{u}^{(\ell)}(t) &= \mathrm{Mods}_\ell\!\left(\mathbf{x}^{(\ell)}(t)\right),
        \quad \ell = 2,\dots,L, \\
    \mathbf{y}(t) &= \rho\!\left(\mathbf{W}_{\text{out}}\, \mathbf{u}^{(L)}(t)
        + \mathbf{b}_{\text{out}} \right).
\end{aligned}
```

## Arguments

  - `cells`: A `Tuple` of pre-instantiated layers. If a layer is not already a `StatefulLayer`, it will be wrapped automatically to maintain continuous reservoir memory.
  - `readout`: The final readout layer (e.g., `LinearReadout`) applied to the output of the last cell.

## Keyword arguments

  - `states_modifiers`: Per-layer modifier(s) applied to each layer’s state before it
    feeds into the next layer (and the readout for the last layer). Accepts `nothing`,
    a single layer, a vector/tuple of length `L`, or per-layer collections. Defaults to no modifiers.

Composition:

  - `state_modifiers`: Per-layer modifier(s) applied to each layer’s state before it
    feeds into the next layer (and the readout for the last layer). Accepts `nothing`,
    a single layer, a vector/tuple of length `L`, or per-layer collections. Defaults to no modifiers.
  - `readout_activation`: Activation for the final linear readout. Default: `identity`.

## Inputs

  - `x :: AbstractArray (in_dims, batch)`

## Returns

  - Output `y :: (out_dims, batch)`.
  - Updated layer state (NamedTuple) containing states for all cells, modifiers, and readout.

## Parameters

  - `cells :: NTuple{L,NamedTuple}` — parameters for each cell in the sequence.
  - `states_modifiers :: NTuple{L,Tuple}` — per-layer tuples of modifier parameters (empty tuples if none).
  - `readout` — parameters for the readout layer.

> Exact field names for cells, modifiers, and readout follow their respective layer definitions.

## States

  - `cells :: NTuple{L,NamedTuple}` — states for each cell in the sequence.
  - `states_modifiers :: NTuple{L,Tuple}` — per-layer tuples of modifier states.
  - `readout` — states for the readout layer.
"""
@concrete struct DeepESN <: AbstractEchoStateNetwork{(:cells, :states_modifiers, :readout)}
    cells
    states_modifiers
    readout
end

function DeepESN(
        cells::Tuple,
        readout;
        states_modifiers = nothing
    )
    n_layers = length(cells)

    # Ensure cells are wrapped in StatefulLayer to maintain continuous memory
    stateful_cells = map(c -> c isa StatefulLayer ? c : StatefulLayer(c), cells)

    # Handle state modifiers
    mods = states_modifiers === nothing ? ntuple(_ -> nothing, n_layers) : states_modifiers
    mods_per_layer = map(_coerce_layer_mods, mods) |> Tuple

    return DeepESN(stateful_cells, mods_per_layer, readout)
end

function initialparameters(rng::AbstractRNG, desn::DeepESN)
    ps_cells = map(l -> initialparameters(rng, l), desn.cells) |> Tuple
    mods = desn.states_modifiers === nothing ? ntuple(_ -> (), length(desn.cells)) :
        desn.states_modifiers
    ps_mods = map(
        layer_mods -> (
            layer_mods === nothing ? () :
                map(l -> initialparameters(rng, l), layer_mods) |> Tuple
        ),
        mods
    ) |> Tuple

    ps_ro = initialparameters(rng, desn.readout)
    return (cells = ps_cells, states_modifiers = ps_mods, readout = ps_ro)
end

function initialstates(rng::AbstractRNG, desn::DeepESN)
    st_cells = map(l -> initialstates(rng, l), desn.cells) |> Tuple

    mods = desn.states_modifiers === nothing ? ntuple(_ -> (), length(desn.cells)) :
        desn.states_modifiers

    st_mods = map(
        layer_mods -> (
            layer_mods === nothing ? () :
                map(l -> initialstates(rng, l), layer_mods) |> Tuple
        ),
        mods
    ) |> Tuple

    st_ro = initialstates(rng, desn.readout)
    return (cells = st_cells, states_modifiers = st_mods, readout = st_ro)
end

function _partial_apply(desn::DeepESN, inp, ps, st)
    n_layers = length(desn.cells)
    current_inp = inp

    # Using ntuple ensures strict type-stability across arbitrary hybrid layers
    new_states = ntuple(n_layers) do idx
        cell_out, st_cell_i = apply(desn.cells[idx], current_inp, ps.cells[idx], st.cells[idx])

        mod_out, st_mods_i = _apply_seq(
            desn.states_modifiers[idx], cell_out,
            ps.states_modifiers[idx], st.states_modifiers[idx]
        )

        current_inp = mod_out
        return (cell = st_cell_i, mod = st_mods_i)
    end

    new_cell_st = map(x -> x.cell, new_states)
    new_mods_st = map(x -> x.mod, new_states)

    return current_inp, (; cells = new_cell_st, states_modifiers = new_mods_st)
end

function (desn::DeepESN)(inp, ps, st)
    out, new_st = _partial_apply(desn, inp, ps, st)
    inp_t, st_ro = apply(desn.readout, out, ps.readout, st.readout)
    return inp_t, merge(new_st, (readout = st_ro,))
end

function resetcarry!(rng::AbstractRNG, desn::DeepESN, st; init_carry = nothing)
    n_layers = length(desn.cells)

    @inline function _layer_outdim(idx)
        st_i = st.cells[idx]
        if st_i.carry === nothing
            return desn.cells[idx].cell.out_dims
        else
            return size(first(st_i.carry), 1)
        end
    end

    @inline function _init_for(idx)
        if init_carry === nothing
            return nothing
        elseif init_carry isa Function
            sz = _layer_outdim(idx)
            return (_asvec(init_carry(rng, sz)),)
        elseif init_carry isa Tuple || init_carry isa AbstractVector
            f = init_carry[idx]
            sz = _layer_outdim(idx)
            return f === nothing ? nothing : (_asvec(f(rng, sz)),)
        else
            throw(ArgumentError("init_carry must be nothing, a Function, or a Tuple/Vector of Functions"))
        end
    end

    new_cells = ntuple(
        idx -> begin
            st_i = st.cells[idx]
            new_carry = _init_for(idx)
            merge(st_i, (; carry = new_carry))
        end, n_layers
    )

    return (;
        cells = new_cells,
        states_modifiers = st.states_modifiers,
        readout = st.readout,
    )
end

function collectstates(desn::DeepESN, data::AbstractMatrix, ps, st::NamedTuple)
    newst = st
    collected = Any[]
    n_layers = length(desn.cells)

    for inp in eachcol(data)
        current_inp = inp

        # Type-stable folding for arbitrary custom cells
        new_states = ntuple(n_layers) do idx
            cell_out, st_cell_i = apply(desn.cells[idx], current_inp, ps.cells[idx], newst.cells[idx])

            mod_out, st_mods_i = _apply_seq(
                desn.states_modifiers[idx], cell_out,
                ps.states_modifiers[idx], newst.states_modifiers[idx]
            )

            current_inp = mod_out
            return (cell = st_cell_i, mod = st_mods_i)
        end

        push!(collected, copy(current_inp))

        newst = (;
            cells = map(x -> x.cell, new_states),
            states_modifiers = map(x -> x.mod, new_states),
            readout = newst.readout,
        )
    end

    @assert !isempty(collected)
    states = eltype(data).(reduce(hcat, collected))

    return states, newst
end
