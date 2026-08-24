@doc raw"""
    DeepReservoir(cells, readout; state_modifiers=nothing, make_stateful=true)

Deep Reservoir Network wrapper, generalizing deep architectures [Gallicchio2017](@cite).

`DeepReservoir` composes, for `L = length(cells)` layers:
  1) a sequence of recurrent reservoir cells and ordinary Lux layers,
  2) zero or more per-layer `state_modifiers[ℓ]` applied to the layer's state, and
  3) a final `readout` layer from the last layer's features to the output.

## Arguments

  - `cells`: A nonempty tuple or vector of pre-instantiated layers. Every
    [`AbstractReservoirRecurrentCell`](@ref) is wrapped in [`StatefulLayer`](@ref)
    by default. Ordinary Lux layers are left unchanged. Continuous-time
    [`AbstractSciMLProblemReservoir`](@ref) cells are not supported because they
    require sequence-level integration through their specialized `collectstates` path.
  - `readout`: Readout layer from the last layer's features to the output.

## Keyword arguments

  - `make_stateful`: A boolean or collection of one or `L` booleans. Scalar and
    one-element inputs broadcast to every layer. When true, recurrent cells are
    wrapped in [`StatefulLayer`](@ref); ordinary Lux layers remain unchanged.
    Default: `true`.
  - `state_modifiers`: Per-layer modifiers applied before the next layer. A
    single modifier is broadcast to all layers. A length-`L` tuple/vector assigns
    one entry per layer; each entry may itself be a tuple/vector of modifiers.
    For a one-layer model, a tuple/vector of multiple modifiers is interpreted as
    that layer's modifier sequence. Default: `nothing`.

## Inputs

  - `x :: AbstractArray (in_dims, batch)`

## Returns

  - Output `y :: (out_dims, batch)`.
  - Updated layer state (`NamedTuple`) containing states for all cells, modifiers, and readout.

## Parameters

  - `cells :: NTuple{L,NamedTuple}` — parameters for each cell in the sequence.
  - `state_modifiers :: NTuple{L,Tuple}` — per-layer tuples of modifier parameters (empty tuples if none).
  - `readout` — parameters for the readout layer.

  > Exact field names for modifiers/readout follow their respective layer definitions.

## States

  - `cells :: NTuple{L,NamedTuple}` — states for each cell in the sequence.
  - `state_modifiers :: NTuple{L,Tuple}` — per-layer tuples of modifier states.
  - `readout` — states for the readout layer.

"""
@concrete struct DeepReservoir <:
    AbstractReservoirComputer{(:cells, :state_modifiers, :readout)}
    cells
    state_modifiers
    readout
end

function DeepReservoir(
        cells,
        readout;
        state_modifiers = nothing,
        make_stateful = true
    )
    cells_tuple = cells isa Tuple ? cells :
        cells isa AbstractVector ? Tuple(cells) :
        throw(ArgumentError("cells must be a nonempty tuple or vector of Lux layers"))
    n_layers = length(cells_tuple)
    n_layers > 0 || throw(ArgumentError("cells must contain at least one layer"))
    all(cell -> cell isa AbstractLuxLayer, cells_tuple) || throw(
        ArgumentError("every entry in cells must be an AbstractLuxLayer")
    )
    any(cell -> cell isa AbstractSciMLProblemReservoir, cells_tuple) && throw(
        ArgumentError(
            "DeepReservoir does not support AbstractSciMLProblemReservoir cells; " *
                "use their sequence-level collectstates interface instead"
        )
    )

    make_stateful_per_layer = __asvec(make_stateful, n_layers)
    all(flag -> flag isa Bool, make_stateful_per_layer) || throw(
        ArgumentError("make_stateful must be a Bool or a collection of Bool values")
    )
    wrapped_cells = ntuple(n_layers) do idx
        cell = cells_tuple[idx]
        should_wrap = make_stateful_per_layer[idx]
        should_wrap && cell isa AbstractReservoirRecurrentCell ? StatefulLayer(cell) : cell
    end

    raw_modifiers = if state_modifiers === nothing
        ntuple(Returns(nothing), n_layers)
    elseif n_layers == 1 &&
            (state_modifiers isa Tuple || state_modifiers isa AbstractVector) &&
            length(state_modifiers) != 1
        (state_modifiers,)
    else
        Tuple(__asvec(state_modifiers, n_layers))
    end
    modifiers_per_layer = map(__coerce_layer_mods, raw_modifiers)

    return DeepReservoir(wrapped_cells, modifiers_per_layer, readout)
end

@inline function __apply_deep_layers(
        ::Tuple{}, ::Tuple{}, inp, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}
    )
    return inp, (), ()
end

@inline function __apply_deep_layers(
        cells::Tuple, modifiers::Tuple, inp, ps_cells::Tuple,
        ps_modifiers::Tuple, st_cells::Tuple, st_modifiers::Tuple
    )
    cell_output, cell_state = apply(
        first(cells), inp, first(ps_cells), first(st_cells)
    )
    output, modifier_state = __apply_seq(
        first(modifiers), cell_output, first(ps_modifiers), first(st_modifiers)
    )
    final_output, remaining_cell_states, remaining_modifier_states =
        __apply_deep_layers(
        Base.tail(cells), Base.tail(modifiers), output,
        Base.tail(ps_cells), Base.tail(ps_modifiers),
        Base.tail(st_cells), Base.tail(st_modifiers)
    )
    return final_output, (cell_state, remaining_cell_states...),
        (modifier_state, remaining_modifier_states...)
end

function __partial_apply(dres::DeepReservoir, inp, ps, st)
    output, cell_states, modifier_states = __apply_deep_layers(
        dres.cells, dres.state_modifiers, inp,
        ps.cells, ps.state_modifiers, st.cells, st.state_modifiers
    )
    return output, (; cells = cell_states, state_modifiers = modifier_states)
end

function collectstates(dres::DeepReservoir, data::AbstractMatrix, ps, st::NamedTuple)
    return __collectstates(nothing, dres, data, ps, st)
end

function initialparameters(rng::AbstractRNG, dres::DeepReservoir)
    ps_cells = map(layer -> initialparameters(rng, layer), dres.cells) |> Tuple
    mods = dres.state_modifiers === nothing ? ntuple(_ -> (), length(dres.cells)) :
        dres.state_modifiers
    ps_mods = map(
        layer_mods -> (
            layer_mods === nothing ? () :
                map(layer -> initialparameters(rng, layer), layer_mods) |> Tuple
        ),
        mods
    ) |> Tuple

    ps_ro = initialparameters(rng, dres.readout)
    return (cells = ps_cells, state_modifiers = ps_mods, readout = ps_ro)
end

function initialstates(rng::AbstractRNG, dres::DeepReservoir)
    st_cells = map(layer -> initialstates(rng, layer), dres.cells) |> Tuple

    mods = dres.state_modifiers === nothing ? ntuple(_ -> (), length(dres.cells)) :
        dres.state_modifiers

    st_mods = map(
        layer_mods -> (
            layer_mods === nothing ? () :
                map(layer -> initialstates(rng, layer), layer_mods) |> Tuple
        ),
        mods
    ) |> Tuple

    st_ro = initialstates(rng, dres.readout)
    return (cells = st_cells, state_modifiers = st_mods, readout = st_ro)
end

__carry_dimensions(cell) = (__cell_out_dims(cell),)
__carry_dimensions(cell::Union{MemoryESNCell, MemoryResESNCell}) =
    (cell.out_dims, cell.out_dims)
__carry_dimensions(cell::RMNCell) = (
    __cell_out_dims(cell.nonlinear_reservoir),
    __cell_out_dims(cell.linear_reservoir),
)
__carry_dimensions(cell::LocalInformationFlow) = __carry_dimensions(cell.cell)

function __reset_deep_carry(
        rng::AbstractRNG, layer::StatefulLayer, st, initializer
    )
    initializer === nothing && return merge(st, (; carry = nothing))
    initializer isa Function || throw(
        ArgumentError("each init_carry entry must be nothing or a function")
    )
    dimensions = st.carry === nothing ? __carry_dimensions(layer.cell) :
        map(state -> size(state, 1), st.carry)
    carry = map(dim -> __asvec(initializer(rng, dim)), dimensions)
    return merge(st, (; carry))
end

__reset_deep_carry(::AbstractRNG, ::AbstractLuxLayer, st, ::Nothing) = st
function __reset_deep_carry(::AbstractRNG, layer::AbstractLuxLayer, _, initializer)
    throw(
        ArgumentError(
            "cannot initialize carry for non-stateful layer $(typeof(layer)); " *
                "use nothing for that layer"
        )
    )
end

function resetcarry!(rng::AbstractRNG, dres::DeepReservoir, st; init_carry = nothing)
    n_layers = length(dres.cells)
    initializers = Tuple(__asvec(init_carry, n_layers))
    new_cells = ntuple(n_layers) do idx
        __reset_deep_carry(rng, dres.cells[idx], st.cells[idx], initializers[idx])
    end

    return (;
        cells = new_cells,
        state_modifiers = st.state_modifiers,
        readout = st.readout,
    )
end
