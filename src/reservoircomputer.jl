@doc raw"""
    ReservoirComputer(reservoir, state_modifiers, readout)
    ReservoirComputer(reservoir, readout)

Generic reservoir computing container that wires together:
  1) a `reservoir` (any Lux-compatible layer producing features),
  2) (Optional) zero or more `state_modifiers` applied sequentially to the reservoir features,
  3) a `readout` layer (typically [`LinearReadout`](@ref)).

The container exposes a standard `(x, ps, st) -> (y, st′)` interface and
utility functions to initialize parameters/states, stream sequences to collect
features, and install trained readout weights.

## Arguments

- `reservoir`: a layer that consumes inputs and produces feature vectors.
- `state_modifiers`: a tuple (or vector converted to `Tuple`) of layers applied
  after the reservoir (optional. May be empty).
- `readout`: the final trainable layer mapping features to outputs.

## Inputs

- `x`: input to the reservoir (shape determined by the reservoir).
- `ps`: reservoir computing parameters.
- `st`: reservoir computing states.

## Returns

- `(y, st′)` where `y` is the readout output and `st′` contains the updated
  states of the reservoir, modifiers, and readout.
"""
struct ReservoirComputer{R, S, L} <:
    AbstractReservoirComputer{(:reservoir, :state_modifiers, :readout)}
    reservoir::R
    state_modifiers::S
    readout::L

    function ReservoirComputer(reservoir::R, state_modifiers::S, readout::L) where {R, S, L}
        mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
            Tuple(state_modifiers) : (state_modifiers,)
        mods = __wrap_layers(mods_tuple)

        return new{R, typeof(mods), L}(reservoir, mods, readout)
    end
end

function ReservoirComputer(reservoir, readout)
    return ReservoirComputer(reservoir, (), readout)
end

function initialparameters(rng::AbstractRNG, rc::AbstractReservoirComputer)
    ps_res = initialparameters(rng, rc.reservoir)
    ps_mods = map(l -> initialparameters(rng, l), rc.state_modifiers) |> Tuple
    ps_ro = initialparameters(rng, rc.readout)
    return (reservoir = ps_res, state_modifiers = ps_mods, readout = ps_ro)
end

function initialstates(rng::AbstractRNG, rc::AbstractReservoirComputer)
    st_res = initialstates(rng, rc.reservoir)
    st_mods = map(l -> initialstates(rng, l), rc.state_modifiers) |> Tuple
    st_ro = initialstates(rng, rc.readout)
    return (reservoir = st_res, state_modifiers = st_mods, readout = st_ro)
end

function __require_nonempty_data(data::AbstractMatrix, context::AbstractString)
    size(data, 2) ≥ 1 || throw(
        ArgumentError("$context data must have at least one column, got $(size(data, 2)).")
    )
    return nothing
end

@inline __apply_seq(::Tuple{}, inp, ::Tuple{}, ::Tuple{}) = inp, ()

@inline function __apply_seq(layers::Tuple, inp, ps::Tuple, st::Tuple)
    out, head_st = apply(first(layers), inp, first(ps), first(st))
    out, tail_st = __apply_seq(Base.tail(layers), out, Base.tail(ps), Base.tail(st))
    return out, (head_st, tail_st...)
end

@inline function __apply_state_modifier(layer, features, _, ps, st)
    return apply(layer, features, ps, st)
end

@inline function __apply_state_modifiers(
        ::Tuple{}, features, _, ::Tuple{}, ::Tuple{}
    )
    return features, ()
end

@inline function __apply_state_modifiers(
        modifiers::Tuple, features, inp, ps::Tuple, st::Tuple
    )
    out, head_st = __apply_state_modifier(
        first(modifiers), features, inp, first(ps), first(st)
    )
    out, tail_st = __apply_state_modifiers(
        Base.tail(modifiers), out, inp, Base.tail(ps), Base.tail(st)
    )
    return out, (head_st, tail_st...)
end

@inline function __state_modifier_output_dims(::Any, state_dims::Int, ::Int)
    return state_dims
end

@inline function __state_modifier_output_dims(
        ::Tuple{}, state_dims::Int, ::Int
    )
    return state_dims
end


@inline function __state_modifier_output_dims(
        modifiers::Tuple, state_dims::Int, input_dims::Int
    )
    output_dims = __state_modifier_output_dims(first(modifiers), state_dims, input_dims)
    return __state_modifier_output_dims(Base.tail(modifiers), output_dims, input_dims)
end

function __resolve_readout_in_dims(
        readout_in_dims::Nothing, modifiers::Tuple, state_dims::Int, input_dims::Int
    )
    return __state_modifier_output_dims(modifiers, state_dims, input_dims)
end

function __resolve_readout_in_dims(
        readout_in_dims::IntegerType, ::Tuple, ::Int, ::Int
    )
    readout_in_dims > 0 || throw(
        ArgumentError("`readout_in_dims` must be positive, got $readout_in_dims.")
    )
    return Int(readout_in_dims)
end

function __partial_apply(rc::AbstractReservoirComputer, inp, ps, st)
    out, st_res = apply(rc.reservoir, inp, ps.reservoir, st.reservoir)
    out, st_mods = __apply_state_modifiers(
        rc.state_modifiers, out, inp, ps.state_modifiers, st.state_modifiers
    )
    return out, (reservoir = st_res, state_modifiers = st_mods)
end

function (rc::AbstractReservoirComputer)(inp, ps, st)
    out, new_st = __partial_apply(rc, inp, ps, st)
    out, st_ro = apply(rc.readout, out, ps.readout, st.readout)
    return out, merge(new_st, (readout = st_ro,))
end

function collectstates(
        rc::AbstractReservoirComputer, data::AbstractMatrix, ps, st::NamedTuple
    )
    return __collectstates(rc.reservoir, rc, data, ps, st)
end

function __collectstates(
        ::AbstractSciMLProblemReservoir, ::AbstractReservoirComputer,
        ::AbstractMatrix, _, ::NamedTuple
    )
    return error(
        "collectstates for a `SciMLProblemReservoir` requires the " *
            "`RCODEReservoirExt` extension. Load `OrdinaryDiffEq`, `SciMLBase`, " *
            "and `DataInterpolations` to enable continuous-time reservoirs."
    )
end

function __collectstates(
        _, rc::AbstractReservoirComputer, data::AbstractMatrix, ps, st::NamedTuple
    )
    __require_nonempty_data(data, "collectstates")
    newst = st
    nsteps = size(data, 2)
    cols = eachcol(data)
    x1 = first(cols)
    current_state, partial_st = __partial_apply(rc, x1, ps, newst)
    state_dims = size(current_state, 1)
    states = similar(data, eltype(current_state), state_dims, nsteps)
    states[:, 1] .= current_state
    newst = merge(partial_st, (readout = newst.readout,))
    for (idx, inp) in Base.Iterators.drop(Base.enumerate(cols), 1)
        current_state, partial_st = __partial_apply(rc, inp, ps, newst)
        states[:, idx] .= current_state
        newst = merge(partial_st, (readout = newst.readout,))
    end

    return states, newst
end

__set_readout_weight(ps_readout::NamedTuple, wro) = merge(ps_readout, (; weight = wro))

function addreadout!(
        rc::AbstractReservoirComputer, output_matrix::AbstractMatrix,
        ps::NamedTuple, st::NamedTuple
    )
    hasproperty(ps, :readout) || throw(
        ArgumentError("model parameters must contain a `readout` entry")
    )
    hasproperty(rc, :readout) && rc.readout isa SVMReadout && __svmreadout_matrix_error()
    new_readout = __set_readout_weight(ps.readout, output_matrix)
    return merge(ps, (readout = new_readout,)), st
end

function Base.show(io::IO, rc::ReservoirComputer)
    print(io, "ReservoirComputer(\n")

    print(io, "    reservoir = ")
    show(io, rc.reservoir)
    print(io, ",\n")

    print(io, "    state_modifiers = ")
    if isempty(rc.state_modifiers)
        print(io, "()")
    else
        print(io, "(")
        for (i, m) in enumerate(rc.state_modifiers)
            i > 1 && print(io, ", ")
            show(io, m)
        end
        print(io, ")")
    end
    print(io, ",\n")

    print(io, "    readout = ")
    show(io, rc.readout)
    print(io, "\n)")

    return
end

@doc raw"""
    resetcarry!(rng, rc::ReservoirComputer, st; init_carry=nothing)
    resetcarry!(rng, rc::ReservoirComputer, ps, st; init_carry=nothing)

Reset (or set) the hidden-state carry of a model in the echo state network family.

When a function is supplied as `init_carry`, it is called as `init_carry(rng, dim, 1)`
once per state held by the cell (one for most cells, a hidden state and a memory
state for [`RMNCell`](@ref)). An existing carry provides the leading dimensions;
otherwise they are taken from the cell. When `init_carry=nothing`,
the carry is cleared and the cell's initializer is used on the next call. This does
not require the cell to expose an output dimension.

For continuous-time reservoirs ([`ContinuousESN`](@ref), [`LSM`](@ref) and
[`SciMLProblemReservoir`](@ref)) the carry is the ODE state vector, and `init_carry` is
called as `init_carry(rng, dim, 1)` and flattened to a vector. `dim` is `out_dims` for
`ContinuousESN`, `length(prob.u0)` for `SciMLProblemReservoir`, and `out_dims` for `LSM`,
where `init_carry` sets only the membrane potentials (the first half of the state)
and the remaining half of the state is set to zero.

The history and clock of every [`DelayLayer`](@ref) held by the model, in
`st.input_delay` or in `st.state_modifiers`, are reset as well: with
`init_carry=nothing` the history is cleared and rebuilt from the layer's `init_delay`
on the next call, and with a function the history is set to
`init_carry(rng, in_dims, num_delays)`. The clock restarts at zero. Other modifier
and readout states are unchanged.

For [`LIFESN`](@ref) the cached `start_states` and the input buffer of the
[`LocalInformationFlow`](@ref) cell are reset together with the carry: with
`init_carry=nothing` both are cleared and sampled again from the cell's `init_state` and
`init_buffer` on the next call, and with a function `start_states` is set to the same
states as the carry. The cell's `rng` is never advanced, so a random `init_state` and
`init_buffer` are sampled again with the same values as in a fresh state.

## Arguments

- `rng`: Random number generator (used if a new carry is sampled/created).
- `rc`: A reservoir computing network model.
- `st`: Current model states.
- `ps`: Optional model parameters. Returned unchanged.

## Keyword arguments

- `init_carry`: Controls the initialization of the new carry.
  - `nothing` (default): remove/clear the carry (forces the cell to reinitialize
    from its own `init_state` on next use).
  - `f`: a function following standard from
    [WeightInitializers.jl](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers)

## Returns

- `resetcarry!(rng, rc, st; ...) -> st′`:
  Updated states with `st′.reservoir.carry` set to `nothing` or a tuple with one
  state per recurrent state of the cell.
- `resetcarry!(rng, rc, ps, st; ...) -> (ps, st′)`:
  Same as above, but also returns the unchanged `ps` for convenience.

"""
function resetcarry!(
        rng::AbstractRNG, rc::AbstractReservoirComputer, st; init_carry = nothing
    )
    carry = get(st.reservoir, :carry, nothing)
    if init_carry === nothing
        new_state = nothing
    else
        cell = __carry_cell(rc.reservoir)
        dims = carry === nothing ? __carry_dimensions(cell) :
            map(state -> size(state, 1), carry)
        new_state = map(dim -> __init_carry_state(cell, init_carry, rng, dim), dims)
    end
    new_res = __reset_window(rc.reservoir, merge(st.reservoir, (; carry = new_state)), new_state)
    new_st = __reset_modifier_delays(rng, rc, merge(st, (reservoir = new_res,)), init_carry)
    return haskey(st, :input_delay) ?
        merge(
            new_st,
            (input_delay = __reset_delay(rng, rc.input_delay, st.input_delay, init_carry),)
        ) : new_st
end

function __reset_modifier_delays(rng::AbstractRNG, rc, st, init_carry)
    new_mods = map(
        (layer, st_layer) -> __reset_delay(rng, layer, st_layer, init_carry),
        rc.state_modifiers, st.state_modifiers
    )
    return merge(st, (state_modifiers = new_mods,))
end

function resetcarry!(
        rng::AbstractRNG, rc::AbstractReservoirComputer,
        ps, st; init_carry = nothing
    )
    return ps, resetcarry!(rng, rc, st; init_carry = init_carry)
end
