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

function __partial_apply(rc::AbstractReservoirComputer, inp, ps, st)
    out, st_res = apply(rc.reservoir, inp, ps.reservoir, st.reservoir)
    out,
        st_mods = __apply_seq(
        rc.state_modifiers, out, ps.state_modifiers, st.state_modifiers
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
    states = similar(data, state_dims, nsteps)
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
        ::AbstractReservoirComputer, output_matrix::AbstractMatrix,
        ps::NamedTuple, st::NamedTuple
    )
    @assert hasproperty(ps, :readout)
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

When a function is supplied as `init_carry`, an existing carry provides its leading
dimension; otherwise the reservoir output size is used. When `init_carry=nothing`,
the carry is cleared and the cell's initializer is used on the next call. This does
not require the cell to expose an output dimension.

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
  Updated states with `st′.cell.carry` set to `nothing` or `(h0,)`.
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
        if carry === nothing
            sz = __cell_out_dims(rc.reservoir.cell)
        else
            state = first(carry)
            sz = size(state, 1)
        end
        new_state = init_carry(rng, sz, 1)
        new_state = (new_state,)
    end
    new_cell = merge(st.reservoir, (; carry = new_state))
    return merge(st, (reservoir = new_cell,))
end

function resetcarry!(
        rng::AbstractRNG, rc::AbstractReservoirComputer,
        ps, st; init_carry = nothing
    )
    return ps, resetcarry!(rng, rc, st; init_carry = init_carry)
end
