
@doc raw"""
    ReservoirComputer(reservoir, states_modifiers, readout)
    ReservoirComputer(reservoir, readout)

Generic reservoir computing container that wires together:
  1) a `reservoir` (any Lux-compatible layer producing features),
  2) (Optional) zero or more `states_modifiers` applied sequentially to the reservoir features,
  3) a `readout` layer (typically [`LinearReadout`](@ref)).

The container exposes a standard `(x, ps, st) -> (y, st′)` interface and
utility functions to initialize parameters/states, stream sequences to collect
features, and install trained readout weights.

## Arguments

- `reservoir`: a layer that consumes inputs and produces feature vectors.
- `states_modifiers`: a tuple (or vector converted to `Tuple`) of layers applied
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
       AbstractReservoirComputer{(:reservoir, :states_modifiers, :readout)}
    reservoir::R
    states_modifiers::S
    readout::L

    function ReservoirComputer(reservoir::R, state_modifiers::S, readout::L) where {R, S, L}
        mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
                     Tuple(state_modifiers) : (state_modifiers,)
        mods = _wrap_layers(mods_tuple)

        return new{R, typeof(mods), L}(reservoir, mods, readout)
    end
end

function ReservoirComputer(reservoir, readout)
    return ReservoirComputer(reservoir, (), readout)
end

function initialparameters(rng::AbstractRNG, rc::AbstractReservoirComputer)
    ps_res = initialparameters(rng, rc.reservoir)
    ps_mods = map(l -> initialparameters(rng, l), rc.states_modifiers) |> Tuple
    ps_ro = initialparameters(rng, rc.readout)
    return (reservoir = ps_res, states_modifiers = ps_mods, readout = ps_ro)
end

function initialstates(rng::AbstractRNG, rc::AbstractReservoirComputer)
    st_res = initialstates(rng, rc.reservoir)
    st_mods = map(l -> initialstates(rng, l), rc.states_modifiers) |> Tuple
    st_ro = initialstates(rng, rc.readout)
    return (reservoir = st_res, states_modifiers = st_mods, readout = st_ro)
end

@inline function _apply_seq(layers::Tuple, inp, ps::Tuple, st::Tuple)
    new_st_parts = Vector{Any}(undef, length(layers))
    for idx in eachindex(layers)
        inp, sti = apply(layers[idx], inp, ps[idx], st[idx])
        new_st_parts[idx] = sti
    end
    return inp, tuple(new_st_parts...)
end

function _partial_apply(rc::AbstractReservoirComputer, inp, ps, st)
    out, st_res = apply(rc.reservoir, inp, ps.reservoir, st.reservoir)
    out,
    st_mods = _apply_seq(
        rc.states_modifiers, out, ps.states_modifiers, st.states_modifiers)
    return out, (reservoir = st_res, states_modifiers = st_mods)
end

function (rc::AbstractReservoirComputer)(inp, ps, st)
    out, new_st = _partial_apply(rc, inp, ps, st)
    out, st_ro = apply(rc.readout, out, ps.readout, st.readout)
    return out, merge(new_st, (readout = st_ro,))
end

function collectstates(
        rc::AbstractReservoirComputer, data::AbstractMatrix, ps, st::NamedTuple)
    newst = st
    nsteps = size(data, 2)
    cols = eachcol(data)
    @assert !isempty(cols)
    x1 = first(cols)
    current_state, partial_st = _partial_apply(rc, x1, ps, newst)
    state_dims = size(current_state, 1)
    states = similar(data, state_dims, nsteps)
    states[:, 1] .= current_state
    newst = merge(partial_st, (readout = newst.readout,))
    for (idx, inp) in Base.Iterators.drop(Base.enumerate(cols), 1)
        current_state, partial_st = _partial_apply(rc, inp, ps, newst)
        states[:, idx] .= current_state
        newst = merge(partial_st, (readout = newst.readout,))
    end

    return states, newst
end

_set_readout_weight(ps_readout::NamedTuple, wro) = merge(ps_readout, (; weight = wro))

function addreadout!(::AbstractReservoirComputer, output_matrix::AbstractMatrix,
        ps::NamedTuple, st::NamedTuple)
    @assert hasproperty(ps, :readout)
    new_readout = _set_readout_weight(ps.readout, output_matrix)
    return merge(ps, (readout = new_readout,)), st
end

function Base.show(io::IO, rc::ReservoirComputer)
    print(io, "ReservoirComputer(\n")

    print(io, "    reservoir = ")
    show(io, rc.reservoir)
    print(io, ",\n")

    print(io, "    state_modifiers = ")
    if isempty(rc.states_modifiers)
        print(io, "()")
    else
        print(io, "(")
        for (i, m) in enumerate(rc.states_modifiers)
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

If an existing carry is present in `st.cell.carry`, its leading dimension is used to
infer the state size. Otherwise the reservoir output size is taken from
`rc.reservoir.cell.out_dims`. When `init_carry=nothing`, the carry is cleared; the initializer
from the struct construction will then be used. When a
function is provided, it is called to create a new initial hidden state.

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
        rng::AbstractRNG, rc::AbstractReservoirComputer, st; init_carry = nothing)
    carry = get(st.reservoir, :carry, nothing)
    if carry === nothing
        outd = rc.reservoir.cell.out_dims
        sz = outd
    else
        state = first(carry)
        sz = size(state, 1)
    end

    if init_carry === nothing
        new_state = nothing
    else
        new_state = init_carry(rng, sz, 1)
        new_state = (new_state,)
    end
    new_cell = merge(st.reservoir, (; carry = new_state))
    return merge(st, (reservoir = new_cell,))
end

function resetcarry!(rng::AbstractRNG, rc::AbstractReservoirComputer,
        ps, st; init_carry = nothing)
    return ps, resetcarry!(rng, rc, st; init_carry = init_carry)
end
