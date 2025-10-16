abstract type AbstractEchoStateNetwork{Fields} <: AbstractReservoirComputer{Fields} end

_wrap_layer(x) = x isa Function ? WrappedFunction(x) : x
_wrap_layers(xs::Tuple) = map(_wrap_layer, xs)

@inline function _apply_seq(layers::Tuple, inp, ps::Tuple, st::Tuple)
    new_st_parts = Vector{Any}(undef, length(layers))
    for idx in eachindex(layers)
        inp, sti = apply(layers[idx], inp, ps[idx], st[idx])
        new_st_parts[idx] = sti
    end
    return inp, tuple(new_st_parts...)
end

@inline function _fillvec(x, n::Integer)
    v = Vector{typeof(x)}(undef, n)
    @inbounds @simd for i in 1:n
        v[i] = x
    end
    return v
end

@inline _asvec(::Tuple{}, n::Integer) = _fillvec(nothing, n)

@inline function _asvec(comp::Tuple, n::Integer)
    len = length(comp)
    if len == n
        return collect(comp)
    elseif len == 1
        return _fillvec(comp[1], n)
    else
        error("Expected length $n or 1, got $len")
    end
end

@inline function _asvec(comp::AbstractVector, n::Integer)
    len = length(comp)
    if len == n
        return collect(comp)
    elseif len == 1
        return _fillvec(comp[1], n)
    else
        error("Expected length $n or 1, got $len")
    end
end

@inline _asvec(::Nothing, n::Integer) = _fillvec(nothing, n)

@inline _asvec(comp, n::Integer) = _fillvec(comp, n)

@inline _asvec(x) = (ndims(x) == 2 ? vec(x) : x)

function _coerce_layer_mods(x)
    x === nothing ? () :
    x isa Tuple ? x :
    x isa AbstractVector ? Tuple(x) :
    (x,)
end

_set_readout_weight(ps_readout::NamedTuple, wro) = merge(ps_readout, (; weight = wro))

function collectstates(esn::AbstractEchoStateNetwork, data::AbstractMatrix, ps, st::NamedTuple)
    newst = st
    collected = Any[]
    for inp in eachcol(data)
        state_t, partial_st = _partial_apply(esn, inp, ps, newst)
        push!(collected, copy(state_t))
        newst = merge(partial_st, (readout = newst.readout,))
    end
    states = eltype(data).(reduce(hcat, collected))
    @assert !isempty(collected)
    states_raw = reduce(hcat, collected)
    states = eltype(data).(states_raw)
    return states, newst
end

function addreadout!(::AbstractEchoStateNetwork, output_matrix::AbstractMatrix,
        ps::NamedTuple, st::NamedTuple)
    @assert hasproperty(ps, :readout)
    new_readout = _set_readout_weight(ps.readout, output_matrix)
    return merge(ps, (readout = new_readout,)), st
end

@doc raw"""
    resetcarry!(rng, esn::AbstractEchoStateNetwork, st; init_carry=nothing)
    resetcarry!(rng, esn::AbstractEchoStateNetwork, ps, st; init_carry=nothing)

Reset (or set) the hidden-state carry of a model in the echo state network family.

If an existing carry is present in `st.cell.carry`, its leading dimension is used to
infer the state size. Otherwise the reservoir output size is taken from
`esn.cell.cell.out_dims`. When `init_carry=nothing`, the carry is cleared; the initialzer
from the struct construction will then be used. When a
function is provided, it is called to create a new initial hidden state.

## Arguments

- `rng`: Random number generator (used if a new carry is sampled/created).
- `esn`: An echo state network model.
- `st`: Current model states.
- `ps`: Optional model parameters. Returned unchanged.

## Keyword arguments

- `init_carry`: Controls the initialization of the new carry.
  - `nothing` (default): remove/clear the carry (forces the cell to reinitialize
    from its own `init_state` on next use).
  - `f`: a function called as `f(rng, sz, batch)`, following standard from
    [WeightInitializers.jl](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers)

## Returns

- `resetcarry!(rng, esn, st; ...) -> st′`:
  Updated states with `st′.cell.carry` set to `nothing` or `(h0,)`.
- `resetcarry!(rng, esn, ps, st; ...) -> (ps, st′)`:
  Same as above, but also returns the unchanged `ps` for convenience.

"""
function resetcarry!(rng::AbstractRNG, esn::AbstractEchoStateNetwork, st; init_carry = nothing)
    carry = get(st.cell, :carry, nothing)
    if carry === nothing
        outd = esn.cell.cell.out_dims
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
    new_cell = merge(st.cell, (; carry = new_state))
    return merge(st, (cell = new_cell,))
end

function resetcarry!(rng::AbstractRNG, esn::AbstractEchoStateNetwork,
        ps, st; init_carry = nothing)
    return ps, resetcarry!(rng, esn, st; init_carry = init_carry)
end
