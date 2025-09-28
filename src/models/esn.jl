@concrete struct ESN <: AbstractLuxContainerLayer{(:cell, :states_modifiers, :readout)}
    cell::Any
    states_modifiers::Any
    readout::Any
end

_wrap_layer(x) = x isa Function ? WrappedFunction(x) : x
_wrap_layers(xs::Tuple) = map(_wrap_layer, xs)

function ESN(in_dims::IntegerType, res_dims::IntegerType,
        out_dims::IntegerType, activation = tanh;
        readout_activation = identity,
        state_modifiers = (),
        kwargs...)
    cell = StatefulLayer(ESNCell(in_dims => res_dims, activation; kwargs...))
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
                 Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return ESN(cell, mods, ro)
end

function initialparameters(rng::AbstractRNG, esn::ESN)
    ps_cell = initialparameters(rng, esn.cell)
    ps_mods = map(l -> initialparameters(rng, l), esn.states_modifiers) |> Tuple
    ps_ro = initialparameters(rng, esn.readout)
    return (cell = ps_cell, states_modifiers = ps_mods, readout = ps_ro)
end

function initialstates(rng::AbstractRNG, esn::ESN)
    st_cell = initialstates(rng, esn.cell)
    st_mods = map(l -> initialstates(rng, l), esn.states_modifiers) |> Tuple
    st_ro = initialstates(rng, esn.readout)
    return (cell = st_cell, states_modifiers = st_mods, readout = st_ro)
end

@inline function _apply_seq(layers::Tuple, inp, ps::Tuple, st::Tuple)
    new_st_parts = Vector{Any}(undef, length(layers))
    for idx in eachindex(layers)
        inp, sti = apply(layers[idx], inp, ps[idx], st[idx])
        new_st_parts[idx] = sti
    end
    return inp, tuple(new_st_parts...)
end

function (esn::ESN)(inp::AbstractVector, ps, st)
    out, st_cell = apply(esn.cell, inp, ps.cell, st.cell)
    out, st_mods = _apply_seq(
        esn.states_modifiers, out, ps.states_modifiers, st.states_modifiers)
    out, st_ro = apply(esn.readout, out, ps.readout, st.readout)
    return out, (cell = st_cell, states_modifiers = st_mods, readout = st_ro)
end

function reset_carry(rng::AbstractRNG, esn::ESN, ps, st; init_carry = nothing)
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
    return ps,
    (cell = new_cell, states_modifiers = st.states_modifiers, readout = st.readout)
end

_set_readout_weight(ps_readout::NamedTuple, wro) = merge(ps_readout, (; weight = wro))

function collectstates(esn::ESN, data::AbstractMatrix, ps, st::NamedTuple)
    newst = st
    collected = Any[]
    for inp in eachcol(data)
        cell_y, st_cell = apply(esn.cell, inp, ps.cell, newst.cell)
        state_t, st_mods = _apply_seq(
            esn.states_modifiers, cell_y, ps.states_modifiers, newst.states_modifiers)
        push!(collected, copy(state_t))
        newst = (cell = st_cell, states_modifiers = st_mods, readout = newst.readout)
    end
    states = eltype(data).(reduce(hcat, collected))
    @assert !isempty(collected)
    states_raw = reduce(hcat, collected)
    states = eltype(data).(states_raw)
    return states, newst
end

function train!(esn::ESN, train_data::AbstractMatrix, target_data::AbstractMatrix,
        ps, st, train_method = StandardRidge(0.0);
        washout::Int = 0, return_states::Bool = false)
    states, newst = collectstates(esn, train_data, ps, st)
    states_wo, targets_wo = washout > 0 ? _apply_washout(states, target_data, washout) :
                            (states, target_data)
    wro = train(train_method, states_wo, targets_wo)
    ps2 = (cell = ps.cell,
        states_modifiers = ps.states_modifiers,
        readout = _set_readout_weight(ps.readout, wro))

    return return_states ? ((ps2, newst), states_wo) : (ps2, newst)
end
