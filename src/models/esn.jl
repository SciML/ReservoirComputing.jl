@concrete struct ESN <: AbstractEchoStateNetwork{(:cell, :states_modifiers, :readout)}
    cell
    states_modifiers
    readout
end

function ESN(in_dims::IntegerType, res_dims::IntegerType,
    out_dims::IntegerType, activation=tanh;
    readout_activation=identity,
    state_modifiers=(),
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
    return (cell=ps_cell, states_modifiers=ps_mods, readout=ps_ro)
end

function initialstates(rng::AbstractRNG, esn::ESN)
    st_cell = initialstates(rng, esn.cell)
    st_mods = map(l -> initialstates(rng, l), esn.states_modifiers) |> Tuple
    st_ro = initialstates(rng, esn.readout)
    return (cell=st_cell, states_modifiers=st_mods, readout=st_ro)
end

function (esn::ESN)(inp, ps, st)
    out, st_cell = apply(esn.cell, inp, ps.cell, st.cell)
    out, st_mods = _apply_seq(
        esn.states_modifiers, out, ps.states_modifiers, st.states_modifiers)
    out, st_ro = apply(esn.readout, out, ps.readout, st.readout)
    return out, (cell=st_cell, states_modifiers=st_mods, readout=st_ro)
end

function resetcarry(rng::AbstractRNG, esn::ESN, st; init_carry=nothing)
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

    new_cell = merge(st.cell, (; carry=new_state))
    return (cell=new_cell, states_modifiers=st.states_modifiers, readout=st.readout)
end

function collectstates(esn::ESN, data::AbstractMatrix, ps, st::NamedTuple)
    newst = st
    collected = Any[]
    for inp in eachcol(data)
        cell_y, st_cell = apply(esn.cell, inp, ps.cell, newst.cell)
        state_t, st_mods = _apply_seq(
            esn.states_modifiers, cell_y, ps.states_modifiers, newst.states_modifiers)
        push!(collected, copy(state_t))
        newst = (cell=st_cell, states_modifiers=st_mods, readout=newst.readout)
    end
    states = eltype(data).(reduce(hcat, collected))
    @assert !isempty(collected)
    states_raw = reduce(hcat, collected)
    states = eltype(data).(states_raw)
    return states, newst
end

function addreadout!(::ESN, output_matrix::AbstractMatrix,
    ps::NamedTuple, st::NamedTuple)
    @assert hasproperty(ps, :readout)
    new_readout = _set_readout_weight(ps.readout, output_matrix)
    return (cell=ps.cell,
        states_modifiers=ps.states_modifiers,
        readout=new_readout), st
end
