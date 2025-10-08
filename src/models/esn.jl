"""
    ESN(in_dims, res_dims, out_dims, activation=tanh;
        leak_coefficient=1.0, init_reservoir=rand_sparse, init_input=scaled_rand,
        init_bias=zeros32, init_state=randn32, use_bias=false,
        state_modifiers=(), readout_activation=identity)

Build a ESN.
"""
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

function _partial_apply(esn::ESN, inp, ps, st)
    out, st_cell = apply(esn.cell, inp, ps.cell, st.cell)
    out, st_mods = _apply_seq(
        esn.states_modifiers, out, ps.states_modifiers, st.states_modifiers)
    return out, (cell=st_cell, states_modifiers=st_mods)
end

function (esn::ESN)(inp, ps, st)
    out, new_st = _partial_apply(esn, inp, ps, st)
    out, st_ro = apply(esn.readout, out, ps.readout, st.readout)
    return out, merge(new_st, (readout=st_ro,))
end
