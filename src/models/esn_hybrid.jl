@concrete struct HybridESN <: AbstractEchoStateNetwork{(:cell, :states_modifiers, :readout, :knowledge_model)}
    cell
    knowledge_model
    states_modifiers
    readout
end

function HybridESN(km,
    km_dims::IntegerType, in_dims::IntegerType,
    res_dims::IntegerType, out_dims::IntegerType,
    activation=tanh;
    state_modifiers=(),
    readout_activation=identity,
    include_collect::BoolType=True(),
    kwargs...)

    esn_inp_size = in_dims + km_dims
    cell = StatefulLayer(ESNCell(esn_inp_size => res_dims, activation; kwargs...))
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
                 Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims + km_dims => out_dims, readout_activation;
        include_collect=static(include_collect))
    km_layer = km isa WrappedFunction ? km : WrappedFunction(km)
    return HybridESN(cell, km_layer, mods, ro)
end

function initialparameters(rng::AbstractRNG, hesn::HybridESN)
    ps_cell = initialparameters(rng, hesn.cell)
    ps_km = initialparameters(rng, hesn.knowledge_model)
    ps_mods = map(l -> initialparameters(rng, l), hesn.states_modifiers) |> Tuple
    ps_ro = initialparameters(rng, hesn.readout)
    return (cell=ps_cell, knowledge_model=ps_km, states_modifiers=ps_mods, readout=ps_ro)
end

function initialstates(rng::AbstractRNG, hesn::HybridESN)
    st_cell = initialstates(rng, hesn.cell)
    st_km = initialstates(rng, hesn.knowledge_model)
    st_mods = map(l -> initialstates(rng, l), hesn.states_modifiers) |> Tuple
    st_ro = initialstates(rng, hesn.readout)
    return (cell=st_cell, knowledge_model=st_km, states_modifiers=st_mods, readout=st_ro)
end

function _partial_apply(hesn::HybridESN, inp, ps, st)
    k_t, st_km = hesn.knowledge_model(inp, ps.knowledge_model, st.knowledge_model)
    xin = vcat(k_t, inp)
    r, st_cell = apply(hesn.cell, xin, ps.cell, st.cell)
    rstar, st_mods = _apply_seq(hesn.states_modifiers, r, ps.states_modifiers, st.states_modifiers)
    feats = vcat(k_t, rstar)
    return feats, (cell=st_cell, states_modifiers=st_mods, knowledge_model=st_km)
end

function (hesn::HybridESN)(inp, ps, st)
    feats, new_st = _partial_apply(hesn, inp, ps, st)
    y, st_ro = apply(hesn.readout, feats, ps.readout, st.readout)
    return y, merge(new_st, (readout=st_ro,))
end
